"""
DNAアプタマー生成用Multi-Objective GFlowNet (MOGFN)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import math
from torch.distributions import Categorical
from tqdm import tqdm

from dna_aptamer_mogfn.algorithms.base import BaseAlgorithm
from dna_aptamer_mogfn.algorithms.conditional_transformer import CondGFNTransformer
from dna_aptamer_mogfn.algorithms.mogfn_utils import (
    mean_pairwise_distances, generate_simplex, thermometer, 
    plot_pareto, pareto_frontier, compute_hypervolume
)
from dna_aptamer_mogfn.utils import str_to_tokens, tokens_to_str
from dna_aptamer_mogfn.metrics import get_all_metrics


class MOGFN(BaseAlgorithm):
    """Multi-Objective GFlowNet for DNA Aptamer Generation"""
    
    def __init__(self, cfg, task, tokenizer, task_cfg, **kwargs):
        super(MOGFN, self).__init__(cfg, task, tokenizer, task_cfg)
        self.setup_vars(kwargs)
        self.init_policy()

    def setup_vars(self, kwargs):
        """変数の初期化"""
        cfg = self.cfg
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        
        # タスク関連
        self.max_len = cfg.max_len
        self.min_len = cfg.min_len
        self.obj_dim = self.task.obj_dim
        
        # GFN関連
        self.train_steps = cfg.train_steps
        self.random_action_prob = cfg.random_action_prob
        self.batch_size = cfg.batch_size
        self.reward_min = cfg.reward_min
        self.therm_n_bins = cfg.therm_n_bins
        self.beta_use_therm = cfg.beta_use_therm
        self.pref_use_therm = cfg.pref_use_therm
        self.gen_clip = cfg.gen_clip
        self.sampling_temp = cfg.sampling_temp
        self.sample_beta = cfg.sample_beta
        self.beta_cond = cfg.beta_cond
        self.pref_cond = cfg.pref_cond
        self.beta_scale = cfg.beta_scale
        self.beta_shape = cfg.beta_shape
        self.pref_alpha = cfg.pref_alpha
        self.beta_max = cfg.beta_max
        self.reward_type = cfg.reward_type
        
        # 学習パラメータ
        self.pi_lr = cfg.pi_lr
        self.z_lr = cfg.z_lr
        self.wd = cfg.wd
        
        # 評価関連
        self.eval_metrics = cfg.eval_metrics
        self.eval_freq = cfg.eval_freq
        self.num_samples = cfg.num_samples
        self.k = cfg.k
        self.simplex_bins = cfg.simplex_bins
        self.use_eval_pref = cfg.use_eval_pref
        self.num_pareto_points = cfg.num_pareto_points
        self.pareto_freq = cfg.pareto_freq
        self.unnormalize_rewards = cfg.unnormalize_rewards
        
        # モデル設定
        self.model_cfg = cfg.model

    def init_policy(self):
        """ポリシーネットワークの初期化"""
        # モデルの初期化
        self.model = CondGFNTransformer(
            max_len=self.max_len,
            vocab_size=len(self.tokenizer.full_vocab),
            num_actions=len(self.tokenizer.non_special_vocab) + 1,  # +1 for stop action
            num_hid=self.model_cfg.num_hid,
            num_layers=self.model_cfg.num_layers,
            num_head=self.model_cfg.num_head,
            dropout=self.model_cfg.dropout,
            batch_size=self.model_cfg.batch_size,
            objectives=self.task.objectives,
            therm_n_bins=self.therm_n_bins,
            beta_use_therm=self.beta_use_therm,
            pref_use_therm=self.pref_use_therm
        ).to(self.device)
        
        # オプティマイザーの初期化
        self.opt = torch.optim.Adam(
            self.model.model_params(), 
            lr=self.pi_lr, 
            weight_decay=self.wd
        )
        self.opt_Z = torch.optim.Adam(
            self.model.Z_param(), 
            lr=self.z_lr, 
            weight_decay=self.wd
        )
        
        # シンプレックスの生成（パレート条件付け用）
        self.simplex = generate_simplex(self.obj_dim, self.simplex_bins)
        
        # 評価用設定
        if hasattr(self.task_cfg, 'eval_pref'):
            self.eval_pref = torch.tensor(self.task_cfg.eval_pref).float()
        else:
            self.eval_pref = torch.ones(self.obj_dim).float() / self.obj_dim

    def optimize(self, task, init_data=None):
        """最適化の実行"""
        losses = []
        rewards = []
        
        desc_str = 'rew {:.2f}, hv {:.2f}, r2 {:.2f}, hsri {:.2f}, loss {:.2f}, rs {:.2f}'
        pb = tqdm(range(self.train_steps), desc=desc_str.format(0, 0, 0, 0, 0, 0))
        
        # 初期評価
        hv, r2, hsri = 0, 0, 0
        
        for i in pb:
            loss, r = self.train_step(task, self.batch_size)
            losses.append(loss)
            rewards.append(r)
            
            if i != 0 and i % self.eval_freq == 0:
                with torch.no_grad():
                    samples, all_rews, rs, mo_metrics, topk_metrics, fig = self.evaluation(task, plot=True)
                
                hv = mo_metrics.get("hypervolume", 0)
                r2 = mo_metrics.get("r2", 0)
                hsri = mo_metrics.get("hsri", 0)
                
                # ログ出力
                eval_metrics = {
                    'topk_rewards': topk_metrics[0].mean(),
                    'topk_diversity': topk_metrics[1].mean(),
                    'sample_r': rs.mean(),
                    'eval_step': i,
                    **mo_metrics
                }
                
                # サンプル配列をWandBテーブルとして記録
                if i % (self.eval_freq * 5) == 0:  # 5回の評価に1回
                    sample_table = []
                    for j, seq in enumerate(samples[:10]):  # 上位10配列
                        scores = task.score([seq])[0]
                        sample_table.append([
                            j+1, seq, len(seq), 
                            float(scores[0]) if len(scores) > 0 else 0.0,  # 目的1
                            float(scores[1]) if len(scores) > 1 else 0.0   # 目的2
                        ])
                    
                    try:
                        import wandb
                        if hasattr(wandb, 'Table'):
                            eval_metrics['sample_sequences'] = wandb.Table(
                                columns=["Rank", "Sequence", "Length", "Length_Score", "GC_Score"],
                                data=sample_table
                            )
                    except ImportError:
                        pass
                
                self.log(eval_metrics, commit=False)
                
                if fig is not None:
                    self.log({'pareto_front': fig}, commit=False)
                
                # パレートフロンティアの保存
                if i % self.pareto_freq == 0:
                    self._save_pareto_results(task)
            
            # 訓練ログ
            metrics = {
                'train_loss': loss,
                'train_rewards': r,
                'step': i,
                'learning_rate': self.opt.param_groups[0]['lr'],
                'z_learning_rate': self.opt_Z.param_groups[0]['lr'],
            }
            
            # モデルパラメータの統計
            if i % 100 == 0:  # 100ステップごと
                grad_norm = 0
                for p in self.model.parameters():
                    if p.grad is not None:
                        grad_norm += p.grad.data.norm(2) ** 2
                grad_norm = grad_norm ** 0.5
                
                metrics.update({
                    'gradient_norm': grad_norm,
                    'model_param_norm': sum(p.data.norm(2) ** 2 for p in self.model.parameters()) ** 0.5
                })
            
            # Betaパラメータの分布
            if i % 50 == 0:  # 50ステップごと
                cond_var, (prefs, beta) = self._get_condition_var(train=True, bs=16)
                metrics.update({
                    'beta_mean': beta.mean().item(),
                    'beta_std': beta.std().item(),
                    'pref_0_mean': prefs[:, 0].mean().item(),
                    'pref_1_mean': prefs[:, 1].mean().item() if prefs.shape[1] > 1 else 0.0
                })
            
            self.log(metrics)
            
            pb.set_description(desc_str.format(
                rs.mean() if 'rs' in locals() else 0,
                hv, r2, hsri,
                np.mean(losses[-10:]),
                np.mean(rewards[-10:])
            ))
        
        return {
            'losses': losses,
            'train_rs': rewards,
            'hypervol_rel': hv
        }

    def train_step(self, task, batch_size):
        """1ステップの訓練"""
        cond_var, (prefs, beta) = self._get_condition_var(train=True, bs=batch_size)
        states, logprobs = self.sample(batch_size, cond_var)
        
        log_r = self.process_reward(states, prefs, task).to(self.device)
        
        self.opt.zero_grad()
        self.opt_Z.zero_grad()
        
        # Trajectory Balance Loss
        loss = (logprobs - beta * log_r).pow(2).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gen_clip)
        
        self.opt.step()
        self.opt_Z.step()
        
        return loss.item(), log_r.mean().item()

    def _get_condition_var(self, train=True, bs=1):
        """条件変数の生成"""
        if train:
            # 訓練時はランダムに選択
            pref_indices = torch.randint(0, len(self.simplex), (bs,))
            prefs = torch.tensor(self.simplex[pref_indices]).float().to(self.device)
        else:
            # 評価時は評価用設定を使用
            prefs = self.eval_pref.unsqueeze(0).repeat(bs, 1).to(self.device)
        
        # Betaパラメータの生成
        if self.beta_cond:
            if self.beta_use_therm:
                beta_raw = torch.distributions.Beta(self.beta_shape, 1).sample((bs,)).to(self.device)
                beta_raw_scaled = beta_raw * self.beta_max
                beta = thermometer(beta_raw_scaled, self.therm_n_bins, 0, self.beta_max)
            else:
                beta_raw_scaled = torch.distributions.Beta(self.beta_shape, 1).sample((bs,)).to(self.device) * self.beta_max
                beta = beta_raw_scaled
        else:
            beta_raw_scaled = torch.ones(bs).to(self.device) * self.sample_beta
            beta = beta_raw_scaled
        
        # 条件変数の結合
        if self.pref_use_therm and self.pref_cond:
            prefs_encoded = thermometer(prefs, self.therm_n_bins, 0, 1)
            prefs_encoded = prefs_encoded.view(bs, -1)
        else:
            prefs_encoded = prefs
        
        if self.beta_use_therm:
            cond_var = torch.cat([beta, prefs_encoded], dim=1)
        else:
            cond_var = torch.cat([beta.unsqueeze(1), prefs_encoded], dim=1)
        
        return cond_var, (prefs, beta_raw_scaled)

    def sample(self, episodes, cond_var=None, train=True):
        """配列のサンプリング"""
        states = [''] * episodes
        traj_logprob = torch.zeros(episodes).to(self.device)
        
        if cond_var is None:
            cond_var, _ = self._get_condition_var(train=train, bs=episodes)
        
        active_mask = torch.ones(episodes).bool().to(self.device)
        x = str_to_tokens(states, self.tokenizer).to(self.device)
        lens = torch.ones(episodes).long().to(self.device)  # [CLS]の分で1から開始
        
        uniform_pol = torch.empty(episodes).fill_(self.random_action_prob).to(self.device)
        
        for t in range(self.max_len):
            if not active_mask.any():
                break
                
            # パディングマスクの作成
            mask = torch.zeros_like(x, dtype=torch.bool)
            for i, length in enumerate(lens):
                if length < x.size(1):
                    mask[i, length:] = True
            
            logits = self.model(x, cond_var, mask, lens=lens)
            
            # 最小長制約（コピーして変更）
            if t < self.min_len:
                logits = logits.clone()
                logits[:, 0] = -1000  # Stop actionを無効化
                
            if t == 0:
                traj_logprob += self.model.Z(cond_var)
            
            # サンプリング
            sampling_dist = Categorical(logits=logits / self.sampling_temp)
            policy_dist = Categorical(logits=logits)
            actions = sampling_dist.sample()
            
            # ランダムアクション（訓練時）
            if train and self.random_action_prob > 0:
                uniform_mix = torch.bernoulli(uniform_pol).bool()
                random_actions = torch.randint(
                    int(t < self.min_len), logits.shape[1], (episodes,)
                ).to(self.device)
                actions = torch.where(uniform_mix, random_actions, actions)
            
            # ログ確率の計算
            log_probs = policy_dist.log_prob(actions)
            traj_logprob = traj_logprob + log_probs * active_mask.float()
            
            # 状態の更新
            new_active_mask = active_mask.clone()
            for i, action in enumerate(actions):
                if active_mask[i]:
                    if action == 0:  # Stop action
                        new_active_mask[i] = False
                    else:
                        token = self.tokenizer.sampling_vocab[action - 1]
                        states[i] += token
                        lens[i] += 1
            active_mask = new_active_mask
            
            # 入力の更新
            x = str_to_tokens(states, self.tokenizer).to(self.device)
            
            # 最大長に達した場合の処理
            if lens.max() >= self.max_len:
                break
        
        return states, traj_logprob

    def process_reward(self, states, prefs, task):
        """報酬の処理"""
        # 無効な状態をフィルタリング
        valid_states = [s for s in states if len(s) >= self.min_len]
        if not valid_states:
            return torch.zeros(len(states)).to(self.device)
        
        # スコア計算
        scores = task.score(valid_states)
        scores = torch.tensor(scores).float().to(self.device)
        
        # 選好による重み付け
        if self.reward_type == "convex":
            weighted_scores = torch.sum(scores * prefs[:len(scores)], dim=1)
        else:
            weighted_scores = torch.sum(scores, dim=1)
        
        # 報酬の変換
        rewards = torch.clamp(weighted_scores, min=self.reward_min)
        log_rewards = torch.log(rewards)
        
        # 無効な状態に対する処理
        if len(valid_states) < len(states):
            full_rewards = torch.full((len(states),), float('-inf')).to(self.device)
            full_rewards[:len(valid_states)] = log_rewards
            return full_rewards
        
        return log_rewards

    def evaluation(self, task, plot=False):
        """評価の実行"""
        # サンプル生成
        all_samples = []
        all_rewards = []
        
        for _ in range(self.num_samples // self.batch_size):
            samples, _ = self.sample(self.batch_size, train=False)
            valid_samples = [s for s in samples if len(s) >= self.min_len]
            
            if valid_samples:
                rewards = task.score(valid_samples)
                all_samples.extend(valid_samples)
                all_rewards.extend(rewards)
        
        if not all_samples:
            return [], [], [], {}, [[], []], None
        
        all_rewards = np.array(all_rewards)
        
        # パレートフロンティア計算
        pareto_samples, pareto_rewards = pareto_frontier(
            np.array(all_samples), all_rewards, maximize=True
        )
        
        # メトリクス計算
        mo_metrics = get_all_metrics(
            pareto_rewards, self.eval_metrics,
            hv_ref=np.min(all_rewards, axis=0) - 0.1,
            num_obj=self.obj_dim
        )
        
        # トップK解析
        topk_rewards, topk_diversity = self._analyze_topk(all_samples, all_rewards)
        
        # プロット
        fig = None
        if plot and len(pareto_rewards) > 0:
            fig = plot_pareto(pareto_rewards, all_rewards)
        
        return all_samples, all_rewards, all_rewards.mean(axis=0), mo_metrics, [topk_rewards, topk_diversity], fig

    def _analyze_topk(self, samples, rewards):
        """トップK解の解析"""
        if len(samples) == 0:
            return np.array([]), np.array([])
        
        # 各目的でトップK
        topk_rewards = []
        topk_diversity = []
        
        for obj_idx in range(self.obj_dim):
            obj_rewards = rewards[:, obj_idx]
            topk_indices = np.argsort(obj_rewards)[-self.k:]
            topk_samples = [samples[i] for i in topk_indices]
            
            topk_rewards.append(np.mean(obj_rewards[topk_indices]))
            topk_diversity.append(mean_pairwise_distances(topk_samples))
        
        return np.array(topk_rewards), np.array(topk_diversity)

    def _save_pareto_results(self, task):
        """パレートフロンティアの結果を保存"""
        samples, rewards = self.sample(self.num_pareto_points, train=False)
        valid_samples = [s for s in samples if len(s) >= self.min_len]
        
        if valid_samples:
            scores = task.score(valid_samples)
            pareto_samples, pareto_rewards = pareto_frontier(
                np.array(valid_samples), scores, maximize=True
            )
            
            self.update_state({
                'pareto_samples': pareto_samples.tolist(),
                'pareto_rewards': pareto_rewards.tolist(),
                'all_samples': valid_samples,
                'all_rewards': scores.tolist()
            })
            self.save_state()
