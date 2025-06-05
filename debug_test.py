#!/usr/bin/env python3
"""
デバッグテスト
"""

import torch
import hydra
from omegaconf import DictConfig
from src.dna_aptamer_mogfn.utils import AptamerTokenizer
from src.dna_aptamer_mogfn.tasks.simple_dna import SimpleDNATask
from src.dna_aptamer_mogfn.algorithms.mogfn import MOGFN

torch.autograd.set_detect_anomaly(True)

def debug_test():
    # 設定をハードコード
    tokenizer = AptamerTokenizer()
    
    task = SimpleDNATask(
        objectives=['length', 'gc_content'],
        eval_pref=[0.5, 0.5],
        score_max=[1.0, 1.0],
        min_len=30,
        max_len=60,
        ideal_length=45,
        ideal_gc_content=0.5
    )
    
    # アルゴリズム設定
    from omegaconf import OmegaConf
    cfg = OmegaConf.create({
        'train_steps': 10,
        'batch_size': 4,
        'pi_lr': 0.0001,
        'z_lr': 0.001,
        'wd': 0.0001,
        'max_len': 60,
        'min_len': 30,
        'random_action_prob': 0.01,
        'sampling_temp': 1.0,
        'gen_clip': 10,
        'reward_min': 1e-80,
        'reward_max': 100,
        'reward_type': 'convex',
        'beta_use_therm': True,
        'pref_use_therm': True,
        'beta_cond': True,
        'pref_cond': True,
        'beta_scale': 1,
        'beta_shape': 32,
        'pref_alpha': 1.0,
        'beta_max': 32,
        'therm_n_bins': 50,
        'sample_beta': 4,
        'eval_metrics': ['hypervolume', 'r2', 'hsri'],
        'eval_freq': 100,
        'num_samples': 256,
        'k': 10,
        'simplex_bins': 10,
        'use_eval_pref': False,
        'num_pareto_points': 500,
        'pareto_freq': 200,
        'unnormalize_rewards': False,
        'state_save_path': './outputs/mogfn_state.pkl.gz',
        'model': {
            '_target_': 'dna_aptamer_mogfn.algorithms.conditional_transformer.CondGFNTransformer',
            'max_len': 60,
            'vocab_size': 10,
            'num_actions': 5,
            'num_hid': 64,
            'num_layers': 3,
            'num_head': 8,
            'bidirectional': False,
            'dropout': 0.0,
            'batch_size': 4
        }
    })
    
    algorithm = MOGFN(cfg, tokenizer, task, device='cuda')
    
    print("=== デバッグテスト開始 ===")
    try:
        algorithm.init_policy()
        
        # 1ステップだけ実行
        loss, r = algorithm.train_step(task, 4)
        print(f"成功: loss={loss:.4f}, reward={r:.4f}")
        
    except Exception as e:
        print(f"エラー: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_test()
