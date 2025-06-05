#!/usr/bin/env python3
"""
DNAアプタマー生成のためのMulti-Objective GFlowNets メインスクリプト
"""

import hydra
import warnings
import random
import logging
import os
import sys
from pathlib import Path
import torch
import numpy as np
from omegaconf import OmegaConf, DictConfig

from src.dna_aptamer_mogfn.utils import flatten_config

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not available, logging will be limited")


def set_seed(seed):
    """ランダムシードを設定"""
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def init_run(cfg):
    """実行環境の初期化"""
    trial_id = cfg.trial_id
    if cfg.job_name is None:
        cfg.job_name = f"aptamer_mogfn_{trial_id}"
    
    cfg.seed = random.randint(0, 100000) if cfg.seed is None else cfg.seed
    set_seed(cfg.seed)
    
    cfg = OmegaConf.to_container(cfg, resolve=True)  # 設定の補間を解決
    cfg = DictConfig(cfg)
    
    print("=== 設定 ===")
    print(OmegaConf.to_yaml(cfg))
    
    # 設定をファイルに保存
    with open('hydra_config.yaml', 'w') as f:
        f.write(OmegaConf.to_yaml(cfg))
    
    return cfg


@hydra.main(version_base=None, config_path='./configs', config_name='main')
def main(config):
    """メイン関数"""
    random.seed(None)  # マルチラン時のランダムシード初期化
    
    # ログ設定の準備
    log_config = flatten_config(OmegaConf.to_container(config, resolve=True), sep='/')
    log_config = {'/'.join(('config', key)): val for key, val in log_config.items()}
    
    # wandbの初期化
    if WANDB_AVAILABLE and config.wandb_mode != 'disabled':
        wandb.init(
            project=config.project_name,
            config=log_config,
            mode=config.wandb_mode,
            group=config.group_name,
            name=config.exp_name,
            tags=config.exp_tags
        )
        config['job_name'] = wandb.run.name
    
    config = init_run(config)  # ランダムシードはここで固定
    
    ret_val = float('NaN')
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # 警告を無視
        
        try:
            logging.info("トークナイザーを初期化中...")
            tokenizer = hydra.utils.instantiate(config.tokenizer)
            
            logging.info("タスクを初期化中...")
            task = hydra.utils.instantiate(config.task, tokenizer=tokenizer)
            
            # プロジェクトルート（必要に応じて調整）
            project_root = Path(os.getcwd()).resolve()
            all_seqs, all_targets = task.task_setup(config, project_root=project_root)
            
            logging.info("アルゴリズムを初期化中...")
            algorithm = hydra.utils.instantiate(
                config.algorithm,
                task=task,
                tokenizer=tokenizer,
                cfg=config.algorithm,
                task_cfg=config.task
            )
            
            logging.info("最適化を開始中...")
            print(f"\n=== DNAアプタマー生成開始 ===")
            print(f"目的関数: {task.objectives}")
            print(f"配列長: {task.min_len}-{task.max_len}")
            print(f"訓練ステップ: {config.algorithm.train_steps}")
            print(f"デバイス: {algorithm.device}")
            print("============================\n")
            
            metrics = algorithm.optimize(task, init_data=None)
            
            # メトリクス名を整理
            metrics = {key.split('/')[-1]: val for key, val in metrics.items()}
            ret_val = metrics.get('hypervol_rel', float('NaN'))
            
            print(f"\n=== 最適化完了 ===")
            print(f"最終ハイパーボリューム: {ret_val:.4f}")
            print("===================\n")
            
        except Exception as err:
            logging.exception(f"エラーが発生しました: {err}")
            ret_val = float('NaN')
    
    if WANDB_AVAILABLE and config.wandb_mode != 'disabled':
        wandb.finish()
    
    return ret_val


if __name__ == "__main__":
    main()
    sys.exit()
