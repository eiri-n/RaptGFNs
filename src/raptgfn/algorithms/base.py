"""
アルゴリズムの基底クラス
"""

import pickle
import gzip
import os

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


class BaseAlgorithm:
    """アルゴリズムの基底クラス"""
    
    def __init__(self, cfg, task, tokenizer, task_cfg, **kwargs):
        self.cfg = cfg
        self.task = task
        self.tokenizer = tokenizer
        self.task_cfg = task_cfg
        self.state = {}
        self.state_save_path = getattr(cfg, 'state_save_path', './algorithm_state.pkl.gz')

    def optimize(self, task, init_data=None):
        """最適化を実行（サブクラスで実装）"""
        raise NotImplementedError("Override this method in your class")
    
    def log(self, metrics, commit=True):
        """メトリクスをログ出力"""
        if WANDB_AVAILABLE:
            try:
                wandb.log(metrics, commit=commit)
            except wandb.errors.Error:
                # WandBが初期化されていない場合
                print(f"Metrics: {metrics}")
        else:
            print(f"Metrics: {metrics}")
    
    def update_state(self, metrics):
        """状態を更新"""
        for k, v in metrics.items():
            if k in self.state.keys():
                self.state[k].append(v)
            else:
                self.state[k] = [v]

    def save_state(self):
        """状態を保存"""
        os.makedirs(os.path.dirname(self.state_save_path), exist_ok=True)
        with gzip.open(self.state_save_path, 'wb+') as f:
            pickle.dump(self.state, f)
            
    def load_state(self, path=None):
        """状態を読み込み"""
        path = path or self.state_save_path
        if os.path.exists(path):
            with gzip.open(path, 'rb') as f:
                self.state = pickle.load(f)
        else:
            print(f"State file not found: {path}")
