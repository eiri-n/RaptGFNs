"""
DNAアプタマー生成タスクの基底クラス
"""

import numpy as np


class BaseTask:
    """タスクの基底クラス"""
    
    def __init__(self, tokenizer, obj_dim, max_len, transform=lambda x: x, batch_size=1, **kwargs):
        self.tokenizer = tokenizer
        self.obj_dim = obj_dim
        self.transform = transform
        self.batch_size = batch_size
        self.max_len = max_len

    def _evaluate(self, x, out, *args, **kwargs):
        """評価関数（サブクラスで実装）"""
        raise NotImplementedError

    def score(self, str_array):
        """スコア計算（サブクラスで実装）"""
        raise NotImplementedError
    
    def task_setup(self, *args, **kwargs):
        """タスクセットアップ（サブクラスで実装）"""
        raise NotImplementedError
