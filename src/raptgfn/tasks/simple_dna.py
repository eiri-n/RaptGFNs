"""
NUPACKを使わない簡単なDNAアプタマー生成タスク
構造計算を行わず、配列の特性のみを評価
"""

import numpy as np
from omegaconf import ListConfig

from raptgfn.tasks.base import BaseTask


class SimpleDNATask(BaseTask):
    """
    NUPACK不要の簡単なDNAアプタマー生成タスク
    
    利用可能な目的関数:
    - length: 配列長を評価（理想的な長さへの近さ）
    - gc_content: GC含有量を評価（理想的なGC比率への近さ）
    - base_balance: 各塩基の均等度を評価
    - dinucleotide_diversity: ジヌクレオチド（2塩基）の多様性を評価
    - complexity: 配列の複雑さ（繰り返しパターンの少なさ）を評価
    """
    
    def __init__(
        self,
        regex_list,
        max_len,
        min_len,
        num_start_examples,
        tokenizer,
        objectives,
        transform=lambda x: x,
        **kwargs
    ):
        obj_dim = len(objectives)
        super().__init__(tokenizer, obj_dim, max_len, transform, **kwargs)
        self.regex_list = None
        self.min_len = min_len
        self.max_len = max_len
        self.num_start_examples = num_start_examples
        self.max_reward_per_dim = kwargs["max_score_per_dim"]
        self.score_max = kwargs.get("score_max", [1.0] * obj_dim)
        self.objectives = objectives
        
        # 目的関数のパラメータ
        self.ideal_length = kwargs.get("ideal_length", (min_len + max_len) // 2)
        self.ideal_gc_content = kwargs.get("ideal_gc_content", 0.5)  # 50% GC content

    def task_setup(self, *args, **kwargs):
        """タスクセットアップ"""
        return [], []

    def score(self, candidates):
        """
        各候補配列のマルチ目的スコアを計算

        Args:
            candidates (list or np.array): DNAアプタマー配列（文字列形式）

        Returns:
            scores (np.array): マルチ目的スコア。形状: [n_candidates, n_objectives]
        """
        scores_dict = self.compute_scores(candidates, objectives=self.objectives)
        scores = [scores_dict[obj] for obj in self.objectives]
        scores = np.stack(scores, axis=-1).astype(np.float64)
        
        # 正規化
        scores = scores / np.array(self.score_max)
        return scores

    def compute_scores(self, sequences, objectives="length"):
        """
        DNAアプタマー配列のスコアを計算

        Args:
            sequences (list): DNA配列のリスト
            objectives (str or list): 計算する目的関数

        Returns:
            dict: 各目的関数の結果
        """
        dict_return = {}
        
        if "length" in objectives:
            length_scores = self._compute_length_score(sequences)
            dict_return["length"] = length_scores
            
        if "gc_content" in objectives:
            gc_scores = self._compute_gc_content_score(sequences)
            dict_return["gc_content"] = gc_scores
            
        if "base_balance" in objectives:
            balance_scores = self._compute_base_balance_score(sequences)
            dict_return["base_balance"] = balance_scores
            
        if "dinucleotide_diversity" in objectives:
            diversity_scores = self._compute_dinucleotide_diversity_score(sequences)
            dict_return["dinucleotide_diversity"] = diversity_scores
            
        if "complexity" in objectives:
            complexity_scores = self._compute_complexity_score(sequences)
            dict_return["complexity"] = complexity_scores

        # 結果の返却
        if isinstance(objectives, (list, ListConfig)):
            if len(objectives) > 1:
                return dict_return
            else:
                return dict_return[objectives[0]]
        else:
            return dict_return[objectives]

    def _compute_length_score(self, sequences):
        """配列長スコアを計算（理想的な長さへの近さ）"""
        length_scores = []
        for seq in sequences:
            length_diff = abs(len(seq) - self.ideal_length)
            # 理想的な長さに近いほど高いスコア
            score = 1.0 / (1.0 + length_diff / self.ideal_length)
            length_scores.append(score)
        return np.array(length_scores)

    def _compute_gc_content_score(self, sequences):
        """GC含有量スコアを計算（理想的なGC比率への近さ）"""
        gc_scores = []
        for seq in sequences:
            if not self._validate_sequence(seq):
                gc_scores.append(0.0)
                continue
                
            seq_upper = seq.upper()
            gc_count = seq_upper.count('G') + seq_upper.count('C')
            gc_content = gc_count / len(seq) if len(seq) > 0 else 0.0
            
            # 理想的なGC含有量に近いほど高いスコア
            gc_diff = abs(gc_content - self.ideal_gc_content)
            score = 1.0 / (1.0 + gc_diff * 4)  # 4倍のペナルティで敏感に
            gc_scores.append(score)
        return np.array(gc_scores)

    def _compute_base_balance_score(self, sequences):
        """塩基バランススコアを計算（各塩基の均等度）"""
        balance_scores = []
        for seq in sequences:
            if not self._validate_sequence(seq):
                balance_scores.append(0.0)
                continue
                
            seq_upper = seq.upper()
            if len(seq_upper) == 0:
                balance_scores.append(0.0)
                continue
                
            # 各塩基の頻度を計算
            base_counts = {
                'A': seq_upper.count('A'),
                'T': seq_upper.count('T'),
                'G': seq_upper.count('G'),
                'C': seq_upper.count('C')
            }
            
            # 理想的には各塩基が25%ずつ
            ideal_count = len(seq_upper) / 4
            variance = sum((count - ideal_count) ** 2 for count in base_counts.values())
            
            # 分散が小さいほど高いスコア
            score = 1.0 / (1.0 + variance / (ideal_count ** 2))
            balance_scores.append(score)
        return np.array(balance_scores)

    def _compute_dinucleotide_diversity_score(self, sequences):
        """ジヌクレオチド多様性スコアを計算"""
        diversity_scores = []
        for seq in sequences:
            if not self._validate_sequence(seq) or len(seq) < 2:
                diversity_scores.append(0.0)
                continue
                
            seq_upper = seq.upper()
            
            # 全ジヌクレオチドの種類数をカウント
            dinucleotides = set()
            for i in range(len(seq_upper) - 1):
                dinucleotides.add(seq_upper[i:i+2])
            
            # 最大16種類のジヌクレオチドが存在可能（4^2）
            diversity_ratio = len(dinucleotides) / 16.0
            diversity_scores.append(diversity_ratio)
        return np.array(diversity_scores)

    def _compute_complexity_score(self, sequences):
        """配列複雑さスコアを計算（繰り返しパターンの少なさ）"""
        complexity_scores = []
        for seq in sequences:
            if not self._validate_sequence(seq):
                complexity_scores.append(0.0)
                continue
                
            seq_upper = seq.upper()
            if len(seq_upper) < 4:
                complexity_scores.append(0.5)  # 短すぎる場合は中程度のスコア
                continue
            
            # 4塩基の繰り返しパターンをチェック
            tetranucleotides = {}
            for i in range(len(seq_upper) - 3):
                tetra = seq_upper[i:i+4]
                tetranucleotides[tetra] = tetranucleotides.get(tetra, 0) + 1
            
            # 最も頻繁な4塩基パターンの頻度を計算
            max_repeat = max(tetranucleotides.values()) if tetranucleotides else 1
            total_tetras = len(seq_upper) - 3
            
            # 繰り返しが少ないほど高いスコア
            repeat_ratio = max_repeat / total_tetras if total_tetras > 0 else 1.0
            complexity_score = 1.0 - repeat_ratio
            complexity_scores.append(max(0.0, complexity_score))
        return np.array(complexity_scores)

    def _validate_sequence(self, sequence):
        """DNA配列の妥当性をチェック"""
        if not sequence:
            return False
        valid_bases = set('ATCG')
        return all(base.upper() in valid_bases for base in sequence)

    def _generate_random_sequence(self, length):
        """指定された長さのランダムDNA配列を生成"""
        bases = ['A', 'T', 'G', 'C']
        return ''.join(np.random.choice(bases) for _ in range(length))
