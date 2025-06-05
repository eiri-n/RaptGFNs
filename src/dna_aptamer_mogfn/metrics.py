"""
DNAアプタマー生成用メトリクス計算
"""

import numpy as np

try:
    from pymoo.factory import get_performance_indicator
    PYMOO_AVAILABLE = True
except ImportError:
    try:
        from pymoo.indicators.hv import HV
        PYMOO_AVAILABLE = True
    except ImportError:
        PYMOO_AVAILABLE = False

from dna_aptamer_mogfn.algorithms.mogfn_utils import compute_hypervolume


def get_all_metrics(solutions, eval_metrics, **kwargs):
    """
    パレートフロンティアの解に対してメトリクスを計算
    
    Args:
        solutions: パレートフロンティアの解
        eval_metrics: 評価するメトリクスのリスト
        **kwargs: 追加パラメータ
    
    Returns:
        dict: 計算されたメトリクス
    """
    metrics = {}
    
    if "hypervolume" in eval_metrics:
        if PYMOO_AVAILABLE and "hv_ref" in kwargs:
            try:
                hv_indicator = get_performance_indicator('hv', ref_point=kwargs["hv_ref"])
                # pymooは最小化を想定するため負値にする
                metrics["hypervolume"] = hv_indicator.do(-solutions)
            except:
                # フォールバック実装
                metrics["hypervolume"] = compute_hypervolume(solutions, kwargs.get("hv_ref"))
        else:
            # 簡単なハイパーボリューム計算
            metrics["hypervolume"] = compute_hypervolume(solutions, kwargs.get("hv_ref"))
    
    if "r2" in eval_metrics and "r2_prefs" in kwargs and "num_obj" in kwargs:
        try:
            metrics["r2"] = r2_indicator_set(kwargs["r2_prefs"], solutions, np.ones(kwargs["num_obj"]))
        except:
            metrics["r2"] = 0.0
    
    if "hsri" in eval_metrics and "num_obj" in kwargs:
        try:
            metrics["hsri"] = calculate_simple_hsr(solutions, kwargs["num_obj"])
        except:
            metrics["hsri"] = 0.0
    
    # 追加のDNAアプタマー固有メトリクス
    if "diversity" in eval_metrics:
        metrics["diversity"] = calculate_diversity(solutions)
    
    if "coverage" in eval_metrics:
        metrics["coverage"] = calculate_objective_space_coverage(solutions)
    
    return metrics


def r2_indicator_set(preferences, solutions, weights):
    """R2指標の簡単な実装"""
    if len(solutions) == 0:
        return 0.0
    
    # 簡単なR2近似
    r2_values = []
    for pref in preferences:
        weighted_solutions = solutions * pref
        max_vals = np.max(weighted_solutions, axis=0)
        r2_values.append(np.sum(max_vals))
    
    return np.mean(r2_values)


def calculate_simple_hsr(solutions, num_obj):
    """簡単なHSR（Hypervolume-based Success Rate）計算"""
    if len(solutions) == 0:
        return 0.0
    
    # 目的空間の範囲を計算
    obj_ranges = np.max(solutions, axis=0) - np.min(solutions, axis=0)
    
    # 各目的の正規化された範囲の積
    normalized_ranges = obj_ranges / (np.max(obj_ranges) + 1e-8)
    hsr = np.prod(normalized_ranges)
    
    return float(hsr)


def calculate_diversity(solutions):
    """解の多様性を計算"""
    if len(solutions) <= 1:
        return 0.0
    
    # ペアワイズ距離の平均
    distances = []
    for i in range(len(solutions)):
        for j in range(i+1, len(solutions)):
            dist = np.linalg.norm(solutions[i] - solutions[j])
            distances.append(dist)
    
    return np.mean(distances) if distances else 0.0


def calculate_objective_space_coverage(solutions):
    """目的空間のカバレッジを計算"""
    if len(solutions) == 0:
        return 0.0
    
    # 各目的の範囲を計算
    ranges = np.max(solutions, axis=0) - np.min(solutions, axis=0)
    
    # 正規化されたカバレッジ
    normalized_ranges = ranges / (1.0 + ranges)  # 0-1の範囲に正規化
    
    return np.mean(normalized_ranges)


def calculate_hypervolume_improvement(current_solutions, previous_solutions, reference_point=None):
    """ハイパーボリュームの改善を計算"""
    if reference_point is None:
        all_solutions = np.vstack([current_solutions, previous_solutions]) if len(previous_solutions) > 0 else current_solutions
        reference_point = np.min(all_solutions, axis=0) - 0.1
    
    current_hv = compute_hypervolume(current_solutions, reference_point)
    previous_hv = compute_hypervolume(previous_solutions, reference_point) if len(previous_solutions) > 0 else 0.0
    
    return current_hv - previous_hv
