#!/bin/bash
# DNAアプタマー生成実行スクリプト

set -e

echo "=== DNAアプタマー生成用Multi-Objective GFlowNets ==="
echo "プロジェクトディレクトリ: $(pwd)"
echo "Python バージョン: $(python --version)"
echo "PyTorch バージョン: $(python -c 'import torch; print(torch.__version__)')"
echo ""

# NUPACKを使わない簡単なタスクでの実行
echo "=== 配列長とGC含有量の最適化 ==="
python main.py task=simple_length_gc algorithm=mogfn tokenizer=aptamer \
    exp_name=length_gc_optimization

echo ""
echo "=== 塩基バランスと多様性の最適化 ==="
python main.py task=simple_balance_diversity algorithm=mogfn tokenizer=aptamer \
    exp_name=balance_diversity_optimization

echo ""
echo "=== 三目的最適化（長さ、GC含有量、複雑さ） ==="
python main.py task=simple_three_objectives algorithm=mogfn tokenizer=aptamer \
    exp_name=three_objectives_optimization

echo ""
echo "=== 全実験完了 ==="
