#!/bin/bash
# DNAアプタマー生成 - 基本学習ジョブスクリプト
# 使用方法: ./scripts/train_job.sh

set -e

# =============================================================================
# 設定
# =============================================================================

# プロジェクト設定
PROJECT_DIR="/home/matsumoto/raptgfn"
CONDA_ENV="dna_aptamer_test"  # 必要に応じて変更

# 実験設定
EXPERIMENT_NAME="dna_aptamer_training_$(date +%Y%m%d_%H%M%S)"
WANDB_MODE="online"  # online, offline, disabled
SEED=42

# ハードウェア設定
CUDA_VISIBLE_DEVICES="0"  # 使用するGPU
NUM_WORKERS=4

# =============================================================================
# 環境準備
# =============================================================================

echo "=== DNAアプタマー生成 学習ジョブ開始 ==="
echo "開始時刻: $(date)"
echo "プロジェクトディレクトリ: $PROJECT_DIR"
echo "実験名: $EXPERIMENT_NAME"
echo ""

# プロジェクトディレクトリに移動
cd "$PROJECT_DIR"

# Conda環境の活性化
echo "Conda環境を活性化中: $CONDA_ENV"
source ~/anaconda3/etc/profile.d/conda.sh
conda activate "$CONDA_ENV"

# 環境情報表示
echo "Python バージョン: $(python --version)"
echo "PyTorch バージョン: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA利用可能: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "GPU数: $(python -c 'import torch; print(torch.cuda.device_count())')"
echo ""

# GPU設定
export CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES"
echo "使用GPU: $CUDA_VISIBLE_DEVICES"

# =============================================================================
# 学習実行
# =============================================================================

echo "=== 学習開始 ==="

# ログディレクトリ作成
LOG_DIR="outputs/$EXPERIMENT_NAME"
mkdir -p "$LOG_DIR"

# 基本的な二目的最適化（配列長とGC含有量）
echo "--- 配列長とGC含有量の最適化 ---"
python main.py \
    task=simple_length_gc \
    algorithm=mogfn \
    tokenizer=aptamer \
    exp_name="${EXPERIMENT_NAME}_length_gc" \
    seed=$SEED \
    wandb_mode=$WANDB_MODE \
    algorithm.train_steps=2000 \
    algorithm.batch_size=64 \
    algorithm.eval_freq=200 \
    algorithm.pareto_freq=400 \
    2>&1 | tee "$LOG_DIR/length_gc_training.log"

echo ""
echo "--- 塩基バランスと多様性の最適化 ---"
python main.py \
    task=simple_balance_diversity \
    algorithm=mogfn \
    tokenizer=aptamer \
    exp_name="${EXPERIMENT_NAME}_balance_diversity" \
    seed=$SEED \
    wandb_mode=$WANDB_MODE \
    algorithm.train_steps=2000 \
    algorithm.batch_size=64 \
    algorithm.eval_freq=200 \
    algorithm.pareto_freq=400 \
    2>&1 | tee "$LOG_DIR/balance_diversity_training.log"

echo ""
echo "--- 三目的最適化（長さ、GC含有量、複雑さ） ---"
python main.py \
    task=simple_three_objectives \
    algorithm=mogfn \
    tokenizer=aptamer \
    exp_name="${EXPERIMENT_NAME}_three_objectives" \
    seed=$SEED \
    wandb_mode=$WANDB_MODE \
    algorithm.train_steps=3000 \
    algorithm.batch_size=64 \
    algorithm.eval_freq=200 \
    algorithm.pareto_freq=400 \
    2>&1 | tee "$LOG_DIR/three_objectives_training.log"

# =============================================================================
# 結果まとめ
# =============================================================================

echo ""
echo "=== 学習完了 ==="
echo "終了時刻: $(date)"
echo "結果保存場所: $LOG_DIR"

# 結果ファイルの一覧表示
echo ""
echo "生成されたファイル:"
find "outputs" -name "*${EXPERIMENT_NAME#dna_aptamer_training_}*" -type f | sort

echo ""
echo "WandBプロジェクト: raptgfn"
echo "すべての学習が正常に完了しました。"
