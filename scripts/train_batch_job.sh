#!/bin/bash
# DNAアプタマー生成 - バッチ処理ジョブスクリプト
# 複数の設定での並列実行や詳細な評価を行うスクリプト
# 使用方法: ./scripts/train_batch_job.sh

set -e

# =============================================================================
# 設定
# =============================================================================

# プロジェクト設定
PROJECT_DIR="/home/matsumoto/raptgfn"
CONDA_ENV="dna_aptamer_test"

# 実験設定
BASE_EXP_NAME="dna_aptamer_batch_$(date +%Y%m%d_%H%M%S)"
WANDB_MODE="online"
SEEDS=(42 123 456)  # 複数のシードで実行
GPU_IDS=(0)  # 利用可能なGPU ID

# 学習パラメータ
TRAIN_STEPS=5000
BATCH_SIZES=(32 64 128)
LEARNING_RATES=(0.0001 0.0005 0.001)

# =============================================================================
# 関数定義
# =============================================================================

# ログ出力関数
log_info() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] INFO: $1"
}

log_error() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: $1" >&2
}

# GPU使用率チェック関数
check_gpu_usage() {
    if command -v nvidia-smi &> /dev/null; then
        nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
    else
        echo "nvidia-smi not available"
    fi
}

# 学習実行関数
run_training() {
    local task=$1
    local exp_name=$2
    local seed=$3
    local batch_size=$4
    local lr=$5
    local gpu_id=$6
    
    log_info "学習開始: $exp_name (GPU:$gpu_id, seed:$seed, batch:$batch_size, lr:$lr)"
    
    CUDA_VISIBLE_DEVICES=$gpu_id python main.py \
        task=$task \
        algorithm=mogfn \
        tokenizer=aptamer \
        exp_name="$exp_name" \
        seed=$seed \
        wandb_mode=$WANDB_MODE \
        algorithm.train_steps=$TRAIN_STEPS \
        algorithm.batch_size=$batch_size \
        algorithm.pi_lr=$lr \
        algorithm.eval_freq=500 \
        algorithm.pareto_freq=1000 \
        algorithm.num_samples=512 \
        algorithm.num_pareto_points=1000 \
        2>&1 | tee "outputs/${exp_name}_training.log"
    
    local exit_code=$?
    if [ $exit_code -eq 0 ]; then
        log_info "学習完了: $exp_name"
    else
        log_error "学習失敗: $exp_name (exit code: $exit_code)"
    fi
    
    return $exit_code
}

# =============================================================================
# メイン処理
# =============================================================================

log_info "=== DNAアプタマー生成 バッチ学習ジョブ開始 ==="
log_info "ベース実験名: $BASE_EXP_NAME"

# 環境準備
cd "$PROJECT_DIR"
source ~/anaconda3/etc/profile.d/conda.sh
conda activate "$CONDA_ENV"

log_info "環境情報:"
log_info "Python: $(python --version)"
log_info "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
log_info "CUDA利用可能: $(python -c 'import torch; print(torch.cuda.is_available())')"

# GPU情報表示
log_info "GPU使用状況:"
check_gpu_usage

# ログディレクトリ作成
BATCH_LOG_DIR="outputs/batch_logs_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BATCH_LOG_DIR"

# =============================================================================
# バッチ学習実行
# =============================================================================

# タスクリスト
TASKS=("simple_length_gc" "simple_balance_diversity" "simple_three_objectives")

# 実験カウンタ
total_experiments=0
successful_experiments=0
failed_experiments=0

# 実験実行
for task in "${TASKS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        for batch_size in "${BATCH_SIZES[@]}"; do
            for lr in "${LEARNING_RATES[@]}"; do
                # GPU選択（ラウンドロビン）
                gpu_id=${GPU_IDS[$((total_experiments % ${#GPU_IDS[@]}))]}
                
                # 実験名生成
                exp_name="${BASE_EXP_NAME}_${task}_s${seed}_b${batch_size}_lr${lr//./_}"
                
                log_info "実験 $((total_experiments + 1)): $exp_name"
                
                # GPU使用率チェック（高負荷時は待機）
                while true; do
                    gpu_usage=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i $gpu_id 2>/dev/null || echo "0")
                    if [ "$gpu_usage" -lt 80 ]; then
                        break
                    fi
                    log_info "GPU $gpu_id の使用率が高いため待機中... (${gpu_usage}%)"
                    sleep 30
                done
                
                # 学習実行
                if run_training "$task" "$exp_name" "$seed" "$batch_size" "$lr" "$gpu_id"; then
                    ((successful_experiments++))
                else
                    ((failed_experiments++))
                fi
                
                ((total_experiments++))
                
                # 短い休憩
                sleep 5
            done
        done
    done
done

# =============================================================================
# 結果集計と分析
# =============================================================================

log_info "=== バッチ学習完了 ==="
log_info "総実験数: $total_experiments"
log_info "成功: $successful_experiments"
log_info "失敗: $failed_experiments"

# 結果分析スクリプト作成
cat > "$BATCH_LOG_DIR/analyze_results.py" << 'EOF'
#!/usr/bin/env python3
"""バッチ学習結果の分析スクリプト"""

import os
import glob
import json
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def analyze_batch_results(base_exp_name):
    """バッチ学習結果を分析"""
    results = []
    
    # 結果ファイルを検索
    pattern = f"outputs/*{base_exp_name.split('_')[-1]}*/wandb/*/files/wandb-summary.json"
    summary_files = glob.glob(pattern)
    
    for summary_file in summary_files:
        try:
            with open(summary_file, 'r') as f:
                data = json.load(f)
            
            # 実験名から設定を抽出
            exp_path = Path(summary_file).parent.parent.parent.name
            parts = exp_path.split('_')
            
            result = {
                'experiment': exp_path,
                'task': '_'.join(parts[3:-3]),
                'seed': int(parts[-3][1:]),
                'batch_size': int(parts[-2][1:]),
                'learning_rate': float(parts[-1].replace('lr', '').replace('_', '.')),
                'final_hypervolume': data.get('hypervolume', 0),
                'final_loss': data.get('loss', float('inf')),
                'training_time': data.get('_runtime', 0)
            }
            results.append(result)
            
        except Exception as e:
            print(f"Error processing {summary_file}: {e}")
    
    if not results:
        print("結果ファイルが見つかりませんでした")
        return
    
    df = pd.DataFrame(results)
    
    # 結果保存
    df.to_csv(f"{base_exp_name}_analysis.csv", index=False)
    
    # ベストパフォーマンスの表示
    print("=== ベストパフォーマンス ===")
    best_by_task = df.groupby('task')['final_hypervolume'].max()
    for task, best_hv in best_by_task.items():
        best_row = df[(df['task'] == task) & (df['final_hypervolume'] == best_hv)].iloc[0]
        print(f"{task}: HV={best_hv:.4f}, seed={best_row['seed']}, "
              f"batch={best_row['batch_size']}, lr={best_row['learning_rate']}")
    
    # 可視化
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Hypervolume vs Batch Size
    df.groupby(['task', 'batch_size'])['final_hypervolume'].mean().unstack().plot(
        kind='bar', ax=axes[0, 0], title='Hypervolume vs Batch Size')
    
    # Hypervolume vs Learning Rate
    df.groupby(['task', 'learning_rate'])['final_hypervolume'].mean().unstack().plot(
        kind='bar', ax=axes[0, 1], title='Hypervolume vs Learning Rate')
    
    # Training Time vs Batch Size
    df.groupby(['task', 'batch_size'])['training_time'].mean().unstack().plot(
        kind='bar', ax=axes[1, 0], title='Training Time vs Batch Size')
    
    # Loss vs Hypervolume
    for task in df['task'].unique():
        task_df = df[df['task'] == task]
        axes[1, 1].scatter(task_df['final_loss'], task_df['final_hypervolume'], 
                          label=task, alpha=0.7)
    axes[1, 1].set_xlabel('Final Loss')
    axes[1, 1].set_ylabel('Final Hypervolume')
    axes[1, 1].set_title('Loss vs Hypervolume')
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig(f"{base_exp_name}_analysis.png", dpi=300, bbox_inches='tight')
    print(f"分析完了: {base_exp_name}_analysis.csv, {base_exp_name}_analysis.png")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        analyze_batch_results(sys.argv[1])
    else:
        print("使用方法: python analyze_results.py <base_exp_name>")
EOF

# 分析実行
log_info "結果分析を実行中..."
cd "$BATCH_LOG_DIR"
python analyze_results.py "$BASE_EXP_NAME"

log_info "バッチ学習ジョブが完了しました"
log_info "結果分析ファイル: $BATCH_LOG_DIR"
