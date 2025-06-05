#!/bin/bash
#SBATCH --job-name=dna_aptamer_mogfn
#SBATCH --output=outputs/slurm_%j.out
#SBATCH --error=outputs/slurm_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

# DNAアプタマー生成 - SLURM HPC ジョブスクリプト
# 高性能計算環境での実行用
# 使用方法: sbatch scripts/train_slurm_job.sh

# =============================================================================
# SLURM環境設定
# =============================================================================

echo "=== SLURM ジョブ情報 ==="
echo "ジョブID: $SLURM_JOB_ID"
echo "ノード: $SLURM_JOB_NODELIST"
echo "開始時刻: $(date)"
echo "作業ディレクトリ: $PWD"
echo ""

# 環境変数設定
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export CUDA_VISIBLE_DEVICES=0

# =============================================================================
# プロジェクト設定
# =============================================================================

# プロジェクト設定
PROJECT_DIR="/home/matsumoto/dna_aptamer_mogfn"
CONDA_ENV="dna_aptamer_test"

# 実験設定
EXPERIMENT_NAME="dna_aptamer_hpc_${SLURM_JOB_ID}"
WANDB_MODE="online"
SEED=${SLURM_ARRAY_TASK_ID:-42}

# 学習パラメータ（HPCでの長時間学習用）
TRAIN_STEPS=10000
BATCH_SIZE=128
EVAL_FREQ=1000
PARETO_FREQ=2000

# =============================================================================
# 環境準備
# =============================================================================

echo "=== 環境準備 ==="

# プロジェクトディレクトリに移動
cd "$PROJECT_DIR"

# モジュールロード（HPC環境に応じて調整）
# module load cuda/11.8
# module load python/3.10

# Conda環境活性化
source ~/anaconda3/etc/profile.d/conda.sh
conda activate "$CONDA_ENV"

# 環境情報表示
echo "Python: $(python --version)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA利用可能: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "GPU数: $(python -c 'import torch; print(torch.cuda.device_count())')"
echo "CPUコア数: $SLURM_CPUS_PER_TASK"
echo "メモリ: ${SLURM_MEM_PER_NODE}MB"
echo ""

# GPU情報
if command -v nvidia-smi &> /dev/null; then
    echo "GPU情報:"
    nvidia-smi
    echo ""
fi

# 出力ディレクトリ作成
mkdir -p "outputs/slurm_jobs"
LOG_DIR="outputs/slurm_jobs/job_${SLURM_JOB_ID}"
mkdir -p "$LOG_DIR"

# =============================================================================
# 学習実行
# =============================================================================

echo "=== 学習開始 ==="

# 学習実行関数
run_hpc_training() {
    local task=$1
    local suffix=$2
    
    echo "--- $task の学習開始 ---"
    
    python main.py \
        task=$task \
        algorithm=mogfn \
        tokenizer=aptamer \
        exp_name="${EXPERIMENT_NAME}_${suffix}" \
        seed=$SEED \
        wandb_mode=$WANDB_MODE \
        algorithm.train_steps=$TRAIN_STEPS \
        algorithm.batch_size=$BATCH_SIZE \
        algorithm.pi_lr=0.0001 \
        algorithm.z_lr=0.001 \
        algorithm.eval_freq=$EVAL_FREQ \
        algorithm.pareto_freq=$PARETO_FREQ \
        algorithm.num_samples=1024 \
        algorithm.num_pareto_points=2000 \
        algorithm.k=20 \
        hydra.run.dir="$LOG_DIR/${suffix}" \
        2>&1 | tee "$LOG_DIR/${suffix}_training.log"
    
    local exit_code=$?
    echo "学習終了: $task (exit code: $exit_code)"
    return $exit_code
}

# メイン学習タスク実行
SUCCESS_COUNT=0
TOTAL_COUNT=0

# 基本タスク（軽量）
echo "=== 基本タスクでのウォームアップ ==="
if run_hpc_training "simple_length_gc" "length_gc"; then
    ((SUCCESS_COUNT++))
fi
((TOTAL_COUNT++))

if run_hpc_training "simple_balance_diversity" "balance_diversity"; then
    ((SUCCESS_COUNT++))
fi
((TOTAL_COUNT++))

# 高度なタスク（計算量大）
echo "=== 高度なタスクでの本格学習 ==="
if run_hpc_training "simple_three_objectives" "three_objectives"; then
    ((SUCCESS_COUNT++))
fi
((TOTAL_COUNT++))

# NUPACK使用タスク（利用可能な場合）
if python -c "import nupack" 2>/dev/null; then
    echo "=== NUPACK使用タスク ==="
    
    if run_hpc_training "nupack_energy_length" "nupack_energy_length"; then
        ((SUCCESS_COUNT++))
    fi
    ((TOTAL_COUNT++))
    
    if run_hpc_training "nupack_energy_pairs" "nupack_energy_pairs"; then
        ((SUCCESS_COUNT++))
    fi
    ((TOTAL_COUNT++))
else
    echo "NUPACK が利用できません。簡単なタスクのみ実行します。"
fi

# =============================================================================
# 後処理と結果分析
# =============================================================================

echo "=== 後処理 ==="

# 結果サマリー作成
cat > "$LOG_DIR/job_summary.txt" << EOF
=== SLURM ジョブサマリー ===
ジョブID: $SLURM_JOB_ID
ノード: $SLURM_JOB_NODELIST
開始時刻: $(cat $LOG_DIR/../start_time.txt 2>/dev/null || echo "Unknown")
終了時刻: $(date)
実行時間: $SECONDS 秒

実験設定:
- 実験名: $EXPERIMENT_NAME
- シード: $SEED
- 学習ステップ: $TRAIN_STEPS
- バッチサイズ: $BATCH_SIZE

結果:
- 総タスク数: $TOTAL_COUNT
- 成功タスク数: $SUCCESS_COUNT
- 失敗タスク数: $((TOTAL_COUNT - SUCCESS_COUNT))

生成ファイル:
$(find $LOG_DIR -type f -name "*.log" -o -name "*.pkl*" -o -name "*.yaml" | sort)
EOF

# リソース使用量記録
if command -v sacct &> /dev/null; then
    echo "=== リソース使用量 ===" >> "$LOG_DIR/job_summary.txt"
    sacct -j $SLURM_JOB_ID --format=JobID,JobName,Partition,AllocCPUS,State,ExitCode,Elapsed,MaxRSS,MaxVMSize >> "$LOG_DIR/job_summary.txt"
fi

# 結果ファイル圧縮（オプション）
if [ "$SUCCESS_COUNT" -gt 0 ]; then
    echo "結果ファイルを圧縮中..."
    tar -czf "$LOG_DIR/results_${SLURM_JOB_ID}.tar.gz" -C "$LOG_DIR" \
        --exclude="*.log" --exclude="wandb" .
fi

# =============================================================================
# 完了処理
# =============================================================================

echo "=== ジョブ完了 ==="
echo "終了時刻: $(date)"
echo "総実行時間: $SECONDS 秒"
echo "成功率: $SUCCESS_COUNT/$TOTAL_COUNT"
echo "結果保存場所: $LOG_DIR"

# 結果サマリー表示
cat "$LOG_DIR/job_summary.txt"

# 正常終了の場合は0、一部失敗は1、全て失敗は2で終了
if [ "$SUCCESS_COUNT" -eq "$TOTAL_COUNT" ]; then
    echo "全てのタスクが正常に完了しました"
    exit 0
elif [ "$SUCCESS_COUNT" -gt 0 ]; then
    echo "一部のタスクが失敗しました"
    exit 1
else
    echo "全てのタスクが失敗しました"
    exit 2
fi
