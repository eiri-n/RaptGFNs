#!/bin/bash
# DNAアプタマー生成 - デバッグ・テスト用軽量ジョブスクリプト
# 短時間で動作確認を行うためのスクリプト
# 使用方法: ./scripts/train_debug_job.sh

set -e

# =============================================================================
# 設定（デバッグ用に軽量化）
# =============================================================================

PROJECT_DIR="/home/matsumoto/dna_aptamer_mogfn"
CONDA_ENV="dna_aptamer_test"

EXPERIMENT_NAME="debug_test_$(date +%Y%m%d_%H%M%S)"
WANDB_MODE="disabled"  # デバッグ時はWandBを無効化
SEED=42

# 軽量学習パラメータ
TRAIN_STEPS=100
BATCH_SIZE=16
EVAL_FREQ=50
PARETO_FREQ=50

# =============================================================================
# 関数定義
# =============================================================================

check_prerequisites() {
    echo "=== 前提条件チェック ==="
    
    # Python環境チェック
    if ! command -v python &> /dev/null; then
        echo "ERROR: Pythonが見つかりません"
        exit 1
    fi
    
    # 必要なパッケージチェック
    local packages=("torch" "numpy" "hydra" "omegaconf")
    for pkg in "${packages[@]}"; do
        if ! python -c "import $pkg" 2>/dev/null; then
            echo "WARNING: $pkg がインストールされていません"
        else
            echo "OK: $pkg"
        fi
    done
    
    # GPU利用可能性チェック
    local cuda_available=$(python -c "import torch; print(torch.cuda.is_available())" 2>/dev/null || echo "False")
    echo "CUDA利用可能: $cuda_available"
    
    # ディスク容量チェック
    local free_space=$(df -h "$PROJECT_DIR" | awk 'NR==2 {print $4}')
    echo "利用可能ディスク容量: $free_space"
    
    echo ""
}

run_quick_test() {
    local task=$1
    local suffix=$2
    
    echo "--- クイックテスト: $task ---"
    
    timeout 300 python main.py \
        task=$task \
        algorithm=mogfn \
        tokenizer=aptamer \
        exp_name="${EXPERIMENT_NAME}_${suffix}" \
        seed=$SEED \
        wandb_mode=$WANDB_MODE \
        algorithm.train_steps=$TRAIN_STEPS \
        algorithm.batch_size=$BATCH_SIZE \
        algorithm.eval_freq=$EVAL_FREQ \
        algorithm.pareto_freq=$PARETO_FREQ \
        algorithm.num_samples=32 \
        algorithm.num_pareto_points=50 \
        hydra.job.chdir=False \
        2>&1 | tee "outputs/debug_${suffix}.log"
    
    local exit_code=$?
    if [ $exit_code -eq 0 ]; then
        echo "✓ $task テスト成功"
        return 0
    elif [ $exit_code -eq 124 ]; then
        echo "⚠ $task テストタイムアウト（5分）"
        return 1
    else
        echo "✗ $task テスト失敗 (exit code: $exit_code)"
        return 1
    fi
}

# =============================================================================
# メイン処理
# =============================================================================

echo "=== DNAアプタマー生成 デバッグジョブ開始 ==="
echo "開始時刻: $(date)"
echo ""

# 環境準備
cd "$PROJECT_DIR"
source ~/anaconda3/etc/profile.d/conda.sh
conda activate "$CONDA_ENV"

# 前提条件チェック
check_prerequisites

# 出力ディレクトリ準備
mkdir -p outputs/debug_logs

# =============================================================================
# クイックテスト実行
# =============================================================================

echo "=== クイックテスト開始（各タスク最大5分） ==="

TEST_RESULTS=()

# 最も軽量なタスクから順にテスト
echo "1. 最軽量タスクテスト"
if run_quick_test "simple_length_gc" "length_gc"; then
    TEST_RESULTS+=("simple_length_gc:PASS")
else
    TEST_RESULTS+=("simple_length_gc:FAIL")
fi

echo ""
echo "2. 中程度タスクテスト"
if run_quick_test "simple_balance_diversity" "balance_diversity"; then
    TEST_RESULTS+=("simple_balance_diversity:PASS")
else
    TEST_RESULTS+=("simple_balance_diversity:FAIL")
fi

echo ""
echo "3. 重量タスクテスト"
if run_quick_test "simple_three_objectives" "three_objectives"; then
    TEST_RESULTS+=("simple_three_objectives:PASS")
else
    TEST_RESULTS+=("simple_three_objectives:FAIL")
fi

# =============================================================================
# 設定妥当性チェック
# =============================================================================

echo ""
echo "=== 設定妥当性チェック ==="

# 設定ファイルの構文チェック
echo "設定ファイル構文チェック:"
for config_file in configs/main.yaml configs/algorithm/mogfn.yaml configs/task/*.yaml; do
    if python -c "
import yaml
try:
    with open('$config_file') as f:
        yaml.safe_load(f)
    print('✓ $config_file')
except Exception as e:
    print('✗ $config_file: ' + str(e))
"; then
        :
    fi
done

# モデルサイズ推定
echo ""
echo "モデルパラメータ数推定:"
python -c "
import sys
sys.path.append('src')
from dna_aptamer_mogfn.algorithms.conditional_transformer import CondGFNTransformer
model = CondGFNTransformer(vocab_size=5, pad_token=0, max_len=50)
params = sum(p.numel() for p in model.parameters())
print(f'総パラメータ数: {params:,}')
print(f'推定メモリ使用量: {params * 4 / 1024**2:.1f} MB')
" 2>/dev/null || echo "モデルサイズ推定失敗"

# =============================================================================
# 結果まとめ
# =============================================================================

echo ""
echo "=== テスト結果まとめ ==="
echo "終了時刻: $(date)"

PASS_COUNT=0
TOTAL_COUNT=${#TEST_RESULTS[@]}

for result in "${TEST_RESULTS[@]}"; do
    task=${result%:*}
    status=${result#*:}
    if [ "$status" = "PASS" ]; then
        echo "✓ $task"
        ((PASS_COUNT++))
    else
        echo "✗ $task"
    fi
done

echo ""
echo "成功率: $PASS_COUNT/$TOTAL_COUNT"

# 推奨事項
echo ""
echo "=== 推奨事項 ==="
if [ $PASS_COUNT -eq $TOTAL_COUNT ]; then
    echo "🎉 全てのテストが成功しました！"
    echo "本格的な学習を実行できます:"
    echo "  ./scripts/train_job.sh"
elif [ $PASS_COUNT -gt 0 ]; then
    echo "⚠ 一部のテストが失敗しました"
    echo "失敗したタスクの設定を確認してください"
    echo "成功したタスクのみで学習を実行することもできます"
else
    echo "❌ 全てのテストが失敗しました"
    echo "以下を確認してください:"
    echo "1. 環境の依存関係"
    echo "2. GPU利用可能性"
    echo "3. 設定ファイルの構文"
    echo "4. ログファイルのエラーメッセージ"
fi

# ログファイルの場所を表示
echo ""
echo "詳細ログ:"
for log_file in outputs/debug_*.log; do
    if [ -f "$log_file" ]; then
        echo "  $log_file"
    fi
done

exit $((TOTAL_COUNT - PASS_COUNT))
