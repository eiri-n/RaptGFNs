# DNAアプタマー生成 ジョブスクリプト使用ガイド

このディレクトリには、DNAアプタマー生成プロジェクトの学習を実行するための複数のジョブスクリプトが含まれています。

## スクリプト一覧

### 1. `train_debug_job.sh` - デバッグ・動作確認用
```bash
./scripts/train_debug_job.sh
```

**用途**: 初回実行時の動作確認、環境テスト
**特徴**:
- 軽量パラメータ（100ステップ）
- 5分タイムアウト
- WandBログ無効
- 前提条件チェック機能

**使用場面**: 
- 初めてプロジェクトを実行する時
- 環境設定に問題がないか確認したい時
- 新しい設定を試す前の動作確認

### 2. `train_job.sh` - 基本学習用
```bash
./scripts/train_job.sh
```

**用途**: 標準的な学習実行
**特徴**:
- 2000-3000ステップ学習
- 3つの基本タスク実行
- WandBログ記録
- 詳細な環境情報出力

**使用場面**:
- 通常の学習実行
- 基本的な結果が欲しい時
- 個人PC・ワークステーション環境

### 3. `train_batch_job.sh` - バッチ処理・ハイパーパラメータ探索用
```bash
./scripts/train_batch_job.sh
```

**用途**: 複数設定での大規模実験
**特徴**:
- 複数シード・バッチサイズ・学習率の組み合わせ
- 自動結果分析
- GPU使用率監視
- 失敗実験の自動記録

**使用場面**:
- 最適なハイパーパラメータを見つけたい時
- 論文用の統計的に有意な結果が必要な時
- 長時間実行可能な環境

### 4. `train_slurm_job.sh` - HPC環境用
```bash
sbatch scripts/train_slurm_job.sh
```

**用途**: 高性能計算クラスタでの実行
**特徴**:
- SLURM ジョブスケジューラ対応
- 長時間学習（10000ステップ）
- リソース使用量記録
- 自動結果圧縮

**使用場面**:
- 大学・研究機関のHPCクラスタ
- 大規模な学習が必要な時
- バッチジョブとしての実行

### 5. `run_experiments.sh` - 既存の簡易実行スクリプト
```bash
./scripts/run_experiments.sh
```

**用途**: 簡単な実験実行
**特徴**: シンプルな3タスク実行

## 推奨実行順序

### 初回実行
```bash
# 1. 動作確認
./scripts/train_debug_job.sh

# 2. 問題なければ基本学習
./scripts/train_job.sh
```

### 本格的な研究
```bash
# ハイパーパラメータ探索
./scripts/train_batch_job.sh

# または HPC環境での大規模学習
sbatch scripts/train_slurm_job.sh
```

## 出力結果の場所

### 基本的な出力先
```
outputs/
├── dna_aptamer_experiment/          # 設定ファイルのデフォルト
├── <experiment_name>_<task>/        # 各実験の結果
│   ├── .hydra/                     # Hydra設定
│   ├── main.log                    # 実行ログ
│   └── wandb/                      # WandB記録
├── batch_logs_<timestamp>/          # バッチ処理結果
└── slurm_jobs/                     # SLURM ジョブ結果
```

### 重要なファイル
- `mogfn_state.pkl.gz`: パレートフロンティア状態（配列と報酬）
- `wandb/*/files/wandb-summary.json`: WandB結果サマリー
- `*_training.log`: 詳細な学習ログ

## トラブルシューティング

### よくあるエラーと対処法

1. **CUDA device error**
   ```bash
   # GPU使用状況確認
   nvidia-smi
   # GPU IDを指定して実行
   CUDA_VISIBLE_DEVICES=0 ./scripts/train_job.sh
   ```

2. **Memory error**
   - バッチサイズを小さくする: `algorithm.batch_size=32`
   - 最大長を短くする: `task.max_len=30`

3. **Import error**
   ```bash
   # 環境確認
   conda activate dna_aptamer_test
   pip install -r requirements.txt
   ```

4. **WandB error**
   - オフラインモードで実行: `wandb_mode=offline`
   - 無効化: `wandb_mode=disabled`

### パフォーマンス調整

- **学習速度向上**: バッチサイズ増加、学習率調整
- **メモリ使用量削減**: バッチサイズ減少、max_len短縮
- **結果精度向上**: train_steps増加、num_pareto_points増加

## カスタマイズ

各スクリプトの冒頭設定セクションで以下を調整可能:

- `TRAIN_STEPS`: 学習ステップ数
- `BATCH_SIZE`: バッチサイズ
- `EVAL_FREQ`: 評価頻度
- `WANDB_MODE`: WandBモード
- `CUDA_VISIBLE_DEVICES`: 使用GPU

## 連絡先・サポート

問題が発生した場合は、以下を確認してください:
1. デバッグスクリプトの結果
2. ログファイルのエラーメッセージ
3. 環境の依存関係

詳細は `README.md` または設定ファイル `configs/` を参照してください。
