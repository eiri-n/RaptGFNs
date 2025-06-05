# DNAアプタマー生成のためのMulti-Objective GFlowNets

このプロジェクトは、学術論文「Multi-Objective GFlowNets」に基づいて、DNAアプタマー生成に特化した機能を抽出・再構築したものです。

## 概要

Multi-Objective Generative Flow Networks (MOGFNs)を使用してDNAアプタマーを生成し、複数の目的関数を同時に最適化します。

## 機能

- **MOGFN-PC**: パレート条件付きアプローチによるマルチ目的最適化
- **DNAアプタマー生成**: 配列の特性に基づく目的関数計算
- **複数の目的関数**:
  - 配列長の最適化（理想的な長さへの近さ）
  - GC含有量の最適化（理想的なGC比率への近さ）
  - 塩基バランスの評価（各塩基の均等度）
  - ジヌクレオチド多様性の評価
  - 配列複雑さの評価（繰り返しパターンの少なさ）

## インストール

### 前提条件

- Python 3.8以降
- CUDA対応GPU（推奨）

### 1. リポジトリのクローンとインストール

```bash
git clone <repository-url>
cd dna_aptamer_mogfn
pip install -e .
```

### 2. 依存関係のインストール

```bash
pip install -r requirements.txt
```

## 使用方法

### 基本的な実行

```bash
# デフォルト設定（配列長とGC含有量の最適化）
python main.py

# 塩基バランスと多様性の最適化
python main.py task=simple_balance_diversity

# 三目的最適化（長さ、GC含有量、複雑さ）
python main.py task=simple_three_objectives
```

### バッチ実行

```bash
# 全ての実験を実行
bash scripts/run_experiments.sh
```

### コマンドライン引数のカスタマイズ

```bash
# 実験名とグループ名を指定
python main.py task=simple_length_gc algorithm=mogfn tokenizer=aptamer \
    exp_name=my_experiment group_name=aptamer_generation

# ハイパーパラメータの調整
python main.py task=simple_length_gc algorithm=mogfn tokenizer=aptamer \
    algorithm.train_steps=200 algorithm.batch_size=64

# 配列長の制限
python main.py task=simple_length_gc algorithm=mogfn tokenizer=aptamer \
    task.min_len=20 task.max_len=50
```

### 設定ファイルのカスタマイズ

設定ファイルは `configs/` ディレクトリにあります：

- `configs/task/`: タスク設定（目的関数、制約など）
- `configs/algorithm/`: アルゴリズム設定（ハイパーパラメータなど）
- `configs/tokenizer/`: トークナイザー設定

## 利用可能な目的関数

### SimpleDNATask で利用可能な目的関数

- **length**: 配列長の最適化（理想的な長さへの近さ）
- **gc_content**: GC含有量の最適化（理想的なGC比率への近さ）
- **base_balance**: 塩基バランスの評価（各塩基の均等度）
- **dinucleotide_diversity**: ジヌクレオチド多様性の評価
- **complexity**: 配列複雑さの評価（繰り返しパターンの少なさ）

## 出力

実行結果は以下の形式で出力されます：

- **生成されたDNAアプタマー配列**: 最適化されたDNA配列
- **パレートフロンティア**: 複数目的の最適解集合
- **ハイパーボリューム**: 最適化性能の指標
- **スコア情報**: 各目的関数の評価値

## 設定例

### 配列長とGC含有量の最適化
```yaml
# configs/task/simple_length_gc.yaml
objectives:
  - length      # 配列長の最適化
  - gc_content  # GC含有量の最適化
min_len: 30
max_len: 60
ideal_length: 45
ideal_gc_content: 0.5
```

### 三目的最適化
```yaml
# configs/task/simple_three_objectives.yaml
objectives:
  - length      # 配列長
  - gc_content  # GC含有量
  - complexity  # 配列複雑さ
```

## トラブルシューティング

### 依存関係のエラー
- Python 3.8以降を使用していることを確認してください
- PyTorchが正しくインストールされていることを確認してください

### メモリ不足エラー
- バッチサイズを小さくしてください: `algorithm.batch_size=32`
- GPU使用時はVRAMを確認してください

### 収束が遅い場合
- 学習ステップ数を増やしてください: `algorithm.train_steps=500`
- 学習率を調整してください: `algorithm.pi_lr=0.0005`

## 特徴

### NUPACKフリー設計
このプロジェクトは元々のNUPACK依存を削除し、より簡単に実行できるように設計されています：

- **軽量な目的関数**: 構造計算なしで配列特性を評価
- **クロスプラットフォーム**: Linux以外の環境でも実行可能
- **高速実行**: 複雑な構造計算を回避することで高速化

### カスタマイズ可能
- 新しい目的関数を簡単に追加可能
- 配列の制約条件を柔軟に設定
- ハイパーパラメータの細かい調整が可能

## ライセンス

MIT License - 元のMulti-Objective GFlowNetsプロジェクトのライセンスに従います。

## 参考文献

```bibtex
@article{jain2023multi,
  title={Multi-Objective GFlowNets},
  author={Jain, Moksh and Raparthy, Sharath and Hernandez-Garcia, Alex and Rector-Brooks, Jarrid and Bengio, Yoshua and Miret, Santiago and Bengio, Emmanuel},
  journal={arXiv preprint arXiv:2310.12372},
  year={2023}
}
```
