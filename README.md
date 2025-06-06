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
cd raptgfn
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

## コードの概略説明とコードマップ

### プロジェクト構造

```
src/raptgfn/
├── algorithms/           # GFlowNetsアルゴリズムの実装
│   ├── mogfn.py         # Multi-Objective GFlowNet メインアルゴリズム
│   ├── conditional_transformer.py  # 条件付きTransformerニューラルネット
│   ├── mogfn_utils.py   # GFlowNet用ユーティリティ関数
│   └── base.py          # アルゴリズム基底クラス
├── tasks/               # タスクと目的関数の実装
│   ├── simple_dna.py    # DNAアプタマー目的関数（メイン実装）
│   └── base.py          # タスク基底クラス
├── utils/               # ユーティリティとトークナイザー
│   └── __init__.py      # DNAアプタマー用トークナイザーとヘルパー関数
└── metrics.py           # 評価メトリクス（ハイパーボリューム等）
```

### 1. 目的関数の実装 (`src/raptgfn/tasks/simple_dna.py`)

**SimpleDNATask**クラスが全ての目的関数を実装：

- **`_compute_length_score()`**: 配列長の最適化
  - 理想的な長さ（ideal_length）への近さを評価
  - スコア = 1.0 / (1.0 + |実際の長さ - 理想長さ| / 理想長さ)

- **`_compute_gc_content_score()`**: GC含有量の最適化
  - 理想的なGC比率（ideal_gc_content、デフォルト50%）への近さを評価
  - スコア = 1.0 / (1.0 + |実際のGC% - 理想GC%| × 4)

- **`_compute_base_balance_score()`**: 塩基バランスの評価
  - A, T, G, Cの均等度を評価（理想的には各25%）
  - 分散ベースの評価でバランスの良さを測定

- **`_compute_dinucleotide_diversity_score()`**: ジヌクレオチド多様性
  - 2塩基の組み合わせの多様性を評価
  - 最大16種類の可能な組み合わせに対する比率

- **`_compute_complexity_score()`**: 配列複雑さ
  - 繰り返しパターンの少なさを評価
  - より複雑な配列ほど高スコア

### 2. GFlowNetsの設計 (`src/raptgfn/algorithms/mogfn.py`)

**MOGFN**クラスがMulti-Objective GFlowNetを実装：

#### 主要コンポーネント：
- **パレート条件付けアプローチ**: `_get_condition_var()`
  - βパラメータ（温度）とプリファレンスベクトルで多目的を制御
  - Thermometer encoding（温度計エンコーディング）で細かい制御

- **サンプリング機能**: `sample()`
  - 条件付きポリシーから配列を生成
  - Causal sampling（因果的サンプリング）で逐次配列生成

- **報酬処理**: `process_reward()`
  - 複数目的スコアを単一報酬に変換
  - プリファレンスに基づく重み付け結合

- **訓練ループ**: `train_step()`
  - GFlowNet損失の計算と最適化
  - パーティション関数Zの学習

#### 特徴的な実装：
- **Thermometer encoding**: 連続値を離散ビンに分割して細かい制御
- **パレートフロンティア評価**: 定期的な多目的最適化性能評価
- **適応的サンプリング**: 訓練進行に応じたサンプリング戦略

### 3. ニューラルネットワーク (`src/raptgfn/algorithms/conditional_transformer.py`)

**CondGFNTransformer**クラスが条件付きTransformerを実装：

#### アーキテクチャ：
- **入力**: DNA配列トークン + 条件ベクトル（β + プリファレンス）
- **エンコーダ**: 標準的なTransformer Encoder
  - 位置エンコーディング（PositionalEncoding）
  - Multi-head attention
  - Causal mask（因果的マスク）で逐次生成を制御

- **条件付け機構**:
  - `cond_embed`: 条件ベクトルをTransformer次元に埋め込み
  - `Z_mod`: パーティション関数Z用の専用層
  - 配列表現と条件表現をconcatenateして最終出力

#### 重要な設計選択：
- **Batch-first形式**: 効率的なバッチ処理
- **動的条件次元**: 目的関数数に応じて条件ベクトルサイズを調整
- **分離された最適化**: モデルパラメータとZパラメータを別々に最適化

### 4. 評価とメトリクス (`src/raptgfn/metrics.py`)

**多目的最適化評価**:
- **ハイパーボリューム**: パレートフロンティアの質を測定
- **R2指標**: プリファレンスベースの評価
- **多様性メトリクス**: 解の分散度を評価
- **PyMOO統合**: 高度な多目的最適化メトリクス

### 5. トークナイザーとユーティリティ (`src/raptgfn/utils/`)

**AptamerTokenizer**:
- DNA塩基（A, T, C, G）専用のトークナイザー
- 特殊トークン（[CLS], [SEP], [PAD]等）のサポート
- 効率的なエンコード/デコード処理

**ヘルパー関数**:
- `str_to_tokens()`: 文字列→トークン変換
- `tokens_to_str()`: トークン→文字列変換
- `validate_dna_sequence()`: DNA配列の妥当性チェック
- `gc_content()`: GC含量計算

### 実行フロー

1. **初期化** (`main.py`)
   - Hydra設定の読み込み
   - トークナイザー、タスク、アルゴリズムの初期化

2. **訓練ループ** (`MOGFN.optimize()`)
   - バッチサンプリング
   - 目的関数評価
   - GFlowNet損失計算
   - ニューラルネット更新

3. **評価** (`MOGFN.evaluation()`)
   - サンプル生成
   - パレートフロンティア計算
   - メトリクス評価

### カスタマイズポイント

- **新しい目的関数**: `SimpleDNATask`に`_compute_*_score()`メソッドを追加
- **ネットワーク構造**: `CondGFNTransformer`の層数、ヘッド数等を調整
- **サンプリング戦略**: `MOGFN`の条件付け方法を変更
- **評価メトリクス**: `metrics.py`に新しい評価関数を追加

## 参考文献

```bibtex
@article{jain2023multi,
  title={Multi-Objective GFlowNets},
  author={Jain, Moksh and Raparthy, Sharath and Hernandez-Garcia, Alex and Rector-Brooks, Jarrid and Bengio, Yoshua and Miret, Santiago and Bengio, Emmanuel},
  journal={arXiv preprint arXiv:2310.12372},
  year={2023}
}
```
