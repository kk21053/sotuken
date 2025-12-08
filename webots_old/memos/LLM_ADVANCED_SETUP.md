# Advanced LLM Diagnosis System - Setup Guide

Jetson Orin Nano Super用の高度なLLM診断システムのセットアップガイド

## 概要

このシステムは、ルールベース診断の信頼度が低い場合（≤60%）に、LLMを使用した高度な診断を実行します。

### 主要機能

1. **ルールベース診断** (< 1ms)
   - 既存の4つのルールによる高速診断
   - 信頼度スコアの計算

2. **RAGシステム**
   - SpotマニュアルPDFから関連情報を検索
   - 埋め込みベクトルによる意味検索
   - FAISS インデックスによる高速検索

3. **LLM診断** (~3sec)
   - llama.cpp による軽量推論
   - マニュアル情報 + RoboPose生データを統合
   - 低信頼度ケースのみ実行（選択的起動）

### アーキテクチャ

```
入力（脚状態）
    ↓
ルールベース診断 → 信頼度計算
    ↓
信頼度 > 60%? → YES → 結果を返す
    ↓ NO
RAGでマニュアル検索
    ↓
LLMプロンプト構築
    ↓
LLM推論 (Llama 3.2 3B)
    ↓
最終診断結果
```

## 実装済みファイル

### コアモジュール

1. **`rag_manual.py`** (新規作成)
   - `ManualRAG` クラス: PDFからテキスト抽出、埋め込み生成、検索
   - `get_manual_rag()`: シングルトンアクセス

2. **`llm_advanced.py`** (新規作成)
   - `AdvancedLLMAnalyzer` クラス: LLM診断エンジン
   - プロンプト構築、LLM呼び出し、結果パース
   - `get_llm_analyzer()`: シングルトンアクセス

3. **`llm_client.py`** (更新)
   - `infer_with_confidence()` メソッド追加
   - 確率分布の最大値を信頼度として返す

4. **`pipeline.py`** (更新)
   - LLM診断統合
   - 信頼度チェック → 低信頼度時にLLM起動

5. **`config.py`** (更新)
   ```python
   USE_LLM_ADVANCED = False  # LLM診断の有効化
   LLM_CONFIDENCE_THRESHOLD = 0.6  # 信頼度閾値
   LLM_MODEL_PATH = "models/llama-3.2-3b-instruct-q4_k_m.gguf"
   MANUAL_PDF_PATH = "/home/kk21053/sotuken/Spot_IFU-v2.1.2-ja.pdf"
   MANUAL_EMBEDDINGS_CACHE = "data/manual_embeddings"
   ```

### テスト・ドキュメント

- **`test_llm_advanced.py`** (新規作成): 統合テストスクリプト
- **`requirements_llm.txt`** (新規作成): 依存関係リスト
- **`LLM_ADVANCED_SETUP.md`** (このファイル): セットアップガイド

## セットアップ手順

### 前提条件

- Jetson Orin Nano Super (8GB RAM)
- JetPack 5.x または 6.x
- Python 3.8+
- CUDA 11.4+ (JetPackに含まれる)

### 1. 依存関係のインストール

```bash
cd /home/kk21053/sotuken/webots

# 基本パッケージ
pip install -r requirements_llm.txt

# llama-cpp-python (CUDA有効化)
CMAKE_ARGS="-DGGML_CUDA=ON" pip install llama-cpp-python
```

**注意**: Jetson上でビルドには時間がかかります（20-30分）。

### 2. モデルのダウンロード

```bash
# モデルディレクトリ作成
mkdir -p models
cd models

# Llama 3.2 3B Instruct (Q4_K_M量子化版)
# オプション1: Hugging Faceから直接
wget https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q4_K_M.gguf \
  -O llama-3.2-3b-instruct-q4_k_m.gguf

# オプション2: 別マシンでダウンロード後、scpで転送
# scp llama-3.2-3b-instruct-q4_k_m.gguf jetson@<IP>:/home/kk21053/sotuken/webots/models/
```

**ファイルサイズ**: 約2.5GB

### 3. 埋め込みモデルのダウンロード（初回自動）

初回実行時、`sentence-transformers` が自動でモデルをダウンロードします:
- モデル: `paraphrase-multilingual-MiniLM-L12-v2`
- サイズ: 約500MB
- キャッシュ: `~/.cache/torch/sentence_transformers/`

### 4. 設定の有効化

`controllers/diagnostics_pipeline/config.py` を編集:

```python
# LLM診断を有効化
USE_LLM_ADVANCED = True

# モデルパスを確認（必要に応じて変更）
LLM_MODEL_PATH = "models/llama-3.2-3b-instruct-q4_k_m.gguf"
MANUAL_PDF_PATH = "/home/kk21053/sotuken/Spot_IFU-v2.1.2-ja.pdf"
```

### 5. テスト実行

```bash
cd /home/kk21053/sotuken/webots

# 統合テスト
python test_llm_advanced.py
```

**期待される出力**:

```
TEST 1: RAG System (PDF Manual Search)
✓ RAG initialized with XXX chunks
...
✅ RAG test passed

TEST 2: Confidence Calculation
...
✅ Confidence calculation test passed

TEST 3: LLM Diagnosis
✓ LLM analyzer initialized
...
✅ LLM diagnosis test passed

TEST 4: Pipeline Integration
...
✅ Integration test passed

🎉 All tests passed!
```

## 使用方法

### 基本的な使い方

診断パイプラインは自動的にLLMを使用します（`USE_LLM_ADVANCED=True`の場合）:

```python
from diagnostics_pipeline.pipeline import DiagnosticsPipeline

# パイプライン初期化（LLM自動ロード）
pipeline = DiagnosticsPipeline(session_id="test_001")

# 診断実行（信頼度が低い場合、自動でLLM起動）
# ... 既存のコードと同じ
```

### ログ出力例

```
[pipeline] Initializing advanced LLM analyzer...
[llm_advanced] Loading model: models/llama-3.2-3b-instruct-q4_k_m.gguf
[llm_advanced] Model loaded successfully
[rag] Loading cached embeddings
[rag] Loaded 1234 chunks from cache
[pipeline] Advanced LLM enabled (confidence threshold: 0.6)

...

[pipeline] FL: Low confidence (45.2%), invoking LLM...
[llm_advanced] Running LLM diagnosis for FL...
[llm_advanced] Parsed distribution: {...}
[pipeline] FL: LLM diagnosis complete (new confidence: 87.3%)
```

## パフォーマンス

### Jetson Orin Nano Super 実測値（予測）

| 処理 | 時間 | メモリ使用量 |
|------|------|--------------|
| ルールベース診断 | < 1ms | 最小 |
| RAG検索 | ~50ms | ~500MB (初回のみ) |
| LLM推論 | ~3秒 | ~3GB |
| **合計（LLM使用時）** | **~3秒** | **~3.5GB** |

### メモリ最適化

- 埋め込みキャッシュ: 初回生成後はディスクから読み込み
- シングルトンパターン: モデルは1インスタンスのみ
- Q4量子化: メモリ使用量を1/4に削減

### スループット

- ルールベースのみ: **600+ legs/sec**
- LLM併用（10%がLLM起動）: **~3 legs/sec** (LLM bottleneck)

## トラブルシューティング

### 1. llama-cpp-python のインストール失敗

**症状**: CUDA関連のビルドエラー

**解決策**:
```bash
# CUDAパスを明示
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

CMAKE_ARGS="-DGGML_CUDA=ON" pip install llama-cpp-python --no-cache-dir
```

### 2. メモリ不足エラー

**症状**: `CUDA out of memory`

**解決策**:
- より小さいモデルを使用: Llama 3.2 1B (Q4_K_M)
- GPU層数を削減: `config.py` で `n_gpu_layers = 20`
- バッチサイズを削減（実装済み: batch_size=1）

### 3. PDFから埋め込みを生成できない

**症状**: `[rag] Warning: PDF not found`

**解決策**:
```bash
# PDFパスを確認
ls -lh /home/kk21053/sotuken/Spot_IFU-v2.1.2-ja.pdf

# config.pyのパスを更新
MANUAL_PDF_PATH = "/正しいパス/Spot_IFU-v2.1.2-ja.pdf"
```

### 4. 推論が遅い

**症状**: LLM診断に5秒以上かかる

**解決策**:
- GPU層数を増やす: `n_gpu_layers = 33`（全層）
- CUDA有効化を確認: `llama-cpp-python` のビルドログ
- より小さいコンテキスト: `n_ctx = 1024`

### 5. テストで "LLM not available"

**症状**: `⚠️ LLM not available`

**原因**: モデルファイルが存在しない

**解決策**:
```bash
# モデルの存在確認
ls -lh models/llama-3.2-3b-instruct-q4_k_m.gguf

# モデルを再ダウンロード（セットアップ手順2参照）
```

## 高度な設定

### カスタムモデルの使用

他のGGUFモデルも使用可能:

```python
# config.py
LLM_MODEL_PATH = "models/your-custom-model.gguf"

# pipeline初期化時
from diagnostics_pipeline.llm_advanced import get_llm_analyzer
analyzer = get_llm_analyzer(model_path="models/your-model.gguf")
```

### 信頼度閾値の調整

```python
# config.py
LLM_CONFIDENCE_THRESHOLD = 0.5  # より多くのケースでLLM起動
LLM_CONFIDENCE_THRESHOLD = 0.8  # LLM起動を抑制（高速化）
```

### RAGチューニング

```python
# rag_manual.py の ManualRAG.__init__()
rag = ManualRAG(
    chunk_size=300,       # チャンクサイズ縮小（精度向上）
    chunk_overlap=50,     # オーバーラップ削減（速度向上）
    model_name="paraphrase-multilingual-mpnet-base-v2",  # より高精度なモデル
)
```

## パフォーマンスチューニング

### GPU最適化

```python
# llm_advanced.py
llm = Llama(
    model_path=model_path,
    n_gpu_layers=33,      # 全層をGPUに（Jetson Orin Nano Superの場合）
    n_ctx=2048,           # コンテキストサイズ
    n_batch=512,          # バッチサイズ
    n_threads=4,          # CPUスレッド数
)
```

### メモリ vs 速度トレードオフ

| 設定 | メモリ | 速度 | 精度 |
|------|--------|------|------|
| Q4_K_M, n_ctx=2048, 全層GPU | 3GB | 速い | 高 |
| Q4_K_M, n_ctx=1024, 全層GPU | 2GB | 最速 | 中 |
| Q4_K_M, n_ctx=2048, 20層GPU | 2GB | 中 | 高 |
| Q2_K, n_ctx=1024, 全層GPU | 1.5GB | 速い | 低 |

## 開発者向け情報

### ファイル構造

```
webots/
├── controllers/
│   └── diagnostics_pipeline/
│       ├── llm_advanced.py       # LLM診断エンジン
│       ├── rag_manual.py         # RAGシステム
│       ├── llm_client.py         # ルールベース + 信頼度
│       ├── pipeline.py           # 統合パイプライン
│       └── config.py             # 設定
├── models/                       # LLMモデル（.gguf）
├── data/
│   └── manual_embeddings/        # 埋め込みキャッシュ
│       ├── manual_index.json     # チャンク情報
│       └── manual_embeddings.npy # 埋め込みベクトル
├── test_llm_advanced.py          # テストスクリプト
├── requirements_llm.txt          # 依存関係
└── LLM_ADVANCED_SETUP.md         # このファイル
```

### API
