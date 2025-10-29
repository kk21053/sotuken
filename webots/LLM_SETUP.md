# ローカルLLM診断機能

## 概要

診断パイプラインにローカルで動作するLLM（Large Language Model）を統合しました。
Ollama + Llama 3.2を使用して、4足歩行ロボットの故障診断を行います。

## セットアップ済み環境

- **Ollama**: ローカルLLM推論サーバー（~/.local/bin/bin/ollama）
- **Python ollama client**: Python APIクライアント（uv経由でインストール済み）
- **モデル**:
  - llama3.2:1b（1.3GB）- 軽量・高速
  - llama3.2:3b（2.0GB）- より高性能

## 使用方法

### 1. Ollamaサーバーの起動

```bash
# バックグラウンドで起動
~/.local/bin/bin/ollama serve > /tmp/ollama.log 2>&1 &

# ログ確認
tail -f /tmp/ollama.log
```

### 2. 診断モードの選択

`webots/controllers/diagnostics_pipeline/config.py` で設定:

```python
# ルールベース診断（推奨）
USE_LLM = False

# LLM診断（実験的）
USE_LLM = True
LLM_MODEL = "llama3.2:3b"
```

### 3. テスト実行

```bash
cd /home/kk21053/sotuken/webots
source ../.venv/bin/activate
python3 test_llm.py
```

## LLMモデルの比較

| モデル | サイズ | 速度 | 精度 | 推奨用途 |
|--------|--------|------|------|----------|
| llama3.2:1b | 1.3GB | 超高速 | 低 | テスト・開発 |
| llama3.2:3b | 2.0GB | 高速 | 中 | 実験的使用 |
| llama3:8b | 4.7GB | 中速 | 高 | 本番環境（未テスト） |

## 現在の推奨設定

**USE_LLM = False（ルールベース）** を推奨します。

理由:
- ✅ 高速（< 1ms）
- ✅ 正確（仕様通りのルール）
- ✅ 安定した動作
- ✅ 外部依存なし

LLM（llama3.2:1b/3b）の問題点:
- ❌ ルール推論が不安定
- ❌ 数値比較の精度が低い
- ❌ 推論時間が長い（100-500ms）

## LLMの改善案

より良い結果を得るには:

1. **より大きなモデルを使用**
   ```bash
   ~/.local/bin/bin/ollama pull llama3:8b
   # config.pyでLLM_MODEL = "llama3:8b"
   ```

2. **Few-shot学習の追加**
   - プロンプトに診断例を追加

3. **ファインチューニング**
   - 診断データでモデルを微調整

## ファイル構成

```
webots/
├── test_llm.py                          # LLMテストスクリプト
└── controllers/
    └── diagnostics_pipeline/
        ├── config.py                    # LLM設定（USE_LLM, LLM_MODEL）
        ├── llm_client.py                # LLM/ルールベース実装
        └── ...
```

## トラブルシューティング

### Ollamaサーバーが起動しない

```bash
# プロセス確認
ps aux | grep ollama

# 手動起動
~/.local/bin/bin/ollama serve
```

### モデルが見つからない

```bash
# インストール済みモデル確認
~/.local/bin/bin/ollama list

# モデルダウンロード
~/.local/bin/bin/ollama pull llama3.2:3b
```

### Python ollama パッケージのエラー

```bash
cd /home/kk21053/sotuken
source .venv/bin/activate
uv pip install ollama
```

## まとめ

- ローカルLLM環境は**完全にセットアップ済み**
- 現時点では**ルールベース診断を推奨**（高速・正確）
- LLMは**オプション機能**として実装済み
- より大きなモデルで性能改善の可能性あり
