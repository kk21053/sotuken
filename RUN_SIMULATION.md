# Webots シミュレーション実行手順

## クイックスタート

### 1. Webotsを起動

```bash
# Webotsがインストールされている場合
webots
```

### 2. ワールドファイルを開く

Webotsメニューから:
- `File` → `Open World...`
- `/home/kk21053/sotuken/webots/worlds/sotuken_world.wbt` を選択

### 3. シミュレーションを実行

- 画面上部の **再生ボタン（▶）** をクリック
- またはキーボードショートカット: `Ctrl+P`

## 実行されること

1. **環境マネージャー起動** (0秒)
   - `scenario.ini` からシナリオを読み込み
   - 砂埋没、罠、ツタなどを配置
   - コンソールに設定内容を表示

2. **Spot自己診断開始** (1秒後)
   - 4本の脚を順番に診断
   - 各脚につき4回の試行（約2秒/脚）
   - 合計約8秒で全脚診断完了

3. **ドローン観測** (同時並行)
   - Spotの上空でホバリング
   - トリガー受信時に高頻度観測
   - RoboPoseでSpotの姿勢推定

4. **診断完了** (約10秒後)
   - コンソールに結果表示
   - ログファイルに保存
   - Spotは待機状態へ

## コンソール出力例

```
[environment] Environment Manager started
[environment] Scenario: sand_burial
[environment] Sand burial active
[environment]                foot          : front_left
[environment]                radius (m)    : 0.300
[environment]                height (m)    : 0.180
[environment] Environment setup complete

[spot] Controller initialized
[spot] Starting self-diagnosis sequence
[spot] === Diagnosing leg FL (1/4) ===
[spot] Leg FL - Trial 1/4 (dir=+)
[spot] Sent trigger: FL trial 1 dir=+

[drone] Controller initialized
[drone] Trigger received: FL trial 1 dir=+
[drone] Sending 9 observations for FL trial 1

[spot] Trial 1 complete - collected 31 samples
[spot] Leg FL - Trial 2/4 (dir=+)
...
```

## シナリオ変更

### 砂埋没シナリオ

`webots/config/scenario.ini`:
```ini
scenario=sand_burial
buriedFoot=front_left
topLevel=0.03
```

### 罠シナリオ

```ini
scenario=foot_trap
buriedFoot=front_right
```

### ツタ絡まりシナリオ

```ini
scenario=foot_vine
buriedFoot=rear_left
```

### 正常シナリオ（障害なし）

```ini
scenario=none
```

## 結果の確認

### 1. コンソール出力

Webotsコンソールに各コントローラーのログが表示されます。

### 2. ログファイル

診断結果は以下に保存されます：

```
webots/controllers/diagnostics_pipeline/logs/
├── leg_diagnostics_events.jsonl      # 各試行の詳細
└── leg_diagnostics_sessions.jsonl    # セッション集約結果
```

### 3. JSONLファイルの閲覧

```bash
# 最新のイベントログを表示
tail -f webots/controllers/diagnostics_pipeline/logs/leg_diagnostics_events.jsonl

# セッション結果を整形表示
cat webots/controllers/diagnostics_pipeline/logs/leg_diagnostics_sessions.jsonl | python -m json.tool
```

## トラブルシューティング

### Spotが動かない

**症状**: Spotが静止したまま

**確認点**:
1. コンソールにエラーメッセージがないか確認
2. Spotコントローラーが起動しているか確認
3. シミュレーションが一時停止していないか確認

**対処**:
- シミュレーションをリセット（Ctrl+Shift+T）
- ワールドをリロード

### ドローンが反応しない

**症状**: ドローンがトリガーを受信しない

**確認点**:
1. Emitter/Receiverが正しく設定されているか
2. チャンネル番号が一致しているか

**対処**:
- ワールドファイルを確認
- extensionSlotにEmitter/Receiverが追加されているか確認

### コンソールにエラーが表示される

**よくあるエラー**:

```
[spot] Warning: Motor front left shoulder abduction motor not found
```
→ モーター名が間違っている可能性。Spot PROTOファイルを確認。

```
[drone] Warning: receiver not found
```
→ ワールドファイルでReceiverが追加されていない。extensionSlotを確認。

## 高度な使用方法

### LLM推論を有効化

デフォルトではLLMはスタブモード（一様分布）で動作します。

実際のLLMを使用する場合:

```bash
# 必要なパッケージをインストール
pip install transformers torch

# シミュレーション実行
# 初回はモデルのダウンロードに時間がかかります
```

### カスタムパラメータ調整

`webots/controllers/diagnostics_pipeline/config.py` で以下を変更可能:

- `TRIAL_COUNT`: 試行回数（デフォルト: 4）
- `TRIAL_ANGLE_DEG`: 角度変化（デフォルト: 7度）
- `SELF_CAN_THRESHOLD`: 可動判定しきい値（デフォルト: 0.70）
- `FUSION_WEIGHTS`: 融合重み（デフォルト: drone=0.4, llm=0.6）

### デバッグモード

より詳細なログを出力する場合、各コントローラーにprint文を追加してください。

## 次のステップ

1. 異なるシナリオで実行し、診断精度を確認
2. ログファイルを分析し、判定ロジックを改善
3. LLMモデルを変更して性能比較
4. パラメータチューニングで最適値を探索
