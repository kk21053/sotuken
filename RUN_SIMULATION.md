# シミュレーション実行手順# Webots シミュレーション実行手順



## クイックスタート## クイックスタート



### 1. 環境設定### 1. Webotsを起動



まず、各脚の環境を設定します。```bash

# Webotsがインストールされている場合

```bashwebots

cd /home/kk21053/sotuken/webots```

python set_environment.py NONE BURIED TRAPPED NONE

```### 2. ワールドファイルを開く



上記の例では:Webotsメニューから:

- FL (前左脚): NONE (正常)- `File` → `Open World...`

- FR (前右脚): BURIED (埋まる)- `/home/kk21053/sotuken/webots/worlds/sotuken_world.wbt` を選択

- RL (後左脚): TRAPPED (挟まる)

- RR (後右脚): NONE (正常)### 3. シミュレーションを実行



### 2. Webotsの起動- 画面上部の **再生ボタン（▶）** をクリック

- またはキーボードショートカット: `Ctrl+P`

```bash

webots /home/kk21053/sotuken/webots/worlds/sotuken_world.wbt## 実行されること

```

1. **環境マネージャー起動** (0秒)

### 3. シミュレーション実行   - `scenario.ini` からシナリオを読み込み

   - 砂埋没、罠、ツタなどを配置

Webotsの再生ボタンをクリックしてシミュレーションを開始します。   - コンソールに設定内容を表示



### 4. 結果確認2. **Spot自己診断開始** (1秒後)

   - 4本の脚を順番に診断

#### コンソール出力を見る   - 各脚につき4回の試行（約2秒/脚）

シミュレーション実行中、コンソールに診断の進行状況と最終結果が表示されます。   - 合計約8秒で全脚診断完了



#### スクリプトで結果を確認3. **ドローン観測** (同時並行)

```bash   - Spotの上空でホバリング

cd /home/kk21053/sotuken/webots   - トリガー受信時に高頻度観測

python view_result.py   - RoboPoseでSpotの姿勢推定

```

4. **診断完了** (約10秒後)

## 詳細設定   - コンソールに結果表示

   - ログファイルに保存

### 環境設定の対話モード   - Spotは待機状態へ



```bash## コンソール出力例

cd /home/kk21053/sotuken/webots

python set_environment.py```

```[environment] Environment Manager started

[environment] Scenario: sand_burial

対話形式で各脚の環境を設定できます。[environment] Sand burial active

[environment]                foot          : front_left

### 利用可能な環境タイプ[environment]                radius (m)    : 0.300

[environment]                height (m)    : 0.180

- **NONE**: 正常状態[environment] Environment setup complete

- **BURIED**: 砂に埋まっている

- **TRAPPED**: 障害物に挟まれている[spot] Controller initialized

- **TANGLED**: ツタなどに絡まっている[spot] Starting self-diagnosis sequence

- **MALFUNCTION**: センサー故障[spot] === Diagnosing leg FL (1/4) ===

[spot] Leg FL - Trial 1/4 (dir=+)

### シナリオ設定ファイル[spot] Sent trigger: FL trial 1 dir=+



直接編集する場合:[drone] Controller initialized

```bash[drone] Trigger received: FL trial 1 dir=+

nano /home/kk21053/sotuken/webots/config/scenario.ini[drone] Sending 9 observations for FL trial 1

```

[spot] Trial 1 complete - collected 31 samples

## トラブルシューティング[spot] Leg FL - Trial 2/4 (dir=+)

...

### Webotsが起動しない```



```bash## シナリオ変更

# Webotsのプロセスを確認

ps aux | grep webots### 砂埋没シナリオ



# 必要に応じてプロセスをキル`webots/config/scenario.ini`:

pkill webots```ini

```scenario=sand_burial

buriedFoot=front_left

### 診断が開始されないtopLevel=0.03

```

1. コンソール出力を確認

2. コントローラーが正しく起動しているか確認### 罠シナリオ

3. Webotsを再起動

```ini

### 結果が表示されないscenario=foot_trap

buriedFoot=front_right

```bash```

# ログファイルを確認

ls -la /home/kk21053/sotuken/webots/controllers/diagnostics_pipeline/logs/### ツタ絡まりシナリオ



# 最新のログを表示```ini

tail -f /home/kk21053/sotuken/webots/controllers/diagnostics_pipeline/logs/leg_diagnostics_sessions.jsonlscenario=foot_vine

```buriedFoot=rear_left

```

## 診断結果の見方

### 正常シナリオ（障害なし）

### 出力形式

```ini

```scenario=none

================================================================================```

診断結果

================================================================================## 結果の確認

セッションID: spot_diagnosis_20251013_123456

診断時刻:     2025-10-13T12:34:56### 1. コンソール出力



--------------------------------------------------------------------------------Webotsコンソールに各コントローラーのログが表示されます。

  脚   |   動作状態   |  確信度  |  拘束原因  |   期待値   

--------------------------------------------------------------------------------### 2. ログファイル

  FL   |     動く     |  0.850   |    正常    |    正常    

  FR   |   動かない   |  0.120   |   埋まる   |   埋まる   診断結果は以下に保存されます：

  RL   |   動かない   |  0.180   |   挟まる   |   挟まる   

  RR   |     動く     |  0.920   |    正常    |    正常    ```

--------------------------------------------------------------------------------webots/controllers/diagnostics_pipeline/logs/

├── leg_diagnostics_events.jsonl      # 各試行の詳細

転倒状態: 転倒していない (確率: 0.050)└── leg_diagnostics_sessions.jsonl    # セッション集約結果

```

診断精度: 4/4 (100.0%)

### 3. JSONLファイルの閲覧

================================================================================

``````bash

# 最新のイベントログを表示

### 診断精度の評価tail -f webots/controllers/diagnostics_pipeline/logs/leg_diagnostics_events.jsonl



- **100%**: 全ての脚で正しく診断できた# セッション結果を整形表示

- **75%**: 4脚中3脚が正しいcat webots/controllers/diagnostics_pipeline/logs/leg_diagnostics_sessions.jsonl | python -m json.tool

- **50%**: 4脚中2脚が正しい```

- **25%以下**: 診断精度が低い（設定やアルゴリズムの見直しが必要）

## トラブルシューティング

## 高度な使用例

### Spotが動かない

### 複数シナリオのテスト

**症状**: Spotが静止したまま

```bash

#!/bin/bash**確認点**:

# test_scenarios.sh1. コンソールにエラーメッセージがないか確認

2. Spotコントローラーが起動しているか確認

cd /home/kk21053/sotuken/webots3. シミュレーションが一時停止していないか確認



# シナリオ1: 1脚が埋まっている**対処**:

echo "シナリオ1: FL脚が埋まっている"- シミュレーションをリセット（Ctrl+Shift+T）

python set_environment.py BURIED NONE NONE NONE- ワールドをリロード



# シナリオ2: 対角線の脚が問題あり### ドローンが反応しない

echo "シナリオ2: FLとRRに問題"

python set_environment.py TRAPPED NONE NONE TANGLED**症状**: ドローンがトリガーを受信しない



# シナリオ3: 全脚正常**確認点**:

echo "シナリオ3: 全脚正常"1. Emitter/Receiverが正しく設定されているか

python set_environment.py NONE NONE NONE NONE2. チャンネル番号が一致しているか

```

**対処**:

### ログの分析- ワールドファイルを確認

- extensionSlotにEmitter/Receiverが追加されているか確認

```bash

# 全セッションのログを表示### コンソールにエラーが表示される

cat /home/kk21053/sotuken/webots/controllers/diagnostics_pipeline/logs/leg_diagnostics_sessions.jsonl | python -m json.tool

**よくあるエラー**:

# 特定の脚の診断履歴

grep "FL" /home/kk21053/sotuken/webots/controllers/diagnostics_pipeline/logs/leg_diagnostics_events.jsonl```

```[spot] Warning: Motor front left shoulder abduction motor not found

```

## 注意事項→ モーター名が間違っている可能性。Spot PROTOファイルを確認。



1. 環境設定を変更した後は、Webotsを再起動してください```

2. シミュレーション中はコントローラーのログを確認してください[drone] Warning: receiver not found

3. 診断には約30-60秒かかります（4脚×4試行）```

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
