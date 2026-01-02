# webots_new（簡潔版）

`webots/` の診断システムと同等の外部I/F（設定・ログ・結果表示）を、短く読みやすく書き直した版です。

## 1. できること（外から見える機能）

- `config/scenario.ini` を元に、各脚（FL/FR/RL/RR）の環境（NONE/BURIED/TRAPPED/TANGLED/MALFUNCTION）を設定
- Webotsで Spot が各脚 6 試行の自己診断を実行し、Drone が観測と統合診断を行う
- 結果を JSONL に保存し、`view_result.py` で最新セッションを表示

## 2. 実行手順

### 2.1 環境設定（任意）

例（FLだけ BURIED）:

```bash
python3 webots_new/set_environment.py BURIED NONE NONE NONE
```

- 反映先: `webots_new/config/scenario.ini` と `webots_new/worlds/sotuken_world.wbt`

### 2.2 Webots 実行

- Webotsで `webots_new/worlds/sotuken_world.wbt` を開いて実行

補足: ターミナルで `webots` だけを実行すると「最後に開いた world」が開かれることがあります。
確実に `webots_new` の world を開くには次のどちらかを使ってください。

```bash
# 1) world を引数で指定
webots webots_new/worlds/sotuken_world.wbt

# 2) このリポジトリに同梱した起動スクリプト
./webots_new/run_webots.sh
```
- world 内の controller 名は以下（既存と同じ）:
  - Spot: `spot_self_diagnosis`
  - Drone: `drone_circular_controller`

### 2.3 結果表示

```bash
python3 webots_new/view_result.py
```

## 3. 入出力（I/F）

### 3.1 Spot→Drone メッセージ（customData）

既存互換（同じ形式）:

- `TRIGGER|leg_id|trial_index|direction|start_time|duration_ms`
- `JOINT_ANGLES|leg_id|trial_index|a0|a1|a2`
- `SELF_DIAG|leg_id|trial_index|theta_samples|theta_avg|theta_final|tau_avg|tau_max|tau_nominal|safety|self_can_raw`

### 3.2 ログ出力

Drone コントローラが以下に出力します（`view_result.py` が読む場所）:

- `webots_new/controllers/drone_circular_controller/logs/leg_diagnostics_events.jsonl`
- `webots_new/controllers/drone_circular_controller/logs/leg_diagnostics_sessions.jsonl`

## 4. 実装の対応関係（最小）

- Spot: `webots_new/controllers/spot_self_diagnosis/spot_self_diagnosis.py`
- Drone: `webots_new/controllers/drone_circular_controller/drone_circular_controller.py`
- 統合診断: `webots_new/controllers/diagnostics_pipeline/`
