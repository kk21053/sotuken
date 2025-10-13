# 4足歩行ロボット故障診断システム# sotuken - Spot脚故障診断システム



## 概要四足ロボットSpotの脚故障診断を、ドローン観測（RoboPose）とLLMを組み合わせて実現するシステムです。



このプロジェクトは、4足歩行ロボット(Spot)の故障原因をドローンとの協調診断を用いて特定する研究のシミュレーション実装です。## 概要



## 実施環境このシステムは以下の3つのコンポーネントで構成されています：



- **シミュレーター**: Webots1. **Spot自己診断** - 各脚について4回の試行（+2回、-2回）を実行し、内部センサーで可動度を測定

- **ロボット**: Boston Dynamics Spot2. **ドローン観測** - RoboPoseを用いてSpotの姿勢・関節角を推定し、外部から可動度を評価

- **ドローン**: DJI Mavic 2 PRO3. **LLM判定** - 自己診断とドローン観測の結果を統合し、故障原因を推定



## システム構成## システム構成



### 環境設定### ディレクトリ構造



各脚(FL, FR, RL, RR)に対して以下の環境を個別に設定できます:```

sotuken/

- **NONE**: 正常├── webots/

- **BURIED**: 埋まる（砂に埋まっている）│   ├── worlds/

- **TRAPPED**: 挟まる（障害物に固定されている）│   │   └── sotuken_world.wbt          # Webotsシミュレーション環境

- **TANGLED**: 絡まる（ツタなどに絡まっている）│   ├── controllers/

- **MALFUNCTION**: 故障（センサー故障など）│   │   ├── spot_self_diagnosis/       # Spot側コントローラー

│   │   ├── drone_circular_controller/ # ドローン側コントローラー

### 診断フロー│   │   ├── environment_manager/       # 環境設定マネージャー

│   │   └── diagnostics_pipeline/      # 診断パイプラインライブラリ

1. **Spotの自己診断** (4回/脚)│   ├── config/

   - 各脚を動かして、動くか動かないかを判断│   │   └── scenario.ini               # シナリオ設定

   - 診断基準: `追従性スコア*0.4 + 速度スコア*0.25 + トルクスコア*0.25 + 安全性スコア*0.1`│   └── protos/                        # カスタムPROTOファイル

   - 各脚を動かす際、ドローンに通知└── README.md

```

2. **ドローンの観測**

   - RoboPoseを用いてロボットの姿勢を監査### 主要コンポーネント

   - 脚の実際の動きを診断

   - 胴体の動きも考慮#### 1. diagnostics_pipeline モジュール



3. **確率への変換**診断システムの中核となるライブラリ：

   - Spot: 診断結果を確率に変換し、シグモイド関数で差を明確化 → `spot_can`

   - ドローン: 診断結果を確率に変換し、シグモイド関数で差を明確化 → `drone_can`- `config.py` - 設定定数（試行回数、角度、しきい値など）

   - ドローン: 各脚の拘束状況(NONE, BURIED, TRAPPED, TANGLED, MALFUNCTION)を確率分布として推論- `models.py` - データモデル（脚状態、試行結果、セッション状態）

- `self_diagnosis.py` - Spot自己診断データの集約

4. **Spotからドローンへの通信**- `drone_observer.py` - ドローン観測データの集約（RoboPoseベース）

   - `spot_can`をドローンへ送信- `llm_client.py` - LLM推論インタフェース

- `fusion.py` - 確率融合ロジック

5. **ルールベースLLMによる統合**- `logger.py` - JSONL形式でのロギング

   - ドローンが`spot_can`、`drone_can`、拘束状況の確率分布を分析- `utils.py` - ユーティリティ関数

   - 以下のルールに基づいて最終判定:

     - ①両方が0.7以上 → 動く、拘束原因=正常#### 2. Spotコントローラー (`spot_self_diagnosis.py`)

     - ②両方が0.3以下 → 動かない、拘束原因=確率分布の最大値

     - ③片方が0.7以上、もう片方が0.3以下 → 動かない、拘束原因=故障- 4本の脚（FL, FR, RL, RR）を順番に診断

     - ④片方が中間値 → 一部動く、拘束原因=確率分布の最大値- 各脚につき4回の試行（+7°×2回、-7°×2回）

- 各試行前にドローンへトリガー通知を送信

6. **転倒判定**- センサーデータ（関節角、速度、トルク）を収集

   - ドローンがSpotの姿勢を確認

   - 転倒しているかどうかを確率で判定#### 3. ドローンコントローラー (`drone_circular_controller.py`)



7. **結果の表示**- Spotの上空でホバリング

   - 各脚の動作状態（動く/動かない/一部動く）- トリガー受信時に高頻度観測モード（18Hz）に切り替え

   - 各脚の拘束原因- RoboPoseシミュレータでSpotの姿勢・関節角を推定

   - 転倒状態（転倒している/していない）- 観測データを収集・送信



## 使い方#### 4. 環境マネージャー (`environment_manager.py`)



### 1. 環境設定- シナリオ設定に応じて環境を構築

- 砂埋没、罠、ツタ絡まりなどの障害物を配置

#### コマンドライン方式

```bash## 実行方法

cd webots

python set_environment.py NONE BURIED TRAPPED NONE### 1. 前提条件

```

- Webots R2023b以降

引数の順序: FL FR RL RR- Python 3.8以降

- （オプション）transformers, torch（LLM推論を使用する場合）

#### 対話方式

```bash### 2. シナリオ設定

cd webots

python set_environment.py`webots/config/scenario.ini` で診断シナリオを設定：

```

```ini

### 2. シミュレーション実行# シナリオ選択: sand_burial / foot_trap / foot_vine / none

scenario=sand_burial

Webotsで`webots/worlds/sotuken_world.wbt`を開いて実行します。

# 対象の脚: front_left / front_right / rear_left / rear_right

### 3. 結果の確認buriedFoot=front_left



#### コンソール出力# 砂埋没パラメータ

シミュレーション実行中、コンソールに診断結果が表示されます。topLevel=0.03

friction=1.8

#### スクリプトでの確認sand.radius=0.3

```bashsand.height=0.18

cd webots```

python view_result.py

```### 3. Webotsでシミュレーション実行



最新の診断結果が読みやすい形式で表示されます。1. Webotsを起動

2. `webots/worlds/sotuken_world.wbt` を開く

## ディレクトリ構造3. シミュレーションを開始（再生ボタン）



```### 4. 実行フロー

webots/

├── config/1. **環境マネージャー** がシナリオ設定を読み込み、障害物を配置

│   └── scenario.ini          # シナリオ設定ファイル2. **Spot** が自己診断を開始

├── controllers/   - 各脚ごとに4回の試行を実行

│   ├── diagnostics_pipeline/ # 診断パイプライン（コアロジック）   - 試行前にドローンへトリガー送信

│   │   ├── config.py        # 設定定数3. **ドローン** がトリガーを受信し高頻度観測

│   │   ├── self_diagnosis.py # Spot自己診断   - RoboPoseでSpotを観測

│   │   ├── drone_observer.py # ドローン観測   - 観測データを収集

│   │   ├── llm_client.py    # ルールベースLLM4. 各試行完了後、データが集約される

│   │   ├── fusion.py        # 確率統合5. 全脚の診断完了後、結果がログ出力される

│   │   └── ...

│   ├── spot_self_diagnosis/  # Spotコントローラー## 診断結果の見方

│   ├── drone_circular_controller/ # ドローンコントローラー

│   └── environment_manager/  # 環境管理コントローラー### コンソール出力

├── worlds/

│   └── sotuken_world.wbt     # Webotsワールドファイル```

├── set_environment.py        # 環境設定スクリプト[spot] === Diagnosing leg FL (1/4) ===

└── view_result.py           # 結果表示スクリプト[spot] Leg FL - Trial 1/4 (dir=+)

```[drone] Trigger received: FL trial 1 dir=+

[drone] Sending 9 observations for FL trial 1

## 技術仕様...

```

### 診断指標

### ログファイル

**Spot自己診断:**

- 追従性スコア: 目標角度への追従精度診断結果は `webots/controllers/diagnostics_pipeline/logs/` に保存されます：

- 速度スコア: 関節のピーク速度

- トルクスコア: モーター負荷- `leg_diagnostics_events.jsonl` - 各試行の詳細データ

- 安全性スコア: 安全レベル- `leg_diagnostics_sessions.jsonl` - セッション単位の集約結果



**ドローン観測:**## 仕様詳細

- 関節角度変化量

- 末端位置の変位### 診断パラメータ

- 胴体姿勢の変化

- 経路の直進性- **試行回数**: 4回/脚（config.TRIAL_COUNT）

- **試行パターン**: ["+", "+", "-", "-"]（config.TRIAL_PATTERN）

### シグモイド変換- **角度変化**: ±7度（config.TRIAL_ANGLE_DEG）

- **試行時間**: 0.5秒（config.TRIAL_DURATION_S）

診断スコアを確率に変換する際、シグモイド関数を使用して差を明確化:- **可動判定しきい値**: 0.70（config.SELF_CAN_THRESHOLD）

- 閾値: 0.5

- 急峻度: 15.0### 故障原因クラス

- 高信頼域: 0.7以上

- 低信頼域: 0.3以下- **NONE**: 正常動作

- **BURIED**: 砂などに埋まっている

## ライセンス- **TRAPPED**: 障害物に挟まれている

- **ENTANGLED**: ツタなどに絡まっている

卒業研究用プロジェクト

### 確率融合

最終的な故障原因は、ドローン観測（40%）とLLM判定（60%）の重み付き平均で決定：

```
p_final(c) = 0.4 × p_drone(c) + 0.6 × p_llm(c)
```

## トラブルシューティング

### Spotが動かない

- Spotコントローラーのコンソール出力を確認
- モーター名が正しく取得できているか確認

### ドローンが観測しない

- Emitter/Receiverの設定を確認
- チャンネル番号が一致しているか確認（Spot: channel 1, Drone: channel 1受信）

### LLMエラー

- transformersライブラリがインストールされているか確認
- モデルのダウンロードに時間がかかる場合があります
- スタブモードで動作確認（一様分布を返す）

## 開発者向け情報

### カスタマイズポイント

1. **診断パラメータ調整**: `diagnostics_pipeline/config.py`
2. **RoboPose実装変更**: `drone_circular_controller.py` の `RoboPoseSimulator` クラス
3. **LLMモデル変更**: `diagnostics_pipeline/llm_client.py` の `LLM_PRIMARY` 設定
4. **シナリオ追加**: `config/scenario.ini` と `environment_manager.py`

### テスト

現在はWebotsシミュレーションでの統合テストのみ。
単体テストの追加を推奨。

## ライセンス

（ライセンス情報を記載）

## 参考文献

- Webots: https://cyberbotics.com/
- Boston Dynamics Spot: https://www.bostondynamics.com/products/spot
- RoboPose論文: （該当論文を記載）

