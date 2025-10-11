# sotuken - Spot脚故障診断システム

四足ロボットSpotの脚故障診断を、ドローン観測（RoboPose）とLLMを組み合わせて実現するシステムです。

## 概要

このシステムは以下の3つのコンポーネントで構成されています：

1. **Spot自己診断** - 各脚について4回の試行（+2回、-2回）を実行し、内部センサーで可動度を測定
2. **ドローン観測** - RoboPoseを用いてSpotの姿勢・関節角を推定し、外部から可動度を評価
3. **LLM判定** - 自己診断とドローン観測の結果を統合し、故障原因を推定

## システム構成

### ディレクトリ構造

```
sotuken/
├── webots/
│   ├── worlds/
│   │   └── sotuken_world.wbt          # Webotsシミュレーション環境
│   ├── controllers/
│   │   ├── spot_self_diagnosis/       # Spot側コントローラー
│   │   ├── drone_circular_controller/ # ドローン側コントローラー
│   │   ├── environment_manager/       # 環境設定マネージャー
│   │   └── diagnostics_pipeline/      # 診断パイプラインライブラリ
│   ├── config/
│   │   └── scenario.ini               # シナリオ設定
│   └── protos/                        # カスタムPROTOファイル
└── README.md
```

### 主要コンポーネント

#### 1. diagnostics_pipeline モジュール

診断システムの中核となるライブラリ：

- `config.py` - 設定定数（試行回数、角度、しきい値など）
- `models.py` - データモデル（脚状態、試行結果、セッション状態）
- `self_diagnosis.py` - Spot自己診断データの集約
- `drone_observer.py` - ドローン観測データの集約（RoboPoseベース）
- `llm_client.py` - LLM推論インタフェース
- `fusion.py` - 確率融合ロジック
- `logger.py` - JSONL形式でのロギング
- `utils.py` - ユーティリティ関数

#### 2. Spotコントローラー (`spot_self_diagnosis.py`)

- 4本の脚（FL, FR, RL, RR）を順番に診断
- 各脚につき4回の試行（+7°×2回、-7°×2回）
- 各試行前にドローンへトリガー通知を送信
- センサーデータ（関節角、速度、トルク）を収集

#### 3. ドローンコントローラー (`drone_circular_controller.py`)

- Spotの上空でホバリング
- トリガー受信時に高頻度観測モード（18Hz）に切り替え
- RoboPoseシミュレータでSpotの姿勢・関節角を推定
- 観測データを収集・送信

#### 4. 環境マネージャー (`environment_manager.py`)

- シナリオ設定に応じて環境を構築
- 砂埋没、罠、ツタ絡まりなどの障害物を配置

## 実行方法

### 1. 前提条件

- Webots R2023b以降
- Python 3.8以降
- （オプション）transformers, torch（LLM推論を使用する場合）

### 2. シナリオ設定

`webots/config/scenario.ini` で診断シナリオを設定：

```ini
# シナリオ選択: sand_burial / foot_trap / foot_vine / none
scenario=sand_burial

# 対象の脚: front_left / front_right / rear_left / rear_right
buriedFoot=front_left

# 砂埋没パラメータ
topLevel=0.03
friction=1.8
sand.radius=0.3
sand.height=0.18
```

### 3. Webotsでシミュレーション実行

1. Webotsを起動
2. `webots/worlds/sotuken_world.wbt` を開く
3. シミュレーションを開始（再生ボタン）

### 4. 実行フロー

1. **環境マネージャー** がシナリオ設定を読み込み、障害物を配置
2. **Spot** が自己診断を開始
   - 各脚ごとに4回の試行を実行
   - 試行前にドローンへトリガー送信
3. **ドローン** がトリガーを受信し高頻度観測
   - RoboPoseでSpotを観測
   - 観測データを収集
4. 各試行完了後、データが集約される
5. 全脚の診断完了後、結果がログ出力される

## 診断結果の見方

### コンソール出力

```
[spot] === Diagnosing leg FL (1/4) ===
[spot] Leg FL - Trial 1/4 (dir=+)
[drone] Trigger received: FL trial 1 dir=+
[drone] Sending 9 observations for FL trial 1
...
```

### ログファイル

診断結果は `webots/controllers/diagnostics_pipeline/logs/` に保存されます：

- `leg_diagnostics_events.jsonl` - 各試行の詳細データ
- `leg_diagnostics_sessions.jsonl` - セッション単位の集約結果

## 仕様詳細

### 診断パラメータ

- **試行回数**: 4回/脚（config.TRIAL_COUNT）
- **試行パターン**: ["+", "+", "-", "-"]（config.TRIAL_PATTERN）
- **角度変化**: ±7度（config.TRIAL_ANGLE_DEG）
- **試行時間**: 0.5秒（config.TRIAL_DURATION_S）
- **可動判定しきい値**: 0.70（config.SELF_CAN_THRESHOLD）

### 故障原因クラス

- **NONE**: 正常動作
- **BURIED**: 砂などに埋まっている
- **TRAPPED**: 障害物に挟まれている
- **ENTANGLED**: ツタなどに絡まっている

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

