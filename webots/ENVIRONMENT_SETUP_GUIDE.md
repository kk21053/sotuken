# 環境設定システム - 完全ガイド

## 概要

`set_environment.py` は、Webots シミュレーション環境を完全に制御するツールです。
このスクリプトは以下を自動的に更新します：

1. **config/scenario.ini** - 環境パラメータと脚ごとの設定
2. **worlds/sotuken_world.wbt** - ワールドファイル内のオブジェクト位置

## 使い方

### コマンドライン モード

```bash
python3 set_environment.py <FL> <FR> <RL> <RR>
```

引数は脚の順序で指定：
- FL: Front Left（前左）
- FR: Front Right（前右）
- RL: Rear Left（後左）
- RR: Rear Right（後右）

有効な環境タイプ：
- **NONE**: 正常環境（障害物なし）
- **BURIED**: 砂に埋まった状態
- **TRAPPED**: 足が挟まった状態
- **TANGLED**: ツタに絡まった状態
- **MALFUNCTION**: 故障（シミュレーションでは未実装）

### 例

```bash
# 全脚正常
python3 set_environment.py NONE NONE NONE NONE

# FL脚のみ砂に埋まった状態
python3 set_environment.py BURIED NONE NONE NONE

# FL脚が挟まった状態
python3 set_environment.py TRAPPED NONE NONE NONE

# FL脚が砂に埋まり、FR脚が挟まった状態（複数環境）
python3 set_environment.py BURIED TRAPPED NONE NONE
```

### 対話モード

引数なしで実行すると対話モードになります：

```bash
python3 set_environment.py
```

各脚の環境を順番に入力できます。

## 実行フロー

### 1. scenario.ini の更新

以下のフィールドが更新されます：

```ini
[DEFAULT]
scenario = foot_trap          # 'none', 'sand_burial', 'foot_trap', 'foot_vine'
fl_environment = TRAPPED      # 各脚の環境
fr_environment = NONE
rl_environment = NONE
rr_environment = NONE
trappedFoot = front_left      # 該当する脚（後方互換性）
```

### 2. ワールドファイルの更新

各環境オブジェクトの `translation` フィールドが更新されます：

- **BURIED_BOX** (砂のボックス)
  - 表示: `translation 0.0 0.0 0.0`
  - 非表示: `translation 0.0 0.0 -10.0`

- **FOOT_TRAP** (足トラップ)
  - 表示: `translation 0.48 0.17 0.0`
  - 非表示: `translation 0.0 0.0 -10.0`

- **FOOT_VINE** (ツタ)
  - 表示: `translation 0.48 0.17 0.05`
  - 非表示: `translation 0.0 0.0 -10.0`

### 3. バックアップ

ワールドファイルは自動的にバックアップされます：

```
worlds/sotuken_world_backup_YYYYMMDD_HHMMSS.wbt
```

## 重要な注意事項

### ⚠️ Webots の再起動が必要

環境設定を変更した後は、**必ず Webots を完全に再起動**してください：

```bash
# Webots を完全終了
pkill -9 webots
pkill -9 webots-bin

# 2秒待機してから再起動
sleep 2
webots worlds/sotuken_world.wbt
```

**理由**: Webotsは実行中にワールドファイルをメモリにキャッシュしているため、
ファイルを更新しても、再起動しない限り変更は反映されません。

### 🔍 設定の確認

現在の環境設定を確認するには：

```bash
python3 test_environment_visibility.py
```

出力例：
```
砂のボックス（BURIED環境）
  DEF名: BURIED_BOX
  位置: (0.000, 0.000, -10.000)
  状態: ❌ 非表示（z < -5m）

足トラップ（TRAPPED環境）
  DEF名: FOOT_TRAP
  位置: (0.480, 0.170, 0.000)
  状態: ✅ 表示
```

## トラブルシューティング

### 問題: 環境が反映されない

**解決策**:
1. Webots を完全に終了: `pkill -9 webots; pkill -9 webots-bin`
2. 2秒待機: `sleep 2`
3. 再起動: `webots worlds/sotuken_world.wbt`

### 問題: 複数の環境が同時に表示される

**原因**: ワールドファイルが古い状態で保存されていた

**解決策**:
```bash
# 環境を再設定
python3 set_environment.py NONE NONE NONE NONE

# Webots再起動
pkill -9 webots; pkill -9 webots-bin; sleep 2
webots worlds/sotuken_world.wbt
```

### 問題: バックアップファイルが多すぎる

**解決策**:
```bash
# 古いバックアップを削除（7日以上前）
find worlds/ -name "sotuken_world_backup_*.wbt" -mtime +7 -delete
```

## 自動テスト

環境設定の動作テストを実行：

```bash
bash quick_env_test.sh
```

このスクリプトは以下をテストします：
1. 全脚 NONE
2. FL脚 BURIED
3. FL脚 TRAPPED
4. 複数環境（FL=BURIED, FR=TRAPPED）

## システム構成

```
webots/
├── set_environment.py              # メイン環境設定スクリプト
├── test_environment_visibility.py  # 環境オブジェクト確認ツール
├── quick_env_test.sh              # 自動テストスクリプト
├── config/
│   └── scenario.ini               # 環境設定ファイル
├── worlds/
│   └── sotuken_world.wbt          # Webotsワールドファイル
└── controllers/
    └── environment_manager/       # 実行時の環境管理
        └── environment_manager.py
```

## 技術詳細

### なぜワールドファイルを直接編集するのか？

**問題**: `environment_manager.py` は起動時にオブジェクトを配置しますが、
Webotsが実行中にワールドファイルに変更を書き込むため、次回起動時に
古い状態がロードされることがあります。

**解決策**: `set_environment.py` がワールドファイルを直接編集することで、
起動前の初期状態から正しい環境が設定されます。

### 非表示の仕組み

オブジェクトを `(0, 0, -10)` に移動することで画面外に配置します。
これにより：
- レンダリング負荷が軽減される
- 物理シミュレーションに影響しない
- カメラビューから完全に消える

## 今後の拡張

- [ ] 複数の脚に異なる環境を同時適用（現在は1種類のみ）
- [ ] 環境オブジェクトの位置カスタマイズ
- [ ] GUI ベースの環境設定ツール
- [ ] 環境プリセットの保存/読み込み

## 参考コマンド集

```bash
# 環境設定 → Webots再起動 の完全フロー
python3 set_environment.py BURIED NONE NONE NONE && \
pkill -9 webots; pkill -9 webots-bin; sleep 2 && \
webots worlds/sotuken_world.wbt

# 設定確認
python3 test_environment_visibility.py

# scenario.ini の内容確認
cat config/scenario.ini | grep -E "(scenario|_environment)"

# バックアップ一覧
ls -lh worlds/sotuken_world_backup_*.wbt
```

---

**作成日**: 2025-10-14  
**最終更新**: 2025-10-14
