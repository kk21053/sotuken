#!/usr/bin/env python3
"""
完全環境設定スクリプト

使い方:
    python set_environment.py NONE BURIED TRAPPED NONE
    
    または対話モード:
    python set_environment.py

引数の順序: FL FR RL RR
設定可能な環境: NONE, BURIED, TRAPPED, TANGLED, MALFUNCTION

このスクリプトは以下を変更します:
1. config/scenario.ini - 環境設定
2. worlds/sotuken_world.wbt - ワールドファイルのオブジェクト位置
"""

import sys
import configparser
import re
import shutil
from pathlib import Path
from datetime import datetime

# 設定ファイルのパス
CONFIG_PATH = Path(__file__).parent / "config" / "scenario.ini"
WORLD_FILE = Path(__file__).parent / "worlds" / "sotuken_world.wbt"

# 有効な環境タイプ
VALID_ENVIRONMENTS = ["NONE", "BURIED", "TRAPPED", "TANGLED", "MALFUNCTION"]

# 脚のID
LEG_IDS = ["FL", "FR", "RL", "RR"]

# 脚名の変換マップ
LEG_NAMES = {
    "FL": "front_left",
    "FR": "front_right", 
    "RL": "rear_left",
    "RR": "rear_right"
}


def validate_environment(env):
    """環境タイプの検証"""
    env = env.upper()
    if env not in VALID_ENVIRONMENTS:
        print(f"エラー: '{env}' は無効な環境です。")
        print(f"有効な環境: {', '.join(VALID_ENVIRONMENTS)}")
        return None
    return env


def update_world_file(environments):
    """ワールドファイルのオブジェクト位置を更新
    
    各環境オブジェクトの初期位置を設定:
    - 使用する環境: 表示位置に配置
    - 使用しない環境: 非表示位置 (0, 0, -10) に配置
    """
    if not WORLD_FILE.exists():
        print(f"警告: ワールドファイルが見つかりません: {WORLD_FILE}")
        return False
    
    # バックアップを作成
    backup_file = WORLD_FILE.parent / f"{WORLD_FILE.stem}_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wbt"
    shutil.copy2(WORLD_FILE, backup_file)
    print(f"バックアップ作成: {backup_file.name}")
    
    # ワールドファイルを読み込み
    content = WORLD_FILE.read_text()
    
    # 環境タイプをチェック
    has_buried = "BURIED" in environments
    has_trapped = "TRAPPED" in environments
    has_tangled = "TANGLED" in environments
    
    # BURIED_BOX の位置を更新
    # 親は常に (0, 0, 0)、子要素を移動する
    content = re.sub(
        r'(DEF BURIED_BOX Group \{\s*translation\s+)[-\d.]+\s+[-\d.]+\s+[-\d.]+',
        r'\g<1>0.0 0.0 0.0',
        content,
        count=1
    )
    
    # 子要素（BURIED_BOTTOM, TOP, LEFT, RIGHT, FRONT, BACK）の位置を更新
    buried_children = [
        ('BURIED_BOTTOM', 0.45, 0.17, 0.0),
        ('BURIED_TOP', 0.45, 0.17, 0.12),
        ('BURIED_LEFT', 0.36, 0.17, 0.08),
        ('BURIED_RIGHT', 0.54, 0.17, 0.08),
        ('BURIED_FRONT', 0.45, 0.26, 0.08),
        ('BURIED_BACK', 0.45, 0.08, 0.08),
    ]
    
    if has_buried:
        # 表示位置：FL脚の位置に配置
        for child_name, x, y, z in buried_children:
            pattern = rf'(DEF {child_name} Solid \{{\s*translation\s+)[-\d.]+\s+[-\d.]+\s+[-\d.]+'
            replacement = rf'\g<1>{x} {y} {z}'
            content = re.sub(pattern, replacement, content, count=1)
        print("  BURIED_BOX → 表示（子要素を表示位置に配置）")
    else:
        # 非表示位置：全ての子要素を地下100mに
        for child_name, _, _, _ in buried_children:
            pattern = rf'(DEF {child_name} Solid \{{\s*translation\s+)[-\d.]+\s+[-\d.]+\s+[-\d.]+'
            replacement = r'\g<1>0.0 0.0 -100.0'
            content = re.sub(pattern, replacement, content, count=1)
        print("  BURIED_BOX → 非表示（子要素を地下100mに配置）")
    
    # FOOT_TRAP の位置を更新
    if has_trapped:
        # 表示位置に設定
        content = re.sub(
            r'(DEF FOOT_TRAP Solid \{\s*translation\s+)[-\d.]+\s+[-\d.]+\s+[-\d.]+',
            r'\g<1>0.48 0.17 0.0',
            content,
            count=1
        )
        print("  FOOT_TRAP → 表示")
    else:
        # 非表示位置に設定（-100で完全に非表示）
        content = re.sub(
            r'(DEF FOOT_TRAP Solid \{\s*translation\s+)[-\d.]+\s+[-\d.]+\s+[-\d.]+',
            r'\g<1>0.0 0.0 -100.0',
            content,
            count=1
        )
        print("  FOOT_TRAP → 非表示")
    
    # FOOT_VINE の位置を更新
    if has_tangled:
        # 表示位置に設定
        content = re.sub(
            r'(DEF FOOT_VINE Solid \{\s*translation\s+)[-\d.]+\s+[-\d.]+\s+[-\d.]+',
            r'\g<1>0.48 0.17 0.05',
            content,
            count=1
        )
        print("  FOOT_VINE → 表示")
    else:
        # 非表示位置に設定（-100で完全に非表示）
        content = re.sub(
            r'(DEF FOOT_VINE Solid \{\s*translation\s+)[-\d.]+\s+[-\d.]+\s+[-\d.]+',
            r'\g<1>0.0 0.0 -100.0',
            content,
            count=1
        )
        print("  FOOT_VINE → 非表示")
    
    # ワールドファイルに書き込み
    WORLD_FILE.write_text(content)
    print(f"ワールドファイル更新: {WORLD_FILE}")
    
    return True


def update_scenario_config(environments):
    """scenario.iniを更新"""
    # 設定ファイルを読み込み（物理パラメータのみ保持）
    old_config = configparser.ConfigParser()
    if CONFIG_PATH.exists():
        old_config.read(CONFIG_PATH)
    
    # 新しい設定を作成
    config = configparser.ConfigParser()
    config['DEFAULT'] = {}
    
    # 物理パラメータを引き継ぐ（DEFAULTセクションのみ）
    if 'DEFAULT' in old_config:
        for key in ['toplevel', 'friction', 'bounce', 'material',
                    'sand.radius', 'sand.height', 'sand.color',
                    'trap.offsetx', 'trap.offsety', 'trap.offsetz', 'trap.friction', 'trap.bounce', 'trap.material',
                    'vine.offsetx', 'vine.offsety', 'vine.offsetz', 'vine.rotation', 'vine.friction', 'vine.bounce', 'vine.material']:
            if key in old_config['DEFAULT']:
                config['DEFAULT'][key] = old_config['DEFAULT'][key]
    
    # デフォルト値を設定（存在しない場合）
    config['DEFAULT'].setdefault('scenario', 'none')
    config['DEFAULT'].setdefault('toplevel', '0.10')
    config['DEFAULT'].setdefault('friction', '1000.0')
    config['DEFAULT'].setdefault('bounce', '0.0')
    config['DEFAULT'].setdefault('material', 'sand')
    
    # 各脚の環境を設定
    active_scenario = 'none'
    active_foot = None
    
    for leg_id, env in zip(LEG_IDS, environments):
        leg_name = LEG_NAMES[leg_id]
        config['DEFAULT'][f'{leg_id}_environment'] = env
        
        # 最初に見つかった環境をシナリオとして設定（後方互換性）
        if active_scenario == 'none':
            if env == "BURIED":
                active_scenario = 'sand_burial'
                active_foot = leg_name
                config['DEFAULT']['buriedFoot'] = leg_name
            elif env == "TRAPPED":
                active_scenario = 'foot_trap'
                active_foot = leg_name
                config['DEFAULT']['trappedFoot'] = leg_name
            elif env == "TANGLED":
                active_scenario = 'foot_vine'
                active_foot = leg_name
                config['DEFAULT']['tangledFoot'] = leg_name
    
    config['DEFAULT']['scenario'] = active_scenario
    
    # 設定を保存
    with CONFIG_PATH.open('w') as f:
        config.write(f)
    
    print(f"\n環境設定を更新しました: {CONFIG_PATH}")
    print("=" * 60)
    for leg_id, env in zip(LEG_IDS, environments):
        print(f"  {leg_id} ({LEG_NAMES[leg_id]:12s}): {env}")
    print("=" * 60)


def interactive_mode():
    """対話モード"""
    print("=" * 60)
    print("環境設定スクリプト - 対話モード")
    print("=" * 60)
    print(f"有効な環境: {', '.join(VALID_ENVIRONMENTS)}")
    print()
    
    environments = []
    for leg_id in LEG_IDS:
        while True:
            env = input(f"{leg_id} ({LEG_NAMES[leg_id]}) の環境を入力 (デフォルト: NONE): ").strip()
            if not env:
                env = "NONE"
            env = validate_environment(env)
            if env is not None:
                environments.append(env)
                break
    
    return environments


def main():
    """メイン処理"""
    print()
    
    # コマンドライン引数をチェック
    if len(sys.argv) == 1:
        # 対話モード
        environments = interactive_mode()
    elif len(sys.argv) == 5:
        # コマンドライン引数から取得
        environments = []
        for arg in sys.argv[1:]:
            env = validate_environment(arg)
            if env is None:
                sys.exit(1)
            environments.append(env)
    else:
        print("使い方:")
        print("  対話モード:         python set_environment.py")
        print("  コマンドライン:     python set_environment.py FL FR RL RR")
        print()
        print("例:")
        print("  python set_environment.py NONE BURIED TRAPPED NONE")
        print()
        print(f"有効な環境: {', '.join(VALID_ENVIRONMENTS)}")
        print(f"脚の順序: {' '.join(LEG_IDS)}")
        sys.exit(1)
    
    # 設定を更新
    print("\n" + "=" * 70)
    print("環境設定の適用")
    print("=" * 70)
    
    # 1. scenario.ini を更新
    print("\n[1/2] scenario.ini を更新中...")
    update_scenario_config(environments)
    
    # 2. ワールドファイルを更新
    print("\n[2/2] ワールドファイルを更新中...")
    if update_world_file(environments):
        print("\n" + "=" * 70)
        print("✅ 環境設定が完了しました")
        print("=" * 70)
        print("\n⚠️  Webotsが起動中の場合:")
        print("   1. Webotsを完全に終了してください (Ctrl+C または pkill)")
        print("   2. 再度Webotsを起動してください")
        print("\n   これにより、新しい環境設定が確実に反映されます。\n")
    else:
        print("\n⚠️  ワールドファイルの更新に失敗しました")
        print("   scenario.ini のみが更新されています\n")


if __name__ == "__main__":
    main()
