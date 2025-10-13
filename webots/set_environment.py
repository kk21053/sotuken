#!/usr/bin/env python3
"""
簡易環境設定スクリプト

使い方:
    python set_environment.py NONE BURIED TRAPPED NONE
    
    または対話モード:
    python set_environment.py

引数の順序: FL FR RL RR
設定可能な環境: NONE, BURIED, TRAPPED, TANGLED, MALFUNCTION
"""

import sys
import configparser
from pathlib import Path

# 設定ファイルのパス
CONFIG_PATH = Path(__file__).parent / "config" / "scenario.ini"

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


def update_scenario_config(environments):
    """scenario.iniを更新"""
    # 設定ファイルを読み込み
    config = configparser.ConfigParser()
    if CONFIG_PATH.exists():
        config.read(CONFIG_PATH)
    
    if 'DEFAULT' not in config:
        config['DEFAULT'] = {}
    
    # 各脚の環境を設定
    for leg_id, env in zip(LEG_IDS, environments):
        leg_name = LEG_NAMES[leg_id]
        config['DEFAULT'][f'{leg_id}_environment'] = env
        
        # シナリオタイプの設定（後方互換性のため）
        if env == "BURIED":
            config['DEFAULT']['scenario'] = 'sand_burial'
            config['DEFAULT']['buriedFoot'] = leg_name
        elif env == "TRAPPED":
            config['DEFAULT']['scenario'] = 'foot_trap'
            config['DEFAULT']['trappedFoot'] = leg_name
        elif env == "TANGLED":
            config['DEFAULT']['scenario'] = 'foot_vine'
            config['DEFAULT']['tangledFoot'] = leg_name
    
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
    update_scenario_config(environments)
    print("\n設定が完了しました。Webotsを再起動して変更を反映してください。\n")


if __name__ == "__main__":
    main()
