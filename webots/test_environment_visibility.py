#!/usr/bin/env python3
"""
環境オブジェクトの可視性テストスクリプト
Webotsを起動せずにワールドファイルを直接編集して、
BURIED_BOXとFOOT_TRAPの位置を確認します。
"""

import re
from pathlib import Path

WORLD_FILE = Path(__file__).parent / "worlds" / "sotuken_world.wbt"

def check_object_position(world_content, object_name):
    """オブジェクトの位置を検索"""
    # DEF OBJECT_NAME から次のDEFまでの範囲を取得
    pattern = rf'DEF {object_name}.*?(?=\nDEF|\Z)'
    match = re.search(pattern, world_content, re.DOTALL)
    
    if not match:
        return None, "Object not found"
    
    object_section = match.group(0)
    
    # translation フィールドを検索
    trans_pattern = r'translation\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)'
    trans_matches = list(re.finditer(trans_pattern, object_section))
    
    if not trans_matches:
        return None, "No translation field found"
    
    # 最初のtranslation（親ノード）を取得
    first_match = trans_matches[0]
    x, y, z = map(float, first_match.groups())
    
    return (x, y, z), "OK"

def main():
    print("=" * 70)
    print("環境オブジェクト可視性チェック")
    print("=" * 70)
    
    if not WORLD_FILE.exists():
        print(f"エラー: {WORLD_FILE} が見つかりません")
        return
    
    content = WORLD_FILE.read_text()
    
    objects = {
        "BURIED_BOX": "砂のボックス（BURIED環境）",
        "FOOT_TRAP": "足トラップ（TRAPPED環境）",
        "FOOT_VINE": "ツタ（TANGLED環境）"
    }
    
    print(f"\nワールドファイル: {WORLD_FILE}")
    print()
    
    for obj_def, description in objects.items():
        pos, status = check_object_position(content, obj_def)
        
        print(f"{description}")
        print(f"  DEF名: {obj_def}")
        
        if pos:
            x, y, z = pos
            if z < -50.0:
                visibility = "❌ 非表示（z < -50m）"
            else:
                visibility = "✅ 表示"
            
            print(f"  位置: ({x:.3f}, {y:.3f}, {z:.3f})")
            print(f"  状態: {visibility}")
        else:
            print(f"  エラー: {status}")
        
        print()
    
    # scenario.ini の設定を確認
    config_file = Path(__file__).parent / "config" / "scenario.ini"
    if config_file.exists():
        print("-" * 70)
        print("scenario.ini の環境設定:")
        print("-" * 70)
        
        config_content = config_file.read_text()
        for line in config_content.split('\n'):
            if '_environment' in line or 'scenario' in line:
                print(f"  {line}")
    
    print("=" * 70)

if __name__ == "__main__":
    main()
