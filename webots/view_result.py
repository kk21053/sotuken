#!/usr/bin/env python3
"""
診断結果表示スクリプト

診断結果を読みやすい形式で表示します。
"""

import json
import sys
from pathlib import Path
from datetime import datetime

# ログディレクトリ
LOG_DIR = Path(__file__).parent / "controllers" / "diagnostics_pipeline" / "logs"
SESSION_LOG = LOG_DIR / "leg_diagnostics_sessions.jsonl"


def load_latest_result():
    """最新の診断結果を読み込む"""
    if not SESSION_LOG.exists():
        print(f"エラー: ログファイルが見つかりません: {SESSION_LOG}")
        return None
    
    # 最後の行を読み込む
    with SESSION_LOG.open('r') as f:
        lines = f.readlines()
        if not lines:
            print("エラー: ログファイルが空です")
            return None
        
        try:
            return json.loads(lines[-1])
        except json.JSONDecodeError as e:
            print(f"エラー: ログファイルの解析に失敗しました: {e}")
            return None


def format_movement_status(status):
    """動作状態を日本語に変換"""
    status_map = {
        "MOVES": "動く",
        "CAN_NOT_MOVE": "動かない",
        "PARTIALLY_MOVES": "一部動く"
    }
    return status_map.get(status, status)


def format_cause(cause):
    """拘束原因を日本語に変換"""
    cause_map = {
        "NONE": "正常",
        "BURIED": "埋まる",
        "TRAPPED": "挟まる",
        "TANGLED": "絡まる",
        "MALFUNCTION": "故障"
    }
    return cause_map.get(cause, cause)


def format_fallen_status(is_fallen):
    """転倒状態を日本語に変換"""
    return "転倒している" if is_fallen else "転倒していない"


def display_result(result):
    """診断結果を表示"""
    print()
    print("=" * 80)
    print("診断結果")
    print("=" * 80)
    print(f"セッションID: {result.get('session_id', 'N/A')}")
    print(f"診断時刻:     {result.get('timestamp', 'N/A')}")
    print()
    
    # 各脚の結果
    print("-" * 80)
    print(f"{'脚':^6} | {'動作状態':^12} | {'確信度':^8} | {'拘束原因':^10} | {'期待値':^10}")
    print("-" * 80)
    
    legs = result.get('legs', {})
    for leg_id in ['FL', 'FR', 'RL', 'RR']:
        if leg_id not in legs:
            print(f"{leg_id:^6} | {'データなし':^12} | {'N/A':^8} | {'N/A':^10} | {'N/A':^10}")
            continue
        
        leg_data = legs[leg_id]
        movement = format_movement_status(leg_data.get('movement_status', 'N/A'))
        confidence = leg_data.get('p_can', 0.0)
        cause = format_cause(leg_data.get('cause', 'N/A'))
        expected_cause = format_cause(leg_data.get('expected_cause', 'N/A'))
        
        print(f"{leg_id:^6} | {movement:^12} | {confidence:^8.3f} | {cause:^10} | {expected_cause:^10}")
    
    print("-" * 80)
    print()
    
    # 転倒状態
    is_fallen = result.get('is_fallen', False)
    fallen_prob = result.get('fallen_probability', 0.0)
    print(f"転倒状態: {format_fallen_status(is_fallen)} (確率: {fallen_prob:.3f})")
    print()
    
    # 正解率（期待値との比較）
    correct_count = 0
    total_count = 0
    for leg_id in ['FL', 'FR', 'RL', 'RR']:
        if leg_id in legs:
            leg_data = legs[leg_id]
            if leg_data.get('cause') == leg_data.get('expected_cause'):
                correct_count += 1
            total_count += 1
    
    if total_count > 0:
        accuracy = (correct_count / total_count) * 100
        print(f"診断精度: {correct_count}/{total_count} ({accuracy:.1f}%)")
        print()
    
    print("=" * 80)
    print()


def main():
    """メイン処理"""
    result = load_latest_result()
    if result is None:
        sys.exit(1)
    
    display_result(result)


if __name__ == "__main__":
    main()
