#!/usr/bin/env python3
"""
診断結果の確認スクリプト - シンプル版
使い方: python3 check_result.py
"""
import json
from pathlib import Path
from datetime import datetime

def check_latest_diagnosis():
    """最新の診断結果を表示"""
    
    # ログディレクトリのパス
    log_dir = Path("controllers/spot_self_diagnosis/logs")
    
    # 最新の診断ログを探す
    diagnosis_logs = sorted(log_dir.glob("diagnosis_*.jsonl"))
    if not diagnosis_logs:
        print("❌ エラー: 診断ログが見つかりません")
        print(f"📁 確認場所: {log_dir.absolute()}")
        return
    
    latest_log = diagnosis_logs[-1]
    
    # ファイル名から日時を抽出
    filename = latest_log.stem  # diagnosis_20251012_012023
    timestamp = filename.replace('diagnosis_', '')
    
    print("=" * 80)
    print(f"📊 Spot脚診断システム - 結果表示")
    print(f"📁 ログファイル: {latest_log.name}")
    print(f"🕒 実行日時: {timestamp[:8]}-{timestamp[9:11]}:{timestamp[11:13]}:{timestamp[13:15]}")
    print("=" * 80)
    print()
    
    # ログファイルを読み込む
    with open(latest_log, 'r') as f:
        lines = f.readlines()
    
    if not lines:
        print("❌ エラー: ログファイルが空です")
        return
    
    # メタデータ(1行目)
    metadata = json.loads(lines[0])
    print(f"📋 実験設定:")
    print(f"   - 脚の数: {metadata.get('num_legs', 4)}")
    print(f"   - 1脚あたりの試行回数: {metadata.get('trials_per_leg', 4)}")
    print()
    
    # 各脚の結果(2行目以降)
    print("🦿 各脚の診断結果:")
    print("=" * 80)
    
    results = []
    for line in lines[1:]:
        if line.strip():
            leg_result = json.loads(line)
            results.append(leg_result)
    
    # 脚の順番でソート
    leg_order = {"FL": 0, "FR": 1, "RL": 2, "RR": 3}
    results.sort(key=lambda x: leg_order.get(x['leg_id'], 99))
    
    # 各脚の結果を表示
    correct_count = 0
    total_count = 0
    
    for leg_result in results:
        leg_id = leg_result['leg_id']
        detected_cause = leg_result['cause_final']
        confidence = leg_result['conf_final']
        
        # 期待される結果
        expected_cause = "BURIED" if leg_id == "FL" else "NONE"
        
        # 正解判定
        is_correct = (detected_cause == expected_cause)
        if is_correct:
            correct_count += 1
        total_count += 1
        
        # 記号で表示
        status_mark = "✓" if is_correct else "✗"
        
        print(f"{leg_id}脚:")
        print(f"  検出結果: {detected_cause:10} (信頼度: {confidence:.3f})")
        print(f"  期待値:   {expected_cause:10}")
        print(f"  判定:     {status_mark} {'正解' if is_correct else '不正解'}")
        print()
    
    # 精度を計算
    accuracy = (correct_count / total_count * 100) if total_count > 0 else 0
    
    print("=" * 80)
    print(f"📈 診断精度: {correct_count}/{total_count} = {accuracy:.1f}%")
    print("=" * 80)
    print()
    
    # 詳細情報へのリンク
    print("💡 ヒント:")
    print(f"   - 詳細データを見るには: python3 check_detail.py")
    print(f"   - 生ログを見るには: cat {latest_log}")
    print()

if __name__ == "__main__":
    check_latest_diagnosis()
