#!/usr/bin/env python3
"""
Elbow motor test results viewer
"""
import json
import math
from pathlib import Path

def view_latest_results():
    log_dir = Path("/home/kk21053/sotuken/webots/controllers/environment_manager/logs")
    
    # Find latest diagnosis log
    diagnosis_logs = sorted(log_dir.glob("diagnosis_*.jsonl"))
    if not diagnosis_logs:
        print("❌ No diagnosis logs found")
        return
    
    latest_log = diagnosis_logs[-1]
    print("=" * 80)
    print(f"📊 Latest Diagnosis Results (Elbow Motor Test)")
    print(f"📁 File: {latest_log.name}")
    print("=" * 80)
    print()
    
    # Read all observations
    observations = []
    with open(latest_log, 'r') as f:
        for line in f:
            if line.strip():
                observations.append(json.loads(line))
    
    if not observations:
        print("❌ No observations found in log")
        return
    
    # Group by leg
    leg_data = {}
    for obs in observations:
        leg_id = obs['leg_id']
        if leg_id not in leg_data:
            leg_data[leg_id] = []
        leg_data[leg_id].append(obs)
    
    # Analyze each leg
    print("🦿 Individual Leg Results (Elbow Motor - index 2)")
    print("=" * 80)
    
    for leg_id in ["FL", "FR", "RL", "RR"]:
        if leg_id not in leg_data:
            print(f"\n❌ {leg_id}: No data")
            continue
        
        trials = leg_data[leg_id]
        
        print(f"\n【{leg_id}脚】 Expected: {'BURIED (砂に埋没)' if leg_id == 'FL' else 'NONE (正常)'}")
        print("-" * 80)
        
        # Calculate average displacement
        displacements = []
        for trial in trials:
            trial_num = trial['trial_index']
            direction = trial['direction']
            end_disp = trial['end_displacement']
            displacements.append(end_disp)
            
            # Show each trial
            theta_cmd = trial.get('theta_cmd_deg', [])
            theta_meas = trial.get('theta_meas_deg', [])
            
            if theta_cmd and theta_meas:
                cmd_change = theta_cmd[-1] - theta_cmd[0]
                meas_change = theta_meas[-1] - theta_meas[0]
                print(f"  Trial {trial_num} ({direction:>2}): "
                      f"cmd={cmd_change:+6.2f}°, "
                      f"actual={meas_change:+6.2f}°, "
                      f"end_disp={end_disp:.4f}m")
        
        avg_disp = sum(displacements) / len(displacements) if displacements else 0
        print(f"\n  平均変位: {avg_disp:.4f}m")
        print(f"  BURIED判定閾値: 0.0100m (1cm)")
        
        if avg_disp < 0.01:
            print(f"  → 検出: BURIED ⚠️")
        else:
            print(f"  → 検出: NONE ✓")
    
    # Get final diagnosis
    print("\n" + "=" * 80)
    print("🎯 Final Diagnosis (from LLM)")
    print("=" * 80)
    
    # Find diagnosis log
    diag_summary_logs = sorted(log_dir.glob("diagnosis_summary_*.jsonl"))
    if diag_summary_logs:
        latest_summary = diag_summary_logs[-1]
        with open(latest_summary, 'r') as f:
            for line in f:
                if line.strip():
                    summary = json.loads(line)
                    diagnoses = summary.get('diagnoses', {})
                    
                    for leg_id in ["FL", "FR", "RL", "RR"]:
                        if leg_id in diagnoses:
                            result = diagnoses[leg_id]
                            status = result.get('status', 'UNKNOWN')
                            confidence = result.get('confidence', 0.0)
                            
                            expected = "BURIED" if leg_id == "FL" else "NONE"
                            match = "✓" if status == expected else "✗"
                            
                            print(f"{leg_id}: {status:8} (confidence: {confidence:.3f}) "
                                  f"[Expected: {expected}] {match}")
    
    print("\n" + "=" * 80)
    print("📈 Analysis: Elbow Motor Performance")
    print("=" * 80)
    print()
    print("肘モーター(index 2)の特性:")
    print("  - 肘を曲げると足先が大きく動く")
    print("  - FL脚が砂に埋まっていると肘が曲がらない(はず)")
    print("  - 他の脚は自由に肘が曲がる")
    print()
    print("結果の見方:")
    print("  - FL脚の実際の角度変化が小さい → 砂に拘束されている → 成功")
    print("  - FL脚が3°近く動く → 砂でも肘が曲がる → 失敗")
    print("  - 他の脚が3°動く → 正常動作 → 成功")
    print("=" * 80)

if __name__ == "__main__":
    view_latest_results()
