#!/usr/bin/env python3
"""RRの誤診断を分析するスクリプト"""

import json
from pathlib import Path

def main():
    log_dir = Path('controllers/drone_circular_controller/logs')
    session_log = log_dir / 'leg_diagnostics_sessions.jsonl'
    
    # 最新のセッションを取得
    with open(session_log, 'r') as f:
        lines = f.readlines()
        latest_session = json.loads(lines[-1])
        session_id = latest_session['session_id']
    
    print(f'=== セッション: {session_id} ===\n')
    
    # イベントログから全データを収集
    events_log = log_dir / 'leg_diagnostics_events.jsonl'
    
    for leg_name in ['FL', 'FR', 'RL', 'RR']:
        print(f'=== {leg_name} ===')
        
        # このセッションの観測データを全て収集
        observations = []
        with open(events_log, 'r') as f:
            for line in f:
                data = json.loads(line)
                if (data.get('session_id') == session_id and 
                    data.get('leg_name') == leg_name):
                    observations.append(data)
        
        if not observations:
            print(f'  観測データなし\n')
            continue
        
        # 集計結果を探す
        aggregation_data = None
        for obs in observations:
            if obs.get('event') == 'aggregation':
                aggregation_data = obs
                break
        
        if aggregation_data:
            feat = aggregation_data.get('aggregated_features', {})
            dist = aggregation_data.get('cause_distribution', {})
            
            print(f"  観測回数: {aggregation_data.get('trial_count', 0)}")
            print(f"  delta_theta: {feat.get('delta_theta_deg', 0):.2f}°")
            print(f"  end_disp: {feat.get('end_disp_mm', 0):.2f}mm")
            print(f"  最終診断: {aggregation_data.get('final_cause', 'N/A')}")
            print(f"  確信度: {aggregation_data.get('confidence', 0):.3f}")
            
            if dist:
                print(f"  診断分布:")
                for cause, prob in sorted(dist.items(), key=lambda x: -x[1])[:3]:
                    print(f"    {cause}: {prob:.3f}")
        else:
            # 観測データから特徴量を確認
            print(f"  観測回数: {len(observations)}")
            if observations:
                last_obs = observations[-1]
                if 'features' in last_obs:
                    feat = last_obs['features']
                    print(f"  (最後の観測) delta_theta: {feat.get('delta_theta_deg', 0):.2f}°")
                    print(f"  (最後の観測) end_disp: {feat.get('end_disp_mm', 0):.2f}mm")
        
        print()

if __name__ == '__main__':
    main()
