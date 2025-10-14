#!/usr/bin/env python3
import json
from pathlib import Path

# 診断結果のJSONを探す
result_dir = Path('controllers/spot_self_diagnosis/results')
if result_dir.exists():
    json_files = sorted(result_dir.glob('diagnosis_result_*.json'), 
                       key=lambda p: p.stat().st_mtime, reverse=True)
    if json_files:
        latest = json_files[0]
        print(f'最新結果: {latest.name}\n')
        
        with open(latest, 'r') as f:
            result = json.load(f)
            
        for leg in ['FL', 'FR', 'RL', 'RR']:
            if leg in result.get('legs', {}):
                leg_data = result['legs'][leg]
                feat = leg_data.get('features', {})
                print(f'=== {leg} ===')
                print(f"  診断: {leg_data.get('diagnosis', 'N/A')}")
                print(f"  確信度: {leg_data.get('confidence', 0):.3f}")
                print(f"  delta_theta: {feat.get('delta_theta_deg', 0):.2f}°")
                print(f"  end_disp: {feat.get('end_disp_mm', 0):.2f}mm")
                print()
else:
    print('結果ディレクトリが見つかりません')
