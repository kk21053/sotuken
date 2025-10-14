import json
import glob
import os

# 最新の診断結果を取得
result_dirs = sorted(glob.glob("drone_diagnosis_*"), reverse=True)
if not result_dirs:
    print("診断結果が見つかりません")
    exit(1)

latest_dir = result_dirs[0]
print(f"最新の診断結果: {latest_dir}\n")

# features.jsonが存在しない場合はログから直接解析
features_file = os.path.join(latest_dir, "features.json")
if not os.path.exists(features_file):
    print("features.jsonが見つかりません。ログを確認してください。")
    
    # ログファイルから特徴量を探す
    log_files = glob.glob(os.path.join(latest_dir, "*.log"))
    for log_file in log_files:
        print(f"\n{log_file}を確認中...")
        with open(log_file, 'r') as f:
            for line in f:
                if 'RR' in line and ('中央値' in line or 'TRAPPED' in line):
                    print(line.strip())
else:
    # features.jsonから読み込み
    with open(features_file, 'r') as f:
        data = json.load(f)
        
    for leg in ['FL', 'FR', 'RL', 'RR']:
        print(f"\n===== {leg} =====")
        features = data['leg_features'][leg]
        print(f"trials: {features['trials']}")
        print(f"delta_theta: {features['delta_theta']:.4f}°")
        print(f"end_disp: {features['end_disp']:.4f}mm")
        print(f"avg_disp: {features['avg_disp']:.4f}mm")
        print(f"straightness: {features['straightness']:.4f}")
        
        # 期待値と比較
        expected = data.get('expected_causes', {}).get(leg, 'unknown')
        actual = data.get('diagnosed_causes', {}).get(leg, 'unknown')
        print(f"期待: {expected}, 実際: {actual}")
        
        if expected != actual:
            print(f"⚠️ 誤診断!")
