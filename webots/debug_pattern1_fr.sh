#!/bin/bash
# FR脚のデバッグ情報を取得するスクリプト

echo "パターン1(全NONE)のFR脚デバッグ情報を取得中..."

# 環境設定
python3 set_environment.py NONE NONE NONE NONE > /dev/null 2>&1

# キャッシュクリア
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null

# Webots終了
pkill -9 webots > /dev/null 2>&1
pkill -9 webots-bin > /dev/null 2>&1
sleep 2

# Webotsを実行（デバッグ出力を保持）
echo "Webots実行中..."
timeout 180 webots --mode=fast --no-rendering worlds/sotuken_world.wbt 2>&1 | grep -E "(FR|TRAPPED|BURIED|NONE判定)" | tail -100

echo ""
echo "診断結果:"
python3 view_result.py 2>&1 | grep -A 20 "診断結果"
