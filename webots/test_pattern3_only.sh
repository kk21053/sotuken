#!/bin/bash
# パターン3のみを実行してデバッグ出力を取得

echo "パターン3(FL=TRAPPED)を実行中..."

# 環境設定
python3 set_environment.py TRAPPED NONE NONE NONE > /dev/null 2>&1

# キャッシュクリア
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null

# Webots終了
pkill -9 webots > /dev/null 2>&1
pkill -9 webots-bin > /dev/null 2>&1
sleep 2

# Webotsを実行（デバッグ出力を保持）
echo "Webots実行中..."
timeout 180 webots --mode=fast --no-rendering worlds/sotuken_world.wbt 2>&1 | grep -E "\[(TRAPPED|BURIED|TANGLED)判定\].*FL" | head -20

echo ""
echo "診断結果:"
python3 view_result.py 2>&1 | grep -A 25 "診断結果"
