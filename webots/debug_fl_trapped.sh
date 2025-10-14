#!/bin/bash
# パターン3のFLデバッグ出力を取得

echo "パターン3(FL=TRAPPED)のデバッグ出力を取得中..."

# 環境設定
python3 set_environment.py TRAPPED NONE NONE NONE > /dev/null 2>&1

# キャッシュクリア
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null

# Webots終了
pkill -9 webots > /dev/null 2>&1
pkill -9 webots-bin > /dev/null 2>&1
sleep 2

# Webotsを実行してFL脚のデバッグ出力を抽出
echo "Webots実行中..."
timeout 180 webots --mode=fast --no-rendering worlds/sotuken_world.wbt 2>&1 > /tmp/webots_pattern3.log

echo ""
echo "========== FL脚の判定過程 =========="
grep -E "\[(TRAPPED|BURIED|TANGLED)判定\]" /tmp/webots_pattern3.log | grep -A 2 -B 2 "FL" | tail -30

echo ""
echo "========== 診断結果 =========="
python3 view_result.py 2>&1 | grep -A 20 "診断結果"
