#!/bin/bash
# FR脚のデバッグ情報を詳細に取得

echo "パターン1(全NONE)のFR脚デバッグ情報を取得中..."

# 環境設定
python3 set_environment.py NONE NONE NONE NONE > /dev/null 2>&1

# キャッシュクリア
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null

# Webots終了
pkill -9 webots > /dev/null 2>&1
pkill -9 webots-bin > /dev/null 2>&1
sleep 2

# Webotsを実行してログ保存
echo "Webots実行中..."
timeout 180 webots --mode=fast --no-rendering worlds/sotuken_world.wbt 2>&1 > /tmp/webots_debug.log

echo ""
echo "========== FR脚のデバッグ出力 =========="
grep -E "FR|TRAPPED判定|BURIED判定|NONE判定" /tmp/webots_debug.log | tail -50

echo ""
echo "========== 診断結果 =========="
python3 view_result.py 2>&1 | grep -A 20 "診断結果"
