#!/bin/bash

echo ""
echo "========================================================================"
echo "包括的診断精度検証"
echo "========================================================================"

# パターン1: 全ての脚がNONE
echo ""
echo "--------------------------------------------------------------------"
echo "パターン1: 全ての脚がNONE"
echo "--------------------------------------------------------------------"
python3 set_environment.py NONE NONE NONE NONE > /dev/null 2>&1
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
pkill -9 webots > /dev/null 2>&1
pkill -9 webots-bin > /dev/null 2>&1
sleep 2
timeout 180 webots --mode=fast --no-rendering worlds/sotuken_world.wbt > /dev/null 2>&1
python3 view_result.py | grep -A 20 "診断結果"

# パターン2: FL=BURIED
echo ""
echo "--------------------------------------------------------------------"
echo "パターン2: FL=BURIED"
echo "--------------------------------------------------------------------"
python3 set_environment.py BURIED NONE NONE NONE > /dev/null 2>&1
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
pkill -9 webots > /dev/null 2>&1
pkill -9 webots-bin > /dev/null 2>&1
sleep 2
timeout 180 webots --mode=fast --no-rendering worlds/sotuken_world.wbt > /dev/null 2>&1
python3 view_result.py | grep -A 20 "診断結果"

# パターン3: FL=TRAPPED
echo ""
echo "--------------------------------------------------------------------"
echo "パターン3: FL=TRAPPED"
echo "--------------------------------------------------------------------"
python3 set_environment.py TRAPPED NONE NONE NONE > /dev/null 2>&1
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
pkill -9 webots > /dev/null 2>&1
pkill -9 webots-bin > /dev/null 2>&1
sleep 2
timeout 180 webots --mode=fast --no-rendering worlds/sotuken_world.wbt > /dev/null 2>&1
python3 view_result.py | grep -A 20 "診断結果"

echo ""
echo "========================================================================"
echo "検証完了"
echo "========================================================================"
