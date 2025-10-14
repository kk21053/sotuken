#!/bin/bash
# 環境設定の動作テストスクリプト

echo "======================================================================"
echo "環境設定動作テスト"
echo "======================================================================"
echo ""

# テストケース1: 全てNONE
echo "[テスト1] 全脚NONE設定"
python3 set_environment.py NONE NONE NONE NONE
echo ""
echo "結果確認:"
python3 test_environment_visibility.py | grep -E "(状態:|scenario|environment)"
echo ""
read -p "Enter キーで次のテストへ..."
echo ""

# テストケース2: BURIED
echo "[テスト2] FL脚BURIED設定"
python3 set_environment.py BURIED NONE NONE NONE
echo ""
echo "結果確認:"
python3 test_environment_visibility.py | grep -E "(状態:|scenario|environment)"
echo ""
read -p "Enter キーで次のテストへ..."
echo ""

# テストケース3: TRAPPED
echo "[テスト3] FL脚TRAPPED設定"
python3 set_environment.py TRAPPED NONE NONE NONE
echo ""
echo "結果確認:"
python3 test_environment_visibility.py | grep -E "(状態:|scenario|environment)"
echo ""
read -p "Enter キーで次のテストへ..."
echo ""

# テストケース4: 複数環境
echo "[テスト4] 複数環境（FL=BURIED, FR=TRAPPED）"
echo "注意: 現在のワールドファイルは1つの環境のみサポート"
python3 set_environment.py BURIED TRAPPED NONE NONE
echo ""
echo "結果確認:"
python3 test_environment_visibility.py | grep -E "(状態:|scenario|environment)"
echo ""

echo "======================================================================"
echo "テスト完了"
echo "======================================================================"
echo ""
echo "Webotsで視覚的に確認する場合:"
echo "  pkill -9 webots; pkill -9 webots-bin; sleep 2 && webots worlds/sotuken_world.wbt"
