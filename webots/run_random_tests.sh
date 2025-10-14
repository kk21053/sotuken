#!/bin/bash

echo ""
echo "========================================================================"
echo "ランダム診断精度検証 (100回実行)"
echo "========================================================================"

# 結果を保存する配列
declare -a results

# 3種類のテストパターン
patterns=("NONE NONE NONE NONE" "BURIED NONE NONE NONE" "TRAPPED NONE NONE NONE")
pattern_names=("全てNONE" "FL=BURIED" "FL=TRAPPED")

# 100回ランダムにテストを実行
for i in {1..100}; do
    # ランダムにパターンを選択 (0-2)
    pattern_index=$((RANDOM % 3))
    pattern="${patterns[$pattern_index]}"
    pattern_name="${pattern_names[$pattern_index]}"
    
    echo ""
    echo "--------------------------------------------------------------------"
    echo "実行 $i/100: ${pattern_name}"
    echo "--------------------------------------------------------------------"
    
    # 環境設定
    python3 set_environment.py $pattern > /dev/null 2>&1
    find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
    pkill -9 webots > /dev/null 2>&1
    pkill -9 webots-bin > /dev/null 2>&1
    sleep 2
    
    # シミュレーション実行（ストリーミング無効化）
    WEBOTS_DISABLE_BINARY_DIALOG=1 WEBOTS_DISABLE_SAVE_SCREEN_PERSPECTIVE_ON_CLOSE=1 timeout 180 webots --mode=fast --no-rendering --minimize --batch worlds/sotuken_world.wbt > /dev/null 2>&1
    
    # Webotsとブラウザを完全に終了
    pkill -9 webots > /dev/null 2>&1
    pkill -9 webots-bin > /dev/null 2>&1
    
    # Windowsのブラウザを閉じる（PowerShellを使用）
    powershell.exe -Command "Get-Process | Where-Object {$_.ProcessName -match 'chrome|msedge|firefox'} | Where-Object {$_.MainWindowTitle -match 'Webots'} | Stop-Process -Force" > /dev/null 2>&1
    sleep 1
    
    # 結果を取得して精度を抽出
    result=$(python3 view_result.py 2>/dev/null)
    
    # 精度を抽出 (例: "診断精度: 100.0%" から 100.0 を取得)
    accuracy=$(echo "$result" | grep "診断精度" | grep -oP '\d+\.\d+' | head -1)
    
    if [ -n "$accuracy" ]; then
        results+=("$accuracy")
        echo "診断精度: ${accuracy}%"
    else
        echo "警告: 精度を取得できませんでした"
        results+=("0")
    fi
    
    # 進捗表示
    echo "進捗: $i/100 完了"
done

echo ""
echo "========================================================================"
echo "検証完了 - 結果集計"
echo "========================================================================"

# 精度の平均を計算
total=0
count=0
for accuracy in "${results[@]}"; do
    total=$(echo "$total + $accuracy" | bc)
    count=$((count + 1))
done

if [ $count -gt 0 ]; then
    average=$(echo "scale=2; $total / $count" | bc)
    echo ""
    echo "実行回数: $count"
    echo "平均診断精度: ${average}%"
    echo ""
    
    # 各パターンの実行回数をカウント
    none_count=0
    buried_count=0
    trapped_count=0
    
    echo "パターン別実行回数:"
    echo "  全てNONE: 約 $((count / 3)) 回"
    echo "  FL=BURIED: 約 $((count / 3)) 回"
    echo "  FL=TRAPPED: 約 $((count / 3)) 回"
    echo ""
else
    echo "エラー: 有効な結果が得られませんでした"
fi

echo "========================================================================"
