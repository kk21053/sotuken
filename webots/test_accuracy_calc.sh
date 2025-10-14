#!/bin/bash

echo "精度計算のテスト"
echo "===================="
echo "仮定: 100回実行"
echo "  - 67回が100.0%"
echo "  - 33回が75.0%"
echo ""

# 結果を保存する配列
declare -a results

# 67回の100.0%を追加
for i in {1..67}; do
    results+=("100.0")
done

# 33回の75.0%を追加
for i in {1..33}; do
    results+=("75.0")
done

echo "結果配列に追加完了"
echo "配列の要素数: ${#results[@]}"
echo ""

# 精度の平均を計算
total=0
count=0
for accuracy in "${results[@]}"; do
    total=$(echo "$total + $accuracy" | bc)
    count=$((count + 1))
done

if [ $count -gt 0 ]; then
    average=$(echo "scale=2; $total / $count" | bc)
    echo "========================================================================"
    echo "検証完了 - 結果集計"
    echo "========================================================================"
    echo ""
    echo "実行回数: $count"
    echo "平均診断精度: ${average}%"
    echo ""
    
    # 詳細な内訳
    echo "詳細内訳:"
    echo "  100.0%の回数: 67回"
    echo "  75.0%の回数: 33回"
    echo ""
    
    # 計算式を表示
    total_display=$(echo "scale=2; 67 * 100.0 + 33 * 75.0" | bc)
    echo "計算式: (67 × 100.0 + 33 × 75.0) / 100"
    echo "      = ${total_display} / 100"
    echo "      = ${average}%"
    echo ""
else
    echo "エラー: 有効な結果が得られませんでした"
fi

echo "========================================================================"
