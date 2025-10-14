"""
複合図（A/B/C）生成スクリプト - 完全日本語対応版
すべてのテキスト・軸名・ラベルを日本語化
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os
import japanize_matplotlib

# 日本語フォント設定（japanize_matplotlibが自動設定）

# 図の作成（全体統合版：A+B+C横並び）
fig_all = plt.figure(figsize=(20, 6))

# ==================== 図A左: Feature nodes規模比較 ====================
ax_a1 = plt.subplot(1, 4, 1)
categories = ['UAV/\nDrone', 'UGV/\nRobot']
x_pos = np.arange(len(categories))
width = 0.35

prior_feature = [350, 250]
ours_feature = [8, 8]

bars_a1 = ax_a1.bar(x_pos - width/2, prior_feature, width, label='先行 (BLS)', color='#FF9933', edgecolor='black', linewidth=1.2)
bars_a2 = ax_a1.bar(x_pos + width/2, ours_feature, width, label='本研究 (Proxy)', color='#CC3366', edgecolor='black', linewidth=1.2)

for i, (p, o) in enumerate(zip(prior_feature, ours_feature)):
    ax_a1.text(x_pos[i] - width/2, p + 10, str(p), ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax_a1.text(x_pos[i] + width/2, o + 10, str(o), ha='center', va='bottom', fontsize=10, fontweight='bold', color='darkred')

ax_a1.text(0.5, 370, '~97%削減', ha='center', va='center', fontsize=9,
          bbox=dict(boxstyle='round,pad=0.4', facecolor='yellow', edgecolor='black', linewidth=1.2))

ax_a1.set_ylabel('Feature nodes 数 (相当)', fontsize=10, fontweight='bold')
ax_a1.set_ylim(0, 400)
ax_a1.set_xticks(x_pos)
ax_a1.set_xticklabels(categories, fontsize=9)
ax_a1.set_title('[A] Feature段階の\n計算規模 (図4b proxy)', fontsize=10, fontweight='bold')
ax_a1.grid(axis='y', alpha=0.3)
ax_a1.legend(loc='upper right', fontsize=8)

# ==================== 図A右: Enhancement nodes規模比較 ====================
ax_a2 = plt.subplot(1, 4, 2)
prior_enhance = [3500, 3000]
ours_enhance = [24, 24]

bars_a3 = ax_a2.bar(x_pos - width/2, prior_enhance, width, label='先行 (BLS)', color='#FF9933', edgecolor='black', linewidth=1.2)
bars_a4 = ax_a2.bar(x_pos + width/2, ours_enhance, width, label='本研究 (Proxy)', color='#66CC66', edgecolor='black', linewidth=1.2)

for i, (p, o) in enumerate(zip(prior_enhance, ours_enhance)):
    ax_a2.text(x_pos[i] - width/2, p + 100, str(p), ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax_a2.text(x_pos[i] + width/2, o + 100, str(o), ha='center', va='bottom', fontsize=10, fontweight='bold', color='darkgreen')

ax_a2.text(0.5, 3700, '~99%削減', ha='center', va='center', fontsize=9,
          bbox=dict(boxstyle='round,pad=0.4', facecolor='yellow', edgecolor='black', linewidth=1.2))

ax_a2.set_ylabel('Enhancement nodes 数 (相当)', fontsize=10, fontweight='bold')
ax_a2.set_ylim(0, 4000)
ax_a2.set_xticks(x_pos)
ax_a2.set_xticklabels(categories, fontsize=9)
ax_a2.set_title('[A] Enhancement段階の\n計算規模 (図4a proxy)', fontsize=10, fontweight='bold')
ax_a2.grid(axis='y', alpha=0.3)
ax_a2.legend(loc='upper right', fontsize=8)

# ==================== 図B: 性能比較（速度・メモリ） ====================
ax_b = plt.subplot(1, 4, 3)
methods_perf = ['先行\n(BLS)', '本研究\n(ルール)']
x_pos_b = np.arange(len(methods_perf))

inference_median = [500, 0.08]
memory = [4785, 0.8]

# 速度
bars_b1 = ax_b.bar(x_pos_b - 0.18, inference_median, 0.35, label='推論時間 (ms)', 
                   color='#FF9933', edgecolor='black', linewidth=1.2)
ax_b.text(0 - 0.18, 550, '500 ms', ha='center', va='bottom', fontsize=9, fontweight='bold')
ax_b.text(1 - 0.18, 0.15, '0.08 ms', ha='center', va='bottom', fontsize=9, fontweight='bold', color='darkred')
ax_b.text(0.5, 200, '6250倍高速化', ha='center', va='center', fontsize=8,
         bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', edgecolor='black', linewidth=1))

ax_b.set_ylabel('推論時間 (ms/脚)', fontsize=10, fontweight='bold')
ax_b.set_yscale('log')
ax_b.set_ylim(0.05, 1000)
ax_b.set_xticks(x_pos_b - 0.18)
ax_b.set_xticklabels(methods_perf, fontsize=9)
ax_b.set_title('[B] 性能比較\n(速度・メモリ)', fontsize=10, fontweight='bold')
ax_b.grid(axis='y', alpha=0.3)

# メモリ（右軸）
ax_b_right = ax_b.twinx()
bars_b2 = ax_b_right.bar(x_pos_b + 0.18, memory, 0.35, label='メモリ (KB)', 
                         color='#6699CC', edgecolor='black', linewidth=1.2, alpha=0.7)
ax_b_right.text(0 + 0.18, 5000, '4785 KB', ha='center', va='bottom', fontsize=9, fontweight='bold')
ax_b_right.text(1 + 0.18, 3, '0.8 KB', ha='center', va='bottom', fontsize=9, fontweight='bold', color='darkblue')
ax_b_right.set_ylabel('メモリ使用量 (KB)', fontsize=10, fontweight='bold')
ax_b_right.set_yscale('log')
ax_b_right.set_ylim(0.5, 10000)

# 凡例
lines_b1, labels_b1 = ax_b.get_legend_handles_labels()
lines_b2, labels_b2 = ax_b_right.get_legend_handles_labels()
ax_b.legend(lines_b1 + lines_b2, labels_b1 + labels_b2, loc='upper left', fontsize=8)

# ==================== 図C: 精度比較 ====================
ax_c = plt.subplot(1, 4, 4)
methods_c = ['先行\n(BLS)', 'Drone\nのみ', 'Spot\nのみ', '融合\n(0.4/0.6)']
f1_scores = [96.23, 95.00, 92.00, 100.00]
colors_c = ['#FF9933', '#6699CC', '#9966CC', '#FF6699']
x_pos_c = np.arange(len(methods_c))

bars_c = ax_c.bar(x_pos_c, f1_scores, color=colors_c, edgecolor='black', linewidth=1.2)

for i, v in enumerate(f1_scores):
    ax_c.text(i, v + 0.5, f'{v:.2f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

ax_c.axhline(y=100, color='gray', linestyle='--', linewidth=1, alpha=0.7)
ax_c.text(3, 101, '100%\n(n=40)', ha='center', va='bottom', fontsize=8, fontweight='bold')
ax_c.text(1.5, 104, '+3.77 pts', ha='center', va='center', fontsize=9,
         bbox=dict(boxstyle='round,pad=0.4', facecolor='yellow', edgecolor='black', linewidth=1.2))

ax_c.set_ylabel('Macro-F1 スコア (%)', fontsize=10, fontweight='bold')
ax_c.set_ylim(88, 106)
ax_c.set_xticks(x_pos_c)
ax_c.set_xticklabels(methods_c, fontsize=9)
ax_c.set_title('[C] 診断精度\n(Macro-F1, n=10)', fontsize=10, fontweight='bold')
ax_c.grid(axis='y', alpha=0.3)

# 全体タイトル
fig_all.suptitle('図1　四足×ドローン共通診断の"軽量さ"と性能の比較\n' +
                'Feature nodes ≒ 基本特徴 (8), Enhancement nodes ≒ 派生計算 (24)',
                fontsize=13, fontweight='bold', y=0.98)

plt.tight_layout(rect=[0, 0.06, 1, 0.94])

# 脚注
caption_all = (
    '[A]: 先行図4に対応する規模指標（BLSノード数の代理：Feature=基本特徴8、Enhancement=派生計算24）。\n'
    '[B]: 同一環境での推論時間（中央値、p95）とメモリ使用量。\n'
    '[C]: 原因分類のMacro-F1（95%CI, n=10セッション/40診断）。\n'
    '測定条件: Webots R2023b, Ubuntu/WSL2, AMD/Intel x64, time.perf_counter()。出典: 先行図4（BLS）'
)
fig_all.text(0.5, 0.01, caption_all, ha='center', va='bottom', fontsize=8)

# 保存
output_dir = os.path.dirname(__file__)
fig_all.savefig(os.path.join(output_dir, 'fig1_composite_jp_final.png'), dpi=300, bbox_inches='tight')
fig_all.savefig(os.path.join(output_dir, 'fig1_composite_jp_final.pdf'), dpi=300, bbox_inches='tight')
print('✅ 完全日本語対応版（A+B+C横並び）を fig1_composite_jp_final.png / fig1_composite_jp_final.pdf に保存しました。')

plt.close()

print('\n=== 完全日本語対応版図の生成完了 ===')
print('- fig1_composite_jp_final.png/pdf: 全体統合版（A+B+C横並び、日本語フォント対応）')
print('\n【改善点】')
print('✓ 日本語フォント（japanize_matplotlib）による文字化け解消')
print('✓ すべての軸ラベル・タイトルを日本語化')
print('✓ 図内注記・凡例も完全日本語化')
