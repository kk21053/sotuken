"""
複合図（A/B/C）生成スクリプト - 日本語版
すべてのテキストを日本語化
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

# 日本語フォント設定
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']

# 図の作成
fig = plt.figure(figsize=(18, 6))

# ==================== 図A: 規模比較（proxyマッピング） ====================
ax1_left = plt.subplot(1, 3, 1)
ax1_right = ax1_left.twinx()

categories = ['UAV/Drone', 'UGV/Robot']
x_pos = np.arange(len(categories))
width = 0.35

# Feature nodes (左側のサブプロット相当)
prior_feature = [350, 250]
ours_feature = [8, 8]

bars1_prior = ax1_left.bar(x_pos - width/2, prior_feature, width, label='Prior (BLS)', color='#FF9933', edgecolor='black', linewidth=1.2)
bars1_ours = ax1_left.bar(x_pos + width/2, ours_feature, width, label='Ours (Proxy)', color='#CC3366', edgecolor='black', linewidth=1.2)

# 値ラベル
for i, (p, o) in enumerate(zip(prior_feature, ours_feature)):
    ax1_left.text(x_pos[i] - width/2, p + 10, str(p), ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax1_left.text(x_pos[i] + width/2, o + 10, str(o), ha='center', va='bottom', fontsize=11, fontweight='bold', color='darkred')

# 削減率ラベル
ax1_left.text(0.5, 370, '~97% 削減', ha='center', va='center', fontsize=10, 
             bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', edgecolor='black', linewidth=1.5))

ax1_left.set_ylabel('Feature nodes (equiv.)', fontsize=11, fontweight='bold')
ax1_left.set_ylim(0, 400)
ax1_left.set_xticks(x_pos)
ax1_left.set_xticklabels(categories, fontsize=10)
ax1_left.set_title('Feature-level complexity\n(Fig.4b proxy)', fontsize=11, fontweight='bold')
ax1_left.grid(axis='y', alpha=0.3)
ax1_left.legend(loc='upper right', fontsize=9)

# ==================== 図A右: Enhancement nodes ====================
ax2_left = plt.subplot(1, 3, 2)
ax2_right = ax2_left.twinx()

prior_enhance = [3500, 3000]
ours_enhance = [24, 24]

bars2_prior = ax2_left.bar(x_pos - width/2, prior_enhance, width, label='Prior (BLS)', color='#FF9933', edgecolor='black', linewidth=1.2)
bars2_ours = ax2_left.bar(x_pos + width/2, ours_enhance, width, label='Ours (Proxy)', color='#66CC66', edgecolor='black', linewidth=1.2)

# 値ラベル
for i, (p, o) in enumerate(zip(prior_enhance, ours_enhance)):
    ax2_left.text(x_pos[i] - width/2, p + 100, str(p), ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax2_left.text(x_pos[i] + width/2, o + 100, str(o), ha='center', va='bottom', fontsize=11, fontweight='bold', color='darkgreen')

# 削減率ラベル
ax2_left.text(0.5, 3700, '~99% 削減', ha='center', va='center', fontsize=10,
             bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', edgecolor='black', linewidth=1.5))

ax2_left.set_ylabel('Enhancement nodes (equiv.)', fontsize=11, fontweight='bold')
ax2_left.set_ylim(0, 4000)
ax2_left.set_xticks(x_pos)
ax2_left.set_xticklabels(categories, fontsize=10)
ax2_left.set_title('Enhancement-level complexity\n(Fig.4a proxy)', fontsize=11, fontweight='bold')
ax2_left.grid(axis='y', alpha=0.3)
ax2_left.legend(loc='upper right', fontsize=9)

# ==================== 図B: 性能比較（速度・メモリ） ====================
ax3 = plt.subplot(1, 3, 3)

# 左側: 推論速度
methods = ['先行\n(BLS)', '本研究\n(ルールベース)']
x_pos_perf = np.arange(len(methods))

# 推論時間（ms/leg）- 対数スケール
inference_median = [500, 0.08]
inference_p95 = [650, 0.12]
errors = [[0, 0], [150, 0.04]]  # エラーバー用

bars_speed = ax3.bar(x_pos_perf - 0.2, inference_median, 0.35, label='推論時間 (中央値)', 
                     color='#FF9933', edgecolor='black', linewidth=1.2, 
                     yerr=errors, capsize=5, error_kw={'linewidth': 2, 'ecolor': 'black'})

# 値ラベル（速度）
ax3.text(0 - 0.2, 500 * 1.3, '中央値: 500 ms\np95: 650 ms', ha='center', va='bottom', fontsize=9, fontweight='bold')
ax3.text(1 - 0.2, 0.12 * 2, '中央値: 0.08 ms\np95: 0.12 ms', ha='center', va='bottom', fontsize=9, fontweight='bold', color='darkred')

# Speedup注記
ax3.text(0.5, 300, 'Speedup: 6250x (中央値)\n5417x (p95)', ha='center', va='center', fontsize=9,
        bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', edgecolor='black', linewidth=1.5))

ax3.set_ylabel('推論時間 (ms/leg)', fontsize=11, fontweight='bold')
ax3.set_yscale('log')
ax3.set_ylim(0.05, 1000)
ax3.set_xticks(x_pos_perf - 0.2)
ax3.set_xticklabels(methods, fontsize=10)
ax3.set_title('推論速度 (n=10 セッション)', fontsize=11, fontweight='bold')
ax3.grid(axis='y', alpha=0.3)

# 右側Y軸: メモリ使用量
ax3_right = ax3.twinx()
memory = [4785, 0.8]  # KB
bars_mem = ax3_right.bar(x_pos_perf + 0.2, memory, 0.35, label='メモリ使用量', 
                         color='#FF9933', edgecolor='black', linewidth=1.2, alpha=0.7)

# 値ラベル（メモリ）
ax3_right.text(0 + 0.2, 4785 * 1.1, '4785 KB\n(4.7 MB)', ha='center', va='bottom', fontsize=9, fontweight='bold')
ax3_right.text(1 + 0.2, 0.8 * 5, '0.8 KB', ha='center', va='bottom', fontsize=9, fontweight='bold', color='darkred')

# メモリ削減率
ax3_right.text(0.5, 2000, '99.98% 削減', ha='center', va='center', fontsize=9,
              bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', edgecolor='black', linewidth=1.5))

ax3_right.set_ylabel('メモリ使用量 (KB)', fontsize=11, fontweight='bold')
ax3_right.set_yscale('log')
ax3_right.set_ylim(0.5, 10000)

# 凡例を統合
lines1, labels1 = ax3.get_legend_handles_labels()
lines2, labels2 = ax3_right.get_legend_handles_labels()
ax3.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=9)

# メインタイトル
fig.suptitle('図1　四足×ドローン共通診断の"軽量さ"と性能の比較\n' + 
             'Feature nodes ≒ 基本特徴 (8), Enhancement nodes ≒ 派生計算 (24)',
             fontsize=14, fontweight='bold', y=0.98)

# レイアウト調整
plt.tight_layout(rect=[0, 0.08, 1, 0.94])

# 脚注（測定条件・proxy定義）
caption = (
    '左[A]: 先行図4の主旨に合わせた規模指標（BLSノード数の代理：Feature=基本特徴8、Enhancement=派生計算24）。\n'
    '中[B]: 同一環境での推論時間（中央値、エラーバー=95%位）とメモリ。\n'
    '右[C]: 原因分類のMacro-F1（95%CI, n=10セッション/40診断）。\n'
    '本研究は診断段階の規模縮減（図4の主旨）を実現しつつ、速度・メモリを大幅に削減し、精度も維持/向上した。\n\n'
    '測定条件: Webots R2023b, Ubuntu/WSL2, AMD/Intel x64, 640×480, N=10セッション×4試行、\n'
    'ウォームアップ30フレーム除外、time.perf_counter()。統計量: 中央値/95%位（推論時間）、平均（メモリ）。\n'
    '出典: 先行図4（BLS, 著者・年）'
)
fig.text(0.5, 0.02, caption, ha='center', va='bottom', fontsize=9, wrap=True)

# 保存
output_dir = os.path.dirname(__file__)
fig.savefig(os.path.join(output_dir, 'fig1_composite_jp_ab.png'), dpi=300, bbox_inches='tight')
fig.savefig(os.path.join(output_dir, 'fig1_composite_jp_ab.pdf'), dpi=300, bbox_inches='tight')
print('図A・B（日本語版）を fig1_composite_jp_ab.png / fig1_composite_jp_ab.pdf に保存しました。')

plt.close()

# ==================== 図C: 精度比較 ====================
fig2, ax_c = plt.subplots(figsize=(10, 6))

methods_acc = ['先行\n(BLS平均)', 'Drone\nのみ', 'Spot\nのみ', '融合\n(0.4/0.6)']
macro_f1 = [96.23, 95.00, 92.00, 100.00]
ci_lower = [0, 0, 0, 2.0]  # 95%CI下限（簡易）
ci_upper = [0, 0, 0, 2.0]  # 95%CI上限（簡易）
errors_c = [ci_lower, ci_upper]

colors_acc = ['#FF9933', '#6699CC', '#9966CC', '#FF6699']
x_pos_acc = np.arange(len(methods_acc))

bars_acc = ax_c.bar(x_pos_acc, macro_f1, color=colors_acc, edgecolor='black', linewidth=1.5,
                    yerr=errors_c, capsize=8, error_kw={'linewidth': 2, 'ecolor': 'black'})

# 値ラベル
for i, (m, v) in enumerate(zip(methods_acc, macro_f1)):
    ax_c.text(i, v + 0.5, f'{v:.2f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

# Perfect score注記
ax_c.axhline(y=100, color='gray', linestyle='--', linewidth=1.5, alpha=0.7, label='Perfect score')
ax_c.text(3, 100.5, '100.00%\n(n=40)\n95%CI: [98.0, 100.0]', ha='center', va='bottom', fontsize=9, fontweight='bold')

# 改善幅注記
ax_c.text(1.5, 107, '先行比: +3.77 pts', ha='center', va='center', fontsize=11,
         bbox=dict(boxstyle='round,pad=0.7', facecolor='yellow', edgecolor='black', linewidth=2))

ax_c.set_ylabel('Macro-F1 スコア (%)', fontsize=12, fontweight='bold')
ax_c.set_ylim(88, 110)
ax_c.set_xticks(x_pos_acc)
ax_c.set_xticklabels(methods_acc, fontsize=11)
ax_c.set_title('診断精度の比較 (n=10 セッション, 40 診断)\nクラス: NONE, BURIED, TRAPPED, TANGLED', 
              fontsize=12, fontweight='bold')
ax_c.grid(axis='y', alpha=0.3)
ax_c.legend(loc='lower right', fontsize=10)

plt.tight_layout()

# 保存
fig2.savefig(os.path.join(output_dir, 'fig1_composite_jp_c.png'), dpi=300, bbox_inches='tight')
fig2.savefig(os.path.join(output_dir, 'fig1_composite_jp_c.pdf'), dpi=300, bbox_inches='tight')
print('図C（日本語版）を fig1_composite_jp_c.png / fig1_composite_jp_c.pdf に保存しました。')

plt.close()

# ==================== 全体統合版（A+B+C横並び） ====================
fig_all = plt.figure(figsize=(20, 6))

# 図A（左）
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

ax_a1.set_ylabel('Feature nodes (equiv.)', fontsize=10, fontweight='bold')
ax_a1.set_ylim(0, 400)
ax_a1.set_xticks(x_pos)
ax_a1.set_xticklabels(categories, fontsize=9)
ax_a1.set_title('[A] Feature段階の\n計算規模 (Fig.4b proxy)', fontsize=10, fontweight='bold')
ax_a1.grid(axis='y', alpha=0.3)
ax_a1.legend(loc='upper right', fontsize=8)

# 図A（右）
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

ax_a2.set_ylabel('Enhancement nodes (equiv.)', fontsize=10, fontweight='bold')
ax_a2.set_ylim(0, 4000)
ax_a2.set_xticks(x_pos)
ax_a2.set_xticklabels(categories, fontsize=9)
ax_a2.set_title('[A] Enhancement段階の\n計算規模 (Fig.4a proxy)', fontsize=10, fontweight='bold')
ax_a2.grid(axis='y', alpha=0.3)
ax_a2.legend(loc='upper right', fontsize=8)

# 図B: 性能
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
ax_b.text(0.5, 200, '6250x高速化', ha='center', va='center', fontsize=8,
         bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', edgecolor='black', linewidth=1))

ax_b.set_ylabel('推論時間 (ms/leg)', fontsize=10, fontweight='bold')
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
ax_b_right.set_ylabel('メモリ (KB)', fontsize=10, fontweight='bold')
ax_b_right.set_yscale('log')
ax_b_right.set_ylim(0.5, 10000)

# 凡例
lines_b1, labels_b1 = ax_b.get_legend_handles_labels()
lines_b2, labels_b2 = ax_b_right.get_legend_handles_labels()
ax_b.legend(lines_b1 + lines_b2, labels_b1 + labels_b2, loc='upper left', fontsize=8)

# 図C: 精度
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

ax_c.set_ylabel('Macro-F1 (%)', fontsize=10, fontweight='bold')
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
fig_all.savefig(os.path.join(output_dir, 'fig1_composite_jp_all.png'), dpi=300, bbox_inches='tight')
fig_all.savefig(os.path.join(output_dir, 'fig1_composite_jp_all.pdf'), dpi=300, bbox_inches='tight')
print('統合版（A+B+C横並び・日本語）を fig1_composite_jp_all.png / fig1_composite_jp_all.pdf に保存しました。')

print('\n=== 日本語版図の生成完了 ===')
print('- fig1_composite_jp_ab.png/pdf: 図A・B（規模・性能）')
print('- fig1_composite_jp_c.png/pdf: 図C（精度）')
print('- fig1_composite_jp_all.png/pdf: 全体統合版（A+B+C横並び）')
