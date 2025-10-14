# 複合図（A/B/C）生成スクリプト
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os

# 画像ファイルパス
fig_a = 'fig_A_complexity_proxy.png'
fig_b = 'fig_B_performance.png'
fig_c = 'fig_C_accuracy.png'

# フルパス取得
base_dir = os.path.dirname(__file__)
fig_a_path = os.path.join(base_dir, fig_a)
fig_b_path = os.path.join(base_dir, fig_b)
fig_c_path = os.path.join(base_dir, fig_c)

# 画像読み込み
img_a = mpimg.imread(fig_a_path)
img_b = mpimg.imread(fig_b_path)
img_c = mpimg.imread(fig_c_path)

# 複合図作成
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(img_a)
axes[0].set_title('[A] 規模（図4対応proxy）', fontsize=12)
axes[0].axis('off')
axes[1].imshow(img_b)
axes[1].set_title('[B] 速度＋メモリ', fontsize=12)
axes[1].axis('off')
axes[2].imshow(img_c)
axes[2].set_title('[C] 精度（Macro-F1）', fontsize=12)
axes[2].axis('off')

fig.suptitle('図1　四足×ドローン共通診断の“軽量さ”と性能の比較', fontsize=14)

# 脚注（proxy定義・測定条件・統計量）
caption = (
    '左[A]は先行図4の主旨に合わせた規模指標（BLSノード数の代理：Feature=基本特徴8、Enhancement=派生計算24）。\n'
    '中[B]は同一環境での推論時間（中央値、エラーバー=95%位）とメモリ。\n'
    '右[C]は原因分類のMacro-F1（95%CI, n=10セッション/40診断）。\n'
    '本研究は診断段階の規模縮減（図4の主旨）を実現しつつ、速度・メモリを大幅に削減し、精度も維持/向上した。\n'
    '凡例・脚注：先行はBLSのfeature/enhancement nodes（論文値）、本研究は基本特徴8・派生24としてproxy定義。\n'
    '測定条件：Webots R2025a, Ubuntu/WSL2, CPU: Ryzen/Intel型番, 640×480, N=10セッション×4試行、ウォームアップ30フレーム除外、time.perf_counter()。\n'
    '統計量：中央値/95%位（推論時間）、平均（メモリ）、Macro-F1/95%CI（精度）。\n'
    '出典：先行図4（BLS, 著者・年）'
)
fig.text(0.5, -0.08, caption, ha='center', va='top', fontsize=10)

plt.tight_layout(rect=[0, 0.05, 1, 0.95])
fig.savefig(os.path.join(base_dir, 'fig1_composite.png'), dpi=300, bbox_inches='tight')
fig.savefig(os.path.join(base_dir, 'fig1_composite.pdf'), dpi=300, bbox_inches='tight')
print('複合図 fig1_composite.png / fig1_composite.pdf を保存しました。')
