import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.colors import LinearSegmentedColormap

plt.rcParams['font.family'] = ['WenQuanYi Zen Hei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150

# ══════════════════════════════════════════════════════
# 图1：激光器三要素示意
# ══════════════════════════════════════════════════════
fig1, ax = plt.subplots(figsize=(13, 6))
ax.set_xlim(0, 13)
ax.set_ylim(0, 6)
ax.axis('off')
fig1.suptitle('激光器三大要素', fontsize=16, fontweight='bold')

# ── 泵浦源 ──
pump = FancyBboxPatch((0.3, 2.0), 2.4, 2.0,
    boxstyle="round,pad=0.15", facecolor='#fff3cd', edgecolor='#e6a817', lw=2)
ax.add_patch(pump)
ax.text(1.5, 3.35, '① 泵浦源', ha='center', fontsize=13, fontweight='bold', color='#b8860b')
ax.text(1.5, 2.85, '提供能量', ha='center', fontsize=11, color='#555')
ax.text(1.5, 2.45, '（电流 / 闪光灯）', ha='center', fontsize=10, color='#777')

# ── 增益介质 ──
gain = FancyBboxPatch((4.5, 1.5), 4.0, 3.0,
    boxstyle="round,pad=0.15", facecolor='#d4edda', edgecolor='#28a745', lw=2.5)
ax.add_patch(gain)
ax.text(6.5, 3.55, '② 增益介质', ha='center', fontsize=13, fontweight='bold', color='#155724')
ax.text(6.5, 3.0,  '粒子数反转后', ha='center', fontsize=11, color='#555')
ax.text(6.5, 2.5,  '受激辐射放大光', ha='center', fontsize=11, color='#555')
ax.text(6.5, 2.0,  '（半导体 / 气体 / 晶体）', ha='center', fontsize=10, color='#777')

# ── 谐振腔 ──
cav = FancyBboxPatch((10.0, 2.0), 2.4, 2.0,
    boxstyle="round,pad=0.15", facecolor='#cce5ff', edgecolor='#004085', lw=2)
ax.add_patch(cav)
ax.text(11.2, 3.35, '③ 谐振腔', ha='center', fontsize=13, fontweight='bold', color='#004085')
ax.text(11.2, 2.85, '选择波长', ha='center', fontsize=11, color='#555')
ax.text(11.2, 2.45, '（两面反射镜）', ha='center', fontsize=10, color='#777')

# 箭头
ax.annotate('', xy=(4.3, 3.0), xytext=(2.9, 3.0),
    arrowprops=dict(arrowstyle='->', color='#e6a817', lw=2.5))
ax.annotate('', xy=(9.8, 3.0), xytext=(8.7, 3.0),
    arrowprops=dict(arrowstyle='->', color='#28a745', lw=2.5))

ax.text(3.6, 3.3, '能量注入', ha='center', fontsize=10, color='#e6a817')
ax.text(9.25, 3.3, '光放大', ha='center', fontsize=10, color='#28a745')

# 激光输出箭头
ax.annotate('', xy=(12.8, 3.0), xytext=(12.4, 3.0),
    arrowprops=dict(arrowstyle='->', color='red', lw=3))
ax.text(12.95, 3.0, '激光\n输出', ha='left', fontsize=11, color='red', fontweight='bold', va='center')

plt.tight_layout()
fig1.savefig('/home/user/FBG_pro/laser_fig1_elements.png', bbox_inches='tight')
plt.close()

# ══════════════════════════════════════════════════════
# 图2：法布里-珀罗（FP）激光器结构（最常见普通激光器）
# ══════════════════════════════════════════════════════
fig2, ax = plt.subplots(figsize=(14, 7))
ax.set_xlim(-1, 14)
ax.set_ylim(-1.5, 6)
ax.axis('off')
fig2.suptitle('普通半导体激光器结构（法布里-珀罗腔）', fontsize=15, fontweight='bold')

# 主体芯片
chip = mpatches.FancyBboxPatch((1.5, 1.5), 9, 3,
    boxstyle="square,pad=0", facecolor='#e8f4f8', edgecolor='#2c3e50', lw=2.5)
ax.add_patch(chip)

# p型层
p_layer = mpatches.FancyBboxPatch((1.5, 3.5), 9, 1.0,
    boxstyle="square,pad=0", facecolor='#f9c6c6', edgecolor='none')
ax.add_patch(p_layer)
ax.text(6.0, 4.0, 'p型半导体层（空穴）', ha='center', fontsize=11, color='#c0392b')

# 有源区（量子阱）
active = mpatches.FancyBboxPatch((1.5, 2.85), 9, 0.65,
    boxstyle="square,pad=0", facecolor='#a8d8a8', edgecolor='#27ae60', lw=1.5)
ax.add_patch(active)
ax.text(6.0, 3.175, '有源区 / 量子阱（产生激光）', ha='center', fontsize=11,
        color='#155724', fontweight='bold')

# n型层
n_layer = mpatches.FancyBboxPatch((1.5, 1.5), 9, 1.35,
    boxstyle="square,pad=0", facecolor='#c6dcf9', edgecolor='none')
ax.add_patch(n_layer)
ax.text(6.0, 2.15, 'n型半导体层（电子）', ha='center', fontsize=11, color='#1a5276')

# 左侧全反射镜
left_mirror = mpatches.FancyBboxPatch((1.0, 1.5), 0.5, 3.0,
    boxstyle="square,pad=0", facecolor='#2c3e50', edgecolor='#2c3e50')
ax.add_patch(left_mirror)
ax.text(1.25, 0.9, '全反射镜\n(R=100%)', ha='center', fontsize=10, color='#2c3e50')

# 右侧半透射镜
right_mirror = mpatches.FancyBboxPatch((10.5, 1.5), 0.3, 3.0,
    boxstyle="square,pad=0", facecolor='#7f8c8d', edgecolor='#2c3e50', lw=1.5,
    alpha=0.7)
ax.add_patch(right_mirror)
ax.text(10.65, 0.9, '半透射镜\n(R<100%)', ha='center', fontsize=10, color='#2c3e50')

# 电极
ax.plot([3.0, 9.0], [4.5, 4.5], color='#f39c12', lw=5)
ax.text(6.0, 4.85, 'p极（正电极）', ha='center', fontsize=11, color='#e67e22', fontweight='bold')
ax.plot([3.0, 9.0], [1.5, 1.5], color='#7f8c8d', lw=5)
ax.text(6.0, 1.1,  'n极（负电极）', ha='center', fontsize=11, color='#555', fontweight='bold')

# 电流箭头
for x in [4.0, 6.0, 8.0]:
    ax.annotate('', xy=(x, 3.5), xytext=(x, 4.5),
        arrowprops=dict(arrowstyle='->', color='#e74c3c', lw=1.5))
ax.text(9.3, 4.0, '注入\n电流', ha='center', fontsize=10, color='#e74c3c')

# 腔内光子来回振荡
for y_off, color, alpha in [(0.15, '#27ae60', 0.9), (-0.15, '#27ae60', 0.5)]:
    ax.annotate('', xy=(10.3, 3.175+y_off), xytext=(1.7, 3.175+y_off),
        arrowprops=dict(arrowstyle='->', color=color, lw=1.8, alpha=alpha,
                        connectionstyle='arc3,rad=0'))
    ax.annotate('', xy=(1.7, 3.175-y_off), xytext=(10.3, 3.175-y_off),
        arrowprops=dict(arrowstyle='->', color=color, lw=1.8, alpha=alpha,
                        connectionstyle='arc3,rad=0'))
ax.text(6.0, 3.175+0.55, '← 光子在腔内来回振荡放大 →', ha='center', fontsize=10, color='#27ae60')

# 激光输出
ax.annotate('', xy=(13.2, 3.175), xytext=(10.9, 3.175),
    arrowprops=dict(arrowstyle='->', color='red', lw=3.5))
ax.text(13.35, 3.175, '激光输出\n（单色、相干）', ha='left', fontsize=11,
        color='red', fontweight='bold', va='center')

plt.tight_layout()
fig2.savefig('/home/user/FBG_pro/laser_fig2_fp_structure.png', bbox_inches='tight')
plt.close()

# ══════════════════════════════════════════════════════
# 图3：粒子数反转 & 受激辐射原理
# ══════════════════════════════════════════════════════
fig3, axes = plt.subplots(1, 3, figsize=(14, 5))
fig3.suptitle('激光产生的微观原理：受激辐射', fontsize=14, fontweight='bold')

titles = ['① 泵浦：粒子数反转\n（高能级粒子 > 低能级）',
          '② 自发辐射\n（随机方向，触发激光）',
          '③ 受激辐射\n（方向一致，能量放大）']
colors = ['#e74c3c', '#f39c12', '#27ae60']

for ax, title, clr in zip(axes, titles, colors):
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis('off')
    ax.set_title(title, fontsize=11, color=clr, pad=10)

    # 能级线
    ax.plot([1, 9], [2, 2], color='#2c3e50', lw=2.5)
    ax.plot([1, 9], [6, 6], color='#2c3e50', lw=2.5)
    ax.text(0.5, 2, 'E₁\n低能级', ha='center', fontsize=10, va='center', color='#555')
    ax.text(0.5, 6, 'E₂\n高能级', ha='center', fontsize=10, va='center', color='#555')

# 图①：粒子分布
ax = axes[0]
# 高能级：多粒子
for i, x in enumerate([3,4,5,6,7,8]):
    circle = plt.Circle((x, 6), 0.35, color='#e74c3c', zorder=3)
    ax.add_patch(circle)
    ax.text(x, 6, str(i+1), ha='center', va='center', fontsize=9, color='white', fontweight='bold')
# 低能级：少粒子
for x in [3, 5]:
    circle = plt.Circle((x, 2), 0.35, color='#3498db', zorder=3)
    ax.add_patch(circle)
ax.text(5, 4, '泵浦能量 ↑↑\n大量粒子跃迁到高能级', ha='center', fontsize=10, color='#e74c3c')

# 图②：自发辐射
ax = axes[1]
for x in [3,5,7]:
    circle = plt.Circle((x, 6), 0.35, color='#e74c3c', zorder=3)
    ax.add_patch(circle)
# 一个粒子自发跃迁
ax.annotate('', xy=(5, 2.5), xytext=(5, 5.5),
    arrowprops=dict(arrowstyle='->', color='#f39c12', lw=2))
circle = plt.Circle((5, 2), 0.35, color='#3498db', zorder=3)
ax.add_patch(circle)
# 随机方向光子
for angle in [30, 80, 140, 200]:
    rad = np.radians(angle)
    ax.annotate('', xy=(5+1.5*np.cos(rad), 4+1.5*np.sin(rad)), xytext=(5, 4),
        arrowprops=dict(arrowstyle='->', color='#f39c12', lw=1.5, alpha=0.8))
ax.text(5, 0.7, '随机方向光子', ha='center', fontsize=10, color='#f39c12')

# 图③：受激辐射
ax = axes[2]
for x in [3, 5, 7]:
    circle = plt.Circle((x, 6), 0.35, color='#e74c3c', zorder=3)
    ax.add_patch(circle)
# 入射光子触发
ax.annotate('', xy=(4.5, 4), xytext=(1.5, 4),
    arrowprops=dict(arrowstyle='->', color='#27ae60', lw=2.5))
ax.text(3.0, 4.5, '入射光子', ha='center', fontsize=10, color='#27ae60')
# 两个粒子同时跃迁
for x in [5, 7]:
    ax.annotate('', xy=(x, 2.5), xytext=(x, 5.5),
        arrowprops=dict(arrowstyle='->', color='#2ecc71', lw=2))
    circle = plt.Circle((x, 2), 0.35, color='#3498db', zorder=3)
    ax.add_patch(circle)
# 3个同向光子输出
for xi in [5, 7, 8.5]:
    ax.annotate('', xy=(xi+1.2, 4), xytext=(xi, 4),
        arrowprops=dict(arrowstyle='->', color='#27ae60', lw=2.5))
ax.text(7.5, 3.3, '方向相同\n能量增强', ha='center', fontsize=10, color='#27ae60', fontweight='bold')

plt.tight_layout()
fig3.savefig('/home/user/FBG_pro/laser_fig3_stimulated_emission.png', bbox_inches='tight')
plt.close()

# ══════════════════════════════════════════════════════
# 图4：谐振腔选模原理
# ══════════════════════════════════════════════════════
fig4, axes = plt.subplots(2, 1, figsize=(12, 7))
fig4.suptitle('谐振腔：选择波长（纵模）', fontsize=14, fontweight='bold')

L = 1.0   # 腔长（归一化）
x = np.linspace(0, L, 1000)

# 上图：腔内驻波
ax = axes[0]
ax.set_title('腔内驻波：只有整数倍半波长才能稳定存在', fontsize=12)
colors_mode = ['#e74c3c', '#3498db', '#27ae60']
for m, clr in zip([3, 4, 5], colors_mode):
    y = np.sin(m * np.pi * x / L)
    ax.plot(x, y + m*2.5 - 5, color=clr, lw=2,
            label=f'm={m}，波长 λ={2*L/m:.2f}L')
    # 镜面线
    ax.axvline(0,   color='#2c3e50', lw=4)
    ax.axvline(L,   color='#7f8c8d', lw=3, ls='--')

ax.set_xlim(-0.05, 1.15)
ax.set_xticks([0, L])
ax.set_xticklabels(['全反射镜', '半透射镜'], fontsize=11)
ax.set_yticks([])
ax.legend(loc='upper right', fontsize=10)
ax.set_ylabel('光场分布', fontsize=11)
ax.text(0.5, -2.8,
    '条件：腔长 L = m × λ/2  （m为正整数）\n只有满足此条件的波长才能在腔内共振放大',
    ha='center', fontsize=11,
    bbox=dict(facecolor='#fff9c4', edgecolor='gray', boxstyle='round,pad=0.4'))
ax.set_ylim(-4, 8)

# 下图：增益谱 vs 纵模
ax = axes[1]
ax.set_title('增益曲线与纵模：只有增益高于阈值的纵模才能振荡出光', fontsize=12)
lam = np.linspace(780, 830, 1000)
# 增益曲线（高斯型）
gain = 1.2 * np.exp(-((lam - 805)**2) / (2*8**2))
ax.fill_between(lam, gain, alpha=0.15, color='#27ae60')
ax.plot(lam, gain, color='#27ae60', lw=2, label='增益曲线')
# 阈值线
ax.axhline(0.7, color='gray', lw=1.5, ls='--', label='振荡阈值')
ax.text(832, 0.72, '阈值', fontsize=10, color='gray')
# 纵模（离散竖线）
mode_lam = np.arange(784, 828, 3.1)
for lm in mode_lam:
    g = 1.2 * np.exp(-((lm - 805)**2) / (2*8**2))
    clr = 'red' if g > 0.7 else '#aaa'
    ax.vlines(lm, 0, g, color=clr, lw=2.5, alpha=0.85)
# 标注
ax.scatter([805], [1.2], color='red', s=80, zorder=5)
ax.annotate('主模（最强）', xy=(805, 1.2), xytext=(812, 1.15),
    arrowprops=dict(arrowstyle='->', color='red'), fontsize=10, color='red')
ax.set_xlabel('波长 (nm)', fontsize=11)
ax.set_ylabel('增益', fontsize=11)
ax.legend(fontsize=10)
ax.set_xlim(778, 840)
ax.set_ylim(0, 1.4)

plt.tight_layout()
fig4.savefig('/home/user/FBG_pro/laser_fig4_resonator.png', bbox_inches='tight')
plt.close()

print("All laser figures saved.")
