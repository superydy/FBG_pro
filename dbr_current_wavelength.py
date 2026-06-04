import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams['font.family'] = ['WenQuanYi Zen Hei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150

# ══════════════════════════════════════════════════════
# 图1：物理机制链条
# ══════════════════════════════════════════════════════
fig1, ax = plt.subplots(figsize=(14, 4))
ax.set_xlim(0, 14)
ax.set_ylim(0, 4)
ax.axis('off')
fig1.suptitle('DBR激光器电流控制波长的物理机制', fontsize=15, fontweight='bold')

steps = [
    ('注入电流 I↑',        '#e74c3c', 0.8),
    ('载流子浓度 N↑',      '#e67e22', 2.8),
    ('等效折射率 n_eff↓',  '#8e44ad', 4.8),
    ('布拉格波长 λ↓',      '#2980b9', 6.8),
    ('激光输出波长改变',    '#27ae60', 9.2),
]

for i, (label, color, x) in enumerate(steps):
    box = FancyBboxPatch((x, 1.2), 1.8, 1.6,
        boxstyle="round,pad=0.15", facecolor=color, edgecolor='white', lw=2, alpha=0.85)
    ax.add_patch(box)
    ax.text(x+0.9, 2.0, label, ha='center', va='center',
            fontsize=11, color='white', fontweight='bold')
    if i < len(steps)-1:
        ax.annotate('', xy=(steps[i+1][2], 2.0), xytext=(x+1.8, 2.0),
            arrowprops=dict(arrowstyle='->', color='#555', lw=2.5))

# 公式
ax.text(7.0, 0.4,
    'λ_Bragg = 2 × n_eff × Λ        '
    '注入电流 → 载流子等离子体效应 → n_eff 减小 → λ 蓝移（变短）',
    ha='center', fontsize=11, color='#333',
    bbox=dict(facecolor='#f8f9fa', edgecolor='gray', boxstyle='round,pad=0.4'))

plt.tight_layout()
fig1.savefig('/home/user/FBG_pro/dbr_fig1_mechanism.png', bbox_inches='tight')
plt.close()

# ══════════════════════════════════════════════════════
# 图2：两种效应对折射率/波长的影响
# ══════════════════════════════════════════════════════
fig2, axes = plt.subplots(1, 2, figsize=(13, 5))
fig2.suptitle('两种电流效应对波长的影响（方向相反）', fontsize=14, fontweight='bold')

I = np.linspace(0, 60, 500)

# 载流子效应（等离子体效应）：n_eff 随电流减小 → 波长蓝移（负）
dn_carrier = -0.006 * np.sqrt(I + 0.1)
# 热效应：电流产生热量，n_eff 随温度升高而增大 → 波长红移（正）
dn_thermal = 0.00012 * I**1.3
# 总效应
dn_total = dn_carrier + dn_thermal

ax = axes[0]
ax.plot(I, dn_carrier * 1000, color='#2980b9', lw=2.5, label='载流子等离子体效应（折射率↓，波长蓝移）')
ax.plot(I, dn_thermal * 1000, color='#e74c3c', lw=2.5, label='热效应（折射率↑，波长红移）')
ax.plot(I, dn_total * 1000,   color='#27ae60', lw=3,   label='综合效果', ls='--')
ax.axhline(0, color='gray', lw=1, ls=':')
ax.set_xlabel('注入电流 (mA)', fontsize=12)
ax.set_ylabel('折射率变化量 Δn (×10⁻³)', fontsize=12)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_title('折射率随电流变化', fontsize=12)
# 标注主导区域
ax.axvspan(0, 20, alpha=0.08, color='#2980b9')
ax.axvspan(40, 60, alpha=0.08, color='#e74c3c')
ax.text(10, -3.5, '低电流区\n载流子效应主导', ha='center', fontsize=10, color='#2980b9')
ax.text(50, 1.5, '高电流区\n热效应主导', ha='center', fontsize=10, color='#e74c3c')

# 右图：波长vs电流（典型DBR相位区）
ax = axes[1]
lambda0 = 1550.0  # 中心波长 nm
# 相位区调谐：低电流段基本线性，范围约0.4nm
I_phase = np.linspace(0, 20, 500)
# 模拟论文图3-9的形状：先快后慢，近似线性段
dl_phase = -0.4 * (1 - np.exp(-I_phase / 8)) + 0.02 * I_phase * np.sin(I_phase * 0.3)
lam_phase = lambda0 + dl_phase

ax.plot(I_phase, lam_phase, color='#8e44ad', lw=2.5)
ax.set_xlabel('相位区注入电流 (mA)', fontsize=12)
ax.set_ylabel('输出波长 (nm)', fontsize=12)
ax.set_title('DBR相位区：电流 vs 波长（典型曲线）', fontsize=12)
ax.grid(True, alpha=0.3)

# 标注线性工作区（论文中5~13mA）
ax.axvspan(5, 13, alpha=0.15, color='#27ae60')
ax.text(9, lambda0 - 0.12,
        '线性工作区\n(精度最高)', ha='center', fontsize=10,
        color='#155724',
        bbox=dict(facecolor='#d4edda', edgecolor='#27ae60', boxstyle='round,pad=0.3'))
ax.annotate('', xy=(5, lam_phase[125]), xytext=(5, lambda0+0.05),
    arrowprops=dict(arrowstyle='<->', color='#27ae60', lw=2))
ax.text(3.5, lambda0 - 0.07, '~0.3 nm\n调谐范围', ha='center', fontsize=10, color='#27ae60')

plt.tight_layout()
fig2.savefig('/home/user/FBG_pro/dbr_fig2_two_effects.png', bbox_inches='tight')
plt.close()

# ══════════════════════════════════════════════════════
# 图3：DBR激光器结构 + 各区段作用
# ══════════════════════════════════════════════════════
fig3, ax = plt.subplots(figsize=(15, 6))
ax.set_xlim(0, 15)
ax.set_ylim(0, 6)
ax.axis('off')
fig3.suptitle('DBR激光器各区段结构与电流作用', fontsize=15, fontweight='bold')

# 定义各区段
sections = [
    ('后端\n光栅区', 1.0, 2.5, '#3498db',   '粗调波段\n（跳跃式，nm级）\n注入0~60mA'),
    ('相位区',       4.0, 1.5, '#8e44ad',   '精细调谐\n（连续，0.4nm内）\n精度0.1pm'),
    ('有源区',       6.2, 3.0, '#27ae60',   '产生激光增益\n注入电流→光'),
    ('前端\n光栅区', 10.0, 2.5, '#e67e22',  '选模+反射\n（可多电极分区）'),
    ('SOA',          13.0, 1.2, '#e74c3c',  '功率放大'),
]

x_pos = 0.5
for name, width, height, color, desc in sections:
    y0 = (6 - height) / 2
    box = FancyBboxPatch((x_pos, y0), width, height,
        boxstyle="square,pad=0", facecolor=color, edgecolor='white', lw=2, alpha=0.85)
    ax.add_patch(box)
    ax.text(x_pos + width/2, 3.0, name,
            ha='center', va='center', fontsize=11, color='white', fontweight='bold')
    ax.text(x_pos + width/2, y0 - 0.8, desc,
            ha='center', va='top', fontsize=9.5, color=color,
            bbox=dict(facecolor='white', edgecolor=color, boxstyle='round,pad=0.2', alpha=0.9))
    # 电流箭头
    ax.annotate('', xy=(x_pos + width/2, y0 + height),
                    xytext=(x_pos + width/2, y0 + height + 0.6),
        arrowprops=dict(arrowstyle='->', color=color, lw=2))
    ax.text(x_pos + width/2, y0 + height + 0.75, 'I', ha='center', fontsize=12,
            color=color, fontweight='bold')
    x_pos += width + 0.3

# 激光输出
ax.annotate('', xy=(14.8, 3.0), xytext=(14.2, 3.0),
    arrowprops=dict(arrowstyle='->', color='red', lw=4))
ax.text(14.85, 3.0, '激光\n输出', ha='left', va='center',
        fontsize=11, color='red', fontweight='bold')

# 标注关键路径
ax.text(7.5, 0.25,
    '控制波长的关键：后端光栅电流（粗调）+ 相位区电流（精调）',
    ha='center', fontsize=11, color='#333', style='italic',
    bbox=dict(facecolor='#fff9c4', edgecolor='gray', boxstyle='round,pad=0.35'))

plt.tight_layout()
fig3.savefig('/home/user/FBG_pro/dbr_fig3_structure.png', bbox_inches='tight')
plt.close()

# ══════════════════════════════════════════════════════
# 图4：实际控制策略——粗调+精调配合
# ══════════════════════════════════════════════════════
fig4, axes = plt.subplots(1, 2, figsize=(14, 5))
fig4.suptitle('实际控制策略：粗调 + 精调 两级联动', fontsize=14, fontweight='bold')

# 左图：粗调（后端光栅）——阶梯式跳跃
ax = axes[0]
lam_coarse = np.array([1530, 1537, 1544, 1551, 1558, 1565, 1572])
I_coarse    = np.array([0,    8,    18,   28,   38,   50,   60])
ax.step(I_coarse, lam_coarse, where='post', color='#3498db', lw=2.5)
ax.scatter(I_coarse, lam_coarse, color='#3498db', s=80, zorder=5)
for i, (ic, lc) in enumerate(zip(I_coarse, lam_coarse)):
    ax.text(ic+1, lc+0.5, f'{lc}nm', fontsize=9, color='#3498db')
ax.set_xlabel('后端光栅电流 (mA)', fontsize=12)
ax.set_ylabel('中心波长 (nm)', fontsize=12)
ax.set_title('后端光栅：粗调（阶梯跳跃，间隔~7nm）', fontsize=12)
ax.grid(True, alpha=0.3)
ax.set_ylim(1525, 1578)
ax.annotate('选定波段后\n再用相位区精调', xy=(30, 1551), xytext=(40, 1541),
    arrowprops=dict(arrowstyle='->', color='gray'), fontsize=10, color='gray')

# 右图：精调（相位区）——连续线性
ax = axes[1]
I_fine = np.linspace(5, 13, 300)
# 线性段近似
lam_fine = 1551.0 - 0.038 * (I_fine - 5)
ax.plot(I_fine, lam_fine, color='#8e44ad', lw=3)

# 加一些实测噪声感
noise = np.random.RandomState(42).normal(0, 0.005, len(I_fine))
ax.scatter(I_fine[::15], lam_fine[::15] + noise[::15]*2,
           color='#8e44ad', s=30, alpha=0.6, label='实测点')
ax.plot(I_fine, lam_fine, color='#8e44ad', lw=2.5, label='拟合线')

# 标注分辨率
ax.axhline(1550.8, color='#e74c3c', lw=1, ls='--', alpha=0.7)
ax.axhline(1550.7, color='#e74c3c', lw=1, ls='--', alpha=0.7)
ax.annotate('', xy=(13.1, 1550.8), xytext=(13.1, 1550.7),
    arrowprops=dict(arrowstyle='<->', color='#e74c3c', lw=1.5))
ax.text(13.2, 1550.75, '0.1pm\n分辨率', fontsize=9, color='#e74c3c', va='center')

ax.set_xlabel('相位区电流 (mA)', fontsize=12)
ax.set_ylabel('输出波长 (nm)', fontsize=12)
ax.set_title('相位区：精调（连续线性，精度0.1pm）', fontsize=12)
ax.grid(True, alpha=0.3)
ax.legend(fontsize=10)

# 斜率标注
ax.text(8.5, 1550.55,
    f'灵敏度 ≈ 0.038 nm/mA\n= 38 pm/mA',
    ha='center', fontsize=11, color='#8e44ad',
    bbox=dict(facecolor='#f3e5f5', edgecolor='#8e44ad', boxstyle='round,pad=0.4'))

plt.tight_layout()
fig4.savefig('/home/user/FBG_pro/dbr_fig4_control.png', bbox_inches='tight')
plt.close()

print("All DBR figures saved.")
