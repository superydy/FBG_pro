import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.signal import chirp, spectrogram

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['figure.dpi'] = 150

# ── 参数 ──────────────────────────────────────────────
fs = 10000       # 采样率 Hz
T  = 1.0         # 持续时间 s
t  = np.linspace(0, T, int(fs * T), endpoint=False)
f0, f1 = 10, 500  # 起始/终止频率 Hz

up   = chirp(t, f0=f0, f1=f1, t1=T, method='linear', phi=90)
down = chirp(t, f0=f1, f1=f0, t1=T, method='linear', phi=90)

# ══════════════════════════════════════════════════════
# 图1：时域波形对比
# ══════════════════════════════════════════════════════
fig1, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
fig1.suptitle('Chirp Signal — Time-Domain Waveform\n啁啾信号时域波形', fontsize=14, fontweight='bold')

for ax, sig, label, color in zip(
        axes,
        [up, down],
        ['Up-Chirp  (上啁啾: 10 Hz → 500 Hz)', 'Down-Chirp  (下啁啾: 500 Hz → 10 Hz)'],
        ['steelblue', 'tomato']):
    ax.plot(t[:3000], sig[:3000], color=color, lw=0.8)
    ax.set_ylabel('Amplitude', fontsize=11)
    ax.set_title(label, fontsize=12)
    ax.axhline(0, color='gray', lw=0.5, ls='--')
    ax.set_ylim(-1.4, 1.4)
    ax.grid(True, alpha=0.3)

axes[-1].set_xlabel('Time (s)', fontsize=11)
plt.tight_layout()
fig1.savefig('/home/user/FBG_pro/fig1_waveform.png', bbox_inches='tight')
plt.close()

# ══════════════════════════════════════════════════════
# 图2：瞬时频率 vs 时间
# ══════════════════════════════════════════════════════
fig2, ax = plt.subplots(figsize=(10, 4))
fig2.suptitle('Instantaneous Frequency vs Time\n瞬时频率随时间变化', fontsize=14, fontweight='bold')

freq_up   = f0 + (f1 - f0) * t / T
freq_down = f1 - (f1 - f0) * t / T

ax.plot(t, freq_up,   color='steelblue', lw=2.5, label='Up-Chirp（上啁啾）')
ax.plot(t, freq_down, color='tomato',    lw=2.5, label='Down-Chirp（下啁啾）', ls='--')
ax.fill_between(t, freq_up, freq_down, alpha=0.08, color='purple')
ax.set_xlabel('Time (s)', fontsize=12)
ax.set_ylabel('Frequency (Hz)', fontsize=12)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_xlim(0, T)
plt.tight_layout()
fig2.savefig('/home/user/FBG_pro/fig2_freq_vs_time.png', bbox_inches='tight')
plt.close()

# ══════════════════════════════════════════════════════
# 图3：时频谱图（Spectrogram）
# ══════════════════════════════════════════════════════
fig3, axes = plt.subplots(1, 2, figsize=(13, 5))
fig3.suptitle('Spectrogram (Time-Frequency Analysis)\n时频谱图', fontsize=14, fontweight='bold')

for ax, sig, label, cmap in zip(
        axes,
        [up, down],
        ['Up-Chirp', 'Down-Chirp'],
        ['Blues', 'Reds']):
    f_arr, t_arr, Sxx = spectrogram(sig, fs=fs, nperseg=256, noverlap=200)
    im = ax.pcolormesh(t_arr, f_arr, 10*np.log10(Sxx + 1e-12),
                       cmap=cmap, shading='gouraud', vmin=-60)
    ax.set_ylim(0, 600)
    ax.set_xlabel('Time (s)', fontsize=11)
    ax.set_ylabel('Frequency (Hz)', fontsize=11)
    ax.set_title(label, fontsize=12)
    fig3.colorbar(im, ax=ax, label='Power (dB)')

plt.tight_layout()
fig3.savefig('/home/user/FBG_pro/fig3_spectrogram.png', bbox_inches='tight')
plt.close()

# ══════════════════════════════════════════════════════
# 图4：脉冲压缩（Pulse Compression）
# ══════════════════════════════════════════════════════
# 发射宽脉冲啁啾，接收后用匹配滤波器压缩
tx = chirp(t, f0=f0, f1=f1, t1=T, method='linear', phi=90)
tx_windowed = tx * np.hanning(len(t))

# 匹配滤波 = 发射信号的时间反转共轭
mf = np.conj(tx_windowed[::-1])
compressed = np.fft.ifft(np.fft.fft(tx_windowed) * np.fft.fft(mf, n=len(tx_windowed)))
compressed = np.abs(compressed)
compressed /= compressed.max()

fig4, axes = plt.subplots(2, 1, figsize=(11, 6))
fig4.suptitle('Pulse Compression via Matched Filter\n脉冲压缩（匹配滤波）', fontsize=14, fontweight='bold')

axes[0].plot(t, tx_windowed, color='steelblue', lw=0.8)
axes[0].set_title('Transmitted Chirp Pulse（发射啁啾宽脉冲）', fontsize=12)
axes[0].set_ylabel('Amplitude', fontsize=11)
axes[0].grid(True, alpha=0.3)

# 找峰值位置居中显示
pk = np.argmax(compressed)
w  = 500
sl = slice(max(0, pk-w), min(len(t), pk+w))
axes[1].plot(t[sl] - t[pk], compressed[sl], color='tomato', lw=1.5)
axes[1].set_title('Compressed Pulse after Matched Filter（压缩后窄脉冲）', fontsize=12)
axes[1].set_ylabel('Normalized Amplitude', fontsize=11)
axes[1].set_xlabel('Relative Time (s)', fontsize=11)
axes[1].grid(True, alpha=0.3)
axes[1].annotate('主峰（极窄！）', xy=(0, 1.0), xytext=(0.01, 0.85),
                 arrowprops=dict(arrowstyle='->', color='black'), fontsize=11)

plt.tight_layout()
fig4.savefig('/home/user/FBG_pro/fig4_pulse_compression.png', bbox_inches='tight')
plt.close()

# ══════════════════════════════════════════════════════
# 图5：啁啾 FBG 示意图（概念图）
# ══════════════════════════════════════════════════════
fig5, ax = plt.subplots(figsize=(12, 5))
fig5.suptitle('Chirped FBG Concept\n啁啾光纤布拉格光栅（Chirped FBG）示意', fontsize=14, fontweight='bold')
ax.set_xlim(0, 10)
ax.set_ylim(-2.5, 3.5)
ax.axis('off')

# 光纤主体
ax.fill_between([0.5, 9.5], [-0.4, -0.4], [0.4, 0.4], color='#cce5ff', zorder=1)
ax.plot([0.5, 9.5], [-0.4, -0.4], color='gray', lw=1.5)
ax.plot([0.5, 9.5], [ 0.4,  0.4], color='gray', lw=1.5)
ax.text(5, -0.85, 'Optical Fiber（光纤）', ha='center', fontsize=11, color='gray')

# 光栅条纹：间距从大变小（啁啾）
x_start = 1.5
spacings = np.linspace(0.38, 0.13, 30)
x = x_start
for sp in spacings:
    ax.fill_between([x, x+sp*0.4], [-0.38, -0.38], [0.38, 0.38], color='#2166ac', alpha=0.7, zorder=2)
    x += sp

ax.annotate('', xy=(x_start, 1.5), xytext=(x_start + 2.5, 1.5),
            arrowprops=dict(arrowstyle='<->', color='steelblue', lw=2))
ax.text(x_start + 1.25, 1.8, '较大周期 Λ₁\n→ 较长波长 λ₁', ha='center', fontsize=10, color='steelblue')

ax.annotate('', xy=(x-1.5, 1.5), xytext=(x, 1.5),
            arrowprops=dict(arrowstyle='<->', color='tomato', lw=2))
ax.text(x-0.75, 1.8, '较小周期 Λ₂\n→ 较短波长 λ₂', ha='center', fontsize=10, color='tomato')

ax.annotate('入射宽带光', xy=(0.5, 0), xytext=(-0.5, 0.8),
            arrowprops=dict(arrowstyle='->', color='orange', lw=2),
            fontsize=11, color='orange')
ax.annotate('反射（多波长）', xy=(0.5, 0), xytext=(-0.5, -0.8),
            arrowprops=dict(arrowstyle='<-', color='purple', lw=2),
            fontsize=11, color='purple')
ax.annotate('透射光', xy=(9.5, 0), xytext=(10.2, 0.8),
            arrowprops=dict(arrowstyle='<-', color='green', lw=2),
            fontsize=11, color='green')

ax.text(5, -2.2,
        'Λ₁ > Λ₂  →  λ_Bragg = 2nΛ  →  不同位置反射不同波长',
        ha='center', fontsize=11, style='italic',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='#fff9c4', edgecolor='gray'))

plt.tight_layout()
fig5.savefig('/home/user/FBG_pro/fig5_chirped_fbg.png', bbox_inches='tight')
plt.close()

print("All figures saved.")
