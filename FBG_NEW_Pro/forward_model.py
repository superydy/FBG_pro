"""
物理正向模型
================
核心思想：形状传感的"正向"过程是完全已知的解析公式，不需要 ANSYS。

流程：
    随机曲率 (k1, k2)  --Bishop标架积分-->  3D真值曲线 P(s)
                       \--欧拉-伯努利梁公式-->  各芯应变 ε_i  --FBG公式-->  Δλ_i (+噪声)

这样可以零成本生成任意多 (Δλ, 3D曲线) 训练样本。

约定（forward 与 inverse 全程一致）：
    - 材料坐标系采用 Bishop 标架 {M1, M2, T}（平行移动，无扭转）
    - 曲率矢量 κ_vec = k1·M1 + k2·M2，位于横截面内
    - 第 i 芯横截面坐标 p_i = r_i·(cosθ_i·M1 + sinθ_i·M2)
    - 弯曲应变 ε_bend_i = κ_vec · p_i = r_i·(k1·cosθ_i + k2·sinθ_i)
"""
import numpy as np
import config as C


# ─────────────────────────────────────────────────────────────
# 1. 随机曲率生成：带限随机（低频傅里叶叠加），保证曲线光滑
# ─────────────────────────────────────────────────────────────
def random_curvature_profile(n_points, ds, rng, n_modes=C.N_MODES,
                             kappa_max=C.KAPPA_MAX):
    """生成一条光滑随机曲率剖面 k1(s), k2(s)。返回 shape (n_points,) 两个数组。"""
    L = (n_points - 1) * ds
    s = np.arange(n_points) * ds

    def one_component():
        comp = np.zeros(n_points)
        for m in range(1, n_modes + 1):
            amp = rng.normal(0, 1.0) / m          # 高频幅值衰减 -> 光滑
            phase = rng.uniform(0, 2 * np.pi)
            comp += amp * np.sin(2 * np.pi * m * s / L + phase)
        return comp

    k1 = one_component()
    k2 = one_component()

    # 归一化到合理曲率范围
    scale = rng.uniform(0.2, 1.0) * kappa_max
    norm = np.max(np.sqrt(k1**2 + k2**2)) + 1e-9
    k1 *= scale / norm
    k2 *= scale / norm
    return k1, k2


# ─────────────────────────────────────────────────────────────
# 2. Bishop 标架积分：曲率 -> 3D 真值曲线
#    ODE:  T' = k1·M1 + k2·M2
#          M1' = -k1·T
#          M2' = -k2·T
#    P(s) = ∫ T ds
# ─────────────────────────────────────────────────────────────
def integrate_bishop(k1, k2, ds):
    """由曲率分量积分出 3D 曲线。返回 positions (n,3), T,M1,M2 (n,3)。"""
    n = len(k1)
    T  = np.zeros((n, 3)); M1 = np.zeros((n, 3)); M2 = np.zeros((n, 3))
    P  = np.zeros((n, 3))
    # 初始标架：沿 z 轴出发
    T[0]  = np.array([0.0, 0.0, 1.0])
    M1[0] = np.array([1.0, 0.0, 0.0])
    M2[0] = np.array([0.0, 1.0, 0.0])

    for i in range(n - 1):
        # 四阶精度不必要，弧长步长很小，用半隐式 Euler + 重正交化即可
        dT  = (k1[i] * M1[i] + k2[i] * M2[i]) * ds
        dM1 = (-k1[i] * T[i]) * ds
        dM2 = (-k2[i] * T[i]) * ds
        T[i+1]  = T[i]  + dT
        M1[i+1] = M1[i] + dM1
        M2[i+1] = M2[i] + dM2
        # 重正交归一化（防漂移）
        T[i+1]  /= np.linalg.norm(T[i+1]) + 1e-12
        M1[i+1] -= np.dot(M1[i+1], T[i+1]) * T[i+1]
        M1[i+1] /= np.linalg.norm(M1[i+1]) + 1e-12
        M2[i+1]  = np.cross(T[i+1], M1[i+1])
        # 位置积分（梯形）
        P[i+1] = P[i] + 0.5 * (T[i] + T[i+1]) * ds
    return P, T, M1, M2


# ─────────────────────────────────────────────────────────────
# 3. 曲率 -> 各芯应变 -> Δλ
# ─────────────────────────────────────────────────────────────
def strains_from_curvature(k1, k2, eps_axial, delta_T,
                           core_radii=C.CORE_RADII, core_angles=C.CORE_ANGLES):
    """计算每个芯的总应变（弯曲 + 轴向）。返回 (n, n_cores)。"""
    n = len(k1)
    n_cores = len(core_radii)
    eps = np.zeros((n, n_cores))
    for c in range(n_cores):
        r, th = core_radii[c], core_angles[c]
        eps_bend = r * (k1 * np.cos(th) + k2 * np.sin(th))
        eps[:, c] = eps_bend + eps_axial            # 轴向项各芯相同
    return eps


def wavelength_from_strain(eps, delta_T,
                           lambda0=C.LAMBDA0, p_e=C.P_E, k_T=C.K_T):
    """FBG 公式：Δλ = λ0·(1-pe)·ε + K_T·ΔT。返回 (n, n_cores)。"""
    dlam_strain = lambda0 * (1 - p_e) * eps
    dlam_temp   = k_T * delta_T                      # 温度对所有芯一致
    return dlam_strain + dlam_temp


def add_noise(dlam, rng, sigma=C.NOISE_DLAMBDA):
    """加入解调仪波长噪声。"""
    return dlam + rng.normal(0, sigma, size=dlam.shape)


# ─────────────────────────────────────────────────────────────
# 4. 一条完整样本
# ─────────────────────────────────────────────────────────────
def make_one_sample(rng, n_points=C.N_POINTS, ds=C.DS, with_noise=True):
    """返回 dict: dlambda(n,4 含噪), dlambda_clean, positions(n,3),
    k1,k2, delta_T, eps_axial。"""
    k1, k2 = random_curvature_profile(n_points, ds, rng)
    # 轴向应变（拉伸）小幅随机 + 温度
    eps_axial = rng.normal(0, 50e-6)                 # ~50 με
    delta_T   = rng.uniform(-C.TEMP_RANGE, C.TEMP_RANGE)

    P, T, M1, M2 = integrate_bishop(k1, k2, ds)
    eps = strains_from_curvature(k1, k2, eps_axial, delta_T)
    dlam_clean = wavelength_from_strain(eps, delta_T)
    dlam = add_noise(dlam_clean, rng) if with_noise else dlam_clean.copy()

    return dict(dlambda=dlam, dlambda_clean=dlam_clean, positions=P,
                k1=k1, k2=k2, delta_T=delta_T, eps_axial=eps_axial)


if __name__ == "__main__":
    # 自检：跑一条样本，打印关键量
    rng = np.random.default_rng(C.SEED)
    s = make_one_sample(rng)
    print("Δλ shape      :", s["dlambda"].shape)
    print("positions shape:", s["positions"].shape)
    print("末端坐标 (m)   :", np.round(s["positions"][-1], 4))
    print("曲率范围 (1/m) :", np.round([s["k1"].min(), s["k1"].max()], 2))
    print("Δλ 范围 (nm)   :", np.round([s["dlambda"].min(), s["dlambda"].max()], 4))
    print("总弧长 (m)     :", round(C.DS * (C.N_POINTS - 1), 3))
