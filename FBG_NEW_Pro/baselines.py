"""
传统重建算法（baseline，论文里的对比对象）
============================================
传统流程：Δλ -> 应变 -> 加权最小二乘解曲率 (k1,k2) -> 逐段积分 -> 3D坐标

逐段积分是传统方法的"原罪"：误差一段一段往后传，弧长越长端点误差越大。
本文的神经网络方法就是要绕开这个累积机制。

提供三个 baseline：
    1. recover_curvature_lsq : 四芯 -> (k1,k2)，加权最小二乘 + 中心芯温度解耦
    2. reconstruct_bishop     : Bishop 标架逐段积分（较好的经典法）
    3. reconstruct_frenet     : Frenet-Serret 逐段积分（经典法，曲率->0 处有奇异）
"""
import numpy as np
import config as C


# ─────────────────────────────────────────────────────────────
# 1. 四芯 Δλ -> 曲率 (k1, k2)  [加权最小二乘 + 温度解耦]
# ─────────────────────────────────────────────────────────────
def recover_curvature_lsq(dlam, lambda0=C.LAMBDA0, p_e=C.P_E, k_T=C.K_T,
                          core_radii=C.CORE_RADII, core_angles=C.CORE_ANGLES,
                          weights=None):
    """
    dlam: (n, 4)   ->   返回 k1_hat, k2_hat: (n,)
    用中心芯(core 0)估计 轴向+温度 公共项，外芯减去公共项得纯弯曲应变，
    再对外芯做最小二乘解 (k1,k2)。
    """
    n = dlam.shape[0]
    # Step 1: 总应变估计（先不分离温度）
    #   中心芯只含 轴向+温度 公共项：dlam_c = λ0(1-pe)·eps_axial + K_T·ΔT
    #   外芯含 弯曲 + 公共项。用"外芯 - 中心芯"消去公共项即可得弯曲贡献。
    dlam_common = dlam[:, 0:1]                       # (n,1) 中心芯
    dlam_bend = dlam[:, 1:] - dlam_common            # (n, 3) 仅弯曲（公共项抵消）
    eps_bend = dlam_bend / (lambda0 * (1 - p_e))     # (n,3)

    # Step 2: 最小二乘  eps_bend_j = r·(k1·cosθ_j + k2·sinθ_j)
    outer_idx = [1, 2, 3]
    A = np.stack([core_radii[outer_idx] * np.cos(core_angles[outer_idx]),
                  core_radii[outer_idx] * np.sin(core_angles[outer_idx])], axis=1)  # (3,2)
    if weights is not None:
        W = np.diag(weights)
        ATA = A.T @ W @ A
        ATb = A.T @ W @ eps_bend.T                   # (2, n)
    else:
        ATA = A.T @ A
        ATb = A.T @ eps_bend.T                       # (2, n)
    sol = np.linalg.solve(ATA, ATb)                  # (2, n)
    k1_hat, k2_hat = sol[0], sol[1]
    return k1_hat, k2_hat


# ─────────────────────────────────────────────────────────────
# 2. Bishop 逐段积分重建
# ─────────────────────────────────────────────────────────────
def reconstruct_bishop(k1, k2, ds=C.DS):
    n = len(k1)
    T  = np.zeros((n, 3)); M1 = np.zeros((n, 3)); M2 = np.zeros((n, 3))
    P  = np.zeros((n, 3))
    T[0]  = [0, 0, 1]; M1[0] = [1, 0, 0]; M2[0] = [0, 1, 0]
    for i in range(n - 1):
        T[i+1]  = T[i]  + (k1[i] * M1[i] + k2[i] * M2[i]) * ds
        M1[i+1] = M1[i] + (-k1[i] * T[i]) * ds
        T[i+1] /= np.linalg.norm(T[i+1]) + 1e-12
        M1[i+1] -= np.dot(M1[i+1], T[i+1]) * T[i+1]
        M1[i+1] /= np.linalg.norm(M1[i+1]) + 1e-12
        M2[i+1] = np.cross(T[i+1], M1[i+1])
        P[i+1] = P[i] + 0.5 * (T[i] + T[i+1]) * ds
    return P


# ─────────────────────────────────────────────────────────────
# 3. Frenet-Serret 逐段积分重建（经典法，曲率小处不稳定）
#    κ = sqrt(k1²+k2²),  φ = atan2(k2,k1),  τ ≈ dφ/ds
# ─────────────────────────────────────────────────────────────
def reconstruct_frenet(k1, k2, ds=C.DS):
    n = len(k1)
    kappa = np.sqrt(k1**2 + k2**2)
    phi   = np.unwrap(np.arctan2(k2, k1))
    tau   = np.gradient(phi, ds)                     # 挠率近似

    T = np.zeros((n, 3)); N = np.zeros((n, 3)); B = np.zeros((n, 3))
    P = np.zeros((n, 3))
    T[0] = [0, 0, 1]; N[0] = [1, 0, 0]; B[0] = np.cross(T[0], N[0])
    for i in range(n - 1):
        # Frenet-Serret 方程
        T[i+1] = T[i] + kappa[i] * N[i] * ds
        N[i+1] = N[i] + (-kappa[i] * T[i] + tau[i] * B[i]) * ds
        B[i+1] = B[i] + (-tau[i] * N[i]) * ds
        # 重正交
        T[i+1] /= np.linalg.norm(T[i+1]) + 1e-12
        N[i+1] -= np.dot(N[i+1], T[i+1]) * T[i+1]
        N[i+1] /= np.linalg.norm(N[i+1]) + 1e-12
        B[i+1] = np.cross(T[i+1], N[i+1])
        P[i+1] = P[i] + 0.5 * (T[i] + T[i+1]) * ds
    return P


# ─────────────────────────────────────────────────────────────
# 误差度量
# ─────────────────────────────────────────────────────────────
def pointwise_error(P_pred, P_true):
    """每点欧氏误差 (n,)。"""
    return np.linalg.norm(P_pred - P_true, axis=1)

def endpoint_error(P_pred, P_true):
    return float(np.linalg.norm(P_pred[-1] - P_true[-1]))


if __name__ == "__main__":
    # 自检：噪声下 baseline 的端点误差应随弧长增大
    import forward_model as fm
    rng = np.random.default_rng(1)
    s = fm.make_one_sample(rng)
    k1h, k2h = recover_curvature_lsq(s["dlambda"])
    P_bishop = reconstruct_bishop(k1h, k2h)
    P_frenet = reconstruct_frenet(k1h, k2h)
    P_true   = s["positions"]
    print("Bishop 端点误差 (mm):", round(endpoint_error(P_bishop, P_true) * 1000, 2))
    print("Frenet 端点误差 (mm):", round(endpoint_error(P_frenet, P_true) * 1000, 2))
    print("曲率反演 RMSE (1/m) :",
          round(float(np.sqrt(np.mean((k1h - s["k1"])**2 + (k2h - s["k2"])**2))), 3))
