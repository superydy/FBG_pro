"""
对比评估 + 出图
================
对测试集比较各方法的重建误差，产出论文核心结果：
    1. 误差随弧长增长曲线  -> error_vs_arclength.png（最重要的图）
    2. 端点误差/RMSE 统计表 -> 终端打印 + metrics.csv
    3. 单条样本 3D 重建对比 -> sample_reconstruction.png

方法：
    Frenet  : 传统 Frenet-Serret 逐段积分（baseline）
    Bishop  : Bishop 标架逐段积分（较强 baseline）
    Net     : 本文神经网络（需 torch + checkpoint.pt，缺失则自动跳过）

用法：
    python3 evaluate.py        # 有 torch+权重则含网络，否则只比 baseline
"""
import os
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config as C
import baselines as B

plt.rcParams["font.family"] = ["WenQuanYi Zen Hei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 150

HERE = os.path.dirname(__file__)


# ─────────────────────────────────────────────────────────────
# 可选：加载网络（无 torch 或无权重则返回 None）
# ─────────────────────────────────────────────────────────────
def try_load_net():
    ckpt_path = os.path.join(HERE, "checkpoint.pt")
    if not os.path.exists(ckpt_path):
        return None
    try:
        import torch
        from model import build_model
    except Exception:
        return None
    ck = torch.load(ckpt_path, map_location="cpu")
    model = build_model(ck.get("name", "lstm"))
    model.load_state_dict(ck["model"]); model.eval()
    mean, std = ck["mean"], ck["std"]

    def predict(X):                              # X: (m,n,4) numpy
        import torch
        with torch.no_grad():
            xt = (torch.tensor(X, dtype=torch.float32) - mean) / std
            return model(xt).numpy()
    return predict


# ─────────────────────────────────────────────────────────────
# 主评估
# ─────────────────────────────────────────────────────────────
def main(max_samples=None):
    d = np.load(os.path.join(HERE, "data", "test.npz"))
    X, Y = d["X"], d["Y"]
    n_points = int(d["n_points"]); ds = float(d["ds"])
    m = X.shape[0] if max_samples is None else min(max_samples, X.shape[0])
    arclen = np.arange(n_points) * ds

    methods = ["Frenet", "Bishop"]
    net_predict = try_load_net()
    if net_predict is not None:
        methods.append("Net(本文)")
        Y_net = net_predict(X[:m])
        print("已加载神经网络模型，纳入对比。")
    else:
        print("未发现 torch/checkpoint，仅对比传统 baseline。")

    # 误差累积：每点误差对所有样本求均值
    err_curves = {meth: np.zeros(n_points) for meth in methods}
    endpoint = {meth: [] for meth in methods}

    for i in range(m):
        k1h, k2h = B.recover_curvature_lsq(X[i])
        recon = {
            "Frenet": B.reconstruct_frenet(k1h, k2h, ds),
            "Bishop": B.reconstruct_bishop(k1h, k2h, ds),
        }
        if net_predict is not None:
            recon["Net(本文)"] = Y_net[i]
        for meth in methods:
            e = B.pointwise_error(recon[meth], Y[i])
            err_curves[meth] += e
            endpoint[meth].append(e[-1])
    for meth in methods:
        err_curves[meth] /= m

    # ── 图1：误差 vs 弧长（核心图）──
    fig, ax = plt.subplots(figsize=(9, 5.5))
    colors = {"Frenet": "#e74c3c", "Bishop": "#2980b9", "Net(本文)": "#27ae60"}
    for meth in methods:
        ax.plot(arclen, err_curves[meth] * 1000, lw=2.5,
                color=colors.get(meth), label=meth)
    ax.set_xlabel("弧长 / 距离 (m)", fontsize=12)
    ax.set_ylabel("平均重建误差 (mm)", fontsize=12)
    ax.set_title("重建误差随距离的累积（长距离形状传感核心指标）", fontsize=13)
    ax.legend(fontsize=11); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "error_vs_arclength.png"), bbox_inches="tight")
    plt.close()

    # ── 图2：单条样本 3D 重建对比 ──
    i0 = 0
    k1h, k2h = B.recover_curvature_lsq(X[i0])
    P_true = Y[i0]
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(*P_true.T * 1000, color="k", lw=3, label="真值")
    ax.plot(*B.reconstruct_bishop(k1h, k2h, ds).T * 1000,
            color="#2980b9", lw=1.8, ls="--", label="Bishop")
    if net_predict is not None:
        ax.plot(*Y_net[i0].T * 1000, color="#27ae60", lw=1.8, ls=":", label="Net(本文)")
    ax.set_xlabel("X (mm)"); ax.set_ylabel("Y (mm)"); ax.set_zlabel("Z (mm)")
    ax.set_title("单条样本三维重建对比"); ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "sample_reconstruction.png"), bbox_inches="tight")
    plt.close()

    # ── 表：统计指标 ──
    print("\n{:<12} {:>14} {:>14} {:>14}".format(
        "方法", "端点误差均值(mm)", "端点误差std(mm)", "全程RMSE(mm)"))
    rows = []
    for meth in methods:
        ep = np.array(endpoint[meth]) * 1000
        rmse = np.sqrt(np.mean(err_curves[meth] ** 2)) * 1000
        print("{:<12} {:>14.2f} {:>14.2f} {:>14.2f}".format(
            meth, ep.mean(), ep.std(), rmse))
        rows.append([meth, ep.mean(), ep.std(), rmse])

    with open(os.path.join(HERE, "metrics.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "endpoint_mean_mm", "endpoint_std_mm", "rmse_mm"])
        w.writerows(rows)

    print("\n已保存: error_vs_arclength.png, sample_reconstruction.png, metrics.csv")


if __name__ == "__main__":
    main()
