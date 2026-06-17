"""
数据集生成
============
批量调用正向模型，生成 (Δλ, 3D坐标) 训练/验证/测试集，存成 .npz。

用法：
    python3 data_generation.py            # 用 config 里的默认规模
    python3 data_generation.py --quick    # 小规模 smoke test（几秒钟）

输出文件：
    data/train.npz  data/val.npz  data/test.npz
    每个含: X (m,n,4) 含噪Δλ,  Y (m,n,3) 3D坐标真值,
            K (m,n,2) 真值曲率,  meta(温度/轴向应变)
"""
import os
import sys
import time
import numpy as np
import config as C
import forward_model as fm


def build_dataset(n_samples, rng, n_points=C.N_POINTS, ds=C.DS):
    X = np.zeros((n_samples, n_points, C.N_CORES), dtype=np.float32)
    Y = np.zeros((n_samples, n_points, 3),         dtype=np.float32)
    K = np.zeros((n_samples, n_points, 2),         dtype=np.float32)
    meta = np.zeros((n_samples, 2), dtype=np.float32)   # [delta_T, eps_axial]
    for i in range(n_samples):
        s = fm.make_one_sample(rng, n_points=n_points, ds=ds)
        X[i] = s["dlambda"]
        Y[i] = s["positions"]
        K[i, :, 0] = s["k1"]
        K[i, :, 1] = s["k2"]
        meta[i] = [s["delta_T"], s["eps_axial"]]
    return X, Y, K, meta


def main():
    quick = "--quick" in sys.argv
    n_tr, n_va, n_te = (200, 50, 50) if quick else (C.N_TRAIN, C.N_VAL, C.N_TEST)

    out_dir = os.path.join(os.path.dirname(__file__), "data")
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(C.SEED)

    for name, n in [("train", n_tr), ("val", n_va), ("test", n_te)]:
        t0 = time.time()
        X, Y, K, meta = build_dataset(n, rng)
        path = os.path.join(out_dir, f"{name}.npz")
        np.savez_compressed(path, X=X, Y=Y, K=K, meta=meta,
                            ds=C.DS, n_points=C.N_POINTS)
        print(f"[{name}] {n} 条样本 -> {path}  "
              f"({time.time()-t0:.1f}s, {os.path.getsize(path)/1e6:.1f} MB)")

    print("完成。X=(样本, 点数, 4芯Δλ)  Y=(样本, 点数, xyz)")


if __name__ == "__main__":
    main()
