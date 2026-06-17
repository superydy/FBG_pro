"""
训练脚本
==========
需要 PyTorch。本机/服务器执行：
    python3 data_generation.py        # 先生成数据
    python3 train.py                  # 训练 LSTM（默认）
    python3 train.py --model transformer
    python3 train.py --finetune real_data.npz   # 用少量真实数据微调（缩小 sim-to-real）

自动选择 GPU（若可用），无 GPU 也能在 CPU 上跑（慢一些）。
"""
import os
import sys
import argparse
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

import config as C
from model import build_model, ShapeLoss


def load_npz(path):
    d = np.load(path)
    X = torch.tensor(d["X"], dtype=torch.float32)
    Y = torch.tensor(d["Y"], dtype=torch.float32)
    return X, Y


def normalizer(X):
    """对输入 Δλ 做标准化，返回 (mean, std) 以及标准化函数。"""
    mean = X.mean(dim=(0, 1), keepdim=True)
    std = X.std(dim=(0, 1), keepdim=True) + 1e-8
    return mean, std


def run_epoch(model, loader, loss_fn, opt, device, train=True):
    model.train(train)
    tot, n = 0.0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        if train:
            opt.zero_grad()
        pred = model(xb)
        loss, _ = loss_fn(pred, yb)
        if train:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        tot += loss.item() * xb.size(0); n += xb.size(0)
    return tot / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="lstm", choices=["lstm", "transformer"])
    ap.add_argument("--finetune", default=None, help="真实数据 npz，用于微调")
    ap.add_argument("--epochs", type=int, default=C.EPOCHS)
    ap.add_argument("--out", default="checkpoint.pt")
    args = ap.parse_args()

    torch.manual_seed(C.SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("设备:", device)

    here = os.path.dirname(__file__)
    Xtr, Ytr = load_npz(os.path.join(here, "data", "train.npz"))
    Xva, Yva = load_npz(os.path.join(here, "data", "val.npz"))

    mean, std = normalizer(Xtr)
    Xtr = (Xtr - mean) / std
    Xva = (Xva - mean) / std

    tr = DataLoader(TensorDataset(Xtr, Ytr), batch_size=C.BATCH_SIZE, shuffle=True)
    va = DataLoader(TensorDataset(Xva, Yva), batch_size=C.BATCH_SIZE)

    model = build_model(args.model).to(device)
    loss_fn = ShapeLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=C.LR, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)

    best = 1e9
    for ep in range(1, args.epochs + 1):
        tl = run_epoch(model, tr, loss_fn, opt, device, train=True)
        with torch.no_grad():
            vl = run_epoch(model, va, loss_fn, opt, device, train=False)
        sched.step()
        if vl < best:
            best = vl
            torch.save({"model": model.state_dict(), "mean": mean, "std": std,
                        "name": args.model}, os.path.join(here, args.out))
        if ep % 5 == 0 or ep == 1:
            print(f"epoch {ep:3d} | train {tl:.4e} | val {vl:.4e} | best {best:.4e}")

    # ── 可选：用真实数据微调（缩小 sim-to-real gap）──
    if args.finetune:
        print("微调 with", args.finetune)
        Xr, Yr = load_npz(args.finetune)
        Xr = (Xr - mean) / std
        ft = DataLoader(TensorDataset(Xr, Yr), batch_size=min(32, len(Xr)), shuffle=True)
        for g in opt.param_groups:
            g["lr"] = C.LR * 0.1
        for ep in range(1, 21):
            fl = run_epoch(model, ft, loss_fn, opt, device, train=True)
            if ep % 5 == 0:
                print(f"  finetune ep {ep} | loss {fl:.4e}")
        torch.save({"model": model.state_dict(), "mean": mean, "std": std,
                    "name": args.model}, os.path.join(here, "checkpoint_finetuned.pt"))

    print("训练完成，最优模型已保存。")


if __name__ == "__main__":
    main()
