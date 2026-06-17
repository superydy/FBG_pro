"""
神经网络模型 + 自定义损失
============================
任务：序列到序列回归
    输入 X: (batch, n_points, 4)   四芯 Δλ 沿光纤分布
    输出 Y: (batch, n_points, 3)   3D 坐标

核心创新点（区别于"随便套个 LSTM"）：
    自定义损失 = 逐点位置误差 + 端点累积误差惩罚 + 弧长/平滑约束
    端点惩罚专门针对"长距离误差发散"这一长距离形状传感的核心痛点。

提供两个模型：
    ShapeLSTM        : 双向 LSTM（主推，序列建模天然合适）
    ShapeTransformer : Transformer 编码器（消融/对比用）

注意：本文件需要 PyTorch。远程环境未装 torch，请在本机/服务器
    pip install torch  后运行 train.py。
"""
import torch
import torch.nn as nn
import config as C


# ─────────────────────────────────────────────────────────────
# 模型 1：双向 LSTM
# ─────────────────────────────────────────────────────────────
class ShapeLSTM(nn.Module):
    def __init__(self, in_dim=C.N_CORES, hidden=C.HIDDEN_SIZE,
                 layers=C.NUM_LAYERS, out_dim=3):
        super().__init__()
        self.input_norm = nn.LayerNorm(in_dim)
        self.lstm = nn.LSTM(in_dim, hidden, layers,
                            batch_first=True, bidirectional=True,
                            dropout=0.1 if layers > 1 else 0.0)
        self.head = nn.Sequential(
            nn.Linear(hidden * 2, hidden), nn.GELU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):                       # x: (B, n, 4)
        x = self.input_norm(x)
        h, _ = self.lstm(x)                     # (B, n, 2*hidden)
        return self.head(h)                     # (B, n, 3)


# ─────────────────────────────────────────────────────────────
# 模型 2：Transformer 编码器（对比）
# ─────────────────────────────────────────────────────────────
class ShapeTransformer(nn.Module):
    def __init__(self, in_dim=C.N_CORES, d_model=C.HIDDEN_SIZE,
                 nhead=8, layers=C.NUM_LAYERS, out_dim=3, max_len=C.N_POINTS):
        super().__init__()
        self.embed = nn.Linear(in_dim, d_model)
        self.pos = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)
        enc = nn.TransformerEncoderLayer(d_model, nhead, d_model * 4,
                                         dropout=0.1, batch_first=True,
                                         activation="gelu")
        self.encoder = nn.TransformerEncoder(enc, layers)
        self.head = nn.Linear(d_model, out_dim)

    def forward(self, x):
        x = self.embed(x) + self.pos[:, :x.size(1)]
        h = self.encoder(x)
        return self.head(h)


# ─────────────────────────────────────────────────────────────
# 自定义损失（创新点）
# ─────────────────────────────────────────────────────────────
class ShapeLoss(nn.Module):
    """
    L = L_point + w_end·L_endpoint + w_smooth·L_smooth + w_len·L_length
        L_point    : 逐点坐标 MSE
        L_endpoint : 末端点误差（直接压制累积误差发散）
        L_smooth   : 二阶差分（曲线光滑，物理合理）
        L_length   : 总弧长一致性（防止整体缩放漂移）
    """
    def __init__(self, w_end=C.ENDPOINT_W, w_smooth=0.1, w_len=0.5, ds=C.DS):
        super().__init__()
        self.w_end, self.w_smooth, self.w_len, self.ds = w_end, w_smooth, w_len, ds

    def forward(self, pred, target):
        # 逐点
        l_point = torch.mean(torch.sum((pred - target) ** 2, dim=-1))
        # 端点
        l_end = torch.mean(torch.sum((pred[:, -1] - target[:, -1]) ** 2, dim=-1))
        # 光滑（二阶差分）
        d2 = pred[:, 2:] - 2 * pred[:, 1:-1] + pred[:, :-2]
        l_smooth = torch.mean(torch.sum(d2 ** 2, dim=-1))
        # 弧长
        seg_p = torch.norm(pred[:, 1:] - pred[:, :-1], dim=-1).sum(dim=1)
        seg_t = torch.norm(target[:, 1:] - target[:, :-1], dim=-1).sum(dim=1)
        l_len = torch.mean((seg_p - seg_t) ** 2)

        total = (l_point + self.w_end * l_end
                 + self.w_smooth * l_smooth + self.w_len * l_len)
        return total, dict(point=l_point.item(), end=l_end.item(),
                           smooth=l_smooth.item(), length=l_len.item())


def build_model(name="lstm"):
    return ShapeTransformer() if name == "transformer" else ShapeLSTM()


if __name__ == "__main__":
    # 形状自检（无需训练）
    m = build_model("lstm")
    x = torch.randn(8, C.N_POINTS, C.N_CORES)
    y = m(x)
    loss_fn = ShapeLoss()
    loss, parts = loss_fn(y, torch.randn_like(y))
    print("LSTM out:", tuple(y.shape), "loss:", round(loss.item(), 4), parts)
    n_param = sum(p.numel() for p in m.parameters())
    print(f"参数量: {n_param/1e6:.2f} M")
