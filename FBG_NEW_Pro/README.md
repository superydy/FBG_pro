# FBG_NEW_Pro —— 长距离四芯光纤光栅 3D 形状传感（算法方向）

面向**学硕（计算机方向）**的完整研究工程：用**深度学习重建**替代传统**逐段积分**，
解决长距离形状传感的**累积误差发散**问题。全程**不需要 ANSYS、不需要昂贵真值设备**。

---

## 一句话创新点

> 将形状重建从"逐段积分"转化为"序列到序列学习"，并在损失函数中引入**端点累积误差惩罚**，
> 在 N 米尺度上将端点重建误差较 Frenet/Bishop 显著降低。

传统方法（Frenet/Bishop）是逐段积分，误差一段段往后传，弧长越长端点误差越大；
神经网络直接由四芯 Δλ 序列映射到 3D 坐标，不做顺序积分，从而抑制误差累积。

---

## 整体流程

```
随机曲率(k1,k2) ──Bishop积分──► 3D真值曲线 ───┐
       │                                      │ 训练对 (X, Y)
       └──梁公式──► 各芯应变 ──FBG公式──► Δλ(+噪声)─┘
                                              │
        ┌─────────────────────────────────────┘
        ▼
  神经网络 (LSTM/Transformer) ：Δλ序列 ─► 3D坐标
        │
        ▼  对比
  Frenet积分 / Bishop积分 (baseline)
```

---

## 文件说明

| 文件 | 作用 | 是否需 torch |
|------|------|:---:|
| `config.py` | 全局参数（物理常数、光纤几何、噪声、超参数） | 否 |
| `forward_model.py` | **物理正向模型**：曲率→真值曲线、应变、Δλ | 否 |
| `data_generation.py` | 批量生成训练/验证/测试集（.npz） | 否 |
| `baselines.py` | 传统重建：加权最小二乘解曲率 + Frenet/Bishop 积分 | 否 |
| `model.py` | 网络（双向LSTM / Transformer）+ **自定义损失** | 是 |
| `train.py` | 训练（自动用GPU；支持真实数据微调） | 是 |
| `evaluate.py` | 对比出图：误差-弧长曲线、3D重建、指标表 | 部分 |

---

## 快速开始

```bash
# 1) 安装依赖（本机/服务器）
pip install -r requirements.txt          # 训练需要 torch

# 2) 生成数据
python3 data_generation.py --quick       # 小规模自检（几秒）
python3 data_generation.py               # 正式 10 万条（CPU 约 10+ 分钟）

# 3) 训练
python3 train.py                         # 默认 LSTM
python3 train.py --model transformer     # 消融对比

# 4) 评估出图
python3 evaluate.py                      # 生成 error_vs_arclength.png 等
```

> 远程环境未预装 torch，但 `forward_model.py / data_generation.py / baselines.py /
> evaluate.py(baseline部分)` 都已验证可独立运行。

---

## ⚠️ 投稿前务必做的两件事（决定论文生死）

1. **替换真实噪声参数**：`config.py` 里 `NOISE_DLAMBDA` 改成你解调仪说明书的波长重复性。
   仿真噪声越接近真实，sim-to-real gap 越小。
2. **做真实实验验证**：买不同直径 PVC 水管（DN15/25/40/50，游标卡尺测外径），
   缠绕四芯光纤采 15~20 组真实 Δλ，存成同格式 npz，用 `train.py --finetune` 微调后，
   在真实数据上报告精度。**论文核心结论必须基于真实数据，不能只有仿真。**

---

## 实验 / 写作路线

| 阶段 | 任务 |
|------|------|
| 1 | 跑通仿真→训练→评估全流程，确认网络优于 baseline |
| 2 | 替换真实噪声模型，重训 |
| 3 | 圆柱缠绕实物实验，采真实数据，微调 + 测试 |
| 4 | 消融（去掉端点损失/换网络）+ 对比（Frenet/Bishop/本文）|
| 5 | 写论文：引言→原理→方法→实验→结论 |

---

## 参考对比方法（真实文献）

- Moore & Rogge, *Optics Express* (2012) — 多芯光纤 Frenet-Serret 形状重建（经典 baseline）
- 形状传感中 Bishop / 平行移动标架用于消除直线段奇异性
- 多芯 FBG 抗扭转 / 加权融合相关工作（用于方法章节背景）
