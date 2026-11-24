# -*- coding: utf-8 -*-
"""
2026-11-24 主任务：残差学习 + 点目标/边缘加权损失（第一次修复尝试）

目标：验证 H2 假设「过平滑是亮点丢失根因」。
手段：
  1. 残差学习：网络输出学 Δ = 高质 - 低质，最终重建 = 低质 + Δ
  2. 加权损失：对强点区域 / 边缘区域加权，让 loss 盯住稀疏强反射

数据：复用 11/19 的自仿真（已知真点 + 已知消声区），见 1119/us_recon_test/train_recon.py

四指标对比：PSNR / SSIM / CNR / gCNR（+ 点目标恢复率作第五个）

设计要点（带你理清逻辑，不是代跑）：
  - baseline   = 11/19 的原始 UNet + L1Loss
  - 修复版 A   = 只加残差（skip 连接）
  - 修复版 B   = 只加加权损失
  - 修复版 A+B = 残差 + 加权（2x2 消融）
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ================= 全局仿真参数（与 11/19 一致，保证可比）=================
c = 1540.0
fs = 40e6
N = 64
pitch = 0.3e-3
sigma = 0.05e-6
elem_x = (np.arange(N) - (N - 1) / 2) * pitch

nx, nz = 256, 512
x_grid = np.linspace(-10e-3, 10e-3, nx)
z_grid = np.linspace(2e-3, 40e-3, nz)
X, Z = np.meshgrid(x_grid, z_grid)
z_max = 40e-3
t_axis = np.arange(0, 2 * z_max / c, 1 / fs)

# 消声区（固定，用于 CNR/gCNR 背景）
anechoic_region = {"x_range": (-8e-3, -4e-3), "z_range": (10e-3, 18e-3)}
anechoic_mask = (X >= anechoic_region["x_range"][0]) & (X <= anechoic_region["x_range"][1]) \
              & (Z >= anechoic_region["z_range"][0]) & (Z <= anechoic_region["z_range"][1])

# #这个区域里没有放任何真点，理论上是"应该全黑"的区域。所以：

# 如果这里有信号 → 那是伪影；
# 用它当"背景"，就能算 CNR（对比度噪声比）。
# 这就是"已知消声区"的含义——你知道这里不该有任何东西，所以这里的信号一定是伪影。


# ================= 仿真：低质图 / 理想高质图 =================
def make_low_image(scatterers):
    """K=1 单角度仿真 -> 低质图（含栅瓣/旁瓣伪影）"""
    rf = np.zeros((N, len(t_axis)))
    for xk, zk, amp in scatterers:
        tau = (zk + np.sqrt((xk - elem_x) ** 2 + zk ** 2)) / c
        rf += amp * np.exp(-((t_axis[None, :] - tau[:, None]) ** 2) / (2 * sigma ** 2))
    tau_img = (Z[:, :, None] + np.sqrt((X[:, :, None] - elem_x[None, None, :]) ** 2
                                       + Z[:, :, None] ** 2)) / c
    idx = np.clip(np.round(tau_img * fs).astype(int), 0, len(t_axis) - 1)
    return rf[np.arange(N)[None, None, :], idx].sum(axis=2)


def make_ideal_image(scatterers, sigma_pt=0.3e-3):
    """理想高质图：每个真点 = 一个高斯点，无伪影"""
    img = np.zeros((nz, nx))
    for xk, zk, amp in scatterers:
        dx = X - xk
        dz = Z - zk
        img += amp * np.exp(-(np.square(dx) + np.square(dz)) / (2.0 * sigma_pt * sigma_pt))
    return img


def random_scatterers(n_pts=3):
    """随机摆放 n_pts 个真点（避开消声区）"""
    pts = []
    while len(pts) < n_pts:
        xk = np.random.uniform(-8e-3, 8e-3)
        zk = np.random.uniform(6e-3, 36e-3)
        if (-8e-3 <= xk <= -4e-3) and (10e-3 <= zk <= 18e-3):
            continue
        amp = np.random.uniform(0.3, 1.0)
        pts.append([xk, zk, amp])
    return np.array(pts)


def gen_pair(noise=0.0):
    """生成一对 (低质图, 高质图)，归一化到 [0,1]；noise 为低质图加的噪声标准差（相对峰值）"""
    sc = random_scatterers()
    low = make_low_image(sc)
    ideal = make_ideal_image(sc)
    low = low / (np.abs(low).max() + 1e-8)
    ideal = ideal / (np.abs(ideal).max() + 1e-8)
    if noise > 0:
        low = low + np.random.randn(*low.shape).astype(np.float32) * noise
    return low, ideal


# ================= UNet（复用 11/19 骨架，加残差开关）=================
class ResUNet(nn.Module):
    """
    残差 UNet：输出学「残差 Δ」，最终重建 = 输入 + Δ。
    residual=True 时走残差分支；False 时退化为普通 UNet（等价 baseline）。
    """

    def __init__(self, in_ch=1, out_ch=1, base=16, residual=True):
        super().__init__()
        self.residual = residual

        def blk(a, b):
            return nn.Sequential(nn.Conv2d(a, b, 3, padding=1), nn.BatchNorm2d(b), nn.ReLU(True),
                                 nn.Conv2d(b, b, 3, padding=1), nn.BatchNorm2d(b), nn.ReLU(True))

        self.e1 = blk(in_ch, base); self.p = nn.MaxPool2d(2)
        self.e2 = blk(base, base * 2)
        self.e3 = blk(base * 2, base * 4)
        self.b = blk(base * 4, base * 8)
        self.u3 = nn.Sequential(nn.Conv2d(base * 8, base * 4 * 4, 1), nn.PixelShuffle(2)); self.d3 = blk(base * 8, base * 4)
        self.u2 = nn.Sequential(nn.Conv2d(base * 4, base * 2 * 4, 1), nn.PixelShuffle(2)); self.d2 = blk(base * 4, base * 2)
        self.u1 = nn.Sequential(nn.Conv2d(base * 2, base * 4, 1), nn.PixelShuffle(2)); self.d1 = blk(base * 2, base)
        self.out = nn.Conv2d(base, out_ch, 1)

    def forward(self, x):
        e1 = self.e1(x); e2 = self.e2(self.p(e1)); e3 = self.e3(self.p(e2)); b = self.b(self.p(e3))
        d3 = self.d3(torch.cat([self.u3(b), e3], 1))
        d2 = self.d2(torch.cat([self.u2(d3), e2], 1))
        d1 = self.d1(torch.cat([self.u1(d2), e1], 1))
        delta = self.out(d1)
        if self.residual:
            return x + delta          # 残差：重建 = 低质 + Δ
        return delta                  # 普通：直接输出重建


# ================= 加权损失 =================
def build_weight_map(low_img, w_strong=5.0, w_edge=2.0, w_depth=0.0):
    """
    从「低质图」构造空间权重图：
      - 强点区域：低质图里高强度像素（已归一化到 0~1，取强度本身当权重）
      - 边缘区域：用梯度幅度（中心差分）捕捉边缘
      - 深度补偿（可选）：沿 z 方向线性加大权重，救回深部弱点
    权重 = 1 + w_strong * intensity + w_edge * grad_norm + w_depth * depth

    depth 是沿行方向（z 方向，图的上到下 = 浅到深）的归一化斜坡。
    当 w_depth > 0 时，深处（z=34mm 那个弱点）权重更高，
    这正是「弱点也加权」的策略，用于对比「只对强点加权」。
    """
    intensity = torch.clamp(low_img, 0, 1)

    # 梯度幅度（用中心差分近似，避免额外依赖）
    gx = intensity[..., 2:, 1:-1] - intensity[..., :-2, 1:-1]
    gy = intensity[..., 1:-1, 2:] - intensity[..., 1:-1, :-2]
    grad = torch.zeros_like(intensity)
    grad[..., 1:-1, 1:-1] = torch.sqrt(gx ** 2 + gy ** 2 + 1e-8)
    grad = grad / (grad.max() + 1e-8)

    # 深度斜坡：沿 H 维（z 方向）从 0 到 1
    H = intensity.shape[-2]
    depth = torch.linspace(0, 1, H, device=intensity.device).view(1, 1, H, 1)
    depth = depth.expand_as(intensity)

    w = 1.0 + w_strong * intensity + w_edge * grad + w_depth * depth
    return w


class WeightedL1Loss(nn.Module):
    """
    加权 L1：loss = mean(w * |pred - target|)。
    权重 w 对强点/边缘/深处取大值，让稀疏强反射不被背景均值淹没。
    """

    def __init__(self, w_strong=5.0, w_edge=2.0, w_depth=0.0):
        super().__init__()
        self.w_strong = w_strong
        self.w_edge = w_edge
        self.w_depth = w_depth

    def forward(self, pred, target, low_img):
        w = build_weight_map(low_img, w_strong=self.w_strong,
                             w_edge=self.w_edge, w_depth=self.w_depth)
        return (w * torch.abs(pred - target)).mean()


# ================= 指标（复用 us-recon/metrics 的思路，本地内联）=================
def psnr(a, b):
    a = np.asarray(a, dtype=np.float64); b = np.asarray(b, dtype=np.float64)
    mse = np.mean((a - b) ** 2)
    return float("inf") if mse == 0 else 10 * np.log10(1.0 / mse)


def ssim(a, b, C1=1e-4, C2=9e-4):
    a = np.asarray(a, dtype=np.float64); b = np.asarray(b, dtype=np.float64)
    ma, mb = a.mean(), b.mean()
    va, vb = max(a.var(), 0.0), max(b.var(), 0.0)
    cov = np.mean((a - ma) * (b - mb))
    l = (2 * ma * mb + C1) / (ma ** 2 + mb ** 2 + C1)
    c = (2 * np.sqrt(va) * np.sqrt(vb) + C2) / (va + vb + C2)
    s = (cov + C2 / 2) / (np.sqrt(va) * np.sqrt(vb) + C2 / 2)
    return l * c * s


def cnr(img, in_mask, bg_mask):
    img = np.asarray(img, dtype=np.float64)
    mu_in = img[in_mask].mean()
    mu_bg = img[bg_mask].mean()
    sg_bg = img[bg_mask].std()
    return (mu_in - mu_bg) / (sg_bg + 1e-8)


def gcnr(img, in_mask, bg_mask, bins=256):
    img = np.asarray(img, dtype=np.float64)
    vi, vb = img[in_mask], img[bg_mask]
    rmin, rmax = min(vi.min(), vb.min()), max(vi.max(), vb.max())
    hi, _ = np.histogram(vi, bins=bins, range=(rmin, rmax))
    hb, _ = np.histogram(vb, bins=bins, range=(rmin, rmax))
    hi = hi / hi.sum(); hb = hb / hb.sum()
    return 1.0 - np.sum(np.minimum(hi, hb))


def point_recovery(img_rec, img_ref, mask):
    return np.abs(img_rec[mask]).max() / (np.abs(img_ref[mask]).max() + 1e-8)


# ================= 主流程 =================
if __name__ == "__main__":
    # 配置：可通过命令行切换残差/加权开关，做 2x2 消融
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--residual", type=int, default=1, help="1=残差 0=普通")
    ap.add_argument("--weighted", type=int, default=1,
                    help="0=不加权(L1) 1=强点+边缘加权 2=强点+边缘+深度补偿")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--noise", type=float, default=0.0,
                    help="低质图加的高斯噪声标准差（相对峰值），用于触发过平滑丢点")
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. 训练集
    n_train = 200
    lows, ideals = [], []
    for _ in range(n_train):
        lo, hi = gen_pair(noise=args.noise)
        lows.append(lo); ideals.append(hi)
    lows = np.array(lows)[:, None, :, :].astype(np.float32) #(200,1,512,256) 样本数200，通道1灰度图，高512X宽256
    ideals = np.array(ideals)[:, None, :, :].astype(np.float32)

    # 2. 测试集（固定 3 个真点，难度梯度：浅强 -> 深弱）
    # 点3 幅度 0.4（深且弱），让 baseline 在噪声下明显丢点，成为 H2 假设的靶子
    test_sc = np.array([[0.0e-3, 12e-3, 1.0], [0.0e-3, 25e-3, 0.7], [0.0e-3, 34e-3, 0.4]])
    test_low = make_low_image(test_sc); test_low = test_low / (np.abs(test_low).max() + 1e-8)
    if args.noise > 0:
        test_low = test_low + np.random.randn(*test_low.shape).astype(np.float32) * args.noise
    test_ideal = make_ideal_image(test_sc); test_ideal = test_ideal / (np.abs(test_ideal).max() + 1e-8)

    # 3. 模型 + 优化
    model = ResUNet(residual=bool(args.residual)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    l1 = nn.L1Loss()
    # 加权模式：1=强点+边缘，2=强点+边缘+深度补偿
    if args.weighted == 1:
        wloss = WeightedL1Loss(w_strong=5.0, w_edge=2.0, w_depth=0.0)
    elif args.weighted == 2:
        wloss = WeightedL1Loss(w_strong=5.0, w_edge=2.0, w_depth=3.0)
    else:
        wloss = None

    Xt = torch.from_numpy(lows).to(device)
    Yt = torch.from_numpy(ideals).to(device)
    bs = 16

    print(f"配置: residual={bool(args.residual)} weighted={args.weighted} | "
          f"epochs={args.epochs} | 设备={device}")

    # 4. 训练
    for ep in range(args.epochs):
        model.train()
        perm = torch.randperm(n_train)
        for i in range(0, n_train, bs):
            idx = perm[i:i + bs]
            xb, yb = Xt[idx], Yt[idx]
            opt.zero_grad()
            out = model(xb)
            if wloss is not None:
                loss = wloss(out, yb, xb)   # 加权：传入低质图算权重
            else:
                loss = l1(out, yb)
            loss.backward(); opt.step()

    # 5. 推理测试图
    model.eval()
    with torch.no_grad():
        tin = torch.from_numpy(test_low[None, None]).float().to(device)
        rec = model(tin).cpu().numpy()[0, 0]
    rec = np.clip(rec, 0, None); rec = rec / (np.abs(rec).max() + 1e-8)

    # 6. 四指标 + 点目标恢复率
    def pt_mask(x0, z0, hw=1.0e-3, hh=1.5e-3):
        return (np.abs(X - x0) <= hw) & (np.abs(Z - z0) <= hh)

    # 消声区当背景，真点1当目标
    in_mask = pt_mask(test_sc[0][0], test_sc[0][1])

    print("\n===== 四指标对比（baseline=低质图 vs 修复版重建）=====")
    print(f"PSNR:  低质={psnr(test_low, test_ideal):6.2f}dB | 重建={psnr(rec, test_ideal):6.2f}dB")
    print(f"SSIM:  低质={ssim(test_low, test_ideal):.4f}   | 重建={ssim(rec, test_ideal):.4f}")
    print(f"CNR:   低质={cnr(test_low, in_mask, anechoic_mask):8.2f} | 重建={cnr(rec, in_mask, anechoic_mask):8.2f}")
    print(f"gCNR:  低质={gcnr(test_low, in_mask, anechoic_mask):.4f}  | 重建={gcnr(rec, in_mask, anechoic_mask):.4f}")

    print("\n===== 点目标恢复率（核心：亮点有没有保住）=====")
    for i, (xk, zk, amp) in enumerate(test_sc):
        m = pt_mask(xk, zk)
        r = point_recovery(rec, test_ideal, m)
        print(f"  真点{i+1}(z={zk*1e3:.0f}mm): 恢复率={r:.3f}")

    # 7. 出图
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(1, 3, figsize=(15, 8))
    for ax, im, t in zip(axes, [test_low, rec, test_ideal], ["低质图(K=1)", "残差+加权重建", "理想高质图"]):
        disp = 20 * np.log10(np.abs(im) / np.abs(im).max() + 1e-6)
        ax.imshow(disp, extent=[x_grid[0]*1e3, x_grid[-1]*1e3, z_grid[-1]*1e3, z_grid[0]*1e3],
                  cmap="gray", aspect="auto", vmin=-40, vmax=0)
        ax.set_title(t); ax.set_xlabel("x (mm)"); ax.set_ylabel("z (mm)")
    plt.tight_layout()
    out = f"residual{args.residual}_weighted{args.weighted}.png"
    plt.savefig(out, dpi=150)
    print(f"\n已保存对比图 {out}")