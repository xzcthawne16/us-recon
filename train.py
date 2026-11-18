# -*- coding: utf-8 -*-
"""
train.py —— us-recon 训练主程序

用法：
    python train.py --config configs/base.yaml

功能（对应学习任务 4.2 节）：
    1. 读 yaml 配置（数据/batch/lr/epoch/seed/loss）
    2. 固定随机种子（torch/np/random/cudnn）
    3. 造数据 -> 建模型 -> 建损失
    4. 训练循环，打印 epoch/loss/psnr/ssim
    5. 指标落盘 metrics_log.csv
"""
import argparse
import csv
import os
import random

import numpy as np
import torch
import yaml

from data.dataset import build_dataloaders
from models.unet import build_model
from losses.losses import build_loss
from metrics import psnr, ssim


# ===================== 1. 读配置 =====================
def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ===================== 2. 固定随机种子 =====================
def set_seed(seed):
    """固定所有随机源，保证训练可复现。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)          # 多卡也固定
    torch.backends.cudnn.deterministic = True  # cudnn 走确定算法
    torch.backends.cudnn.benchmark = False     # 关闭自动选算法（会引入不确定性）


# ===================== 3. 训练 =====================
def train(cfg, device):
    t = cfg["train"]

    # 数据
    train_loader, val_loader = build_dataloaders(cfg, t["seed"])

    # 模型 + 损失 + 优化器
    model = build_model(cfg).to(device)
    loss_fn = build_loss(cfg)
    opt = torch.optim.Adam(model.parameters(), lr=t["lr"])

    log_path = cfg["logging"]["metrics_csv"]
    print_every = cfg["logging"]["print_every"]

    # 写 csv 表头
    with open(log_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "train_loss", "val_loss", "val_psnr", "val_ssim"])

    print(f"设备: {device} | 模型参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    print(f"训练 {t['epochs']} epoch, batch={t['batch_size']}, lr={t['lr']}, seed={t['seed']}\n")

    for epoch in range(1, t["epochs"] + 1):
        # ---- 训练阶段 ----
        model.train()
        train_losses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
            train_losses.append(loss.item())
        avg_train_loss = float(np.mean(train_losses))

        # ---- 验证阶段 ----
        model.eval()
        val_losses, val_psnr_l, val_ssim_l = [], [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                val_losses.append(loss_fn(pred, yb).item())
                # 逐样本算 PSNR/SSIM（转 numpy）
                for i in range(pred.shape[0]):
                    p = pred[i, 0].cpu().numpy()
                    g = yb[i, 0].cpu().numpy()
                    val_psnr_l.append(psnr(p, g))
                    val_ssim_l.append(ssim(p, g))

        avg_val_loss = float(np.mean(val_losses))
        avg_psnr = float(np.mean(val_psnr_l))
        avg_ssim = float(np.mean(val_ssim_l))

        # 落盘 csv
        with open(log_path, "a", newline="") as f:
            w = csv.writer(f)
            w.writerow([epoch, avg_train_loss, avg_val_loss, avg_psnr, avg_ssim])

        # 打印
        if epoch % print_every == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/{t['epochs']} | "
                  f"train_loss {avg_train_loss:.4f} | "
                  f"val_loss {avg_val_loss:.4f} | "
                  f"PSNR {avg_psnr:.2f} dB | SSIM {avg_ssim:.4f}")

    print(f"\n训练完成！指标已写入 {log_path}")
    return model


# ===================== 4. 入口 =====================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed = cfg["train"]["seed"]
    set_seed(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    train(cfg, device)


if __name__ == "__main__":
    main()
