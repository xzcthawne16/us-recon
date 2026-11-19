# -*- coding: utf-8 -*-
"""
CNR（对比度噪声比，Contrast-to-Noise Ratio）

定义：CNR = (mean_in - mean_bg) / sigma_bg
- mean_in : 目标区像素均值
- mean_bg : 背景区像素均值
- sigma_bg: 背景区像素标准差（噪声/伪影水平）

含义：衡量目标区相对背景有多「突出」。伪影多 -> 背景标准差大 -> CNR 低；
      伪影被清除 -> 背景变干净 -> CNR 上升。
      这正是 H1 假设「消声区 CNR 应上升」的测量工具。
"""
import numpy as np


def cnr(img, in_mask, bg_mask):
    """计算对比度噪声比。

    Args:
        img:     2D numpy 数组（重建图）
        in_mask: 目标区 bool 掩码（True 表示目标区）
        bg_mask: 背景区 bool 掩码（True 表示背景区）
    Returns:
        CNR 值（float）
    """
    img = np.asarray(img, dtype=np.float64)
    in_mask = np.asarray(in_mask, dtype=bool)
    bg_mask = np.asarray(bg_mask, dtype=bool)

    mean_in = img[in_mask].mean()
    mean_bg = img[bg_mask].mean()
    sigma_bg = img[bg_mask].std()          # 默认 ddof=0，总体标准差（与手算一致）

    # 背景标准差为 0 时（背景完全均匀），CNR 趋于无穷
    if sigma_bg == 0:
        return float("inf") if mean_in != mean_bg else 0.0

    return (mean_in - mean_bg) / sigma_bg
