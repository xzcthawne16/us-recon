# -*- coding: utf-8 -*-
"""
基础评价指标：PSNR、SSIM（NumPy 实现）

PSNR: 峰值信噪比，衡量重建图与真值的逐像素差异。
SSIM: 结构相似性，衡量亮度/对比度/结构三方面的相似度。

注意：SSIM 的方差项在数值上可能为负，需 clamp(min=0) 保护
（对应学习任务第 5 块「负方差 clamp」的复习点）。
"""
import numpy as np
from math import log10


def psnr(a, b, data_range=1.0):
    """峰值信噪比（Peak Signal-to-Noise Ratio）。

    Args:
        a, b: 两个同形状数组（重建图、真值）
        data_range: 数据动态范围（归一化到 [0,1] 时为 1.0）
    Returns:
        PSNR 值（dB），完全相同时返回 inf
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    mse = np.mean((a - b) ** 2)
    if mse == 0:
        return float("inf")
    return 10 * log10((data_range ** 2) / mse)


def ssim(a, b, C1=1e-4, C2=9e-4):
    """结构相似性（Structural Similarity，全局单值版）。

    用整图均值/方差/协方差计算，适合快速评估。
    Args:
        a, b: 两个同形状数组
        C1, C2: 稳定常数，避免分母为 0
    Returns:
        SSIM 值（-1 ~ 1，1 表示完全相同）
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)

    mu_a, mu_b = a.mean(), b.mean()
    var_a, var_b = a.var(), b.var()
    cov = np.mean((a - mu_a) * (b - mu_b))

    # 负方差保护：clamp(min=0)，避免开方/除零异常
    var_a = max(var_a, 0.0)
    var_b = max(var_b, 0.0)

    l = (2 * mu_a * mu_b + C1) / (mu_a ** 2 + mu_b ** 2 + C1)
    c = (2 * np.sqrt(var_a) * np.sqrt(var_b) + C2) / (var_a + var_b + C2)
    s = (cov + C2 / 2) / (np.sqrt(var_a) * np.sqrt(var_b) + C2 / 2)
    return l * c * s
