# -*- coding: utf-8 -*-
"""
点目标恢复率（Point Target Recovery Ratio）

定义：recovery = DL重建图中某点目标峰值 / 高质参考图中同一点峰值

含义：直接量化「亮点丢失」。若重建完整保留了该点 -> 比值 >= 1（甚至更强）；
      若该点被抑制/丢失 -> 比值 < 1（越接近 0 丢失越严重）。

      这是 H1（证明没删真组织）和 H2/H3（量化亮点丢失）的核心测量工具。
"""
import numpy as np


def point_recovery(img_rec, img_ref, in_mask):
    """计算点目标恢复率 = 重建图峰值 / 高质参考图峰值。

    Args:
        img_rec: 重建图（DL 输出），2D numpy 数组
        img_ref: 高质参考图，2D numpy 数组（与 img_rec 同形状）
        in_mask: 目标点区域 bool 掩码（圈出要评估的那一个点散射体）
    Returns:
        恢复率（float，>=0）。恢复率 >= 1 表示保住，< 1 表示丢失。
    """
    img_rec = np.asarray(img_rec, dtype=np.float64)
    img_ref = np.asarray(img_ref, dtype=np.float64)
    in_mask = np.asarray(in_mask, dtype=bool)

    peak_rec = np.abs(img_rec[in_mask]).max()
    peak_ref = np.abs(img_ref[in_mask]).max()

    if peak_ref == 0:
        return float("inf") if peak_rec > 0 else float("nan")

    return peak_rec / peak_ref
