# -*- coding: utf-8 -*-
"""
FWHM（半高全宽，Full Width at Half Maximum）

定义：过峰值点，沿一条剖面（profile），找到信号强度下降到峰值 -6 dB 的两个点，
      这两点之间的宽度。

含义：超声中常用 -6 dB 宽度（幅度降到峰值一半对应的功率下降点）来衡量
      「一个点散射体在图像里有多宽」。宽度越小 -> 分辨率越高；
      若重建把点抹糊（过平滑），该点的 FWHM 会变大。

      这正是量化「亮点丢失/过平滑」的分辨率指标。
"""
import numpy as np


def fwhm(profile, axis_units=None, db=-6.0):
    """计算一条剖面的 -6dB 全宽。

    Args:
        profile:    1D numpy 数组（过峰值点的剖面，如某一行/列）
        axis_units: 可选，横轴坐标数组（如深度 z 或横向 x），
                    提供时返回物理宽度，否则返回像素数
        db:         阈值（dB），默认 -6dB
    Returns:
        宽度（像素数或物理单位）；找不到两侧下降点时返回 nan
    """
    profile = np.asarray(profile, dtype=np.float64)

    # 取幅值的峰值，转 dB（相对峰值 = 0 dB）
    peak = np.abs(profile).max()
    if peak == 0:
        return float("nan")

    db_profile = 20 * np.log10(np.abs(profile) / peak + 1e-12)

    peak_idx = int(np.argmax(np.abs(profile)))

    # 从峰值向左找第一个 <= db 阈值的点
    left = None
    for i in range(peak_idx, -1, -1):
        if db_profile[i] <= db:
            left = i
            break

    # 从峰值向右找第一个 <= db 阈值的点
    right = None
    for i in range(peak_idx, len(profile)):
        if db_profile[i] <= db:
            right = i
            break

    if left is None or right is None:
        return float("nan")

    if axis_units is not None:
        return float(np.abs(axis_units[right] - axis_units[left]))
    return float(right - left)
