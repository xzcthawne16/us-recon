# -*- coding: utf-8 -*-
"""
gCNR（广义对比度噪声比，Generalized CNR）

定义：两个区域（目标/背景）强度分布的交叠面积（Overlap Area）。
      1 - 交叠面积 = gCNR，取值 0~1。

含义：CNR 的局限在于它假设背景是高斯分布（只用均值和标准差概括）。
      gCNR 不依赖分布假设，直接用直方图算两个区域强度分布「重叠多少」。
      gCNR 越高（越接近 1）表示两个区域越容易区分（分布越分离）。

      这正是 H1 假设里「两区域分布交叠面积」的指标。
"""
import numpy as np


def gcnr(img, in_mask, bg_mask, bins=256, range_min=None, range_max=None):
    """计算广义 CNR（分布交叠面积）。

    Args:
        img:     2D numpy 数组
        in_mask: 目标区 bool 掩码
        bg_mask: 背景区 bool 掩码
        bins:    直方图分箱数
        range_min/range_max: 强度范围，默认取两区域合并后的 min/max
    Returns:
        gCNR 值（0~1），越大越可区分
    """
    img = np.asarray(img, dtype=np.float64)
    in_mask = np.asarray(in_mask, dtype=bool)
    bg_mask = np.asarray(bg_mask, dtype=bool)

    vals_in = img[in_mask]
    vals_bg = img[bg_mask]

    if range_min is None:
        range_min = min(vals_in.min(), vals_bg.min())
    if range_max is None:
        range_max = max(vals_in.max(), vals_bg.max())

    # 归一化直方图（除以各自像素总数，得到概率密度）
    hist_in, _ = np.histogram(vals_in, bins=bins, range=(range_min, range_max))
    hist_bg, _ = np.histogram(vals_bg, bins=bins, range=(range_min, range_max))
    hist_in = hist_in / hist_in.sum()
    hist_bg = hist_bg / hist_bg.sum()

    # 交叠面积 = 对每个 bin 取两个分布的最小值之和
    overlap = np.sum(np.minimum(hist_in, hist_bg))
    return 1.0 - overlap
