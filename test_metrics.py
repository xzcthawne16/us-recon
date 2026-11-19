# -*- coding: utf-8 -*-
"""
指标函数自测：每个指标用「已知答案」的 toy 输入断言输出 ≈ 手算值。

运行：python test_metrics.py
全部通过会打印 PASS；有失败会打印 FAIL 并指出差异。
"""
import numpy as np
from metrics import psnr, ssim, cnr, gcnr, fwhm, point_recovery


def close(a, b, tol=1e-6):
    return abs(a - b) < tol


def test_cnr():
    """手算答案 ≈ 63.64"""
    img = np.array([
        [100, 102, 10, 12],
        [98, 100, 8, 10],
    ], dtype=np.float64)
    in_mask = np.array([
        [True, True, False, False],
        [True, True, False, False],
    ])
    bg_mask = ~in_mask

    val = cnr(img, in_mask, bg_mask)
    # mean_in=100, mean_bg=10, sigma_bg=sqrt(2)≈1.414 -> (90)/1.414≈63.64
    expected = (100 - 10) / np.sqrt(2.0)
    assert close(val, expected), f"CNR: got {val}, expected {expected}"
    print(f"CNR  PASS  {val:.4f} (期望 {expected:.4f})")


def test_gcnr():
    """两个完全不重叠的区域 -> 交叠面积 0 -> gCNR = 1"""
    img = np.array([
        [100, 100, 0, 0],
        [100, 100, 0, 0],
    ], dtype=np.float64)
    in_mask = np.array([
        [True, True, False, False],
        [True, True, False, False],
    ])
    bg_mask = ~in_mask

    val = gcnr(img, in_mask, bg_mask, bins=100)
    assert close(val, 1.0), f"gCNR: got {val}, expected 1.0"
    print(f"gCNR PASS  {val:.4f} (期望 1.0000)")


def test_fwhm():
    """一个标准高斯点，-6dB 全宽应接近 2*sqrt(2ln2)*sigma"""
    sigma = 3.0
    x = np.linspace(-20, 20, 4001)  # 细采样，避免离散误差
    profile = np.exp(-(x ** 2) / (2 * sigma ** 2))

    val = fwhm(profile, axis_units=x, db=-6.0)
    # -6dB 对应幅度降为峰值的 0.5（因为 20log10(0.5)≈-6.02dB）
    # 高斯剖面幅度降到一半：exp(-(dx)^2/(2σ^2))=0.5 -> dx=σ*sqrt(2ln2)
    expected = 2 * sigma * np.sqrt(2 * np.log(2))
    assert close(val, expected, tol=0.02), f"FWHM: got {val}, expected {expected}"
    print(f"FWHM PASS  {val:.4f} (期望 {expected:.4f})")


def test_point_recovery():
    """重建图峰值是参考图的一半 -> 恢复率 = 0.5（亮点丢失了一半）"""
    img_ref = np.array([
        [0, 0, 0, 0],
        [0, 200, 0, 0],
        [0, 0, 0, 0],
    ], dtype=np.float64)
    img_rec = img_ref * 0.5  # 重建后峰值减半

    mask = img_ref > 10  # 圈出那个点

    val = point_recovery(img_rec, img_ref, mask)
    assert close(val, 0.5), f"point_recovery: got {val}, expected 0.5"
    print(f"point_recovery PASS  {val:.4f} (期望 0.5000)")


def test_psnr_ssim():
    """完全相同的两张图 -> PSNR=inf, SSIM=1"""
    a = np.random.RandomState(0).rand(16, 16)
    assert psnr(a, a) == float("inf")
    assert close(ssim(a, a), 1.0, tol=1e-4)
    print("PSNR/SSIM PASS  (相同图 -> PSNR=inf, SSIM=1)")


if __name__ == "__main__":
    test_cnr()
    test_gcnr()
    test_fwhm()
    test_point_recovery()
    test_psnr_ssim()
    print("\n全部指标自测通过 ✅")
