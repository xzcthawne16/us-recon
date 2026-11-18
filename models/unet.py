# -*- coding: utf-8 -*-
"""
UNet 网络定义（超声重建用）

结构：编码器(4层下采样) -> 瓶颈 -> 解码器(4层上采样，PixelShuffle)
上采样采用 PixelShuffle（亚像素卷积），避免转置卷积的棋盘伪影。

本模块从 11/9 的 stepC_full_rebuild.py 中抽取并参数化，
供 train.py / eval.py 通过 config 里的 base_channels 控制规模。
"""
import torch
import torch.nn as nn


class UNet(nn.Module):
    """单通道超声重建 UNet。

    Args:
        in_ch:  输入通道数（灰度超声 = 1）
        out_ch: 输出通道数（重建图 = 1）
        base:   基础通道数，控制网络规模（默认 32）
    """

    def __init__(self, in_ch=1, out_ch=1, base=32):
        super().__init__()
        self.enc1 = self._block(in_ch, base)
        self.pool = nn.MaxPool2d(2)

        self.enc2 = self._block(base, base * 2)
        self.enc3 = self._block(base * 2, base * 4)
        self.enc4 = self._block(base * 4, base * 8)
        self.bottleneck = self._block(base * 8, base * 16)

        # 上采样：PixelShuffle(2) 将 (C*4, H, W) -> (C, 2H, 2W)
        self.up4 = nn.Sequential(nn.Conv2d(base * 16, base * 8 * 4, 1), nn.PixelShuffle(2))
        self.dec4 = self._block(base * 16, base * 8)

        self.up3 = nn.Sequential(nn.Conv2d(base * 8, base * 4 * 4, 1), nn.PixelShuffle(2))
        self.dec3 = self._block(base * 8, base * 4)

        self.up2 = nn.Sequential(nn.Conv2d(base * 4, base * 2 * 4, 1), nn.PixelShuffle(2))
        self.dec2 = self._block(base * 4, base * 2)

        self.up1 = nn.Sequential(nn.Conv2d(base * 2, base * 4, 1), nn.PixelShuffle(2))
        self.dec1 = self._block(base * 2, base)

        self.out = nn.Conv2d(base, out_ch, 1)

    def _block(self, in_c, out_c):
        """编码/解码的基本单元：Conv -> BN -> ReLU 重复两次。"""
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        # 编码
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))

        # 解码（上采样 + 跳跃连接）
        d4 = self.dec4(torch.cat([self.up4(b), e4], 1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], 1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], 1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], 1))
        return self.out(d1)


def build_model(cfg):
    """根据 config 里的 model 段构造网络。

    Args:
        cfg: 解析后的 config 字典
    Returns:
        nn.Module 实例
    """
    name = cfg["model"]["name"]
    if name == "unet":
        return UNet(
            in_ch=cfg["data"]["channel"],
            out_ch=cfg["data"]["channel"],
            base=cfg["model"]["base_channels"],
        )
    raise ValueError(f"未知模型名称: {name}")
