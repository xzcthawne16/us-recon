# -*- coding: utf-8 -*-
"""
损失函数层

把 config 里的 loss 类型字符串映射为实际的损失函数。
当前支持 l1 / mse；后续 H2 假设（强点/边缘加权损失）在此扩展。
"""
import torch.nn as nn


def build_loss(cfg):
    """根据 config 里的 train.loss 构造损失函数。

    Args:
        cfg: 解析后的 config 字典
    Returns:
        nn.Module 损失函数实例
    """
    loss_name = cfg["train"]["loss"].lower()

    if loss_name == "l1":
        return nn.L1Loss()
    if loss_name == "mse":
        return nn.MSELoss()

    # TODO: 在此扩展 H2 假设需要的「强点/边缘加权损失」
    # if loss_name == "weighted_l1":
    #     return WeightedL1Loss(...)

    raise ValueError(f"未知损失类型: {loss_name}")
