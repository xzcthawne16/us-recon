# -*- coding: utf-8 -*-
"""
数据层：占位数据生成 + DataLoader 构造

占位阶段用随机张量模拟「低质图 -> 高质图」配对，
接口设计成以后可无缝替换为真实超声数据（PICMUS hdf5）：
只需新增一个 Dataset 类，并在 build_dataloaders 里按 config 的
data.mode 选择即可，train.py 无需改动。
"""
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class RandomPairDataset(Dataset):
    """占位数据集：随机张量当「低质输入」和「高质目标」。

    为了给网络一个可学习、且能反映指标变化的任务，这里让
    「高质目标」=「低质输入」经过轻微高斯平滑，模拟
    「低质(带噪声) -> 高质(干净)」的映射关系。
    """

    def __init__(self, n, channel, height, width, noise=0.1):
        self.n = n
        self.channel = channel
        self.height = height
        self.width = width
        self.noise = noise

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        rng = np.random.default_rng(idx)  # 用 idx 当种子，保证可复现这样同一条数据（同一个 idx）每次生成的内容都一样——保证可复现。如果每次都用全局随机数，同一个 idx 两次取出来的图会不一样，你就没法复现实验了。这是「可复现」的第一个关键细节。
        # 高质目标：随机平滑图（几团高斯亮点，模拟超声点散射体）
        target = np.zeros((self.channel, self.height, self.width), dtype=np.float32)
        n_blobs = 3
        for _ in range(n_blobs):
            cy = rng.integers(0, self.height)
            cx = rng.integers(0, self.width)
            amp = rng.uniform(0.5, 1.0)
            yy, xx = np.mgrid[0:self.height, 0:self.width]
            sigma = rng.uniform(2.0, 5.0)
            blob = amp * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * sigma ** 2))
            target[0] += blob

        #这一段在模拟超声图：超声图本质就是「黑背景上的一些亮点（点散射体）」。所以「高质目标图」= 3 个高斯亮点。这个 exp(...) 是二维高斯函数，你把它想成「在 (cy,cx) 处堆一座小山」，离中心越远越矮（越暗）。
        
        # 低质输入 = 高质目标 + 高斯噪声（模拟低质图）
        low = target + rng.normal(0, self.noise, target.shape).astype(np.float32)
        low = np.clip(low, 0, 1)
        target = np.clip(target, 0, 1)



        #「低质图 = 高质图 + 噪声」——这就是模拟「低质」的最简方式。clip(0,1) 是把数值限制在 0~1 之间（因为加了噪声可能超范围）。
        
        #返回一个元组 (输入, 标签)。这是 PyTorch 的约定：__getitem__ 返回的 (x, y)，DataLoader 会自动打包成 batch。
        return torch.from_numpy(low), torch.from_numpy(target)
    


def build_dataloaders(cfg, seed):
    """根据 config 构造 train / val 两个 DataLoader。

    Args:
        cfg:  解析后的 config 字典
        seed: 随机种子（用于打乱，保证可复现）
    Returns:
        train_loader, val_loader
    """
    d = cfg["data"]
    t = cfg["train"]

    #从 config 里读「训练 160 条、验证 40 条」，造两个数据集。
    train_ds = RandomPairDataset(
        d["n_samples"], d["channel"], d["height"], d["width"]
    )
    val_ds = RandomPairDataset(
        d["n_val"], d["channel"], d["height"], d["width"]
    )



#关键在 generator=g + manual_seed(seed)：

# shuffle=True 意味着训练时每个 epoch 会打乱数据顺序。但「打乱」也是随机的，如果不用固定种子的 generator，每次打乱顺序都不一样。
# 用固定 seed 的 generator，保证「打乱顺序」也是可复现的。这直接对应任务第 4.3 节的变式验证：「seed 42 → 0，预测 loss 是否逐位一致」——只有数据顺序也可复现，这个验证才公平。
    g = torch.Generator()
    g.manual_seed(seed)

    train_loader = DataLoader(
        train_ds, batch_size=t["batch_size"], shuffle=True, generator=g
    )
    val_loader = DataLoader(
        val_ds, batch_size=t["batch_size"], shuffle=False
    )
    return train_loader, val_loader


#data/ 层把「数据从哪来、长什么样」封装起来，对外只暴露一个 build_dataloaders(cfg, seed)。​ train.py 只调用这一个函数，不关心数据是随机张量还是真实 PICMUS——以后换数据只改这一个文件。