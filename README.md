# us-recon

少角度超声（Plane-Wave / Sparse-Angle）图像重建。

## 一句话问题

> 用深度学习从**少角度（低质）超声图像**重建出接近**多角度复合（高质）**的图像，
> 同时解决两个关键问题：**如何证明重建只去伪影、不损伤真组织**，以及**重建中亮点丢失的根因**。

## 目录结构

```
us-recon/
├─ configs/            # yaml 配置
├─ data/               # 数据读取、预处理、patch 切分
├─ models/             # 网络定义（UNet）
├─ losses/             # 损失函数
├─ metrics/            # PSNR/SSIM/CNR/gCNR/FWHM/点目标恢复率
├─ train.py            # 训练入口
├─ eval.py             # 评估入口（待补）
└─ requirements.txt
```

## 快速开始

```bash
# 安装依赖
pip install -r requirements.txt

# 一键复现占位训练（随机张量数据）
python train.py --config configs/base.yaml
```

## 结果表（待补）

| 配置 | 数据 | PSNR (dB) | SSIM | 备注 |
|---|---|---|---|---|
| base.yaml | 随机张量占位 | 待填 | 待填 | 骨架验证 |
| （后续） | PICMUS | 待填 | 待填 | 真实数据 |
| （后续） | 自仿真点散射体 | 待填 | 待填 | 已知真点/伪影 |
