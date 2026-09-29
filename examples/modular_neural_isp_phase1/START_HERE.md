# 原始 Modular Neural ISP 第一阶段工程包

本包包含工程覆盖文件、复现说明及合成联调产物。

## 应用代码

```bash
git clone --depth 1 https://github.com/SamsungLabs/modular_neural_isp.git
cp -a overlay/. modular_neural_isp/
cd modular_neural_isp
```

请在解压目录执行上述命令。随后按 README_PHASE1.md 安装依赖并运行。
覆盖文件基于 2026-09-29 获取的官方源码；模型权重仍从官方仓库获取。

## 包内容

- overlay/：所有新增和修改的工程文件。
- evidence/：本次实际执行的结果；全部是合成工程联调数据，不能当作真实画质评估。
- evidence/trained/：短训练选出的权重与配置；仅用于验证输出可重载。
- LICENSE.md：原始源码及模型许可。

已验证：原模型 CPU 前向、两轮原 loss 训练、最佳模型重载测试、官方全链路 demo、15 项测试。
尚待资源验证：真实数据完整训练及真实 Naive/ARIS 调用。详情见 overlay/docs/verification.md。
