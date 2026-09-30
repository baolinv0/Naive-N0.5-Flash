# 安装与启动

在本 examples/modular_neural_isp_phase1 目录执行：

```bash
git clone --depth 1 https://github.com/SamsungLabs/modular_neural_isp.git
cp -a overlay/. modular_neural_isp/
cd modular_neural_isp
```

然后按 [overlay/README_PHASE1.md](overlay/README_PHASE1.md) 安装环境、运行 baseline 和 campaign。覆盖文件基于本项目原始官方源码。

本轮默认 P512 dev 评测，每次运行重载 checkpoint。original/MSE/L1、Adam/AdamW、学习率可通过配置调整；模型结构保持固定。ARIS-Code 读取 campaign 的真实 dev 结果，再提交下一轮 proposal。最终测试单独执行。

- overlay/：最新代码、配置、测试和说明。
- review/：原审查意见及复现证据，完整保留。
- evidence/review_fix/：本轮 CPU 合成数据联调证据。
- Modular_Neural_ISP_Phase1.zip：历史 baseline 包，未包含本轮修复。

真实数据完整训练、真实 Naive/ARIS 驱动和多 seed 提升确认仍需对应资源。具体完成边界见 [本轮验证记录](overlay/docs/review-fix-verification.md)。
