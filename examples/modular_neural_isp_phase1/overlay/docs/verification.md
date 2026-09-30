> 历史记录：本文件描述最初 baseline 阶段。当前功能、协议和范围以 [engineering-contract.md](engineering-contract.md) 和 [review-fix-verification.md](review-fix-verification.md) 为准。

# 实施与验证记录

验证日期：2026-09-29。基础源码：SamsungLabs/modular_neural_isp 当日公开源码。执行环境：Linux、Python 3.11、CPU、PyTorch 2.5.1+cpu、torchvision 0.20.1+cpu。

## 子 agent 分工

| 分工 | 交付 |
|---|---|
| ISP 实现 | 修复场景配对、HDF5 重读、验证平均值；训练和测试输出统一结果文件 |
| 调用适配 | Naive 工具格式解析、SSE 服务、工具历史转换、ARIS 配置说明 |
| 运行器 | YAML 配置、后台训练、最佳权重传递、独立测试进程、状态和报告入口 |
| 独立审查 | 检查正常使用流程及原配方保留情况；发现的 temperature=0 兼容问题已修复 |
| 主 agent | 依赖安装、真实模型集成运行、官方全链路 demo、文档和打包 |

## 已实际执行

1. 官方 style-0 photofinishing 权重 CPU 前向：输出为 1×3×256×256，全部有限，值域 0～1。
2. `scripts/make_smoke_data.py` 生成 2/1/1 张训练、验证和测试合成图；输入尺寸 256，batch 1，2 epochs，seed 7。原始模型、全部原 loss、Adam 和调度器实际参与训练，从官方 style-0 权重初始化。
3. 运行器执行 train → 保存最佳权重 → 独立 test 进程重新加载 → 保存图片和指标，最终状态 `completed`，约 86 秒。
4. 未修改的 `main/demo.py` 加载官方 s24_base denoising、style-0 photofinishing 和 enhancement 权重；处理合成 PNG-16，保存七个中间阶段和最终 JPG，进程退出码 0。使用 metadata 白平衡，没有重新估计 AWB。
5. `python -m pytest -q`：**15 passed**，耗时 12.86 秒。覆盖数据配对、HDF5 重读、验证平均、进程启动与结果传递、正常 SSE 工具循环、等待心跳和确定性生成参数。20 条警告来自依赖弃用提示。
6. `git diff --check` 通过。审查确认原模型结构、loss 配方、优化器、调度、增强和官方输入处理保持不变。

合成联调数值：

| 输出 | 数值 |
|---|---:|
| 验证集逐图平均 PSNR | 24.40784073 |
| 验证集原 batch 聚合 PSNR | 24.40783978 |
| 测试集 PSNR | 24.43007414 |
| 测试集 SSIM | 0.90919542 |
| 测试图数量 | 1 |

上述数值仅证明训练测试链路产生了可读取的结果，不代表真实相机数据效果，也不能用于判断改善或复现论文精度。验证与测试继续使用各自的原始处理协议。

## 尚待实际资源验证

- 真实数据集上的完整训练与论文指标对照。
- 真实 DNG 输入与重新估计 AWB 的可选分支。
- 真实 Naive 模型服务及 ARIS 进程驱动的调用闭环；当前适配器测试使用注入生成器。

这次未接入此前的改善方案，也未实现搜索策略或新的风格迁移模块。
