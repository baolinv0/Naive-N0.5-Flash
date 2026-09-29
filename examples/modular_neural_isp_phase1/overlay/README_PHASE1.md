# 原始 Modular Neural ISP：第一阶段工程

本阶段先运行 Samsung 官方方案。保留模型结构、loss 权重、Adam、学习率调度和原始预处理；新增内容是数据配对修复、运行入口、结果记录及可选的 Naive 调用适配。尚未接入此前提出的风格迁移、teacher、新 loss 或自动优化搜索。

## 安装

在官方仓库根目录应用本工程文件，然后使用独立环境：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt -r requirements-phase1-dev.txt
python -m pytest -q
```

CPU 验证使用 Python 3.11、PyTorch 2.5.1+cpu 和 torchvision 0.20.1+cpu。原始 perceptual loss 首次运行会下载 VGG19 权重，请预留下载时间。Naive 服务使用另一个环境，避免覆盖 ISP 的固定依赖。

## 先跑工程联调

```bash
python scripts/make_smoke_data.py --output smoke_data
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python -m tm_research.cli baseline --config smoke_data/baseline.yaml
python -m tm_research.cli wait --run exp_001 --runs-dir smoke_data/runs --timeout 600
python -m tm_research.cli report --run exp_001 --runs-dir smoke_data/runs
```

将 `exp_001` 换成实际返回的 run ID。`baseline` 启动后台训练，返回后可随时通过 `status` 查询。运行器会在训练完成后读取最佳权重，再启动独立的测试进程。

合成联调设置为 2 张训练图、1 张验证图、1 张测试图，256×256、batch 1、2 epochs、固定 seed 7，从官方 style-0 权重初始化，使用原始 loss 和优化器。这只验证工程链路，不是完整训练或画质基准。

## 官方完整推理链路

生成上述数据后，可直接调用未修改的官方 demo：

```bash
python main/demo.py \
  --input-file smoke_data/test/denoised_raw_images/synthetic_test_000.png \
  --device cpu \
  --denoising-model-path denoising/models/s24_base.pth \
  --photofinishing-model-path photofinishing/models/photofinishing_s24-style-0.pth \
  --enhancement-model-path enhancement/models/enhancement_s24-style-0.pth \
  --output-dir smoke_data/full_demo --save-intermediate
```

该命令使用官方预训练模块完成降噪、基于 metadata 的颜色转换、photofinishing、上采样及 enhancement。此次验证沿用 metadata 的白平衡，没有启用重新估计 AWB。真实 DNG 可通过同一官方 demo 输入；本次未验证真实 DNG。

## 真实数据训练

编辑 `configs/baseline.example.yaml`，填入 train、validation、test 的输入图、GT 和 metadata 目录。输入为官方预处理后的 PNG，GT 为 JPG/JPEG，metadata 为 JSON；同一场景的文件名主干应一致，输入允许 `_denoised` 后缀。请沿用官方数据预处理流程。

```bash
python -m tm_research.cli commands --config configs/baseline.example.yaml
python -m tm_research.cli baseline --config configs/baseline.example.yaml
python -m tm_research.cli wait --run exp_001 --runs-dir runs
python -m tm_research.cli report --run exp_001 --runs-dir runs
```

未指定训练参数时沿用原默认值：600 epochs、batch 8、输入 512、每 4 epochs 验证一次、12 workers。最后一个 epoch 也会验证，以便短任务输出最佳权重。`init_checkpoint` 仅加载模型参数，不恢复优化器或 epoch。原始测试默认四分之一分辨率缩放保持不变；验证集与测试集指标应分别解读。

每次运行输出 `runs/exp_NNN/`：

- `config.json`、`commands.json`、`state.json`：实际参数、命令与状态。
- `train.log`、`test.log`：训练及测试日志。
- `train/models/*-best.pth`、`train/config/`：最佳模型和配套配置。
- `train/metrics.json`：逐图平均 PSNR、原 batch 聚合 PSNR、最佳模型位置。
- `test/metrics.json`、`test/per_image.csv`、`test/images/`：测试指标和图片。

所有运行的最终结果追加到 `runs/results.jsonl`。

## Naive 与 ARIS 调用

参见 [docs/naive-setup.md](docs/naive-setup.md) 和 `configs/aris.example.json`。已有兼容服务时直接配置其地址；需要原始 Transformers 推理时，可启动 `python -m tm_research.naive_adapter`。适配器支持 SSE、工具调用和工具结果回传，并在推理等待期间发送心跳。

当前已验证模拟生成器驱动的正常工具调用闭环；尚未运行真实 Naive 权重或 ARIS 进程。部署真实服务后，让 ARIS 依次调用本工程的 baseline、wait、report，即可接入相同的训练测试入口。

## 本次实际验证

已完成官方权重 CPU 前向、两轮真实模型训练、最佳权重重载测试，以及官方完整 demo 推理。详细结果见 [docs/verification.md](docs/verification.md)。真实数据长训练和真实 Naive 服务联调需要数据、GPU 及服务环境。

官方源代码及模型沿用仓库的 `LICENSE.md`。
