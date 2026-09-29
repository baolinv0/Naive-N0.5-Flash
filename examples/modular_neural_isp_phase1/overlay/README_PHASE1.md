# Modular Neural ISP：baseline 与受控研究循环

保留 Samsung 官方 photofinishing 模型、RAW/metadata 前处理与原始 recipe，增加可选 MSE/L1、AdamW/学习率配置，以及由 ARIS-Code + Naive 驱动的串行 campaign。未接入历史增强方案、风格迁移或教师监督。

## 安装

将本目录覆盖文件应用到 Samsung 官方 ISP 仓库，在该仓库中执行：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt -r requirements-phase1-dev.txt
python -m pytest -q
```

本轮 CPU 环境为 Python 3.12.14、PyTorch 2.5.1+cpu。原始 loss 首次构造需要 VGG19 权重；纯 MSE/L1 不加载 VGG。Naive 服务使用独立推理环境。

## 单次训练与 dev 评测

```bash
python scripts/make_smoke_data.py --output smoke_data
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python -m tm_research.cli baseline --config smoke_data/baseline.yaml
python -m tm_research.cli wait --run exp_001 --runs-dir smoke_data/runs --timeout 600
python -m tm_research.cli report --run exp_001 --runs-dir smoke_data/runs
```

实际 run ID 以返回值为准。训练后自动重载最佳权重评测 **dev**。配置中的 test 不会随普通实验执行。

合成 fixture 使用 2 张 train、1 张 dev、1 张 test、256×256、2 epochs、batch1、seed7，从官方 style-0 权重初始化。它只验证功能，不代表真实相机画质。正式训练默认 600 epochs、batch8、输入512、评测 P512；评测大小由 `eval_size` 指定，smoke 明确为 P256。

## 可调训练方案

```yaml
loss_family: original  # original / mse / l1
optimizer: adam       # adam / adamw
learning_rate: 0.0001
weight_decay: 0.0000001
eval_size: 512
seed: 7
```

未设置时保留官方默认值。每个 campaign 固定数据、初始化、seed、预算和协议，只接受 recipe 字段变化。`init_checkpoint` 仅加载权重，不代表恢复优化器或 epoch。

验证、best 选择、重载评测和候选比较统一为逐图 RGB PSNR 后平均；用 float64 计算 MSE，最小 MSE 为 1e-12。P512/P256 使用与 validation 一致的模型前向路径；官方 quarter/full 推理作为独立协议，不能混入同一排行榜。

## 结果驱动的研究

编辑 `configs/baseline.example.yaml` 填好真实 train/dev 路径后：

```bash
python -m tm_research.cli campaign init --config configs/baseline.example.yaml --campaign-dir campaigns/pilot --max-trials 4 --no-gain-limit 3 --wait
python -m tm_research.cli campaign next --campaign-dir campaigns/pilot
```

`next` 返回已有真实 dev 结果和下一轮需要回答的问题。Naive 经 ARIS-Code 读取这些结果，写入 proposal JSON，再调用 `campaign submit`。每次提交必须包含假设、所依据的结果及可执行 recipe。控制器运行、比较和记录；达到预算或连续无收益条件后停止。完整命令与 ARIS 任务见 [docs/tm_research_task.md](docs/tm_research_task.md)。

本地模拟 proposer 只能证明控制器能消费结果并执行不同方案；真实 Naive 的研究行为需要 live 服务联调。

## 封存测试

```bash
python -m tm_research.cli campaign finalize --campaign-dir campaigns/pilot
python -m tm_research.cli final-test --campaign-dir campaigns/pilot
```

先冻结候选，再单独运行 test。日常研究 prompt 不包含测试分数。多 seed 统计和仅 GTM/LTM 更新的归因实验属于后续阶段，当前不声称已确认 PSNR 提升。

## 原始全 ISP 推理

官方 `main/demo.py` 仍可加载 denoising、photofinishing、enhancement 权重执行完整链路。本次只修改训练、评价及研究控制入口，不修改模型结构或全 ISP demo。

## 验证范围与入口

- [本轮 review 修复与实测记录](docs/review-fix-verification.md)
- [Naive / ARIS 服务设置](docs/naive-setup.md)
- [研究合同](docs/engineering-contract.md)
- [上一阶段历史验证](docs/verification.md)

源码及模型许可遵循原仓库 LICENSE.md。
