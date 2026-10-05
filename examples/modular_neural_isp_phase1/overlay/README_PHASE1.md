# Modular Neural ISP：baseline 与受控研究循环

保留 Samsung 官方 photofinishing 模型、RAW/metadata 前处理与原始 recipe，增加可选 MSE/L1、AdamW/学习率配置，以及由 ARIS-Code + Naive 驱动的串行 campaign。未接入历史增强方案、风格迁移或教师监督。

验证当前快慢系统请先阅读 [部署与真实闭环验收使用手册](docs/validation-manual_CN.md)：先验实际节点原生执行，再验真实 Naive 四轮反馈链。手册说明受控初始化、宿主 JSON 接入和 W 的通过标准；下面保留基础接口示例。

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

HDF5 缓存复用前会检查输入、GT、metadata 的实际路径、配对文件清单和前处理设置；旧 `COMPLETE` 标记本身不再足以证明缓存匹配。validation loss 按图像数量加权，避免尾批次影响曲线。PSNR 选优规则不变。

## 结果驱动的研究

编辑 `configs/baseline.example.yaml` 填好真实 train/dev 路径后：

```bash
python -m tm_research.cli campaign init --config configs/baseline.example.yaml --campaign-dir campaigns/pilot --max-trials 4 --no-gain-limit 3 --min-delta 0.01 --wait
python -m tm_research.cli campaign next --campaign-dir campaigns/pilot
```

`next` 返回已有真实 dev 结果和下一轮需要回答的问题。Naive 经 ARIS-Code 读取这些结果，写入 proposal JSON，再调用 `campaign submit`。每次提交必须包含假设、所依据的结果及可执行 recipe。控制器运行、比较和记录；达到预算或连续无收益条件后停止。完整命令与 ARIS 任务见 [docs/tm_research_task.md](docs/tm_research_task.md)。

本地模拟 proposer 只能证明控制器能消费结果并执行不同方案；真实 Naive 的研究行为需要 live 服务联调。

`raw_best` 保存最高实测分数；`best` 仅在增幅严格超过冻结的 `min_delta` 后更新并清零无收益计数，冻结候选也使用这个有效 `best`。示例 `0.01 dB` 是工程初值，真实 pilot 前应结合实际有用增益与评测波动确定，并在初始化时固定，不能根据后续结果临时更改。同一权重重复 DEV 只能估计评测波动；训练随机性的判断还需要配对 seed 复训。完整的 baseline＋3 次真实提案启动与证据要求见 [ARIS pilot](docs/aris-campaign-task.md#real-naive-pilot-baseline-plus-three-proposals)。

## 冻结与授权后的测试

```bash
python -m tm_research.cli campaign finalize --campaign-dir campaigns/pilot
```

先依据 `next_action` 冻结候选。当前 pilot 禁止 TEST；`final-test` 仅供另有明确 TEST 授权的冻结流程使用。日常研究 prompt 不包含测试分数。可选确认模块对冻结 baseline/winner 进行配对 seed 复训；仅 GTM/LTM 更新的归因实验仍是独立任务，当前不声称已确认 PSNR 提升。

## 原始全 ISP 推理

官方 `main/demo.py` 仍可加载 denoising、photofinishing、enhancement 权重执行完整链路。本次只修改训练、评价及研究控制入口，不修改模型结构或全 ISP demo。

## 验证范围与入口

- [数据与科学证据可靠性修复](docs/reliability-fixes-20261005.md)
- [本轮 review 修复与实测记录](docs/review-fix-verification.md)
- [二次 review 阈值修复与验证](docs/rereview-followup.md)
- [Naive / ARIS 服务设置](docs/naive-setup.md)
- [研究合同](docs/engineering-contract.md)
- [上一阶段历史验证](docs/verification.md)

源码及模型许可遵循原仓库 LICENSE.md。


## 可选快慢科研模式

新增 `campaign init --control ... --profile ...`、提交时独立 `--decision ...`、版本化逐图反馈、事件触发的复盘及冻结配方确认。科学 YAML、严格三键 proposal 和原有选优阈值保持兼容。作业先 prepare，再保存 active 引用，再启动；未知活性禁止重复训练，确认/校准预算预留，诊断失败不覆盖有效 DEV 分数。

可执行命令和授权模板见 [fast-slow-research.md](docs/fast-slow-research.md)。模板中的空资源不是授权；真实数据/GPU、Naive＋ARIS transcript、多 seed 收益和独立泛化证据仍待实际运行。合成测试只验证工程行为。

首次注册的可执行确认计划在任务开始前冻结为主计划；后续计划披露为探索性结果，不能替换主计划的 Q 结论或最终权重规则。profile 的事前来源按准备运行时保存的内容判断；仅路径相同不足以证明分组未改。确认报告与 campaign Q 共用实际结果校验，原生协议或配置失效会阻止结论提升。
