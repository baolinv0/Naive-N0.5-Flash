# Modular Neural ISP Phase 1：对照自动科研方案的代码审查

- 审查日期：2026-09-29
- 仓库：`baolinv0/Naive-N0.5-Flash`
- 固定版本：`53fa505c5ecba44f89f21fc8fbf3c090d2ab25c8`
- 范围：`examples/modular_neural_isp_phase1` 的源码、文档、测试；并对照上一版《Architecture and Research Contract》《Implementation Plan》。
- 操作：只读审查；未向 GitHub 提交修改。
- 结论：**实现了缩减后的原始 ISP baseline 工程与 Naive 协议适配；未完成 ARIS + Naive 驱动的、自适应 PSNR 优化研究系统。**

## 1. 为什么不能称为完整实现

原方案要求：Naive 提假设 → 提交受控训练 recipe → ARIS 执行 → 固定 evaluator 返回 dev PSNR → 比较 → 下一假设 → 多 seed 确认 → 冻结后最终测试。

本实现的 `docs/engineering-contract.md` 则明确写了“original ... baseline only”，固定 loss、optimizer、schedule，并排除“new losses/search”。这是**实施范围被缩小**，不是仅仅缺 GPU 导致完整功能未验证。

目前可以执行：

```
人工准备 baseline YAML
  → run_baseline
  → 原 train.py（内部验证与选 best）
  → 可选 test.py 独立进程重载
  → state.json / metrics.json / results.jsonl
```

Naive adapter 单独提供“原生模板 ↔ Chat Completions/SSE/tool_calls”。外部 ARIS-Code 被文档指定为宿主，但这里没有交付任务专用的 adaptive campaign、实验比较与决策循环。

另一方面，**ARIS-Code 是上游已有的独立宿主**。使用它替代原方案的自写 Python agent host 是合理的实现选择，不应因为目录或类名不同就判不合格。缺少的是本任务的功能和验证证据，不是必须重新写宿主。

## 2. 已实现的部分

1. `baseline_utils.paired_files` 按场景 stem 配对输入、GT、metadata，处理 `_denoised` 后缀，检查重复和集合不匹配；不再依赖目录枚举顺序。
2. `dataset.py` 修复 HDF5 缓存句柄被 `with` 关闭后再次复用的问题。
3. `train.validate` 用实际 batch_count 修正分母，同时新增逐图 PSNR；最后一个 epoch 也验证，支持短训练。
4. `runner.py` 创建不同 exp_NNN 输出目录，启动后台训练、保存命令、日志和终态；训练非零退出会标失败并跳过 test。
5. 测试入口重新加载模型权重，保存逐图 CSV、指标和输出图像。
6. Naive adapter 实现模型加载入口、官方 chat template 调用、历史参数字典化、工具标记转换、SSE 心跳；测试使用模拟生成器。

这些功能值得保留。没有必要推倒 baseline 或重写全部 ARIS。

## 3. 原方案八项任务的覆盖

| 原方案任务 | 当前代码 | 判断 |
|---|---|---|
| 合同、类型、preflight | YAML 路径解析、有限字段检查；无正式 campaign/search/data/model/profile 合同 | 部分 |
| Manifest、前处理、数据集 | 按 ID 配对和 HDF5 重读修复；无场景组跨 split 审核、完整缓存确认、P512 统一协议 | 部分 |
| 固定独立 evaluator | test 单独进程、weights-only 重载；但 best 在训练进程选，评测和训练使用同一代码目录，主指标和分辨率未统一 | 部分，不等于原计划的独立裁判 |
| 训练 recipe 与恢复 | 原 recipe 可运行；没有统一 MSE/L1/optimizer 搜索接口，没有完整训练状态恢复 | baseline 子集 |
| ARIS 队列、workspace、预算 | 自有 Popen runner、独立输出和状态文件；没有接 canonical queue、独立代码快照、GPU 池和 campaign 预算 | runner 子集 |
| Naive backend / tools | 兼容协议服务器和模拟测试；没有实测部署 profile 或 live 工具闭环证据 | 适配部分实现、真实使用待验证 |
| 自适应研究循环 | 没有 proposal→result→next proposal 的实现、停止规则或对照判定 | 未实现 |
| 多 seed / 封存 test / TM 归因 | 没有正式比较、finalize、置信区间或 GTM/LTM-only 对照 | 未实现 |

## 4. 需要先处理的关键问题

### F1. 研究入口只支持 baseline，不能按原计划进行受控优化

**位置：** `tm_research/runner.py:48–53,71–107`；`tm_research/cli.py`。

`normalize_config` 不接受 learning_rate、loss、optimizer 等字段。直接加入 `learning_rate: 0.0001` 会得到：

```
Unsupported config fields: ['learning_rate']
```

CLI 只有 baseline/commands/status/wait/report，没有 research/resume/finalize。它能重复运行原始训练，但没有实现统一的训练方案搜索。

**边界：** upstream `train.py` 自身仍有一些学习率/损失 CLI 参数；不能说整个原训练代码完全不可调。问题是当前自动化 runner 的公共接口没有暴露和约束这些参数，也没有迭代研究逻辑。

**最小修正：** 增加受控 recipe 配置（原配方、MSE、L1、学习率、optimizer），模型结构不变；每次保存实际生效 recipe。先做配置搜索，不必立即开放任意源码编辑。

### F2. 保存的 best checkpoint 与要优化的逐图 PSNR 不一致

**位置：** `photofinishing/train.py:278–286`。

```
best_model_idx = log['val_psnr'].index(max(log['val_psnr']))
# 但最终报告 mean_psnr 来自 val_mean_psnr[best_model_idx]
```

虽然记录了逐图平均 PSNR，选 best 仍然用旧 batch 聚合 PSNR。本实现的小合同明确允许保留此行为；但它不满足原计划的主指标。

**最小数值复现（不是实际模型成绩）：**

| checkpoint | 两张图的 MSE | 逐图 PSNR 平均 | batch MSE 后 PSNR |
|---|---|---:|---:|
| A | 0.01、0.0001 | 30.0000 dB | 22.9671 dB |
| B | 0.002、0.002 | 26.9897 dB | 26.9897 dB |

按新目标应选 A，现有选取逻辑会选 B。实际 helper 计算及相同选择表达式已复现这一反例；没有完整运行 train.py 的两个学习 checkpoint。

**最小修正：** 统一一个权威 `mean_per_image_psnr`，用于验证、选 best、报告和候选比较；旧 batch 指标仅作诊断。并列时仍选较早 checkpoint。增加以上排序反例和不同 batch 划分的回归测试。

### F3. 每次训练后自动跑 test；测试分辨率也不是固定 P512

**位置：** `runner.py:98–106,183–197`；`photofinishing/test.py::test_net`；`docs/tm_research_task.md`。

只要 YAML 有 test，每个 baseline run 都会调用 test.py，并将 test_metrics 放进 get_result/report 的返回中。若把它直接复用于 agent 搜索，原本应封存的 test 结果将反复暴露。**这不是证明已发生数据泄漏，而是当前流程不具备封存测试语义。**

此外，训练/validation 使用 in_size 方图；test 保留高宽分别 /4 的原协议，runner 不传固定 P512 评测选项。例：原图 256×256，validation 可为 256×256，test 为 64×64；原图 4000×3000 时 test 则为 1000×750。它们不必错误，但不能混成同一排行榜指标。

**最小修正：** 日常 trial 只返回 dev 指标；test 从常规配置移出，冻结候选后另行触发。主协议明确 P512，quarter/native 作为单独诊断协议。不要把 validation 与不同协议的 test 分数直接相减。

### F4. completed 没有验证 checkpoint 与指标有效性

**位置：** `runner.py::_worker`，尤其 `197` 行附近。

训练 exit=0 后读取 metrics.json 就能推进到 completed。在允许不配置 test 的路径中，没有验证 checkpoint 是否存在、能否加载、数值是否有限。

**实际故障注入：** 以替代训练脚本正常退出，写入不存在的 best_checkpoint 路径和 NaN mean_psnr；不配置 test。真实 runner 返回：

```
status = completed
checkpoint_exists = false
mean_psnr_is_finite = false
```

这只说明结果验证缺失，**不意味着现有仓库伪造了真实实验结果**。

**最小修正：** 至少检查退出码、权重存在/可重载、覆盖数量和分数有限；独立 evaluator 重算后再进入“可比较结果”。执行完成和科学判定分开：completed 不能等于 improved。初期可以只有 valid/no_gain/improved，没必要增加复杂评分体系。

### F5. Naive 工具转换没有完整处理截断和 schema

**位置：** `tm_research/naive_adapter.py:44–65` 及生成结束的 finish_reason 逻辑。

探针证明：未闭合 `<think>` 后的工具块仍被解析为 tool_calls；缺少 required 参数仍输出空字典；期望 object 的参数可以输出整数 2。截断工具块被保留为普通内容，而 HTTP 最终只报告 stop/tool_calls，不能体现 length 截断。

适配器本身不执行工具，真实外部宿主可能再校验；但当前 bridge 没有证明能阻止这些无效调用向外派发。尤其短输出上限导致的不完整 reasoning/tool call 是常规部署需处理的情况。

**最小修正：** 把真实 generation 终止原因传到协议层；区分 reasoning 与有效回答边界；对已声明工具做 required/type 校验；截断不产生可执行调用。不需要扩展成安全攻防框架。

### F6. 独立输出目录不等于独立实验 workspace 或可恢复调度

**位置：** `runner.py::commands_for_run/run_baseline/get_result/wait_for_result`；`dataset.py::__init__`。

每个 exp 有独立输出，但所有实验仍从相同 repo_dir 读取训练源码，并继承调用环境中的 GPU 可见性。没有 ARIS queue/GPU 分配。共享 HDF5 缓存只凭目录存在就复用，没有准备完成确认；首次并发或预处理打断后可能读不全缓存。worker 被终止后，状态查询也不会自动核对进程并恢复；文档正确说明失败后新 run，而 `--load` 只恢复模型权重。

这些是静态路径结论；本轮没有实际 GPU 争用或缓存中断复现。

**最小修正：** 首版先单实验 GPU 串行运行，先单独完成缓存；要并行时复用 ARIS queue、明确 GPU 池、冻结每 trial 的有效源码/recipe。需要续训时保存 optimizer/scheduler/epoch/RNG；status.json 不能替代恢复逻辑。

### 补充：PSNR helper 仍与原数值合同有差异

`baseline_utils.py:37–40` 使用输入 dtype 的 reduction，无 finite 检查或 MSE epsilon；实测完全匹配返回 inf，NaN 输入返回 nan。与原合同的 FP64、epsilon=1e-12、有限数值规则不一致。建议在统一 evaluator 时一起修，不必为此单独建设大套流程。

## 5. 实际验证范围

### 本轮独立执行

环境：Python 3.13.5、PyTorch 2.10.0+cpu、无 CUDA。

通过 GitHub 连接器读取固定版本完整源码，将所需模块和两份原测试文件保存到本地。网络环境未能克隆完整仓库，因此未声称完成上游环境安装。执行：

```
PYTHONPATH=<review-source-root> python -m pytest -q \
  tests/test_naive_adapter.py tests/test_runner.py
```

**结果：11 passed（6 adapter + 5 runner）。** 这些测试分别使用注入生成器、mock Transformers，以及替代训练/测试脚本。

随后执行 `review_probes.py`，得到前述选择反例、字段拒绝、无效 completed、解析器和数值行为。原始输出在 evidence/。

### 仅阅读仓库记录，未独立复现

仓库 verification.md 记录了 15 项测试、官方权重 CPU 前向、原配方 2 epochs 合成训练、独立测试重载和全 ISP 合成 demo。本轮未完整重跑这些集成流程，也未执行另外 4 项依赖 upstream train/dataset 的测试。

### 未验证

真实 Naive 权重部署、live ARIS-Code 工具调用、真实相机数据 B0/MSE/L1 训练、GPU 并行、完整自适应实验和最终 PSNR 提升。不能以 mock 正常返回替代上述证据。

## 6. 建议工程师下一批交付

### PR-A：把 PSNR 裁判和 best 选择统一

固定 P512、逐图平均、单独 dev/test 路径、best 按同一指标选；增加排序反例和结果重载测试。保留旧协议做显式命名的诊断，不混用。

### PR-B：开放最小训练 recipe

在 runner 接收/传递 learning_rate、固定集合的 loss family、optimizer；加入原 recipe/MSE/L1 同初始化同预算控制组。每项只改变一个因素；模型结构不改。

### PR-C：真正由 ARIS-Code + Naive 做下一步决策

沿用现有兼容服务路线，补项目专用任务 skill 或受控 campaign controller。保存 hypothesis、parent/result ID、effective config、dev score、decision。运行一个有限预算 pilot，至少有三次“读取真实结果后改变下一实验”的动作。预写网格可以做 sanity，但不算自适应研究。

### PR-D：最小有效结果与恢复

检查 checkpoint/指标，分离执行状态与改进判定，限制预算和连续无收益停止；先串行、后 ARIS queue 并行。完整 resume 与多 seed/final test 按实际后续阶段补齐。

## 7. 验收建议

- **原始 ISP baseline Phase 1：** 已有实质代码与被清楚标注的联调记录；可以作为后续底座。当前独立复跑证明了 adapter/runner 的 11 项单测，不覆盖全部真实环境。
- **ARIS × Naive 自动研究工程：** 不能验收为完成；需要能提出不同 recipe、读取可信结果、迭代并停机的实际闭环。
- **PSNR 提升：** 暂无 baseline/candidate 同协议同预算比较与多 seed/holdout 证据，不能判定。

**核心差距不是“还没买够 GPU”，而是当前交付有意只实现 baseline 调用，未实现研究方案变化与证据驱动迭代。**

## 源码与证据链接

以下链接均指向审查固定版本。

- [README / Phase1 范围](https://github.com/baolinv0/Naive-N0.5-Flash/blob/53fa505c5ecba44f89f21fc8fbf3c090d2ab25c8/examples/modular_neural_isp_phase1/README.md)
- [缩减后的 engineering contract](https://github.com/baolinv0/Naive-N0.5-Flash/blob/53fa505c5ecba44f89f21fc8fbf3c090d2ab25c8/examples/modular_neural_isp_phase1/overlay/docs/engineering-contract.md)
- [runner.py](https://github.com/baolinv0/Naive-N0.5-Flash/blob/53fa505c5ecba44f89f21fc8fbf3c090d2ab25c8/examples/modular_neural_isp_phase1/overlay/tm_research/runner.py)
- [cli.py](https://github.com/baolinv0/Naive-N0.5-Flash/blob/53fa505c5ecba44f89f21fc8fbf3c090d2ab25c8/examples/modular_neural_isp_phase1/overlay/tm_research/cli.py)
- [train.py best 选择](https://github.com/baolinv0/Naive-N0.5-Flash/blob/53fa505c5ecba44f89f21fc8fbf3c090d2ab25c8/examples/modular_neural_isp_phase1/overlay/photofinishing/train.py#L274-L286)
- [test.py](https://github.com/baolinv0/Naive-N0.5-Flash/blob/53fa505c5ecba44f89f21fc8fbf3c090d2ab25c8/examples/modular_neural_isp_phase1/overlay/photofinishing/test.py)
- [dataset.py](https://github.com/baolinv0/Naive-N0.5-Flash/blob/53fa505c5ecba44f89f21fc8fbf3c090d2ab25c8/examples/modular_neural_isp_phase1/overlay/photofinishing/dataset.py)
- [baseline_utils.py](https://github.com/baolinv0/Naive-N0.5-Flash/blob/53fa505c5ecba44f89f21fc8fbf3c090d2ab25c8/examples/modular_neural_isp_phase1/overlay/photofinishing/baseline_utils.py)
- [Naive adapter](https://github.com/baolinv0/Naive-N0.5-Flash/blob/53fa505c5ecba44f89f21fc8fbf3c090d2ab25c8/examples/modular_neural_isp_phase1/overlay/tm_research/naive_adapter.py)
- [Naive / ARIS-Code setup](https://github.com/baolinv0/Naive-N0.5-Flash/blob/53fa505c5ecba44f89f21fc8fbf3c090d2ab25c8/examples/modular_neural_isp_phase1/overlay/docs/naive-setup.md)
- [runner 范围和恢复边界](https://github.com/baolinv0/Naive-N0.5-Flash/blob/53fa505c5ecba44f89f21fc8fbf3c090d2ab25c8/examples/modular_neural_isp_phase1/overlay/docs/tm_research_task.md)
- [原有测试目录](https://github.com/baolinv0/Naive-N0.5-Flash/tree/53fa505c5ecba44f89f21fc8fbf3c090d2ab25c8/examples/modular_neural_isp_phase1/overlay/tests)
- [仓库作者验证记录](https://github.com/baolinv0/Naive-N0.5-Flash/blob/53fa505c5ecba44f89f21fc8fbf3c090d2ab25c8/examples/modular_neural_isp_phase1/overlay/docs/verification.md)
- [ARIS 上游 README：独立 ARIS-Code 宿主说明](https://github.com/wanshuiyin/Auto-claude-code-research-in-sleep/blob/2132036060e03e8d0df69a4b21e5971819c0c2d6/README.md)