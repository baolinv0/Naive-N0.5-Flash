# Modular Neural ISP Phase 1 — Code Review

**Reviewed commit:** `53fa505c5ecba44f89f21fc8fbf3c090d2ab25c8`  
**Date:** 2026-09-29

## 结论

当前实现可以作为 Phase 1 baseline 工程底座，但**尚未实现完整的 ARIS + Naive 自动科研闭环**。现阶段实现的是：

- 原始 Samsung photofinishing baseline 的训练/验证/测试链路；
- 场景配对、HDF5 重读、验证统计等基础修正；
- 后台 runner、状态与日志持久化；
- Naive 原生工具格式到 OpenAI-compatible streamed tool calls 的适配。

仍缺少的核心闭环是：

```text
Naive 提出训练方案
    ↓
真实执行实验
    ↓
读取可信 dev PSNR
    ↓
比较 / keep / reject
    ↓
基于结果提出下一实验
```

因此，本阶段不应按“完整自动科研系统已实现”验收。

---

## 1. 已实现且建议保留

### 1.1 数据与 baseline 工程

- 输入、GT、metadata 已改为按 scene stem 匹配，不再依赖目录枚举顺序；
- HDF5 cache 可检测已失效 handle 并重新打开；
- validation 已新增逐图 PSNR 平均；
- runner 能启动独立后台 worker，并保存：
  - config
  - command
  - state
  - train/test log
  - metrics
  - checkpoint 路径
- test 会重新加载训练得到的 checkpoint。

### 1.2 Naive adapter

`tm_research/naive_adapter.py` 已完成：

- OpenAI-style tool-call history → Naive 官方模板参数映射；
- Naive `<tool_call>` 块解析；
- SSE streamed Chat Completions endpoint；
- heartbeat；
- injected generator 测试。

这些基础设施可以继续沿用。

---

## 2. 当前与原设计的主要差距

原设计目标是：

```text
固定任务合同
→ Naive 提假设
→ 受控训练方案
→ 实验
→ 独立 evaluator
→ 可信 PSNR
→ 下一轮 hypothesis
→ freeze candidate
→ multi-seed / sealed test
```

当前实现只完成了前半段基础设施。

| 原设计模块 | 当前状态 |
|---|---|
| baseline 运行 | 已实现 |
| 数据配对修正 | 已实现 |
| Naive 协议适配 | 已实现，但仅 mock / injected generator 验证 |
| 统一可信 evaluator | 部分实现 |
| loss / optimizer / LR 搜索接口 | 未实现 |
| ARIS experiment queue 正式接入 | 未实现 |
| Naive 基于结果自适应研究循环 | 未实现 |
| keep/reject/stop rules | 未实现 |
| multi-seed confirmation | 未实现 |
| sealed test | 未实现 |
| TM-only attribution | 未实现 |

---

## 3. 最高优先级问题

### P0-1：best checkpoint 仍按旧 batch PSNR 选择

当前 `photofinishing/train.py` 中：

```python
best_model_idx = log['val_psnr'].index(max(log['val_psnr']))
```

而新的逐图 PSNR 被记录在：

```python
log['val_mean_psnr']
```

这意味着：

> 训练已经报告了正确的 mean-per-image PSNR，但最终 best checkpoint 仍可能按旧的 batch-level PSNR 选择。

这会直接影响自动优化目标。

#### 反例

两个 checkpoint：

```text
A:
MSE = [0.01, 0.0001]
mean(per-image PSNR) = 30.00 dB
pooled-MSE PSNR      = 22.97 dB

B:
MSE = [0.002, 0.002]
mean(per-image PSNR) = 26.99 dB
pooled-MSE PSNR      = 26.99 dB
```

如果目标是 mean-per-image PSNR，应选 A；旧指标会选 B。

### 建议

统一使用同一个 primary metric：

```text
mean_per_image_psnr
```

用于：

- validation
- checkpoint selection
- candidate comparison
- report

旧 batch PSNR 可以保留为 diagnostic metric，但不能参与 best selection。

---

### P0-2：当前 runner 不能真正执行 Naive 提出的训练方案

`tm_research/runner.py` 的配置白名单目前主要只有：

- epochs
- batch_size
- in_size
- validation_frequency
- num_workers
- seed
- init checkpoint

没有正式暴露：

- learning rate
- objective / loss family
- loss weights
- optimizer
- scheduler

因此当前即使 Naive 提出：

> “下一轮把 LR 从 1e-4 调到 2e-4”

runner 也不能把它当作一个受控 experiment 执行。

### 建议

第一版只开放配置搜索，不开放模型结构：

```text
objective:
- original
- mse
- l1
- mse+l1
- mse+ssim

optimizer:
- Adam
- AdamW

lr:
- bounded range

weight_decay:
- bounded range

schedule:
- constant
- cosine
```

先证明闭环，再开放有限代码搜索。

---

### P0-3：test set 目前会被普通 baseline run 自动调用

当前 `commands_for_run()` 中，只要配置存在 `test` split，就会在每个实验完成后直接运行 test。

这对于 baseline 联调可以接受，但对于 AutoResearch 不合适：

> 自动研究过程中不能每一轮都把 test 结果反馈给 Naive。

否则 sealed test 会失去意义。

### 建议

拆成：

```text
train + dev evaluation
        ↓
AutoResearch loop
        ↓
freeze recipe
        ↓
finalize
        ↓
sealed test
```

默认 experiment 不允许有 test split。

---

### P0-4：训练/验证与 test 的 PSNR protocol 尚未统一

当前：

- train/validation：按 `in_size` 处理，通常是方图；
- test：默认按 upstream 行为缩到原图高宽的 1/4。

两者可以同时存在，但必须明确命名：

```text
P512_dev_psnr
PQuarter_test_psnr
```

不能把二者直接作为同一个 metric 比较。

如果本 campaign 目标定义为 P512，则 dev 与 final test 都应使用同一 protocol。

---

### P0-5：completed 不等于 valid experiment

当前 worker 的主要逻辑是：

```text
train exit code == 0
→ read metrics.json
→ optional test
→ status = completed
```

但没有足够检查：

- checkpoint 是否真实存在；
- checkpoint 是否可加载；
- PSNR 是否 finite；
- evaluator 是否覆盖预期样本数。

因此应将两个概念拆开：

```text
execution_status:
queued / running / completed / failed

scientific_status:
invalid / valid_no_gain / improved / confirmed
```

正常训练完成但没有提升，是 `valid_no_gain`，不是 `failed`。

---

## 4. Naive adapter 仍需补的正常工程行为

当前 parser 对普通 tool block 已可用，但在真实长输出中还需要处理：

- unfinished `<think>`；
- truncated tool block；
- missing required arguments；
- wrong argument types；
- generation 因 max tokens 截断时的 finish reason。

不需要扩展成复杂安全框架，只需要保证：

> 不完整工具调用不会被当成有效实验请求。

另外，当前 README / verification 也明确说明尚未运行真实 Naive 权重服务和 live ARIS 驱动，因此这部分不能标记为端到端已验证。

---

## 5. ARIS 目前还没有真正承担 harness 的实验调度职责

当前 runner 自己负责后台进程。

这可以用于 Phase 1，但正式 AutoResearch 还缺：

- experiment queue；
- GPU allocation；
- per-trial workspace；
- duplicate prevention；
- budget ledger；
- resume/reconcile；
- keep/reject decision；
- stop rules。

建议不要重写一套复杂调度系统。

下一阶段直接复用 ARIS 的：

```text
experiment-queue
run_state
iteration_log
```

当前 `tm_research.runner` 可以继续作为 photofinishing task worker。

---

## 6. 建议下一阶段只做三批工作

### Batch A — 统一裁判

完成：

1. mean-per-image PSNR 成为唯一 primary metric；
2. best checkpoint 按 primary metric 选择；
3. dev/test protocol 明确区分；
4. experiment 默认不跑 test；
5. checkpoint 存在、可加载；
6. PSNR finite；
7. 样本 coverage 正确。

验收：

```text
同一 checkpoint 重复评测一致
batch size / image order 不影响 score
排序反例选择正确
invalid result 不进入 comparison
```

---

### Batch B — 开放受控训练方案

实现统一 experiment config，例如：

```yaml
objective: mse
optimizer: adamw
learning_rate: 0.0002
weight_decay: 0.000001
schedule: cosine
epochs: 60
seed: 2026
```

要求：

- 固定模型结构；
- 固定数据；
- 固定 evaluator；
- 固定训练预算；
- B0 / MSE / L1 都通过同一 trainer。

这一步完成后，Naive 才真正有“实验 action space”。

---

### Batch C — 接上证据驱动研究循环

最小闭环：

```text
Naive
  ↓
proposal #1
  ↓
submit experiment
  ↓
dev PSNR
  ↓
Naive reads result
  ↓
proposal #2 explicitly references result #1
  ↓
next experiment
```

最少做 6-trial pilot，其中至少 3 次后续实验必须能证明：

> 新 hypothesis 是基于前一轮真实实验结果产生，而不是预先写死的 grid。

需要记录：

- proposal
- parent result
- config
- actual metrics
- decision
- next hypothesis

只有这一步完成，才可以说：

> ARIS + Naive AutoResearch loop 已跑通。

---

## 7. 本次测试与证据边界

当前仓库自己的验证记录说明：

- CPU synthetic short training 已跑；
- upstream ISP forward 已跑；
- 15 tests passed；
- Naive adapter 测试使用 injected generator；
- 尚未运行真实 Naive model service；
- 尚未运行真实 ARIS → Naive → ISP 自动实验循环；
- 尚无真实数据 PSNR 提升结论。

本次独立复查额外确认：

- runner / adapter 的单元测试设计与当前实现一致；
- best checkpoint metric 选择存在目标不一致；
- runner 配置还不支持核心训练搜索变量；
- completed 与 scientific-validity 尚未分离。

因此当前最准确的阶段命名仍然是：

```text
Phase 1:
Original Photofinishing Baseline + Naive Protocol Adapter
```

而不是：

```text
Completed AutoResearch System
```

---

## 8. 下一阶段验收条件

建议 Phase 2 只有四个硬条件：

1. **一个不同于 baseline 的 training recipe 能由 Naive 提出并真实执行；**
2. **结果由统一 dev PSNR evaluator 独立返回；**
3. **Naive 下一实验明确基于该实际结果调整；**
4. **整个过程留下可重放的 experiment history。**

先完成这四项，再考虑：

- 30-candidate campaign；
- 多 GPU；
- multi-seed；
- sealed test；
- TM-only attribution。

核心优先级是：

> **先让“假设 → 实验 → 证据 → 下一假设”真实闭环一次，再扩大规模。**
