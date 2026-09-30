# Modular Neural ISP：修复分支二次审查

- 日期：2026-09-29
- 分支：`fix/tm-adaptive-research`
- 审查版本：`c180078360c6716d7aa6a45ae8b209739cb21225`
- 对照：上一轮审查的 Batch A（统一裁判）、Batch B（开放 recipe）、Batch C（反馈驱动研究循环）。
- 操作：只读 GitHub；本地测试选定模块；没有提交、修改或合并远程代码。

## 1. 总结与验收判断

**本轮有实质性修改，不能再沿用“只有 baseline runner、没有实验搜索入口”的旧结论。**

- Batch A：主要代码问题已修复。逐图 PSNR、best 选择、P512 dev 重载、有效性检查和日常 test 隔离已接通。
- Batch B：最小配置搜索已实现。original/MSE/L1、Adam/AdamW、learning_rate、weight_decay 可从配置传入实际训练入口，并记录生效 recipe。
- Batch C：串行 campaign/controller、结果引用、去重、预算/无收益停止、冻结与独立 final-test 已实现；ARIS-Code 作为外部真实 proposing agent 的任务入口已给出。**真实 Naive + ARIS 连续执行尚未验证。**

建议阶段验收为：**配置搜索工程主体通过本次局部复核，可以进入真实 Naive/ARIS 的单卡小规模联调；不可宣称完整自动科研系统或真实数据 PSNR 提升已经验收。**

本次覆盖路径未发现新的已证实 P0 阻塞。以下区分真正未修复问题、待实测能力与后续阶段，不把缺少复杂基础设施重新作为串行 pilot 的阻塞。

## 2. 上次意见逐项闭环

| 上次问题 | 当前实现 | 本次判断 |
|---|---|---|
| best checkpoint 仍按旧 batch PSNR 选择 | `training()` 调用 `best_checkpoint_index(log['val_mean_psnr'])`；validation 的 `psnr` 成为逐图平均别名 | 已修复；排序反例选 A，且 ties 选早期 checkpoint |
| 训练变量未暴露 | runner 允许并传递四项 recipe；`recipe_utils.py` 实现 MSE/L1 与 Adam/AdamW | 已实现最小搜索空间，不只是文档声明 |
| 每轮自动跑 test | 普通 run 始终独立重载 DEV；保存运行配置前移除 test；final-test 须先 finalize | 已修复自动调用路径；不是 OS 级数据访问隔离 |
| dev/test 分辨率口径不一 | 显式 eval_size，默认 P512；dataset 与 test 共用 load_image_pair，PSNR 共用同一 helper；quarter/full 单独命名 | 已统一当前主协议；P256 smoke 不能充当 P512 真数据证据 |
| completed 等于有效结果 | checkpoint 存在、独立进程重载退出码、finite score、样本 coverage、protocol 检查；valid 与执行状态分离 | 已修复；invalid run 不成为 best |
| Naive 未闭合 reasoning/tool 与参数问题 | GenerationResult 传递 length/stop 与 starts_in_think；required/type 检查；长度截断不输出工具调用 | 针对性回归通过；真实模型服务未运行 |
| 缺少下一轮逻辑 | campaign init/next/submit/status/wait/finalize；提案要求引用已记录 run 与分数；禁止重复 recipe 和固定条件改动 | 控制器实现；模拟 proposer 通过；真实 proposer 待联调 |
| HDF5 半成品缓存 | COMPLETE 标记只在构建完成后写入；缺标记时重建 | 代码已处理；本轮未重跑完整 upstream 缓存测试 |
| 并行 GPU/完整训练恢复 | 当前明确串行，不恢复 optimizer/scheduler/RNG；只有宿主重启后读取存活 worker 的状态 | 尚未实现且文档已明确，不应冒充 resume 全功能 |

### 2.1 指标选择的具体验证

数值反例（不是训练成绩）：

| checkpoint | 两图 MSE | 逐图 PSNR 均值 | pooled-MSE PSNR |
|---|---|---:|---:|
| A | 0.01 / 0.0001 | 30.000000 | 22.967086 |
| B | 0.002 / 0.002 | 26.989700 | 26.989700 |

新版 `best_checkpoint_index` 对新主指标选择 A。独立诊断验证了分 batch、换序、单图完全一致时 120 dB、非有限预测不会生成合法 summary。静态核验确认 `training()` 的实际调用已切到新 helper，而不是仅增加孤立函数。

### 2.2 recipe 确实接到训练

`commands_for_run` 将 recipe 传入 `--loss-family / --optimizer / --learning-rate / --weight-decay`。`PixelLoss.forward` 对最终 sRGB 计算 `F.mse_loss` 或 `F.l1_loss`；`make_optimizer` 选择真实 PyTorch Adam/AdamW。original 分支继续调用原 PhotofinishingLoss。

当前故意固定 schedule 为 CosineAnnealingLR，不支持组合损失权重搜索、结构修改或代码搜索。这是合理的最小 action space，并非必须立即实现完整原设计所有候选才能开始 pilot。

### 2.3 controller 确实依赖既有结果

`next_proposal` 返回实际 trial 历史、best、错误与 decision；`_validate_proposal` 校验 based_on 引用的 run 和分数，recipe 只能改四个字段。`_collect` 依据独立 DEV 返回值更新 best/no_gain；固定数据、初始化、seed、epochs、batch、尺寸与验证频率不能随提案改动。达到预算或连续无收益后停止；finalize 后拒绝 submit；final-test 不更改 DEV 选择。

**结果引用验证只能保证引用的记录真实，不能独立证明模型产生了有价值的科研推理。**真实模型身份和反馈驱动提案仍需要 live transcript。

## 3. 当前最重要的缺口：真实 Naive/ARIS 尚未进入证据链

`docs/review-fix-verification.md` 明确写明：三次提案是实施助手逐次读取 CPU 合成实验结果后编写，不是运行 Naive 权重或 ARIS-Code 后得到。应保留这个准确边界。

当前已验证/已有证据的是：

```
实施助手或模拟 proposer
→ proposal.json
→ controller
→ 训练/独立 dev 重载
→ 结果/下一提案
```

需要补的最终连接是：

```
真实 Naive 服务
→ ARIS-Code 的文件/Bash 工具
→ task 文档
→ campaign next/submit/wait
→ 真实结果
→ Naive 自主提出下一配置
```

使用现成 ARIS-Code 作为宿主是合适的，不要求另写 Python host，也不因为没有复制 ARIS 内部 run_state/queue 文件就认定完全没有使用 harness。当前 task 文档提供了调用路径，但路径存在不等于 live 集成已经跑通。

## 4. 新确认的非阻塞改进点：有效增益阈值

位置：`tm_research/campaign.py::_collect`。

当前判断仅为 `new_psnr > best_psnr`。诊断注入 best=25.0、candidate=25.000001 后，结果为 improved，且 no_gain_count 从 2 重置成 0。这是对现有逻辑的刻画，不是真实性能提升。

当存在正常评测/训练数值波动时，极小正差会被文字解释成有效改进。建议区分：

- raw best：仍可记录最高实际分数；
- meaningful improvement：超过冻结的 min_delta 后才重置无收益计数并称为有效候选。

min_delta 应依据重复评测波动确定；例如设置 0.01 dB 作为初始工程值，但不能把该示例当作已确认统计阈值。多 seed 的性能确认仍另做。本项不阻止小规模流程联调。

## 5. 边界与后续阶段，不升级为本轮 P0

### 5.1 配置搜索而非任意代码搜索

目前训练/evaluator 共用固定工作源码，训练进程仍读取 dev GT 并做 best selection，然后另进程重载 best。它不是原设计中的强隔离裁判服务。但在明确只开放配置且固定 model/evaluator 的本期范围下，可先运行 pilot。扩大为任意代码修改前再处理独立源码快照和评测访问边界。

### 5.2 固定预算不是完整恢复

每个 trial 从同一个初始化重新训练，与此前“从不断改善的 best 随意继续训”相比，预算控制已合理。现有 60 epoch 与未来 600 epoch 的 cosine T_max 语义仍不同；尚未实现续训，不能把短训 checkpoint 接长训声称是同一完整轨迹。宿主返回后继续读取一个尚存 worker 的结果，不等于 worker 被杀后的自动恢复。

### 5.3 真实数据与 TM 归因

现存合成 train/dev/test 构造高度重合，作者已明确标为工程联调。不能把本次合成提升当作真实 S24 泛化收益，也不能把 photofinishing 总分上涨直接归因于 GTM/LTM。真实 P512 数据、分组切分、multi-seed、冻结后 test、TM-only attribution 仍待后续阶段。

### 5.4 当前不需要加大系统

不建议此时要求工程师重新建设多 GPU scheduler、复杂权限平台或更多 agents。先完成真实 Naive + ARIS-Code 的少量串行实验，更能判断该系统能否真正工作。

## 6. 本次实际执行的测试

环境：Linux / Python 3.13.5 / PyTorch 2.10.0+cpu / pytest 9.0.2；没有 CUDA。

获取方式：GitHub connector `fetch_file` 读取固定版本，将选定源码与测试转存本地。普通 Git/HTTP 下载在此运行环境 DNS 不可用。测试针对转存的原可执行逻辑；部分复制文件省略纯注释。没有完整安装 Samsung upstream 或下载模型。

| 实际运行 | 结果 | 含义 |
|---|---|---|
| 新版原有 `tests/test_runner.py` | 16 passed | 进程、配置、无效结果、dev/test 路由；训练/重载为仓库自带替代脚本 |
| 新版原有 `tests/test_campaign.py` | 6 passed（分两组各 3） | 多轮反馈、引用、限制、停止、冻结、状态读取；proposer 为模拟 |
| 本次新增针对性诊断 | 15 passed | 指标数学、batch不变性、tool parser、SSE length/stop、协议检查；其中1项刻画 min_delta 缺失 |
| 合计 | 37 passed | 局部代码/协议验证，不是完整 ISP 或真实 Naive 验证 |

一次初始组合运行超过工具命令时限；定位显示在等待子进程而非已证实死锁。拆分重跑后以上各组全部完成。原始超时诊断单独保留，不计为模型代码失败。

没有独立重跑仓库宣称的完整49项测试，没有执行完整真实 ISP 短训，没有重新计算仓库合成权重成绩，没有真实 Naive/ARIS 服务、GPU训练、真实数据或多seed结果。本报告不把作者已有验证记录冒充本轮独立实测。

## 7. 工程师下一步验收建议

只进行一个“baseline + 3个后续提案”的单GPU串行 pilot：

1. 真实 Naive 服务可读写工具，ARIS-Code 明确使用该 executor，不静默换模型。
2. baseline 由当前 worker 实际运行、DEV 实际重载。
3. Naive 读取 campaign next 的真实分数与 decision，生成后续 proposal；每次只在结果返回后提交下一项。
4. 保存真实宿主转录、模型配置、proposal、run日志与指标；trial次数或无收益条件满足后停止。
5. 冻结后才按操作员授权运行 final-test。未涨分仍可验收流程，不要求凑出提升。

成功后再扩真实数据完整训练、多seed和TM归因。无需为了本轮复核重新设计整个系统。

## 8. 来源

所有链接指向本次固定 commit；不是当前 main 的可变状态。

- [runner](https://github.com/baolinv0/Naive-N0.5-Flash/blob/c180078360c6716d7aa6a45ae8b209739cb21225/examples/modular_neural_isp_phase1/overlay/tm_research/runner.py)
- [campaign](https://github.com/baolinv0/Naive-N0.5-Flash/blob/c180078360c6716d7aa6a45ae8b209739cb21225/examples/modular_neural_isp_phase1/overlay/tm_research/campaign.py)
- [CLI](https://github.com/baolinv0/Naive-N0.5-Flash/blob/c180078360c6716d7aa6a45ae8b209739cb21225/examples/modular_neural_isp_phase1/overlay/tm_research/cli.py)
- [Naive adapter](https://github.com/baolinv0/Naive-N0.5-Flash/blob/c180078360c6716d7aa6a45ae8b209739cb21225/examples/modular_neural_isp_phase1/overlay/tm_research/naive_adapter.py)
- [指标与前处理](https://github.com/baolinv0/Naive-N0.5-Flash/blob/c180078360c6716d7aa6a45ae8b209739cb21225/examples/modular_neural_isp_phase1/overlay/photofinishing/baseline_utils.py)
- [train](https://github.com/baolinv0/Naive-N0.5-Flash/blob/c180078360c6716d7aa6a45ae8b209739cb21225/examples/modular_neural_isp_phase1/overlay/photofinishing/train.py)
- [test](https://github.com/baolinv0/Naive-N0.5-Flash/blob/c180078360c6716d7aa6a45ae8b209739cb21225/examples/modular_neural_isp_phase1/overlay/photofinishing/test.py)
- [recipe](https://github.com/baolinv0/Naive-N0.5-Flash/blob/c180078360c6716d7aa6a45ae8b209739cb21225/examples/modular_neural_isp_phase1/overlay/photofinishing/recipe_utils.py)
- [dataset](https://github.com/baolinv0/Naive-N0.5-Flash/blob/c180078360c6716d7aa6a45ae8b209739cb21225/examples/modular_neural_isp_phase1/overlay/photofinishing/dataset.py)
- [runner测试](https://github.com/baolinv0/Naive-N0.5-Flash/blob/c180078360c6716d7aa6a45ae8b209739cb21225/examples/modular_neural_isp_phase1/overlay/tests/test_runner.py)
- [campaign测试](https://github.com/baolinv0/Naive-N0.5-Flash/blob/c180078360c6716d7aa6a45ae8b209739cb21225/examples/modular_neural_isp_phase1/overlay/tests/test_campaign.py)
- [ARIS任务](https://github.com/baolinv0/Naive-N0.5-Flash/blob/c180078360c6716d7aa6a45ae8b209739cb21225/examples/modular_neural_isp_phase1/overlay/docs/aris-campaign-task.md)
- [实施方验证边界](https://github.com/baolinv0/Naive-N0.5-Flash/blob/c180078360c6716d7aa6a45ae8b209739cb21225/examples/modular_neural_isp_phase1/overlay/docs/review-fix-verification.md)