# 科学证据可靠性修复（2026-10-05）

针对 `feat/aris-fast-slow-research-20261003`，修复起点为 `8a7902042b646962800794571f4ea03829d4b2e1`。修复范围是数据缓存、训练评测汇总、反馈来源和冻结确认的可靠性。

## 行为变化

| 问题 | 修复后的行为 |
| --- | --- |
| 相同 GT 路径下更换输入或 metadata，却复用旧缓存 | 缓存目录绑定规范化的输入、GT、metadata 路径和前处理参数；复用前检查源文件与 HDF5 清单的大小、修改时间及 manifest。旧缓存保留，单独的 COMPLETE 标记不再足够。 |
| validation loss 等权平均 batch，尾批次权重过大 | 每个 batch 的均值按图像数加权；分批方式不改变 MSE/L1 汇总，逐图 PSNR 选优继续使用原规则。 |
| CUDA 前向计时未等待设备执行完成 | 在指定 CUDA 设备上同步后开始计时，前向完成后同步再结束；CPU 路径无需同步。 |
| profile 引用存在就满足必需的 groups | 必须有可读的实际标签及完整匹配的分组测量；缺失、损坏或无标签的 profile 进入诊断，原生有效分数仍保留。 |
| 同一路径修改 profile 后仍被称为事前证据 | prepare 保存 profile 内容及 ROI 定义到独立观察元数据；当前内容、启动快照及已有运行快照必须一致。无法证明事前内容的历史或修改后分析标为 retrospective，旧反馈版本保留。 |
| 后续有利确认计划覆盖先前不利结论 | 首个可执行计划在任务启动前冻结为 primary；后续计划作为 exploratory 全部披露。Q 与最终权重规则由 primary 决定，不能通过显式传入后续目录改选。 |
| 默认配置省略值导致合法确认权重被拒绝 | 两侧使用有效默认值比较；省略 `in_size=512`、`validation_frequency=4` 等同显式默认值，真实配置变化仍拒绝。 |
| 确认报告信任缓存任务分数，与 Q 判断不一致 | 共用实际配置、原生状态、协议、样本数和分数校验，并绑定冻结父配置；清单异常、无效配对和嵌套 JSON 错误披露为证据不足。其他有效配对仍计入覆盖；可选 CSV 缺失不抹去有效原生指标。 |
| 最新反馈索引损坏后回退旧摘要 | 决策与 campaign 共用索引、版本、run、路径和摘要身份校验；损坏、丢失或互相矛盾的发布信息不能恢复旧就绪状态。 |

## 验证

Python 3.12.14，PyTorch 2.5.1+cpu，torchvision 0.20.1+cpu；官方 ISP 依赖源码固定于 `5a845f673edfdf92de18dcfc20d204f79f1ba38b`。新回归先复现错误，再验证修复；数据、证据和确认修改均经过独立审查，审查发现的边界问题已补回归并复验。

最终源码 overlay 全量测试：**448 passed，19 subtests passed**，142.87 秒，退出码 0；JUnit 记录零 failure、error、skip。20 条警告来自现有依赖弃用提示。原有 `test_predeclared_seed_checkpoint_rule_does_not_use_other_seed` 的失败已修复，没有禁用断言或跳过测试。

在应用 overlay 的官方 ISP 环境中运行：

```bash
python -m pytest -q
```

新增回归入口：

- `tests/test_isp_data_eval_regressions.py`：真实图像缓存身份、manifest、复用与清理、异步 CUDA 计时控制流。
- `tests/test_profile_identity_regressions.py` 与 `tests/test_evidence.py`：准备阶段内容冻结、组可用性、修改后的来源及损坏 ROI 的可选错误处理。
- `tests/test_confirmation_authority_regressions.py`：主计划与历史披露、默认值、原生证据及父配置绑定、完整配对覆盖、最新反馈发布身份。

最终源码另完成两轮原生 CPU 联调：官方 style-0 权重、完整 original loss、P256、seed 17，每轮 1 epoch，2 张合成 TRAIN 与 1 张合成 DEV。两轮均保存真实 checkpoint，并在独立进程重载 DEV，逐图 CSV 和 50 行 ROI 测量齐全。基线 PSNR 为 25.39509645 dB，脚本提出的较小学习率候选为 25.28061920 dB；控制器保留基线，完成独立 slow stop、max-trials 停止和冻结。profile 在结果前构建，最终反馈完整可读。没有提供或评测 TEST，没有启动确认或 live Naive/ARIS。这些合成数值只用于检查执行，不代表真实画质。

## 兼容性与证据范围

旧缓存不会直接复用或删除；后续使用新的缓存目录。文件身份基于路径、清单和 stat 元数据，不是逐文件内容哈希系统。

单个旧确认计划保持可用；已有多个旧计划却没有主计划绑定时，汇总标为 ambiguous/inconclusive，不能在看过结果后补造预注册。没有事前内容快照的旧 profile 分析保持可读，但不能标为已证明的事前证据。

CUDA 同步分支在 CPU 环境以模拟异步工作测试，未做 GPU 实测。确认方向规则不是显著性检验；同 seed 重复不增加独立 seed 覆盖。同一权重 reload 校准不估计重新训练的随机波动。当前仍是四个 recipe 字段的受控搜索，引用和文本结构合规不证明模型推理正确。

真实 Naive/ARIS、真实相机数据、多 seed 独立确认及等预算搜索策略对照仍需实际实验。原生合成联调仅证明执行和控制流程，不能据此宣称科研或画质收益。
