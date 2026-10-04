# 快慢科研流程：实现与验收记录

2026-10-03。基于 `docs/aris-delegation-policy-20261002` 的 `91d8577`，实现于独立功能分支。运行入口见 [fast-slow-research.md](fast-slow-research.md)，冻结规格见 [原实施方案](plans/aris-fast-slow-research-20261003.md)。

## 已实现

| 环节 | 实现与边界 |
|---|---|
| 固定数据档案 | 共享 ISP 读取与评测变换；TRAIN 校准或预设阈值；稳定 ID、分组、场景来源、固定亮度×梯度掩码；不扫描 TEST |
| 浮点诊断 | 量化前区域 MSE/PSNR、码值亮度偏差、低频 RGB 误差、梯度误差；保存有效面积、完整测量定义和固定掩码身份 |
| 证据包 | baseline/本轮对照逐样本比较、覆盖率、组内变化、案例、训练曲线、实际配置差异、不可用原因、可读取的 observation 来源 |
| 快循环 | 保留三键 proposal 和四个 recipe 字段；control 模式要求独立 decision、最新反馈确认、预测/反证/替代解释；必需证据缺失阻塞，可选失败保留原生事实 |
| 慢循环 | 无效执行与科学停滞分开；事件 review packet、独立决定记录、协议疑点 hold、证据支持的解除；停止/冻结/资源边界优先 |
| 作业执行 | 准备请求 token、持久启动意图、同 ID 恢复、进程身份、独立看门狗、包含清理的总 walltime、失败成本、未知活性保留预留 |
| 冻结确认 | baseline/winner × seed × replicate；实际任务与冻结清单对应；原 DEV 选 checkpoint 后再评测授权独立 split；失败保留、不自动重试、不更新搜索 best |
| 结案 | 四池成本统一核算；外部确认 receipt；W/Q/R/P 分层、实际确认范围、负结果及范围明确的经验导出 |

平均逐图 RGB DEV PSNR、`min_delta` 选优、raw best 和旧模式语义保持原有合同。PNG 供展示；Naive 的现有输入仍是文本，图片路径不会被算作视觉观察。

## 完整回归

最终运行结果：**266 passed，19 subtests passed，101.83 秒，0 失败**。18 个警告来自上游 matplotlib/colour-demosaicing 对依赖旧接口的使用。原始输出：[pytest.log](../evidence/fast_slow_cpu/pytest.log)。

在 overlay 目录运行，官方 ISP 提供 overlay 不包含的 utils/models/loss_utils：

```bash
PYTHONPATH="$ISP_ROOT:$ISP_ROOT/photofinishing" \
  OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  "$PYTHON" -m pytest -q
```

验收环境：Python 3.12、CPU PyTorch 2.5.1、torchvision 0.20.1、NumPy 1.26.4；官方 ISP `5a845f673edfdf92de18dcfc20d204f79f1ba38b`。HTTP 集成测试需要允许本机 socket；该环境使用相应执行权限后通过。

有针对性的回归覆盖：同名错数据集、同面积不同位置掩码、不同滤波定义、共同漏样本、CSV/原生均值不一致、不兼容协议、损坏或不可达详情、旧反馈确认、准备/发布/启动中断、worker/evaluator SIGKILL、残留子孙进程、写入失败、当前 hold 与新增成本、被拒绝的确认计划、冻结任务不完整和声明范围错误。独立任务复核所列 Important 问题均已修复并通过限定范围复核。

最终全分支审查发现三项跨阶段问题，均在一个修复批次中解决：确认实验所有实际启动窗口重查当前父 campaign；已知 TRAIN/DEV 与确认场景重叠限制独立结论和经验晋级；评测 receipt 在 runner 确认终态且资源退出前保留活动状态。新增 28 项回归先复现问题再通过，CPU/GPU、即时/持久 receipt 和真实看门狗恢复均覆盖。最后独立专项复核 **28 passed / 24.88 秒**，I1/I2/I3 全部 addressed，0 未解决发现。审查、修复和复核原文见 [reviews](reviews/final-rereview.md)。

## 真实 ISP 代码的 CPU 闭环

这是合成数据工程验收。使用官方 style-0 初始化权重、2 张 TRAIN、1 张 DEV、P256、seed 7、每轮 2 epoch、batch 1。四轮搜索和两个确认作业均实际训练、重载最佳权重并运行 DEV。每作业上限 120 秒，明确 CPU 分配；配置不含 TEST。提案是预设脚本，并非真实 Naive 输出。

| 搜索 run | 完整 recipe 的变化 | DEV PSNR / dB | 原生 valid | 诊断 |
|---|---|---:|---|---|
| exp_001 | MSE / Adam / LR 0.001 / WD 0 | 21.334834 | true | complete |
| exp_002 | LR → 0.0005 | 23.086680 | true | complete |
| exp_003 | LR → 0.00025 | 25.535658 | true | complete |
| exp_004 | loss → L1，继承有效 best 的 LR 0.00025 | 26.027617 | true | complete |

各轮生成 50 行固定区域测量、版本化反馈和真实配置差异，控制器读取最新证据并保存 decision 后提交。达到四轮上限后冻结 `exp_004`。该数据序列一直改善，因此本次原生 CPU 路径没有触发科学停滞 review；停滞、无效执行和 hold 恢复由独立回归覆盖。

确认目录位于搜索目录之外。`baseline:7:1` 与 `winner:7:1` 均完成；逐图数、原生均值和唯一场景统计一致。确认重复结果为 21.334834 / 26.027617 dB，差 4.692783 dB，仅适用于这个合成单样本范围。报告如实保留 original-DEV adaptive bias、单 seed 和无统计显著性限制。确认阶段未配置 ROI 档案，其逐图/场景证据可用，ROI 不作可用声明。

六作业实际 CPU walltime 合计 **83.939 秒**；分配 GPU 数为 0，测量 GPU-hours 为 0。Naive 推理成本和 token 数没有测量。可移植的实际数值摘要：[native_cpu_acceptance.json](../evidence/fast_slow_cpu/native_cpu_acceptance.json)。

搜索代码对应 `326009c`；确认运行对应 `d15482d` 的模块内容。期间只有缓存、恢复、授权与声明范围修复，模型、数据、初始化、训练预算和评测变换保持相同；控制副本保留原冻结身份。

最终跨阶段修复 `9d0731a` 后，复制更新模块并再次收集既有 CPU 确认及结案报告：任务 ID、run 目录和搜索 best 均未改变，没有重新训练或追加评测。原始均值、确认 pair 数和 W/Q/R/P 范围保持一致。上述训练成绩仍归属于实际执行时的代码；最终保护逻辑由新回归及本次重复收集验证。

## 尚需真实条件验收

- **W：pending。** 需要真实 Naive/ARIS 的 baseline＋三轮消费反馈的提案，以及对应模型/服务身份、按序事件和原始 transcript。预设脚本不会满足 live 门槛。
- **Q：本次只验证合成 original-DEV 范围的确认路径。** 真实任务需按预先授权的种子、预算和独立确认数据执行；上述 PSNR 不代表真实相机收益。
- **R/P：unavailable。** 尚无同预算策略对照或产品条件证据。
- 跨主机恢复保守返回 unknown；支持的自动终止与身份验证是同主机 Linux 进程组。路径映射验证当前可读性，跨机传输由现有 ARIS/共享目录承担。
- TRAIN 精确像素分位数会保留校准数组；大数据可使用事先确定的显式阈值。码值亮度/梯度标签没有 ISO、曝光或语义因果含义。

## 实施时的处理决定

按作出顺序列出决定及判断错误时的成本：

1. 在新 clone 的独立分支工作，避免影响已有 checkout；错误成本为迁移分支。
2. 按文件职责并行分工、统一提交；错误成本为集成返工。
3. 每作业增加同主机看门狗，在 worker 意外退出后仍执行期限；成本为额外监控进程。
4. walltime 包含终止清理，在截止前进入软终止窗口；成本为短时限下可用计算时间减少。
5. 对照使用实际 resolved 科学配置，仅剔除普通搜索未执行的 TEST；错误成本为重新核对对照兼容性。
6. control 的 `mean_per_image_rgb_psnr` 对应原生 `mean_per_image_psnr` 的既有 RGB 定义；错误成本为更正度量身份校验。
7. 可选诊断失败不阻塞，只有声明必需且适用的证据才阻塞；错误成本为后续 campaign 需采用更严格的明确合同。
8. 恢复实际启动前重查当前授权、hold、活性和成本，只排除该作业自身预留；错误成本为等待诊断后恢复。

代码、文档、模板和自动化验收已经交付；真实服务/数据实验的科学判断依赖上述尚未收集的证据。
