# Modular Neural ISP 原始方案：第一阶段跑通计划

版本：v2.1，2026-09-29。范围按用户最新要求收敛。
状态：实施设计已更新；尚未在用户机器部署、训练或取得性能结果。

## 1. 本阶段唯一目标

使用 Samsung 官方 Modular Neural Image Signal Processing 原始方案，跑通数据准备、模型加载、训练、验证、测试、结果保存，再接通 ARIS-Code + Naive 的任务调用与结果读取。

本阶段验收是原始 pipeline 可以手动执行、可以由 agent 调用、结果可以重新加载和查看。PSNR 用于记录 baseline，不要求涨分，不启动优化搜索。

范围承接前文的 photofinishing 训练任务：沿用官方数据前处理、模型结构与原训练配方。官方端到端示例中已有的其他 ISP 模块按原方式调用，不扩展成重新训练所有 ISP 模块的项目。

## 2. 保留原方案的哪些内容

| 部分 | 本阶段决定 |
|---|---|
| 代码起点 | SamsungLabs/modular_neural_isp 官方仓库 |
| 输入 | 按官方示例准备数据与metadata；photofinishing使用其原始输入链路 |
| 模型 | 沿用官方配置、已有模块和对应权重，3D LUT开关与配置匹配 |
| 训练 | 原loss、原optimizer、原learning-rate schedule、原增强策略 |
| 目标图 | 原配对GT，沿用现有数据划分与目标风格 |
| 推理 | 原forward和原示例处理方式 |
| 自动化 | ARIS-Code负责调用，Naive负责执行任务和读结果 |

暂不接入用户此前的任何改善部分，包括Qwen/Vera风格迁移、teacher/伪GT、辅助loss、新MSE训练配方、参考图分支、语义分支、跨相机适配、AE/AWB联合优化，以及自动调参搜索。此前文档中的教师生成与候选实验计划不属于本阶段任务。

## 3. 实施顺序

### 第一步：官方示例推理

使用官方环境说明、对应预训练权重和示例数据，先执行原始推理，保存输出图与实际命令。确认能加载模型并得到正常图像。

如果原示例需要其他ISP模块，按官方组合运行。不要在这个阶段替换模型、额外接入用户自己的增强模块。

### 第二步：原始训练入口短跑

用原数据组织和原训练配方，在少量真实样本上运行2个epochs，完成：读取数据、前向、反向、保存checkpoint、验证、重新加载checkpoint。

短跑只减少数据量和训练时长，用来确认执行链路。它不用于判断算法质量，也不称为完整baseline复现。

### 第三步：原始baseline

短跑通过后，用正式数据划分与原训练配方运行baseline。完整训练预算参考原入口配置与可用算力；若实际缩短预算，报告中直接标明，不宣称复现了论文成绩。

已有权重用于推理、原训练配方重新训练、从已有权重继续微调是不同实验，分别记录。不要把官方 `--load` 当成完整断点续训：现有入口加载模型参数，不等于恢复optimizer与scheduler。

### 第四步：验证和测试

调用原训练验证与测试入口，保存checkpoint、平均及逐图PSNR、代表输出图。保留各自原始尺寸规则：512训练验证和quarter-resolution测试分开命名、分开报告，不强行把两者改成同一协议。

### 第五步：接入ARIS与Naive

手动链路通过后，由ARIS-Code中的Naive完成一次相同流程：读取任务和配置、启动原始训练、读取日志与指标、调用测试、汇总结果。

首版Naive只负责把原始流程执行正确，不自行改loss、结构或超参数。训练在普通进程中运行，程序负责等待；无需让模型持续生成轮询消息。

### 第六步：完成报告

交付一条手动命令链和一条agent调用链，并列出实际数据、训练长度、指标口径和输出位置。本阶段到此结束，后续优化另开实验。

## 4. 仅处理直接影响执行或数值正确性的工程问题

保持算法原样，不等于保留已确认的读取与统计错误。只修实际影响本次执行的问题，并将这些小修改单独列出。

| 已核对的问题 | 最小处理 |
|---|---|
| 训练/测试按目录位置zip，存在配对不确定性 | 按实际样本ID对应input、GT和metadata |
| HDF5缓存句柄在with中关闭后再次使用 | 直接按需打开并关闭文件 |
| validate用从0开始的batch索引作除数 | 按实际计数累计，消除错误分母 |
| batch级PSNR不等于逐图平均PSNR | 记录原日志，并另外输出正确的逐图PSNR均值，明确列名 |

PSNR的补充统计为每张图先计算MSE，再转PSNR，最后对图像平均。它是测量修正，不改变训练loss，不作为算法改进计入收益。修正后的baseline使用同一统计实现。

不重写Dataset，不重建评价服务，不增加teacher字段，不扩建缓存框架。现有HDF5预打包batch结构继续使用。

需要恢复中断训练时再给现有checkpoint补optimizer、scheduler和epoch；先跑通短实验，不把完整恢复系统作为首次推理的前置条件。

## 5. 最小自动化实现

直接复用ARIS-Code的文件工具、Bash、会话记录和自定义provider。先提供两个任务接口即可：

```python
run_baseline(config: dict) -> str   # 调原训练命令，返回run_id
get_result(run_id: str) -> dict     # 返回状态、日志、指标与产物路径
```

这些是待实现的包装接口，不是上游已有API。runner只负责传递原参数、启动进程、指定输出目录、收集结果；不重新实现训练器。

每次运行只保存：实际命令、配置、日志、checkpoint、metrics和输出图。运行编号采用exp_001这样的顺序编号。先串行运行，一个Naive会话即可。

Naive服务已经兼容ARIS时直接配置；仅有原始推理代码时才补工具协议转换。ARIS-Code当前使用流式Chat Completions，适配层需提供SSE与结构化tool_calls。通过一次真实的“读配置→执行短命令→读结果”验证接通。

Naive部署尚未就绪时，原始ISP链路仍先独立跑通；最终agent调用验收需要真实Naive服务，不用其他模型结果代替。

## 6. Agent分工

| Agent | 任务 | 完成标准 |
|---|---|---|
| A：原方案执行 | 官方示例、原训练入口、必要读取/统计修正 | 手动完成推理→训练→验证→测试 |
| B：ARIS/Naive接入 | 配置已有runtime，按需适配Naive工具协议 | 真实Naive能执行命令并读取结果 |
| C：闭环集成 | 两个包装接口、独立输出目录、结果汇总 | agent调用与手动调用使用同一原始流程 |

A和B可并行；C在A通过后接入，再与B联调。原方案中的风格迁移agent与优化实验任务本阶段取消。主负责人最后检查一次完整链路，不做逐项打分式评审。

## 7. 验收只看四件事

1. 官方模型和输入能生成正常输出图。
2. 原训练配方能完成训练、验证、checkpoint保存与重新加载。
3. 测试结果和图像对应，指标统计含义清楚。
4. ARIS + Naive能调用同一流程并读取真实结果。

无需证明PSNR提高，无需候选竞赛、多seed比较或教师消融。正式baseline没有跑完时，只报告工程链路已通过、完整训练仍在进行。

## 8. 给实施agent的任务说明

> 先跑通Samsung官方Modular Neural ISP原始pipeline。使用官方模型、数据前处理和原训练配方，先推理、再短训、再正式baseline，最后接ARIS-Code与Naive。只修影响当前运行和统计正确性的实际问题。不要引入用户历史改进，不生成伪GT，不添加风格迁移或新loss，不启动性能搜索。记录实际命令、配置、日志、checkpoint与结果，让下一位工程师可以直接复跑。

## 参考入口

- [Samsung官方仓库](https://github.com/SamsungLabs/modular_neural_isp)
- [Photofinishing说明](https://github.com/SamsungLabs/modular_neural_isp/blob/main/photofinishing/README.md)
- [ARIS-Code运行时](https://github.com/wanshuiyin/Auto-claude-code-research-in-sleep/blob/aris-code/README.md)
- [Naive模型与工具模板](https://github.com/baolinv0/Naive-N0.5-Flash)

本文件为v2.1当前执行范围，替代此前v2中的风格迁移与优化计划。
