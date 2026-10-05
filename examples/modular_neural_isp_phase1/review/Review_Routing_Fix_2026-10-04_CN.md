# 路由与 Agent 指令一致性修复

基于本轮对 `2afcafee` 的源码审查，先修复明确的执行衔接缺陷。本轮没有增加通用 runtime、开放新的 TM 干预变量或替代部署环境验收。

## 已验证并修复

原 `campaign next` 在平台期返回 `diagnose / review_required=true / proposal_allowed=false`，却仍要求 `Stop when proposal_allowed is false`；ARIS 任务第 3、7 步又把这个状态引向 finalize。已删除这条错误映射。

`next_action.action` 现在决定行动，`proposal_allowed` 只控制是否允许 submit。提示词明确标出当前动作，只有允许的 `propose` 才附上写 proposal 与 submit 指令。暂停不会自动终结研究。

| 动作 | 当前宿主指令 |
|---|---|
| wait | 收集同一作业；没有 active 时保留已记录的暂停 |
| diagnose | 查反馈和执行状态；需要事件复盘时读取当前 trigger 的 review-packet，保存决定，再取新路由 |
| propose | 只有提交门也允许时提出并运行一轮实验 |
| finalize / report_stop | 冻结有效且尚未冻结的候选或报告停止；无效 baseline 不尝试冻结 |
| request_scope_change | 暂停并提交范围变更要求；现有 campaign 不扩大权限 |
| confirm | 使用已授权的冻结确认计划；恢复既有任务，终态任务不重跑 |

两份任务文档的早期步骤和后续 fast/slow 说明使用同一合同。路由与提交门不一致时检查当前状态并重新获取路由，不推断新的提交或冻结权限。

本轮只修改生成给 Agent 的指令和任务文档；确定性路由、预算门、实际训练、评测指标、四个 recipe 字段和 TEST 权限保持原实现。

## 实际验证与边界

- 修改前新增回归：**9 failed**，复现旧提示词缺少当前路由操作及错误停止指令；原始输出保留。
- 修改后新增路由回归共 **10 项**，覆盖七动作、首次平台期复盘、重载后的待诊断状态和提交门阻塞。
- 根任务汇总回归：**142 passed，4 deselected，19 subtests passed，21.89 秒**。
- 独立限定范围复核：路由＋慢决定 **47 passed**；既有路由优先级／硬停止 **2 passed**，无阻塞发现。见 [复核记录](../overlay/docs/reviews/20261004/routing-review.md)。
- 源码编译和源码／文档 whitespace 检查通过。测试命令、原始 RED 与汇总输出见 [本轮证据](../overlay/evidence/review_routing_20261004/README.md)。

本轮没有重新运行整套原生进程测试。`2afcafee` 保存的 **246 passed / 115 failed** 仍是此前完整尝试的实际结果，不能用本轮限定范围通过替换它。生产 runner 没有绕过 PID／存活检查，目标 GPU 节点与实际容器配置仍须完成原生全量验收。

## 接下来验证什么

1. 在实际部署节点确认 PID 与 `/proc` 一致，重跑原生全量回归；保持进程保护逻辑。
2. 使用已有采集／登记入口运行真实 baseline＋三次逐轮 Naive 提案，保留结果→读取反馈→下一提案的顺序；自然触发复盘时验证它实际执行。无需预设 PSNR 上涨，TEST 保持禁止访问。
3. 再固定四组策略的同等预算与确认方法：无 Agent 搜索、Naive 聚合分数、Naive 丰富反馈、Naive 丰富反馈＋慢复盘。比较确认收益、有效实验率、实际总耗时、推理成本和人工介入。
4. 为可量化预测逐项记录满足／不满足／信息不足，并将该结果关联到下一次解释取舍；当前非空 prediction/falsifier 字段尚不能证明这件事。

真实服务／GPU 授权和任务数据未提供，本次没有执行 live Naive 或真实 ISP 实验。W 仍待真实闭环验收，R/P 仍缺相应证据。当前研究目标是固定 photofinishing 模型的训练方案搜索；开放 TM 机制研究与产品影调质量验证属于后续授权和实验范围。
