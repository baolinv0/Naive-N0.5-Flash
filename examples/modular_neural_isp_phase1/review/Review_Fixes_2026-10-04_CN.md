# 2026-10-04 评审修复回应

针对 [Code_Fast_Slow_Review_CN.md](Code_Fast_Slow_Review_CN.md) 中的两处控制／验收衔接问题，以及预指定确认权重的一致性问题，已在同一功能分支完成修复。修复基线为 `f3ee3784`（源码与被审查的 `c00c84c8` 相同，只增加原评审文件）。

## 改动与验收

| 评审问题 | 当前行为 | 关键回归 |
|---|---|---|
| F1：慢决定只留在日志 | `pending_slow_decision` 持久化动作；停止、待授权、诊断、待冻结分别控制后续路由和实际启动。重启、旧决定重放、准备作业恢复均不能绕过。只有匹配当前事件且通过现有授权／证据／预算检查的慢复盘才能继续非终止状态 | 四种动作重载后仍阻断；错误事件、普通 fast 决定和旧提案不放行；中断写入重放恢复；排队 recipe 与放行决定一致 |
| F2：W 缺少真实登记路径 | `capture-workflow` 保留实际 CLI 输入输出；Naive adapter 在模型加载与 HTTP 绑定成功后登记实际加载身份，并保存请求／原始生成。`register-workflow` 确定性关联反馈、模型输出、实际 submit 和终态 run；保存原始来源、生成索引并关联 campaign | baseline＋三轮 HTTP／CLI 工程闭环；缺失／乱序／篡改／未消费反馈保持 pending；登记中断重试；显示别名与实际权重路径区分 |
| F3：最终报告与实际测试用不同权重 | `campaign final-checkpoint` 明确采用完整、关联且预声明的确认任务，保存不可替换的 `final_checkpoint_ref`；报告、memory、controlled／legacy final-test 使用同一引用。报告读取不自动采用权重 | 指定 seed 权重实际传给 evaluator；缺失或失败拒绝回退；多计划须明确选择；重复测试复用；确认配置必须与冻结父实验一致 |

终止状态上的另一种新 slow-review 动作会在写入任何记录前被明确拒绝；可用 `closure` 记录分析而不改变控制状态。`report_stop` 后可以明确冻结，并在既有授权内确认；`request_scope_change` 需要新的授权 campaign。确认候选存在与最终权重已采用是不同状态。

四个 recipe 字段、平均逐图 RGB DEV PSNR／`min_delta` 选优、固定初始化和 TEST 边界保持原合同。本次没有增加常驻 Agent 或开放结构／数据／预算修改权限。

## 本次实际验证

- 根任务汇总回归：**132 passed，4 deselected，19 subtests passed，21.51 秒**，覆盖新增慢决定、最终权重、工作流登记及既有 control／adapter／研究报告测试。完整命令见 [本次验证记录](../overlay/evidence/review_fixes_20261004/README.md)。
- 两组独立限定范围复核：F1/F3 **9 passed**，F2 **10 passed**；初次复核提出的全部问题已处理，最终复核无未关闭发现。见 [F1/F3 复核](../overlay/docs/reviews/20261004/F1-F3-rereview.md) 和 [F2 复核](../overlay/docs/reviews/20261004/F2-rereview.md)。
- 当前全量重跑：**115 failed，246 passed，19 subtests passed，156.91 秒**，退出码 1。原始失败输出完整保留，不能把历史 266 项全通过当成本次结果。
- 当前容器 `os.getpid()` 为 5 时，`/proc/self/stat` 返回 7137，`NSpid` 为 `7137 5`。原生进程身份／监督测试在此环境失效，训练入口拒绝不可验证 owner（状态 125），导致依赖有效 baseline 的下游断言失败。生产 runner 未改动，未弱化身份或退出检查；部署环境须重新跑原生全量集成。
- 完整日志专项归类为 62 项无效 winner 的前置失败、28 项无效 baseline 的下游失败、25 项既有 runner 边界失败；未发现新 F1/F2/F3 回归。日志不能单独证明 115 项失败全部只有环境原因；两份 [归类报告](../overlay/docs/reviews/20261004/full-suite-triage-F1-F3.md)、[F2 归类](../overlay/docs/reviews/20261004/full-suite-triage-F2.md) 保留了这一限制。
- 新增状态／权重回归使用持久化原生结果 fixture 和有限执行边界替身；W 接受分支使用明确标记的模型加载替身、微型 train/eval 子进程及测试专用同步监督。因此它们证明工程接线，不能证明生产看门狗或 live Naive 能力。
- 用当前模块重新收集既有合成 CPU 确认：两个任务、run ID、搜索 best 和 frozen 不变，无新增训练或评测；W 仍 pending，Q 仅 supported_in_scope（合成 original DEV），R/P unavailable。

源文件 compile 和源码／文档的 `git diff --check` 通过；原始 pytest 日志保留其尾部空白，不改写测量输出。实施者的 RED→GREEN、首次独立审查与最终复核原文保存在 [本轮审查目录](../overlay/docs/reviews/20261004/)。

## 真实联调入口与剩余条件

按 [workflow-evidence.md](../overlay/docs/workflow-evidence.md) 使用现有 ARIS 宿主和真实 Naive adapter：保留捕获的完整反馈 JSON，将模型实际返回的 proposal／decision 原样提交，baseline 后顺序执行三轮，再登记并查看 W。当前转换器支持文档中的明确 JSON 合同，不猜测未提供格式的 ARIS 导出；通用终端转录可供人工审阅，但不会自动变成 accepted 事件。

本次未提供真实 Naive 服务／权重、实际 GPU 授权或真实任务数据，因而没有执行 baseline＋三次真实模型后续提案，也没有访问 TEST。下一次实际验收应在 PID 与 `/proc` 一致的部署环境完成全量回归和上述真实会话，检查模型是否利用诊断改变实验决定；自然触发慢复盘时，验证停止／暂停／继续的持久控制。真实 PSNR 收益、研究策略效果和产品证据仍待测量。
