# ARIS Fast/Slow Research：方案与代码复查

- 审查日期：2026-10-04。
- 仓库：baolinv0/Naive-N0.5-Flash。
- 分支：feat/aris-fast-slow-research-20261003。
- 固定提交：c00c84c8901be4c0a8ce041d71897563a596e65a（2026-10-04 提交）。
- 对照：6d5ea1323c6834d0594ca7ad8e2edd04179593b4；中间还包含 91d8577 的委派规则文档。
- 操作：只读审查；未修改远程源码，未运行真实 ISP、Naive、ARIS 或访问任何真实 TEST。

## 结论

这版是从“数值反馈驱动的 recipe 搜索”到“带诊断证据、事件复盘、冻结确认的研究工作流”的实质升级，不能再套用第一轮 baseline-only 的结论。

建议接受总体架构及已实现的证据、授权、确认接口；保留现有主指标与固定架构。进入无人值守运行前，应明确补齐慢循环决定与控制状态的衔接。自动 W 验收还缺可见的真实事件采集/登记路径。当前可以准备人工监督的真实联调，但不能把作者的模拟/合成测试当成 live Naive 自动研究证据。

本次不是完整源码审计，也不是重新执行作者全部测试：重点静态阅读了 control、campaign、research、confirmation、evidence、diagnostics、CLI 和工作流/验证文档的相关路径；完整转存并逐字节核对 control.py 与 test_control.py，独立执行其 13 项测试。另执行 10 个函数级路由/持久化探针；其中 4 个复现同一慢决定衔接问题，不能把“探针通过”误读成该行为正确。

## 一、相对上一版的实际升级

| 维度 | 上一版 | 本版 |
|---|---|---|
| 反馈 | 主要是整体 DEV PSNR、best/no_gain | 按 sample_id 配对、覆盖率、分组/ROI、训练曲线、案例和实际配置差异 |
| 假设 | 文本 hypothesis + based_on | 独立 decision：预测、反证、替代解释、观测引用、最新反馈确认 |
| 参照关系 | 容易混淆参考结果与继承配置 | baseline、best_before、construction_base、comparison、based_on、固定初始化各自登记 |
| 复盘 | 连续无收益则停止 | 必需诊断缺失、有效停滞、证据冲突、hold、冻结和资源边界的事件路由 |
| 执行 | 串行进程与持久状态 | 持久 launch request、活性检查、看门狗、预算/预留与跨阶段关联 |
| 确认 | 主要是一次 final-test | 冻结 baseline/winner × seed × replicate，原 DEV 与独立确认范围分开 |
| 主张 | 流程/性能的文字限制 | W/Q/R/P 分层报告、带范围的经验导出 |

代码依据：[S1]–[S9]。这仍然是固定 photofinishing 与四个 recipe 字段的配置搜索，不是新 TM 网络，也不是 Naive 权重升级。加入诊断和复盘可能改善搜索决策，但没有实验可直接证明它比上一版或简单 HPO 更有效。

## 二、最值得保留的设计

### 2.1 反馈不再只有一个分数

证据通过稳定 sample_id 对齐，配对范围不足时标记 matched_subset；不同科学配置或评测协议不自动作为公平对照。主指标仍是 mean-per-image RGB PSNR。

ROI 诊断实际使用浮点输出，包括 MSE、PSNR、码值亮度偏差、低频误差、梯度误差，并记录区域面积、滤波定义和有效掩码。PNG-8 只用于展示。由此可以区分“整体色调更接近 GT”与“纹理更接近 GT”的证据，但不能由码值亮度直接推断真实照度、ISO 或语义。[S3,S4]

Naive 当前仍是文本观测合同；保存图片路径不代表它已经看过图。[S1,S2]

### 2.2 证据引用、构造父项和实验对照分开

`based_on_run_id` 表示参考了哪条证据；`construction_base_run_id` 表示部分 recipe 从哪个 effective best 继承；`comparison_run_id` 表示想和哪个实验对照。训练仍从 campaign 固定初始化重新开始。[S3,S5]

因此，“引用失败的 MSE 实验，只提交 LR 补丁”不会自动继承 MSE。需要试 MSE+新 LR 时应提交完整所需 recipe。该语义已写入文档，值得保留。

原 recipe→MSE/L1 同时改变了主损失形式及辅助损失组合，不能单凭一次对比称为“证明 MSE 损失形式更好”。

### 2.3 独立确认与原 DEV 稳定性分开

原 DEV 仍负责选 checkpoint；独立数据在 checkpoint 冻结后才评测，不能用于搜索/选权重。确认任务独立于搜索去重，并保留失败或不完整配对。相同原 DEV 上增加 seed 只能测搜索后的稳定性，不能消除该 DEV 已被自适应使用的偏差。[S6]

### 2.4 四层主张不混用

- W：真实 Naive/ARIS 工作流是否有记录支持。
- Q：冻结 recipe 在指定数据/seed/协议范围内是否有改善证据。
- R：快慢策略是否优于等预算对照。
- P：是否满足实际产品条件。

W 不要求涨分；Q 不自动证明 R/P。`supported_in_scope` 也明确不等于统计显著或跨相机泛化。[S2,S7]

## 三、本次发现与需要明确的接口

### F1 — 慢循环决定被记录后，路由可能重新开放 propose

**优先级：P1（无人值守衔接）；证据：静态调用链＋函数级复现。**

涉及：
- `research.py::record_research_decision`
- `research.py::_persist_decision`
- `research.py::route_next_action`
- `cli.py` 的 `campaign record-decision`

当前逻辑：慢决定保存 action/trigger，并把 trigger 登记为 handled；路由遇到已处理的 valid_plateau 就返回 propose。它不根据记录下来的 `report_stop`、`request_scope_change`、`finalize` 或普通 `diagnose` 维持对应控制状态。`record_research_decision` 对 `protocol_suspect` 有专门 hold 处理，但不是所有 action 的通用执行器。[S5,S7,S8]

复现场景：无硬停止、无 active job、预算足够，连续两次有效 no_gain，反馈可读。先得到 diagnose/review_required=true；持久化带同一 trigger 的 slow_review 决定，再路由。

| 记录的慢决定 | 后续 action | review_required |
|---|---|---|
| report_stop | propose | false |
| request_scope_change | propose | false |
| finalize | propose | false |
| diagnose（未设置 protocol_suspect） | propose | false |

原始输出在 `evidence/slow_action_*.json`。测试执行的是原函数体 `_persist_decision + route_next_action`，真实 control 模块；只替代 JSON 写入依赖。没有执行完整 ARIS 宿主或完整 record-decision 校验链，静态阅读用于确认外层没有另外应用这些 action。

**准确解释：**这不是已发生越权实验；四字段和预算门仍有效。问题是“停止/暂停/待人决定”的意图没有成为恢复后可读取的控制状态。若设计意图是 record 只归档，则需要明确的 apply/ack 步骤，且在该步骤完成前不能把事件当成已处理后自动继续。若宿主人工随后调用 finalize，当然能停止，但不能据此保证自动恢复路径。

**最小修改：**选择唯一语义：保存 `pending_slow_decision`，由 route 尊重其动作；只有允许继续的、匹配当前 trigger 的 slow_review 才解除事件门。report_stop 应持久停止，request_scope_change 应停在等待授权/新 campaign，diagnose 应保留待诊断状态，finalize 应执行或等待显式冻结完成。无需新 Agent runtime，也无需扩大权限系统。

**回归：**慢决定保存后再次 next、宿主重启后 next、然后 submit，均应遵守该决定；不是只检查日志写成功或 review_required 变 false。

### F2 — 自动 W 验收器存在，但真实事件采集/登记路径未接通

**优先级：P1（自动验收接入），不等于无法进行人工审阅的 live pilot；证据：静态。**

`research.py::_w_claim` 要求顶层 `workflow_evidence`，特定 schema 的 service 配置、startup 记录，以及有顺序和 decision/feedback 引用的 JSONL 事件。普通 `script` 终端转录不是这种 JSONL。[S7]

当前 CLI 未提供显式登记 workflow evidence 的入口；campaign 初始化不写此字段。adapter 在本次相对上一版差异中未修改，现有服务桥不产出这套 startup/transcript schema。对应成功测试由测试代码构造 events/service/startup，并直接写 `campaign.json['workflow_evidence']`，这证明检查器能检查格式，不证明 live 宿主已能生成这些证据。[S8,S10]

**最小修改：**复用现有 ARIS 原始会话导出，增加一个确定性转换/登记工具（或现有 host 的事件 hook），从真实 tool results 生成结构化索引，保留原始转录；由操作员/控制器登记 manifest。服务启动日志记录真实模型与权重身份。禁止由研究 LLM事后编写“accepted”事件替代原始运行证据。

验收应通过正常 CLI/host 路径得到 W，不要求操作员手工编辑 campaign.json。没有转换器前，可人工检查原始 live 转录，但自动 W 应继续 pending。

### F3 — predeclared_seed 最终权重规则与 final-test 入口可能指向不同 checkpoint

**优先级：P2（后续可选确认流程）；当前不访问 TEST 的 pilot 不受影响；证据：静态。**

`confirmation.py::_canonical_plan/report_confirmation` 接受 `final_checkpoint_rule.kind=predeclared_seed`，报告对应 winner seed 的 checkpoint。然而 `campaign.py::_controlled_final_test` 和 legacy final_test 都取 `state['frozen']['run_id']` 对应的搜索阶段权重，没有读取 confirmation 的 final_checkpoint。[S6,S9]

若用户声明最终采用某个预指定 seed，报告可能指向确认权重，而最终 TEST 实际评的是搜索权重。默认 retain_search_checkpoint 没有此问题。

处理二选一：暂时只支持 retain_search_checkpoint，并明确 predeclared_seed 仅作另行导出、不驱动 final-test；或者新增明确冻结的 final_checkpoint_ref，让报告、导出和实际最终评测指向同一权重。当前不得为了测试此路径读取真实 TEST；用替代 evaluator 验证实际传入的权重身份即可。

## 四、独立测试与证据边界

### 本次实际执行

1. 当前 `control.py`、`test_control.py` 的完整内容经 GitHub connector 获取后转存；Git blob 标识逐字节一致，详见 acquisition.json。
2. `PYTHONPATH=probe_src python -m pytest -q tests/test_control.py`：13 passed。
3. `PYTHONPATH=probe_src python -m pytest -q tests/test_focused_review.py`：10 个探针执行完成；6 个核查既有路由/控制行为，4 个复现 F1。同一个问题的四个动作变体不是四个独立缺陷。
4. 没有修改远程仓库，没有真实训练，没有调用真实 Naive/ARIS，没有运行真实 TEST。

### 作者记录，不是本次重跑

固定版本验证文档记录 266 tests＋19 subtests 通过，以及 4 次真实 ISP 代码的合成 CPU 搜索、2 次 original-DEV 确认。提案是预设脚本，数据规模为 2 TRAIN/1 DEV、P256；四个搜索实验全部改善，原生 CPU 轨迹没有触发科学停滞 slow review，slow 的异常和停滞由回归测试覆盖。[S2]

这些能支持工程部件的运行证据，不能支持真实 Naive 的判断质量、真实相机改善或快慢策略优越性。本次也未重新复算该六个训练作业。作者已保留这些范围限制，不应把其合成分数包装成新模型性能。

### 网络与测试范围

容器直接 git 下载因 DNS 失败；随后归档/原始下载尝试亦未取得代码。实际可执行部分来自连接器获取并核对的源文件，不把失败下载计为成功 clone。本包提供局部探针环境，不是完整 ISP 可运行分发。

## 五、下一步最小交付

先修 F1，并补 F2 的最小采集/登记链；不要继续增加通用基础设施。保持已授权模型、数据、评测、预算与四字段搜索范围。当前 pilot 不访问 TEST。

真实 Naive/ARIS 运行 baseline＋3 次后续提案；各提案在前一轮结果返回后产生，引用实际浮点诊断/逐图证据。保存 proposal、decision、feedback、实际训练/DEV产物和原始宿主转录。若自然出现有效停滞，验收一次“复盘→决定→实际状态/下一实验”的完整 slow 路径；未出现则如实只报告实际覆盖的路径，不编造失败或触发。

停止/冻结本身也可产生慢循环事件，但不能以此声称已经验证“停滞诊断提高科研效率”。工程流程不要求涨分。真实 recipe 改善、多 seed 独立确认和等预算策略对照另行按授权执行。

**最重要的判断：这版增强了研究环境和证据质量，尚未证明增强了研究智能或 PSNR。下一轮应交付真实模型消费这些证据并改变实验决定的记录，而不是继续扩大框架。**

## 来源（固定提交路径）

所有路径均位于 `https://github.com/baolinv0/Naive-N0.5-Flash/blob/c00c84c8901be4c0a8ce041d71897563a596e65a/examples/modular_neural_isp_phase1/overlay/`。

- [S1] `docs/fast-slow-research.md`。
- [S2] `docs/fast-slow-verification.md`。
- [S3] `tm_research/evidence.py`，参照关系、实际配置差异、按 ID 对齐及 CSV 检查。
- [S4] `tm_research/diagnostics.py`，425–536，区域统计与浮点测量。
- [S5] `tm_research/campaign.py`，收集、决策、授权、next/submit、冻结路径。
- [S6] `tm_research/confirmation.py`，1–220、570–692，计划及报告。
- [S7] `tm_research/research.py`，路由、记录决定、W/Q 主张构建。
- [S8] `tm_research/cli.py`，完整命令分发。
- [S9] `tm_research/campaign.py`，492–末尾，最终评测权重选择。
- [S10] `tests/test_research.py`，`test_recognized_w_events_link_actual_runs_and_feedback_before_each_proposal`。
- [S11] `tm_research/control.py` / `tests/test_control.py`，完整文件与独立重跑。

未把摘录范围之外的整个源码树声称为逐行审查通过。