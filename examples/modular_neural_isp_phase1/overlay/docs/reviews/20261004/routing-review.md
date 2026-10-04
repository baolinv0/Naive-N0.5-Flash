# 路由与提示词一致性专项复核

日期：2026-10-04。复核范围为工作区相对 `2afcafee` 的 `campaign.py::_action_prompt()` / `next_proposal()` 提示词改动、`test_next_action_prompt.py`，以及 `aris-campaign-task.md` 和 `fast-slow-research.md` 的对应任务合同。只读代码复核；未改生产源码、未提交 Git、未运行真实模型或训练。

**结论：APPROVED。未发现本次范围内的阻塞问题。**

## 核查结果

- `next_action.action` 是唯一动作选择依据。`proposal_allowed` 仍是提交 guard，false 不再发出结束、冻结或提交指令。只有 `propose` 且 guard 为 true 时提示词包含实际提交步骤。
- 七个现有动作均有一致的指令：`wait` 收集原作业或保留暂停；`diagnose` 检查现有证据和状态；`propose` 提交一轮；`finalize` 冻结有效且未冻结的候选；`report_stop` 报告停止；`request_scope_change` 保持当前范围并请求新的授权 campaign；`confirm` 仅推进已授权、未完成的冻结确认任务。
- 平台期第一次 `diagnose + review_required=true` 明确要求读取当前 trigger 的 review-packet、记录 scoped slow review，再读取新路由。已持久化的 diagnosis 不再要求重复记录相同复盘，仍需匹配的显式 continuation 才能提交。
- 实际 `route_next_action()` 的既有优先顺序保留：活动或 unknown 作业先 wait/diagnose；冻结、hard stop、hold 和资源边界仍由确定性控制决定；提示词未为复盘增加越过这些边界的许可。
- 文档第 3–7 步与后部 fast/slow 段已统一，不再保留 false guard → 跳结束步骤的旧流程。文档明确无有效候选时报告失败、已有冻结时复用、已有确认计划时恢复而非重建。
- 生产代码 diff 仅修改提示词构造及其条件拼接。未修改路由、recipe 权限、选优分数、授权合同、执行监督或 TEST 权限。

## 本次实际验证

在仓库根目录，使用现有 `.venv/bin/python`，`PYTHONPATH=examples/modular_neural_isp_phase1/overlay`，`OMP_NUM_THREADS=2 MKL_NUM_THREADS=2`：

```bash
.venv/bin/python -m pytest -q \
  examples/modular_neural_isp_phase1/overlay/tests/test_next_action_prompt.py \
  examples/modular_neural_isp_phase1/overlay/tests/test_slow_decisions.py
```

结果：**47 passed in 0.22s**。涵盖本次新增的 10 个提示词用例和现有持久化慢决定/续行边界用例。

```bash
.venv/bin/python -m pytest -q \
  examples/modular_neural_isp_phase1/overlay/tests/test_research.py \
  -k 'invalid_is_not_scientific_plateau or diagnose_precedes_plateau'
```

结果：**2 passed, 15 deselected in 0.01s**。验证诊断优先级及 hard stop 不被科学平台期替代。

本结论仅关闭已指出的动作合同冲突，不代表真实 Naive/ARIS 宿主已按这些指令完成工具调用，也不代表目标部署节点的原生回归、科研收益或产品收益已验收。
