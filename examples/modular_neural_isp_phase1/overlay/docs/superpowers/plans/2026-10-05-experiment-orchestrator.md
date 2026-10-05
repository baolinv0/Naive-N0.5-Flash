# Naive/Qwen Experiment Orchestrator Implementation Plan

**Goal:** 实现已批准的五组外层实验编排器及开源Qwen辅助诊断接口。

**Architecture:** 独立策略/模型/数值分析模块，通过现有campaign API串行执行和确认。模型只生成提案或辅助报告，原生评测保持研究分数的权威来源。

**Spec:** [设计规格](../specs/2026-10-05-experiment-orchestrator-design.md)。

## 全局约束

- 四字段离散150配方；同一比较固定模型/数据/init/seed/epochs/P<size>。
- 不直接改campaign.json，不自动重训失败，不暴露CONFIRM/TEST给模型。
- Naive主提案；默认同Naive slow；Qwen启用必须显式记录角色和视觉输入。
- fixture来源明确；未知usage为null；未接资源不宣称真实主实验完成。

## 任务

1. [x] 测试先行实现空间、Scalar投影、Random/TPE和真实引用sidecar；运行单模块测试。
2. [x] 测试先行实现CostLedger、SSE模型服务、有限授权Qwen图像输入和辅助报告；用本地HTTP服务验证。
3. [x] 测试先行实现符号翻转/Holm/区间反演、缺失分母、CSV/图表导出；对照手算和独立枚举。
4. [x] 实现manifest校验、独立配置生成、一次一步可恢复runner、统一确认和CLI；验证暂停、拒绝、重启与身份。
5. [x] 接通五组各一次fixture；更新资源示例/文档；全量回归和独立代码审查。GitHub交付以此变更的实际提交记录为准。

## 审查重点

Scalar历史自由文本可能泄漏诊断；重复TPE和非法Naive提案不能无限重试；模型usage缺失不能虚构tokens或金额；未知提交不能重复launch；小样本不可有界区间与缺失确认不能宣布显著胜出。

2026-10-05 CPU 全量验证：646 passed，19 subtests passed，20 条已有依赖弃用警告；无失败、错误或跳过。独立审阅的授权配置、两级预算、Slow配方执行、浮点ties、恢复及超时边界问题已修复并加入回归。未加载真实Naive/Qwen权重，未执行正式科研主实验。
