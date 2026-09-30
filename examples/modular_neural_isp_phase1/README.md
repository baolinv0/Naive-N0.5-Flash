# Modular Neural ISP：baseline + 受控研究循环

基于 Samsung 官方模型，现已增加可调训练 recipe 与结果驱动 campaign。本轮实现针对 review 的 A/B/C：统一评测、开放配置、接通提案—执行—比较—下一轮。模型结构不变，未接入风格迁移或历史改善。

- [安装入口](START_HERE.md)
- [运行说明](overlay/README_PHASE1.md)
- [当前研究合同](overlay/docs/engineering-contract.md)
- [本轮实现与实测记录](overlay/docs/review-fix-verification.md)
- [二次 review 后续修复与验证](overlay/docs/rereview-followup.md)
- [ARIS 任务入口](overlay/docs/tm_research_task.md)
- [原审查意见与证据](review/README.md)

原 [Phase 1 ZIP](Modular_Neural_ISP_Phase1.zip) 保留为历史交付，仅包含 baseline 阶段；最新功能请使用当前 overlay。CPU 合成联调不能替代真实数据训练或 live Naive/ARIS 验证，不能据此宣称 PSNR 提升。

overlay 应用到 Samsung ISP 仓库，不能覆盖本 Naive 模型仓库。许可见 [LICENSE.md](LICENSE.md)。
