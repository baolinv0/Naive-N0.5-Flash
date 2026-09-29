# Original Modular Neural ISP — Phase 1

基于 Samsung 原始 Modular Neural ISP 跑通工程链路，未接入风格迁移或新的算法改善。

- [下载完整工程包](Modular_Neural_ISP_Phase1.zip)：代码、运行说明、短训练权重与实际输出。
- [安装入口](START_HERE.md)：克隆官方 ISP 仓库后应用 overlay。
- [运行说明](overlay/README_PHASE1.md)
- [实际验证记录](overlay/docs/verification.md)
- [Naive / ARIS 适配说明](overlay/docs/naive-setup.md)

已实际验证官方完整推理链路，以及原 photofinishing 短训练 → 最佳权重保存 → 独立进程重载测试；15 项测试通过。
验证使用 CPU 和合成数据。真实数据完整训练、真实 Naive 模型服务及 ARIS 调用仍需对应资源。

本目录的 overlay 应用到 Samsung ISP 仓库，不能直接覆盖本 Naive 模型仓库。ISP 代码与模型许可见 [LICENSE.md](LICENSE.md)。
