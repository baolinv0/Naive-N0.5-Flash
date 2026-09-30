# 二次 review 后续修改与验证

日期：2026-09-30。依据 `review/Code_ReReview_CN.md`，保留已实现的统一评测、受控 recipe 和串行研究控制器。本次只处理有效增益阈值，以及真实 Naive＋ARIS 小规模 pilot 的操作说明。

## 修改内容

- `campaign init --min-delta` 将阈值固定到 campaign 状态；提案不能修改它。默认 `0.01 dB` 是工程初值，真实 pilot 应先按目标环境的重复评测波动确定并显式传入。
- `raw_best` 保存所有有效结果中的最高实测分数；`best` 只在新分数严格大于当前有效最佳分数加阈值时更新。只有这种改善才标记 `improved` 并清零无收益计数。
- 微小正差保留原始分数，但计为 `no_gain`。比较基准保持为上次有效最佳值，避免连续微小改善移动阈值；累计超过阈值时仍能更新。
- `campaign next` 返回这两个记录、冻结阈值和明确判定。提案继承与 finalize/final-test 选择使用有效 `best`。
- 旧 campaign 没有阈值字段时保留原来的零阈值规则，不重算历史；需要新阈值时新建 campaign。
- [ARIS 任务](aris-campaign-task.md) 明确 baseline＋3 次后续提案、单卡串行、阈值校准、宿主转录、模型身份与逐轮反馈证据。真实模型服务可以独立于训练 GPU。

## 本次实际验证

环境：Linux、Python 3.12.14、PyTorch 2.5.1+cpu，无 CUDA。

先在旧实现上运行新增回归：5 项失败，包括复现 `25.000000 → 25.000001 dB` 被错误标为 `improved`；另有新 CLI/参数及旧状态兼容尚不存在导致的预期失败。修改后 campaign 测试 **11 passed**。

将当前 campaign、CLI 和测试覆盖到已有的官方 ISP 源码＋overlay 安装验证目录，执行：

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=. python -m pytest -q
```

结果：**54 passed，19 subtests passed，12.77 秒**。20 条警告是依赖弃用提示。完整输出见 `evidence/rereview_followup/pytest.log`。新增回归验证了微小涨分不会延长搜索或替换冻结候选、恰好等于阈值不提升、累计越过阈值才重置计数、CLI 阈值传入和状态续读、旧 campaign 兼容，以及负数/NaN 阈值拒绝。独立只读代码复核未发现本次改动的阻塞问题。

runner/campaign 测试仍使用替代训练/评测脚本和模拟提案，adapter 测试使用注入生成器。本次没有重新进行真实 ISP campaign 训练，没有运行真实 Naive 或 ARIS。上一轮 CPU 合成训练证据保留原样，见 [上一轮验证](review-fix-verification.md)，不计作本轮 live 证据。

## 真实 pilot 状态

当前环境未找到 `aris` 和 ARIS 配置，默认 `127.0.0.1:8000` 无监听，也没有 NVIDIA 设备。未获得可用的其他 Naive 服务地址或真实数据配置，因此本次没有执行真实 Naive＋ARIS pilot。

后续需要可运行的 ARIS 宿主、真实 Naive executor、实验 GPU，以及填写实际路径和固定预算的 ISP 配置。执行现有 [pilot 入口](aris-campaign-task.md#real-naive-pilot-baseline-plus-three-proposals)，保存三次“读取前轮结果→生成不同配置→实际执行”的原始记录。没有涨分仍可验收流程；本次修改不构成真实 PSNR 提升、多 seed 或 TM 归因的证据。
