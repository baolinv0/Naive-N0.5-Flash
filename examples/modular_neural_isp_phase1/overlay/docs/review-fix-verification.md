# Review 修复与实际验证

日期：2026-09-29。范围为审查建议的 A/B/C：统一裁判、开放训练 recipe、接上结果驱动 campaign。当前代码已超出原始 baseline-only 阶段，但不将其表述为完整原方案全部验收。

## 审查项落实

| 问题 | 本轮处理 |
|---|---|
| runner 无法传 loss/LR/optimizer | original/MSE/L1、Adam/AdamW、learning_rate、weight_decay 配置入口，保存生效 recipe |
| best 按旧 batch PSNR | validation、best、独立重载和比较共用 float64 逐图 PSNR；MSE下限1e-12 |
| 每轮暴露 test、分辨率不一 | 日常只跑 dev；默认 P512，smoke P256；finalize 后单独 final-test；quarter/full 显式单列 |
| completed 接受无效结果 | checkpoint存在、独立加载成功、覆盖图数与有限分数；执行状态和 valid/no_gain/improved 分开 |
| Naive 截断/参数 | 传播 stop/length、屏蔽未完成 reasoning/tool、required/type 校验 |
| 缺少反馈驱动迭代 | 串行 campaign、真实结果引用、提案执行/比较、预算停止、持久状态与 ARIS任务入口 |
| 不完整 HDF5 缓存 | 完成标记缺失时重建；不引入复杂缓存系统 |

模型结构与官方原始损失类未修改。当前属于配置搜索，evaluator 与训练仍共用固定源码；不宣称存在独立隔离的裁判服务。

## 实际执行的测试

环境：Linux、Python3.12.14、PyTorch2.5.1+cpu，无 CUDA。

`OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python -m pytest -q`：**49 passed**、19 subtests，25.53秒。20条警告为依赖弃用提示。覆盖指标排序反例/不同 batch 划分、缓存、真实训练入口的 best保存选择、recipe梯度/优化器、进程重载、三次模拟反馈变化、dev/test分开、固定预算与Naive协议。最后一次引用分数容差修正后，相关 runner/campaign 22项重新通过（13.83秒）。

最终交付又按“官方源码归档＋当前overlay”的干净安装路径复验：**49 passed**、19 subtests、30.23秒，记录于install_pytest.log。

单测中 runner使用替代训练脚本，adapter使用模拟生成器；这些不冒充真实模型或服务实测。真实模型证据单列如下。独立审查未发现当前范围的阻塞问题，见 integration-review.md。

## 真实 ISP CPU 合成 campaign

所有 trial：官方同一 style-0 初始权重、seed7、1 epoch、2张训练图/1张dev、batch1、输入256、P256、相同Cosine调度规则。每个 trial 都实际训练，随后在另一进程加载 checkpoint 并计算 dev 分数。

**提案由本轮助手逐次读取结果后编写；没有运行 Naive 模型或 ARIS-Code 进程。** 它证明controller到真实训练/评测的数据流，而不证明Naive研究能力。fixture的train/dev/test共用合成图形构造，不能用于泛化或真实画质结论。

| Trial | Loss | LR | dev PSNR (dB) | 判定 |
|---|---|---|---:|---|
| exp_001 | original | 0.0001 | 22.67446859 | valid |
| exp_002 | mse | 0.0001 | 22.26742613 | no_gain |
| exp_003 | l1 | 0.0001 | 22.47858588 | no_gain |
| exp_004 | original | 0.0002 | 24.80136079 | improved |

实际决策轨迹：原recipe结果→试MSE；MSE无收益→试L1；两种纯像素loss均不及原recipe→保留原recipe并只改变LR。每条proposal引用已完成run与实测分数，见证据目录proposal_1/2/3.json及next_1/2/3.json。

控制器达到4个trial后以max_trials停止，冻结exp_004，再单独运行合成test，得到24.80138451 dB、1张、P256、valid。日常trial未执行test。这个分数仅用于验证最终测试入口，不代表真实封存测试收益。

另外单独跑过原recipe短训与重载。同一权重两次独立评测的差为0.00001531 dB，处于0.0001 dB内；不宣称浮点计算逐位一致。原始记录见repeatability.json。

## 运行方式与限制

本环境的工具沙箱会在命令退出后终止后台子进程。初次campaign init产生的queued任务尚未启动，因此以内部_worker前台执行同一任务后收集；后续三次submit均用`--wait`并保持执行session。常规主机可保留后台worker；受限宿主使用`--wait`。没有将被杀worker自动恢复或优化器续训描述为已实现功能。

尚未实测或尚未实现：真实Naive+ARIS的连续三轮提案/live transcript、真实相机数据P512完整训练、多seed确认、仅GTM/LTM更新的归因、GPU队列并行、完整optimizer/RNG恢复。这些在后续资源/阶段补齐。本轮没有宣称真实数据PSNR提升，也没有把合成结果作为论文精度复现。

## 证据位置

GitHub本工程目录`evidence/review_fix/`含完整campaign状态、三份提案与对应反馈、各run配置/实际命令/日志/指标/逐图CSV、选中权重与配置、冻结测试和pytest日志。路径记录保留当时工作环境；复跑时通过示例YAML指定本机路径。
