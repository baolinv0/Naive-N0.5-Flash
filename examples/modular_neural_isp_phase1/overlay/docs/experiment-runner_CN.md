# 五组实验编排器使用说明

本工具把已授权的单 campaign 接口组织成五组配对搜索、统一冻结确认和预注册分析。研究依据是[科研循环实验方案](plans/naive-research-loop-experiments-20261005_CN.md)，部署前置条件见[部署与验收手册](validation-manual_CN.md)。命令行入口为 `python -m tm_research.experiment`，执行目录是已应用 overlay 的 ISP runtime 根目录。

工程 fixture 可以验证配置、请求记录和执行接口。它不代表真实 Naive 闭环 W 已验收，也不代表五组主实验完成、PSNR 提升或科研策略优势。真实 Naive 权重路径、Qwen-VL 权重/服务、TRAIN/DEV/CONFIRM 数据、GPU 和全部资源授权仍须由部署者提供；示例中的 `PLACEHOLDER`、`null` 不是已有授权。

## 1. 准备冻结 manifest

在 ISP 环境安装已有训练与开发依赖，再安装 `requirements-experiments.txt` 中固定版本的 Optuna。模型推理环境独立准备，本工具不下载大模型，不启动权重服务。

```bash
python -m pip install -r requirements.txt -r requirements-phase1-dev.txt \
  -r requirements-experiments.txt
python -m tm_research.experiment --help
```

复制并编辑 [experiment.example.json](../configs/experiment.example.json)。相对路径按 manifest 文件所在目录解析。示例有五组、十个搜索块、每块三个新的确认 seed，是待评审的主实验形状；它没有填入预算，不能直接运行。搜索 seed 和确认 seed 是示例安排，正式冻结前须核对初始化与预注册要求。

必须填写或核对以下输入：

- `authorization_ref` 和 `source_revision` 与实际有效的 `control_template` 完全一致。control 必须已有完整数值资源合同；外层不能替单 campaign 新增训练 GPU、数据访问权或确认授权。
- `baseline_config` 固定实际 TRAIN/DEV、初始化、训练轮数和 P512 等协议；普通搜索的 `test_permission=false`。`profile_ref` 指向真实 DEV profile，不使用该功能时显式设为 `null`。
- `max_trials` 包含 baseline 和失败尝试，不能超过原 control 或 150 个离散配方。五组均提交完整四字段；仅完整快慢组的 plateau 阈值设为 2，其余设为 `K+1`。
- `experiment_limits` 填写全实验训练 GPU 小时上限和墙钟上限；`campaign_walltime_seconds` 是各 campaign 从初始化起相同的墙钟上限，包含排队和确认等待，不得超过全局上限。`campaign_model_limits` 为每个 campaign 填写相同的正整数调用次数、输入/输出预约单位和累计上限；`model_limits` 是外层全局上限，须覆盖全部 Naive campaign 的相同分配。单调用上限也须兼容；正式五组十块有30个 Naive campaign，不能只填一份或用共享总额代替等额上限。非 LLM 组不预约模型调用。
- `models.naive` 填真实 `base_url` 和固定模型/权重身份。需要凭据的服务使用 `api_key_env` 指定既有环境变量名称，不在 JSON 中写密钥。
- `confirmation` 只有在 control 已明确授权后才保留。正式独立确认须在 control 中配置 `kind=independent`、授权 CONFIRM split 和全部确认 seeds；示例不会替你创建这些权限。无确认授权时设为 `null`；`analyze` 要求预注册确认 seeds，否则返回明确错误。

本地 Naive 权重由部署者在单独环境加载。现有文本 adapter 的形状如下；路径必须替换成当前环境可访问的真实权重，版本、依赖和设备按[Naive 部署说明](naive-setup.md)核对：

```bash
python -m tm_research.naive_adapter \
  --model-id /PLACEHOLDER/actual-naive-weights \
  --host 127.0.0.1 --port 8000
```

服务须提供流式 Chat Completions、完整 JSON 回复、正常 `stop` 和 `[DONE]`。provider 保存原始输入、响应和失败；截断或无效 JSON 不经宿主修补为成功，也不自动重发未知请求。没有 endpoint 时只能完成配置与 fixture 检查。

## 2. Naive 与 Qwen 的角色

主实验的 Naive 是提案者；`slow_reviewer=naive` 是默认值。完整组的 slow 使用同一 Naive 的新 messages 会话和独立请求 ID，不能称为两个统计独立专家。

slow 若选择 `propose`，须给出完整四字段 `requested_change`。编排器直接执行这个已批准配方，并保留原始 slow sidecar；提案的 hypothesis/observation 取自其 prediction/observation，不再调用 Fast 替换配方。slow 的停止或诊断决定按原生门禁处理，hold 不会被自动解除。这一提示与调用规则应在真实先导前冻结。

开源 Qwen-VL 的 `models.qwen.enabled` 默认是 `false`。启用后，辅助描述必须基于授权 run 目录中的有限 DEV PNG/JPEG 图像字节；仅给路径不能证明模型看过图片。两种 rich 组使用一致的抽样与提示，Scalar 不获得诊断或 Qwen 文本。Qwen 输出的 `observations/limitations` 保存来源，不能修改 PSNR、winner、预算、路由或统计结论。权威数值来自原生独立重载评测。

如果明确把 `slow_reviewer` 改为 `qwen`，这是另一个 `qwen_review` 变体。必须固定 Qwen 模型身份和角色并单独披露，不能把它混写成默认同 Naive slow 的主实验结果。CONFIRM/TEST 不发送给任一模型。

## 3. 一次一步执行

```bash
python -m tm_research.experiment validate --manifest /PLACEHOLDER/experiment.json
python -m tm_research.experiment generate \
  --manifest /PLACEHOLDER/experiment.json --output /PLACEHOLDER/experiment-output
python -m tm_research.experiment status --experiment /PLACEHOLDER/experiment-output
python -m tm_research.experiment step --experiment /PLACEHOLDER/experiment-output
python -m tm_research.experiment step --experiment /PLACEHOLDER/experiment-output --wait
```

`validate` 只校验 manifest；`generate` 固定输入并生成每组/块独立的 config、control、runs 与确认计划，不启动模型或训练。相同输入重复生成可返回已有索引，输入变化必须使用新的输出目录。输出不能覆盖数据目录。

`status` 读取状态。`step` 每次只调用一次 runner，最多推进一个有成本动作；`--wait` 对现有训练作业最多等待30秒再收集，不循环跑完全部 campaign。独立确认评测本身仍由原生作业 watchdog 限时。已结算 campaign 按冻结的随机顺序轮转，既有活跃作业优先收集。需要继续时，根据输出状态显式再次调用。没有可执行工作的 `wait/diagnose` 会暂停；非法提案、预算耗尽、HTTP 结果未知或提交未决时，应先检查保存的请求和原生状态。

外层 request/receipt 和 native campaign 是恢复依据。不要直接编辑 `campaign.json`，不要删除失败记录再补跑，不要对未知提交手工执行第二次 launch。若模型请求/提交已发生而结果无法证实，保留不可判定状态并完成明确恢复，再继续下一步。连续两次提案拒绝达到有界上限时暂停，不强行 finalize。工具不会仅因 CLI 返回退出码 0 就保证研究完成；暂停和等待都须读 JSON `status`。

所有搜索组/块冻结后，才可推进统一确认：

```bash
python -m tm_research.experiment confirm \
  --experiment /PLACEHOLDER/experiment-output --wait
python -m tm_research.experiment status --experiment /PLACEHOLDER/experiment-output
python -m tm_research.experiment analyze \
  --experiment /PLACEHOLDER/experiment-output --output /PLACEHOLDER/analysis-output
```

每次 `confirm` 也只推进一个有成本动作，不隐式跑完整个确认矩阵。primary plan 在搜索前冻结，失败遵循 `preserve_no_retry`。统一确认及统计不会重新打开模型搜索；独立 CONFIRM 仅由评测程序读取。

## 4. 成本预约与实际使用

Qwen 图像请求若没有显式 `input_token_reservation`，会按包含 base64 的 UTF-8 请求长度预约；真实 P512 图片很容易因此在 HTTP 前超出额度。应先用目标 Qwen 服务校准视觉+文本输入预约和硬上下文限制，再冻结该字段与相同 campaign 上限；不能为了发送图片把实际 tokens 虚报为0。辅助请求被拒绝时保存 unavailable，不伪造诊断。

模型请求共享同一个全局 ledger，同时受每 campaign 的相同上限约束。预约在同一锁下同时检查全局和 campaign 余额，任一不足就拒绝；Naive fast/slow 和 Qwen 辅助/审查调用都先预约再结算。失败、截断和未知状态也保留预约与原始记录。文本输入默认以 UTF-8 请求字节数加每条 message 的 512 个模板预约单位保守记账；显式文本预约取该值与配置值的较大者。图片的校准预约由服务规则决定。预约单位不是 tokenizer 测得的 token，也不是已证明的硬上下文界限；服务端仍须限制输出（包括 reasoning）及上下文。

ledger 区分 booked/reserved 与 actual。服务没有 usage 时，实际输入/输出/总 tokens 和金额保持 `null`，不填 0；预约值不能当作实际费用。没有价格或计费记录时不能宣称推理免费。模型与训练 GPU 成本分别披露，不能以少几个 trial 推出总成本更低。

正式五组、十块、每 campaign `K=12` 搜索 slots 和 `S=3` 个 baseline/winner 配对 seeds，最多是 `5×10×(12+2×3)=900` 次训练，另有最多 300 次独立 CONFIRM 评测。当前原生确认的独立评测也按完整作业上限预约，不能只拨 900 个作业上限的 GPU 时间。

令 `H = allocated_gpus × job_walltime_seconds / 3600`，每 campaign 至少分配搜索 `12H` 与确认 `4×3H`，即 `24H`；主实验全局分配至少为 `5×10×(12+4×3)×H = 1200H`。校准等保留另加，实际耗时单独测量。每个 control 的 `total_gpu_hours` 必须含自己的 confirmation/calibration reserves；外层 `total_gpu_hours` 必须覆盖全部 campaign 分配。默认不共享 baseline/确认来缩减账目。

## 5. 分析输出与结论范围

`analyze` 从经原生身份核对的 primary 确认提取数值记录，先平均 campaign 内确认 seeds，再计算同块策略差。预注册策略、块数和 `delta_strategy` 从冻结索引读取，不从“成功完成的子集”重构。`analysis.json` 保留原始数值记录、scope、变体和 primary 资格；工程 fixture、original DEV 确认或 Qwen reviewer 变体不能进入默认独立确认主比较。另输出 `campaigns.csv`、`comparisons.csv`；安装 matplotlib 时生成 `confirmation_gains.png`，缺少该可选依赖时明确记录图表不可用。

三个主比较使用符号翻转检验、同一检验的区间反演和 Holm 校正；同时区间为 98.33%。缺失/失败不补成 0，覆盖不完整时主结论不可判定。小 R 不能给出有限区间时标记 unbounded 并保留 `null` 边界，JSON 不写 Infinity/NaN。探索性子集或工程 fixture 的数值可以用于检查输出，但不能据此宣称正式研究获胜；报告保留 `evidence_mode` 和冻结实验身份。

CLI 的协议门禁和授权图像路径检查没有实现强宿主隔离。真正的 CONFIRM/TEST 保密、服务账号/挂载权限、GPU 调度和工具白名单仍由外部宿主配置。现有 Naive W 自动验收只适用于规定的 adapter/capture 路径；外层模型请求日志或第三方服务不是自动 W 接受证据。

## 6. 本次交付的验证证据

2026-10-05 在已准备的 CPU 环境、当前 source overlay 上运行完整 pytest：646 passed、19 subtests passed，无失败/错误/跳过，20 条已有依赖弃用警告。覆盖五组合成子进程 fixture、实际本地 SSE 请求→原生提交、真实 DEV 图像字节传递、两级预算、拒绝/恢复/超时和统计独立枚举。该结果不代表真实 Naive/Qwen 权重推理、真实图像质量提升、W 验收或900次主实验已经完成。
