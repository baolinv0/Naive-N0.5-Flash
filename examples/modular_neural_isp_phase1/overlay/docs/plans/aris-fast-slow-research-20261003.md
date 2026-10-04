# ARIS 快慢科研系统 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** 在现有 Naive＋ARIS＋photofinishing 工程上，形成“证据丰富的快循环＋事件触发的慢循环＋独立确认”，提高单任务研究的可解释性、可靠性和有效产出。

**Architecture:** 复用 ARIS 宿主、Naive adapter、runner 和 campaign。新增小型证据汇总、授权检查、慢循环路由和确认模块；模型负责提出与解释假设，确定性代码负责实验执行、比较和边界检查。慢循环是 ARIS 的按需任务，不是第二个常驻 Agent runtime。

**Tech Stack:** 现有 Python/PyTorch、标准库 JSON/CSV、现有 Bash/CLI、现有 YAML 配置；首版不新增数据库、消息队列、向量库或付费服务。

**Spec:** 本文第 1–11 节为设计规格，第 12–14 节为依赖该规格的实施任务与验收。方案状态为建议，尚未修改代码或执行训练。

日期：2026-10-03。面向：当前 photofinishing 研究工程负责人、实施 agent、独立 reviewer。

修订：v1.1，整合“DEV 评测后自动生成诊断证据包”的反馈。优先交付复用 CSV 的 V1，再按需要加入量化前区域诊断 V1.1；两者均不改变现有选优规则。本文中的 V1/V1.1 指诊断能力发布阶段，不代表功能已实现。

## Global Constraints

- 以 docs/aris-delegation-policy-20261002 的 91d8577f63231344c8b03c555dbde9511212db8b 为建议开发基线；该提交包含 fix/tm-adaptive-research 的 6d5ea1323c6834d0594ca7ad8e2edd04179593b4。
- 当前问题为 paired photofinishing、P512、mean per-image RGB PSNR；每个搜索 campaign 固定模型、数据、初始化规则、seed、训练预算和评测协议。
- 首版允许变化仍是 loss_family、optimizer、learning_rate、weight_decay；不开放任意模型代码生成。
- 保留 completed、valid、improved 的区别，以及 effective best、raw_best、min_delta 的既有含义。
- 现有严格三字段 proposal JSON 保持不变；研究预测和最新反馈确认放在独立 decision 文件，由新 CLI 参数引用。
- 新 control 参数独立于 runner 科学配置，不把政策字段塞进 normalize_config 当前拒绝的 YAML 字段集合。
- 旧 campaign 不原地升级、不重算历史 best、不补造预测；没有 min_delta 的旧记录继续执行旧零阈值兼容规则。
- 当前 live pilot 不访问 TEST；确认集必须真实存在、身份明确且已授权，不能把 TEST 改名使用。
- 功能实现优先；不新增内容哈希、防篡改体系、通用审批平台或机械 reviewer 评分。
- 外部 GPU、Naive 服务、真实数据路径和数值预算必须来自实际可用资源与既有授权；本方案不代填或授权支出。

## Review Focus

1. 逐图记录乱序、重名或缺失时必须正确对齐或明确无法比较，不能靠 CSV 行号。
2. 当前 trial 成为新 best 时，诊断仍须与提交前 best 比较，不能与自身比较。
3. 宿主在 worker 启动附近中断时，不应丢失 active run 后重复训练；运行状态未知时停止新提交。
4. 确认数据不得用于 checkpoint 选择；同配方跨 seed 复训不得触发搜索去重，也不得改写搜索 best。
5. 缺分组、缺视觉能力、缺成本数据或旧格式时，明确 unavailable/unknown，不能生成看似完整的研究结论。

## 1 结论与范围

推荐采用中等规模的增量改造：保留现有执行闭环，增加“证据装配、边界约束、慢循环决策、确认阶段”四类能力。

可选路线比较：

| 路线 | 改动 | 优点 | 限制 |
|---|---|---|---|
| 仅改提示词和任务文档 | 人工读取日志并写复盘 | 可立即试运行 | 证据消费不稳定，恢复后容易漏读，确认依赖手工 |
| 增量快慢系统，推荐 | 小型 Python 模块＋现有 CLI | 可验收、可恢复、可逐步上线 | 需要明确产物结构与兼容规则 |
| 新建科研平台 | 新 runtime、调度器、通用任务语言 | 长期通用性较高 | 当前任务尚未验证，工程成本先于研究收益 |

快循环：在一个已冻结问题内，读取实验结果，提出允许的改变，执行并选择候选。

慢循环：在停滞、证据矛盾、协议问题或结案时，重新判断解释是否成立、下一实验是否值得、是否需要确认或扩大范围。

慢循环不等于多想几次，也不等于每轮请人批准。其核心输出是“下一种行动及依据”。

本方案不把原问题扩展为通用语义分割、任意恢复网络或跨任务 AutoML。任务迁移留待当前流程和收益得到验证。

## 2 当前代码事实与差距

核查时三个 HEAD 为 main 91add900、fix 6d5ea132、docs 91d8577，未发生变化。

| 现有位置 | 已具备 | 需要改善 |
|---|---|---|
| tm_research/campaign.py | baseline、next、submit、停止、冻结、历史证据、effective/raw best | next 主要给总分；缺少诊断包、研究行动路由、确认阶段 |
| tm_research/runner.py | 独立 DEV 重载、样本数/协议/有限值检查、状态文件 | 未限制整次作业时长；未可靠识别死 worker；启动与 active 持久化存在间隙 |
| photofinishing/test.py | 逐图 PSNR/SSIM、输出 PNG-8、粗略耗时 | 缺稳定样本映射、冻结分组、逐图对照和可用于诊断的浮点附加统计 |
| photofinishing/train.py | val_mean_psnr、checkpoint 列表等结构日志；细项在文本/TensorBoard | 需要统一小型 epoch telemetry，避免 agent 解析大量终端文本 |
| docs/aris-campaign-task.md | 已有委派与证据原则 | 需要明确快慢路由、decision 文件和恢复入口 |
| naive_adapter.py | 协议桥接 | 首版无需修改，不承担研究决策 |
| 配置加载 | runner.normalize_config 严格限定字段 | 新 control 使用独立入口；仓库没有单独 config.py |

必须保留的证据边界：

- 已提交四次 CPU 合成数据训练记录；不是 Naive＋ARIS live 实测，也不证明真实相机泛化。
- 原生评测 PSNR 来自浮点 tensor；输出图是 PNG-8，不能用保存图重算并冒充同一指标。
- test.py 的 time_seconds 未做完整设备同步和部署基准设计，不能当作端侧延迟。
- TensorBoard 图片只覆盖相关最后一批样本，不代表全部失败样本。
- min_delta 的重载校准只处理评测数值波动，不处理训练随机性和适应 DEV 的选择偏差。
- 一个分组表现差只能提示解释，不能自动定位某个 ISP 模块的因果责任。
- 当前缺口主要是加工和传递：逐图结果已经落盘，下一轮决策接口主要消费 aggregate PSNR；不能把当前系统概括为“只产生一个分数”。
- 当前 metadata 明确使用 cam_illum、ccm，不能据此推断存在 ISO、曝光时间或真实噪声强度。
- original 使用多个中间输出及其约束；mse/l1 的 PixelLoss 仅监督最终 sRGB。original→mse 是监督位置和损失构成的组合改变。
- based_on 是历史证据引用；partial recipe 当前合并到 effective best 的 recipe，每轮仍从固定初始化重训。

## 3 组件关系与职责

~~~mermaid
flowchart TD
    H["人确定问题和资源边界"] --> C["冻结控制契约"]
    C --> A["ARIS与Naive"]
    A --> P["快循环提案"]
    P --> R["现有runner与独立DEV"]
    R --> E["证据汇总"]
    E --> Q{"下一行动"}
    Q -->|"继续"| A
    Q -->|"触发慢循环"| S["竞争解释和区分实验"]
    S -->|"范围内"| A
    S -->|"需改变边界"| H
    Q -->|"冻结候选"| F["独立确认"]
    F --> K["有限结论和研究经验"]
~~~

| 角色 | 工作 | 不应承担 |
|---|---|---|
| 人类研究 owner | 定义价值、主要目标、可变范围、资源及重要范围变化 | 每个 trial 的机械点击批准 |
| Naive proposer | 消费证据，提出预测、反证条件及 recipe | 修改裁判后宣布获胜 |
| ARIS host | 工具循环、会话、文件读写、按需 reviewer | 第二套训练调度/恢复框架 |
| runner/campaign | 真实运行、结果资格、比较、冻结、停止 | 判断语言解释是否科学正确 |
| slow reviewer | 竞争解释、决定最小区分实验、限定结论 | 用共识替代实验或自动扩大权限 |
| confirmation | 复验冻结候选，生成配对证据 | 搜索新的候选或重新择优 |

日常运行保持一个 proposer；独立 reviewer 在慢循环事件或结案时启动。多视角同时讨论用于新研究问题或重大矛盾，不用于每一次小幅 LR 调整。

## 4 文件与事实来源

以下为运行产物的逻辑位置，不要求移动现有 run 目录。

| 产物 | 唯一职责与写入者 |
|---|---|
| campaign/control.json | 初始化时冻结的人类授权记录、范围、资源上限、确认/TEST 权限；后续范围变更使用新 campaign |
| campaign/campaign.json | 当前科学配置、trial 状态、实测结果引用、best/raw_best、阈值和停止条件；controller 写 |
| run/state.json、train/、dev/ | 实际进程和训练评测产物；runner/评测脚本写 |
| campaign/data_manifest.json | 样本、场景、split、固定标签与来源；在首轮前由数据准备步骤生成 |
| campaign/dev_profile/ | profile.json 记录冻结的图像分组规则；V1.1 增加 roi_profile.json 与固定掩码；一次性生成，共享给所有 trial |
| campaign/feedback/exp_NNN/r001/ | 从已存产物生成的 summary、paired rows、cases；可重建为新 revision，已被引用版本保留，不另造分数 |
| run/observation_config.json、run/dev/roi_metrics.csv | 前者是本次观测配置快照；后者仅在 V1.1 开启时由 float 评测生成，不作为新的主评分 |
| campaign/decisions/decision_NNN.json | Agent 的预测、反证条件、最新反馈确认、解释及行动建议；提交时保存快照 |
| campaign/research_decisions.jsonl | 已提交研究决定及 slow review 的追加式索引；不可改写原来的预测 |
| campaign/confirmation/confirmation.json | 冻结确认协议、arm/seed 状态及结果引用；独立于 search best |
| campaign/report.md、control.md | 可读报告与授权视图；从上面事实生成，不是第二套手工维护状态 |
| research_memory.jsonl | 有适用条件和证据链接的经验记录；结案时追加或追加撤销/替代记录 |

所有实验引用使用 campaign_id＋run_id，避免不同目录都有 exp_001 时混淆。使用可读 ID 和源码提交号，不新增内容哈希。

### 4.1 控制契约

新模式通过 campaign init --control 引入 control.json；现有训练 YAML 保持科学配置。

control 至少包含：

- schema_version、campaign_id、parent_campaign_id、question。
- authorization_ref：用户已作出的具体授权或现成资源配额引用；不能由 Agent 虚构“用户已同意”。
- source_revision、approved_config_ref、protocol_id、primary_metric。
- allowed_recipe_fields，以及已批准时才填写的 LR/weight_decay 数值范围。
- limits：每次 TRAIN＋DEV 作业 walltime、总训练 GPU-hours、实际分配 GPU 数、search trial 上限、确认预算、校准预算；推理费用使用既有 provider/host 配额。
- slow_policy：停滞触发配置；默认建议连续两次 valid no_gain 触发一次 review，初始化前可改。
- confirmation_scope：未授权、仅原 DEV seed 稳定性，或指定独立确认集；可执行的 seed 列表和预算。
- test_permission：当前 pilot 为 false。
- observation_mode：默认 text；视觉能力未实测验证时不得改成 image。
- observation_config：profile_ref、required_for_proposal、案例选择规则及详情访问方式。新模式 V1 的必需项建议为样本对齐、配置差异、全图逐样本比较和文本摘要；ROI、图片拼图与新增 telemetry 默认可选。

科学配置及实际冻结的 min_delta/max_trials 仍由 campaign.json 保存；control 表示获准边界。初始化必须验证实际值在授权范围内，不维护两套可独立改变的运行设置。

授权缺字段不等于无限预算。方案模板可将未配置字段标为 null；新模式 init 在缺少必要实际配置时拒绝启动并返回具体缺项。已经覆盖的授权不再次询问。

control.json 只是流程授权记录及程序检查输入，不是防止任意 Bash 绕过的安全沙箱。若需要防止 TEST 被意外读取，使用现有目录挂载/数据访问配置，不扩展成本项目为安全平台。

## 5 诊断反馈如何生成

### 5.1 先确定样本与场景

data_manifest 每个条目包含 sample_id、scene_id、split、input_relpath、gt_relpath、metadata_relpath、tags、annotation_source、annotation_version。

- sample_id 是稳定唯一键，不能按 CSV 行号比较。
- manifest 只扫描当前获准的 TRAIN/DEV；确认数据在对应授权阶段单独准备，未授权 TEST 不枚举、不读取。
- 当前 basename 只在映射唯一时可转为 sample_id；重名、缺失映射需报出。
- scene_id 来自真实采集关系、连拍或同源图信息。没有可靠信息就标 unknown，不能由文件名不同推断场景独立。
- 已知同源 patch、曝光变体和连拍跨 split 时，报告具体冲突并暂停宣称独立确认；不自行重分数据。
- 当前 pilot 保留原划分；如发现确实需要重划分，走新协议与新 campaign。

### 5.2 一次性 profile：图像分组与图内区域分开

新增 scripts/build_dev_profile.py，复用现有 paired_files/load_image_pair 及实际评测使用的色彩转换、裁剪/缩放、尺寸和坐标约定，生成样本映射和候选无关的 profile。记录 profile_revision、规则参数、来源和评测变换；不运行候选模型，不读取 TEST。旧 run 生成的新 profile 必须标为事后探索，不能声称是实验前已冻结的分析。

| 层次 | 第一版内容 | 含义与限制 |
|---|---|---|
| 图像分组，V1 | 固定输入或 GT 的亮度、暗像素比例、亮像素比例、梯度统计；已有可靠 metadata/人工标注 | 描述哪类整图改善或退化；只能使用实际可得字段 |
| 图内区域，V1.1 | 固定 GT 的低/中/高码值亮度区域，与低/中/高梯度区域的少量组合 | 描述同一张图内哪些位置发生误差，所有候选使用相同 mask |
| 语义区域，后续可选 | 可靠人脸/肤色等标注，或一次性离线模型输出 | 必须保留标注来源及误差边界，不能由亮度或梯度自动替代 |

可从经过同一评测变换的 TRAIN 图像/GT 校准阈值，再冻结并用于 DEV。使用 GT 并不违反 DEV 的开发用途；DEV 仍不能被称为独立泛化确认。若阈值确实基于 DEV 一次性设定，记录来源并将结论限定为探索性。不得看过每轮 candidate 后重新设阈值。

亮度第一版可用归一化 GT 输出码值 R′G′B′ 的加权量 Y_code=0.2126R′+0.7152G′+0.0722B′；这不是线性光亮度、照度或 ISO。参数、值域、阈值及边界归属都写入 profile。梯度算法也需固定；尚无纹理/边缘分类算法时应直接命名“低/中/高梯度”，不能把它们自动解释为平坦/纹理/强边缘。

若用“暗部占比高”划分整图，暗像素阈值应是冻结的共同阈值；逐图分位数会使占比近似固定，不能再用这个占比区分图像。图像标签可重叠，报告 n_images、n_scenes、overlapping 和 unknown，不得相加后当作总体样本数。当前已知 metadata 不支持把“低码值区域”命名为“高 ISO 低光场景”。

ROI profile 额外记录 mask_source、mask 坐标、图像尺寸、有效像素数、min_pixels 规则及聚合定义。阈值、mask 和变换均与候选输出无关。动态发现的高误差位置另记为案例 ROI，不能替代这些固定分析区域。

### 5.3 每轮自动生成的证据包

| 文件 | 内容 |
|---|---|
| summary.json | execution/validity、diagnostics_status、proposal_ready、协议、覆盖、overall、comparisons、actual_changes、带 ID 的 observations、边界和缺失项 |
| paired_rows.csv | 按 sample_id 对齐的当前/参考 PSNR、SSIM、差值、scene_id、comparison_role；每种参照单独识别 |
| paired_comparison.json | 固定 baseline、提交前 best、本轮明确对照各自的配对结果、分组汇总、配置差异与可比性 |
| cases.json | 退化、持续困难、改善、普通四类，每类最多 2 例，去重后最多 8 例；理由和证据来源明确 |
| curves.json | 已有 epoch/val PSNR、checkpoint 位置，以及新 telemetry 可得项；不从缺失日志虚构曲线 |
| details_index.json | 输入、GT、输出、日志等路径与来源；仅包含获准 TRAIN/DEV |

固定个数是信息压缩默认值，不是统计阈值。agent 可以按需查全部逐图表，摘要不得隐藏总体覆盖和缺失情况。

先冻结参照身份，再启动训练；新 trial 成为 best 后不能改写本轮参照。

| 记录 | 唯一含义 |
|---|---|
| baseline_run_id | campaign 的固定有效 baseline，用于累计变化 |
| best_before_run_id | 提交本轮前的 effective best，用于记录保留决策背景 |
| comparison_run_id | 本轮实验的实际比较对象；decision 可显式指定有效历史 run，未指定时默认为 best_before |
| construction_base_run_id | partial recipe 实际合并依据；按现有实现等于提交前 best，不能由 based_on 冒充 |
| based_on_run_id | Agent 在旧 proposal 中引用的证据 run，不自动变成配方父节点或训练起点 |
| init_checkpoint_ref | campaign 的固定初始化权重，各轮从这里重训；不从 best/based_on checkpoint 接续训练 |

显式 comparison 不存在、无有效结果或协议不兼容时明确拒绝该比较/提案，不能静默换成 best。参考角色可以指向同一 run，但含义分别记录。baseline 自身没有前序 trial，相关字段为 null，不伪造自我增益。

controller 保存完整 resolved_config，并与 construction base 自动计算 actual_changed_fields；每个 comparison 另外计算相对该对照的 config_delta。仅排除输出目录、日志目录、时间戳等运行身份字段；TRAIN/DEV 路径或 manifest、初始化权重等科学资源身份仍需核对，不能笼统忽略所有路径。检查模型、数据、seed、预算和评测等固定科学字段一致。既保留四个 recipe 字段的实际差异，也标明由其导出的变化，例如 LR 改动影响当前 cosine scheduler 的 eta_min=LR/100。不可只抄 Agent 的 requested_change。

original→mse/l1 必须附 supervision_change：从多个中间输出及组合损失变为仅最终 sRGB 的像素损失。该结果可描述整个监督配置的效果，不能单独识别“损失函数形式”或某个 TM 模块的原因。固定科学条件不同的 run 标 comparable=false，只展示差异，不能输出纯 recipe 的因果结论。

主要聚合定义：

- Δ_i = PSNR_current(i) − PSNR_reference(i)。
- overall_delta 是同一完整样本集合上的 mean(Δ_i)，不是 pooled MSE PSNR。
- group_delta 是固定组内 mean(Δ_i)，同时报告组大小和可用 scene 数。
- 每个组同时报告改善/近似持平/退化数量和案例 ID；样本覆盖不全时只输出明确标注的 matched_subset 描述，不能冒充完整 DEV 的 overall_delta。
- 胜/平/负若使用容差，该容差须单独定义并冻结；不能把全局 min_delta 直接当逐图容差。
- 分组太小不自动拒绝数据，但标 small_sample、给出真实数量，不写稳定趋势结论。
- 组之间重叠时不对组均值再求平均来恢复 overall。
- 原始评测有效但诊断文件缺失时，保留 valid 分数。诊断使用独立 complete/partial/failed/unavailable 状态；协议/样本身份不一致时暂停受影响的比较。

required_for_proposal 在初始化时指定必需章节。必需证据缺失则 proposal_ready=false，下一行动为 diagnose，优先重建后处理产物；不为了补一份 JSON 重训。可选 ROI、展示图或训练细项缺失时允许带缺口继续。diagnostics_status 本身不改 valid、improved、best、raw_best、min_delta，也不新增画质晋级条件。暂停原因消除后不应每次 next 反复触发同一 review。

必需章节按实验状态定义适用性：baseline 无前序参照，其配对增益和实际变更标 not_applicable，不视为缺证；已明确结束的 invalid/failed trial 提供真实失败状态、原因和可得日志即可，不能强求不存在的有效逐图比较或伪造数值。能否继续由现有停止条件、运行状态和故障处理决定；运行状态 unknown 仍阻止新启动。

### 5.4 八个案例怎样选

| 案例类型 | 默认上限 | 选择规则 |
|---|---|---|
| 本轮退化 | 2 | 对 comparison 的逐图 Δ 最低，且为负；无退化不强填 |
| 持续困难 | 2 | 最近 K 个有效且配方不同的 run 中持续位于固定定义的低分集合；候选规则为 K=3、每轮末位 20%，均须事先冻结 |
| 本轮改善 | 2 | 对 comparison 的 Δ 最高，且为正；无改善不强填 |
| 普通对照 | 2 | baseline 后、查看候选前按固定规则选定的非极端样本；样本不足时可退化为固定全体抽样并标注 |

持续困难只表示探索历史中的反复低分，不等于多 seed 复现或同一原因。有效历史不足 K 时标 evidence_limited，不称“长期失败”；可展示当前低分样本，但必须另记 selection_reason=current_low_score。普通样本若随机抽取，使用独立固定 RNG，不能消耗训练 RNG 状态。

按照上述顺序和 sample_id 排序打破并列；重复样本只生成一张卡，记录所有入选理由，从各类剩余候选补位。原始全图指标、历史窗口、名额和去重规则写入 profile；上限八例不是必须凑满八例。保留完整逐图表，Agent 随时可检查极端样本之外的结果。

每张案例卡保存 case_id、sample_id、选择理由、固定标签、baseline/comparison/candidate 分数、可得区域变化、同一坐标系的 ROI、输入/GT/输出和图例路径。V1 只需文本卡和已有图像索引；不把拼图生成设为闭环前置条件。

V1.1 可用浮点误差变化图定位动态 ROI，保存 crop 坐标和定位规则。对照、候选、GT 使用同位置同尺度裁剪；误差图共用色标范围。自动选出的最差 crop 属于事后诊断，不能替代固定 mask、成为新排行榜或证明大部分图像都有同一问题。

### 5.5 训练观测与浮点区域测量

V1 先从现有训练产物提取 best epoch、实际训练 epoch、验证序列、末段趋势和缺失项；没有训练侧同指标时，不推断训练/验证差距。波动、平台期等摘要使用显式窗口和算法，保留原数列入口。新增 train.py telemetry 不作为 CSV 反馈上线的前置条件。

需要时在 train.py 增加少量 epoch telemetry，不改变优化器步进、随机数调用或 checkpoint 选择：

epoch、learning_rate、loss_family、带定义的 train_loss、val_mean_per_image_psnr、num_images、finite、elapsed_seconds、checkpoint_path。

不同 loss family 的总损失量纲不同，不用于跨 recipe 排名。训练/验证差距若需要比较，只能使用同一定义的指标，并说明训练数据含增强、DEV 不含增强等差异。没有相同度量就不计算“过拟合差距”。

V1.1 在 test.py::test_net() 中，对当前同尺寸、同对齐的浮点 output/GT 计算附加诊断；保持原 PSNR 路径及张量不被诊断代码原位修改。核心评测与可选诊断分开记录错误。只对保存 PNG-8 计算的量必须标 display_only，不能冒充浮点评测。

第 i 张图的固定区域 g 使用 MSE(i,g)=Σ_x M(i,g,x)·Σ_c(output−GT)² / [3·Σ_x M(i,g,x)]。PSNR 的值域、epsilon 和边界按原评测约定明确记录，诊断中若使用 epsilon=1e-12 则写入配置；不要未经核实改动主 PSNR。

| 诊断量 | 固定定义要求 | 能描述，不能直接证明 |
|---|---|---|
| 区域 MSE/PSNR | 同一 mask、有效像素数、min_pixels、值域；空或太小区域为 null 并给原因 | 区域误差大小，不定位网络内部责任 |
| 码值亮度偏差 | mean(Y_output−Y_GT)，正值为偏亮 | 输出码值倾向，不是物理曝光偏差 |
| 低频 RGB 误差 | 固定低通核、sigma、边界方式及 MSE 归一化 | 缓慢变化的 RGB 误差，不等于感知色差 ΔE |
| 梯度误差 | 固定梯度算子、尺度、范数和边界方式 | 局部变化不匹配，不单独证明过度去噪、锐化或 halo |

对空间滤波先在完整图像上应用相同算子，不能先将 mask 外置零再滤波。若要声称误差来自区域内部，按滤波支持半径侵蚀固定 mask，并报告侵蚀后有效面积；小区域消失就返回 unavailable。具体核与阈值属于 V1.1 配置，启用前固定，不在 Agent 每轮决策中变化。

roi_metrics.csv 使用长表，每行对应 sample_id＋roi_id＋metric_name，保存 value、n_pixels、region_fraction、effective_mask_id、valid_reason、profile_revision。这样低通、梯度和未侵蚀的像素指标各自拥有正确的有效面积；不能用一个 n_pixels 冒充所有度量的面积。candidate/reference 的同一种度量使用完全相同的有效 mask，并按同一有效样本集合比较。首要区域摘要是每图 ROI 指标的等权均值；如另报像素加权 pooled MSE/PSNR，必须单列 micro，不能与每图宏平均 macro 混用。像素数只说明区域覆盖，不能把像素当独立图像扩大样本量。

### 5.6 输入 Agent 的三层内容

- 固定研究合同：任务、四个允许字段、固定初始化/预算/seed/协议、PSNR 选优和 TEST 边界。
- 当轮摘要：最新 execution/result、actual_changes、参照身份、overall 与分组变化、带来源的 observation ID、最多 8 个 case 索引、最近几轮结果、当前 best、剩余资源及可执行行动。
- 按需详情：某 run 的逐图差、曲线、指定 group/case、相关 TRAIN/DEV 日志。

配置、命令、checkpoint 和完整日志属于按需原始证据，不全部塞进每轮提示词。观察 ID 在一个反馈版本内可为 O1/O2，完整引用使用 campaign_id/run_id/feedback_revision/observation_id。每条 observation 包含 metric、comparison_run_id、group/roi、真实覆盖数、finding、source、case_ids 和 interpretation_limits。程序输出测量事实；“过度平滑”“收敛不足”等原因只能出现在 Agent 假设中。

feedback_revision 使用 r001/r002 等可读序号。补诊断、升级分析规则或修正测量后产生新版本；被 decision 引用过的 summary、配对表、案例卡及 observation 保留原版本，不能将旧 O1 改写成另一事实。latest 仅作读取新结果的入口，历史 decision 指向具体版本。相同来源和相同规则的重复 collect/next 复用已有版本，不因重复查询产生版本；不需要内容哈希体系。

建议的摘要合同如下。所有 run/ID/数值仅演示结构；null 表示实际生成时必须填写实测值，不能作为示例结果提交：

~~~json
{
  "schema_version": 1,
  "feedback_revision": "r001",
  "run_id": "exp_002",
  "comparison_run_id": "exp_001",
  "best_before_run_id": "exp_001",
  "construction_base_run_id": "exp_001",
  "actual_changed_fields": {"loss_family": {"from": "original", "to": "mse"}},
  "selection": {"dev_psnr": null, "controller_result": null},
  "observations": [{
    "id": "O1", "finding": "由脚本根据真实结果生成",
    "metric": "mean_paired_psnr_delta_db", "group": "dark_fraction_high",
    "value": null, "n_images": null, "case_ids": ["C01"],
    "source": "paired_comparison.json#/comparison/groups/dark_fraction_high"
  }],
  "interpretation_limits": [
    "original 到 mse 同时改变监督位置和损失构成",
    "当前单 seed，区域误差不能单独定位内部原因"
  ],
  "detail_refs": {"cases": "cases.json", "curves": "curves.json"}
}
~~~

该示例是 feedback 对象，不是 proposal；不能把这些字段直接添加到现有严格三键 proposal 顶层。

当前 naive_adapter.py 使用 AutoTokenizer/AutoModelForCausalLM，这条链路按 text-only 验收。Naive 接收结构化数值和案例文本，图像先供人检查；PNG 路径不等于模型已经看过图。未来接 VLM 时，视觉观察另带观察者身份、输入案例、来源与不确定性，不替代数值裁判。

分机部署的 details_index 使用 artifact_id、相对路径以及 ARIS 可访问的共享根目录或既有远程读取工具。next 直接返回精简数值摘要；必需证据路径在提交前检查可读性。训练机绝对路径不能未经映射就声称 Agent 能读。不可达标 unavailable，不能补写“已观察”。

首版 pilot 只有四轮，history 全量即可；长期运行时最近 6 轮＋baseline/best＋全部历史索引，确保摘要不会丢掉曾被否定的相同方向。摘要按条数截取并携带完整产物路径，不按 token 截断 JSON。

新增任务指令：“先引用 observation ID 陈述测量事实，再列仍未排除的解释。选择一个当前允许的 recipe 改动，指出实际对照、保持不变的条件，以及什么结果会支持或削弱该解释。解释不能区分时明确写出，不能从区域误差直接推断模型内部原因。”

### 5.7 对快慢系统的作用

快循环每次读取这个廉价、确定性的证据包；提出可验证的下一步。慢循环仍只在已定义的停滞、矛盾、数据/协议问题或结案事件启动，读更完整的证据并比较竞争解释。增加诊断不意味着每轮增加一组辩论 Agent。

V1 的首要验收是能自动回答：哪些样本组涨跌、本轮实际改变什么、参照是谁、下一实验试图区分哪两种解释。proposal 引用证据只能证明提案引用了已记录反馈，不能证明策略实际因反馈而改变；是否真正提高研究效率，需后续等预算反馈消融或预注册对照，不能由引用次数代替。

## 6 快循环接口与研究判断

保留现有 recipe/hypothesis/based_on JSON。V1 诊断试点可先将 observation ID、竞争解释和实验判据写入 hypothesis/based_on.observation，由 next 返回证据摘要；不需要为试点放宽 _validate_proposal。完整快慢新模式再增加独立 --decision 参数。不得直接提交 evidence_refs、experiment_design 等未经支持的新顶层字段。

下面仅为结构示例，run_id 与文字不是实测结果：

~~~json
{
  "schema_version": 1,
  "decision_id": "decision_003",
  "kind": "fast_proposal",
  "feedback_ref": "feedback/exp_003/r001/summary.json",
  "latest_seen_run_id": "exp_003",
  "reference_run_ids": ["exp_001", "exp_003"],
  "comparison_run_id": "exp_003",
  "observation_refs": ["campaign_demo/exp_003/r001/O1"],
  "observation": "填写真实观测及证据路径",
  "prediction": "写清本次改变预计影响的既定指标或分组",
  "falsifier": "写清何种结果会削弱该解释",
  "alternative_explanation": "一个仍然可能成立的替代解释",
  "action": "propose",
  "requested_change": {"learning_rate": 0.00005},
  "claim_scope": "exploratory_dev"
}
~~~

验证器只检查结构、引用一致、预测非空、requested_change 与实际提交 recipe 一致、当前边界，不对科学解释打分。requested_change 表示提交的 recipe 字段和值，可包含为复现旧对照而明确填写的原值；程序生成的 actual_changed_fields/config_delta 才表示相应参照下实际发生的变化。comparison_run_id 可选，省略时使用 best_before；明确提供却无效时拒绝，不回退。observation_refs 必须解析到本 campaign 的真实证据，引用本身不等于假设已被支持。

若指定的实验对照不是当前 best，Agent 应提交足以复现目标配方的完整四字段 recipe，再改变拟检验的字段。只传一个 LR override 会继承 best 的其他字段，可能无法实现“固定旧实验的 MSE，只改 LR”。验证器展示实际相对 comparison 的差异；超出预期的差异必须修正提案或明确为组合实验，不能沿用单因素归因。

latest_seen_run_id 必须匹配最新 terminal trial；based_on 仍允许引用历史对照。不强迫科学推理只看上一轮，同时防止漏掉最近失败。

新版 submit 的工作顺序：

1. 刷新实际运行状态；确认没有 active 或不明任务。
2. 检查 control、科学协议、预算及旧的 proposal 合法性。
3. 检查 decision 引用了最新 feedback；保留不可覆盖的预测快照。
4. 快照保存 baseline/best_before/comparison/construction_base、resolved_config、init_checkpoint_ref 及自动 config delta；准备 run，持久化 active，再启动 worker。
5. 等待/续读同一 run；完成后运行原有效性与 best 选择。
6. 自动生成诊断证据，分别保存 diagnostics_status 和 proposal_ready；路由下一行动。必需诊断失败时先修复反馈，不覆盖有效科学结果。
7. 不把 no_gain 当作执行失败，不把 invalid 当作证明 recipe 无效。

## 7 慢循环触发与决策

每次新 terminal 事件执行一次廉价的确定性路由。反复 status/next/wait 不应重复召唤 reviewer。

| 优先级 | 触发 | 行动 |
|---|---|---|
| 1 | 评测协议疑似错误、数据污染、TEST 意外暴露、任务活性不明、必需诊断不可用 | 暂停受影响的新提交，先诊断；可自动重建的反馈先由脚本修复，未解决时再 review |
| 2 | 资源耗尽或下一作业无法获得有界额度 | 停止启动，整理现有结果 |
| 3 | max_trials/no_gain_limit 等搜索硬停止 | 禁止 continue_search；可在独立预授权范围内确认或结案 |
| 4 | PSNR 与已记录视觉观察冲突、分组损益明显且解释不清 | slow review，限制当前 claim，不事后换目标 |
| 5 | 连续 valid no_gain 达到预设触发点 | slow review，判断下一允许实验能否区分解释 |
| 6 | 冻结候选或结案 | 独立 review 确认计划及结论边界 |

“明显”不能由程序随意猜：自动触发需初始化前定义可计算规则；没有此规则时由带样本证据的人工/agent observation 触发，并标为诊断意见。每次有效增益不必立即启动慢循环。

保留现有 no_gain_count 把 invalid 计入停止预算的行为；另计算 consecutive_valid_no_gain 供科学停滞触发。操作失败不等于科研停滞。

slow review 输出六项：

1. 目前能确定的事实。
2. 至多三个竞争解释及各自证据。
3. 哪两个解释会对同一实验产生不同预测。
4. 最便宜且能改变决定的下一步：先查已有证据，再考虑新训练。
5. 所需改动、成本和是否在现有授权范围内。
6. 不确定性与停止条件。

允许行动为 wait、diagnose、propose、finalize、confirm、request_scope_change、report_stop。review_required 是是否召唤 reviewer 的独立布尔字段，不引入第二套动作枚举。

慢循环结果通过新增 campaign record-decision --decision 写入，不依赖下一次训练提交才能保存。对于明确标记的协议疑点，controller 在 campaign.json 保存 research_hold={decision_id,reason,evidence_refs}；该字段表示暂停受影响提交，不改原始 trial 结果。后续决定用 resolves_decision_id 引用原疑点，并提供新诊断证据。只有证明未改变固定协议的操作问题可在原 campaign 解除；确需更换评测协议则保留 hold 并建立新 campaign。任何解除都不能突破原 stop/frozen/预算。普通研究解释失败不设置 hold。

- continue/propose 只在剩余 search 空间和预算允许时执行。
- confirm 只执行已批准且与冻结候选相符的计划。
- 协议修补可以自动准备并在小 TRAIN/DEV fixture 验证；正式采用新裁判需相应授权、新 campaign 和同协议基线。
- 扩展 loss/模型/数据/主要目标需新 campaign；有事先明确授权则不重复询问。
- 资源硬上限和已冻结状态不能被一句 slow review 的 continue 推翻。
- 两个解释对所有允许实验预测相同时，应报告“当前空间无法区分”，不能编造信息增益。

一次只改一因素是解释偏好；联合改变允许字段时只能检验组合效果。一个 recipe 字段也可能对应多项实际机制变化，例如 loss_family。报告根据实际解析差异判断可归因范围，无收益仍可产生有价值的“削弱某解释”的结果。若多种解释都可导致同一结果，使用“支持/不支持/暂不能区分”，不把相关观察写成唯一原因。

## 8 运行边界与恢复

首个有人值守 pilot 可先使用现成 scheduler walltime＋--wait。完整无人值守版必须补齐以下小范围工程能力：

### 8.1 准备与启动拆分

内部 prepare_run 创建 run/config/commands/state，但不启动；controller 先保存 active 引用，再调用 start_prepared_run。worker 使用 per-run lock，重复启动同一运行只返回已有任务，不出现两份训练。

如果宿主在准备后中断，恢复时只检查已记录 run。若已明确从未启动且没有训练产物，可启动同一 prepared run；若是否仍在运行不明，则暂停。不能因日志暂时安静就创建新 trial。

### 8.2 活性与停止

记录 worker 所在主机、PID/进程开始身份、启动时间、阶段、scheduler job_id（若存在）及退出状态。活性读取优先现成 scheduler 状态或可靠进程检查；跨主机无法判断时返回 unknown。不把“日志久未增长”单独当作死亡证据。

运行时限覆盖整个 TRAIN＋独立 DEV 作业。取消必须到实际子进程/作业组，不只杀等待中的 controller。TERM 后给有限清理时间，再按现有平台方式结束残留；保留部分日志，记 interrupted/invalid。不承诺 optimizer、scheduler 或 RNG 续训。

run/state.json 由 worker 写；健康检查和 stop request 使用独立小文件，避免 controller 与 worker 覆写同一个状态临时文件。request_run_stop 只报告已请求，须等 worker/scheduler 确认退出才标已停止。确实死亡后的状态收敛在 per-run lock 下只做一次；重复 collect 不重复追加 trial 或计费。

wait --timeout 只是调用等待上限，不等于作业已超时，也不等于训练失败。

### 8.3 资源计量

- 每次启动前保留下一作业的最大可用额度；作业结束后按实际已分配 GPU 数×占用时长结算，不使用 GPU 利用率替代计费。
- 校准、失败尝试、搜索、确认都计入总额度。
- 确认预算在搜索前预留；不能搜索耗尽后才发现无法确认。
- 状态未知的作业保留已预留额度，避免重复花费。
- 推理 token/费用由 ARIS 或服务提供方已有计量/限额负责。拿不到数据就标 unavailable，不当作零，也不自行宣称严格费用受控。
- 为防止硬上限失效，执行器或现有 scheduler 必须能限制实际作业；配置文件声明本身不够。

不机械限制“每次故障只修一次”。只在有新诊断证据、科学协议不变、执行状态已知和资源允许时继续修复；相同失败且没有新证据就暂停。

## 9 独立确认与结论

### 9.1 分开四类验收

| 等级 | 可回答 | 不足以回答 |
|---|---|---|
| W 工作流验收 | 真实 Naive 经 ARIS 连续消费反馈并执行 | 算法必然涨分 |
| Q 配方收益 | 冻结 recipe 相对 baseline 的稳定收益 | Agent 优于其他搜索方式 |
| R 研究策略收益 | 等预算下搜索方法更有效 | Agent 本体已经自我训练或普遍变强 |
| P 产品收益 | 预先定义的画质/设备/用户条件改善 | 未覆盖场景和视频稳定性 |

W：真实 baseline＋三次反馈后的提案，四次 valid 实际运行、原始 transcript、模型身份、服务配置和工具产物。无涨分仍可通过。出现 invalid 不擅自加轮数补齐。

Q：只比较冻结 baseline 和 winner，允许同一 recipe 跨配对 seed 复训。建议预算允许时做三个配对 seed；这只是小规模确认，不是统计充分性的保证。

R：在确有研究需求时追加预先定义的非适应搜索或反馈消融。事后重排已被反馈选出的候选不能充当随机搜索对照。简单 PSNR 项目若随机搜索同样好且更便宜，可以采用随机搜索作为快循环 proposer。

P：需要独立约定画质目标和样本覆盖；当前 P512 静态 PSNR 不自动支持自然度、全分辨率、跨相机和 temporal 结论。

### 9.2 确认协议

confirmation.json 固定 parent_campaign、baseline/winner 完整配方、代码/数据/评测协议、配对 seeds、预算、主要指标、实际有用增益标准、失败处理、独立数据的角色和授权。

确认任务键是 arm＋seed＋replicate，记录 run_id。终态成功任务不得覆盖；重复命令返回已有状态。确认调用底层 runner，不经 search submit 的重复 recipe 判定，不更新搜索 best。

确认计划在开始前指定执行顺序和失败重试规则，默认不自动补 seed、不删失败 pair；不完整时报告 inconclusive。最终用于产品或授权 TEST 的 checkpoint 也必须预先指定选择规则，例如保留搜索时已冻结的 checkpoint，或采用预先指定 seed 的复训 checkpoint；不能事后挑确认集中最高分的 seed。

训练 checkpoint 仍使用原 DEV 选择。只有 checkpoint 冻结后才评测独立确认集。不能将确认集放到 train.validation，也不能把 TEST 更名确认集。

若没有预留独立确认集：

- 可以在获准范围内做原 DEV 的 seed 稳定性复验；
- 结论明确仍有适应 DEV 的选择偏差；
- 不自动创建新 split 或访问 TEST；
- 如要检验泛化，准备新的数据/授权计划。

确认报告输出各 seed 的配对差、各场景差异、覆盖与失败。不能把 seed×图片当作全部独立样本。输出 supported、not_supported 或 inconclusive，并解释所依据的已固定标准；不把三个 seed 都上涨称为普遍显著。

首版建议的方向性判定规则：先由研究任务给出实际有用增益 delta_useful_db，它与数值抖动 min_delta 分开；完整配对结果均为正且其平均超过 delta_useful_db 时，标 supported_in_scope；平均不大于零时标 not_supported；其他完整结果或任何不完整结果标 inconclusive。这是小规模确认规则，不是统计显著性检验。该规则及阈值必须在确认开始前冻结；缺少 delta_useful_db 时只能生成描述性报告，不能自行填一个阈值作晋级判定。

若确认集结果被用于后续调参，该集合已变为开发证据；下一次泛化确认需要新的未参与选择证据。

## 10 研究经验怎样累积

首版使用简单 JSONL，不训练 Naive，不增加向量数据库。

每条经验包含适用模型/协议/数据条件、观察、尝试、反证、替代解释、evidence_refs、status，以及可检索的标签。经验状态：

- observed：一次探索观察。
- tentative：有初步对照或重复，但范围仍窄。
- confirmed_in_scope：达到已固定确认要求。
- contradicted/superseded：后续证据反对或协议已变，保留旧记录并追加说明。

只取与当前任务条件匹配的少量记录输入下一 campaign，同时包含相关负结果。不能将“本次 MSE 无收益”升级为“MSE 永远不适合 TM”。

报告里分开：目标模型质量、研究成本、人工介入、有效实验率、到达有效候选的时间。自动化比例只是过程指标。

若未来要声称“积累经验使 Agent 更会研究”，需在未见过的问题上，对照冻结记忆版本，固定研究模型和资源。当前首版仅实现可追溯的经验复用。

## 11 一个完整的快慢循环例子

以下为假设性说明，不是仓库实测结果。

1. baseline 已完成；MSE trial 的总体增益不足 min_delta。
2. 自动证据包显示暗图组改善、强梯度组退步；这只是数值切片，不直接称 halo。
3. 快循环提出 H1：步长不合适；H2：最终 sRGB 像素监督与 original 的约束差异影响了这些区域。
4. 当前允许空间内优先固定 MSE、optimizer、weight decay，只改变 LR；保存预测和反证条件。如果 MSE trial 未成为 best，指定它为 comparison，并提交完整四字段 recipe，避免 partial LR override 意外继承 original。
5. 如果较低 LR 改善了收敛表现，但组间退步仍然存在，降低 H1 可信度；不能直接宣布 H2 已证实。
6. 若还能在预算内做有区分力的原 loss×LR 对照，则继续；否则 slow review 报告当前可识别性不足。
7. 若证据支持研究约束缺失，提出下一 campaign：仅增加一个明确 loss 项及有限权重集合，给出成本与退出条件。只有已授权才启动。
8. 冻结候选后做独立确认；如果确认不支持收益，记录负结果与经验，不再借同一确认集调出赢家。
9. 若 PSNR 与人工固定面板意见冲突，报告两者并存；需要改变产品目标时再启动新问题。

这一例子展示：慢循环负责改变下一步研究决策，而不是自动加大搜索空间。

## 12 模块与接口

所有代码路径以前缀 examples/modular_neural_isp_phase1/overlay/ 为准。

| 文件 | 操作 | 职责 |
|---|---|---|
| tm_research/control.py | 新增 | 授权记录加载、范围与资源资格检查 |
| scripts/build_dev_profile.py | 新增 | 一次性 profile 入口，调用共享配对/变换与 profile builder，不运行模型 |
| tm_research/evidence.py | 新增 | manifest 映射、参照解析、配置差异、调用统计、case/曲线/observation 和统一反馈装配 |
| tm_research/diagnostics.py | 新增 | 确定性的 profile/配对分组统计；V1.1 扩展 float 区域量与 mask 运算，不做选优或语言归因 |
| tm_research/research.py | 新增 | 轻量行动路由、decision 校验、slow review 输入和经验导出 |
| tm_research/confirmation.py | 新增 | 冻结配方的配对复训和确认报告 |
| tm_research/runner.py | 修改 | 登记诊断状态/产物；V1.1 传递观测配置；后续准备/启动拆分、运行活性、作业时限、成本产物 |
| tm_research/campaign.py | 修改 | 新模式引用、best-before、反馈装配与 decision 接入 |
| tm_research/cli.py | 修改 | control/decision/feedback/confirmation 的薄 CLI |
| photofinishing/train.py | 按需修改 | 增补已有日志缺少的 epoch telemetry，不是 V1 前置条件 |
| photofinishing/test.py | V1.1 修改 | 在原浮点评测加入 sample_id/profile 映射与可选 float 区域诊断，原 CSV 可兼容保留 |
| docs/aris-campaign-task.md | 修改 | 快慢循环步骤与默认权限 |
| docs/engineering-contract.md、tm_research_task.md | 修改 | 新模式和旧记录边界 |
| tests/test_control.py、test_evidence.py、test_diagnostics.py、test_research.py、test_confirmation.py | 新增 | 对应功能契约 |
| tests/test_campaign.py、test_runner.py、test_isp_baseline.py | 修改 | 兼容及执行风险回归 |

建议接口均为未实现设计：

evidence.py 是唯一反馈装配入口；diagnostics.py 为其数值函数库，同时供 test.py 调用区域计算。runner 只登记底层诊断结果，campaign 在完整历史可用时调用 build_run_feedback。不要让 runner 和 campaign 分别生成互相竞争的 summary。

观测配置从独立 control.observation_config 或诊断试点的明确 profile 参数传入，run 保存快照。V1 后处理不需改 runner 科学 YAML；V1.1 可给 test.py 增加 --diagnostics-profile，并由 runner 的内部可选关键字参数传递。若最终选择把字段加入科学 YAML，必须同时扩展 normalize_config 允许字段和对应测试；不能假定未知字段会被接受。

~~~python
# control.py
load_control(path: str) -> dict
check_action(control: dict, state: dict, action: dict, usage: dict) -> dict

# evidence.py
load_manifest(path: str) -> dict
build_run_feedback(campaign_dir: str, run_id: str) -> dict
load_feedback_detail(feedback_dir: str, section: str, key: str | None = None) -> dict
resolve_trial_references(state: dict, proposal: dict, decision: dict | None) -> dict
diff_scientific_config(resolved: dict, reference: dict) -> dict

# diagnostics.py：纯测量/统计，不修改 scientific best
build_dev_profile(config: dict, rules: dict, output_dir: str) -> dict
compare_per_image(candidate_rows: list[dict], reference_rows: list[dict],
                  profile: dict) -> dict
measure_fixed_regions(output, target, masks: dict, metric_config: dict) -> list[dict]

# research.py
route_next_action(state: dict, control: dict, feedback: dict) -> dict
validate_research_decision(decision: dict, state: dict, feedback: dict) -> dict
record_research_decision(campaign_dir: str, decision: dict) -> dict
build_review_packet(campaign_dir: str, trigger: dict) -> dict
export_research_memory(campaign_dir: str) -> list[dict]

# campaign.py：旧位置参数不变
initialize_campaign(config, campaign_dir, max_trials=5, no_gain_limit=3,
                    wait=True, *, min_delta=DEFAULT_MIN_DELTA, control_ref=None)
next_proposal(campaign_dir)
submit_proposal(campaign_dir, proposal, wait=True, *, decision_ref=None)

# runner.py：新增函数
inspect_run(run_id: str, runs_dir: str = "runs", *, now=None) -> dict
request_run_stop(run_id: str, runs_dir: str = "runs", *, reason: str) -> dict
evaluate_frozen_checkpoint(config: dict, checkpoint: dict, split_spec: dict,
                           output_dir: str) -> dict

# confirmation.py
initialize_confirmation(search_campaign_dir: str, confirmation_dir: str,
                        plan: dict, *, control_ref: str) -> dict
run_confirmation_next(confirmation_dir: str, *, wait: bool = True) -> dict
report_confirmation(confirmation_dir: str) -> dict
~~~

统一返回约定：

- check_action 返回 allowed、reason、requires_human_decision、remaining_budget；不把自然语言当执行命令。
- route_next_action 返回 action、trigger_id、reason、evidence_refs、review_required；trigger_id 由 campaign/run/事件类型组成，不重复产生同一 review。
- feedback 返回 schema_version、feedback_revision、latest_run_id、quality、diagnostics_status、proposal_ready、actual_changed_fields、overall、comparisons、groups、observations、cases、curves、usage、unavailable、detail_refs。quality 表达原实验资格，不能混入诊断失败。
- 新版 next 保留旧字段，追加 feedback、feedback_ref、next_action；legacy campaign 无诊断时明确标注。
- decision_ref 指向文件；验证通过后复制内容到已保存的 trial/决策快照，避免外部文件后改影响历史。
- record_research_decision 持锁追加 slow/observation 决定、去重 decision_id，并处理具备证据的 research_hold 设置/解除；不执行训练，不改变分数。
- evaluate_frozen_checkpoint 只启动独立评测；split_spec 是已授权确认数据，不写入训练的 validation 配置；返回与现有 validate_metrics 一致的有效性证据。
- control_ref 在新 campaign 初始化时保存冻结副本。旧 campaign 没有 control_ref 时保持 legacy 行为，不声称新增边界已受控。

建议新增 CLI：

~~~bash
# 以下是设计接口，当前代码尚不支持新增参数与命令
python -m tm_research.cli campaign init --config configs/naive-pilot.yaml --campaign-dir campaigns/pilot --control control.json --max-trials 4 --no-gain-limit 3 --min-delta <已校准值> --wait
python -m tm_research.cli campaign next --campaign-dir campaigns/pilot
python -m tm_research.cli campaign submit --campaign-dir campaigns/pilot --proposal proposal.json --decision decision.json --wait
python -m tm_research.cli campaign feedback --campaign-dir campaigns/pilot --run exp_002
python -m tm_research.cli campaign review-packet --campaign-dir campaigns/pilot
python -m tm_research.cli campaign record-decision --campaign-dir campaigns/pilot --decision slow-review.json
python -m tm_research.cli confirmation init --campaign-dir campaigns/pilot --confirmation-dir confirmations/pilot --plan confirmation-plan.json --control control.json
python -m tm_research.cli confirmation next --confirmation-dir confirmations/pilot --wait
python -m tm_research.cli confirmation report --confirmation-dir confirmations/pilot
~~~

反馈详情复用文件读取即可，首版不必为每个字段创建 CLI。review-packet 生成材料，不调用另一个通用 Agent 框架。

## 13 实施任务与依赖

最小诊断试点先做 T2a→T3→T4 中的 next 注入与原三键提示词，复用现有执行限制；不必等待新增 telemetry、ROI、慢循环和确认模块。完整新模式再完成 T1 与 T4 的 decision 接入，然后 T5。T2b 是独立的 V1.1 区域扩展，T6 可在接口固定后并行，T7 依赖 T1/T3/T6，T8 做整体验收。多人实施时 runner/campaign/cli 由一个集成人统一合并，避免并行改同一函数。

### T1 固定新模式控制契约

**Files:** 新增 control.py、test_control.py；修改 campaign.py 初始化、cli.py init 和对应文档。

**Consumes:** 现有 normalize_config、initialize_campaign。  
**Produces:** load_control、check_action，新模式 control_ref 与冻结副本。

- [ ] 定义并测试 test_control_blocks_missing_limits、test_existing_authorization_reused、test_legacy_campaign_keeps_decisions：缺必要额度拒绝启动、已有授权不再请求、旧结果不重算。
- [ ] 运行新测试，确认未实现时失败。
- [ ] 实现独立 control 解析；初始化检查 actual config 在授权范围内；旧 runner YAML 不增加政策字段。
- [ ] 运行 pytest -q tests/test_control.py tests/test_campaign.py；新旧契约均通过。
- [ ] 提交这一独立改动及范围说明。

### T2a 形成稳定样本清单和图像分组，V1

**Files:** 新增 scripts/build_dev_profile.py、evidence.py 的 manifest 部分、diagnostics.py 的 profile 部分、test_evidence.py、test_diagnostics.py；原则上不改 train/test 主路径。

**Consumes:** paired_files、load_image_pair、已有 per_image.csv、获准 TRAIN/DEV。  
**Produces:** load_manifest、build_dev_profile、固定 sample_id/scene_id 映射、profile.json 与图像标签。

- [ ] 使用重名、乱序、unknown、同源场景的 fixture 检验映射和边界；同场景跨集报告事实，不擅自重分数据。
- [ ] 实现 profile CLI 和共享 builder；复用实际评测变换；只扫描指定获准集合。
- [ ] 固定图像特征/阈值来源、候选无关标签和案例规则；已有产物回填标 retrospective。
- [ ] 运行 pytest -q tests/test_evidence.py tests/test_diagnostics.py；确认同样本在不同 run 的 ID/标签保持一致。
- [ ] 提交 profile 样例及用法。若现有 CSV 无法唯一映射，先明确缺项，再做最小身份字段补充。

### T2b 增加量化前区域观测，V1.1，可后置

**Files:** diagnostics.py、test_diagnostics.py、test.py、runner.py 可选观测参数、test_runner.py、test_isp_baseline.py；仅需要时增补 train.py telemetry。

**Consumes:** T2a profile、评测同尺寸 float output/GT。  
**Produces:** roi_profile.json、固定 masks、roi_metrics.csv、诊断状态与 artifact 引用。

- [ ] 在已知常量/梯度张量上检查区域 MSE 分母、bias 符号、macro/micro 区别、空 ROI 与小 ROI、滤波边界和侵蚀后面积。
- [ ] 检查 candidate/reference 使用同一 mask 和有效样本集合；不同尺寸/变换/profile_revision 不可静默拼接。
- [ ] 实现可选参数传递与 float 测量，保证诊断不原位修改 output/GT、不调用训练随机过程。
- [ ] 同一 fixture checkpoint 开关诊断，验证原 PSNR、样本数、valid、best 判定一致；诊断失败保留核心 DEV 结果。
- [ ] 如旧 checkpoint 仍在而需要补区域量，显式安排获准 DEV 重评并计入额度，不能从 PNG-8 补造 float 指标；缺 checkpoint 时标 unavailable。
- [ ] 运行 pytest -q tests/test_diagnostics.py tests/test_isp_baseline.py tests/test_runner.py；提交测量定义与限制。

### T3 生成真实反馈包

**Files:** evidence.py、diagnostics.py、test_evidence.py、test_diagnostics.py；修改 campaign.py 的提交前参考记录与终态装配，runner.py 登记已有/可选 artifact 状态。

**Consumes:** T2a manifest、当前/baseline/best-before/explicit comparison 的 CSV、完整科学配置、现有日志。  
**Produces:** build_run_feedback、load_feedback_detail、参照快照与 summary/paired_rows/paired_comparison/cases。

- [ ] 编写 test_pair_join_ignores_row_order、test_missing_pair_is_not_silently_dropped、test_new_best_compares_to_previous_best、test_overlapping_groups_not_pooled。
- [ ] 使用固定小 fixture 验证 mean(delta) 等于同集合 mean_current−mean_reference；重叠组不被相加；无效 run 不生成伪分数。
- [ ] 检查历史 based_on 不改变 partial recipe 的 construction base；explicit comparison 缺失或不同协议不静默替换；新 best 不覆盖 best_before。
- [ ] 自动比较 resolved_config，验证 recipe 差异、导出 scheduler 变化、固定初始化及 original/mse 的监督差异说明；仅忽略运行身份字段，不能忽略数据和初始化路径。
- [ ] 实现按 ID join、八例选择/去重、持续困难历史不足处理、真实覆盖与 unavailable；同一输入生成相同内容，不重写源分数。
- [ ] 保存可读 feedback_revision；补诊断后新版本不覆盖旧引用；重复 next/collect 不制造新版本，旧 decision 仍能解析到原 observation。
- [ ] 从已有 val 曲线提取摘要，缺数据保留缺口；V1.1 的 ROI 文件存在且兼容时再纳入。
- [ ] 运行 pytest -q tests/test_evidence.py tests/test_campaign.py。
- [ ] 提交可独立用于已有 run 的反馈后处理器。可对旧产物生成“事后诊断”，但不得声称当时 agent 已看到它。

### T4 将反馈和预测接入快循环

**Files:** 新增 research.py 的 decision 校验；修改 campaign.py、cli.py；新增 test_research.py，更新 test_campaign.py。

**Consumes:** T3 feedback、现有三键 proposal；完整新模式另依赖 T1 control。  
**Produces:** 新版 next 字段、submit --decision、trial 决策快照。

- [ ] 编写 test_stale_latest_ack_rejected、test_old_reference_with_fresh_ack_allowed、test_legacy_three_key_proposal_unchanged、test_decision_delta_matches_recipe。
- [ ] 在未实现状态运行，确认失败。
- [ ] 先在 next 返回小型摘要、observation ID 和可读详情引用；诊断试点沿用三键 proposal，把实验判据放入已有文本字段。
- [ ] 完整新模式要求最新反馈确认及合法 observation_refs；旧 based_on 仍可引用历史证据，只校验科学文字有无及引用，不打 reviewer 分。
- [ ] 验证诊断 required/optional 的差别：必需证据不可用时阻止新提案并修复后处理，可选证据失败带缺口继续，两者均不改原 valid/best。
- [ ] 验证 baseline 的不适用比较不阻塞首个提案；invalid terminal 的数值缺失不产生伪指标或永久等待，unknown worker 仍阻塞。
- [ ] 验证跨机路径映射和按需读取；路径不可达不能记“已观察”，无视觉链路只传文本观察。
- [ ] 运行 pytest -q tests/test_research.py tests/test_campaign.py。
- [ ] 提交，并用 tiny runner fixture 演示“读取反馈→写预测→实际提交”的完整路径。

### T5 增加事件触发慢循环

**Files:** research.py、test_research.py、cli.py review-packet、ARIS 任务文档。

**Consumes:** T4 最新事件与决定、T1 边界、T3 证据。  
**Produces:** route_next_action、build_review_packet、record_research_decision、slow review 记录。

- [ ] 编写 test_hard_stop_cannot_be_overridden、test_invalid_not_scientific_plateau、test_same_event_not_re_reviewed、test_scope_change_requires_new_campaign。
- [ ] 编写 test_slow_decision_saved_without_trial 和 test_hold_resolution_requires_evidence：停止后仍可保存诊断；解除疑点须引用原决定及新证据，不能解除硬 stop。
- [ ] 运行失败测试后实施优先级路由，保留原 no_gain_count 语义。
- [ ] 任务文档定义六项 slow 输出、行动集合，以及范围内自主/范围外准备决策的规则。
- [ ] 运行 pytest -q tests/test_research.py tests/test_campaign.py。
- [ ] 提交 slow review packet 和一个假设性示例，明确不是实测证据。

### T6 使实际作业有界并可可靠续读

**Files:** runner.py、campaign.py 的 _launch、test_runner.py、test_campaign.py。

**Consumes:** T1 时限/资源、现有 run 状态。  
**Produces:** inspect_run、request_run_stop、内部 prepare/start、usage 产物。

- [ ] 编写 test_active_saved_before_worker_start、test_same_run_worker_lock、test_wait_timeout_does_not_resubmit、test_job_limit_stops_children、test_unknown_liveness_blocks_launch。
- [ ] 使用可控的小型子进程 fixture 复现启动中断与超时，不用真实 GPU 烧算力。
- [ ] 实现 prepare→持久 active→start；per-run lock；实际进程组/调度器停止；TRAIN＋DEV 总时限与成本预留。
- [ ] 运行 pytest -q tests/test_runner.py tests/test_campaign.py。
- [ ] 提交，并说明支持的平台/调度方式；不能测试到的跨主机行为返回 unknown，不承诺通用恢复。

### T7 建立冻结候选确认阶段

**Files:** 新增 confirmation.py、test_confirmation.py；修改 cli.py。

**Consumes:** 已 finalized campaign、T1 确认授权、T6 runner、T3 证据汇总。  
**Produces:** initialize_confirmation、run_confirmation_next、report_confirmation。

- [ ] 编写 test_confirmation_allows_recipe_across_seeds、test_confirmation_is_idempotent、test_confirmation_does_not_mutate_search_best、test_confirmation_never_selects_checkpoint、test_test_split_not_relabelled。
- [ ] 用双配方双 seed fixture 验证任务键、失败保留、同 seed 配对及 incomplete 状态。
- [ ] 实现独立 frozen manifest；训练仍用原 DEV；独立确认评测仅在 checkpoint 冻结后执行。
- [ ] 运行 pytest -q tests/test_confirmation.py tests/test_runner.py tests/test_campaign.py。
- [ ] 提交；数据或预算未授权时只生成可审阅计划，不启动。

### T8 结案、经验和端到端验收

**Files:** research.py 的报告/经验导出、相关 tests；README_PHASE1.md、aris-campaign-task.md、engineering-contract.md、tm_research_task.md。

**Consumes:** T1–T7 的真实产物。  
**Produces:** 分层报告、带边界经验、完整运行文档。

- [ ] 编写 test_claim_level_follows_evidence、test_memory_retains_contradiction、test_no_visual_capability_no_visual_claim、test_unknown_cost_not_zero。
- [ ] 实现 W/Q/R/P 分层事实报告，保留负结果、成本、不确定性与替代解释。
- [ ] 运行相关测试和既有完整 pytest；只把实际命令输出写入验收记录。
- [ ] 在已经授权且可用的服务/GPU/数据上运行真实 baseline＋三提案，保留原始 transcript；没有条件时标 live pending。
- [ ] 有冻结有效候选且确认已授权时再执行 T7；不为让结果好看追加预算或更换数据。
- [ ] 提交代码与文档，保留人工复核一个被削弱的解释和一个未解替代解释的结案入口。

## 14 发布阶段与验收标准

| 阶段 | 必须完成 | 能验收的能力 |
|---|---|---|
| A0：V1 诊断试点 | T2a、T3、T4 的 next 注入与原三键提示词；沿用现有执行限制 | 复用 CSV，交付逐图差、固定图像分组、八例索引、真实配置差异和证据引用；不宣称新控制契约已经生效 |
| A1：完整证据快循环 | T1、T4 的 decision 接入；已有可靠 scheduler 限制 | 最新反馈确认、显式实验对照、预测快照、必需证据检查 |
| A2：V1.1 区域诊断，可后置 | T2b 与 T3 区域汇总扩展 | 量化前 ROI 指标、固定区域对照和可选同位置展示；不改变主排名 |
| B 可持续快慢循环 | T5–T6 | 停滞和冲突触发正确行动，卡死不重复训练，预算约束真实生效 |
| C 科研结案 | T7–T8 | 独立确认、有限 claim 和可追溯经验 |
| D 后续扩展 | 按新证据单独设计 | 有界 loss、模块归因、结构实验、经校准 multi-fidelity |

运行里程碑：

- 诊断包可对已有 run 后处理验证，无需为验证字段拼装启动真实训练。
- 首个真实 V1 验收需 transcript 显示 Naive 读到 observation、引用可访问证据，并提出有对照和反证条件的允许改动；不要求涨分。
- “证据改变了选择”可检查自然语言理由与实际 recipe 是否一致；要断言诊断提高研究收益，另做等预算、有/无诊断的受控比较，不把一次示例当因果证明。
- 工程 fixture 通过，不代表真实 ISP 或真实 Naive。
- CPU 合成实际模型通过，不代表真实数据泛化。
- Naive＋ARIS 四轮完成代表 workflow acceptance。
- frozen recipe 的确认收益代表对应协议范围的算法证据。
- 搜索策略价值和产品收益需要各自对照，不捆绑到首次闭环中。

首版不实施：自动生成任意网络、在线训练研究大模型、自动审美总分、无校准短训筛选、统一跨任务平台、复杂审核评分、内容哈希系统。

## 15 来源与建议的界线

本文“当前代码事实”来自下列读取；其余接口、文件、新状态和测试均为建议设计，不是仓库已经具备的能力。

- [fix 工程契约](https://github.com/baolinv0/Naive-N0.5-Flash/blob/6d5ea1323c6834d0594ca7ad8e2edd04179593b4/examples/modular_neural_isp_phase1/overlay/docs/engineering-contract.md)
- [fix 研究任务](https://github.com/baolinv0/Naive-N0.5-Flash/blob/6d5ea1323c6834d0594ca7ad8e2edd04179593b4/examples/modular_neural_isp_phase1/overlay/docs/tm_research_task.md)
- [campaign.py](https://github.com/baolinv0/Naive-N0.5-Flash/blob/6d5ea1323c6834d0594ca7ad8e2edd04179593b4/examples/modular_neural_isp_phase1/overlay/tm_research/campaign.py)
- [runner.py](https://github.com/baolinv0/Naive-N0.5-Flash/blob/6d5ea1323c6834d0594ca7ad8e2edd04179593b4/examples/modular_neural_isp_phase1/overlay/tm_research/runner.py)
- [test.py](https://github.com/baolinv0/Naive-N0.5-Flash/blob/6d5ea1323c6834d0594ca7ad8e2edd04179593b4/examples/modular_neural_isp_phase1/overlay/photofinishing/test.py)
- [train.py](https://github.com/baolinv0/Naive-N0.5-Flash/blob/6d5ea1323c6834d0594ca7ad8e2edd04179593b4/examples/modular_neural_isp_phase1/overlay/photofinishing/train.py)
- [recipe_utils.py：监督位置、优化器与实际配方](https://github.com/baolinv0/Naive-N0.5-Flash/blob/6d5ea1323c6834d0594ca7ad8e2edd04179593b4/examples/modular_neural_isp_phase1/overlay/photofinishing/recipe_utils.py)
- [baseline_utils.py](https://github.com/baolinv0/Naive-N0.5-Flash/blob/6d5ea1323c6834d0594ca7ad8e2edd04179593b4/examples/modular_neural_isp_phase1/overlay/photofinishing/baseline_utils.py)
- [委派政策](https://github.com/baolinv0/Naive-N0.5-Flash/blob/91d8577f63231344c8b03c555dbde9511212db8b/examples/modular_neural_isp_phase1/overlay/docs/aris-campaign-task.md)
- [历史运行证据边界](https://github.com/baolinv0/Naive-N0.5-Flash/blob/6d5ea1323c6834d0594ca7ad8e2edd04179593b4/examples/modular_neural_isp_phase1/evidence/review_fix/README.md)
