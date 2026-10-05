# Naive/Qwen 实验编排器设计

研究协议：[五组实验方案](../../plans/naive-research-loop-experiments-20261005_CN.md)。实施依据：用户已批准外层编排器，并指定本地 Naive 权重和开源 Qwen 辅助判断。

## 目标与角色

交付可生成配置、串行推进五组独立 campaign、冻结后配对确认、记录模型成本、输出预注册统计的外层工具。Naive 是主提案者。Qwen-VL 可以对授权 DEV 图像作辅助描述，输出保留来源；数值质量、选择、预算和统计结论由现有固定程序处理。Qwen reviewer 是显式单独变体，默认主实验 slow 仍使用同一 Naive。未配置实际 endpoint/数据/预算时，可以完成 dry-run 和明确标记的 fixture 验收。

## 文件与接口

新增模块均放在 `tm_research/`，避免仓库忽略 `experiments/` 目录：

- `experiment_space.py`：`STRATEGIES=('random','tpe','naive_scalar','naive_rich','naive_fast_slow')`；`recipe_space() -> list[dict]`；`validate_space_recipe(recipe) -> dict`；`recipe_key(recipe) -> str`。四字段完整离散150配方。
- `experiment_projection.py`：`project_feedback(packet: dict, strategy: str) -> dict`。Scalar 采用允许字段重建，保留真实 aggregate/execution observation、最新身份、recipe/分数/资源/route；删除诊断与自由文本历史解释。Rich 返回不共享内存的完整副本。
- `experiment_strategies.py`：`suggest_recipe(strategy: str, history: list[dict], seed: int, max_draws: int=10) -> dict`；`sampler_payload(packet: dict, recipe: dict, strategy: str, decision_id: str) -> dict`。Random/TPE提交完整配方，真实引用，算法来源明确。TPE用固定Optuna版本，FAIL不写虚假分数。
- `experiment_costs.py`：`CostLedger(path, limits, *, scope_id=None, scope_limits=None)`；`reserve(request_id, input_tokens, output_tokens, role) -> dict`；`settle(request_id, usage: dict|None, error: str|None=None, **metadata) -> dict`；`summary(scope_id=None) -> dict`。同一锁原子检查全局及 campaign 两级上限，先预约、失败也记录、未知实际tokens/金额不填0。
- `experiment_provider.py`：`ChatClient(config: dict, ledger: CostLedger)`；`complete(messages: list, *, request_id: str, role: str, max_output_tokens: int=4096) -> dict` 返回 `{payload,raw_text,usage,request_id,model,role}`。流式Chat Completions，要求完整stop和[DONE]，保存原始输入/回复/失败，凭据仅来自配置指定环境变量。
- `experiment_vision.py`：`build_vision_messages(packet: dict, image_paths: list[str], allowed_root: str, max_images: int=3) -> list`；`validate_auxiliary_report(payload: dict) -> dict`。只有授权run目录中的有限PNG/JPEG可读，含图片字节、实际资源身份和文本，不能只有路径声称视觉访问。输出`observations/limitations`，显式非数值主分数。
- `experiment_analysis.py`：`sign_flip_test(values: list[float], *, theta: float=0, seed: int=20261005, draws: int=100000) -> dict`；`sign_flip_interval(values, *, alpha=0.05, seed=20261005, draws=100000) -> dict`；`holm_adjust(pvalues: dict[str,float]) -> dict[str,float]`；`analyze_records(records: list[dict], *, strategies: list[str], blocks: list[int], delta_strategy: float=0.1) -> dict`。每record至少含`strategy,block,pairs`，每pair为`seed,status,delta_db`。先平均campaign内seed，缺失明确不完整，三个主比较成一family。
- `experiment_plan.py`：校验外层JSON manifest；为每组/块生成独立config/control/confirmation，不改变baseline科学参数或原control授权范围。
- `experiment_runner.py`：只通过现有campaign/confirmation API推进；一次`step`最多一个有成本动作；先持久保存request/receipt，重启后对照native状态，未知提交不自动重发。
- `experiment_cli.py`：`validate/generate/step/status/confirm/analyze`；无配置的endpoint不启动模型；全部搜索冻结才允许统一CONFIRM。

## Manifest

显式包含`schema_version=1,experiment_id,authorization_ref,source_revision,baseline_config,control_template,profile_ref,strategies,search_seeds,max_trials,min_delta,models,model_limits,campaign_model_limits,campaign_walltime_seconds,experiment_limits`；确认可选，包含预注册`seeds_by_block,delta_useful_db,delta_strategy`。路径按manifest文件目录解析。baseline 必须指向原 control 已批准的配置，CONFIRM相对路径按原授权文件解析。control_template必须是现有有效授权，生成的单campaign预算不得超过模板，不增配GPU或数据权限。max_trials包含baseline；no_gain_limit=K+1；plateau slow阈值完整组2，其余K+1。

Naive endpoint和Qwen endpoint是显式配置，支持用户在外部用本地权重启动服务；工具不下载大模型。provider身份及role冻结。Qwen辅助诊断默认关闭；启用后必须对两种rich组使用同一固定抽样和prompt，Scalar不接收描述。独立CONFIRM只由评测程序读取，不发送给任何模型。

## 调度、恢复与失败

每个组/块独立runs/campaign/models目录；外层会话锁防重复启动。持久request状态分prepared/submitted/settled；未知HTTP生成或提交结果保留inconclusive并等待人工/明确恢复，不自动花费第二次。valid/invalid训练由native状态决定。拒绝提案次数有界，所有尝试和token预约保留。模型回复不由宿主修补为成功；sidecar latest引用、域和route均由原validator检查。无工作wait/diagnose返回暂停；不强行finalize。Naive慢审查请求使用新messages与独立request ID。

没有GPU/真实数据时，测试使用测试专用fixture脚本，产物写engineering_fixture，不登记成live Naive W或scientific strategy证据。真实W现有Naive adapter限制保持透明。

## 分析与成本

native confirmation report重新验证实际权重/配置/协议；仅primary独立确认进入主分析。场景重叠、未知、缺失和失败随结果披露。符号翻转位置检验和区间反演使用同一集合；小R无法获得有界区间时返回明确unbounded，JSON不用Infinity/NaN。R≤20精确，较大R Monte Carlo。三个主比较Holm；98.33%同时区间加有用阈值。实际cost未知保持null，保守预约单列。图表及CSV从只读数值记录派生。

## 验收

先证明150配方唯一完整、Scalar无诊断泄漏、TPE真实观测/失败处理、引用可通过native validator；再测本地HTTP SSE成功/截断/失败预算、授权图像传递、重复运行恢复和五组一次共同接口fixture；最后运行原生全量回归。真实Naive/Qwen验收需用户提供本环境可访问权重或服务以及数据/GPU预算。
