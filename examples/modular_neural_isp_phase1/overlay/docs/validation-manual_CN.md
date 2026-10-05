# ARIS + Naive 快慢科研系统：部署与验收使用手册

本手册用于回答两个先后独立的问题：**现有系统能否可靠执行闭环？真实 Naive 是否使用实验反馈决定下一次实验？** 第一次验收不要求 PSNR 上涨。科研收益要在执行验收后，通过冻结确认和等预算策略对照检验。

源码核对基线：`feat/aris-fast-slow-research-20261003` 的 `0a05bff757cf9c412a527da952499dc31ec5bba7`，2026-10-05。本手册没有运行真实 Naive、真实数据 GPU 训练或目标节点全量回归。此前的限定回归通过不能代替目标部署验收；仓库保留的完整原生测试记录仍有失败，见 [原始全量尝试及部署限制](../evidence/review_fixes_20261004/README.md)。

本手册与[科研循环实验方案](plans/naive-research-loop-experiments-20261005_CN.md)是同一验证链的两个阶段：本手册负责 A 部署验收和 B 真实 Naive 闭环；实验方案负责 D 等预算策略比较及独立 CONFIRM 统计。C 冻结确认是两者之间的接口。除非明确创建正式策略 campaign，否则本手册中的 `max_search_trials=4`、四轮 W 验收和 `slow_policy=2` 不应扩展成正式主实验预算。

## 1. 首次验收做什么

| 阶段 | 操作 | 通过标准 | 能说明什么 |
| --- | --- | --- | --- |
| A：部署 | 实际 Linux 节点、实际容器中做 PID 检查和完整原生回归 | PID 视图一致，原生全量测试通过；没有跳过进程监督 | 执行链路适配部署环境 |
| B：live 闭环 | 真实数据 baseline + 3 次真实 Naive 顺序提案 | 4 次有效运行，结果→模型读取→提案→执行的原始记录可对应；自动报告 `W=accepted` 并核对实际 ARIS 会话 | 模型与执行器接通，确实消费前一轮反馈 |
| C：确认，可选 | 已授权的冻结 baseline/winner 配对 seed 复训 | 按冻结计划完成或如实保留失败、缺失配对 | 当前 recipe 的稳定性或授权独立集效果，见 `Q` |
| D：研究收益，后续 | 预先定义的等预算策略对照 | 收益、有效实验率、耗时、推理费用、人工介入可比较 | 丰富反馈、慢复盘是否值得其成本，见 `R` |

当前 pilot：**TEST 不可见，`test_permission=false`，不执行 `final-test`。** 固定 photofinishing 模型、数据、初始化、seed、训练预算和协议；仅允许改 `loss_family / optimizer / learning_rate / weight_decay`。PNG 路径不是 Naive 的视觉输入；当前为结构化文本观测。

baseline 不由模型提出。4 次 trial 包括 baseline，也包括失败尝试。自然触发停止、预算不足或失败耗尽预算时，如实结束；不要删失败、追加未授权 trial 或修改分数凑齐验收。

## 2. 开始前准备

需要部署者提供：

- 可以执行原生训练和 `/proc` 监督的 Linux 节点及容器；训练 GPU 和 Naive 推理资源分别授权。
- 实际 TRAIN、DEV 数据，以及固定初始化 checkpoint 和对应 config（如采用预训练初始化）。
- 已可加载真实 Naive 权重的独立推理环境；本仓库不安装 ARIS-Code 或提供 GPU。
- 已安装、可用 Bash/文件工具的 ARIS-Code。记录真实宿主版本，保留原始会话导出和人工介入。
- 数值完整的训练资源合同、同一权重重复 DEV 重载得到的 `min_delta`，以及可读共享目录。
- 若要求自动 W：ARIS 宿主能实现第 7 节的原样 JSON 传递和输出保存。仓库没有一条自动安装此宿主接入的命令。

训练与推理可以分机，但服务、capture、campaign 和反馈文件须在参与进程上以**相同绝对路径**可读，节点时钟须同步。控制器的 `allocated_gpus` 记训练作业分配，不包含 Naive 推理 GPU；推理费用另记。

## 3. 安装与固定版本

以下命令在 Linux Bash 执行。把根目录改为实际持久目录；代码仓库、ISP runtime、实验产物分开。不要把 overlay 覆盖到 Naive 权重目录。

```bash
export ISP_VALIDATION_ROOT=/absolute/path/isp-validation
mkdir -p "$ISP_VALIDATION_ROOT"
export ISP_SOURCE="$ISP_VALIDATION_ROOT/Naive-N0.5-Flash"
export ISP_RUNTIME="$ISP_VALIDATION_ROOT/modular_neural_isp"

git clone --branch feat/aris-fast-slow-research-20261003 \
  https://github.com/baolinv0/Naive-N0.5-Flash.git "$ISP_SOURCE"
git -C "$ISP_SOURCE" checkout 0a05bff757cf9c412a527da952499dc31ec5bba7
git clone https://github.com/SamsungLabs/modular_neural_isp.git "$ISP_RUNTIME"
git -C "$ISP_RUNTIME" checkout 5a845f673edfdf92de18dcfc20d204f79f1ba38b
cp -a "$ISP_SOURCE/examples/modular_neural_isp_phase1/overlay/." "$ISP_RUNTIME/"

cd "$ISP_RUNTIME"
python3.12 -m venv .venv
source .venv/bin/activate
export ISP_PYTHON="$ISP_RUNTIME/.venv/bin/python"
```

上述使用已核对的 Python 3.12 系列；先确认节点有该解释器，不把任意最新 Python 当作已验证兼容环境。先按目标 CUDA/驱动环境安装与 ISP requirements 一致的 GPU 轮子，再安装依赖。此固定 upstream 的要求是 PyTorch 2.5.1、torchvision 0.20.1、torchaudio 2.5.1；不要把 Naive 的 Transformers 环境混进 ISP 环境。

```bash
"$ISP_PYTHON" -m pip install -r requirements.txt -r requirements-phase1-dev.txt
"$ISP_PYTHON" -m tm_research.cli --help
"$ISP_PYTHON" -c 'import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())'
mkdir -p "$ISP_VALIDATION_ROOT/evidence"
git -C "$ISP_SOURCE" rev-parse HEAD > "$ISP_VALIDATION_ROOT/evidence/overlay-revision.txt"
git -C "$ISP_RUNTIME" rev-parse HEAD > "$ISP_VALIDATION_ROOT/evidence/upstream-revision.txt"
"$ISP_PYTHON" -m pip freeze > "$ISP_VALIDATION_ROOT/evidence/isp-packages.txt"
```

GPU 验收中 `cuda.is_available()` 应为 true，设备数应符合分配。训练入口使用首个可见 CUDA 设备；通过调度器或 `CUDA_VISIBLE_DEVICES` 约束实际分配。`original` loss 需要 VGG19 权重，提前缓存或允许规定的下载；不能临时改 MSE 来掩盖原始 baseline 的依赖失败。

Naive 权重与依赖按其[官方模型说明](https://huggingface.co/NaiveAI/Naive-N0.5-Flash)在独立环境准备。这里不假定某个显存规格或未核实的 ARIS 安装命令。

## 4. 先验部署，不启动科研搜索

### 4.1 检查 PID 与 `/proc` 视图

在实际运行 runner 的容器内执行，宿主机检查不能替代容器检查：

```bash
"$ISP_PYTHON" - <<'PY'
import json, os, subprocess, sys
from pathlib import Path

pid = os.getpid()
own = Path('/proc/self/stat').read_text()
pid_stat = Path('/proc', str(pid), 'stat').read_text()
proc_pid = int(own.split('(', 1)[0].strip())
def ticks(stat):
    return stat[stat.rfind(')') + 2:].split()[19]
parent_ok = pid == proc_pid and ticks(own) == ticks(pid_stat)
child_code = "import os,json;from pathlib import Path;p=os.getpid();s=Path('/proc/self/stat').read_text();q=int(s.split('(',1)[0].strip());print(json.dumps({'os_pid':p,'proc_self_pid':q,'ok':p==q}))"
child = json.loads(subprocess.check_output([sys.executable, '-c', child_code], text=True))
nspid = next((line for line in Path('/proc/self/status').read_text().splitlines()
              if line.startswith('NSpid:')), 'unavailable')
print(json.dumps({'os_pid': pid, 'proc_self_pid': proc_pid, 'NSpid': nspid,
                  'parent_ok': parent_ok, 'child': child}, indent=2))
if not parent_ok or not child['ok']:
    raise SystemExit('PID /proc view mismatch: fix deployment before native acceptance')
PY
```

通过：parent、child 均为 true。此前环境出现过 `os.getpid()=5`、`/proc/self/stat=7137`，足以使监督链路失败。若不一致，先由部署者修复实际 PID namespace、`/proc` 挂载或节点配置；不跳过身份检查。只调用 runner 自身的身份回环不能排除这个问题。

### 4.2 完整原生回归

在上述 applied ISP root、实际运行环境执行。先安排工程回归预算，原生测试会运行小型训练/评测 fixture。

```bash
cd "$ISP_RUNTIME"
if OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 "$ISP_PYTHON" -m pytest -q \
  > "$ISP_VALIDATION_ROOT/evidence/native-pytest.log" 2>&1; then
  printf 'PASS\n' > "$ISP_VALIDATION_ROOT/evidence/native-pytest-status.txt"
else
  printf 'FAIL\n' > "$ISP_VALIDATION_ROOT/evidence/native-pytest-status.txt"
  cat "$ISP_VALIDATION_ROOT/evidence/native-pytest.log"
  exit 1
fi
```

通过：退出码 0、全量无失败，并保留日志；不是只重跑限定测试或排除 native case。测试通过说明工程链路，不说明真实模型研究能力。若失败，按最早实际失败排查，再重跑原始全量命令；不能把全部失败自动归为环境原因。

可选合成数据入口 `scripts/make_smoke_data.py` 只用于隔离的工程烟测。它生成 2 TRAIN、1 DEV、1 synthetic TEST；不可用它替代真实四轮验收，也不可把其中 PSNR 当产品效果。正式 pilot 直接进入下面的真实数据配置。

## 5. 配置数据、合同和固定诊断

### 5.1 建立本次 pilot 文件

```bash
export ISP_CAMPAIGN="$ISP_RUNTIME/campaigns/live-pilot-001"
export ISP_PROFILE="$ISP_VALIDATION_ROOT/dev-profile-p001"
export ISP_SERVICE="$ISP_VALIDATION_ROOT/naive-session-001"
export ISP_CAPTURE="$ISP_VALIDATION_ROOT/live-capture-001"
export ISP_INPUTS="$ISP_VALIDATION_ROOT/pilot-inputs"
mkdir -p "$ISP_INPUTS"
cp configs/baseline.example.yaml "$ISP_INPUTS/baseline.local.yaml"
cp configs/control.example.json "$ISP_INPUTS/control.local.json"
cp configs/profile-rules.example.json "$ISP_INPUTS/profile-rules.local.json"
```

编辑这三个副本。不要原样运行含 `/path/to`、null 或空数组的模板。

| 文件 | 必须填写/确定 |
| --- | --- |
| `baseline.local.yaml` | `repo_dir`、`python`、`runs_dir` 都写实际绝对路径；TRAIN/DEV 三目录；固定 epochs、batch、seed、初始化和 P512；删除整个 `test:` 块 |
| `control.local.json` | 身份、问题、真实授权引用、源码版本、批准配置引用；全部数值 limits；四字段允许范围；`test_permission=false`；首轮无确认可保持 `confirmation_scope={authorized:false,kind:none}` |
| `profile-rules.local.json` | 保留 TRAIN 校准、TRAIN/DEV 扫描、固定 revision、`retrospective=false`；要验区域反馈，将默认 `roi:false` 改为 **true** |

YAML 相对路径从 **YAML 所在目录**解析。把模板移到 `pilot-inputs` 后，原来的 `repo_dir: ..` 不再指向 ISP root，必须改；`python` 用 `$ISP_PYTHON` 的实际展开路径。正式配置写实际数值与路径，不写 shell 变量字符串。

数据严格配对：输入 PNG、GT JPG/JPEG、metadata JSON 按 scene stem 一一对应，输入末尾 `_denoised` 会剥除。例如 `scene001_denoised.png / scene001.jpg / scene001.json`。重复、缺少或额外 scene 会失败。输入是 photofinishing 预期的已去噪 RAW RGB，不是任意成片 PNG 或直接 DNG；metadata 至少满足现有读取的 `cam_illum / ccm`。不能据此假定有 ISO、曝光时间或真实噪声强度。

采用初始化 checkpoint 时，`init_config_dir` 必须有 `<checkpoint stem>.json`，包含 `use_3d_lut`。每轮从同一初始化开始，不从 `based_on` 的 checkpoint 续训。默认模板 600 epochs 不是轻量验收预算；在启动前选定实际可承受且已授权的固定 pilot 训练预算，不能中途改。

### 5.2 资源合同

`limits` 六项都须填写数字：`job_walltime_seconds` 为有限正数，`allocated_gpus` 为非负整数，pilot 的 `max_search_trials=4`；总 GPUh、确认预留、校准预留均为有限非负数。正式策略实验另建独立 control campaign，将其预注册的 `K`、策略组和确认预算写入新合同，不能修改已启动的 pilot。搜索每次按最大占用先预约：

\[
B_{job}=\frac{\text{allocated\_gpus}\times\text{job\_walltime\_seconds}}{3600},\qquad
B_{total}\ge4B_{job}+B_{confirmation}+B_{calibration}.
\]

这里是保守预约，不是预测平均训练耗时。CPU fixture 可使用 0 GPUh，但仍有有限 walltime。实际训练只有一个可见设备时，按真实分配记账；不能靠填 0 运行 GPU 规避预算。`authorization_ref` 指向已有真实授权记录，模板文字不构成授权。

保留 `slow_policy.consecutive_valid_no_gain=2`；初始化的 `no_gain_limit=3` 是另一条搜索停止条件。两次有效 no_gain 可以先触发科学复盘。合同初始化后会复制冻结，修改外部文件不扩展已运行 campaign 的权限。

```bash
"$ISP_PYTHON" -m tm_research.cli control --control "$ISP_INPUTS/control.local.json"
"$ISP_PYTHON" -m tm_research.cli commands --config "$ISP_INPUTS/baseline.local.yaml" \
  > "$ISP_VALIDATION_ROOT/evidence/command-preview.json"
"$ISP_PYTHON" -m tm_research.cli profile \
  --config "$ISP_INPUTS/baseline.local.yaml" \
  --rules "$ISP_INPUTS/profile-rules.local.json" --output-dir "$ISP_PROFILE" \
  > "$ISP_VALIDATION_ROOT/evidence/profile-build.json"
```

通过：合同和配置校验通过，配对计数符合预期，生成 `$ISP_PROFILE/profile.json` 及 ROI 资产。`commands` 只是预览，不训练；其中 checkpoint 占位符不应拿去直接执行。profile 在搜索前冻结，不能根据候选输出重划区域。TRAIN 全像素分位数构建可能占用较多 RAM，先规划实际数据规模。

### 5.3 固定 `min_delta`

使用目标环境、同一 checkpoint、同一 config、同一 DEV 和 P512，事先做重复独立重载。例如以下原生命令，另两次只换 `result-dir` 为 `reload_2 / reload_3`：

```bash
# 先将这五个变量填写为实际路径，再执行；评测也需要单独安排资源预算。
"$ISP_PYTHON" photofinishing/test.py \
  --model-path "$ISP_CALIBRATION_CHECKPOINT" --config-dir "$ISP_CALIBRATION_CONFIG" \
  --in-testing-dir "$ISP_DEV_INPUT" --gt-testing-dir "$ISP_DEV_GT" \
  --data-testing-dir "$ISP_DEV_METADATA" --eval-size 512 \
  --result-dir "$ISP_VALIDATION_ROOT/evidence/calibration/reload_1"
```

保留每次原始命令、日志和 `metrics.json`；检查 count、协议、finite 一致。根据测量波动和预先声明的容差确定阈值，随后填写：

```bash
export ISP_MIN_DELTA='填写事先确定的非负 dB 数值'
```

`0.01 dB` 仅为旧工程示例，不是实测通用阈值。重复重载不测训练 seed 方差。**当前没有校准启动/登记 CLI**；此事前操作的成本由外部预算记录保存，不会自动写入 campaign 账。不要编辑 `campaign.json` 补造校准 trial。

## 6. 启动真实 Naive 与 ARIS

在已经验证能加载真实权重的独立推理环境启动 adapter。模块在 applied ISP root；不同环境用它自己的解释器，并显式指定 overlay 的 `PYTHONPATH`。下列变量要填实际值：

```bash
export NAIVE_PYTHON=/absolute/path/to/naive-env/bin/python
export NAIVE_MODEL=/absolute/path/to/actual-naive-weights
export ARIS_VERSION='实际已安装 ARIS-Code 版本'
# 单机使用此地址；分机改为绑定后双方可访问的实际接口地址。
export NAIVE_BIND=127.0.0.1
PYTHONPATH="$ISP_RUNTIME" "$NAIVE_PYTHON" -m tm_research.naive_adapter \
  --model-id "$NAIVE_MODEL" --host "$NAIVE_BIND" --port 8000 \
  --workflow-dir "$ISP_SERVICE" --host-version "$ARIS_VERSION"
```

让服务运行在独立终端或已授权的服务调度中。加载并绑定成功后应存在 `service.json / startup.json / requests.jsonl`。每次重启用**新的 service 目录**；capture 绑定一个 campaign 和一次 startup，不能混用。不能用注入 generator、mock model 或手写 startup 当真实权重证据。

ARIS 配置只合并四项，保留已有 reviewer/tool 设置：

```json
{
  "executor_provider": "openai",
  "executor_api_key": "local-naive",
  "executor_base_url": "http://127.0.0.1:8000/v1",
  "executor_model": "/absolute/path/to/actual-naive-weights"
}
```

`executor_model` 应与 `--model-id` 一致；分机 URL 改为实际可达地址。adapter 本身没有 HTTP 认证校验，示例 key 只是兼容配置。ARIS 的既有配置位置见 [服务设置](naive-setup.md)；使用已安装宿主的真实配置/导出方法，本仓库没有 `aris --export-session` 等接口。

另开终端，恢复第 3、5 节变量后，做小请求：

```bash
"$ISP_PYTHON" - <<'PY' > "$ISP_VALIDATION_ROOT/evidence/naive-hello-request.json"
import json, os
print(json.dumps({'model': os.environ['NAIVE_MODEL'], 'stream': True,
                  'temperature': 0, 'max_completion_tokens': 64,
                  'messages': [{'role': 'user', 'content': 'Say hello'}]}))
PY
curl -N "http://$NAIVE_BIND:8000/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  --data-binary @"$ISP_VALIDATION_ROOT/evidence/naive-hello-request.json"
```

接口必须 `stream:true`；它是 SSE。小请求只验证服务协议，不是研究验收。不要用 `/v1/models` 或 GET health 检查此 adapter，它没有这些路由。

已有其他兼容 endpoint 可以驱动真实会话，但**当前自动 W 转换器只支持本仓库 adapter 的原始身份/请求记录与 CLI capture**；第三方服务日志没有自动转换入口，应保留原始记录人工审阅，不能默认 W 自动接通。

## 7. ARIS 宿主的验收接入合同

**先检查这一节能否实现，再启动 B 阶段。** 普通 `aris "Read task..."`、Bash 终端转录或图片路径，均不足以满足自动 W。仓库提供 capture/登记工具，未自动安装原生 ARIS 的消息截取和返回 JSON 保存接入。

宿主必须按下面顺序处理：

1. 调用 captured `campaign next` 或 `campaign feedback`，保存完整返回 JSON。
2. 将**完整对象原样 JSON 序列化**作为单独一条 `user` 或 `tool` message 的 content 发给真实 Naive。任务指令放另一条 message；不能只给内部 `feedback`、路径、摘要，也不能在该 JSON content 前后添加说明或用另一层 Bash 输出对象包装。
3. 要求真实模型可见回复是恰好包含 `proposal`、`decision` 的 JSON 对象，无 Markdown fence。收集 SSE 的 `choices[0].delta.content`；忽略 heartbeat、role、私有推理和 tool-call 参数。结束须正常 `stop` 且有 `[DONE]`，`length` 不是完成的提案。
4. 从模型回复提取两个对象，分别原样保存为 proposal 和 decision 文件。可改 JSON 缩进，不能改 ID、事实、字段、recipe 或补写解释。
5. 用 captured `campaign submit` 提交这两个文件，等待实际训练与独立 DEV 完成。拿到新结果后再发下一轮反馈，不预先一次生成三轮提案。

支持的可见输出形状如下，`{...}` 只是结构示意，不能提交：

```text
{"proposal": {实际严格三键提案}, "decision": {实际有效 fast_proposal sidecar}}
```

严格 proposal 顶层只有 `recipe / hypothesis / based_on`。decision 以 [`decision.example.json`](../configs/decision.example.json) 为 schema 参考：填真实 `feedback_ref / latest_seen_run_id / reference_run_ids / observation_refs` 和预测、反证、替代解释；`requested_change` 与 recipe patch 完全相同。观测引用使用实际 `<campaign_id>/<run_id>/<feedback_revision>/<observation_id>`，不能拿示意 O1 代替。`based_on` 是证据引用；部分 recipe 继承当前 effective best，不是自动继承 `based_on` 的配置。

模型写文件的 tool-call 可以服务普通工作流，但**仅靠 tool-call 写 proposal 文件，不能满足当前自动 W 的上述可见 JSON 来源要求**。若返回无效，保留失败记录、重新请求真实模型，不由操作者修补成“模型成功输出”。

如果已安装的 ARIS 无法保留上述消息/回复，先停在“宿主接入待完成”：可以人工审阅真实会话，自动 W 保持 pending。操作者直接 HTTP 请求能证明模型读过反馈，但不能据此称 ARIS 无人值守运行；记录真实介入次数。W 校验关联来源和版本字符串，也不是对宿主自主性的独立认证。

## 8. 执行 baseline + 三次真实提案

### 8.1 操作者先初始化有合同、有诊断的 campaign

在 applied ISP root，确保前面变量恢复、服务已启动、profile 已冻结。不要让旧任务的 legacy init 示例遗漏 `--control / --profile`。第一次初始化：

```bash
"$ISP_PYTHON" -m tm_research.cli capture-workflow \
  --capture-dir "$ISP_CAPTURE" --service-dir "$ISP_SERVICE" -- \
  campaign init --campaign-dir "$ISP_CAMPAIGN" \
  --config "$ISP_INPUTS/baseline.local.yaml" --control "$ISP_INPUTS/control.local.json" \
  --profile "$ISP_PROFILE/profile.json" --max-trials 4 --no-gain-limit 3 \
  --min-delta "$ISP_MIN_DELTA" --wait \
  > "$ISP_VALIDATION_ROOT/evidence/campaign-init.json"

"$ISP_PYTHON" -m tm_research.cli capture-workflow \
  --capture-dir "$ISP_CAPTURE" --service-dir "$ISP_SERVICE" -- \
  campaign next --campaign-dir "$ISP_CAMPAIGN" \
  > "$ISP_VALIDATION_ROOT/evidence/next-001.json"
```

检查 baseline trial **valid**、重载 DEV 协议/count/finite 正确、反馈文件可读。CLI 退出 0 不代表 baseline 有效：init 也可能返回已停止或无效状态。baseline 是首个 `run_completed` 事件。

### 8.2 将已初始化 campaign 交给真实 ARIS

以下使用现有文档的 ARIS task 调用形式；须先连好第 7 节宿主接入，并在 ISP root 启动：

```bash
cd "$ISP_RUNTIME"
source "$ISP_RUNTIME/.venv/bin/activate"
aris "Read docs/aris-campaign-task.md and resume the EXISTING campaign: $ISP_CAMPAIGN. Do not initialize another campaign. Use the configured real Naive executor. Use the absolute controller interpreter $ISP_PYTHON for every tm_research command. Invoke campaign next/feedback/submit/wait/status/record-decision through capture-workflow with capture-dir $ISP_CAPTURE and service-dir $ISP_SERVICE. The initialized control and profile are authoritative. Only next_action selects actions; proposal_allowed=false alone is not stop. Deliver each complete captured feedback JSON unchanged as a standalone model message and save the model's visible proposal/decision JSON unchanged before submit. Execute at most three sequential candidate trials after the completed baseline within the existing four-trial budget. Preserve the original host session and all interventions. Do not access TEST."
```

此命令是任务入口，不保证你的 ARIS 版本已经实现接入。保留其真实会话导出；记录宿主、模型身份及调用方式，不能用手册中的版本占位字符串代替实际身份。

每个正常候选的宿主命令如下。`proposal-001.json / decision-001.json` 必须来自该轮真实模型回复，不提供可冒充模型输出的预设 recipe。

```bash
# 仅校验，不启动训练。
"$ISP_PYTHON" -m tm_research.cli decision --campaign-dir "$ISP_CAMPAIGN" \
  --decision "$ISP_VALIDATION_ROOT/evidence/decision-001.json"

"$ISP_PYTHON" -m tm_research.cli capture-workflow \
  --capture-dir "$ISP_CAPTURE" --service-dir "$ISP_SERVICE" -- \
  campaign submit --campaign-dir "$ISP_CAMPAIGN" \
  --proposal "$ISP_VALIDATION_ROOT/evidence/proposal-001.json" \
  --decision "$ISP_VALIDATION_ROOT/evidence/decision-001.json" --wait

"$ISP_PYTHON" -m tm_research.cli capture-workflow \
  --capture-dir "$ISP_CAPTURE" --service-dir "$ISP_SERVICE" -- \
  campaign next --campaign-dir "$ISP_CAMPAIGN" \
  > "$ISP_VALIDATION_ROOT/evidence/next-002.json"
```

检查新 terminal run，原样交给模型，再提交 002、003；每次换真实文件名。每轮 sidecar 的 `latest_seen_run_id` 应是紧邻的前一轮 terminal run，引用真实观测，并由实际完整配置生成 `changed_fields`。`original→mse/l1` 同时改变监督位置/辅助损失组合，不能归因成“损失形式单因素更好”。

### 8.3 只依 `next_action` 路由

| action | 宿主动作 |
| --- | --- |
| `wait` | 收集现有作业；不新建实验 |
| `diagnose` | 先按 reason 调查缺失诊断、协议或 unknown 活性；`review_required=true` 时进入当前事件复盘；不直接 finalize |
| `propose` | 仅在 proposal guard 允许时提交真实下一实验 |
| `request_scope_change` | 保持暂停，提出具体新授权范围；不改已冻结合同继续 |
| `finalize / report_stop` | 按路由冻结或结束；不继续搜索 |
| `confirm` | 仅执行已授权、冻结的确认计划 |

普通 diagnose 记录不会自动补回丢失证据或解决 unknown 作业；先检查实际文件与作业状态。需要慢复盘时使用当前 `trigger_id`：

```bash
# ISP_TRIGGER 从本次 next_action 获取，不使用旧事件 ID。
"$ISP_PYTHON" -m tm_research.cli campaign review-packet \
  --campaign-dir "$ISP_CAMPAIGN" --trigger "$ISP_TRIGGER" \
  > "$ISP_VALIDATION_ROOT/evidence/review-packet.json"
# 真实模型读复盘包后，生成 kind=slow_review 的决定文件。
"$ISP_PYTHON" -m tm_research.cli capture-workflow \
  --capture-dir "$ISP_CAPTURE" --service-dir "$ISP_SERVICE" -- \
  campaign record-decision --campaign-dir "$ISP_CAMPAIGN" \
  --decision "$ISP_VALIDATION_ROOT/evidence/slow-review.json"
"$ISP_PYTHON" -m tm_research.cli capture-workflow \
  --capture-dir "$ISP_CAPTURE" --service-dir "$ISP_SERVICE" -- \
  campaign next --campaign-dir "$ISP_CAMPAIGN"
```

`diagnose` 决定会保持诊断；“已写 review”不自动恢复提案。继续需当前事件的有效 `slow_review action=propose`，且权限、预算和 guard 均允许。终止或待改范围决定不能被另一个不同动作覆盖。自然没有慢事件时，记“live 慢复盘未触发”，不改分数/no_gain 制造事件。

**当前 W 要求三个已执行候选 sidecar 都为 `fast_proposal`。** 慢复盘先 `record-decision`；获准继续后取 fresh captured next，让模型另生成实际实验的 `fast_proposal` sidecar 再 submit。直接用 slow_review sidecar submit 虽然控制器可执行，当前 W 仍会 pending。慢复盘记录本身也不证明独立模型审查，保留真实 reviewer/host 消息。

`capture-workflow` 只支持 campaign 的 init/next/feedback/submit/wait/status/record-decision。review-packet、report、memory、finalize、confirmation 用普通 CLI，保留宿主原始记录；不要把它们硬塞进 capture。

## 9. 登记证据并判断 B 是否通过

```bash
"$ISP_PYTHON" -m tm_research.cli register-workflow \
  --campaign-dir "$ISP_CAMPAIGN" --capture-dir "$ISP_CAPTURE" \
  > "$ISP_VALIDATION_ROOT/evidence/workflow-registration.json"
"$ISP_PYTHON" -m tm_research.cli campaign report --campaign-dir "$ISP_CAMPAIGN" \
  > "$ISP_VALIDATION_ROOT/evidence/acceptance.json"
"$ISP_PYTHON" -m tm_research.cli campaign memory --campaign-dir "$ISP_CAMPAIGN" \
  > "$ISP_VALIDATION_ROOT/evidence/research-memory.json"
"$ISP_PYTHON" - <<'PY'
import json, os
from pathlib import Path
report = json.loads((Path(os.environ['ISP_VALIDATION_ROOT']) / 'evidence/acceptance.json').read_text())
print(json.dumps(report['claims']['W'], ensure_ascii=False, indent=2))
PY
```

同时查看 campaign 内 `report.md`。`register-workflow` 成功不是 W 通过。B 的验收需同时满足：

- 4 个有效原生 run 的 checkpoint、重载指标、样本数、协议及实际配置一致。
- 真实权重加载记录与实际服务一致，不是 fixture；三次模型读取的反馈与 captured 返回对象一致。
- 事件链是 baseline 完成，然后三次“前轮反馈产生→模型读取→模型回复→匹配 submit→新 run 完成”，时间/ID/recipe/证据引用都能对应。
- `claims.W.status=accepted`；保留并审阅实际 ARIS 会话，核实工具调用、等待、读取与人工介入，而非只看服务里的宿主版本文字。
- 没有 TEST 访问、无未授权扩范围、无删除失败或补写成功事件。

不要求三次候选都 improved。若合法停止早于四次、某轮 invalid 或宿主接入缺失，则 B 尚未完成；这也可能同时证明控制器正确停止。需要重做时使用明确新授权的新 campaign，不扩大原四次预算或擦除旧记录。

登记可以幂等重试，追加的原始记录可再次登记；已登记部分不能修改。保持服务/请求/capture/campaign/反馈原路径可读，不能登记后移动或整理掉文件。

保留最小证据集：配置/授权/源码版本/profile/min_delta 校准；服务三件套；capture 原始 invocations；实际 ARIS 导出；原样 proposal/decision/slow review；campaign 状态、版本化反馈及决策；每 run 的 config、commands、state、训练与 DEV 日志、checkpoint、metrics、逐图/区域/案例诊断；登记 manifest/events；report/memory；外部推理费用与人工介入表。

再人工检查研究反馈有没有发挥作用，不把字符串校验当预测验收：

| run | 实验前引用的观测 | 预期指标/组/方向 | 实际测量与来源 | 满足/不满足/信息不足 | 保留或放弃的解释 | 下一步 |
| --- | --- | --- | --- | --- | --- | --- |
| 实际 run ID | 实际 observation_refs | 原模型 prediction | 原始指标/诊断路径 | 由证据填写 | 引用该轮模型回复 | 下一提案或停止 |

目前没有自然语言预测自动判定 CLI；这张表是人工复核记录。ROI/分组测量描述错误表现，不自动证明某内部模块或噪声原因。

## 10. 可选冻结确认与后续价值验证

### 10.1 已授权后才执行确认

无确认授权的首轮 pilot 到第 9 节即可。若要确认，初始化 campaign 前就明确 `confirmation_scope.authorized=true`、`kind=original_dev` 或 `independent`、整数 `seeds`、`replicates`、独立数据（如使用）和预算。独立确认的 control 与 plan.scope 须指向同一 `split_spec`（input_dir/gt_dir/metadata_dir 及可选 expected_count），均用实际绝对路径。完整合同见 [快慢科研配置](fast-slow-research.md)。冻结前不能根据结果挑有利 seed；不通过改外部 control 扩大旧 campaign 权限。

复制 [`confirmation.example.json`](../configs/confirmation.example.json) 后填写 seeds、replicates 和 `budget_gpu_hours`。**模板的 `delta_useful_db:null` 会被拒绝：填有限非负数，或删该键做描述性确认。** 首次采用 `final_checkpoint_rule={kind:retain_search_checkpoint}`。原 DEV 确认预算至少预约 `2×seed数×replicates×B_job`；独立确认还有额外冻结后评测阶段，当前代码保守预约再乘 2。

```bash
# 仅在 next_action 允许且没有活动/待执行作业时冻结；已 frozen 则复用。
"$ISP_PYTHON" -m tm_research.cli campaign finalize --campaign-dir "$ISP_CAMPAIGN"
"$ISP_PYTHON" -m tm_research.cli confirmation init \
  --campaign-dir "$ISP_CAMPAIGN" --confirmation-dir "$ISP_CAMPAIGN/confirmation" \
  --plan "$ISP_INPUTS/confirmation.local.json" --control "$ISP_CAMPAIGN/control.json"
"$ISP_PYTHON" -m tm_research.cli confirmation next \
  --confirmation-dir "$ISP_CAMPAIGN/confirmation" --wait
"$ISP_PYTHON" -m tm_research.cli confirmation report \
  --confirmation-dir "$ISP_CAMPAIGN/confirmation"
"$ISP_PYTHON" -m tm_research.cli campaign final-checkpoint \
  --campaign-dir "$ISP_CAMPAIGN"
```

`confirmation next` 一次推进一个 task，按返回状态重复，terminal 后 report，不重新 init 或择优重试失败配对。默认 retain_search_checkpoint 的 final-checkpoint 不传 confirmation-dir；仅另行采用事先固定的 `predeclared_seed` 且对应 task 有效完成时，才传该参数绑定指定确认权重，不能确认后择优挑 seed。确认不更新搜索 best。原 DEV 多 seed 仅支持该适应性 DEV 的稳定性；独立确认必须有授权、无场景重叠的独立资源，不能把 TEST 改名。保留失败/不完整配对，报告支持范围和不确定性。

首次注册的可执行确认计划是冻结的主计划，后续计划只能作为探索性结果披露；Q 与 final-checkpoint 均服从主计划，显式传入另一个 confirmation-dir 也不能事后改选。旧 campaign 若已有多个计划却没有保存主计划绑定，不能据此补造预注册结论。报告会重新检查实际配置、运行状态、协议和分数，不能只凭已完成 task 的缓存值宣布支持。

### 10.2 研究收益和产品证据另验

等预算比较由[科研循环实验方案](plans/naive-research-loop-experiments-20261005_CN.md)统一规定为五组：Random、TPE、Naive Scalar、Naive Rich Fast、Naive Rich Fast/Slow。它扩展了早期四组建议，并把 TPE 作为更强的非 LLM 搜索基线。正式组使用独立 campaign、相同训练/评测门禁和预注册确认，不复用本 pilot 的四轮预算。

正式实验的初始规划为每组 10 个配对搜索块、每个 campaign 最多 12 次训练、冻结后 3 个新种子确认；最多 900 次训练和 300 次独立确认评测。该数字是资源上限，不是已验证的统计功效或耗时预测。正式实验还必须预先冻结 `min_delta`、有用提升、提示、策略随机种子、失败处理和成本采集方法。

**当前没有一组现成 CLI 开关可直接运行这些策略消融。** 需要预先登记独立策略实验及宿主输入/行为控制，不能虚构 `--disable-slow` 等参数。本次 W 通过不能代替 R；报告中的推理成本 unavailable 也不能记成 0。

PSNR 改善不自动证明人脸、高光、自然度或新相机效果，也不自动归因于 TM 模块。产品验证 `P` 保持另行设计，不加入本次 DEV 选择循环。

## 11. 常见阻塞与恢复

| 现象 | 先查什么 | 正确处理 |
| --- | --- | --- |
| `_stage_exec` 拒绝 owner 或 exit 125 | 实际容器的 PID/`/proc` 视图、原始 stage 日志 | 修部署，重跑原生回归；不跳过身份 guard |
| baseline invalid，但 init exit 0 | trial 状态、train/DEV stderr、count/config/checkpoint | 保留失败，按真实预算处理；不是继续冒充有效 baseline |
| 没有区域反馈 | profile 是否初始化前 `roi:true`、ROI 文件是否可读 | 运行前修配置；已开始的 campaign 不临时换 profile |
| `proposal_allowed=false` | `next_action.action / review_required / trigger_id` | wait/diagnose/stop 分开处理，不统一 finalize |
| decision 被拒绝 | 最新反馈、观测 ID、允许字段、requested_change 是否一致 | 把拒绝交给真实模型，保存新回复；不人工改成成功提案 |
| W pending 或登记报错 | report reason；完整 JSON、可见输出、时间顺序、startup/capture 绑定 | 修实际宿主接入，保留原始记录；不手写 events |
| 服务重启或中断 | startup 是否变了、capture 是否仍绑定旧 session | 新 session 用新目录；普通 search 可按状态恢复，当前转换器不合并跨重启 session；自动四轮 W 未完成，新完整验收 campaign 需另行授权 |
| 作业未完成/活性 unknown | captured status/wait 和 `run inspect` | 收同一作业；unknown 时不重复启动 |
| 想取消已有作业 | 实际 run ID 和 YAML 中 runs_dir | 使用下面明确 stop，再收集状态；不直接删目录 |
| OOM/预算耗尽 | 实际显存、作业上限、reserved/consumed | 如实记录；改变固定预算/模型/数据需新授权 campaign |

```bash
"$ISP_PYTHON" -m tm_research.cli capture-workflow \
  --capture-dir "$ISP_CAPTURE" --service-dir "$ISP_SERVICE" -- \
  campaign wait --campaign-dir "$ISP_CAMPAIGN" --timeout 60
"$ISP_PYTHON" -m tm_research.cli run inspect --run "$ISP_RUN_ID" --runs-dir "$ISP_RUNS_DIR"
# 仅在操作者实际决定取消该作业时执行。
"$ISP_PYTHON" -m tm_research.cli run stop --run "$ISP_RUN_ID" --runs-dir "$ISP_RUNS_DIR" \
  --reason 'Operator requested cancellation'
```

最后给出分项结论：A 部署是否通过；B 四轮来源链与真实宿主是否通过；自然慢复盘是否发生、如何改变控制；C 确认支持什么范围；D 策略价值尚有哪些未知。允许“执行已验证，收益未证明”；不要仅因 workflow 成功就声称自主科研或 PSNR 收益已验证。
