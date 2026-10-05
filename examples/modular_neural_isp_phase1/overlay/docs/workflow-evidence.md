# Capturing and registering workflow evidence

`capture-workflow` executes the ordinary campaign CLI, preserves its actual
invocation, input JSON files, returned stdout/stderr and exit code, and binds the
capture to one campaign and adapter startup. `register-workflow` derives an event
index from those records and the adapter's actual model requests/responses, then
registers a manifest under the campaign lock. Neither command needs a manual
`campaign.json` edit. W still requires the existing four valid executed runs,
recorded decisions, available feedback artifacts, and matching actual DEV metrics.
Registration alone does not grant W, Q, R or P and never changes a score or outcome.

## Supported live path

Start the existing Naive bridge on its inference machine, with a **new** evidence
directory. The model loader finishes and HTTP binding succeeds before startup
identity is written. Supply the installed ARIS host version. Mount the evidence
directory readably on the controller machine at the same absolute path. Cross-machine
capture requires synchronized clocks: conversion enforces strict feedback/model/submit
time order and leaves inconsistent records unaccepted.

```bash
python -m tm_research.naive_adapter \
  --model-id /authorized/Naive-weights --host 127.0.0.1 --port 8000 \
  --workflow-dir /evidence/pilot-service --host-version '<installed ARIS version>'
```

Fill `configs/control.local.json` with the existing operator authorization for this
campaign before initialization. Keep using the existing ARIS host and authorized
Naive endpoint. Wrap its normal
controller commands, including baseline initialization and each subsequent
`next`, `feedback`, `submit`, `wait` or `status`, as follows:

```bash
python -m tm_research.cli capture-workflow \
  --capture-dir /evidence/pilot-cli --service-dir /evidence/pilot-service -- \
  campaign init --campaign-dir campaigns/pilot --config configs/pilot.yaml \
  --control configs/control.local.json --max-trials 4 --wait

python -m tm_research.cli capture-workflow \
  --capture-dir /evidence/pilot-cli --service-dir /evidence/pilot-service -- \
  campaign next --campaign-dir campaigns/pilot
```

For this minimal supported capture schema, deliver the **complete parsed JSON
returned by the captured `next` or `feedback` command** as a JSON string in one
model `tool` or `user` message. Ask Naive for visible JSON containing exactly
`{"proposal": <proposal object>, "decision": <decision sidecar>}`. Preserve the
normal sidecar requirements: the latest feedback revision and observations,
prediction, falsifier, alternative explanation, and requested recipe patch.
The host writes these two returned objects to proposal/decision files, unchanged.
The proposal itself retains its normal recipe/hypothesis/based_on schema. Then:

```bash
python -m tm_research.cli capture-workflow \
  --capture-dir /evidence/pilot-cli --service-dir /evidence/pilot-service -- \
  campaign submit --campaign-dir campaigns/pilot \
  --proposal proposal.json --decision decision.json --wait
```

Repeat the result-dependent sequence three times after baseline. `--wait` is the
simplest route; background submits need a returned actual active run and later
captured terminal `wait`/`status` output. Do not generate later proposals before
preceding results return. Failed calls remain raw failures; they cannot establish
a submission or completion. A truncated model generation cannot establish a
proposal.

After the session, register and inspect the ordinary report:

```bash
python -m tm_research.cli register-workflow \
  --campaign-dir campaigns/pilot --capture-dir /evidence/pilot-cli
python -m tm_research.cli campaign report --campaign-dir campaigns/pilot
```

Registering the same capture twice is idempotent. A retry after interrupted
registration restores the derived index and campaign association from the same
raw evidence, including when the manifest was already written. Append-only session extensions
can be registered again; changing already registered raw records is rejected.
Reports validate the registered raw prefix, so later appended unrelated service
requests do not invalidate an already complete capture. Keep all referenced
files at their registered paths. A service restart uses a new service directory;
a capture directory cannot silently switch startup identities.

## Preserved records and checks

- Adapter `startup.json` records session ID, bound endpoint, model/weights path,
  loaded model/tokenizer class and available model revision, and startup time.
  The served model name may be a display alias; `weights_ref` is the actual
  loader `model_path`, and conversion checks that binding. Live capture requires
  nonempty loader path, model class and tokenizer class; arbitrary truthy
  metadata cannot enable it.
  `service.json` references startup and request records. Injected test generators
  are marked `engineering_fixture`, which cannot establish live W.
- Adapter `requests.jsonl` records each completed generation's request messages,
  supported inference settings, raw model text, parsed visible content/tool calls,
  model/session identity and request timing. HTTP authentication headers are never
  saved; arbitrary request-body fields outside the supported inference fields
  are excluded. Prompt/tool content is retained as evidence and should contain
  no credentials.
- CLI `capture.json` binds campaign and service; `invocations.jsonl` retains
  command argv, proposal/decision input snapshots, actual outputs, return codes
  and timing. The capture invokes `tm_research.cli.main` directly and does not
  replace controller execution with user-supplied event objects.
- Campaign `workflow_evidence/manifest.json` preserves the original raw source
  texts and their locations. `events.jsonl` is a derived index. Validation rereads
  the raw sources, verifies their preserved prefix/identity, reparses raw model
  generation, and rederives the exact index each time W is evaluated.

Consumption requires an exact returned feedback payload in a later model request,
an actual model-produced proposal **and decision**, and the later matching
successful submit. The submitted decision must equal its campaign snapshot and
proposal must equal its actual trial. A pathname or an LLM statement that it
read feedback is insufficient. There is no score-gain requirement.

The Python API is
`validate_workflow_evidence(state, directory, provenance) -> list[dict]`; it raises
`ValueError` on missing/malformed/changed provenance or an unsupported fixture
service. The existing W validator retains responsibility for the valid-run count,
actual metrics, decision/observation access, and required ordered event coverage.
Incomplete coverage remains pending.

## Boundaries

No undocumented native ARIS JSONL format is imported. Generic Bash transcripts,
markdown-wrapped responses, shell tool calls that merely write proposal files,
and hand-authored `feedback_consumed`/`proposal_submitted`/`accepted` indexes do
not satisfy this converter. Adapt the existing host's messages to the supported
JSON response contract above, or retain the original session for manual review
and leave automatic W pending. This is a host hook, not a second agent runtime.

This audit trail assumes a trusted operator and filesystem. It detects changed
registered evidence and rejects a claimed event index as raw input; it is not
cryptographic attestation against someone fabricating every source and manifest.
Weights identity records the actual load arguments/classes/available revision,
not a byte-level attestation of all model weights.

The regression tests use an explicitly labelled engineering proposer and tiny
train/DEV substitutes. Their acceptance-branch test explicitly mocks the
successful model-load boundary. In a host with a mismatched `/proc` PID namespace,
the test helper also substitutes synchronous process supervision while running
real train/eval subprocesses. It does not validate the production watchdog.
No test demonstrates live Naive/ARIS research quality, real camera improvement,
GPU execution or access to real TEST.
