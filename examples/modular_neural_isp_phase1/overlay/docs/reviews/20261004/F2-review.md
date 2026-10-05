# F2 independent scoped review — 2026-10-04

Verdict: **changes requested**. No Critical findings; two Important findings and one Minor finding.

Scope: F2 requirement in `Code_Fast_Slow_Review_CN.md`, supplied F2 diff/report, new workflow module/tests/documentation, and the current W validation integration. No source edits, commits, full-suite rerun, real model loading, GPU work, or real TEST access. Narrow probes reused the explicitly labelled engineering fixture; its synchronous supervision substitution does not verify the production runner/watchdog.

## Important — retry cannot finish an interrupted registration

Location: `overlay/tm_research/workflow.py:258–271`, particularly the early return at 266–267.

The manifest is persisted before the campaign association. If `_save` fails (or the process stops) after manifest persistence, retry computes the same manifest and returns success before setting/saving `state['workflow_evidence']`. W remains pending indefinitely despite a successful retry. This violates interrupted-registration idempotency and leaves manual state manipulation or unrelated capture extension as workarounds.

Reproduced without editing campaign state: use the existing `captured_campaign` fixture and inject an `OSError` into `campaign._save` only when its state contains `workflow_evidence`; restore the original save method, then call `register_workflow` again. Observed:

```text
fault_left_manifest= True association_before= False
association_after_retry= False
```

Fix: an identical manifest must still ensure the derived index and campaign association are present/consistent before returning. Add a fault-injection regression covering interruption after manifest persistence and before campaign save.

## Important — declared weights identity is not bound to the actual load argument

Locations: `overlay/tm_research/naive_adapter.py:252–259`; `overlay/tm_research/workflow.py:124–134`.

`make_server` sets `weights_ref=model_name`, independently of `generate.loaded_identity['model_path']`. Any nonempty loaded identity enables live mode. Validation checks service/startup declarations against one another, but never checks those declarations against the recorded loader argument. A normal API caller can load one path and supply a display name/default model name; the generated service evidence then records the display name as the weights reference. The normal `main` currently passes one value to both arguments, but the supported server API and acceptance validator do not enforce that binding.

A narrow engineering probe passed a callback whose simulated successful load identity was `{'model_path': '/actual/other-weights', 'model_class': 'LoadedModel'}` to `make_server(..., model_name='claimed-naive', workflow_dir=..., host_version='test')`. No source/manifest was forged or edited. Observed:

```text
weights_ref= claimed-naive actual_load_argument= /actual/other-weights evidence_mode= live
derive_identity_check= []
```

`_derive` accepted that inconsistent source identity. This probe tests load metadata binding, not real inference. Fix: derive `weights_ref` from the actual recorded load path and validate it against startup loaded identity; validate required identity fields rather than accepting arbitrary truthy metadata. If display aliases are allowed, keep model alias and actual weights path distinct and explicit. Update engineering fixtures to simulate a coherent load identity, and add a mismatched-load-argument regression.

## Minor — malformed registration records escape the typed CLI error boundary

Locations: `overlay/tm_research/workflow.py:145–146,245–246`; CLI exception boundary in `overlay/tm_research/cli.py`.

Report-time validation normalizes malformed-source `IndexError`/`AttributeError` into `ValueError`, but `register_workflow` calls `_derive` without that boundary. A raw CLI record with otherwise valid schema/session/timestamps and `argv: []` produces an uncaught `IndexError`, not the advertised JSON error/exit code. Other wrong JSON container types similarly raise `AttributeError`/`TypeError`.

Narrow probe through `cli.main(['register-workflow', ...])` observed:

```text
malformed_cli_uncaught= IndexError list index out of range
```

Fix: validate raw record/container/argv shapes or normalize malformed-evidence exceptions at the registration API boundary, with a small CLI regression. This fails closed on acceptance, so severity is Minor.

## Positive scope checks

The ordinary capture path executes actual controller commands and retains their returned JSON. Conversion requires exact returned feedback in a subsequent captured model request, an actual parsed generation matching proposal and decision, and a later successful submit tied to the campaign trial/decision snapshot. Derived events cannot be supplied as raw evidence. Registration is locked; report-time validation rereads preserved raw prefixes, rederives events, and retains current W run/metric/decision checks. Append-only post-registration service records are excluded from the old evidence scope. Documentation clearly labels unsupported host export formats and engineering/live limitations. No claim of live Naive/ARIS performance or full production-runner verification is warranted or made here.

Probe scratch location: `/tmp/f2-review-xc6fzfje`. The probes ran in approximately 1.3 seconds and were not a suite rerun.
