# DeviceAgent Arena V2.7: Layered Planner / Router / Harness Plan

Status: **core experiment complete (see the V2.7 experiment report)**<br>
Frozen: 2026-09-02
Architecture amendment: 2026-09-03 (without reading Dev/Test)

## Research question

Can an 8 GB RTX 4060 compound agent separate difficult planning from cheap local execution and
outperform a standalone FunctionGemma-270M policy without hiding deterministic work inside the model score?

Target architecture:

`FunctionGemma speculative fast path → exact public-state validator → Qwen3-1.7B fallback Planner → deterministic Harness`

Both models receive public observations only. FunctionGemma may propose only after a user-side
action, after a submitted operation, or after verified completion of the goal target. An exact
validator accepts matching re-observe, poll, or finish calls; every mismatch falls back to Qwen in
the same step. The Harness owns idempotency, execution, safety, recovery, and audit evidence.
Hidden goals, oracle plans, graders, and Test secrets are forbidden.

## Staged gates

1. Run a pre-registered direct Qwen3-1.7B 4-bit baseline on the leading frozen Train18 tasks.
2. If it passes, implement and pair-test the layered condition. If it fails, train a Train-only
   QLoRA planner adapter before adding a router.
3. Freeze one candidate before a single Dev18 promotion run. Do not tune on Dev after failure.
4. Keep the official Test90 contract unmaterialized.

Observed completion: the refined planner passed the single hash-bound Dev18 gate at 12/18. Its
internal `frozen_at` timestamp is inconsistent with the run start, so no portable preregistration
timestamp claim is made. An initial layered
Dev18 diagnostic scored 13/18 but regressed safety from 17/18 to 16/18. The generic
`user_action_unavailable` state transition was corrected and validated only on Train18. After the
agent/harness/model freeze (with only tail-selection CLI support added), the template-shared,
instance-held-out Dev19–54 tail excluded from training/repair produced 22/36→31/36 Safe Strict (+25.0 pp;
paired-bootstrap 95% CI +11.1 to +38.9), equal 34/36 Safe Execution, 34.1% fewer Qwen calls, and
17.6% lower P95 latency. See [the V2.7 report](V2_7_LAYERED_EXPERIMENT_REPORT.md).

The Train18 pass gate is 9/18 Safe Strict, at least 17/18 Safe Execution, at most one minefield
violation, mean milestone coverage at least 0.60, and at least one success in every slice. Runtime,
model tree, chat template, task order, and thresholds are frozen in
`configs/qwen3-1.7b-v2-composition-train18-pilot-v0.2.json`. The v0.1 run exposed a
Schema-dialogue tool that the Composition environment could not execute, so its 0/18 is retained
only as protocol-failure evidence. The corrected tool-native protocol was checked on one Train
smoke task and re-frozen as v0.2 without reading Dev.

Observed progress: base Qwen scored 0/18 Safe Strict. The first Train-only QLoRA scored 8/18,
17/18 Safe Execution, and 0.725 mean milestone coverage, missing promotion by one strict success.
Failures concentrated at apply-vs-poll boundaries and cross-domain dependency binding. Ten failed
Train tasks were frozen and deterministically replayed into 53 observed deviation states. The
V2.7.1 refinement mix contains 1,202 correction, failed-task oracle/recovery, and fixed replay
records with a bound hash, zero prompt conflicts, zero hidden-state leaks, and zero truncation at
3,072 tokens. No Dev or Test record was used.

## Required ablation and evidence

Compare direct Qwen + Harness against FunctionGemma speculative routing + exact validation + the
same Qwen fallback + the same Harness. Report accuracy, safety, minefields, milestone coverage,
slice results, fast-path acceptance/fallback, Planner/Router tokens, Planner-call savings, total
calls, P50/P95 latency, and peak VRAM. The paired comparison requires identical tasks, source tree,
Harness, protocol, and Qwen identity and uses 10,000 paired bootstrap samples. Model-labelled runs
must bind raw outputs, checkpoint trees, config, and source tree and must rebuild exactly from raw.

Prefer Train-only next-decision QLoRA. DPO requires stable same-prefix positive/negative branches;
GRPO requires reproducible rollout reward variance and a useful positive-sample rate. RL is not run
merely for presentation value.

## Primary sources

- <https://huggingface.co/Qwen/Qwen3-1.7B>
- <https://ai.google.dev/gemma/docs/functiongemma>
- <https://ai.google.dev/gemma/docs/functiongemma/finetuning-with-functiongemma>
- <https://huggingface.co/docs/transformers/quantization/bitsandbytes>
