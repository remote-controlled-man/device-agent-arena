# DeviceAgent Arena V2.7: Layered Tool-Use Agent Experiment Report

Status: **core experiment complete; suitable as the primary resume project**<br>
Hardware: **NVIDIA GeForce RTX 4060 Laptop GPU (8 GB)**<br>
Primary result: **same-task paired evaluation on 36 template-shared, instance-held-out Dev19–54 tasks excluded from training and repair**

## Executive result

V2.7 implements an auditable compound agent:

`FunctionGemma-270M proposal → exact public-state validation → Qwen3-1.7B fallback planner → deterministic harness`

With identical tasks, Qwen planner, harness, protocol, and source tree, the layered condition moves
Safe Strict from **22/36 (61.11%) to 31/36 (86.11%)**, a **+25.00 percentage-point** paired gain
(9 wins, 27 ties, 0 losses; 10,000-sample paired-bootstrap 95% CI **+11.11 to +38.89 pp**).
Safe Execution remains 34/36 and minefield violations remain 2/36. Qwen calls fall from 261 to 172
(-34.10%) and end-to-end P95 latency falls from 59.43 s to 48.99 s (-17.56%). Total model calls,
however, rise from 261 to 294 (+12.64%), input tokens rise 9.93%, and peak VRAM rises by 245 MB.

| Metric | Direct Qwen | Layered agent | Delta |
|---|---:|---:|---:|
| Safe Strict | 61.11% | **86.11%** | **+25.00 pp** |
| Goal Completion | 72.22% | **91.67%** | +19.44 pp |
| Safe Execution | 94.44% | 94.44% | 0 |
| Minefield Violation | 5.56% | 5.56% | 0 |
| Cross-domain output binding | 91.67% | **100%** | +8.33 pp |
| Dual-control failure recovery | 33.33% | **100%** | +66.67 pp |
| Qwen planner calls | 261 | **172** | **-34.10%** |
| End-to-end P50 | 31.14 s | **30.33 s** | -2.59% |
| End-to-end P95 | 59.43 s | **48.99 s** | **-17.56%** |
| Total model calls | 261 | 294 | +12.64% |
| Input tokens | 381,032 | 418,862 | +9.93% |
| Peak VRAM | 2,737 MB | 2,982 MB | +245 MB |

## Problem and architecture

This benchmark evaluates stateful tool use rather than one-shot function classification. Correct
actions depend on public history: the agent may need to disambiguate a target, inspect shared
effects, request a user-side action, bind a real operation ID, poll an asynchronous write, recover
from ambiguous failure, connect outputs across domains, or refuse an infeasible request. Safe
Strict requires correct final state, safe execution, protocol-compliant trajectory, legal calls,
and correct termination.

FunctionGemma-270M uses a native tool-call format and is cheap, but earlier model-only experiments
failed the preregistered long-horizon gate. Qwen3-1.7B is a stronger planner but need not handle
every locally verifiable transition. V2.7 therefore assigns ambiguity, infeasibility, dependency
binding, confirmation, and recovery to Qwen. FunctionGemma may propose only in three states with a
unique public-state answer: re-inspect after a successful user action, poll the exact operation ID
after submission, or finish after verified completion. An exact validator accepts both the function
and critical arguments; every mismatch falls back to Qwen in the same step. Neither model can read
hidden goals, oracle plans, graders, or Test secrets.

## Planner training and data audit

The protocol-corrected Qwen3-1.7B base scored 0/18 on Train18. A first 1,920-record Train-only
next-decision QLoRA reached 8/18, narrowly missing the 9/18 gate. Ten failed tasks were replayed to
locate 53 real deviation states. The refinement set contains 1,202 Train-only curriculum records
over 565 unique prompts: 424 repeated
state corrections, 258 failed-task oracle decisions, 40 recovery decisions, and 480 fixed replay
records. Audits found zero prompt conflicts, zero hidden-marker leaks, zero empty completions, zero
prefix mismatches, and zero truncation at 3,072 tokens.

Incremental NF4 QLoRA used bf16 compute, double quantization, LoRA r=16/alpha=32/dropout=0.05 over
all linear modules, effective batch size 8, and one epoch. It completed 151 steps in 3,367.5 s on the
RTX 4060, peaked at 4,926.8 MB allocated VRAM, and produced a 34.9 MB adapter. The refined planner
passed Train18 at 10/18 and then passed the single hash-bound Dev18 promotion run at 12/18. The
configuration file existed locally before the run, but its internal `frozen_at` timestamp was entered
later than the run start. Filesystem creation time is not portable, so no preregistration timestamp
claim is made.

## Safety regression and correction

The first exploratory layered Dev18 run scored 13/18 but reduced Safe Execution from 17/18 to
16/18. Raw trajectories showed that `user_action_unavailable` incorrectly activated a forced
`inspect_target` fast path, allowing the planner to continue from an invalid state. The generic
state machine was tightened: only `ok=true` responses with `done` or `failed` may enter this fast
path; `unavailable` and `error` fall back to Qwen. The change was validated only on Train18, where
the layered condition scored 13/18 versus direct Qwen's 10/18 with equal 17/18 safety. Agent,
Harness, and model behavior were then frozen; only CLI support for selecting the Dev tail and checking
the promotion receipt was added. Both final runs use the exact same source tree. Dev18 was not re-run
and presented as fresh evidence.

## Evaluation protocol

The final 36 tasks contain 12 examples from each domain and each composition slice. Train and Dev
share all nine instruction/templates; only instances, entity IDs, and states are held out. This is
not unseen-prompt or cross-schema generalization. Both conditions
share task IDs/order, source tree, harness, protocol, Qwen revision and adapter, deterministic
decoding, and planner generation limit. The comparison records per-task wins/losses and uses 10,000
paired-percentile bootstrap samples. The interval is task-level descriptive uncertainty over these
36 tasks, not a production-distribution confidence interval. FunctionGemma proposals were accepted in 107 of 122 eligible
states (87.70%); 15 rejected proposals fell back to Qwen.

Every run stores raw model outputs, parsed and executed calls, token/latency/VRAM telemetry, task
hashes, source-tree identity, model-tree hashes, adapter weights, and chat-template identity. The
layered attestation verifies that each executed model decision parses exactly from the recorded raw
output. Aggregate reports rebuild deterministically from raw trajectories.

## Limits

- Of the five strict failures, three are dual-control failures where the planner rewrites a public
  `available_actions` value as a target-suffixed action key and then loops on an unavailable user
  action. Two are infeasible workspace tasks where the planner fails to stop before a wrong or
  unauthorized mutation. The remaining bottleneck is therefore planner action-value grounding and
  infeasible termination, not the fast path.
- Those latter cases are the two remaining minefield violations. Safe Execution stays 34/36, so no
  broad safety claim is warranted.
- This is a template-shared, instance-held-out 36-task development tail, not unseen-prompt,
  external sealed-test, or production evidence.
- The environment is a deterministic text simulator, not a visual mobile/desktop control stack.
- Total calls and input tokens increase; the measured benefit is fewer expensive planner calls and
  lower tail latency, not lower cost on every axis.
- The exact validator can construct the unique fast-path action. A deterministic fast-path ablation
  that skips FunctionGemma is not yet available, so this experiment does not establish that a learned
  router is better than the corresponding rule.
- Latency comes from one sequential run per condition on this RTX 4060, not repeated interleaved
  measurements. The 17.6% P95 decrease describes this run only.
- Test90 remains unmaterialized.

## Evidence index

- Direct raw: `artifacts/v2/layered/runs/qwen3-1.7b-v2.7.1-refined-direct-dev36-tail-v0.1.jsonl`
- Layered raw: `artifacts/v2/layered/runs/functiongemma-v2.5-fastpath-qwen-v2.7.1-dev36-tail-v0.2.jsonl`
- Paired report: `artifacts/v2/layered/reports/functiongemma-v2.5-fastpath-vs-qwen-v2.7.1-direct-dev36-tail-paired-v0.2.json`
- Promotion receipt: `artifacts/v2/layered/gates/qwen3-1.7b-v2.7.1-refined-direct-dev18-gate-v0.1.json`
- Training result: `artifacts/checkpoints/qwen3-1.7b-deviceops-v2.7-composition-refinement-qlora-v0.2/device_agent_arena_training_result.json`
- Dataset manifest: `data/v2/qwen3-composition-refinement-sft-v2.7.1.manifest.json`
- Explicit post-hoc lineage amendment: `artifacts/v2/layered/training/qwen3-v2.7.1-refinement-lineage-amendment.json`
- Token audit: `artifacts/v2/layered/training/qwen3-v2.7.1-refinement-token-audit-3072.json`
- Release manifest: `artifacts/v2/layered/release-manifest-v2.7.json`

The resume-ready Chinese wording and STAR narrative are in
[RESUME_PROJECT_V2_7.zh-CN.md](RESUME_PROJECT_V2_7.zh-CN.md).
