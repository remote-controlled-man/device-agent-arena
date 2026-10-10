# DeviceAgent Arena V2: Composition and FunctionGemma Experiment Report

Chinese: [V2_COMPOSITION_EXPERIMENT_REPORT.zh-CN.md](V2_COMPOSITION_EXPERIMENT_REPORT.zh-CN.md)

Status: executed; FunctionGemma candidate rejected; Test remains locked<br>
Date: 2026-09-02

## Result

This slice evaluates a complete agent loop rather than one-shot function classification: inspect the target, discover relations, coordinate user-side actions, execute dependencies, poll results, bind an upstream operation ID, verify final state, and refuse composed infeasible requests.

The frozen suite contains Train90, Dev54, and a reserved but unmaterialized Test90 contract over dual control, composed infeasibility, and cross-domain chains; three domains; and linear, fan-out/join, and diamond-gate topologies. A hash-bound amendment limits the legacy `root_seed` to Train/Dev; official Test requires an independent external 256-bit secret. Neither that secret nor `test.jsonl` exists in the repository.

| Condition | Scope | Safe Strict | Worst slice | Minefield |
|---|---:|---:|---:|---:|
| Oracle | Dev54 | 54/54 | 100% | 0% |
| Finish immediately | Dev54 | 0/54 | 0% | — |
| Refuse immediately | Dev54 | 0/54 | 0% | — |
| Public reactive harness | Dev54 | 45/54 | 50% | 0% |
| Public recovery harness | Dev54 | **54/54** | **100%** | **0%** |
| FunctionGemma v2.3 zero-shot | frozen Dev18 | 0/18 | 0% | 0% |
| FunctionGemma v2.5 SFT | frozen Dev18 | **3/18** | 0% | 7/18 |

The 100% result is a deterministic public-contract controller regression, not a learned-model or Test score. It receives no hidden task, goal, oracle plan, or grader truth.

## FunctionGemma training

v2.5 uses 1,985 native next-decision records: 1,260 composition-oracle records and 725 old-capability replay records. The tokenizer audit found 0 truncations, 0 empty completions, and 0 prompt-prefix mismatches at 2,304 tokens. One BF16 full-parameter epoch on an RTX 4060 Laptop used 3,394 MiB peak VRAM and 713.6 seconds. Dev18 improved from 0/18 to 3/18 but failed the preregistered 9/18 gate, with dual control and cross-domain chains both at 0/6.

Raw trajectories show a consistent failure mode: the model learns `inspect_target → inspect_relations`, then mutates the final target prematurely and loops on the same action after `precondition_failed`.

v2.6 adds 3,185 Train-only records: 1,800 hard successful mid/late decisions, 660 deviation-recovery decisions, and 725 replay records. A second BF16 epoch used 3,402 MiB and 1,196.7 seconds. Train18 remained 4/18: ordinary dual control improved, but cross-domain binding stayed 0/6 and infeasibility/safety regressed. Because end-to-end Train performance did not improve, v2.6 did not consume Dev18 again.

## Why RL was not started

Failures are highly deterministic under fixed decoding: relation misinterpretation, repeated invalid mutation, and cross-step binding. SFT has not established a reliable Train loop or stable same-prefix successful/failed branches. DPO therefore lacks trustworthy preference support, while GRPO would face near-zero reward variance and long-horizon credit assignment. The predefined trigger was not met, so the negative result is retained.

## System conclusion

FunctionGemma-270M is useful for schema grounding and local tool decisions but should not own every deterministic execution constraint. The evidence supports a layered design:

- model for semantic interpretation and genuinely open choices;
- harness for schema validation, dependency order, dual-control state, bounded retries, idempotency, and operation binding;
- grader and telemetry for milestone DAGs, minefields, final state, and failure attribution.

The public recovery harness solves 54/54 Dev tasks with zero minefields across all slices, domains, and topologies. This proves that the public contract is sufficient and the tasks are solvable without hidden state; it does not establish learned generalization.

## Evidence boundary

- Suite: `data/v2/deviceops-ul-v2-composition-v0.1/`
- Dev54 SHA-256: `9ddfc87c787b107ead4be1af235a932b49b04b4e968554c919e23b0e028de171`
- Final attested v2.5 raw/report/FAIL receipt: `artifacts/v2/composition/runs/functiongemma-v2.5-stage4-dev18-attested-v2.jsonl`, `reports/functiongemma-v2.5-stage4-dev18-attested-v2.json`, and `gates/functiongemma-v2.5-stage4-dev18-attested-receipt-v0.4.json`
- Training records: `artifacts/checkpoints/functiongemma-270m-deviceops-v2.5-full-sft-stage4/` and `...v2.6-full-sft-stage5/`

The final rerun occurred after audit hardening with no model, data, or policy changes and exactly reproduced 3/18. It is a reproducibility check on the same already-seen Dev, not a new independent sample. The v0.4 receipt binds the checkpoint tree, raw JSONL, exactly regenerated report, frozen protocol/decoder configuration, and every parsed raw model call to the executed call; the public runner rejects generic harnesses that request a model label. This is a local reproducibility attestation, not cryptographic proof against a malicious source-tree owner. A public leaderboard would still require an external evaluator signature. The old v0.1 receipt is retained only as weak historical evidence and cannot unlock future evaluation.

Composition Test90 remains locked. The frozen V1 Qwen Test210 result remains the project's only official Test result.
