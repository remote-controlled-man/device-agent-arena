# DeviceAgent Arena v1 Experiment Report

中文: [V1_EXPERIMENT_REPORT.zh-CN.md](V1_EXPERIMENT_REPORT.zh-CN.md)

## Status

Benchmark sanity, the Train/Dev freeze, and the held-out-tool-name Test are complete. V1.0 failed because its loss mask was invalid. V1.1 corrected the mask but missed the safety gate. V1.2 scores 97.14% Safe Strict with zero boundary violations on Dev140, passing every preregistered promotion check. On the separately materialized Test210, it raises Safe Strict from Qwen3-1.7B base **10.95% to 62.86% (+51.90 pp; paired-bootstrap 95% CI +45.24 to +58.57 pp)**. The held-out naming-profile drop is material: boundary violations rise to 10.48% and dynamic replanning scores zero. V1 is frozen with both findings intact.

## Data evidence

| Artifact | Records | SHA-256 |
|---|---:|---|
| Train tasks | 700 | `f7edf25a0cf5e2eed058fce84516e49c6fe2a817e61e8c6cabfd34fcda22d5c8` |
| Dev tasks | 140 | `176b7e2649593d2bcc8cd11c2735110bbbec216d8babcf7ae0720e513c2a565b` |
| V1.0 full-trajectory SFT | 700 | `6a14857a31af474b0c9baec979b836f442c15cfbdff1bece617c5c9db47d4339` |
| V1.1 next-decision SFT | 4,200 | `885ff7ec3f9f7dcdb4ef40d87135f4577db444585ea4bc5fbaab25a01f7b77bb` |
| V1.2 public-grounded curriculum | 2,000 | `775ed56bf939015864ebac7f1c13dd8a4b7611498b23540ea84d34370b849f2c` |
| Test tasks | 210 | `7728379b6f20046b34dc17b9bf6efe20ecc08c92e1ea923995edb8edd7bee04d` |

Protocol-corrected evidence hashes: V1.2 Test raw run `c75089aac6d2477758b9afc8af90867fd66b3abbcaea6bafb71c14d754f8acb1`; base Test raw run `baeda4165d80689c5b0b8a53d4b67032188e6f3b6d1bcc8f83661eebf5703f0f`; protocol-checked paired report `2f495c456bc380c97d55c050abe40d77449a838c41ed326a7ea54c577cb70857`. The paired artifact marks source commit/tree, Harness, model revision, chat template, NF4, max tokens, and temperature as exact matches.

Protocol erratum: the first formal Test run recorded `max_consecutive_tool_errors=3` in its header, but V1 Environment did not enforce it. After implementing the breaker, the same frozen model, prompt, decoder, and Test were rerun strictly as a protocol correction. All 210 candidate grades and all 210 base grades match the first run exactly; only seven candidate and 29 base termination labels change to `tool_error_limit`. The headline metrics therefore come from the original formal Test and are independently reproduced by the corrected Harness, but the rerun is not presented as a new unseen Test.

Hidden-marker leakage is zero. The V1.1 Qwen3 chat-template audit finds 159,770 completion tokens, 4,796,400 masked prompt tokens, zero empty completions, zero prompt-prefix mismatches, and zero truncations at the frozen 2,048-token limit.

## V1.0 negative result and root cause

The first RTX 4060 QLoRA run genuinely completed: Qwen3-1.7B, two epochs, 176 optimizer steps, 3,561 MiB peak VRAM, 0.4064 train loss, and 0.9935 mean token accuracy. On 14 fixed Dev smoke tasks it nevertheless scored 0% Safe Strict, 14.29% Goal Completion, and 7.14% Boundary Violation, averaging only 0.5 tool calls before frequently terminating early.

Inspection of the locked TRL 1.9.2 pipeline established the cause. V1.0 used whole conversational `messages` records. Qwen3's chat template supplies no assistant mask, while `completion_only_loss=True` only applies a `completion_mask` to prompt-completion datasets. Consequently all system, user, tool, and assistant tokens contributed to V1.0 loss; the training artifact's original `completion-only next-agent decision` description is invalid. The low loss was not policy evidence.

V1.1 changes only the learning representation: each record is `prompt=[public history so far]`, `completion=[one next assistant decision]`. It starts from the base model, preserves all benchmark tasks and graders, and does not inherit the failed adapter.

## V1.1 rollout and second-order data defect

V1.1 completed 525 optimizer steps over 4,200 next-decision records in 6,477 seconds on the RTX 4060, with 0.0818 train loss. On the same 14 Dev smoke tasks it reached 64.29% Safe Strict, 64.29% Goal Completion, 85.71% Safe Execution, 14.29% Boundary Violation, and 75% Safe Recovery. The heuristic scores 28.57% Safe Strict and 50% Safe Recovery on those exact tasks. Ambiguity, direct low-risk, dynamic replanning, and timeout-after-commit were all solved; shared-impact tasks remained at zero and Confirmation Recall was zero.

Event-level inspection exposed a second data defect. The original shared-impact oracle used a device ID before any public observation introduced it, and mutation keys contained hidden task seeds. The model memorized Train `tr-*` IDs, searched only after not-found, and sometimes emitted confirmation as a nonexistent `confirm_operation` tool. Those targets were not derivable from the prompt and are not a model-capacity failure.

V1.2 leaves tasks, worlds, graders, and Dev frozen. Its 2,000 correction records enforce `search → inspect device → inspect routines → exact confirmation` for shared impact and derive idempotency keys only from public action/target context; dropped-write recovery is also replayed with a fresh public key. The real tokenizer audit has zero empty completions, zero prefix mismatches, and zero truncations. Training continues from the V1.1 adapter at a lower learning rate.

V1.2 completed 250 optimizer steps in 3,017 seconds with 0.00868 train loss. A separate reference-grounding audit checks 1,300 `device_id`, `target_id`, and `operation_id` uses; all occur in the corresponding prompt before use. On the same 14-task smoke, every archetype scores 2/2: Safe Strict, Goal, Safety, and Safe Recovery are 100%; Boundary Violation is zero; clarification and confirmation precision/recall are 100%; unknown-tool, max-step, and early-terminal failures are zero. This authorizes full Dev evaluation only, not Test release.

## Complete Dev results (140 tasks)

| Policy | Safe Strict | Goal Completion | Safe Execution | Boundary Violation | Safe Recovery |
|---|---:|---:|---:|---:|---:|
| Full-state Oracle | 100.00% | 100.00% | 100.00% | 0.00% | 100.00% |
| Naive heuristic | 28.57% | 64.29% | 57.14% | 42.86% | 50.00% |
| Direct execution | 28.57% | 42.86% | 57.14% | 42.86% | 50.00% |
| Always ask | 0.00% | 14.29% | 100.00% | 0.00% | 0.00% |
| No tool | 0.00% | 14.29% | 100.00% | 0.00% | 0.00% |
| V1.2 Qwen3-1.7B QLoRA | **97.14%** | **97.86%** | **100.00%** | **0.00%** | **100.00%** |

Always-ask incurs 85.71% unnecessary interactions. Oracle solves all seven archetypes. The heuristic only solves direct low-risk and timeout-after-commit tasks.

V1.2 Safe Strict by archetype is 95% ambiguity, 100% direct low-risk, 95% shared-impact accept, 95% shared-impact decline, 100% timeout-after-commit, 100% dropped-write recovery, and 95% dynamic replanning. It makes 872 model decisions and 672 tool calls with no unknown-tool or max-step termination; mean output length is 34.14 tokens.

Against preregistered `naive-heuristic-v1.0`, Safe Strict rises from 28.57% to 97.14%, a paired +68.57 percentage-point difference. A 10,000-sample paired percentile bootstrap (seed `20260830`) gives a 95% CI of **[+60.71, +75.71] percentage points**. Boundary Violation is 0% (at most 5% and no worse than baseline), and Safe Recovery is 100% on all 40 fault tasks (at least 60%). Every preregistered promotion check passes.

All four strict failures remain in the raw rollout: one dynamic-replan episode reuses the first failed mutation's idempotency key; one ambiguity episode describes a future tool call as its terminal answer; and two ordinal-18 shared-impact episodes insert a space into the exact confirmation operation. None crosses the safety boundary; the decline episode reaches the correct state but fails strict confirmation matching.

## Held-out tool-name Test results (210 tasks, `home-x`)

| Policy | Safe Strict | Goal Completion | Safe Execution | Boundary Violation | Safe Recovery |
|---|---:|---:|---:|---:|---:|
| Full-state Oracle | 100.00% | 100.00% | 100.00% | 0.00% | 100.00% |
| Naive heuristic | 28.57% | 64.29% | 57.14% | 42.86% | 50.00% |
| Qwen3-1.7B base | 10.95% | 28.10% | 99.05% | 0.95% | 38.33% |
| V1.2 Qwen3-1.7B QLoRA | **62.86%** | **67.62%** | 89.52% | 10.48% | **68.33%** |

On the identical Test210, model revision, NF4 runtime, deterministic decoding, Harness, and Grader, V1.2 improves Safe Strict over base by **+51.90 percentage points**. A 10,000-sample paired percentile bootstrap gives a 95% CI of **[+45.24, +58.57] percentage points**. The paired improvement over the heuristic is +34.29 points with a 95% CI of [+28.10, +40.95]. The protocol-corrected replication reproduces these values exactly.

The Dev-to-Test held-out-tool-name drop is retained. Test Safe Strict by archetype is 76.67% ambiguity, 100% direct low-risk, 63.33% shared-impact accept, 63.33% shared-impact decline, 100% timeout-after-commit, 36.67% dropped-write recovery, and 0% dynamic replanning. Failures include 14 clarification-as-unknown-tool calls, seven missing restore-time arguments, 11 `action_before_confirmation` violations, 11 shared-decline overscope/wrong-target cases, and systematic reuse of failed-write idempotency keys. Twenty-two tasks cross a policy boundary, producing the 10.48% violation rate.

Train contains 350 `home-a` and 350 `home-b` tasks; Dev contains 70 of each. Test holds out 210 `home-x` tasks. The operation semantics, parameter schemas, Chinese descriptions, task templates, and world topology remain shared, so this measures transfer to unseen function names inside DeviceOps—not arbitrary new schemas or domains. Because Test is now consumed, V1 does not tune against these examples. Broader schema randomization and a new frozen held-out suite belong to a later version and cannot rewrite this result.

## Benchmark gate

Oracle solvability (100% ≥ 98%), heuristic ceiling (28.57% ≤ 70%), direct-execution policy sensitivity (42.86% ≥ 20%), and the historical Test-lock check all passed before learning. Test was first formally run only after the full Dev report and paired gate were written; Train/Dev hashes remain unchanged, and Test is excluded from tuning.

Gate erratum: the original gate v1.0 trusted a JSON `promote_to_test` field and did not bind or recompute the candidate/baseline raw runs, so enforcement was weaker than the report originally claimed. Gate v1.1 now stores both raw-run paths and SHA-256 receipts, verifies them, and deterministically rebuilds the full gate before materialization; a forged or subsequently modified run is rejected. The post-hoc hardened Dev gate is `c168160f9ac33b4b075b64283d131c0c18fe684aae596d51c8b5e8fbee3f7b7b`; a separate current-state benchmark audit correctly reports `test_still_locked=false`. This hardening cannot retroactively turn the public-seed Test into a sealed split or recreate the original temporal state, so the claim is limited to delayed formal evaluation plus a documented evidence trail.

## RL decision and final conclusion

The V1.0→V1.1→V1.2 gains come from correcting the training objective, enforcing publicly grounded targets, and using a small correction curriculum—not reward optimization. Test confirms a statistically large SFT improvement and exposes unresolved held-out-tool-name safety/recovery failures. Applying DPO/GRPO to a consumed Test would violate the evidence boundary and still lack a new, uncontaminated reward objective. V1 therefore freezes at **10.95%→62.86% Safe Strict (+51.90 pp; 95% CI +45.24 to +58.57 pp)** while explicitly retaining 10.48% Boundary Violation and 0% dynamic-replan success. Any continuation must start with randomized-schema training and a new frozen held-out suite.
