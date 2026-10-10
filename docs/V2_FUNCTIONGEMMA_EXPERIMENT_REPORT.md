# DeviceAgent Arena V2: FunctionGemma Experiment Report

## Conclusion

FunctionGemma-270M was trained and evaluated on unseen Dev tasks; this is not a planned or simulated result.

- On unseen Dev tasks under shifted-schema conditions, schema-randomized SFT raised schema-a/schema-b from `0/7` to `5/7`. The transformation rules themselves were present in training, so this is not claimed as a wholly unseen-schema result.
- Strict success on the original schema remained `14/21 = 66.67%`.
- The model still failed to reliably execute the full shared-impact chain under the original schema: bind an exact confirmation, pause the routine, then mutate the device.
- v2.3 used thresholds frozen before candidate evaluation; v2.4.1 was retrospectively checked against the same thresholds. Both failed that requirement, so the executable gate blocks Dev210 and Test remains unmaterialized.
- The evidence supports using a 270M model for routine tool use and schema adaptation, but not as a standalone planner for precise multi-entity safety workflows.

## V1 versus V2

V1 is the first frozen DeviceAgent Arena suite, Qwen3-1.7B baseline, and Test210 result. That Test has already been consumed and is not reused for tuning.

V2 continues the same project while fixing protocol, parsing, idempotency, and shared-impact seams. It uses unseen Dev data and a locked Test. The completed V2 scope here is the schema-shift/distractor core slice; the Test seed remains absent.

## Infrastructure finding

The first v2.2 Dev21 run scored `0/21`, but raw trajectories showed valid nested calls such as:

```text
apply_change(action=set_mode, payload={mode=cinema}, target_id=dv-bed-0-a)
```

The old FunctionGemma parser only accepted flat arguments. It discarded the nested payload and converted the turn into an early finish. After adding recursive nested-object/list parsing, the same checkpoint on the same 21 tasks recovered to `14/21`. The pre-fix zero is therefore an infrastructure negative result, not a model score.

## Data and training

| Stage | Records | Training | Train loss | Peak VRAM | Purpose |
|---|---:|---:|---:|---:|---|
| v2.2 Stage-1 | 4,400 | 1 epoch / 275 steps / 25m14s | 0.12306 | 3,380 MB | Native multi-turn SFT with leaked IDs and incorrect shared labels removed |
| v2.3 Stage-2 | 5,800 | 0.502 epoch / 182 steps / 24m40s | 0.01553 | 3,394 MB | identity/schema-a/schema-b randomization plus 1,400 shared hard cases |
| v2.4.1 Stage-3 | 2,277 | 0.506 epoch / 72 steps / 7m39s | 0.00320 | 3,397 MB | 1,000 critical states, 702 counterfactual role-binding examples, and 575 replay examples |

All stages used full-parameter BF16 training on an 8 GB RTX 4060 Laptop GPU. v2.3 and v2.4.1 passed a 2,304-token audit with zero truncation and verified completion-only prompt prefixes.

## Paired Dev21 results

All rows use the same 21 unseen tasks, with three tasks per archetype. The mixed condition contains seven tasks each for identity, schema-a, and schema-b.

| Checkpoint / condition | Strict | identity | schema-a | schema-b | Shared accept | Shared decline |
|---|---:|---:|---:|---:|---:|---:|
| v2.2 identity, parser fixed | 14/21 | 14/21 | — | — | 0/3 | 0/3 |
| v2.2 mixed | 4/21 | 4/7 | 0/7 | 0/7 | 0/3 | 0/3 |
| v2.3 identity | 14/21 | 14/21 | — | — | 0/3 | 0/3 |
| v2.3 mixed | 14/21 | 4/7 | 5/7 | 5/7 | 0/3 | 2/3 |
| v2.4.1 identity | 14/21 | 14/21 | — | — | 0/3 | 0/3 |
| v2.4.1 mixed | 14/21 | 4/7 | 6/7 | 4/7 | 1/3 | 2/3 |

v2.2 model-only fault recovery was 100% in the **identity condition**; it was only `2/6 = 33.33%` under mixed schema. Enabling the SharedImpactGuard kept identity strict success at `14/21` while raising safe execution from `66.67%` to `95.24%`. A public-state safety middleware can suppress unsafe behavior, but it does not replace correct model planning.

## Promotion gate

The gate was frozen before candidate evaluation. It required all of the following:

- identity at least `14/21`;
- mixed at least `9/21`;
- schema-a and schema-b at least `2/7` each;
- at most four mixed boundary violations;
- at least one identity shared-impact success.

Both v2.3 and v2.4.1 passed the first five numerical/schema checks but scored `0/6` on identity shared-impact tasks. Both receipts are `FAIL`. The historical v2 receipts bind candidate paths, identity/mixed conditions, 21 task IDs, suite, and report-file hashes, but their raw headers predate checkpoint-tree attestation. They are weak historical evidence and cannot unlock new evaluation. The current v4 gate additionally requires raw JSONL, checkpoint-tree, protocol/decoder, and per-decision raw-output bindings; the next candidate must be rerun by the hardened runner. v2.3's numerical thresholds were frozen before candidate evaluation. v2.4.1 is a retrospective check against those thresholds, not an independent preregistration.

## Why RL is not the next step

The remaining errors already have explicit supervised next-action labels. Moving directly to GRPO or DPO would add reward variance and reward-hacking risk before the model exhibits stable non-zero success on the target chain. A 21-task gate is also too small to justify online optimization.

RL should only be reconsidered after:

1. a larger model or structured decoder produces stable non-zero identity shared success;
2. rollouts contain auditable success/failure branches and non-zero reward-gradient rates;
3. gains reproduce on a larger Dev set rather than one schema or ID pattern; and
4. model-only and guarded conditions remain separately reported.

## Current decision

- The V1 Qwen Test210 result remains the only official Test result.
- v2.3 is the best-balanced schema-generalization research checkpoint, but is not promoted to Dev210/Test.
- v2.4.1 is a useful negative result: confirmation behavior and mixed shared tasks improved, but retrospective evaluation against the previously frozen thresholds still failed.
- The next meaningful comparison is a larger FunctionGemma/Qwen-class model, constrained decoding, or an explicit entity-role state machine—not more epochs of the same SFT recipe.

## Evidence index

- v2.2 training: `artifacts/checkpoints/functiongemma-270m-deviceops-v2.2-full-sft-stage1/device_agent_arena_training_result.json`
- v2.3 training: `artifacts/checkpoints/functiongemma-270m-deviceops-v2.3-full-sft-stage2/device_agent_arena_training_result.json`
- v2.4.1 training: `artifacts/checkpoints/functiongemma-270m-deviceops-v2.4.1-full-sft-stage3/device_agent_arena_training_result.json`
- v2.3 historical weak receipt: `artifacts/v2/training/functiongemma-270m-v2.3-dev21-promotion-receipt-v2.json`
- v2.4.1 historical weak receipt: `artifacts/v2/training/functiongemma-270m-v2.4.1-dev21-retrospective-receipt-v2.json`
- V2 plan: `docs/V2_PROJECT_PLAN.md`
