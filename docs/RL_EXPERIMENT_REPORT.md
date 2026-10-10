English | [简体中文](RL_EXPERIMENT_REPORT.zh-CN.md)

# DeviceAgent Arena: RTX 4060 Online Agent-RL Experiment

Status: exploratory experiment completed; no candidate promoted

Versions: RL task suite v0.3 / reward v0.3 / evidence v0.2

Hardware: NVIDIA GeForce RTX 4060 Laptop GPU, 8,188 MiB VRAM

## 1. Result

The RTX 4060 completed both Qwen3-1.7B 4-bit QLoRA and multi-turn environment
GRPO. Task-SFT peaked at 3,507 MiB allocated memory; the warm and direct GRPO
smokes peaked at 6,378.5 and 6,806.5 MiB sampled device usage, both below the
pre-registered 7,700 MiB guard.

Training feasibility did not imply an RL gain. On the frozen, structurally held-out
Dev8 smoke:

| Route | Before GRPO | After GRPO | Change |
|---|---:|---:|---:|
| task-SFT warm start | 4/8 (50.0%) | 3/8 (37.5%) | -12.5 pp |
| generic Tool-SFT, without v0.3 task imitation | 3/8 (37.5%) | 3/8 (37.5%) | 0.0 pp |

Neither route met the 90% non-zero reward-variance and gradient gates, so neither
was scaled to Dev32 and Test64 was not touched. The supported conclusion is that
the GPU is not the limiting factor; exploration, all-failure groups, and trajectory
truncation are.

## 2. Stateful environment, not a one-shot classifier

Each episode repeatedly selects an action, executes it in the same stateful tool
environment used by evaluation, consumes the resulting observation or error, and
recovers until termination. Hidden device state is available only through tool
observations. Strict success is recomputed from final environment state and required
trajectory events, never from the agent's claim.

Suite v0.3 freezes 128 train, 32 dev, and 64 test tasks across `reconfigure`,
`configure_save`, and `recall_preset` workflows; four valid initial-state profiles;
explicit and inspect-first information modes; clarification; seeded transient faults;
and 3–7 required actions. Complete structural combinations are disjoint across
splits, while every held-out atomic factor occurs in train. Oracle solves all 224
tasks; a legacy fixed macro solves only 5/16 of a Dev control slice.

## 3. SFT is a baseline, not an RL prerequisite

Task SFT teaches the interaction protocol, argument copying, clarification, and legal
termination. It is treated as an initialization and baseline. Because
[ToolRL](https://arxiv.org/html/2504.13958v1) reports cases where direct GRPO
generalizes better than task-SFT initialization, the experiment freezes a
compute-matched comparison:

- task-SFT v0.4 → warm GRPO;
- generic clean-v0.6 Tool-SFT → direct GRPO, without v0.3 trajectory imitation.

The GRPO configurations differ only in the initial adapter: Qwen3-1.7B, NF4 4-bit,
LoRA r16, Dr. GRPO, four generations, four optimizer updates, seed 42, the same 12
training tasks, ten environment steps, and 320 completion tokens. “Direct” here
means no v0.3 task SFT, not an entirely un-post-trained foundation model.

## 4. Measured training evidence

Task-SFT v0.4 contains 719 unique Oracle decisions. Twenty-three clarification
decisions are each repeated three additional times, yielding 788 stored records; the
manifest explicitly distinguishes unique from oversampled supervision. It completed
50/50 updates in 681.11 seconds, peaked at 3,507.22 MiB allocated VRAM, and produced
adapter SHA-256 `610793194d17...1b7f294`.

| Metric | Warm GRPO | Direct GRPO |
|---|---:|---:|
| optimizer updates | 4/4 | 4/4 |
| runtime | 113.68 s | 113.20 s |
| updates with non-zero reward std | 2/4 (50%) | 2/4 (50%) |
| updates with non-zero gradient | 2/4 (50%) | 1/4 (25%) |
| mean completion clipped ratio | 43.75% | 50.00% |
| peak allocated VRAM | 4,668.55 MiB | 4,717.33 MiB |
| peak sampled device usage | 6,378.5 MiB | 6,806.5 MiB |
| 7,700 MiB memory gate | passed | passed |
| learning-signal gate | **rejected** | **rejected** |
| Dev8 promotion gate | **rejected** | **rejected** |

Group-relative advantage vanishes when every rollout for a prompt receives the same
failure return. Heavy length clipping further removes or corrupts sparse terminal
signals. These are the mechanism-level explanations for the failed smokes.

## 5. Reward and anti-gaming controls

Reward v0.3 uses a hard terminal separation: success is `+1`, failure is `-1`, and
all process shaping is capped within `±0.1`. Missing required clarification, wrong
final state, forbidden side effects, repeated failures, or budget exhaustion cannot
be accumulated into a positive return. This repairs an earlier v0.7 reward-gaming
failure, whose historical artifacts remain auditable.

`verify-rl-evidence` binds config, data, dependency lock, adapters, source identity,
the frozen Dev selection, and raw runs by hash. It regrades every trajectory from
`final_state/events`, regenerates reports and comparisons, and separately reports
canonical-integrity, learning-signal, and promotion gates. Both v1.0 smokes are
artifact-verified and canonical-source-clean but rejected by the latter two gates.

## 6. v1.1 short-horizon, 640-token replication

After v1.0 was rejected, `train-short-v0.1` froze eight tasks using executed Oracle
trajectories. The set spans all three workflows, explicit/inspect modes,
clarification, and triggered faults; every task is solvable within five agent
decisions. Seed, reward, LoRA, sampling, and the ten-step environment budget remained
fixed. Only the completion budget changed from 320 to 640, with eight updates—one
round over the eight frozen prompts. Training also persisted 32 complete rollouts per
route with task seed, state/events, reward breakdown, and source hashes.

| Metric | Warm short640 | Direct short640 |
|---|---:|---:|
| optimizer updates | 8/8 | 8/8 |
| runtime | 491.12 s | 498.99 s |
| strict success in training rollouts | 2/32 (6.25%) | 11/32 (34.375%) |
| normal terminal-tool completion | 32/32 | 32/32 |
| non-zero reward-std / gradient updates | 6/8 (75%) | 2/8 (25%) |
| mean TRL completion-clipped ratio | 53.125% | 65.625% |
| peak allocated VRAM | 4,908.00 MiB | 5,212.93 MiB |
| peak sampled device usage | 7,322.5 MiB | 7,682.5 MiB |
| 7,700 MiB memory gate | passed | passed (17.5 MiB margin) |
| 90% learning-signal gate | **rejected** | **rejected** |
| Dev8 | not run | not run |

This corrects the earlier diagnosis: under 320 tokens the policy did not reach the
terminal tool, whereas all 64 v1.1 samples did. The ten-step environment limit was
not the primary blocker. Token-level truncation masks remain high. More importantly,
initialization creates opposite group degeneracies: warm rollouts are mostly
all-failure, while direct has prompts whose four samples are all successful. Both
produce zero group variance. Direct's higher training-rollout success is therefore
an explorability diagnostic, not evidence of better held-out performance.

Neither candidate passed the pre-registered signal gate, so Dev8, Dev32, and Test64
were not run. At the v1.1 decision point, the next defensible experiment was
difficulty-matched dynamic sampling:
prefer prompts whose recent group success is neither 0% nor 100%, with bounded
resampling of equal-return groups and explicit sampling-cost/distribution logs.
Direct short640 left only 17.5 MiB under the guard, so context or group size should
not increase. This was not implemented at v1.1; v1.4 later implemented the
mixed-first bounded-resampling part, reported in the next section, and still failed
the gate.

The repository contains a tested turn-credit research seam, but it is **not wired
into the GRPO loss used here**. No turn-level RL result is claimed.

## 7. v1.2–v1.4 termination semantics and bounded resampling

Independent review showed that v1.1's TRL `clipped_ratio` was not an environment
length-truncation rate. All 64 trajectories had called `finish_task`, but TRL checked
only for a final EOS/PAD token and ignored `done=true` in the tool response. v1.2
restored protocol-terminal rows in the loss mask, then exposed a second issue: the
tool loop kept generating repeated terminal calls. It is retained as a diagnostic,
not an algorithm result.

v1.3 detects the terminal tool response at the post-tool generation boundary,
drops that already-generated invalid assistant turn, and prevents further repeated
tool-loop turns; it does not eliminate this one probe generation. All 32 rollouts
terminated at the protocol layer, none was genuinely length-truncated, and runtime
fell from 538.17 s in v1.2 to 289.49 s (-46.2%). Non-zero gradients remained 6/8,
isolating group variance as the next bottleneck.

v1.4 freezes bounded mixed-first resampling: at most three different-task groups per
update; accept a success/failure-mixed group first, otherwise the first group with
reward std≥0.05. Rejected groups remain in the audit and count against the budget.

| Metric | Warm v1.3 | Warm v1.4 bounded resampling |
|---|---:|---:|
| optimizer updates | 8 | 8 |
| generated rollouts / groups | 32 / 8 | 84 / 21 (cap 96 / 24) |
| accepted mixed / shaping / zero-variance | 3 / 0 / 5¹ | 4 / 3 / 1 |
| accepted non-zero reward-std / gradients | 6/8 (75%)² | 7/8 (87.5%) |
| genuine length truncations | 0/32 | 0/84 |
| runtime | 289.49 s | 678.71 s |
| peak device usage | 7,436.5 MiB | 7,630.5 MiB |
| 90% signal gate | rejected | **rejected** |
| Dev8 / Test64 | not run | not run |

¹ Using v1.4's population-std≥0.05 shaping threshold, v1.3's three non-mixed
non-zero-gradient groups fall in the zero/low-variance bucket. ² The exact
`reward_std>0` gradient count remains 6/8.

All four successes among 84 proposals were selected into accepted groups (4/32;
rejected 0/52). The v1.4 7/8 is therefore a result-conditioned training-group
utilization rate, not directly comparable to v1.3's natural 6/8 and not evidence of
policy or success-rate improvement. Runtime grew 2.34× for one additional non-zero
accepted update and one additional mixed group.

The locked TRL 1.9.2 native `clipped_ratio` remains near 100% because it is a
last-token EOS/PAD heuristic, not an environment truncation rate; evidence renames it
`mean_token_no_eos_ratio`. Audit shows 0/84 genuine length truncations and 0/84 loss
masks. Historical v1.4 loss/reward/reward_std align to accepted groups, but auxiliary
completion/tool trainer means mix rejected proposals. The verifier excludes those
means from conclusions and recomputes proposal/accepted/rejected costs from audit.
The trainer now isolates accepted per-group auxiliary means for future runs, while
cumulative `num_tokens` still reports the cost of all proposals. v1.4 is not rerun
to chase a number. It still misses the gate, which is not relaxed post hoc.

## 8. Limits

- Dev8 is an exploratory gate: one task equals 12.5 pp.
- Each route uses one training seed and four updates; there is no significance or
  algorithm-superiority claim.
- The environment is synthetic, not physical hardware or acoustic ASR.
- Test64 was not run; no RL generalization gain belongs in a resume yet.
- Defensible claims are the reproducible multi-turn RL system, structural holdout,
  compute-matched negative control, reward-gaming audit, and automatic stop gates on
  an 8 GB consumer GPU.
