# ContractGraph Harness + FunctionGemma Verifier Experiment Report

Status: **the experiment loop is complete; the ContractGraph harness passed engineering validation; the learned verifier failed its Dev gates; Test-XSchema was never generated or revealed.**

Chinese version: [CONTRACTGRAPH_EXPERIMENT_REPORT.zh-CN.md](CONTRACTGRAPH_EXPERIMENT_REPORT.zh-CN.md)

## Bottom line

The evidence does not support the claim that fine-tuning a 270M model produces a reliable planner for arbitrary unseen tools. It supports a narrower and more useful conclusion: for tool-use tasks with explicit preconditions, effects, retry rules, and invariants, the **deterministic ContractGraph runtime is the reliability layer; the learned verifier is an experimental ranker that did not earn promotion.**

- On synthetic cross-schema Dev, a scripted baseline solved **16/64 (25%)**, while ContractGraph solved **64/64 (100%)** with 100% argument validity, zero contract violations, and 100% recovery. This is an engineering regression under public Contract IR, not a learned generalization score.
- FunctionGemma-270M v0.8 reached **66.0% Top-1** on the fully held-out Dev-XSchema family, versus 18.5% no-ranker and 25.0% generic-rule baselines. However, invalid-argument negative accuracy was only 60.0% and wrong-or-redundant-repair accuracy only 41.33%, below the preregistered 70% per-slice gate.
- The raw verifier completed **74/200 (37.0%)** multi-step rollouts. Its +12pp paired gain over the 25.0% generic baseline had a 95% bootstrap CI of **[+6.5pp, +17.5pp]**, but its **41.95% contract-violation rate** and **15/40 (37.5%)** recovery over all fault-planned tasks failed their gates.
- A post-hoc, explicitly non-promotional diagnostic applied the same ContractGraph feasibility mask to all rankers: generic rule **100%**, learned verifier **90%**, no-ranker **22.5%**. Learned versus generic was **-10pp, 95% CI [-14.5pp, -6.0pp]**. In this explicit-contract domain, the learned component adds no value.

The candidate was therefore not frozen and sealed Test-XSchema was not generated. Keeping Test sealed is an experimental-control outcome, not a missing best-case score.

## What was built

The runtime path is:

`visible request → candidate action → JSON Schema validation → ContractGraph feasibility/repair → tool execution → postcondition verification → bounded recovery`

The implementation provides:

1. A Contract IR bound to tool-schema SHA-256 digests, with roles, preconditions, effects, observable facts, retry policy, and ambiguous-commit handling. Unknown contracts, digest mismatches, and repair cycles fail closed.
2. A stateful execution engine that accepts only an allowlisted `AgentContext`; hidden `TaskSpec`, goals, fault plans, and grader fields are absent. Episode isolation, batch handling, structured failures, and postconditions are tested.
3. Hash-bound Train/Dev data, model revision, configuration, adapters, and the records/score-report/catalog inputs used by rollouts. Source-freeze tooling exists, but this candidate failed Dev and therefore received neither a source-freeze receipt nor a one-time two-domain sealed Test-XSchema suite.
4. A FunctionGemma-270M NF4 QLoRA sequence-classification/listwise ranking pipeline, evaluated on a shared candidate pool against no-ranker, schema-only generic-rule, and ContractGraph-oracle policies.

## Data and hardware receipt

v0.8 contains 10,000 candidate records: 8,000 Train, 1,000 Dev-ID, and 1,000 Dev-XSchema. Each state group contains five candidates and exactly one positive. Train contains six tool/argument/fact namespace variants across meeting-room and studio domains; `aliased-node-v1` is held out in full for Dev-XSchema. Schema-valid negatives account for 87.475% of all negatives, 50.94% of records belong to counterfactual groups whose label flips after a state change, and hidden runtime fields, duplicate inputs, and duplicate IDs are all zero.

- Dataset SHA-256: `90094aab62cb569c58cd33fa1da8cda3bd2ffcc50bbcbda56eb2e2b8244977de`
- Model: `google/functiongemma-270m-it`
- Model revision: `39eccb091651513a5dfb56892d3714c1b5b8276c`
- RTX 4060 Laptop full run: 1 epoch / 100 updates / 1,230.08 seconds
- Peak allocated / reserved: 1,902.322 / 2,240 MiB; zero token truncation
- Config SHA-256: `2a623a3e0a485e777ee6b59372735bfae14a75da849afa6855168c4e441a6dbd`

`train_loss=5.6729` and the 94.53% train-subset diagnostic accuracy only confirm that training ran; neither is treated as agent capability.

## Static cross-schema results

All rows use the 1,000-record Dev-XSchema set with 200 five-way state groups. Test is absent from selection.

| Version | Main change | Top-1 | Pairwise | Balanced | Positive recall | Invalid negative | Wrong/redundant repair | Decision |
|---|---|---:|---:|---:|---:|---:|---:|---|
| v0.3 | binary verifier | 63.5% | 90.88% | 73.69% | 57.0% | 100% | 98.89% | reject |
| v0.5 | second train domain + tool-order shuffle | 66.5% | 91.63% | 61.44% | 27.5% | 50.0% | 100% | reject |
| v0.6 | listwise + auxiliary BCE | 60.5% | 90.13% | 71.06% | 59.0% | 72.0% | 51.33% | reject |
| v0.7 | public Contract IR input | 59.0% | 79.50% | 70.56% | 62.5% | 86.0% | 70.67% | reject |
| v0.8 | tool/argument/fact metamorphic aliases | **66.0%** | **90.06%** | **83.75%** | **91.5%** | 60.0% | 41.33% | reject |

v0.8 reached 96.0% Top-1 on Dev-ID but 66.0% on Dev-XSchema. The 30pp gap is why in-distribution accuracy is not used as a cross-schema claim.

## Multi-step rollouts

Rollouts start from 200 Dev-XSchema state snapshots and run for at most six steps. Forty tasks are scheduled for one transient failure. A policy that fails before reaching a mutate action does not physically trigger that fault, but the task remains in the primary recovery denominator; this prevents a policy from inflating recovery by avoiding the injection point. Conditional recovery over actually injected tasks is retained separately.

| Version | Learned strict success | Generic | No-ranker | Oracle | Learned violation rate | Primary recovery | Gate |
|---|---:|---:|---:|---:|---:|---:|---|
| v0.5 | 38.5% | 25.0% | 3.5% | 100% | 72.66% | 40.0% | fail |
| v0.6 | **44.5%** | 25.0% | 3.5% | 100% | 54.74% | 40.0% | fail |
| v0.8 | 37.0% | 25.0% | 3.5% | 100% | 41.95% | 37.5% | fail |

v0.8 succeeds on 90% of ready states but only 12% of unknown and unbound states and 34% of bound-but-untuned states. Traces show that the model often chooses the first step correctly, then repeats `bind_node` or `tune_node` after the state has already changed until the six-step budget is exhausted. Better static positive recall therefore did not translate into better closed-loop behavior.

## Negative-result ledger

| Finding | Detection | Resolution and claim boundary |
|---|---|---|
| Old Test64 `required_tools` were read by parser coverage tests | evidence-boundary audit | downgraded to a consumed regression set; never called held-out again |
| Windows CRLF changed the dataset hash | preflight hash verification | forced LF output and regenerated the manifest; old receipt retained |
| v0.3.1 placed every positive first | position-distribution audit | dataset retired; candidate order randomized and audited |
| fixed tool order created a lexical shortcut | cross-alias error review | shuffled tool order from v0.5; old scores excluded from promotion |
| v0.5 positive recall was only 27.5% | per-slice static metrics | moved to listwise; v0.6 still failed |
| public Contract IR did not automatically generalize | v0.7 Dev-XSchema | added tool/argument/fact aliases; v0.8 still failed |
| 66% static Top-1 did not improve rollouts | v0.8 rollout | isolated repeated-repair errors after state changes |
| recovery originally counted only tasks that reached the injection point | independent review found learned triggered only 15/40 planned faults | primary denominator changed to all fault-planned tasks; conditional recovery retained; v0.5/v0.6/v0.8 rerun |
| masked learned ranker trailed rules by 10pp | fair post-hoc same-mask diagnostic | learned verifier judged unnecessary for this domain; not promotion evidence |

## Why DPO/GRPO was not continued here

The bottleneck is not missing preference optimization. Public Contract IR already lets a generic deterministic policy reach 100%, while the learned ranker reaches 90% under the same mask. Applying DPO or GRPO to a component dominated by a simpler baseline would add training complexity without product value. Separate project tracks retain real DPO/GRPO OOM, zero-variance, and rejected-candidate evidence; this track does not train again merely to add technique names.

A future learned-planner version should first introduce tasks where contracts cannot decide semantic conflicts, cost trade-offs among multiple valid paths, or incomplete contracts, then preregister a new gate and benchmark version. It must not generate Test after seeing this Dev result.

## Reproducible evidence

- v0.8 data manifest: [`data/train/manifest_contract_verifier_v0.8.json`](../data/train/manifest_contract_verifier_v0.8.json)
- Full training receipt: [`artifacts/training/functiongemma-270m-contract-verifier-listwise-v0.8.1.json`](../artifacts/training/functiongemma-270m-contract-verifier-listwise-v0.8.1.json)
- Static Dev report: [`artifacts/reports/functiongemma-contract-verifier-listwise-v0.8.1-dev.json`](../artifacts/reports/functiongemma-contract-verifier-listwise-v0.8.1-dev.json)
- Rollout and post-hoc diagnostic: [`artifacts/reports/functiongemma-contract-verifier-listwise-v0.8.1-dev-rollouts.json`](../artifacts/reports/functiongemma-contract-verifier-listwise-v0.8.1-dev-rollouts.json)
- Preregistered plan: [`docs/CONTRACTGRAPH_PROJECT_PLAN.md`](CONTRACTGRAPH_PROJECT_PLAN.md)

## Resume boundary

Safe claims: built the ContractGraph harness; achieved 64/64 synthetic cross-schema Dev with zero contract violations and zero hidden runtime fields; trained and audited a 10k-record FunctionGemma verifier on an RTX 4060; diagnosed and rejected the learned component when a 100% deterministic baseline dominated it.

Unsafe claims: learned verifier achieved 100%; sealed Test passed; results transfer to physical devices or production traffic; SFT/RL improved reliability; zero-shot generalization to arbitrary new tools. In a short resume, lead with the ContractGraph runtime and auditable evaluation; use the rejected verifier as an interview-depth negative experiment.
