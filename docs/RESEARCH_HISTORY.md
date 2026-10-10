# Research history

The public project is organized around the V2.7 Compound Agent result. Earlier lines
remain useful because they show which hypotheses failed and how the final architecture
was selected; they are not mixed into the headline comparison.

## V1: stateful Planner post-training

The first complete benchmark established the Harness, deterministic environment,
next-decision QLoRA, held-out tool naming, and trajectory-aware grading. It also
exposed a large Dev-to-Test gap, nonzero boundary violations, and a dynamic-replanning
failure rather than hiding them.

- [V1 experiment report](V1_EXPERIMENT_REPORT.md)
- [V1 中文实验报告](V1_EXPERIMENT_REPORT.zh-CN.md)

## FunctionGemma: small-model boundary

Multiple full-parameter SFT stages made FunctionGemma's native tool-call format
stable but did not make the 270M model a reliable long-horizon Planner. This negative
result motivated using it only where public state makes the next action exactly
verifiable.

- [FunctionGemma experiment report](V2_FUNCTIONGEMMA_EXPERIMENT_REPORT.md)
- [FunctionGemma 中文实验报告](V2_FUNCTIONGEMMA_EXPERIMENT_REPORT.zh-CN.md)
- [Composition experiment report](V2_COMPOSITION_EXPERIMENT_REPORT.md)
- [Composition 中文实验报告](V2_COMPOSITION_EXPERIMENT_REPORT.zh-CN.md)

## Online RL and preference optimization

GRPO and sequential preference experiments were implemented and measured on the same
consumer GPU. They demonstrated feasibility but did not establish a held-out gain,
so V2.7 does not claim an RL improvement.

- [RL experiment report](RL_EXPERIMENT_REPORT.md)
- [RL 中文实验报告](RL_EXPERIMENT_REPORT.zh-CN.md)

## ContractGraph

The contract runtime achieved deterministic engineering gains, while a learned
listwise verifier failed its promotion gate. The learned candidate was rejected and
the sealed test remained ungenerated.

- [ContractGraph report](CONTRACTGRAPH_EXPERIMENT_REPORT.md)
- [ContractGraph 中文报告](CONTRACTGRAPH_EXPERIMENT_REPORT.zh-CN.md)

These records are research history, not additional evidence for the V2.7 headline.
