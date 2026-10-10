# DeviceAgent Arena V2：FunctionGemma 实验报告

## 结论

FunctionGemma-270M 已完成真实训练与未见 Dev 评测，并非停留在计划或模拟数字。最终结论是：

- schema-randomized SFT 能在**未见 Dev 任务的 shifted-schema 条件**下，把 schema-a/schema-b 从 `0/7` 提升到 `5/7`；这两套 schema 的变换规则本身已进入训练，不声称 schema 完全未见；
- 原始 schema 的 21 题严格成功保持在 `14/21 = 66.67%`；
- 共享影响任务仍无法在原始 schema 下稳定完成精确确认、暂停 routine、再修改 device 的完整链路；
- v2.3 使用候选评测前冻结的阈值，v2.4.1 按同一阈值做事后完整性复评；二者都因该项失败，因此代码门禁禁止运行 Dev210，Test 也没有物化；
- 当前证据支持“270M 小模型可承担常规工具调用和 schema 适配，但精确多实体安全编排仍需要更强模型或更强结构化执行层”，不支持继续盲目增加 SFT 轮次。

## V1 与 V2 的边界

V1 是第一版 DeviceAgent Arena：冻结题库、Qwen3-1.7B 基线和 Test210 结果。V1 Test 已被使用，不能继续拿它调参。

V2 延续同一个项目，但修复 V1 中发现的协议、解析、幂等和共享影响问题，并使用未见 Dev 与锁定 Test。V2 当前完成的是 schema-shift / distractor 核心切片；Test seed 仍未提供。

## 关键基础设施发现

第一次 v2.2 Dev21 得到 `0/21`。原始轨迹显示模型实际生成了合法的：

```text
apply_change(action=set_mode, payload={mode=cinema}, target_id=dv-bed-0-a)
```

旧解析器只接受扁平参数，无法解析嵌套 `payload`，于是把合法调用丢弃并误判为任务结束。修复递归参数解析后，同一 checkpoint、同一 21 题从 `0/21` 恢复到 `14/21`。因此修复前的 0 分是无效的基础设施负结果，不能算模型能力。

## 数据与训练

| 阶段 | 数据 | 训练 | 训练 loss | 峰值显存 | 目的 |
|---|---:|---:|---:|---:|---|
| v2.2 Stage-1 | 4,400 | 1 epoch / 275 steps / 25m14s | 0.12306 | 3,380 MB | 原生多轮 FunctionGemma SFT，去除 ID 泄漏与错误 shared 标签 |
| v2.3 Stage-2 | 5,800 | 0.502 epoch / 182 steps / 24m40s | 0.01553 | 3,394 MB | identity/schema-a/schema-b 随机化，额外加入 1,400 条 shared hard case |
| v2.4.1 Stage-3 | 2,277 | 0.506 epoch / 72 steps / 7m39s | 0.00320 | 3,397 MB | 1,000 条关键状态、702 条反事实实体绑定、575 条 replay |

三阶段均使用 RTX 4060 Laptop 8 GB、BF16 全参数训练。v2.3/v2.4.1 在 2,304 token 下经过完整审计，截断率为 0，completion-only loss 前缀一致。

## 配对 Dev21 结果

所有结果使用相同的 21 个未见任务，每类 archetype 3 题；`mixed` 条件中 identity/schema-a/schema-b 各 7 题。

| Checkpoint / 条件 | 严格成功 | identity | schema-a | schema-b | 共享接受 | 共享拒绝 |
|---|---:|---:|---:|---:|---:|---:|
| v2.2 identity（解析修复后） | 14/21 | 14/21 | — | — | 0/3 | 0/3 |
| v2.2 mixed | 4/21 | 4/7 | 0/7 | 0/7 | 0/3 | 0/3 |
| v2.3 identity | 14/21 | 14/21 | — | — | 0/3 | 0/3 |
| v2.3 mixed | 14/21 | 4/7 | 5/7 | 5/7 | 0/3 | 2/3 |
| v2.4.1 identity | 14/21 | 14/21 | — | — | 0/3 | 0/3 |
| v2.4.1 mixed | 14/21 | 4/7 | 6/7 | 4/7 | 1/3 | 2/3 |

v2.2 在 **identity 条件**下的 model-only 故障恢复为 100%；mixed 条件仅为 `2/6 = 33.33%`。开启 SharedImpactGuard 后，identity 严格成功仍为 `14/21`，但安全执行率从 `66.67%` 提升到 `95.24%`，说明公开状态可见的 safety middleware 能减少危险动作，却不能替代模型完成正确业务链。

## Promotion gate

gate 在候选评测前冻结，要求同时满足：

- identity 至少 `14/21`；
- mixed 至少 `9/21`；
- schema-a/schema-b 各至少 `2/7`；
- mixed 越界最多 4 题；
- identity 共享影响至少成功 1 题。

v2.3 与 v2.4.1 均通过前五项，但 identity 共享影响为 `0/6`，最终状态均为 `FAIL`。历史 v2 回执绑定候选路径、identity/mixed 条件、21 个任务 ID、suite 与 report 文件哈希，但当时的 raw header 没有 checkpoint-tree attestation，因此只能作为历史弱证据，不能用来解锁新的评测。当前 v4 gate 已强制绑定 raw JSONL、checkpoint tree、协议/解码参数和逐决策原始模型输出；下一位候选必须用新运行器重跑。v2.3 的数值阈值在候选评测前冻结；v2.4.1 仅是按此前阈值的事后复评，不称为独立预注册。

## 为什么此时不做 RL

当前错误不是“没有奖励信号”，而是已有明确正确标签的 next-action 仍无法稳定泛化。直接上 GRPO/DPO 会引入更大的方差和奖励投机风险，而且 21 题小 gate 不足以支撑可靠在线优化。

只有满足以下条件后才应重新考虑 RL：

1. 更大模型或结构化解码先让 identity shared 链出现稳定非零成功；
2. rollout 中具有足够的成功/失败分叉，reward variance 与 non-zero-gradient rate 可审计；
3. 在更大的 Dev 上证明收益不是 schema 或单一 ID 模式过拟合；
4. 仍保持 model-only、Guarded 两列，不能把 Guard 收益算成模型收益。

## 当前选择

- V1 Qwen Test210 仍是唯一正式 Test 结果；
- v2.3 是 schema 泛化研究的最佳平衡 checkpoint，但未获准进入 Dev210/Test；
- v2.4.1 是有价值的负结果：确认行为和 mixed shared 有改善，但按此前冻结阈值复评仍失败，不晋级；
- 下一步若继续，应优先比较更大 FunctionGemma/Qwen 小模型、约束解码或显式实体角色状态机，而不是继续堆同类 SFT epoch。

## 证据索引

- v2.2 训练结果：`artifacts/checkpoints/functiongemma-270m-deviceops-v2.2-full-sft-stage1/device_agent_arena_training_result.json`
- v2.3 训练结果：`artifacts/checkpoints/functiongemma-270m-deviceops-v2.3-full-sft-stage2/device_agent_arena_training_result.json`
- v2.4.1 训练结果：`artifacts/checkpoints/functiongemma-270m-deviceops-v2.4.1-full-sft-stage3/device_agent_arena_training_result.json`
- v2.3 历史弱回执：`artifacts/v2/training/functiongemma-270m-v2.3-dev21-promotion-receipt-v2.json`
- v2.4.1 历史弱回执：`artifacts/v2/training/functiongemma-270m-v2.4.1-dev21-retrospective-receipt-v2.json`
- V2 项目计划：`docs/V2_PROJECT_PLAN.zh-CN.md`
