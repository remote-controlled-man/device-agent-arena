# DeviceAgent Arena v1 实验报告

English: [V1_EXPERIMENT_REPORT.md](V1_EXPERIMENT_REPORT.md)

## 状态

- Benchmark sanity：完成；
- Train/Dev 冻结：完成；
- Test：使用第三套 held-out 工具命名，在 Dev 晋级门禁后才首次正式运行；210-task 评测与 base 对照均完成；
- V1.0 full-trajectory QLoRA：完成，但真实 rollout 为负结果；
- V1.1 next-decision QLoRA：完成；14-task smoke 为 64.29%，但安全门禁未通过；
- V1.2 public-grounded curriculum：训练与完整 140-task Dev 均完成；Safe Strict 97.14%，Boundary Violation 0%；
- 学习模型 Dev rollout：相对冻结 heuristic 提升 68.57 个百分点，paired bootstrap 95% CI 为 [60.71, 75.71] 个百分点；
- 学习模型 Test rollout：Qwen3-1.7B base 10.95%→V1.2 62.86%，提升 51.90 个百分点，paired bootstrap 95% CI 为 [45.24, 58.57] 个百分点；
- DPO / GRPO：V1.2 已通过晋级门禁，当前没有为追主指标而启动 RL 的必要。

## 数据凭证

| 产物 | 记录数 | SHA-256 |
|---|---:|---|
| Train tasks | 700 | `f7edf25a0cf5e2eed058fce84516e49c6fe2a817e61e8c6cabfd34fcda22d5c8` |
| Dev tasks | 140 | `176b7e2649593d2bcc8cd11c2735110bbbec216d8babcf7ae0720e513c2a565b` |
| V1.0 full-trajectory SFT | 700 | `6a14857a31af474b0c9baec979b836f442c15cfbdff1bece617c5c9db47d4339` |
| V1.1 next-decision SFT | 4,200 | `885ff7ec3f9f7dcdb4ef40d87135f4577db444585ea4bc5fbaab25a01f7b77bb` |
| V1.2 public-grounded curriculum | 2,000 | `775ed56bf939015864ebac7f1c13dd8a4b7611498b23540ea84d34370b849f2c` |
| Test tasks | 210 | `7728379b6f20046b34dc17b9bf6efe20ecc08c92e1ea923995edb8edd7bee04d` |

协议修正后的凭证：V1.2 Test raw run `c75089aac6d2477758b9afc8af90867fd66b3abbcaea6bafb71c14d754f8acb1`；base Test raw run `baeda4165d80689c5b0b8a53d4b67032188e6f3b6d1bcc8f83661eebf5703f0f`；带协议一致性检查的 paired report `2f495c456bc380c97d55c050abe40d77449a838c41ed326a7ea54c577cb70857`。后者的 source commit/tree、Harness、model revision、chat template、NF4、max tokens 与 temperature 检查全部为 true。

协议勘误：首轮正式 Test 的 run header 记录了 `max_consecutive_tool_errors=3`，但当时 V1 Environment 没有真正执行该熔断。实现熔断后，以完全相同的模型、prompt、decoder 和 Test 做了一次协议修正复跑。candidate 与 base 各 210 个 grade 均与首轮逐题完全一致，只有 candidate 7 条、base 29 条的 termination 改为 `tool_error_limit`。因此主指标既来自首轮正式 Test，也被修正后的 Harness 精确复现；本次复跑不包装成新的未见 Test。

两版 SFT 隐藏 marker 泄漏均为 0。V1.1 从 700 条 oracle trajectory 导出 4,200 个逐决策样本，平均每条 6 个、实际随轨迹长度变化；真实 Qwen3 chat-template 审计得到 159,770 个 completion token、4,796,400 个 prompt token、0 个空 completion、0 个 prompt-prefix 错位、0/4,200 截断，证明 `completion_only_loss` 有明确且非空的监督边界。

## V1.0 QLoRA 负结果与根因

第一次 QLoRA 确实在 RTX 4060 上完成：Qwen3-1.7B、2 epochs、176 optimizer steps、峰值显存 3,561 MiB、训练 loss 0.4064、token accuracy 0.9935。但 14 个固定 Dev smoke task 上 Safe Strict 仍为 0%，Goal Completion 14.29%，Boundary Violation 7.14%；平均只调用 0.5 次工具，模型经常提前 `[ANSWER]` / `[REFUSE]`。

这不是“数据拟合得很好但模型太小”。对锁定的 TRL 1.9.2 数据管线检查发现：V1.0 使用 `messages` 整轨迹格式；Qwen3 chat template 不提供 assistant mask，而 `completion_only_loss=True` 只会对 prompt-completion 数据的 `completion_mask` 生效。因此 V1.0 实际让 system、user、tool 和 assistant 全部 token 参与 loss，训练产物中原先的 `training_objective=completion-only next-agent decision` 字段不成立。极低 loss 主要说明模型复述整段模板容易，不能证明策略学习成功。

修复原则是保持 benchmark、grader、Train/Dev task 完全不变，只把学习样本改为 `prompt=[当前公开历史]`、`completion=[唯一下一步 assistant 决策]`。V1.1 从 base model 重新训练，不继承失败 adapter。

## V1.1 真实 rollout 与第二层数据缺陷

V1.1 在 RTX 4060 上完成 4,200 个逐决策样本、525 optimizer steps、1 epoch，训练耗时 6,477 秒，train loss 0.0818。相同 14 个 Dev smoke task 上：

| Policy | Safe Strict | Goal Completion | Safe Execution | Boundary Violation | Safe Recovery |
|---|---:|---:|---:|---:|---:|
| Naive heuristic | 28.57% | 64.29% | 57.14% | 42.86% | 50.00% |
| V1.0 full-trajectory QLoRA | 0.00% | 14.29% | 92.86% | 7.14% | 0.00% |
| V1.1 next-decision QLoRA | **64.29%** | 64.29% | 85.71% | 14.29% | **75.00%** |

V1.1 已证明逐决策监督有效：歧义、低风险、动态重规划、timeout-after-commit 四类均为 100%，平均工具调用从失败版 0.5 次恢复到 5.36 次。但它仍不能晋级：shared-impact 两类为 0%，Confirmation Recall 为 0，Boundary Violation 14.29% 高于 5% 门槛。

逐事件检查发现第二层数据问题：原 oracle 在 shared-impact 的第一步直接使用尚未由任何公开观察暴露的设备 ID；写操作的幂等键还包含隐藏 task seed。模型因此会背出 Train 的 `tr-*` ID，收到 not-found 后才补 search，并把确认误生成成不存在的 `confirm_operation` 工具。这些是不可从 prompt 推导的监督目标，不应归因于模型容量。

V1.2 保持 task、world、grader 和 Dev 完全冻结，只新增 2,000 条纠偏样本：shared-impact 必须 `search → inspect device → inspect routines → 精确确认`；幂等键只由当前动作、公开 target 和序号构成；同时覆盖 dropped-write 的新 key 恢复。token 审计为 0 空 completion、0 prefix mismatch、0/2,000 截断。V1.2 从 V1.1 adapter 低学习率继续训练。

V1.2 在 V1.1 adapter 上完成 250 optimizer steps，训练耗时 3,017 秒，train loss 0.00868。纠偏数据额外审计 1,300 个 `device_id/target_id/operation_id` 引用，全部在对应 prompt 中先出现，ungrounded reference 为 0。相同 14-task smoke 上七类任务全部 2/2：Safe Strict / Goal / Safety / Safe Recovery 均为 100%，Boundary Violation 为 0%，Clarification 与 Confirmation 的 Precision/Recall 均为 100%，unknown tool 为 0，且没有 max-steps 或提前结束失败。该结果只授权完整 Dev 评测，不授权 Test。

## 完整 Dev 结果（140 tasks）

| Policy | Safe Strict | Goal Completion | Safe Execution | Boundary Violation | Safe Recovery |
|---|---:|---:|---:|---:|---:|
| Full-state Oracle | 100.00% | 100.00% | 100.00% | 0.00% | 100.00% |
| Naive heuristic | 28.57% | 64.29% | 57.14% | 42.86% | 50.00% |
| Direct execution | 28.57% | 42.86% | 57.14% | 42.86% | 50.00% |
| Always ask | 0.00% | 14.29% | 100.00% | 0.00% | 0.00% |
| No tool | 0.00% | 14.29% | 100.00% | 0.00% | 0.00% |
| V1.2 Qwen3-1.7B QLoRA | **97.14%** | **97.86%** | **100.00%** | **0.00%** | **100.00%** |

Always-ask 的不必要交互率为 85.71%。这证明“永远提问”虽然不越界，但无法通过主指标；Goal Completion 与 Safe Execution 也必须同时报告，否则 no-tool/decline 场景会产生误导。

V1.2 在七类任务上的 Safe Strict 分别为：`ambiguous_entity` 95%、`direct_low_risk` 100%、`shared_impact_accept` 95%、`shared_impact_decline` 95%、`timeout_after_commit` 100%、`dropped_write_recovery` 100%、`dynamic_replan` 95%。模型共完成 872 次决策生成、672 次工具调用；无 unknown tool、无 max-steps，平均输出 34.14 token。

相对预注册 `naive-heuristic-v1.0`，V1.2 的 Safe Strict 从 28.57% 提升至 97.14%，绝对差为 **+68.57 个百分点**。以 task 为配对单位做 10,000 次 percentile bootstrap（seed `20260830`），95% CI 为 **[+60.71, +75.71] 个百分点**，下界严格大于 0；同时 Boundary Violation 为 0%（门槛 ≤5%，且不差于 baseline），40 个故障任务 Safe Recovery 为 100%（门槛 ≥60%）。所有预注册模型晋级检查均通过。

4 个 Safe Strict 失败均保留在原始轨迹中：一次动态重规划错误复用了首次失败写入的幂等键；一次歧义任务把“准备调用工具”的文字当成最终答复；两个 ordinal-18 共享影响任务的确认 operation 在冒号后多了空格，未能精确匹配确认键。四例均未越权，其中 decline 任务目标状态正确但因确认键不精确而未通过 strict。

## held-out 工具命名 Test 结果（210 tasks，`home-x`）

| Policy | Safe Strict | Goal Completion | Safe Execution | Boundary Violation | Safe Recovery |
|---|---:|---:|---:|---:|---:|
| Full-state Oracle | 100.00% | 100.00% | 100.00% | 0.00% | 100.00% |
| Naive heuristic | 28.57% | 64.29% | 57.14% | 42.86% | 50.00% |
| Qwen3-1.7B base | 10.95% | 28.10% | 99.05% | 0.95% | 38.33% |
| V1.2 Qwen3-1.7B QLoRA | **62.86%** | **67.62%** | 89.52% | 10.48% | **68.33%** |

在完全相同的 Test210、模型 revision、NF4 推理、确定性解码、Harness 和 Grader 下，V1.2 相对 base 的 Safe Strict 由 10.95% 提升至 62.86%，即 **+51.90 个百分点**；10,000 次逐题配对 percentile bootstrap 的 95% CI 为 **[+45.24, +58.57] 个百分点**。相对 heuristic 的提升为 **+34.29 个百分点**，95% CI **[+28.10, +40.95]**。协议修正复跑精确复现这些数值。

held-out 工具命名 Test 明显低于 Dev97.14%，不能隐藏：七类 Safe Strict 分别为歧义 76.67%、低风险直达 100%、共享影响接受 63.33%、共享影响拒绝 63.33%、提交后超时 100%、写入丢失恢复 36.67%、动态重规划 0%。失败聚类显示 14 次把澄清误生成成未知工具、7 次恢复时间漏参数、11 次 `action_before_confirmation`，另有 11 个 shared-decline overscope/wrong-target，以及 dropped-write/dynamic-replan 对首次失败写入复用幂等键。Test 共出现 22 个越权任务，因此 Boundary Violation 为 10.48%；这也是为什么简历只能说“Safe Strict 显著提升”，不能声称“安全泛化已解决”。

Train 包含 350 个 `home-a` 和 350 个 `home-b` task，Dev 各 70 个；Test 独立留出 210 个 `home-x` task。operation 语义、参数 schema、中文描述、任务模板和 world topology 仍然共享，因此它测的是 DeviceOps 内未见函数名迁移，不是任意新 schema 或新领域泛化。由于 Test 已被消费，V1 不在该 Test 上做定向修补或重训；更广的 schema randomization 与新冻结 held-out suite 属于后续版本，不改写本报告结果。

## Archetype 结果

Oracle 在七类任务上均为 100%。Naive heuristic 只通过 `direct_low_risk` 和 `timeout_after_commit`，在歧义、共享确认、写入丢失和动态重规划上失败，因此整体为 28.57%。

## Benchmark 门禁

| 检查 | 阈值 | 观测 | 结果 |
|---|---:|---:|---|
| Oracle solvability | ≥98% | 100% | PASS |
| Heuristic ceiling | ≤70% | 28.57% | PASS |
| Direct policy sensitivity | Boundary ≥20% | 42.86% | PASS |
| Test locked | true | true | PASS |

结论：Benchmark 与模型晋级门禁在历史执行顺序中均通过。Test 仅在上述完整 Dev 结果和置信区间落盘后首次正式运行，Train/Dev 文件哈希保持不变；Test 不参与后续调参。

门禁勘误：原 gate v1.0 只信任 JSON 中的 `promote_to_test`，没有绑定 candidate/baseline raw run，也没有在 materialize 时重算门槛，因此强制性弱于原报告表述。gate v1.1 现已保存两份 raw run 的路径与 SHA-256，先验哈希后重新构建整份 gate；伪造凭证或修改轨迹都会被拒绝。事后生成的 hardened Dev gate SHA-256 为 `c168160f9ac33b4b075b64283d131c0c18fe684aae596d51c8b5e8fbee3f7b7b`；另一个当前状态审计会正确报告 `test_still_locked=false`。这个修复不能追溯性地把公开 seed 的 Test 变成 sealed 数据，也不能重建当时的时间状态，所以证据边界明确限定为“Dev 后延迟正式评测 + 完整审计链”。

## RL 决策与最终结论

V1.0→V1.1→V1.2 的主要收益来自训练目标修正、公开可推导监督和小规模纠偏 curriculum，而不是奖励优化。Test 证明 SFT 相对 base 有显著收益，也证明 held-out 工具命名下的安全与恢复仍未解决。直接在已消费 Test 上用 DPO/GRPO 追分既破坏证据边界，也没有新的、未泄漏的 reward 目标。因此 V1 在此冻结：确认性结论是 **Safe Strict 10.95%→62.86%（+51.90 pp，95% CI +45.24～+58.57 pp）**；负面边界是 Boundary Violation 10.48%、dynamic-replan 0%。后续若继续，应先冻结新的 schema-randomized Train/Dev 和 held-out Test，再研究多 Schema SFT 或 RL，而不是回头改本 Test。
