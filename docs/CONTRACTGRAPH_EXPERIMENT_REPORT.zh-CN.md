# ContractGraph Harness + FunctionGemma Verifier 实验报告

状态：**实验闭环已完成；ContractGraph Harness 通过工程验证；learned verifier 未通过 Dev Gate；Test-XSchema 未生成、未揭示。**

English version: [CONTRACTGRAPH_EXPERIMENT_REPORT.md](CONTRACTGRAPH_EXPERIMENT_REPORT.md)

## 结论先行

本项目最终没有得到“微调一个 270M 模型就能可靠规划任意新工具”的结论。实验支持的是一个更
具体、也更可信的结论：对于带显式前置条件、效果、重试和不变量的 Tool-Use 任务，
**ContractGraph 确定性运行时是可靠性主体；learned verifier 可以做候选排序研究，但当前版本
没有达到可晋级标准。**

- 合成跨 Schema Dev 上，脚本式基线为 **16/64（25%）**，ContractGraph 为
  **64/64（100%）**，参数有效率 100%、契约违规 0、故障恢复 100%。这是一项公开 Contract IR
  下的工程回归，不是小模型泛化成绩。
- FunctionGemma-270M v0.8 在完全留出的 Dev-XSchema 上取得 **66.0% Top-1**，高于
  no-ranker 18.5% 和通用规则 25.0%，但 `invalid_arguments` 仅 60.0%，
  `wrong_or_redundant_repair` 仅 41.33%，未达到每类 70% 的预注册 Gate。
- 裸 verifier 多步闭环为 **74/200（37.0%）**，虽然比通用规则 25.0% 高 12pp，配对
  Bootstrap 95% CI 为 **[+6.5pp, +17.5pp]**，但契约违规率达 **41.95%**、按全部
  fault-planned 任务计算的恢复率仅 **15/40（37.5%）**，因此仍被拒绝。
- 失败后追加、明确标为不可晋级的 post-hoc 诊断显示：在相同 ContractGraph 可行性掩码下，
  通用规则 **100%**、learned verifier **90%**、no-ranker **22.5%**；learned 相对通用规则
  **-10pp，95% CI [-14.5pp, -6.0pp]**。这证明当前显式契约已经让确定性方法足够，继续为该域
  训练 verifier 没有收益。

因此，本轮按预注册规则不冻结候选、不生成 sealed Test-XSchema，也不把 learned verifier 写成
性能提升。未揭示 Test 是实验纪律的结果，不是缺失的“最好数字”。

## 系统做了什么

运行链路是：

`用户可见指令 → 候选动作 → JSON Schema 校验 → ContractGraph 可行性/修复 → 工具执行 → 后置验证 → 有界恢复`

核心实现包括：

1. 与工具 Schema SHA-256 绑定的 Contract IR，描述角色、前置条件、效果、可观察事实、重试和
   ambiguous-commit 策略；未知契约、Digest 不一致和修复环均 fail closed。
2. 只接收 allowlist `AgentContext` 的状态化执行引擎；不接触 `TaskSpec`、隐藏 goal、故障计划或
   grader；episode 隔离、batch、结构化错误和后置条件均有自动测试。
3. 固化 Train/Dev 数据、模型 revision、配置、adapter，以及 rollout 所用 records / score report /
   catalog 的 SHA-256。项目提供 source freeze 工具，但本候选未过 Dev，因此没有生成源码冻结回执，
   也没有一次性生成两个新域的 sealed Test-XSchema。
4. FunctionGemma-270M 的 NF4 QLoRA sequence-classification/listwise 排序链路，以及 no-ranker、
   schema-only 通用规则、ContractGraph oracle 的共享候选池对照。

## 数据与硬件回执

v0.8 数据共 10,000 条候选记录：Train 8,000、Dev-ID 1,000、Dev-XSchema 1,000；每个状态组
5 个候选且恰有 1 个正例。Train 使用会议室与演播室两个语义域的 6 组工具/参数/事实命名变形，
Dev-XSchema 完整留出 `aliased-node-v1`。其中 87.475% 的负例在 JSON Schema 上合法，50.94%
记录属于状态变化后标签翻转的反事实组；隐藏运行字段、重复输入和重复 ID 均为 0。

- 数据 SHA-256：`90094aab62cb569c58cd33fa1da8cda3bd2ffcc50bbcbda56eb2e2b8244977de`
- 模型：`google/functiongemma-270m-it`
- 模型 revision：`39eccb091651513a5dfb56892d3714c1b5b8276c`
- RTX 4060 Laptop 完整训练：1 epoch / 100 updates / 1,230.08 s
- 峰值 allocated / reserved：1,902.322 / 2,240 MiB；token 截断 0
- 配置 SHA-256：`2a623a3e0a485e777ee6b59372735bfae14a75da849afa6855168c4e441a6dbd`

`train_loss=5.6729` 和训练子集 94.53% 诊断准确率只用于确认训练过程，不作为 Agent 能力指标。

## 静态跨 Schema 结果

下表全部来自 1,000 条 Dev-XSchema 候选、200 个五选一状态组；Test 未参与任何选择。

| 版本 | 主要变化 | Top-1 | Pairwise | Balanced | 正例召回 | Invalid 负例 | 错误/冗余修复负例 | 结论 |
|---|---|---:|---:|---:|---:|---:|---:|---|
| v0.3 | binary verifier | 63.5% | 90.88% | 73.69% | 57.0% | 100% | 98.89% | 高排序、低正例召回，拒绝 |
| v0.5 | 新训练域 + 工具顺序随机化 | 66.5% | 91.63% | 61.44% | 27.5% | 50.0% | 100% | 多数判负式退化，拒绝 |
| v0.6 | listwise + 辅助 BCE | 60.5% | 90.13% | 71.06% | 59.0% | 72.0% | 51.33% | 难负例失衡，拒绝 |
| v0.7 | 加入 public Contract IR | 59.0% | 79.50% | 70.56% | 62.5% | 86.0% | 70.67% | 契约输入仍未泛化，拒绝 |
| v0.8 | 工具/参数/事实三重别名变形 | **66.0%** | **90.06%** | **83.75%** | **91.5%** | 60.0% | 41.33% | 两类负例低于 70%，拒绝 |

v0.8 的 Dev-ID Top-1 为 96.0%，而 Dev-XSchema 为 66.0%；30pp 差距表明不能用同分布结果替代
跨 Schema 泛化。

## 多步闭环结果

闭环从 200 个 Dev-XSchema 状态快照出发，最多执行 6 步；固定 40 个任务被预定一次瞬时故障。
如果策略在到达 mutate 前就失败，故障不会实际触发，但主恢复率仍把该任务计入分母，防止策略
通过避开故障虚高恢复率；另报告只针对实际触发任务的条件恢复率。

| 版本 | Learned 严格成功 | 通用规则 | No-ranker | Oracle | Learned 契约违规率 | 主恢复率 | Gate |
|---|---:|---:|---:|---:|---:|---:|---|
| v0.5 | 38.5% | 25.0% | 3.5% | 100% | 72.66% | 40.0% | 失败 |
| v0.6 | **44.5%** | 25.0% | 3.5% | 100% | 54.74% | 40.0% | 失败 |
| v0.8 | 37.0% | 25.0% | 3.5% | 100% | 41.95% | 37.5% | 失败 |

v0.8 的 ready 状态成功率为 90%，但 unknown 与 unbound 均只有 12%，bound-but-untuned 为
34%。轨迹显示模型通常能选对第一步，却会在状态已经改变后重复 `bind_node` 或 `tune_node`，
把局部合理动作重复到 6 步上限。这解释了为何静态正例召回改善，却没有转化为闭环提升。

## 完整负结果台账

| 发现 | 如何发现 | 处理与最终口径 |
|---|---|---|
| 旧 Test64 被 parser coverage 测试读取 `required_tools` | 证据边界审计 | 降级为已消费回归集；不再称 held-out |
| Windows CRLF 导致数据文件 Hash 与 Manifest 不一致 | 训练前 Hash 校验 | 固定 LF 写入并重新生成 Manifest；旧回执保留 |
| v0.3.1 正例总在候选首位 | 位置分布审计 | 该数据退役；每组候选随机化并审计位置分布 |
| 工具列表顺序固定形成词面捷径 | 跨别名错误复盘 | v0.5 起随机化工具顺序；未把旧高分用于晋级 |
| binary v0.5 正例召回仅 27.5% | 分类型静态指标 | 改 listwise，但 v0.6 仍未通过 |
| public Contract IR 未自动带来泛化 | v0.7 Dev-XSchema | 加工具、参数、事实命名变形；v0.8 仍失败 |
| 静态 66% 未转化为闭环提升 | v0.8 rollout | 定位重复修复与状态更新后的排序错误 |
| 恢复率曾只统计实际触发故障的任务 | 独立终审发现 learned 仅触发 15/40 个计划故障 | 主指标改为全部 fault-planned 任务；保留条件恢复率并重跑 v0.5/v0.6/v0.8 |
| Contract 掩码下 learned 比规则低 10pp | post-hoc 同掩码公平对照 | 判定该域不需要 learned verifier；不可用于晋级 |

## 为什么没有继续做 DPO / GRPO

当前瓶颈不是“模型没有偏好信号”，而是显式 Contract IR 已足以让规则基线达到 100%，同时
learned ranker 在同一掩码下只有 90%。继续用 DPO/GRPO 优化一个被强基线支配的组件，会增加
训练复杂度，却没有明确的产品价值。项目其他研究线已经实际实现并保留 DPO/GRPO 的 OOM、
零方差和未晋级负结果；本轮不再为了技术名词重复训练。

若未来重启 learned planner，前提应是新 benchmark 包含 Contract 无法决定的语义冲突、多合法路径
成本权衡或不完整契约，并重新预注册 Gate；不能在看过本轮 Dev 后生成 Test 来追分。

## 可复核证据

- v0.8 数据 Manifest：[`data/train/manifest_contract_verifier_v0.8.json`](../data/train/manifest_contract_verifier_v0.8.json)
- 完整训练回执：[`artifacts/training/functiongemma-270m-contract-verifier-listwise-v0.8.1.json`](../artifacts/training/functiongemma-270m-contract-verifier-listwise-v0.8.1.json)
- 静态 Dev 报告：[`artifacts/reports/functiongemma-contract-verifier-listwise-v0.8.1-dev.json`](../artifacts/reports/functiongemma-contract-verifier-listwise-v0.8.1-dev.json)
- 多步与 post-hoc 诊断：[`artifacts/reports/functiongemma-contract-verifier-listwise-v0.8.1-dev-rollouts.json`](../artifacts/reports/functiongemma-contract-verifier-listwise-v0.8.1-dev-rollouts.json)
- ContractGraph 计划与预注册门槛：[`docs/CONTRACTGRAPH_PROJECT_PLAN.zh-CN.md`](CONTRACTGRAPH_PROJECT_PLAN.zh-CN.md)

## 简历边界

可以写：实现 ContractGraph Harness、跨 Schema 合成 Dev 64/64、零契约违规、零隐藏字段、4060 上
完成 10k 条 FunctionGemma verifier 训练与多版失败归因、因强规则基线 100% 而主动拒绝 learned
候选。

不能写：learned verifier 达到 100%、通过 sealed Test、在真实设备或线上流量验证、SFT/RL 带来
可靠性提升、任意新工具零样本泛化。若简历只允许 2–3 条，主写 ContractGraph 运行时与可审计
评测；learned verifier 作为面试中的负实验展开。
