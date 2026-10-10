# DeviceAgent Arena V2：组合交互与 FunctionGemma 实验报告

English: [V2_COMPOSITION_EXPERIMENT_REPORT.md](V2_COMPOSITION_EXPERIMENT_REPORT.md)

状态：已执行，FunctionGemma 候选未晋级；Test 保持锁定<br>
日期：2026-09-02

## 1. 结论

本轮不是单次函数分类，而是完整 Agent loop：读取目标、发现关系、协调用户侧动作、按依赖执行、轮询结果、绑定上游 operation ID、验证终态，并在组合约束不可满足时拒绝。

冻结 Suite 包含 Train90、Dev54，以及“仅预留协议、尚未物化”的 Test90，覆盖 `dual_control`、`composed_infeasible`、`cross_domain_chain` 三个切片，home / vehicle / workspace 三个领域，以及 linear / fan-out-join / diamond-gate 三种拓扑。旧 manifest 的 `root_seed` 现通过哈希绑定 amendment 明确限定为 Train/Dev；正式 Test 必须使用仓库外独立 256-bit secret，secret 与 `test.jsonl` 均不存在于仓库中。

核心结论：

- FunctionGemma-270M model-only 零样本在冻结 Dev18 为 **0/18**；成功轨迹 SFT 后为 **3/18（16.67%）**，未通过预注册的 9/18 gate；
- recovery SFT 在 Train18 上把普通 dual-control 学到 3/3，但总体仍为 **4/18（22.22%）**，与前一阶段相同，并牺牲不可行判断；因此停止继续 SFT，不消费新的 Dev；
- 只读取公开 observation 的 reactive harness 在完整 Dev54 为 **45/54（83.33%）**；加入用户失败后的重新观测与有界重试后，public recovery harness 为 **54/54（100%）**、minefield 0%；
- 100% 是确定性公开契约控制器的工程回归，不是模型权重分数，也不是 Test 泛化分数。

## 2. 评测合同

任务隐藏完整 world、goal、milestone DAG、minefield 与 oracle plan。Agent 只看到用户请求、公开上下文、7 个工具 schema、先前 tool call/result 和用户返回。

Safe Strict 同时要求：

1. 最终状态满足；
2. 所有 required milestone 完成；
3. milestone precedence DAG 成立；
4. terminal 行为正确；
5. 无越权、错误目标、不可行时变更、提前变更或伪造 output binding；
6. 成本和调用合法性通过。

空拒绝、直接完成和只检查终态均不能通过。组合不可行使用同指令的 feasible / infeasible counterfactual pair；跨域目标动作必须携带运行时真实生成的上游 operation ID。

## 3. 可解性与弱基线

| 条件 | 范围 | Safe Strict | 最差切片 | Minefield | 说明 |
|---|---:|---:|---:|---:|---|
| Oracle | Dev54 | 54/54 | 100% | 0% | 仅证明可解，不参与部署比较 |
| Finish immediately | Dev54 | 0/54 | 0% | — | 排除空完成捷径 |
| Refuse immediately | Dev54 | 0/54 | 0% | — | 排除一律拒绝捷径 |
| Public reactive | Dev54 | 45/54 | 50% | 0% | 用户首次失败后停止 |
| Public recovery | Dev54 | **54/54** | **100%** | **0%** | 公开状态重观测、重试、依赖与绑定控制器 |

Public recovery 在三个切片、三个领域和三种拓扑上均为 100%；不可行 precision / recall / TNR、dual user-failure recovery 和 cross-domain output binding 也均为 100%。它不接收 hidden task 或 grader truth。

## 4. FunctionGemma 实验

### 4.1 零样本迁移

v2.3 checkpoint 在 composition Dev18 为 **0/18**，平均 milestone coverage 7.36%，安全执行 100%。几乎所有首个高层决策都错误地调用 `request_user_action` 并编造 action key，说明失败不是 parser，而是策略与领域迁移。

### 4.2 v2.5 成功轨迹 SFT

- 数据：1,985 条 FunctionGemma-native next-decision；1,260 条 composition oracle 记录 + 725 条旧能力回放；
- 审计：最长 2,288 / 2,304 tokens，截断 0，空 completion 0，prompt-prefix mismatch 0；
- 训练：BF16 全参数，1 epoch，125 updates，RTX 4060 Laptop，峰值 3,394 MiB，713.6 秒；
- Dev18：**3/18（16.67%）**，平均 milestone coverage 48.33%，minefield 7/18；仅 feasible-control 3/3 成功，dual-control 与 cross-domain 均 0/6；
- gate：FAIL；9/18 主门槛、三切片非零、恢复、不可行识别、output binding 与安全门槛均未同时通过。

原始轨迹显示模型学会 `inspect_target → inspect_relations`，但随后经常直接操作最终目标；收到 `precondition_failed` 后重复“检查目标→同一错误动作”。

### 4.3 v2.6 recovery SFT

v2.6 加入成功轨迹中后段 hard states，以及提前执行、重复 precondition failure、用户动作 unavailable 后的 Train-only 纠正状态：

- 数据：3,185 条；1,800 hard-success、660 recovery、725 replay；
- 审计：最长 2,288 / 2,304 tokens，截断 0，空 completion 0；
- 训练：从 v2.5 继续 BF16 全参数 1 epoch，200 updates，峰值 3,402 MiB，1,196.7 秒；
- Train18：仍为 **4/18**；dual-control 由 2/6 提至 3/6、re-observation 到 100%，但 cross-domain 仍 0/6，不可行与安全能力退化。

由于 Train loop 都没有总体增益，v2.6 未再次消费 Dev18。继续增加 epoch 会扩大能力挤压风险，不能由训练 loss 代替端到端证据。

## 5. 为什么不做 DPO / GRPO

当前失败在确定性解码下高度一致，主要是错误关系解释、错误动作后的循环和跨步变量绑定。SFT 尚未建立可靠 Train loop，缺少稳定成功/失败同前缀分叉；此时 DPO 没有可信 preference support，GRPO 也会遇到近零 reward variance 与长程 credit assignment。RL 触发条件不满足，因此保留负结果并停止，而不是为简历制造收益。

## 6. 系统设计结论

270M 模型适合 schema grounding 与局部工具决策，但不应独自承担所有确定性执行约束。可部署分层应为：

- 模型：理解用户语义、处理真正开放的选择；
- Harness：schema 校验、依赖顺序、用户侧状态机、错误熔断、幂等与 operation binding；
- Grader / telemetry：milestone DAG、minefield、终态与逐层失败归因。

该分层不是用隐藏答案替模型做题，而是把可验证的确定性职责从概率模型中剥离。

## 7. 证据边界与路径

- Suite：`data/v2/deviceops-ul-v2-composition-v0.1/`
- Dev54 SHA-256：`9ddfc87c787b107ead4be1af235a932b49b04b4e968554c919e23b0e028de171`
- v2.5 Dev18 最终验真的 raw / report / FAIL receipt：`artifacts/v2/composition/runs/functiongemma-v2.5-stage4-dev18-attested-v2.jsonl`、`reports/functiongemma-v2.5-stage4-dev18-attested-v2.json`、`gates/functiongemma-v2.5-stage4-dev18-attested-receipt-v0.4.json`
- v2.6 Train18 raw / report：`artifacts/v2/composition/{runs,reports}/`
- public recovery Dev54 raw / report：`artifacts/v2/composition/runs/public-recovery-dev54.jsonl` 与 `artifacts/v2/composition/reports/public-recovery-dev54.json`
- 训练结果：`artifacts/checkpoints/functiongemma-270m-deviceops-v2.{5,6}-full-sft-stage*/device_agent_arena_training_result.json`

最终重跑发生在审计加固后，期间没有修改模型、训练数据或策略；结果与第一次 Dev18 完全一致，因此它是同一已见 Dev 的可复现性核验，不是新的独立样本。v0.4 回执同时验证 checkpoint-tree hash、raw JSONL、由 raw 精确重算的 report、冻结的模型协议与解码参数，并逐步核对 `raw_model_output` 解析出的调用和实际执行调用；公开 runner 会拒绝普通 Harness 写出 model-labelled raw。该回执是本地可复现性 attestation，并不对恶意修改仓库源码的操作者提供密码学证明；正式第三方榜单仍需仓库外评测器签名。旧 v0.1 回执保留为历史弱证据，不再用于解锁。

V2 composition Test90 仍锁定。当前唯一正式 Test 结论仍是冻结的 V1 Qwen Test210；不得把 Dev54 的 harness 100% 写成 Test 或模型泛化成绩。
