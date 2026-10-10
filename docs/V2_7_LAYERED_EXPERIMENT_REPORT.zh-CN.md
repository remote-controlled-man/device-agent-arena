# DeviceAgent Arena V2.7：分层 Tool-Use Agent 实验报告

状态：**核心实验完成，可作为简历主项目**<br>
硬件：**NVIDIA GeForce RTX 4060 Laptop GPU（8 GB）**<br>
正式主结果：**未参与训练/修复的 Dev19–54，共 36 个同模板、实例级留出任务的配对评测**

## 结论先行

本轮最终做出的不是“让 270M 模型独立规划复杂任务”，而是一个可审计的 Compound Agent：

`FunctionGemma-270M 快路径提案 → 精确公开状态校验 → Qwen3-1.7B Planner 回退 → 确定性 Harness 执行`

在完全相同的 36 个同模板、实例级留出 development 任务、Qwen Planner、Harness、协议和源码树下，分层
方案相对 Qwen direct：

| 指标 | Qwen direct | 分层 Agent | 变化 |
|---|---:|---:|---:|
| Safe Strict | 22/36（61.11%） | **31/36（86.11%）** | **+25.00 pp** |
| 配对胜 / 平 / 负 | — | **9 / 27 / 0** | 无逐题退化 |
| 95% paired bootstrap CI | — | **+11.11～+38.89 pp** | 36 题 task-level 描述区间，10,000 次重采样 |
| Goal Completion | 72.22% | **91.67%** | +19.44 pp |
| Safe Execution | 94.44% | **94.44%** | 持平 |
| Minefield Violation | 5.56% | **5.56%** | 持平 |
| 跨域 output binding | 91.67% | **100%** | +8.33 pp |
| dual-control 失败恢复 | 33.33% | **100%** | +66.67 pp |
| Qwen Planner 调用 | 261 | **172** | **-34.10%** |
| 端到端 P50 延迟 | 31.14 s | **30.33 s** | -2.59% |
| 端到端 P95 延迟 | 59.43 s | **48.99 s** | **-17.56%** |
| 总模型调用 | 261 | 294 | **+12.64%** |
| 总输入 token | 381,032 | 418,862 | **+9.93%** |
| 峰值显存 | 2,737 MB | 2,982 MB | +245 MB |

因此，可信结论是：**小模型快路径减少了大模型 Planner 调用，并改善了长尾延迟和任务完成率；
但它没有减少总调用数或总输入 token，也没有修复剩余两条安全失败。**

## 1. 问题定义

设备 Agent 不是一次函数分类。一个任务可能要求 Agent：

1. 先检查多个目标与依赖；
2. 发现动作需要用户侧完成，等待并重新观察；
3. 提交异步操作，使用真实 operation ID 轮询；
4. 遇到超时或失败时判断是否已经提交，避免重复写入；
5. 在跨域链路中把上游工具输出绑定到下游参数；
6. 对不可行任务拒绝，对已完成任务正确结束。

正确动作随公开轨迹变化，因此评分对象是**完整闭环策略与最终世界状态**，而不是单步工具名。
主指标 Safe Strict 同时要求：目标完成、执行安全、轨迹满足协议、调用合法、结束正确。

## 2. 为什么采用分层架构

FunctionGemma-270M 有原生 tool-call 格式、速度快、显存小，但前序正式实验表明它独立承担
长链规划时无法通过 9/18 的预注册门槛。Qwen3-1.7B 的规划能力更好，但每一步都调用它会把
简单且可验证的动作也交给大模型。

V2.7 将两类决策拆开：

- **Qwen Planner**：处理歧义澄清、不可行判断、跨域依赖、共享影响确认和失败恢复；
- **FunctionGemma Router**：只在三个可由公开状态唯一确定的状态提案：用户动作成功后复查、
  异步写入提交后轮询、目标已验证完成后结束；
- **Exact Validator**：函数名与关键参数必须同时匹配预期动作，否则在同一步回退 Qwen；
- **Harness**：拥有工具执行、幂等键、安全边界、故障注入、状态转换和审计记录。

Router 不是隐藏的规则 Agent：它必须真实生成候选；validator 只做窄范围验收，不产生开放域
计划。两个模型都只读取公开指令、Schema、工具回执和对话历史，禁止读取 hidden goal、oracle
plan、grader 或 Test secret。

## 3. Qwen Planner 后训练

### 3.1 第一次结果与失败定位

协议修正后的 Qwen3-1.7B base 在 Train18 为 0/18。第一阶段使用 1,920 条 Train-only
next-decision 数据做 NF4 QLoRA，达到 8/18，距 9/18 Gate 只差一题；失败集中在
`apply_action` / `poll_action` 边界、跨域依赖绑定以及用户失败后的恢复。

### 3.2 真实失败轨迹纠错

系统从 10 条 Train 失败任务的 raw trajectory 中定位 53 个真实偏离状态，并对每个状态生成
oracle 下一动作；再混合失败任务完整 oracle、恢复分支和固定比例 replay，得到 1,202 条训练
课程记录、565 个唯一 prompt。1,202 不是 1,202 个独立失败状态：

| 数据来源 | 数量 |
|---|---:|
| 真实失败状态纠错 | 424 |
| 失败任务 oracle | 258 |
| 失败任务 recovery | 40 |
| 固定 replay | 480 |
| 合计 | **1,202** |

审计结果：Dev/Test 使用为 false，prompt 冲突 0，hidden marker 泄漏 0，completion 空监督 0，
prefix mismatch 0；最大长度从 2,304 调整到 3,072 后截断 0。

### 3.3 4060 单卡训练

从第一阶段 adapter 继续低学习率训练：Qwen3-1.7B、NF4 double quant、bf16、LoRA
`r=16 / alpha=32 / dropout=0.05 / all-linear`，有效 batch size 8，1 epoch，151 steps。
训练耗时 3,367.5 s（56.1 min），峰值分配显存 4,926.8 MB，adapter 大小 34.9 MB。
最终 Train18 达 10/18、Safe Execution 17/18，通过 Train Gate；一次性 Dev18 达 12/18，
通过一次性、哈希绑定的 promotion Gate。配置文件在本机早于 run 创建，但内部 `frozen_at`
误记为 run 开始后的时间；文件系统时间不可移植，因此本报告不声称具有可移植时间证明的“预注册”。

## 4. 分层 Agent 的一次安全回归与修复

首个探索性 Dev18 分层结果为 13/18，但 Safe Execution 从 direct 的 17/18 降到 16/18，
因此不能作为最终结果。raw trajectory 显示：用户动作返回 `user_action_unavailable` 时，快路径
仍强制提案 `inspect_target`，使 Planner 在错误状态上继续执行。

修复原则不是添加任务 ID 特判，而是收紧通用状态机：只有工具回执 `ok=true` 且用户响应属于
`done/failed` 时快路径才可进入；`unavailable/error` 必须同一步回退 Planner。修复只在 Train18
验证：分层方案 13/18，direct 10/18，Safe Execution 同为 17/18。此后保持 Agent、Harness
与模型冻结，仅增加 CLI 的 Dev tail selection / promotion receipt 支持；使用未参与训练/修复的
Dev19–54 做最终配对实验。两份 final run 使用完全相同的源码树，没有重跑 Dev18 冒充新验证集。

## 5. 最终评测设计

- 任务：Dev19–54，36 题；三个域与三个切片各 12 题；Train/Dev 共享 9 个 instruction/template，
  这里只留出任务实例、实体 ID 与状态，不是未见 prompt 或跨 Schema 泛化；
- 对照：同一个 Qwen3-1.7B base revision + 同一个 refinement adapter；
- 控制变量：相同任务 ID/顺序、源码树、Harness、协议、温度 0 和 Qwen 解码长度；
- 统计：逐题胜平负与 10,000 次 paired percentile bootstrap；CI 是这 36 个任务的 task-level
  描述区间，不外推为生产分布置信区间；
- 审计：每次模型原始输出、被执行 tool call、token、延迟、显存、模型树和 adapter SHA 均入 raw；
- 边界：这 36 题是未进入训练/修复的**实例级留出、模板共享 development tail**，不是外部 sealed Test。

FunctionGemma 快路径在 122 个 eligible 状态中接受 107 次，接受率 87.70%，15 次精确校验失败
后回退 Qwen。Qwen 调用减少 89 次，但 Router 新增 122 次调用，所以总调用从 261 增至 294。

## 6. 误差分析与边界

- 最终 5 条 strict failure 中，3 条属于 dual-control：Planner 在用户动作完成后把公开
  `available_actions` 错写为带 target suffix 的 action key，随后重复请求不可用的用户动作并触发
  error limit；2 条属于 workspace 不可行任务，其中一条未及时拒绝，另一条对错误/未授权目标
  尝试写入。这说明剩余瓶颈在 Planner 的 action-value grounding 和不可行终止，而不在快路径。
- 后两条构成 2/36 minefield violation，故 Safe Execution 与 direct 同为 34/36；不能写
  “解决了 Agent 安全问题”。
- 36 题样本仍小；置信区间已显著为正，但不是生产规模或真实用户流量证据。
- 环境是确定性文本模拟器，不包含真实 Android/iOS UI、视觉定位、网络抖动或线上设备差异。
- 总输入 token 和总调用数上升；该架构优化的是**昂贵 Planner 调用与长尾延迟**，不是所有成本。
- FunctionGemma 与 Qwen 当前是顺序加载/调用；若做服务化，可进一步验证并行预取、KV cache、
  batching 与模型驻留策略。
- Exact Validator 已能从公开状态构造唯一动作；当前尚未补“不调用 FunctionGemma、直接执行
  确定性快路径”的消融。因此能证明的是“真实小模型提案 + 精确校验的 Compound Agent”有效，
  不能证明 learned Router 优于纯规则快路径。
- 延迟来自本次 RTX 4060 上每个条件各一次的顺序运行，没有多次交错重复；P95 -17.6% 只描述
  本次实验。
- Test90 仍未物化，不能声称获得 sealed Test 成绩。

## 7. 可复核证据

- 最终 direct raw：`artifacts/v2/layered/runs/qwen3-1.7b-v2.7.1-refined-direct-dev36-tail-v0.1.jsonl`
- 最终 layered raw：`artifacts/v2/layered/runs/functiongemma-v2.5-fastpath-qwen-v2.7.1-dev36-tail-v0.2.jsonl`
- 配对报告：`artifacts/v2/layered/reports/functiongemma-v2.5-fastpath-vs-qwen-v2.7.1-direct-dev36-tail-paired-v0.2.json`
- Dev18 Gate：`artifacts/v2/layered/gates/qwen3-1.7b-v2.7.1-refined-direct-dev18-gate-v0.1.json`
- 训练结果：`artifacts/checkpoints/qwen3-1.7b-deviceops-v2.7-composition-refinement-qlora-v0.2/device_agent_arena_training_result.json`
- 数据 manifest：`data/v2/qwen3-composition-refinement-sft-v2.7.1.manifest.json`
- 事后补齐、明确标注的来源 amendment：`artifacts/v2/layered/training/qwen3-v2.7.1-refinement-lineage-amendment.json`
- token 审计：`artifacts/v2/layered/training/qwen3-v2.7.1-refinement-token-audit-3072.json`
- 最终发布清单：`artifacts/v2/layered/release-manifest-v2.7.json`

raw attestation 会验证：任务与源码树、两层模型树、adapter 权重、chat template、推理参数，以及
每个被采纳动作都能从相应模型 raw output 精确解析得到。聚合报告可从 raw 确定性重算。

## 8. 简历可用结论

这个项目最有价值的部分不是“我微调过两个模型”，而是完整闭环：**发现模型边界 → 设计分层
系统 → 从真实失败轨迹构造训练课程 → 在消费级显卡完成 QLoRA → 哈希绑定 Gate → 同题配对评测
→ 保留安全回归和成本反例 → 用 raw attestation 固化证据**。

简历最终写法见 [V2.7 简历条目与 STAR](RESUME_PROJECT_V2_7.zh-CN.md)。
