# DeviceAgent Arena V2.7：分层 Planner / Router / Harness 计划

状态：**核心实验完成（最终证据见 V2.7 实验报告）**<br>
冻结日期：2026-09-02
架构修订：2026-09-03（修订不读取 Dev/Test）

## 研究问题

在 RTX 4060 8GB 上，把“复杂决策”和“本地确定性执行”拆开，是否能比 FunctionGemma-270M
独立规划更可靠，同时保留小模型的低成本价值？

目标架构：

`FunctionGemma speculative fast path → exact public-state validator → Qwen3-1.7B fallback Planner → deterministic Harness`

- Planner 只依据公开指令、工具 Schema 与轨迹处理完整长链决策；
- FunctionGemma 不承担完整规划，只在三个可由公开状态精确验证的局部状态提案：用户动作后复查、
  动作提交后轮询、目标动作完成后结束；
- validator 只接受函数名与关键参数均精确匹配的提案，否则同一步回退 Qwen；
- Harness 管理幂等键、公开状态、工具执行、安全边界、失败恢复与审计；
- 任一模型都禁止读取隐藏 goal、oracle plan、grader 或 Test secret。

## 分阶段门禁

### A. Direct Planner Train18 pilot

先在冻结 Train18 上运行 Qwen3-1.7B 4-bit、temperature=0 的直接模型基线。配置、模型树、
chat template、任务顺序与阈值均在运行前写入
`configs/qwen3-1.7b-v2-composition-train18-pilot-v0.2.json`。v0.1 首次运行发现 Schema
对话工具被错误暴露给 Composition 环境，导致模型反复请求环境无法执行的确认；该 0/18 仅作为
协议故障证据保留。修复经过 1 条 Train smoke 验证后，以 v0.2 原门槛重新冻结，未读取 Dev。

通过条件：Safe Strict 至少 9/18、Safe Execution 至少 17/18、minefield 至多 1/18、平均
milestone coverage 至少 0.60，且三个切片各至少成功 1 条。

- 通过：实现 Planner + FunctionGemma verifier/router，并做相同任务的配对消融；
- 未通过：先用 Train-only oracle/recovery 轨迹训练 Qwen planner adapter，再重新做新的 Train gate；
- 两条路径都不读取 Dev/Test。

实际结果：base Qwen 为 0/18 Safe Strict；第一阶段 Train-only QLoRA 为 8/18，安全执行
17/18、平均 milestone coverage 0.725，距门槛差 1 题。失败集中在 `apply_action` / `poll_action`
边界和跨域依赖节点绑定。因此冻结 10 条失败任务并重放出 53 个真实偏离状态，形成 1,202 条
V2.7.1 纠错、失败任务 oracle/recovery 与固定比例 replay 数据；数据哈希、零 prompt 冲突、
零隐藏状态泄漏和 3072-token 零截断均已记录。该精修只使用 Train。

### B. Layered Train18 paired ablation

同一 Qwen checkpoint、同一任务顺序、同一 Harness 下比较：

1. Qwen direct + deterministic Harness；
2. FunctionGemma speculative fast path + exact validator + 同一 Qwen fallback + 同一 Harness。

除 Safe Strict 外报告安全执行率、minefield、milestone coverage、切片成功率、快路径接受率、
fallback 率、Planner/Router token、Planner 调用节省、总调用数、P50/P95 延迟和峰值显存。
比较强制同任务、同源码树、同 Harness、同协议与同 Qwen 权重，并做 10,000 次 paired bootstrap。
FunctionGemma 的价值必须来自该配对增量，不能用 Oracle 或规则代理成绩代替。

### C. Dev18 promotion

只允许 Train 阶段冻结出的单一候选运行一次 Dev18。门槛沿用 V2 composition gate：至少 9/18，
并同时满足安全、不可行识别、dual-control 恢复与跨域 output binding 要求。失败后不围绕 Dev
继续调参；成功后才允许扩大到 Dev54。正式 Test90 继续保持未物化。

实际结果：精修 Qwen 在一次性 Dev18 上获得 12/18，通过全部哈希绑定 Gate；配置内部
`frozen_at` 晚于 run，故不声称具有可移植时间证明的“预注册”。首个探索性分层
Dev18 虽为 13/18，但安全执行由 17/18 降至 16/18；仅使用 raw trajectory 诊断并在 Train18
修复通用 `user_action_unavailable` 状态错误。Agent/Harness/模型冻结后（仅增加 tail selection
CLI），在未参与训练/修复、但与 Train 共享 instruction/template 的 Dev19–54 实例上完成最终
36 题配对：Safe Strict 22/36→31/36（+25.0 pp，95% paired bootstrap CI +11.1～+38.9），
安全执行同为 34/36，Qwen 调用 -34.1%，P95 延迟 -17.6%。详见
[V2.7 实验报告](V2_7_LAYERED_EXPERIMENT_REPORT.zh-CN.md)。

## 训练与 RL 决策

- 首选 QLoRA，而不是全参数训练；4-bit NF4 + double quant，单卡 8GB 可运行；
- 先做 next-decision SFT，只有 Train 上端到端指标和同 prefix 好坏分叉同时出现，才考虑 DPO；
- 只有固定 prompt 下多次 rollout 存在稳定 reward 方差、有效正样本比例和安全收益，才考虑 GRPO；
- 禁止因为“想展示 RL”而在没有学习信号时训练。

## 验收边界

- 所有 model-labelled raw 必须由受信 Transformers runner 生成，并绑定模型树、配置和源码树；
- 报告必须能从 raw 精确重算；
- 分层运行必须分别保留 Planner 与 Router 的原始输出和用量；
- Test secret 不进入仓库，Test90 不在开发期生成；
- 对外只陈述真实运行结果，规则 Harness 与模型结果严格分开。

## 研究依据

- Qwen3 官方模型卡：<https://huggingface.co/Qwen/Qwen3-1.7B>
- FunctionGemma compound-system 定位：<https://ai.google.dev/gemma/docs/functiongemma>
- FunctionGemma 微调说明：<https://ai.google.dev/gemma/docs/functiongemma/finetuning-with-functiongemma>
- Transformers bitsandbytes / NF4：<https://huggingface.co/docs/transformers/quantization/bitsandbytes>
