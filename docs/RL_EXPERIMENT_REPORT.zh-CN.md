中文 | [English](RL_EXPERIMENT_REPORT.md)

# DeviceAgent Arena：RTX 4060 在线 Agent RL 实验报告

状态：探索性实验完成，候选未晋级

实验版本：RL task suite v0.3 / reward v0.3 / evidence v0.2

硬件：NVIDIA GeForce RTX 4060 Laptop GPU，8,188 MiB VRAM

## 1. 结论先行

这台 4060 **可以完成** Qwen3-1.7B 的 4-bit QLoRA 和多轮环境 GRPO：任务专用
SFT 峰值分配显存为 3,507 MiB；两条 GRPO smoke 的整卡采样峰值分别为
6,378.5 MiB 和 6,806.5 MiB，均低于预注册的 7,700 MiB 护栏。

但“能训练”不等于“RL 有效”。在冻结的 8 题结构留出 Dev smoke 上：

| 路线 | 训练前 | GRPO 后 | 变化 |
|---|---:|---:|---:|
| 任务 SFT warm start | 4/8（50.0%） | 3/8（37.5%） | -12.5 个百分点 |
| 通用 Tool-SFT 起点，无 v0.3 任务模仿 | 3/8（37.5%） | 3/8（37.5%） | 0.0 个百分点 |

两条路线都未达到 90% 非零奖励方差与非零梯度的学习信号门槛，因而没有扩大到
Dev32，更没有运行 Test64。当前可支持的结论是：**4060 不是瓶颈；当前瓶颈是任务
可探索性、组内全失败和长轨迹截断。**

## 2. 这不是单次 LLM 分类

每个 episode 都在同一有状态工具环境中循环：模型选择动作，模拟器执行工具，环境返回
观察或错误，模型再根据反馈恢复，最后由真实环境状态和必要执行路径评分。Agent 不能直接
读取隐藏状态，只能调用 `get_device_state` 或从工具反馈获得信息。

v0.3 冻结任务包含：

- `reconfigure`、`configure_save`、`recall_preset` 三类工作流；
- `cold`、`connected`、`ready`、`active` 四类合法初始状态；
- 显式信息与必须先检查状态两种模式；
- 缺参澄清、瞬时故障及恢复；
- 3–7 个必要动作的轨迹；
- Train/Dev/Test 完整结构组合两两不相交，但 Dev/Test 的每个原子因子都在 Train
  出现，测试的是组合泛化而非未见工具。

冻结规模为 Train128 / Dev32 / Test64。任务 seed 同时绑定任务取值和故障是否触发；
相同任务与 seed 的环境执行完全可重放。Oracle 在全部 224 题上为 224/224；旧固定宏
策略在 Dev 上仅为 5/16，说明任务不能靠一条固定工具序列通吃。

## 3. 为什么同时做 SFT 与 direct GRPO

SFT 在这里是**行为初始化和对照**，不是 RL 的理论前提。它先教会模型交互工具协议、
参数复制、必要澄清和合法终止；GRPO 则尝试从多轮环境结果中优化策略。与此同时，
[ToolRL](https://arxiv.org/html/2504.13958v1) 报告了 direct GRPO 可能比任务 SFT
初始化具有更好的泛化，因此本实验冻结了计算匹配的两条路线：

- `task-SFT v0.4 → warm GRPO`；
- `clean-v0.6 通用 Tool-SFT → direct GRPO`，不使用 v0.3 任务轨迹模仿。

两条 GRPO 配置除初始 Adapter 外相同：Qwen3-1.7B、NF4 4-bit、LoRA r16、
Dr. GRPO、4 generations、4 optimizer updates、seed 42、相同 12 个训练任务、
最大 10 个环境步骤和 320 completion tokens。这里的 direct 指“无 v0.3 任务 SFT”，
并非从未经任何后训练的原始基座开始。

## 4. 数据与训练记录

### 4.1 任务专用 SFT v0.4

Oracle 从 v0.3 Train128 生成 719 个唯一下一决策监督。为缓解此前模型不愿主动澄清的
缺陷，23 个 `request_clarification` 决策各额外重复 3 次，最终为 788 条记录；重复策略
与唯一记录数均写入 Manifest，而不是把 788 包装成 788 个独立样本。

| 项目 | 实测值 |
|---|---:|
| 优化器步数 | 50/50 |
| 运行时间 | 681.11 s |
| Train loss | 0.099265 |
| 峰值分配显存 | 3,507.22 MiB |
| Adapter SHA-256 | `610793194d17...1b7f294` |

冻结 Dev8 上严格成功率 4/8；2 个缺参任务都发起了澄清，但后续执行仍失败，因此不能把
澄清动作本身当作任务成功。

### 4.2 计算匹配 GRPO smoke

| 指标 | Warm GRPO | Direct GRPO |
|---|---:|---:|
| 优化器步数 | 4/4 | 4/4 |
| 运行时间 | 113.68 s | 113.20 s |
| 非零 reward-std 更新 | 2/4（50%） | 2/4（50%） |
| 非零梯度更新 | 2/4（50%） | 1/4（25%） |
| completion 截断均值 | 43.75% | 50.00% |
| PyTorch allocated 峰值 | 4,668.55 MiB | 4,717.33 MiB |
| 整卡采样占用峰值 | 6,378.5 MiB | 6,806.5 MiB |
| 7,700 MiB 显存 Gate | 通过 | 通过 |
| 学习信号 Gate | **拒绝** | **拒绝** |
| Dev8 晋级 Gate | **拒绝** | **拒绝** |

GRPO 的组内 advantage 依赖同一任务多条采样之间存在回报差异。当一个组的 4 条轨迹
全部失败且回报相同时，reward std 和梯度都为 0；即便有回报差异，过多轨迹在长度上限
处截断，也会使终局奖励稀疏或带噪。这正是本轮失败的主要机制解释。

## 5. 奖励与防投机设计

reward v0.3 使用硬终局门控：成功轨迹为 `+1`，失败轨迹为 `-1`；过程 shaping 总幅度
限制在 `±0.1`。缺失必要澄清、最终状态错误、禁止副作用、重复失败或超出执行上限都不能
靠过程分累加成正回报。主评分永远从环境最终状态和轨迹事件重算，不采信 Agent 自报
“已完成”。

这是对早期 v0.7 中 reward hacking 的修复：早期过程奖励曾允许失败轨迹通过重复合法动作
得到较高回报。历史失败产物被保留，修复后的奖励有单元测试和轨迹重放证据。

## 6. 证据链与停止规则

`verify-rl-evidence` 会：

1. 校验训练配置、训练集、依赖锁、初始/最终 Adapter 和源码身份哈希；
2. 校验冻结 Dev 文件与分层选择清单，拒绝顺序变化、未知或重复任务；
3. 从原始 `final_state/events` 重算每题 grade，再重算 report 和 comparison；
4. 校验候选运行实际加载了对应 GRPO Adapter；
5. 分开输出 artifact/canonical、learning-signal 和 promotion Gate。

本轮两份 evidence 均为：产物可验证、源码规范 Gate 通过，但学习信号和晋级 Gate 拒绝。
按照预先定义的停止规则，不为追求好看的数字更换 seed，也不在失败的 smoke 上继续堆更新。

## 7. v1.1：短轨迹 + 640-token 复现实验

在 v1.0 拒绝后，先用 Oracle 实际执行轨迹冻结 `train-short-v0.1`：8 题覆盖三类工作流、
显式/inspect、澄清与触发故障，全部能在不超过 5 次 Agent 决策内完成。两条路线保持
seed、奖励、LoRA、采样和 10-step 环境上限不变，仅将 completion budget 从 320 提升至
640，并把更新预算设为 8（每个冻结 prompt 一轮）。训练同时保存 32 条完整 rollout，
绑定 task seed、最终状态、事件、reward breakdown、config/data/selection/source 哈希。

| 指标 | Warm short640 | Direct short640 |
|---|---:|---:|
| 优化器步数 | 8/8 | 8/8 |
| 运行时间 | 491.12 s | 498.99 s |
| 训练 rollout 严格成功 | 2/32（6.25%） | 11/32（34.375%） |
| 正常终止并调用终止工具 | 32/32 | 32/32 |
| 非零 reward std / 梯度更新 | 6/8（75%） | 2/8（25%） |
| TRL completion clipped 均值 | 53.125% | 65.625% |
| PyTorch allocated 峰值 | 4,908.00 MiB | 5,212.93 MiB |
| 整卡采样占用峰值 | 7,322.5 MiB | 7,682.5 MiB |
| 7,700 MiB 显存 Gate | 通过 | 通过（余量仅 17.5 MiB） |
| 90% learning-signal Gate | **拒绝** | **拒绝** |
| Dev8 | 未运行 | 未运行 |

v1.1 修正了一个旧解释：320-token 时模型确实未执行到终止工具；640-token 后 64/64 条
采样都正常调用了终止工具，因此环境 10 步不是主阻塞。但 TRL 的 token-level truncated
mask 仍然很高。更重要的是，两种初始化产生相反的组退化：warm 多数轨迹失败，direct
在部分任务上 4 条采样全部成功。全失败和全成功都会令组内 reward std 为零，不能仅凭
direct 的训练 rollout 成功率更高就声称它泛化更好。

两条候选均未通过预注册 signal Gate，因此没有运行 Dev8、Dev32 或 Test64。下一项合理
在 v1.1 收口当时，下一项合理实验是**难度匹配的 dynamic sampling**：按最近组成功比例
优先抽取既非全失败、也非全成功的 prompt，并对全同回报组做有界重采；必须记录额外采样
成本和原始任务分布。由于 direct 640 已距离显存护栏仅 17.5 MiB，不再增加 context/group
size。该方案在 v1.1 时尚未实现；后续 v1.4 已完成其中的 mixed-first 有界重采样，结果见下一节，
但仍未通过 Gate。

当前代码中的 turn-level credit 是独立、已测试的研究 seam，**尚未接入本轮 GRPO loss**，
因此本报告不声称已经完成逐轮信用分配训练。

## 8. v1.2–v1.4：终止语义修复与有界重采样

独立终审发现，v1.1 的 TRL `clipped_ratio` 不能解释为真实长度截断：64/64 条环境轨迹
均已调用 `finish_task`，但 TRL 仅检查末 token 是否 EOS/PAD，且不读取工具响应中的
`done=true`。v1.2 先改 loss mask，证明协议终止轨迹可保留；同时也暴露出 trainer 会在
任务完成后继续生成重复终止调用，因此 v1.2 只作为诊断轮，不作为算法结论。

v1.3 在 post-tool generation 边界识别终止工具响应，丢弃已经生成但无效的下一
assistant turn，并阻止后续重复 tool loop；它并没有省掉这一次探测性 generation。
32/32 rollout 协议终止，真实长度截断 0/32，训练运行时由
v1.2 的 538.17 s 降至 289.49 s（-46.2%）；非零梯度仍为 6/8，说明剩余瓶颈是组内
零方差而非长度或终止集成。

v1.4 预先冻结 mixed-first 有界重采样：每个 update 最多尝试 3 个不同任务 group，优先
接受同时含成功/失败的 terminal-mixed group；否则接受 reward std≥0.05 的首个 shaping
group；所有 rejected group 也落盘并计入预算。结果如下：

| 指标 | Warm v1.3 | Warm v1.4 bounded resampling |
|---|---:|---:|
| 优化器更新 | 8 | 8 |
| 生成 rollout / group | 32 / 8 | 84 / 21（上限 96 / 24） |
| accepted mixed / shaping / zero-variance | 3 / 0 / 5¹ | 4 / 3 / 1 |
| accepted 非零 reward std / 梯度 | 6/8（75%）² | 7/8（87.5%） |
| 真实长度截断 | 0/32 | 0/84 |
| 运行时间 | 289.49 s | 678.71 s |
| 整卡峰值 | 7,436.5 MiB | 7,630.5 MiB |
| 90% signal Gate | 拒绝 | **拒绝** |
| Dev8 / Test64 | 未运行 | 未运行 |

¹ v1.4 的 shaping 阈值为 reward population std≥0.05；按同一阈值回算，v1.3 的 3 个
非 mixed 非零梯度组均低于 0.05，故归入 zero-variance/低方差桶。² 梯度非零仍按精确
`reward_std>0` 计数，因此是 6/8。

84 条 proposal 中成功 4 条，且全部被筛入 accepted（4/32；rejected 0/52）。因此
v1.4 的 7/8 是 mixed-first 结果条件化选择后的训练组利用率，**不能**与 v1.3 的自然
6/8 直接包装成策略、成功率或 RL 效果提升。代价从 32 条增到 84 条 rollout、运行时间
2.34 倍，只多得到 1 个非零梯度 accepted update 和 1 个 mixed group。

本次锁定的 TRL 1.9.2 原生 `clipped_ratio` 在 v1.3/v1.4 仍接近 100%，因为它实际是
“末 token 非 EOS/PAD”
启发式，不是环境截断率；证据层将其改名为 `mean_token_no_eos_ratio`。轨迹审计显示真实
长度截断 0/84、loss mask 0/84。历史 v1.4 trainer 的 reward/reward_std/loss 对齐 accepted
组，但 completion/tool 等辅助均值混入 rejected proposal；自动证据因此不拿这些均值作
结论，并分别从审计重算 proposal/accepted/rejected 成本。训练器已修复为只回写 accepted
组的 per-group 辅助均值；累计 `num_tokens` 仍如实记录全部 proposal 生成成本。修复在后续
运行生效，本轮不为追数重跑。v1.4 仍未达到预注册阈值，不能因为只差
1 个 update 就临时将 90% 改为 87.5%。

## 9. 可复核边界

- Dev8 只有 8 题，单题等于 12.5 个百分点；它只用于 smoke Gate，不支持统计显著性结论。
- 两条 GRPO 各只有一个训练 seed 和 4 个更新，不能支持稳定性或算法优越性结论。
- 环境是合成设备状态机，不代表物理设备、声学 ASR 或生产可靠性。
- Test64 未运行，任何 RL 泛化提升都不能写进简历。
- 可以如实陈述：在 8GB GPU 上完成可重放的多轮 Agent RL 系统、结构留出基准、
  计算匹配对照、奖励投机审计与自动停止门禁。
