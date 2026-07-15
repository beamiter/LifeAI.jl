# LifeAI.jl 当前状态

## 一句话判断

项目已经从 Attention 原理学习推进到一个可训练、可生成并支持 KV Cache / XLA 路径的最小 decoder-only GPT；当前仍处于模型基础设施阶段，尚未形成 agent loop、multimodal perception 或 embodied control loop。

## 当前活动阶段

[`Week 03 — Reproducible Training and Evaluation`](week03_reproducible_training.md) 已 Open，目标是让现有小 GPT 实验可保存、可恢复、可评估、可比较。模型结构、Tokenizer 与真实中文语料实验在该基线建立后分阶段推进。

## 已实现能力

### 1. 模型基本组件

- scaled dot-product attention：同时保留手写版本与基于 `NNlib.batched_mul` 的批量版本，便于原理对照和正确性验证。
- Multi-Head Attention：包括 Q/K/V/O 投影、head reshape / merge 和 causal mask。
- RoPE：支持预计算 cos / sin cache 和增量解码所需的绝对起始位置。
- TransformerBlock：采用 pre-norm、attention residual 和 MLP residual 结构。
- GPTModel：包括 token embedding、多层 TransformerBlock、final norm 和 LM head。

### 2. 数据与训练

- 字符级 Tokenizer，支持编码、解码和未知字符处理。
- 滑动窗口 DatasetLoader，支持 batch、stride 和 `drop_last`；当前按确定性顺序迭代，尚未实现 shuffle。
- 稀疏 next-token cross entropy，不需要构造 dense one-hot target。
- Zygote 常规训练路径。
- Reactant + Enzyme 的 XLA 训练路径，支持稳定 shape 检查和编译复用。

### 3. 生成与推理

- greedy、temperature 和 top-k sampling。
- 普通自回归文本生成。
- 动态 KV Cache，支持 prompt prefill、单 token decode 和 cached generation。
- 固定容量、固定 shape 的 KV Cache，为编译后的增量推理保持执行形状稳定。
- XLA prefill / decode 接口及编译后生成流程。

### 4. 学习与可视化记录

`notebook/` 已覆盖以下主题：

- Attention 的结构与计算过程。
- RoPE 的位置与频率含义。
- prefill 与 decode 的区别。
- KV Cache 为什么有效及其增长过程。
- causal mask、绝对位置和增量解码中的常见错误。
- 动态 cache 与适配 Reactant/XLA 的静态 cache。

这些 notebook 不只是展示结果，也是关键组件学习过程的一部分。

## 验证状态

运行默认测试套件：

```bash
julia --project=. -e 'using Pkg; Pkg.test()'
```

默认套件共 597 项测试通过，验证了 Attention、RoPE、TransformerBlock、GPT、Tokenizer、DatasetLoader、训练/生成和动态 KV Cache。`test/test_xla_kv_cache.jl` 只有设置 `LIFEAI_TEST_XLA=true` 时才会运行，因此默认测试通过不等价于当前机器上已验证全部 XLA 后端。

## 当前边界

以下能力尚未实现，不应从现有 GPT demo 推断为已经具备：

- 面向真实任务和长期运行的模型质量。
- checkpoint 管理、可重复实验配置、系统化评估与性能基线。
- 对话状态、工作记忆、长期记忆和记忆检索。
- 任务规划、工具调用、反思和自主执行循环。
- 图像、音频、空间状态或机器人传感器输入。
- 动作空间、控制器、仿真环境与真实设备适配器。
- 机器人运行所需的实时性、容错和物理安全机制。
- 在线学习、持续学习与个体长期成长。

## 建议的近期里程碑

### Milestone A：建立可恢复、可评估、可比较的实验基线

- 建立版本化 checkpoint、加载与断点续训。
- 完成无泄漏的 train / validation 划分、perplexity 和 global gradient norm clipping。
- 对 full forward、动态 KV Cache 和静态 XLA KV Cache 做一致性与性能基线。

完成标准：同一个可复现实验能够完成训练、验证、保存、加载、恢复、生成与推理性能比较。

### Milestone B：推进模型组件、Tokenizer 与中文训练

- 以独立开关和对照实验加入 RMSNorm、SwiGLU、embedding / lm_head 权重共享。
- 建立 byte-level baseline，再实现并评估 BPE。
- 实现 GQA，并复用 KV Cache correctness / benchmark 验证 cache 布局和 decode 收益。
- 建立来源、清洗、切分和配置可追踪的小型中文语料训练流程。

完成标准：每项结构变化都能与固定 baseline 独立比较，并在版本化 Tokenizer 和中文数据上完成可恢复训练与 validation evaluation。

### Milestone C：建立最小有状态智能体闭环

- 定义与具体机器人无关的 `Observation`、`Action`、`Memory` 和 policy / model 接口。
- 先在一个简单、可重复的模拟环境中跑通“感知 → 记忆 → 决策 → 行动 → 反馈”。
- 保持模型后端可替换，使当前小 GPT、外部模型或后续多模态模型都能接入。

完成标准：智能体可以跨多个 step 保持状态，根据环境反馈改变下一步动作，并用测试或 replay 重现一次完整轨迹。

## 长期能力地图

| 主线 | 当前状态 | 下一关键缺口 |
| --- | --- | --- |
| 模型基本组件 | 已形成最小 GPT 闭环 | checkpoint、evaluation、benchmark、更多架构实验 |
| 高效训练与推理 | 已有 Zygote / XLA 与两类 KV Cache | 后端实测、性能基线、稳定接口 |
| 智能体核心 | 尚未开始 | memory、planning、tools、agent loop |
| 多模态感知 | 尚未开始 | vision / audio / sensor representation |
| 具身闭环 | 尚未开始 | observation/action abstraction、simulation、device adapter |
| 持续学习与生命感 | 处于愿景阶段 | 长期状态、适应、主动性与安全边界 |
| 学习记录 | 已有 Week 01 与多份 notebook | 稳定执行 weekly plan 和月度总结 |
