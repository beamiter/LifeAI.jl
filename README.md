# LifeAI.jl

> 构建有生命感的 AGI，并让它走进真实世界。

LifeAI.jl 是一个长期演进的 Julia/Lux 研究与工程项目。项目不止于实现一个语言模型，而是希望逐步构建能够持续感知、记忆、学习、决策和行动的智能系统，并将这些能力用于机器狗、桌面机器人、具身机器人以及其他可能的智能载体。

这里也是一份公开的学习与构建记录：从关键模型组件开始，理解原理、亲手实现、测试验证、分析性能，再逐步组合成更完整的智能体。

## 项目宗旨

LifeAI.jl 沿四条相互连接的主线持续积累：

1. **模型基础组件**：持续学习和实现 Attention、位置编码、Transformer、训练、推理加速等可复用能力。
2. **智能体核心**：逐步加入记忆、规划、工具使用、反思、多模态理解和持续学习能力。
3. **具身闭环**：让模型通过统一接口连接感知与行动，先在可验证的环境中运行，再走向机器狗、桌面机器人和其他实体设备。
4. **学习过程记录**：用 weekly plan 推动小步交付，用月度总结沉淀结果、实验、失败和认知变化。

这里所说的“有生命感”，不是只让模型表现得像某种人格，而是让系统在长期互动中体现出连续性、状态、记忆、主动性、适应性以及与环境真实连接的行动能力。

## 当前状态

**阶段判断：最小 GPT 的训练、生成、评估与版本化 Tokenizer 已形成；Qwen3 dense 结构、HuggingFace 权重/tokenizer 加载及 Qwen3-0.6B text→text / KV Cache 数值一致性均已验证，智能体与具身层尚未开始。**

Week 01—08 均已 Closed，当前没有 Open 的 Week。[`Week 08 — HuggingFace Qwen3 Tokenizer and Text-to-Text Parity`](notes/week08_hf_tokenizer_text_parity.md) 已完成严格 Qwen3 tokenizer、基础 chat template，以及 text→greedy generation→text parity；Week 06—08 的结构→权重/logits→tokenizer/text 三阶段复现现已闭环。

目前已经具备：

- 手写与批量 scaled dot-product attention、因果遮罩和 Multi-Head Attention。
- 支持 legacy interleaved 与 HF rotate_half 配对的 RoPE、pre-norm TransformerBlock 和 decoder-only GPTModel。
- 可独立切换的 LayerNorm / RMSNorm、GELU / SwiGLU、untied / tied embedding-output projection；legacy 默认保持不变。
- 字符级 Tokenizer、DatasetLoader、next-token loss 和训练循环。
- 无泄漏 train / validation 划分、token-weighted evaluation / perplexity 和 global gradient norm clipping。
- checkpoint v2、设备无关保存/加载、确定性断点续训和 v1 legacy checkpoint 迁移。
- 基于 Zygote 的常规训练，以及 Reactant/Enzyme 驱动的 XLA 训练路径。
- greedy、temperature、top-k 文本生成。
- 动态 KV Cache 的 prefill / decode，以及面向 XLA 的固定形状 KV Cache 和编译后增量解码。
- full / dynamic / static KV Cache correctness matrix，以及 CPU、CUDA GPU、XLA CPU、XLA GPU 四后端 benchmark。
- 严格的 Qwen3 dense config 校验、BF16/F32 safetensors 单文件/分片读取、HF 参数映射与显式 0-based token-id 边界转换。
- Qwen3-0.6B 逐层 hidden states、full logits、dynamic/static cache decode 的真实 HF reference parity；真实权重测试显式 opt-in，默认测试保持离线。
- 严格的 Qwen3 HF tokenizer：NFC、目标 regex、ByteLevel、imported BPE、added/special tokens、artifact/checkpoint 与 provenance fingerprint。
- 无 tools 的 Qwen3 基础 chat template，以及 full/dynamic/static/XLA 的真实 greedy text-generation parity。
- 围绕 Attention、RoPE、prefill/decode 和 KV Cache 的 Pluto 可视化学习笔记。
- 默认测试套件全部通过；Reactant/XLA 专项测试需显式启用。

尚未具备：

- 通用 Jinja、tools/tool-role chat template 与 agent tool loop；可用于真实任务的模型质量仍未评估。
- 长短期记忆、规划、工具使用、反思等完整的 agent loop。
- 视觉、听觉和传感器输入等多模态感知。
- 面向仿真或实体机器人的 observation / action 抽象、控制链路与安全边界。
- 在线或持续学习机制。

更详细的能力盘点、验证范围与建议里程碑见 [`notes/current_status.md`](notes/current_status.md)。

## 演进路线

```text
模型基本组件
    ↓
可训练、可生成、可评估的模型闭环
    ↓
记忆 + 规划 + 工具 + 多模态的智能体核心
    ↓
感知 → 决策 → 行动 → 反馈的具身闭环
    ↓
能长期互动、持续适应的“有生命感”AGI
```

模型组件不会在进入下一阶段后停止建设；它会始终作为底层主线，与智能体和具身实验互相驱动。

## 记录节奏

- **Week plan**：Week 是目标驱动的逻辑迭代，不与自然周绑定。每个 Week 先定义核心问题、交付物和 Close 条件；条件满足后立即复盘关闭，并开启下一个 Week。
- **月度总结**：汇总一段时间内多个 Week 的能力变化、验证证据、关键学习、失败尝试、架构决策和下一阶段重点。
- **状态快照**：每次 Week Close 后按需更新，只描述仓库此刻真实具备的能力，明确区分“已实现”“已验证”和“尚未开始”。

```text
Open Week → 执行与验证 → 满足 Close 条件 → 复盘并更新状态 → Open Next Week
```

记录索引与模板见 [`notes/README.md`](notes/README.md)。

## 仓库结构

```text
src/
├── core/          # Attention、RoPE、Transformer、sampling
├── data/          # Tokenizer 与 DatasetLoader
├── io/            # HuggingFace config / safetensors 权重加载
├── models/        # GPT 模型
├── train/         # Zygote / Reactant-XLA 训练
└── generation/    # 文本生成、动态与固定形状 KV Cache

test/              # 默认测试与可选 XLA 测试
examples/          # 最小训练和生成示例
notebook/          # 可交互的原理与实验记录
notes/             # 当前状态、weekly plan 与月度总结
```

## 开始使用

安装依赖：

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

运行默认测试：

```bash
julia --project=. -e 'using Pkg; Pkg.test()'
```

在具备对应 XLA 后端的环境中运行专项测试：

```bash
LIFEAI_TEST_XLA=true julia --project=. -e 'using Pkg; Pkg.test()'
```

运行字符级 GPT 训练与生成示例（默认使用 Reactant/XLA GPU）：

```bash
julia --project=. examples/minigpt.jl
```

运行 RMSNorm + SwiGLU + tied embedding 的可恢复训练示例：

```bash
julia --project=. examples/modern_gpt.jl
```

也可以直接构建可独立开关的 modern 配置：

```julia
model = GPTModel(
    vocab_size,
    d_model,
    num_heads,
    num_layers;
    norm_type=:rmsnorm,
    mlp_type=:swiglu,
    tie_embeddings=true,
)
```

没有 NVIDIA GPU 时可以尝试 XLA CPU 后端：

```bash
LIFEAI_XLA_BACKEND=cpu julia --project=. examples/minigpt.jl
```

对比 CPU、CUDA GPU、XLA+CPU 和 XLA+GPU 的训练与 KV Cache 推理性能：

```bash
./scripts/benchmark_week03.sh
```

脚本会分别记录首编译/首次执行、post-compile warm-up 和 steady-state 指标，并保留逐 iteration 原始耗时，生成 TSV 原始数据与 Markdown 汇总。默认使用 3 个 warm-up step 和 30 个正式样本，避免一次性 runtime settling 扭曲 p90。配置项和指标口径见 [`Week 03 四后端性能对比`](notes/week03_reproducible_training.md#四后端性能对比)。

其中 XLA+GPU 会额外对比 no-cache、dynamic KV Cache 和 static KV Cache，并报告不同 shape 导致的 executable 数量与 cold compilation 总成本。

运行 Week 04 的五配置 CPU 对照，以及 baseline / modern 四后端 benchmark：

```bash
./scripts/benchmark_week04.sh
```

该脚本使用三个固定 seed 汇总短程 validation 结果，并将 cold compile、warm-up 与 steady-state 性能分开记录。实验用于验证组件可归因性和后端兼容性，不代表真实模型质量排名。
