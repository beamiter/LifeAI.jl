# LifeAI.jl 当前状态

## 一句话判断

项目已经形成一个可训练、可生成、可保存恢复、可评估比较，支持现代组件、KV Cache / XLA 路径，并具备版本化 Tokenizer（character / byte / byte-BPE）与文档级无泄漏中文数据管线的最小 decoder-only GPT；它已能严格加载 HuggingFace Qwen3 dense config 与 BF16 safetensors，并让 Qwen3-0.6B 的逐层 hidden states、logits 和两类 KV Cache decode 与 HF reference 数值对齐，尚未形成 agent loop、multimodal perception 或 embodied control loop。

## 当前活动阶段

[`Week 07 — HuggingFace Weight Loading and Qwen3 Logits Parity`](week07_hf_weight_loading.md) 已于 2026-07-22 Closed，完成「复现 Qwen3 → HF 权重加载 → 推理验证」三阶段计划的第二步：safetensors / BF16 权重加载、HF `config.json` 解析、RoPE rotate_half 适配，以及 Qwen3-0.6B 逐层 hidden states / logits / KV-cache 数值对齐均已验证。当前没有 Open 的 Week；下一阶段计划为 Week 08，导入 HF tokenizer 并完成 text→text 端到端验证。

## 已实现能力

### 1. 模型基本组件

- scaled dot-product attention：同时保留手写版本与基于 `NNlib.batched_mul` 的批量版本（均支持 GQA/MQA 分组），便于原理对照和正确性验证。
- Multi-Head Attention：包括 Q/K/V/O 投影、head reshape / merge 和 causal mask；`head_dim` 可独立于 `d_model` 配置，`use_bias` 可关闭。
- GQA / MQA：`num_kv_heads` 独立可配，K/V 投影与 KV cache 按 KV head 数缩减；manual reference、无物化分组实现与 `repeat_kv` 展开三路等价性已测试钉死。
- QK-Norm（Qwen3 语义）：per-head RMSNorm、独立 q/k scale、位于 head reshape 之后 RoPE 之前，独立开关，关闭时参数树与 legacy 完全一致。
- RoPE：支持预计算 cos / sin cache、可配置 `rope_theta` 和增量解码所需的绝对起始位置；同时支持 legacy `:interleaved` 与 HF Qwen3 `:rotate_half` 配对。
- TransformerBlock：采用 pre-norm、attention residual 和 MLP residual，可独立选择 LayerNorm / RMSNorm 与 GELU / SwiGLU。
- GPTModel：包括 token embedding、多层 TransformerBlock、final norm 和 LM head；支持 embedding / LM head 单 kernel 权重共享。
- legacy 默认仍为 LayerNorm + GELU + untied；modern 配置可通过独立开关组合，不改变旧调用。
- HuggingFace Qwen3 dense 导入：严格解析 config，读取 BF16/F32 safetensors 单文件或 index 分片，完整映射 embedding、attention、QK-Norm、MLP、final norm 与 tied/untied LM head；missing、unexpected、duplicate、shape/dtype/config 错误均 fail closed。
- 显式 `hf_token_ids` 处理 HF 0-based 到 LifeAI 1-based 边界；逐层 trace 与 reference 脚本可验证 embedding、每个 block、final hidden、full logits 和 cache decode logits。

### 2. Tokenizer 与数据

- `AbstractTokenizer` 统一接口：character、byte、byte-BPE 共用 encode / decode / vocab / special-token / fingerprint API，token id 保持 1-based。
- legacy character `Tokenizer` 完整保留，旧调用与旧 checkpoint 不受影响。
- `ByteTokenizer`：对任意有效 UTF-8 无 OOV、可精确 round-trip；`decode_bytes` 始终可逆，`decode` 提供显式 `:strict` / `:replace` 策略。
- `ByteBPETokenizer`：train-only 确定性训练，固定 tie-break，相同语料与配置产生相同 vocabulary、merge ranks 和 fingerprint。
- Tokenizer artifact v1：显式 schema version、normalization、special tokens、vocabulary / merges 与内容指纹，可独立保存、加载与校验，篡改被拒绝。
- 中文数据管线：以 document 为单位记录来源、许可、checksum、变换配置；确定性文档级无泄漏 split；Tokenizer 只在 train split 上拟合；versioned dataset artifact 与显式 EOS 边界语义。
- 滑动窗口 DatasetLoader 与 DocumentDatasetLoader，支持 batch、stride 和 `drop_last`。
- 稀疏 next-token cross entropy；token-weighted validation loss、perplexity 与 `bits_per_byte` 等 byte-normalized 评估。
- checkpoint format v2：版本化、设备无关，支持全部三类 Tokenizer 的保存恢复，并显式迁移 v1 legacy checkpoint。
- 确定性 checkpoint resume、可配置 global gradient norm clipping、Zygote 常规训练路径与 Reactant + Enzyme 的 XLA 训练路径。

### 3. 生成与推理

- greedy、temperature 和 top-k sampling；生成入口对三类 Tokenizer 通用。
- 动态 KV Cache（prompt prefill、单 token decode、cached generation）与固定形状静态 KV Cache（面向编译后增量推理）。
- XLA prefill / decode 接口及编译后生成流程。
- full forward、动态 KV Cache、静态 KV Cache 的 correctness matrix 与 microbenchmark。
- CPU、CUDA GPU、XLA CPU、XLA GPU 独立进程 benchmark，可区分 cold compile、warm-up 和 steady-state。

### 4. 学习与可视化记录

`notebook/` 已覆盖 Attention 结构、RoPE、prefill / decode、KV Cache 原理与常见错误、动态与静态 cache 等主题；这些 notebook 不只是展示结果，也是关键组件学习过程的一部分。

## 验证状态

运行默认测试套件：

```bash
julia --project=. -e 'using Pkg; Pkg.test()'
```

2026-07-22 复核默认套件，共 4025 项测试通过；其中 Week 05 专项 3094 项、Week 06 专项 112 项、Week 07 离线专项 54 项。显式设置 `LIFEAI_TEST_XLA=true` 后，默认套件再次通过，Reactant/XLA 套件 49 / 49 通过，其中新增 rotate_half Qwen3 full/decode smoke 3 项。

使用 `Qwen/Qwen3-0.6B` revision `c1899de289a04d12100db370d81485cdf75e47ca` 的真实 BF16 权重和 Transformers Float32 reference，opt-in integration 35 / 35 通过。final hidden max-abs 为 `7.43866e-5`，full logits 为 `5.67436e-5`，dynamic/static decode logits 均为 `4.48227e-5`，下一 token argmax 全部一致。详细版本、容差、逐层误差、checksum 与内存记录见 Week 07 文档。

Week 05 三 seed（20260720–22）跨 tokenizer 对照记录于 `benchmark_results/week05/`：character / byte / byte_bpe 的 tokens per byte 为 0.3717 / 1.0000 / 0.7139，final BPB 3.0753 / 8.1890 / 6.7614；byte 与 byte-BPE 对 unseen UTF-8 lossless 且 validation unknown 率为 0，character 为 19.6%（其 BPB 不可与 lossless tokenizer 直接排名）。

Week 06 GQA benchmark（CPU）记录于 `benchmark_results/week06/`：固定形状下 KV cache 内存严格按 `num_kv_heads / num_heads` 缩减（8 / 4 / 1 heads 对应 1024 / 512 / 128 KiB），dynamic decode 吞吐 2097 / 2484 / 3131 tok/s，全部配置 correctness 为 true。默认测试、XLA 专项测试和硬件 benchmark 仍是三类不同证据。

## 当前边界

以下能力尚未实现，不应从现有 GPT demo 或已完成的结构 parity 推断为已经具备：

- HF `tokenizer.json` 导入、GPT-2 byte↔unicode 映射与 regex pre-tokenization（Week 08 计划）。
- 面向真实任务和长期运行的模型质量；较大规模真实语料训练。
- 适合 tied embedding 的统一初始化基线、低精度专项与真实规模组件对照。
- 实验注册、超参数搜索、分布式训练和面向生产的性能评估。
- 对话状态、工作记忆、长期记忆和记忆检索。
- 任务规划、工具调用、反思和自主执行循环。
- 图像、音频、空间状态或机器人传感器输入。
- 动作空间、控制器、仿真环境与真实设备适配器。
- 机器人运行所需的实时性、容错和物理安全机制。
- 在线学习、持续学习与个体长期成长。

## 建议的近期里程碑

### Milestone A：建立可恢复、可评估、可比较的实验基线（已完成）

完成记录：Week 03 已于 2026-07-18 Closed；默认测试 654 / 654 通过，四后端基线均完成 correctness 与性能记录。

### Milestone B：推进模型组件、Tokenizer 与中文训练（已完成主体）

- RMSNorm、SwiGLU、embedding / lm_head 权重共享独立开关与对照实验。（Week 04 已完成，2026-07-19 Closed）
- 无 OOV、完全可逆的 byte-level baseline，deterministic byte-BPE、版本化 Tokenizer artifact 与 fingerprint。（Week 05 已完成，2026-07-21 Closed）
- 来源、许可、checksum、文档级切分可追踪的中文语料训练流程与 bits-per-byte 评估口径。（Week 05 已完成）

### Milestone B'：复现 Qwen3 并以 HF 权重验证（进行中）

- 实现 GQA 与 QK-Norm，使模型结构与 Qwen3 dense 同构；复用 KV Cache correctness / benchmark 验证 cache 布局与 decode 收益。（Week 06 已完成，2026-07-22 Closed）
- 解决 RoPE rotate_half 适配，实现 safetensors / bfloat16 权重加载、HF `config.json` 解析与参数名映射；用 token-id fixture 对齐 Qwen3-0.6B 逐层 hidden states、logits 与 KV Cache decode。（Week 07 已完成，2026-07-22 Closed）
- 导入 HF `tokenizer.json`（byte-level BPE、byte↔unicode 映射、regex pre-tokenization、special tokens），完成 text→text 端到端一致性验证。（Week 08 计划）

阶段性完成标准：权重加载与 token-id→logits / KV Cache parity 已由 Week 07 达成；Milestone B' 待 Week 08 完成 tokenizer 与 text→text 端到端一致性后整体结束。

### Milestone C：建立最小有状态智能体闭环

- 定义与具体机器人无关的 `Observation`、`Action`、`Memory` 和 policy / model 接口。
- 先在一个简单、可重复的模拟环境中跑通"感知 → 记忆 → 决策 → 行动 → 反馈"。
- 保持模型后端可替换，使当前小 GPT、Qwen3 复现权重或后续多模态模型都能接入。

完成标准：智能体可以跨多个 step 保持状态，根据环境反馈改变下一步动作，并用测试或 replay 重现一次完整轨迹。

## 长期能力地图

| 主线 | 当前状态 | 下一关键缺口 |
| --- | --- | --- |
| 模型基本组件 | Qwen3 dense 同构结构；HF config / safetensors 权重加载；Qwen3-0.6B logits parity；三类版本化 Tokenizer与中文数据管线 | Week 08 HF tokenizer 与 text→text 对齐 |
| 高效训练与推理 | modern / GQA / rotate_half 已兼容 Zygote / XLA 与两类 KV Cache；Qwen3-0.6B decode 已真实验证 | 低精度计算、更多真实规模与硬件性能验证 |
| 智能体核心 | 尚未开始 | memory、planning、tools、agent loop |
| 多模态感知 | 尚未开始 | vision / audio / sensor representation |
| 具身闭环 | 尚未开始 | observation/action abstraction、simulation、device adapter |
| 持续学习与生命感 | 处于愿景阶段 | 长期状态、适应、主动性与安全边界 |
| 学习记录 | Week 01—07 已 Closed；当前无 Open Week | 规划并 Open Week 08：HF tokenizer 与 text→text parity |
