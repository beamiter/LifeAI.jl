# LifeAI.jl 当前状态

## 一句话判断

项目已经形成一个可训练、可生成、可保存恢复、可评估比较，支持现代组件、KV Cache / XLA 路径，并具备版本化 Tokenizer 与文档级无泄漏中文数据管线的最小 decoder-only GPT；它已能严格加载 HuggingFace Qwen3 dense config、BF16 safetensors 与 Qwen3 tokenizer，使 Qwen3-0.6B 从 text→ids、逐层 logits、full/dynamic/static greedy generation 到最终文本均与 HF reference 对齐，尚未形成 agent loop、multimodal perception 或 embodied control loop。

## 当前活动阶段

[`Week 08 — HuggingFace Qwen3 Tokenizer and Text-to-Text Parity`](week08_hf_tokenizer_text_parity.md) 已于 2026-07-22 Closed，完成「复现 Qwen3 → HF 权重加载 → 推理验证」三阶段计划的最后一步：Qwen3 tokenizer、基础 chat template、text→ids→greedy generation→text 已与 Transformers 对齐。Week 06—08 的结构、权重/logits 与 tokenizer/text 三阶段现已闭环；当前没有 Open 的 Week。

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

- `AbstractTokenizer` 统一接口：character、byte、byte-BPE 与 imported HF Qwen3 共用 encode / decode / vocab / special-token / fingerprint API，token id 保持 1-based。
- legacy character `Tokenizer` 完整保留，旧调用与旧 checkpoint 不受影响。
- `ByteTokenizer`：对任意有效 UTF-8 无 OOV、可精确 round-trip；`decode_bytes` 始终可逆，`decode` 提供显式 `:strict` / `:replace` 策略。
- `ByteBPETokenizer`：train-only 确定性训练，固定 tie-break，相同语料与配置产生相同 vocabulary、merge ranks 和 fingerprint。
- `HFQwen3Tokenizer`：严格导入目标 revision 的 NFC、regex、ByteLevel、151,643-token BPE、151,387 merges 与 26 个 added tokens；支持 HF character/Julia UTF-8 byte spans、byte-exact decode、special-token 语义和 1-based 公共 ids。
- Qwen3 基础 chat template：支持无 tools 的 system/user/assistant、generation prompt 与 thinking 开关；模板 hash 与三份 tokenizer config checksum 纳入 provenance/fingerprint，未知 revision fail closed。
- Tokenizer artifact v1：显式 schema version、normalization、special tokens、vocabulary / merges 与内容指纹，可独立保存、加载与校验，篡改被拒绝。
- 中文数据管线：以 document 为单位记录来源、许可、checksum、变换配置；确定性文档级无泄漏 split；Tokenizer 只在 train split 上拟合；versioned dataset artifact 与显式 EOS 边界语义。
- 滑动窗口 DatasetLoader 与 DocumentDatasetLoader，支持 batch、stride 和 `drop_last`。
- 稀疏 next-token cross entropy；token-weighted validation loss、perplexity 与 `bits_per_byte` 等 byte-normalized 评估。
- checkpoint format v2：版本化、设备无关，支持全部四类 Tokenizer 的保存恢复，并显式迁移 v1 legacy checkpoint。
- 确定性 checkpoint resume、可配置 global gradient norm clipping、Zygote 常规训练路径与 Reactant + Enzyme 的 XLA 训练路径。

### 3. 生成与推理

- greedy、temperature 和 top-k sampling；基础生成入口对全部四类 Tokenizer 通用。
- 动态 KV Cache（prompt prefill、单 token decode、cached generation）与固定形状静态 KV Cache（面向编译后增量推理）。
- XLA prefill / decode 接口及编译后生成流程。
- full forward、动态 KV Cache、静态 KV Cache 的 correctness matrix 与 microbenchmark。
- CPU、CUDA GPU、XLA CPU、XLA GPU 独立进程 benchmark，可区分 cold compile、warm-up 和 steady-state。
- `load_hf_qwen3_bundle` / `generate_hf_text` 串联本地模型、tokenizer、EOS 与 greedy trace；Qwen3-0.6B 的 full、dynamic、static 生成及 host-tokenizer→XLA static 路径已验证。

### 4. 学习与可视化记录

`notebook/` 已覆盖 Attention 结构、RoPE、prefill / decode、KV Cache 原理与常见错误、动态与静态 cache 等主题；这些 notebook 不只是展示结果，也是关键组件学习过程的一部分。

## 验证状态

运行默认测试套件：

```bash
julia --project=. -e 'using Pkg; Pkg.test()'
```

2026-07-22 复核默认套件，共 4086 项测试通过；其中 Week 05 专项 3094 项、Week 06 专项 112 项、Week 07 离线专项 54 项、Week 08 离线专项 61 项。显式设置 `LIFEAI_TEST_XLA=true` 后，默认套件再次通过，Reactant/XLA 套件 52 / 52 通过，其中 Week 08 新增 host tokenizer→XLA static generation 3 项。

使用 `Qwen/Qwen3-0.6B` revision `c1899de289a04d12100db370d81485cdf75e47ca` 的真实 BF16 权重和 Transformers Float32 reference，opt-in integration 35 / 35 通过。final hidden max-abs 为 `7.43866e-5`，full logits 为 `5.67436e-5`，dynamic/static decode logits 均为 `4.48227e-5`，下一 token argmax 全部一致。详细版本、容差、逐层误差、checksum 与内存记录见 Week 07 文档。

同一 revision 的 Week 08 真实 tokenizer/text integration 70 / 70 通过：6 组多语种/Unicode/代码/special-token corpus 和 4 组基础 chat 的 strings/spans/ids 完全一致；raw 与 chat greedy 共 6 step 的 token ids、停止位置和文本完全一致，global logits max-abs `5.054474e-5`。chat prompt 的 full/dynamic/static 输出均为 `"hello"` 并在相同 EOS 停止。

Week 05 三 seed（20260720–22）跨 tokenizer 对照记录于 `benchmark_results/week05/`：character / byte / byte_bpe 的 tokens per byte 为 0.3717 / 1.0000 / 0.7139，final BPB 3.0753 / 8.1890 / 6.7614；byte 与 byte-BPE 对 unseen UTF-8 lossless 且 validation unknown 率为 0，character 为 19.6%（其 BPB 不可与 lossless tokenizer 直接排名）。

Week 06 GQA benchmark（CPU）记录于 `benchmark_results/week06/`：固定形状下 KV cache 内存严格按 `num_kv_heads / num_heads` 缩减（8 / 4 / 1 heads 对应 1024 / 512 / 128 KiB），dynamic decode 吞吐 2097 / 2484 / 3131 tok/s，全部配置 correctness 为 true。默认测试、XLA 专项测试和硬件 benchmark 仍是三类不同证据。

## 当前边界

以下能力尚未实现，不应从现有 GPT demo 或已完成的结构 parity 推断为已经具备：

- 通用 Jinja chat template、Qwen3 tools/tool-role 分支、JSON schema 工具注入与 agent tool loop；Week 08 只完成已冻结的无 tools 基础 chat 子集。
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

### Milestone B'：复现 Qwen3 并以 HF 权重验证（已完成）

- 实现 GQA 与 QK-Norm，使模型结构与 Qwen3 dense 同构；复用 KV Cache correctness / benchmark 验证 cache 布局与 decode 收益。（Week 06 已完成，2026-07-22 Closed）
- 解决 RoPE rotate_half 适配，实现 safetensors / bfloat16 权重加载、HF `config.json` 解析与参数名映射；用 token-id fixture 对齐 Qwen3-0.6B 逐层 hidden states、logits 与 KV Cache decode。（Week 07 已完成，2026-07-22 Closed）
- 导入 HF `tokenizer.json`（byte-level BPE、byte↔unicode 映射、regex pre-tokenization、special tokens），完成基础 chat template 与 text→text 端到端一致性验证。（Week 08 已完成，2026-07-22 Closed）

完成标准已满足：LifeAI.jl 能从本地加载 Qwen3-0.6B 官方权重和同 revision tokenizer，在明确 Float32 容差下与 HF logits 对齐，并以 full / dynamic / static KV Cache 产生完全相同的 greedy token 序列和文本。

### Milestone C：建立最小有状态智能体闭环

- 定义与具体机器人无关的 `Observation`、`Action`、`Memory` 和 policy / model 接口。
- 先在一个简单、可重复的模拟环境中跑通"感知 → 记忆 → 决策 → 行动 → 反馈"。
- 保持模型后端可替换，使当前小 GPT、Qwen3 复现权重或后续多模态模型都能接入。

完成标准：智能体可以跨多个 step 保持状态，根据环境反馈改变下一步动作，并用测试或 replay 重现一次完整轨迹。

## 长期能力地图

| 主线 | 当前状态 | 下一关键缺口 |
| --- | --- | --- |
| 模型基本组件 | Qwen3 dense 同构结构；HF config / safetensors / tokenizer；Qwen3-0.6B text→text parity；四类版本化 Tokenizer 与中文数据管线 | Qwen3 复现阶段已完成；后续按真实任务补组件 |
| 高效训练与推理 | modern / GQA / rotate_half 已兼容 Zygote / XLA 与两类 KV Cache；Qwen3-0.6B decode 已真实验证 | 低精度计算、更多真实规模与硬件性能验证 |
| 智能体核心 | 尚未开始；Qwen3 基础 chat 输入已可作为模型后端 | conversation state、memory、planning、tools、agent loop |
| 多模态感知 | 尚未开始 | vision / audio / sensor representation |
| 具身闭环 | 尚未开始 | observation/action abstraction、simulation、device adapter |
| 持续学习与生命感 | 处于愿景阶段 | 长期状态、适应、主动性与安全边界 |
| 学习记录 | Week 01—08 已 Closed；当前无 Open Week | 复盘后规划下一阶段，候选为最小 agent loop 与工具接口 |
