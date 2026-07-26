# LifeAI.jl 当前状态

## 一句话判断

项目已经形成一个可训练、可生成、可保存恢复、可评估比较，支持现代组件、KV Cache / XLA 路径，并具备版本化 Tokenizer 与文档级无泄漏中文数据管线的 decoder-only GPT；Qwen3 0.6B—32B **六个官方 dense 尺寸全部完成真实权重逐层 parity**（0.6B—4B 全量、8B—32B 流式），并具备镜像 HF 语义的 **native BF16 混合精度推理路径**：0.6B—8B 与 HF BF16 逐层对齐且 16 步 greedy token 完全一致，8B 以 15.26 GiB BF16 树完成本机首个 >4B 全量驻留生成。

## 当前活动阶段

[`Week 16 — Qwen3 XLA BF16 Compiled Decode 与 INT8/INT4 量化`](week16_qwen3_xla_decode_quant.md) 已于 2026-07-26 Closed。XLA BF16 static-cache decode 编译完成：设备端 greedy 闭环 steady **246 tok/s**（eager 的 16.1 倍），16 步 greedy 与 HF BF16 全对。RTN 量化让 8B（INT8，8.22 GiB）与 14B（INT4，8.38 GiB）首次驻留 16.3 GiB GPU：8B token 级行为近乎无损（greedy 14/16，仅近平局翻转），14B INT4 一致率 4/16 如实冻结——无校准 INT4 的生成保真是明确的下一个边界。

[`Week 15 — Qwen3 BF16 CUDA / XLA Accelerated Inference`](week15_qwen3_bf16_accel.md) 已于 2026-07-26 Closed。设备通用向量化 BF16 路径（CPU 上与 Week 14 循环路径逐位相同）在 RTX 5080 上以原生 BF16 张量核运行：0.6B/1.7B/4B CUDA parity 与 16 步 greedy 全部与 HF BF16 一致，吞吐 15.3/14.1/8.1 tok/s（CPU 的 33—92 倍），VRAM ≤ 12.1 GiB；Reactant XLA BF16 编译 prefill 通过（编译 44.8 s、steady 1.36 ms）。推理验证主战场从 CPU 移至 CUDA/XLA；8B GPU 驻留超出 16.3 GiB VRAM 保持边界。

[`Week 14 — Qwen3 Native BF16 Mixed-Precision Compute`](week14_qwen3_bf16_compute.md) 已于 2026-07-26 Closed。独立 BF16 推理路径逐算子镜像 Transformers 4.51.0 混合精度语义；0.6B/1.7B/4B/8B 与 HF BF16 逐层对齐、argmax 零失配、16 步 greedy token 序列完全一致；参数常驻内存减半，8B 完成本机首个 >4B 全量驻留 greedy 生成（峰值 RSS 19.0 GiB）。既有 F32 / 流式 / XLA 路径零改动。

[`Week 13 — Qwen3 Streamed Loading and 8B/14B/32B Real-Weight Parity`](week13_qwen3_streamed_large_weights.md) 已于 2026-07-25 Closed。流式 / 逐层 safetensors 加载（与 in-memory 路径逐位一致）让 8B / 14B / 32B 在峰值 ≤ 8.9 GiB 内完成真实权重逐层 parity，untied LM head 获得真实证据；Qwen3 dense family 六尺寸的真实权重验证闭环就此完成。

[`Week 12 — Qwen3 Dense Family Real-Weight Parity`](week12_qwen3_dense_real_weights.md) 已于 2026-07-25 Closed。1.7B 与 4B 真实权重经分片 safetensors index 加载并在显式容差内完成逐层 hidden/logits/dynamic/static decode parity，loader 零改动一次通过；tied embedding 的三个官方尺寸（0.6B/1.7B/4B）全部实跑。8B/14B/32B 因 Float32 全量加载需 30.5—122.1 GiB、超出本机 30 GiB RAM，保持显式未验证边界。

## 已实现能力

### 1. 模型基本组件

- scaled dot-product attention：同时保留手写版本与基于 `NNlib.batched_mul` 的批量版本（均支持 GQA/MQA 分组），便于原理对照和正确性验证。
- Multi-Head Attention：包括 Q/K/V/O 投影、head reshape / merge 和 causal mask；`head_dim` 可独立于 `d_model` 配置，`use_bias` 可关闭。
- GQA / MQA：`num_kv_heads` 独立可配，K/V 投影与 KV cache 按 KV head 数缩减；manual reference、无物化分组实现与 `repeat_kv` 展开三路等价性已测试钉死。
- QK-Norm（Qwen3 语义）：per-head RMSNorm、独立 q/k scale、位于 head reshape 之后 RoPE 之前，独立开关，关闭时参数树与 legacy 完全一致。
- RoPE：支持预计算 cos / sin cache、可配置 `rope_theta` 和增量解码所需的绝对起始位置；同时支持 legacy `:interleaved` 与 HF Qwen3 `:rotate_half` 配对。
- learned absolute position：full、dynamic/static KV cache 与 XLA decode 共用显式绝对位置；位置表上限 fail closed。
- TransformerBlock：采用 pre-norm、attention residual 和 MLP residual，可独立选择 LayerNorm / RMSNorm 与 GELU / GELU-New / SwiGLU。
- GPTModel：包括 token/可选 position embedding、多层 TransformerBlock、final norm 和 LM head；支持 embedding / LM head 单 kernel 权重共享，并可分离 projection bias 与 LM-head bias。
- legacy 默认仍为 LayerNorm + GELU + untied；modern 配置可通过独立开关组合，不改变旧调用。
- HuggingFace Qwen3 dense 导入：冻结 0.6B / 1.7B / 4B / 8B / 14B / 32B 六个官方规格与 config checksum，可自动识别或显式要求 variant；严格解析 config，读取 BF16/F32 safetensors 单文件或 index 分片，完整映射 embedding、attention、QK-Norm、MLP、final norm 与 tied/untied LM head；missing、unexpected、duplicate、shape/dtype/config 错误均 fail closed。六个真实 checkpoint 全部实跑逐层 parity：0.6B—4B 全量加载，8B—32B 流式加载。
- 流式 safetensors 加载（Week 13）：`open_safetensors_reader` header-only 索引 + `read_safetensors_tensor` 按需读取 + `stream_hf_qwen3_forward` 逐层 trace 与 dynamic decode；embedding 按 token 行从磁盘 gather，单层参数映射与 in-memory loader 共用；与全量路径逐位一致（合成 fixture 与真实 0.6B 验证），32B 峰值 RSS 8.9 GiB。
- native BF16 混合精度推理（Week 14）：`load_hf_qwen3_model(...; weight_dtype=BFloat16)` 位保真加载（参数内存减半）+ `hf_qwen3_bf16_forward` 独立路径，逐算子镜像 HF 语义（RMSNorm/QK-Norm/softmax F32 归一化、RoPE 表 F32 转 BF16、线性 BF16 存储 + 分块 F32 累加 + BF16 舍入）；BF16 cached decode 与全量前向逐位等价；0.6B—8B 与 HF BF16 argmax 零失配、16 步 greedy 完全一致。
- BF16 CUDA/XLA 加速推理（Week 15）：`hf_qwen3_bf16_accel_forward` 设备通用向量化路径（CPU 上与循环路径逐位相同），CUDA eager 用原生 BF16 张量核（CUBLAS/batched_mul，F32 累加），Reactant XLA 可编译同一实现；0.6B—4B GPU parity/greedy 全对，吞吐 8—15 tok/s；CPU batched matmul 显式分派为 F32 累加防止通用 fallback 破坏契约。
- XLA BF16 compiled decode（Week 16）：static KV cache 的 traced prefill/decode/greedy（traced position、动态写、有效前缀掩码），设备端 argmax 闭环使宿主每 token 只取回一个整数；0.6B steady 246 tok/s（eager 16.1×），greedy 与 HF 全对。
- RTN 权重量化（Week 16）：INT8 per-channel / INT4 group 打包 + 混合精度选项；`load_hf_qwen3_quantized` 逐投影流式量化加载；分块反量化线性层；8B INT8 与 14B INT4 首次 GPU 驻留（各约 8.2/8.4 GiB），量化只改权重驻留格式不改计算契约。
- HuggingFace GPT-2 导入：冻结 revision/checksum，严格映射 learned position、LayerNorm bias、fused QKV 与 HF Conv1D `(in, out)`，验证 causal buffers 与 tied LM head；完整 context 参数量 124,439,808。
- 显式 `hf_token_ids` 处理 HF 0-based 到 LifeAI 1-based 边界；逐层 trace 与 reference 脚本可验证 embedding、每个 block、final hidden、full logits 和 cache decode logits。`Lux.parameterlength` 已包含自定义 Q/K-Norm scale，六个 dense topology 的精确参数量与冻结 reference 一致。

### 2. Tokenizer 与数据

- `AbstractTokenizer` 统一接口：character、byte、byte-BPE、imported HF Qwen3 与 imported HF GPT-2 共用 encode / decode / vocab / special-token / fingerprint API，token id 保持 1-based。
- legacy character `Tokenizer` 完整保留，旧调用与旧 checkpoint 不受影响。
- `ByteTokenizer`：对任意有效 UTF-8 无 OOV、可精确 round-trip；`decode_bytes` 始终可逆，`decode` 提供显式 `:strict` / `:replace` 策略。
- `ByteBPETokenizer`：train-only 确定性训练，固定 tie-break，相同语料与配置产生相同 vocabulary、merge ranks 和 fingerprint。
- `HFQwen3Tokenizer`：严格导入目标 revision 的 NFC、regex、ByteLevel、151,643-token BPE、151,387 merges 与 26 个 added tokens；支持 HF character/Julia UTF-8 byte spans、byte-exact decode、special-token 语义和 1-based 公共 ids。
- `HFGPT2Tokenizer`：严格导入 GPT-2 regex、ByteLevel、50,257-token BPE、50,000 merges 与 `<|endoftext|>`；10 组 ASCII/Unicode/空白/控制字节/special corpus 与 HF 完全一致。
- Qwen3 基础 chat template：支持无 tools 的 system/user/assistant、generation prompt 与 thinking 开关；模板 hash 与三份 tokenizer config checksum 纳入 provenance/fingerprint，未知 revision fail closed。
- Tokenizer artifact v1：显式 schema version、normalization、special tokens、vocabulary / merges 与内容指纹，可独立保存、加载与校验，篡改被拒绝。
- 中文数据管线：以 document 为单位记录来源、许可、checksum、变换配置；确定性文档级无泄漏 split；Tokenizer 只在 train split 上拟合；versioned dataset artifact 与显式 EOS 边界语义。
- 滑动窗口 DatasetLoader 与 DocumentDatasetLoader，支持 batch、stride 和 `drop_last`。
- 稀疏 next-token cross entropy；token-weighted validation loss、perplexity 与 `bits_per_byte` 等 byte-normalized 评估。
- checkpoint format v2：版本化、设备无关，支持全部五类 Tokenizer 的保存恢复，并显式迁移 v1 legacy checkpoint。
- 确定性 checkpoint resume、可配置 global gradient norm clipping、Zygote 常规训练路径与 Reactant + Enzyme 的 XLA 训练路径。

### 3. 生成与推理

- greedy、temperature、top-k 和 top-p sampling；基础生成入口对全部五类 Tokenizer 通用。
- 动态 KV Cache（prompt prefill、单 token decode、cached generation）与固定形状静态 KV Cache（面向编译后增量推理）。
- XLA prefill / decode 接口及编译后生成流程。
- full forward、动态 KV Cache、静态 KV Cache 的 correctness matrix 与 microbenchmark。
- CPU、CUDA GPU、XLA CPU、XLA GPU 独立进程 benchmark，可区分 cold compile、warm-up 和 steady-state。
- `load_hf_qwen3_bundle` / `generate_hf_text` 串联本地模型、tokenizer、EOS 与 greedy trace；Qwen3-0.6B 的 full、dynamic、static 生成及 host-tokenizer→XLA static 路径已验证。
- Qwen3 generation config 严格解析、`:sample` / `:config` 与固定 uniform CDF replay 已完成；真实 HF sampled integration 86 / 86 通过，16 步 token/candidate/文本完全一致，概率 global max-abs `5.90086e-6`。
- Qwen3 rotate-half RoPE 已用 Transformers 4.51.0 独立 fixture 验证到 position 40,959；真实 0.6B 的 CPU、CUDA 与 Reactant-XLA GPU cache correctness/benchmark 均有冻结条件和原始 JSON。
- `load_hf_gpt2_bundle` / `generate_hf_text` 串联冻结 GPT-2 模型/tokenizer；embedding、12 blocks、final hidden、full logits 与 full/dynamic/static 8-step greedy text 均通过 Transformers Float32 reference。

### 4. 学习与可视化记录

`notebook/` 已覆盖 Attention 结构、RoPE、prefill / decode、KV Cache 原理与常见错误、动态与静态 cache 等主题；这些 notebook 不只是展示结果，也是关键组件学习过程的一部分。

## 验证状态

运行默认测试套件：

```bash
julia --project=. -e 'using Pkg; Pkg.test()'
```

2026-07-26 复核默认套件，共 `4870 / 4870` 项测试通过；其中 Week 05 专项 3094 项、Week 06 专项 112 项、Week 07 离线专项 54 项、Week 08 离线专项 61 项、Week 09 离线专项 67 项、Week 10 离线专项 37 项、Week 11 专项 91 项、Week 12 离线专项 85 项、Week 13 离线专项 283 项、Week 14 离线专项 77 项、Week 15 离线专项 63 项、Week 16 离线专项 78 项。历史 Reactant/XLA 专项 `52 / 52` 通过；Week 10 与 Week 11 的 XLA smoke 均曾另行复核 `4 / 4` 通过（Week 12—15 未改动既有 XLA F32 路径）。

Week 15 的离线 `63 / 63` 覆盖 CPU batched matmul F32 累加契约、向量化 vs 循环路径逐位一致（tied/untied）与 CUDA/XLA 资产 contract；CUDA opt-in（`LIFEAI_QWEN3_BF16_CUDA_0_6B/1_7B/4B_MODEL_DIR` 同进程 `Pkg.test`）`173 / 173` 通过。CUDA 三尺寸 16 步 greedy 与 HF BF16 全部一致，吞吐 15.3 / 14.1 / 8.1 tok/s，VRAM 峰值 ≤ 12.1 GiB；XLA BF16 prefill 编译 44.8 s、steady 1.36 ms，argmax/greedy 首 token 一致（`scripts/verify_qwen3_bf16_xla.jl`）。

Week 16 离线专项覆盖 RTN round-trip/打包语义（48 项）、量化树驱动 accel 前向（13 项）与 XLA/量化资产 contract。GPU/XLA 实测经脚本冻结：XLA compiled decode（`scripts/verify_qwen3_bf16_xla_decode.jl`）设备端 greedy steady 4.06 ms/token = 246 tok/s，两条编译路径 16 步 greedy 与 HF 全对；8B INT8（`scripts/verify_qwen3_quant_cuda.jl`）树 8.22 GiB、argmax 全对、greedy 14/16（第 15 token 近平局）；14B INT4 g128 树 8.38 GiB、prefill argmax 对、greedy 4/16（第 5 token 分歧）。量化吞吐 0.11—0.61 tok/s（每 token 全量反量化，带宽瓶颈，驻留验证非吞吐目标）。

Week 14 的离线 `77 / 77` 覆盖 BF16 位保真加载、混合精度算子语义、路径确定性与 cached-decode 逐位等价、BF16 资产/parity contract。opt-in：0.6B/1.7B/4B BF16 integration 同进程 `Pkg.test` 中 Week 14 专项 `205 / 205` 通过；8B BF16 integration 用独立进程 + `--heap-size-hint=2G` 跑 `125 / 125` 通过（峰值 RSS 18.53 GiB）。四尺寸 16 步 greedy token 与 HF BF16 全部一致，logits max-abs ≤ 0.72（mean ≤ 0.073），blocks 尺度感知误差 ≤ 1.49e-2，embedding 位精确。

Week 13 的离线 `283 / 283` 覆盖流式 reader 严格性（44 项）、流式 vs in-memory 逐位一致（23 项）与 8B/14B/32B 资产/parity contract；带 `LIFEAI_QWEN3_8B/14B/32B_MODEL_DIR` 的 opt-in 全套 `689 / 689` 通过（integration 406 项，重算逐层流式 parity）。8B/14B/32B 流式峰值 RSS 分别 6.94 / 8.84 / 8.87 GiB（对照 Float32 全量 30.51 / 55.02 / 122.06 GiB），logits max-abs ≤ `3.05e-5`，decode ≤ `4.01e-5`，argmax 全一致。注意 opt-in 协议：tied 尺寸与 streamed 尺寸的 integration 必须分两个 `Pkg.test` 进程运行（同进程堆累积会 OOM）。block 级中间激活用尺度感知容差（scaled ≤ 1e-5，实测最差 6.9e-7），归一化输出用绝对容差。

Week 11 的 `91 / 91` 包含：六个 frozen config 的 revision/checksum、自动识别、错配/RoPE 语义漂移/MoE 拒绝，六套完整 depth/width topology 和精确参数量共 80 项；untied LM head、`Q width > hidden`、GQA dynamic/static cache 的缩小 32B 形态共 11 项。

Week 12 的离线 `85 / 85` 冻结 1.7B/4B 全部资产 checksum、分片清单与实测 parity 结果；带 `LIFEAI_QWEN3_1_7B_MODEL_DIR` / `LIFEAI_QWEN3_4B_MODEL_DIR` 的 opt-in integration 共 `209 / 209` 通过（含 1.7B 57 项、4B 67 项），重算文件尺寸、variant 识别、精确参数量与全部逐层/decode 断言。1.7B 的 final hidden / logits / decode max-abs 为 `1.72e-4 / 9.35e-5 / 3.05e-5`，4B 为 `1.07e-4 / 3.53e-5 / 2.00e-5`，argmax 全一致；4B 真实 3 分片 index 加载 32.0 s，峰值内存在 30 GiB 内。

使用 `Qwen/Qwen3-0.6B` revision `c1899de289a04d12100db370d81485cdf75e47ca` 的真实 BF16 权重和 Transformers Float32 reference，opt-in integration 35 / 35 通过。final hidden max-abs 为 `7.43866e-5`，full logits 为 `5.67436e-5`，dynamic/static decode logits 均为 `4.48227e-5`，下一 token argmax 全部一致。详细版本、容差、逐层误差、checksum 与内存记录见 Week 07 文档。

同一 revision 的 Week 08 真实 tokenizer/text integration 70 / 70 通过：6 组多语种/Unicode/代码/special-token corpus 和 4 组基础 chat 的 strings/spans/ids 完全一致；raw 与 chat greedy 共 6 step 的 token ids、停止位置和文本完全一致，global logits max-abs `5.054474e-5`。chat prompt 的 full/dynamic/static 输出均为 `"hello"` 并在相同 EOS 停止。

Week 09 的官方 sampling reference 使用 Transformers 4.51.0、16 个固定 uniforms 和同一 Float32 compute reference：sampled integration `86 / 86` 通过，raw/filtered/probability global max-abs 分别为 `6.67572e-5`、`3.05176e-5`、`5.90086e-6`。同版本的独立 RoPE fixture 覆盖 position 0/2048/32767/40959，默认专项 `30 / 30` 通过。

Week 10 GPT-2 124M opt-in integration `82 / 82` 通过：tokenizer artifact/checkpoint round-trip 与 10 组 corpus 的 strings/spans/ids/bytes 完全一致；embedding max-abs 为 0，12 blocks 全局最大 `4.8828125e-4`，final hidden `7.05719e-5`，full logits `1.0681152e-4`；full/dynamic/static 的 8-step greedy ids/text 完全一致，step logits global max-abs `1.2207031e-4`。

Qwen3-0.6B CPU benchmark 在 Intel Core Ultra 7 270K Plus 上完成；16/64/256-token prompt 的 dynamic cache decode 分别为 14.28/14.77/10.95 tok/s，256-token 时相对 full recompute 加速 10.33×。RTX 5080 CUDA 的相同三组 dynamic 为 86.06/84.60/67.30 tok/s，static 为 81.55/82.17/81.99 tok/s，三组 correctness 全通过。Reactant-XLA GPU 的 16+2 静态 cache steady decode 为 137.89 tok/s，prefill/decode max-abs `0.01609 / 0.01151`，在 `atol=2e-2, rtol=5e-3` 下通过且 argmax 全一致；cold compile 和 11.59 GiB BFC allocator 成本单独记录。完整 raw samples 见 `benchmark_results/week09/`。

GPT-2 124M 的 16/64/256-token CPU dynamic decode 为 58.71/55.76/33.86 tok/s，CUDA dynamic 为 352.13/339.24/269.92 tok/s；对应 CUDA static 为 329.42/339.46/321.67 tok/s。所有 full/dynamic/static correctness 均通过，完整 warm-up、steady samples、RSS/cache bytes 与同步口径见 `benchmark_results/week10/`。

Week 05 三 seed（20260720–22）跨 tokenizer 对照记录于 `benchmark_results/week05/`：character / byte / byte_bpe 的 tokens per byte 为 0.3717 / 1.0000 / 0.7139，final BPB 3.0753 / 8.1890 / 6.7614；byte 与 byte-BPE 对 unseen UTF-8 lossless 且 validation unknown 率为 0，character 为 19.6%（其 BPB 不可与 lossless tokenizer 直接排名）。

Week 06 GQA benchmark（CPU）记录于 `benchmark_results/week06/`：固定形状下 KV cache 内存严格按 `num_kv_heads / num_heads` 缩减（8 / 4 / 1 heads 对应 1024 / 512 / 128 KiB），dynamic decode 吞吐 2097 / 2484 / 3131 tok/s，全部配置 correctness 为 true。默认测试、XLA 专项测试和硬件 benchmark 仍是三类不同证据。

## 当前边界

以下能力尚未实现，不应从现有 GPT demo 或已完成的结构 parity 推断为已经具备：

- GPT-2 的 WebText 从零训练、论文 zero-shot quality、其他尺寸和非 causal-LM heads；Week 10 只完成 124M 官方 checkpoint 的 Float32 推理/架构复现。
- 通用 Jinja chat template、Qwen3 tools/tool-role 分支、JSON schema 工具注入与 agent tool loop；Week 08 只完成已冻结的无 tools 基础 chat 子集。
- BF16/量化训练、FP8、GPTQ/AWQ 等校准式量化、KV cache/激活量化与 MoE；32B GPU 驻留（INT4 约 16.4 GiB）仍出界。无校准 INT4 的生成保真不足（14B greedy 4/16），量化推理吞吐未优化（每 token 全量反量化）。XLA compiled decode 目前为 0.6B 批 1 greedy；4B XLA、sampling 与 chat 端到端闭环未做。
- Qwen3 128K YaRN / RoPE scaling；六个冻结 checkpoint 的原生 `max_position_embeddings` 均为 40,960，非空 `rope_scaling` 仍 fail closed。
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

### Milestone B''：深化 Qwen3 真实生成与框架性能（已完成）

- 复现官方 temperature/top-k/top-p sampling，比较候选分布并用固定 uniform 流跨框架重放。（Week 09 已完成）
- 验证长位置 RoPE 边界和多 prompt length 的 full/dynamic/static cache correctness。（Week 09 已完成）
- 建立 Qwen3-0.6B CPU、CUDA/XLA 的 load/prefill/decode/RSS/cache benchmark。（Week 09 已完成）

完成标准已满足：官方 sampling 的候选 ids、filtered logits/probabilities 和固定 uniform token 与 Transformers 对齐；真实模型性能结论有原始实验条件和可重复命令，且没有把 BF16 storage 误写为 native BF16 compute。

### Milestone B'''：验证第二种经典 decoder 架构（已完成）

- 以 GPT-2 124M 验证 learned absolute position、LayerNorm + GELU-New、带 bias MHA/MLP 与 HF Conv1D/fused-QKV 权重布局。（Week 10 已完成）
- 严格导入 GPT-2 byte-level BPE，并完成 tokenizer→逐层 logits→KV cache→greedy text parity。（Week 10 已完成）
- 复用 CPU/CUDA/XLA 验证体系，区分官方 checkpoint 推理复现与未执行的 WebText 从零训练/论文质量复现。（Week 10 已完成）

完成标准已满足：checkpoint revision/reference 环境已冻结；GPT-2 124M tokenizer、逐层 hidden/logits、full/dynamic/static generation 与 HF 对齐；默认回归、CUDA 和缩小 XLA smoke 均通过。

### Milestone B''''：补齐 Qwen3 dense family contract（已完成）

- 冻结六个官方 dense 尺寸的 revision、config checksum、width/depth、tied head 与精确参数量。（Week 11）
- 显式识别官方 variant，同时保留兼容 custom dense config；错配 variant fail closed。（Week 11）
- 用完整 topology 和缩小的 untied + 宽 attention 模型覆盖 0.6B 单尺寸真实验证没有触达的结构分支。（Week 11）

完成标准：六个官方 config/topology/参数量进入默认离线测试，tied/untied 与
`Q width > hidden` 的参数映射和 cache 路径均通过；文档不得把 1.7B—32B
结构覆盖写成真实大权重 parity。

完成标准已满足：六个官方尺寸均有 immutable config fixture、完整 topology
和精确参数量；untied + 宽 attention 的缩小 32B 形态通过三路 cache，默认
全套 `4284 / 4284` 通过。

### Milestone B''''': 真实权重 parity 扩展到可实跑的 family 尺寸（已完成）

- 下载 1.7B / 4B 冻结 revision 完整资产并记录全文件 checksum。（Week 12 已完成）
- 用同一 token-id fixture 生成 Transformers Float32 reference，验证逐层
  hidden、logits 与 dynamic/static decode 对齐；两尺寸均走真实分片
  index，1.7B 覆盖 `Q width == hidden` 真实分支。（Week 12 已完成）
- loader 零改动一次通过，未发现尺寸相关 bug；0.6B / GPT-2 既有 parity
  无回归。（Week 12 已完成）

完成标准已满足：两个尺寸在显式容差内逐层对齐且 argmax 一致（默认全套
`4369 / 4369`，opt-in `209 / 209`）；离线 checksum/parity fixture 进入
默认测试；8B+ 不可实跑的边界保持明确。

### Milestone B'''''': 流式加载完成 8B/14B/32B 真实权重验证（已完成）

- header-only safetensors 索引、按需单 tensor 读取与逐层流式
  forward/decode，与 in-memory 路径逐位一致。（Week 13 已完成）
- accelerate disk offload 生成 Float32 同口径 reference；8B/14B/32B
  逐层 parity 全部通过，untied LM head 获真实证据。（Week 13 已完成）
- 尺度感知 block 容差与分进程测试协议进入默认套件与文档。（Week 13
  已完成）

完成标准已满足：六尺寸真实权重逐层验证闭环完成（默认全套
`4652 / 4652`，week13 opt-in `689 / 689`）；32B 峰值 RSS 8.9 GiB 实测
记录在案。

### Milestone B''''''': native BF16 混合精度推理（已完成）

- BF16 位保真加载与参数内存减半；独立推理路径逐算子镜像 HF 混合精度
  契约。（Week 14 已完成）
- 0.6B/1.7B/4B/8B 与 HF BF16 逐层对齐、argmax 零失配、16 步 greedy
  token 序列完全一致。（Week 14 已完成）
- 8B 全量驻留 greedy 生成与内存实测；分块升精度 + 显式 GC + heap
  hint 的内存工程协议。（Week 14 已完成）

完成标准已满足：默认全套 `4729 / 4729`，BF16 opt-in `205 / 205` 与
8B 独立进程 `125 / 125` 全绿；容差、greedy 序列与内存实测冻结进
离线 fixture。

### Milestone B'''''''': BF16 推理落地 CUDA/XLA（已完成）

- 设备通用向量化 BF16 路径，CPU 上与循环路径逐位相同。（Week 15
  已完成）
- 0.6B/1.7B/4B CUDA parity/greedy 全对，吞吐 8—15 tok/s（CPU 的
  33—92 倍）。（Week 15 已完成）
- Reactant XLA BF16 编译 prefill 通过（steady 1.36 ms）。（Week 15
  已完成）

完成标准已满足：默认全套 `4792 / 4792`、CUDA opt-in `173 / 173`
全绿；吞吐/VRAM/编译耗时冻结进离线 fixture；推理主战场移至
CUDA/XLA。

### Milestone B''''''''': XLA 编译生成与量化驻留（已完成）

- XLA BF16 static-cache decode 编译，设备端 greedy 闭环 246 tok/s
  （eager 16.1×），greedy 与 HF 全对。（Week 16 已完成）
- RTN INT8/INT4 量化：8B 与 14B 首次 GPU 驻留；8B token 行为近乎
  无损，14B INT4 漂移精确量化记录。（Week 16 已完成）

完成标准已满足：XLA 吞吐目标 ≥10× 达成；量化离线语义测试、GPU 驻留
实测与行为记录冻结进 fixture；默认全套无回归。

### Milestone C：建立最小有状态智能体闭环（后移）

- 定义与具体机器人无关的 `Observation`、`Action`、`Memory` 和 policy / model 接口。
- 先在一个简单、可重复的模拟环境中跑通"感知 → 记忆 → 决策 → 行动 → 反馈"。
- 保持模型后端可替换，使当前小 GPT、Qwen3 复现权重或后续多模态模型都能接入。

完成标准：智能体可以跨多个 step 保持状态，根据环境反馈改变下一步动作，并用测试或 replay 重现一次完整轨迹。

## 长期能力地图

| 主线 | 当前状态 | 下一关键缺口 |
| --- | --- | --- |
| 模型基本组件 | Qwen3 六尺寸真实权重 parity 全闭环 + native BF16 推理 + CUDA/XLA 加速（XLA compiled decode 246 tok/s）+ INT8/INT4 量化 GPU 驻留（8B/14B）；GPT-2 真实 parity；流式加载；五类版本化 Tokenizer 与中文数据管线 | 校准式量化提升 INT4 保真、量化 gemm 吞吐、XLA sampling/chat 闭环，或下一经典/SOTA 架构 |
| 高效训练与推理 | modern / GQA / rotate_half 已兼容 Zygote / XLA 与两类 KV Cache；Qwen3-0.6B CPU/CUDA/XLA decode 已真实验证 | 低精度、device-resident sampling 与更长上下文优化 |
| 智能体核心 | 尚未开始；Qwen3 基础 chat 输入已可作为模型后端 | conversation state、memory、planning、tools、agent loop |
| 多模态感知 | 尚未开始 | vision / audio / sensor representation |
| 具身闭环 | 尚未开始 | observation/action abstraction、simulation、device adapter |
| 持续学习与生命感 | 处于愿景阶段 | 长期状态、适应、主动性与安全边界 |
| 学习记录 | Week 01—16 已 Closed | 继续以论文/官方 reference、数值 parity、性能原始记录为近期节奏 |
