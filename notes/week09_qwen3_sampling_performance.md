# Week 09 — Qwen3 Sampling Fidelity and Real Inference Performance

> 状态：Closed
>
> 开启记录：2026-07-22
>
> 关闭记录：2026-07-23
>
> 依赖基线：[`Week 08 — HuggingFace Qwen3 Tokenizer and Text-to-Text Parity`](week08_hf_tokenizer_text_parity.md) 已 Closed。
>
> 近期主线：继续复现经典/SOTA 模型与论文，以可定位的数值验证和可重复的性能实验积累 LifeAI.jl 的模型组建、推理与框架能力。

## 核心问题与结论

> 在 Week 06—08 已完成 Qwen3-0.6B 结构、权重、tokenizer 与 greedy text-to-text parity 后，能否进一步严格复现官方实际推荐的 temperature → top-k → top-p 采样语义，并以长位置 correctness 和真实模型 benchmark 回答“结果是否对、代价是多少、瓶颈在哪里”？

Qwen3 官方 `generation_config.json` 默认启用采样，而不是 greedy：thinking 模式使用 `temperature=0.6`、`top_k=20`、`top_p=0.95`。Week 08 的 greedy 对齐适合隔离模型数值正确性，但不能代替真实推荐生成路径。Week 09 因此把验收推进到三层：过滤后的候选 mask / logits / probabilities、固定随机流下的逐 token 结果、真实 Qwen3-0.6B 的 prefill/decode 时间与 cache 内存。

结论是可以：16-step 官方 sampled reference 的 token、候选集合、停止状态和文本完全一致；position 40,959 的 Qwen3 rotate-half RoPE 已由 Transformers 4.51.0 独立 fixture 验证；CPU、CUDA 与 Reactant-XLA GPU 均完成真实 0.6B correctness/benchmark。所有结果继续遵守 BF16 storage → Float32 compute 边界，不宣称 native BF16 已验证。

## 已确认的执行边界

1. **参照对象不漂移**：仍使用 Week 07/08 冻结的 `Qwen/Qwen3-0.6B` revision `c1899de289a04d12100db370d81485cdf75e47ca`、Transformers 4.51.0 和同 revision tokenizer/generation config。
2. **采样对齐不绑定 RNG**：PyTorch 与 Julia 的 RNG/`multinomial` 实现不同。跨框架验收使用显式 `[0,1)` uniform CDF threshold；比较候选 ids、过滤 logits、probabilities 与最终 token。Julia 自身 seeded RNG 只做本地可重复性测试。
3. **处理顺序固定**：temperature scaling → HF top-k threshold（边界 tie 全保留）→ HF top-p removal → categorical draw。任何顺序变化都必须由测试捕获。
4. **greedy 兼容不变**：`generate_hf_text` 默认仍为 `strategy=:greedy`，避免 Week 08 调用静默变成随机；`strategy=:config` 才采用官方文件，`:sample` 允许显式覆盖参数。
5. **dtype 表述严格**：当前官方 BF16 safetensors 被解码为 Float32 参数并以 Float32 计算。本 Week 不把它表述为 native BF16 inference；原生 BF16/CUDA kernel 是后续独立工作。
6. **长上下文分层验证**：默认测试覆盖 Qwen3 `rotate_half` RoPE 到 position 40,959 和越界行为；真实 0.6B benchmark 先选 16/64/256 等可重复长度，不用一次 CPU 40K dense forward 冒充完整长上下文性能验证。
7. **性能证据分层**：correctness、CPU benchmark、CUDA/XLA benchmark 是不同证据。CPU 是本 Week 必做；加速器结果只有在硬件/后端实际可用且完成同步计时后才记录。
8. **默认测试离线**：小模型和合成 tokenizer 验证算法、cache matrix 与 fail-closed；真实权重/reference/benchmark 均通过本地目录显式 opt-in，不在测试中下载。

## 预期接口

```julia
bundle = load_hf_qwen3_bundle(model_dir; max_seq_len=512, revision)

# 明确保持 Week 08 的 deterministic greedy。
greedy = generate_hf_text(bundle, prompt; strategy=:greedy)

# 使用 generation_config.json 中的官方采样参数。
sampled = generate_hf_text(
    bundle,
    prompt;
    strategy=:config,
    cache=:dynamic,
    max_new_tokens=32,
)

# 跨框架 reference：相同 uniform CDF threshold，不要求 RNG 算法相同。
replayed = generate_hf_text(
    bundle,
    prompt;
    strategy=:config,
    sample_uniforms=Float32[0.13, 0.73, 0.42, 0.91],
    capture_logits=true,
    capture_distribution=true,
)
```

`hf_generation_config(bundle.tokenizer)` 返回使用 LifeAI 1-based ids 的已校验配置。sampling trace 记录 uniform、采样概率、candidate count，以及按需返回的候选 ids / filtered logits / probabilities。

## 计划

| 工作项 | 所属主线 | 交付物 | 验收方式 | 状态 |
| --- | --- | --- | --- | --- |
| 冻结官方生成契约 | 模型 / 学习 | generation config 字段、官方 thinking/non-thinking 建议与处理顺序记录 | 与目标 revision 的原始 JSON、官方文档和 Transformers 4.51.0 核对 | 已完成 |
| 严格 generation config | 模型 / 工程 | 类型化配置、1-based special ids、sampling 参数与版本校验 | 缺字段、未知字段、非法范围、tokenizer 冲突全部 fail closed | 已完成 |
| top-p 采样核心 | 模型 / 工程 | temperature/top-k/top-p 分布与显式 uniform categorical | 手算 fixture、top-k tie、top-p 边界、非法输入测试 | 已完成 |
| 三路 sampled generation | 模型 / 工程 | full/dynamic/static 支持 `:sample` / `:config` 与采样 trace | 固定 uniforms 下三路 token 完全相同、概率和为 1 | 已完成（离线） |
| HF sampled reference | 模型 / 学习 | Python exporter + Julia verifier，保存逐步 raw/filtered/probability reference | 固定 uniforms 下候选 mask 与 token 精确一致，数值在冻结容差内 | 已完成 |
| 长位置 RoPE | 模型 / 工程 | position 0/2048/32767/40959 的 rotate-half fixture | 公式、范数、边界与 HF reference 对齐 | 已完成 |
| 真实 cache correctness matrix | 模型 / 工程 | 多 prompt length 的 full/dynamic/static logits 与 sampled token 对照 | 记录 max-abs、argmax/candidate/token、容差和首次分叉 | 已完成 CPU/CUDA 16/64/256 与 XLA 16+2 |
| Qwen3-0.6B CPU benchmark | 框架 / 工程 | load、prefill、decode、tok/s、RSS、理论/观测 cache bytes 原始记录 | 固定线程、warm-up、样本数和 prompt/decode 长度，可重复运行 | 已完成 |
| 加速器可行性 | 框架 / 工程 | CUDA / Reactant-XLA 能力与失败边界记录 | 可用则给同步后 steady-state；不可用则记录硬件/编译阻塞，不伪造结果 | 已完成 |
| 文档与复盘 | 学习 | 结论、瓶颈、失败、下一模型/论文选择标准 | 所有结论能指向命令、reference 或 benchmark 原始文件 | 已完成 |

## Close 条件

只有以下条件全部满足后才能关闭 Week 09：

- 目标 revision 的 `generation_config.json` 被严格解析，官方 temperature/top-k/top-p 与 stop ids 无隐式默认。
- Transformers 与 LifeAI 至少 4 个真实生成 step 的 raw logits、候选 ids、filtered logits、probabilities 在冻结容差内对齐；显式 uniforms 下 token、停止位置和文本完全一致。
- full/dynamic/static 三路 sampled generation 在相同 uniforms 下得到相同 token；默认 greedy 回归不变。
- Qwen3 rotate-half RoPE 在短位置和 position 40,959 均有独立 reference，越界 fail closed。
- 真实 Qwen3-0.6B 至少完成 CPU 上 3 组 prompt length 的 prefill/decode benchmark，记录环境、线程、warm-up、steady-state、RSS 和 cache bytes。
- 默认测试、Week 09 opt-in integration 通过；加速器是否完成必须据实单列。
- 文档明确区分 BF16 storage→Float32 compute 与 native BF16，不能用前者宣称后者已验证。

## 学习重点

- **要理解的概念**：logits warper 顺序、top-k tie、nucleus sampling、随机流与分布正确性的解耦；prefill/decode 的复杂度与 KV cache 内存模型；RoPE 高频/低频维度在长位置的数值行为。
- **要亲手实现的关键组件**：HF-compatible top-p filtering、可重放 categorical draw、sampling trace/reference exporter、真实模型 benchmark harness。
- **要验证的假设**：greedy logits parity 足以推出采样候选分布 parity；dynamic/static cache 的微小数值误差不会在选定 uniforms 下改变 token；0.6B 模型的 Julia CPU 瓶颈主要位于矩阵计算和重复 full-vocabulary projection，而不是 tokenizer。

## 风险与取舍

- sampling token 很容易受 CDF 边界的微小概率误差影响，因此 reference uniforms 必须同时记录到相邻 CDF 边界的距离；单个 token 相同不能替代完整候选分布比较。
- top-p 在 tie、`-Inf` 与排序稳定性上的细节可能因框架实现变化；目标锁定 Transformers 4.51.0，不声称兼容任意版本。
- 真实 0.6B benchmark 的内存和 GC 会污染数据；必须分离模型加载、首次执行、warm-up 与 steady-state，并记录进程峰值 RSS。
- native BF16 牵涉参数类型、CPU/BLAS、CUDA/XLA kernel 和归一化数值策略，不能作为一个 loader 参数草率完成；先用本 Week 性能证据决定后续专项。
- Qwen3 MoE、其他 dense 尺寸、量化和训练配方仍不在本 Week 范围。完成 0.6B 推理 fidelity 后再选择 MoE 或下一经典模型，避免范围无限扩张。

## 实验与过程记录

### 2026-07-22：Open 与第一批实现

- Week 08 Close commit `bd08705` 已推送到 `origin/main`，随后开启本 Week；未跟踪的 `artifacts/` 未被纳入提交。
- 新增 `HFQwen3GenerationConfig` 与 `hf_generation_config`，严格校验官方八个字段；unknown field 和非法采样范围 fail closed。
- 通用 sampling 核心新增 top-p，top-k 改为 HF threshold 语义，因此 cutoff tie 会全部保留。
- `generate_hf_text` 新增 `:sample` / `:config`、固定 uniforms 与候选分布 trace；greedy 默认保持不变。
- 离线 sampled cache matrix 已通过；新增 40,960 长度 RoPE cache 的高位置公式/边界测试。
- 新增 `export_qwen3_sampling_reference.py` 与 `verify_qwen3_sampling_parity.jl`；固定 16 个 uniforms，覆盖 candidate count 为 1、2、3 的 top-p 分布。真实 sampled integration `86 / 86` 通过，16/16 token、候选 ids、停止状态与文本完全一致；raw logits global max-abs `6.67572e-5`，filtered logits `3.05176e-5`，probability `5.90086e-6`。
- 模型已重新下载到持久目录 `/home/yj/models/huggingface/Qwen/Qwen3-0.6B/c1899de289a04d12100db370d81485cdf75e47ca/`；reference 位于其 `lifeai-references/week09-sampling/` 子目录。完整 checksum 和恢复命令见 [`local_model_assets.md`](local_model_assets.md)，不再依赖 `/tmp`。
- `benchmark_qwen3_inference.jl` 已在 Intel Core Ultra 7 270K Plus 上实跑：Julia 1 thread、OpenBLAS 24 threads、每组 3 samples、8 decode tokens。模型 load `12.517 s`，load 后 RSS `5.01 GiB`，peak `6.13 GiB`。
- 16/64/256-token prompt 的 dynamic decode 为 `14.28 / 14.77 / 10.95 tok/s`，相对 full recompute 为 `1.98× / 3.72× / 10.33×`；static 为 `11.03 / 9.90 / 12.49 tok/s`，相对为 `1.53× / 2.49× / 11.79×`。三组 cache correctness 全部通过，decode global max-abs `1.01447105e-4`。原始 samples、cache bytes 与环境见 [`benchmark_results/week09/summary.md`](../benchmark_results/week09/summary.md)。
- 首轮 benchmark 因保留每个 sample 的 cache 导致跨样本内存/GC 干扰；harness 已改为只保存 raw timing、sample 前在计时窗外 GC，最终数据来自修正后的重跑。
- CUDA.jl 报告 `functional=true`，设备为 NVIDIA GeForce RTX 5080、16.59 GB、compute capability 12.0、driver/runtime 13.3；`nvidia-smi` 在当前宿主接口下无法通信，因此后续 CUDA 结果必须以 CUDA.jl 同步执行和显存记录为准。
- 完整默认测试 `4135 / 4135` 通过，其中 Week 09 离线专项 `46 / 46`，Week 03 benchmark contract 新增 3 项 raw-sample 测试；本次未重跑 XLA 专项，因为尚未新增 XLA sampling kernel 或真实加速器路径。

### 2026-07-23：长位置 HF fixture、加速器实测与 Close

- 新增独立 `export_qwen3_rope_reference.py`，直接调用 Transformers 4.51.0 的 `Qwen3RotaryEmbedding` 与 `apply_rotary_pos_emb`，冻结 position 0/2048/32767/40959 的 input/cos/sin/rotated 张量。reference 持久副本位于模型目录 `lifeai-references/week09-rope/`，仓库内保留同 checksum 的小型默认测试 fixture；长位置专项 `30 / 30` 通过。
- RTX 5080 CUDA Float32 实测使用 16/64/256-token prompt、8 decode tokens、3 个 steady-state samples，并在每次模型调用后将 logits 物化到 host 同步。dynamic decode 为 `86.06 / 84.60 / 67.30 tok/s`，static 为 `81.55 / 82.17 / 81.99 tok/s`；256-token 时相对 full recompute 分别为 `9.27× / 11.29×`。三组 cache correctness 全通过，decode global max-abs `9.918213e-5`。
- Reactant-XLA GPU 使用真实 0.6B、16-token prompt、2-token decode、3 samples 完成静态 cache 编译与执行。prefill/decode cold compile + first run 为 `64.745 / 24.771 s`，steady decode `137.89 tok/s`；相对 CPU Float32 reference 的 prefill/decode max-abs 为 `0.0160928 / 0.0115070`，在冻结的 `atol=2e-2, rtol=5e-3` 下通过，并且 prefill 与每个 decode step 的 argmax 全一致。XLA 初始化时建立 11.59 GiB BFC allocator，说明当前 Float32 路径的显存固定成本很高。
- 加速器脚本统一记录设备、revision、load/transfer/setup、同步边界、raw samples、cache bytes 与 correctness。原始结果为 [`qwen3_0.6b_cuda.json`](../benchmark_results/week09/qwen3_0.6b_cuda.json) 和 [`qwen3_0.6b_xla_gpu.json`](../benchmark_results/week09/qwen3_0.6b_xla_gpu.json)，汇总见 [`benchmark_results/week09/summary.md`](../benchmark_results/week09/summary.md)。
- 最终默认测试 `4156 / 4156`、真实 sampled integration `86 / 86`、显式 Reactant-XLA 专项 `52 / 52` 全部通过。默认 Week 09 专项由 `46` 项增至 `67` 项，新增的 21 项来自独立 HF RoPE fixture。

## Close 回顾

- **完成了什么**：官方 generation config 与 temperature→top-k→top-p 采样、固定 uniform 跨框架 replay、16-step HF sampled parity、40,959 长位置 HF RoPE reference，以及真实 0.6B CPU/CUDA/XLA cache correctness 和性能基线均已完成。
- **验证证据**：默认测试 `4156 / 4156`；sampled integration `86 / 86`；Reactant-XLA 专项 `52 / 52`；CPU/CUDA/XLA 原始 JSON、revision、fixture 和 checksum 均已记录。
- **没有完成及原因**：没有 Week 09 Close 阻塞项。native BF16、量化、完整 40K 真实模型 dense forward、其他尺寸与 MoE 从一开始就不属于本 Week 范围，不能从本次 Float32 结果外推。
- **最重要的认知变化**：greedy logits parity 只是强 correctness probe；真实复现必须继续覆盖 logits processor、概率分布、随机流和停止语义。性能上，CUDA eager 仍明显受逐 token host materialization 与 full-vocabulary projection 影响；XLA steady-state 很快，但 cold compile 与 11.59 GiB allocator 是不可忽略的成本。
- **是否满足 Close 条件**：是。所有预先定义的 Close 条件均有代码、fixture、测试或原始 benchmark 证据，Week 09 于 2026-07-23 Closed。
- **带到下一 Week 的问题**：在近期“持续复现经典/SOTA 模型和论文”的主线下，下一步可在 native BF16/量化性能专项、Qwen3 MoE，或另一类经典架构复现之间选择；尚未自动 Open Week 10。
