# Chapter 24 — Qwen3 MoE 架构支持

> 所属 Episode：Episode 06 — Qwen3 MoE 与模型架构扩展
>
> 状态：Open

## Open：核心问题

LifeAI.jl 能否严格复现原始 Qwen3 MoE 的 top-k routing、expert SwiGLU、HuggingFace 权重布局与 cached decode，并把 correctness reference 推进为不计算未选 expert 的可部署实现？

目标契约以 `Qwen/Qwen3-30B-A3B` 的官方 `qwen3_moe` 配置和 Transformers 4.51.0 实现为准：128 experts、每 token 选择 8 experts、Float32 router softmax、选中概率重新归一化。这个原始架构没有 shared expert；其他后续 Qwen MoE 变体不能未经验证复用本章结论。

## 预期结果

本章 Close 时，应当可以展示或验证：

1. tiny Qwen3 MoE 的 router、expert mixture、full logits 与 HuggingFace reference 在显式容差内对齐。
2. 官方 `config.json` 与逐 expert safetensors 名称可以严格加载，missing、unexpected、shape 和不支持的 dense/sparse 混合 schedule 均 fail closed。
3. prefill、dynamic cache 与 static cache decode 一致，并形成 CUDA/XLA 可编译的 sparse dispatch 路径。
4. 至少一个官方真实 checkpoint 的资产 provenance、逐层 parity、内存与性能结果被冻结。

## 计划

| 工作项 | 所属主线 | 交付物 | 验收方式 | 状态 |
| --- | --- | --- | --- | --- |
| top-k router 与 expert SwiGLU correctness oracle | 模型 | `Qwen3SparseMoE`、Float32 routing | 独立手写 reference 对拍 | 已完成 |
| GPT decoder 与 KV cache 集成 | 模型 / 推理 | `GPTModel(..., mlp_type=:qwen3_moe)` | full/dynamic/static logits 一致 | 已完成 |
| `qwen3_moe` config 与权重映射 | 工程 | strict config parser、逐 expert stack loader | tiny HF 命名 fixture、错误输入 fail closed | 已完成 |
| 独立 Transformers tiny reference | 模型 | hidden/router/logits fixture | 跨框架逐层 parity | 已完成 |
| CPU sparse token dispatch | 高效推理 | 只执行被选 token-expert pair 的 gather/compute/combine | 与 dense oracle 对齐，inactive expert 不执行 | 已完成 |
| XLA sparse dispatch | 高效推理 | compact top-k route + route-major gather/matmul/combine | XLA CPU 编译、correctness 与 steady latency | 已完成 |
| CUDA sparse dispatch | 高效推理 | indexed hidden/down/combine kernels，不复制 route 权重 | 无 host fallback、correctness、workspace 与 prefill/decode 性能证据 | 已完成 |
| MoE 按需 expert 流式加载 | 模型 / 工程 | header-only index、路由后按 active expert 读取、cache decode | 分片 fixture、tiny Transformers 逐位 parity、加载集合可观测 | 已完成 |
| 30B-A3B immutable 资产契约 | 模型 / 工程 | revision、config/index、16 分片大小与 SHA256 | 离线 manifest、opt-in 全资产校验 | 已完成 |
| 官方真实权重 parity | 模型 / 工程 | streamed parity 报告 | 逐层/logits/cache 与峰值内存实测 | 计划中 |

## Close 条件

只有以下条件满足后才能关闭本章：

- router top-k、`norm_topk_prob` 和 tie-break 契约有独立 reference。
- tiny checkpoint 的 config、全部 expert 权重、full logits 和 cached decode 与 Transformers 对齐。
- CPU、CUDA、XLA 至少各有一条可验证执行路径；若某后端不支持，必须给出可复现边界而不是静默回退。
- 生产路径不会对每个 token 计算全部 128 experts。
- 真实 checkpoint 采用流式/按需 expert 加载，不要求把 30B Float32 state dict 全量驻留内存。
- Dense Qwen3 和 GPT-2 既有测试无回归。

## 学习重点

- 要理解的概念：router softmax、top-k gating、选中概率重归一化、token-to-expert dispatch、expert 容量与负载均衡。
- 要亲手实现的关键组件：router、expert 参数布局、dispatch/combine、MoE checkpoint loader 与 cache 集成。
- 要验证的假设：先建立全 expert correctness oracle，能够降低 sparse accelerator 实现的数值调试风险。

## 风险与取舍

- CPU 默认路径已经按 expert gather 被选 token；全 expert masked 版本保留为 `qwen3_dense_expert_reference` 数值 oracle。非 CPU Array 的标准 Lux 调用会进入 compact device path；Reactant/XLA CPU 使用可移植 route-major fallback，RTX 4090 D CUDA 通过 package extension 自动启用 indexed kernels。XLA fallback 仍物化 route 权重；CUDA 已消除此临时张量，但还不是 grouped GEMM / tensor-core fused kernel。
- `load_hf_qwen3_moe_model` 仍会把 expert 权重 stack 成三维数组，只适合 tiny fixture；`stream_hf_qwen3_moe_forward` 已提供真实 checkpoint 所需的 header-only、逐层、路由后按 active expert 读取生命周期，但尚未用本地 30B-A3B 资产实跑。
- host `partialsortperm` 不是 CUDA/XLA 路由实现，不能把现有 Float32 CPU 通过写成 accelerator 已支持。
- Qwen3 后续系列可能使用融合 expert tensor、shared expert、不同 attention 或 hybrid layer；必须按各自配置重新建立契约。

## 实验与过程记录

### 2026-08-07：最小 MoE 架构切片

- 新增 `Qwen3SparseMoE`：router 参数 `(experts, hidden)`，expert gate/up/down 按第三维 stack。
- router 在 Float32 上 softmax，保留固定 top-k，并支持 `norm_topk_prob`。
- `GPTModel` / `TransformerBlock` 增加 MoE 构造配置；现有 full、dynamic cache、static cache 路径无需专用分支。
- 新增 strict `load_hf_qwen3_moe_config` 与原始逐 expert 权重名映射。
- 四个按测试内容命名的专项共 `43 / 43` 通过：router 13、expert mixture 6、config/weight mapping 18、cached decode 6。
- Dense 回归：Qwen3 HF weight loading `54 / 54`、dense family `91 / 91` 通过。

### 2026-08-07：独立 Transformers tiny parity

- 新增 `scripts/export_qwen3_moe_tiny_reference.py`，使用项目 `.venv` 中的 Python 3.10.12、PyTorch 2.7.1+cpu、Transformers 4.51.3，从官方 `Qwen3MoeForCausalLM` 独立生成 2-layer / 4-expert / top-2 Float32 checkpoint。
- config、generation config、模型权重、reference tensors 和生成脚本的 SHA256 全部冻结在 `reference.json`；重新生成命令：

  ```bash
  .venv/bin/python scripts/export_qwen3_moe_tiny_reference.py \
    test/fixtures/qwen3_moe_tiny_parity
  ```

- `hf_qwen3_moe_forward_trace` 显式观测 post-attention RMSNorm 后、expert dispatch 前的 router logits；没有用 LifeAI 结果生成 reference。
- 两层共 8 个 token-router 决策的 top-2 expert 顺序和归一化权重全部一致。最大绝对误差：router `5.96e-8`、block `7.45e-9`、final hidden `3.58e-7`、full logits `8.94e-8`、prompt/decode logits `5.96e-8`；embedding 逐位相同。
- Transformers parity 专项 `44 / 44`，Chapter 24 累计 `87 / 87`；默认全套 `5,750 / 5,750` 通过。

### 2026-08-07：CPU sparse token dispatch

- `qwen3_sparse_expert_dispatch` 按 expert gather 非零 routing token，完成 SwiGLU 后按原 token index combine；`Qwen3SparseMoE` 默认前向不再计算未选 token-expert pair。
- `Qwen3MoEDispatchStats` 显式记录 active experts、逐 expert token 数、实际 routed pairs 和 dense oracle pairs，避免仅凭实现名称宣称稀疏。
- inactive expert 测试把三个未选 expert 的 gate/up/down 全部写成 `NaN`：sparse 输出保持全 finite，而 dense oracle 被 `NaN × 0` 污染，证明未选 expert 没有进入矩阵乘法。
- 128 experts / top-8 / 64 tokens / `d_model=128` / expert hidden 64、BLAS 单线程的 7 次 steady CPU 对照：执行 pair 从 `8,192` 降到 `512`（16×），sparse median `1.416 ms`，dense median `6.637 ms`，加速 `4.69×`，max-abs `3.58e-7`。原始结果位于 `benchmark_results/qwen3_moe_sparse_dispatch/cpu_reference.json`。
- sparse dispatch 新增 `16 / 16`，Transformers parity 仍为 `44 / 44`；Chapter 24 累计 `103 / 103`，默认全套 `5,766 / 5,766`。

### 2026-08-07：compact device routing 与 XLA sparse dispatch

- 新增 `qwen3_device_topk_routing`：固定 `top-k` 次 reduction/select，直接生成 `(top_k, tokens)` 的 expert index 与 routing weight，不构造生产用 dense routing table，也不调用 host `partialsortperm`。
- 新增 `qwen3_device_sparse_expert_dispatch`：把 `top_k × tokens` 展平为 route pairs，只 gather 被选 expert 的 gate/up/down 权重，以 batched matmul 执行 SwiGLU，再按 token 维 combine。标准 `Qwen3SparseMoE` Lux 调用对非 CPU `Array` 自动采用该路径。
- CPU 内容化测试覆盖 compact routing 重建、显式最高 expert-index tie-break、dense oracle 数值一致性和未选 `NaN` expert 隔离，共新增 `14 / 14`。
- Reactant/XLA CPU 对标准 Lux 调用完成真实编译与执行，专项 `3 / 3`；kernel 内没有 `findall`、scalar indexing 或 data-dependent Julia branch。
- 128 experts / top-8 / 64 tokens / `d_model=128` / expert hidden 64 的 XLA CPU 冻结实测：route entries 为 `512`，相对 dense `8,192` pairs 缩减 `16×`；编译 `32.126 s`，7 次 steady median `38.616 ms`，对 dense oracle max-abs `9.09e-7`。结果位于 `benchmark_results/qwen3_moe_sparse_dispatch/xla_cpu_reference.json`。
- 该 XLA CPU latency 明显慢于已冻结的 host CPU sparse median `1.416 ms`：当前 route-major 实现会物化每个 route 的选中权重，证明“没有计算未选 expert”尚不等于 accelerator kernel 已高效融合；后续 CUDA 实测继续验证这一开销边界。
- Chapter 24 累计 `117 / 117`，默认全套 `5,780 / 5,780`；XLA 新专项另计 `3 / 3`。

### 2026-08-07：RTX 4090 D CUDA route-major baseline

- 宿主机 `nvidia-smi` 确认 RTX 4090 D、driver `570.153.02`；CUDA.jl 使用 runtime `12.9.0`，设备可用显存约 `23.99 GiB`。此前“CUDA driver 不可用”来自文件/设备隔离沙箱，不能作为宿主机能力结论。
- 标准 Lux 设备前向直接接受 `CuArray` 参数与输入；数值对齐和未选 `NaN` expert 隔离共 `5 / 5`，没有打开 CUDA scalar indexing。
- 同一 128 experts / top-8 / `d_model=128` / expert hidden 64、15 次 steady 实测：单-token decode 为 `0.493 ms`，单线程 CPU sparse 为 `0.0774 ms`（CUDA/CPU `0.157×`）；64-token prefill 为 `0.569 ms`，CPU 为 `1.617 ms`（CUDA `2.84×`）。max-abs 分别为 `2.98e-7` 与 `7.38e-7`。
- 首个 CUDA shape 的 cold kernel/JIT 为 `9.915 s`；64-token case 复用 kernel 后 cold call 为 `8.99 ms`。route pairs 仍严格缩减 `16×`，但单-token 需要物化 `0.75 MiB` 选中权重，64-token 需要 `48 MiB`，解释了 decode launch/gather 开销和下一步融合方向。
- 原始结果位于 `benchmark_results/qwen3_moe_sparse_dispatch/cuda_4090d_reference.json`。这一结果验收基础 CUDA 执行路径，并作为后续 indexed kernels 的冻结优化前 baseline。

### 2026-08-07：CUDA indexed expert kernels

- 将 CUDA 专用实现放入 `LifeAICUDAExt` package extension：普通 `using LifeAI` 不加载或初始化 CUDA；调用方显式加载 CUDA 后，`CuArray` 自动 dispatch 到专用方法，CPU/XLA 仍使用主模块内的 portable fallback。
- 三个 kernel 分别直接从原始 `(hidden, d_model, experts)` 权重按 route expert index 计算 SwiGLU hidden、down projection 和 top-k combine，不再 gather/复制每条 route 的 gate/up/down 矩阵。旧实现保留为 `qwen3_route_major_expert_dispatch` benchmark oracle。
- CUDA 专项扩展为 `9 / 9`：indexed vs dense、indexed vs route-major、inactive `NaN` expert 隔离，以及精确 workspace byte contract 全部通过；max-abs 对 dense oracle ≤ `6.56e-7`。
- 128 experts/top-8 下，单-token 临时空间从 `0.75 MiB` 降到 `6.50 KiB`，64-token 从 `48 MiB` 降到 `416 KiB`，两组都是 `118.15×` 缩减。
- RTX 4090 D 最终重跑的 15 次同进程 steady 对照：单-token indexed `0.365 ms` vs route-major `0.481 ms`（`1.32×`），64-token indexed `0.377 ms` vs route-major `0.592 ms`（`1.57×`）。64-token 相对单线程 CPU sparse 为 `4.06×`；单-token 仍只有 CPU 的 `0.198×`，说明下一瓶颈已从权重物化转为小 kernel launch 与标量 dot-product 效率。
- 原始结果位于 `benchmark_results/qwen3_moe_sparse_dispatch/cuda_4090d_indexed_kernels.json`。该文件 schema 2 也记录 bucketed 对照：小型 `128→64` 的单/64-token bucketed 分别为 `0.416 / 0.449 ms`，均慢于 indexed；这三个直接索引 kernel 是低 workspace baseline，尚未使用 grouped GEMM 或 tensor cores。

### 2026-08-07：路由驱动的 checkpoint expert streaming

- 新增 `stream_hf_qwen3_moe_forward`：先以 header-only reader 严格校验单文件或 `model.safetensors.index.json`，attention 参数仅驻留一层；进入 MoE 后先计算 router，再逐个读取当前 batch 激活 expert 的 gate/up/down 权重，计算完成即释放，不构造完整 expert stack。
- prompt 与单步 dynamic KV-cache decode 共用同一生命周期。返回值显式给出每层 1-based `active_experts` / `decode_active_experts`，使物理权重读取集合可被测试和调用方审计；单 token 每层只读取 top-k 个 expert，30B-A3B 契约下为 8/128。
- 冻结 Transformers tiny checkpoint 上，streamed embedding、router logits、逐层 hidden、final hidden、prompt/full/decode logits 与现有 eager 路径逐位一致；该专项从 `44 / 44` 扩展到 `55 / 55`。
- 新增双分片 MoE checkpoint fixture：正常路径 eager/streamed 逐位一致，index 删除一个 expert tensor 后在计算前 fail closed，共 `5 / 5`。
- Chapter 24 默认专项由 `117 / 117` 增至 `133 / 133`；完整默认套件 `5,796 / 5,796`。本机尚无 Qwen3-30B-A3B 权重，因此这一阶段只关闭加载生命周期与合成/独立 tiny correctness，不宣称真实 30B parity 或真实峰值内存。

### 2026-08-07：Qwen3-30B-A3B immutable 资产契约

- 从官方 `Qwen/Qwen3-30B-A3B` 仓库冻结 immutable revision `ad44e777bcd18fa416d9da3bd8f70d33ebb85d39`。原始 config SHA256 为 `2850ddb3…c2297`，index SHA256 为 `df0d481e…3c24`。
- checkpoint 是 16 个 BF16 safetensors 分片：tensor payload `61,064,245,248` bytes，含 header 的分片文件共 `61,066,575,648` bytes；index 精确包含 `18,867` 个 tensor，对应 `30,532,122,624` 个参数。16 个分片的大小与 LFS SHA256 全部冻结在离线 manifest 和 `Qwen3MoECheckpointSpec`。
- 新增 `verify_qwen3_moe_checkpoint`：config/index 始终校验 SHA256，并验证官方架构、完整 tensor name set、index metadata、分片 assignment 和文件大小；默认再顺序哈希全部 61 GB 权重，`verify_shard_checksums=false` 仅用于显式快速预检。
- 新增 `scripts/verify_qwen3_moe_checkpoint.jl MODEL_DIR [OUTPUT_JSON] [--fast]`，输出 revision、资产大小、校验强度、耗时和逐分片报告。默认测试冻结官方 config/manifest `110 / 110`；双分片 BF16 fixture 对验证器成功路径新增 3 项，Chapter 24 聚合 `246 / 246`。
- 本阶段没有自动下载 61 GB 权重。仓库已经具备下载后的 fail-closed 验证入口，但真实逐层/logits/cache parity 与峰值 RSS 仍需本地完整 checkpoint 才能执行。

### 2026-08-07：BF16 CUDA expert dispatch 与官方投影宽度

- 官方 checkpoint 下载与真实 parity 本轮明确延后；已下载的约 64 MiB partial cache 保留在模型目录，可在后续阶段断点续传，本轮没有后台下载进程。
- CUDA indexed 专项新增 BF16 expert 参数测试：router/input 保持 Float32，gate/up/down 以 BF16 驻留，kernel 显式用 Float32 累加；与同一组 BF16 数值转回 Float32 的 CPU sparse reference 对齐。CUDA 专项由 `9 / 9` 增至 `13 / 13`。
- 新增 `scripts/benchmark_qwen3_moe_cuda_projection_widths.jl`，直接使用官方 `d_model=2,048`、expert hidden `768`、128 experts、top-8。完整三组 expert tensor 的 BF16/F32 容量分别为 `1,207,959,552 / 2,415,919,104` bytes，计时不包含 checkpoint I/O 或 router。
- RTX 4090 D 最终冻结的 15 次 steady：BF16 indexed dispatch 的 1-token / 8-token median 为 `0.180 / 0.689 ms`，相同数值的 Float32 权重为 `0.396 / 1.319 ms`，BF16 加速 `2.20× / 1.91×` 且输出逐位一致。原始结果位于 `benchmark_results/qwen3_moe_sparse_dispatch/cuda_4090d_bf16_projection_widths.json`。
- 实验过“每个 dot-product 一个 thread block”的 shared-memory tiled reduction；在相同官方宽度上比现有“每个输出一个 thread”的 indexed kernel 慢 `2.6—6.0×`，原因是输出维与 route pair 已提供充足并行度，额外拆分产生过多 blocks 和 reduction 同步。该劣化路径未进入生产实现；下一步若继续优化，应做按 expert 分桶后的 GEMM/tensor-core 计算，而不是继续拆分单个 dot-product。

### 2026-08-07：设备端 expert route bucketing

- 新增纯 CUDA `qwen3_cuda_bucket_routes`：atomic count、稳定 device `sortperm!`、device `cumsum` 与 1-based half-open offsets 全程不回传 host；`route_permutation` 保留同 expert 内的原 route 顺序。新增 bucketed gate/up/down kernels 按 expert-major 次序读取权重，并把 down projection 写回原 pair index，combine 语义不变。
- CUDA 专项从 `13 / 13` 增至 `27 / 27`，覆盖 stable permutation、含 inactive expert 的 counts/offsets、所有输出仍为 `CuArray`、策略阈值、真实触发 wide-prefill 生产分支、bucketed 与 indexed 逐位一致，以及既有 BF16/`NaN`/workspace contract。
- 独立 bucketing 基准覆盖 8/64/512/4,096 routes，steady median 为 `0.053 / 0.059 / 0.074 / 0.103 ms`；stable permutation 与 offsets 均对 host MergeSort oracle 验证。原始结果位于 `benchmark_results/qwen3_moe_sparse_dispatch/cuda_4090d_route_bucketing.json`。
- 官方 `2,048→768` BF16 synthetic dispatch 中，bucketed 在 1/8/16 token 上仍慢于 indexed，因此 decode 和小 batch 不启用；32 token 为 `1.896 vs 2.813 ms`（`1.48×`），64 token 为 `2.052 vs 6.730 ms`（`3.28×`），输出逐位一致。小型 `128→64` 即使 64 token 仍慢约 16%，证明切换不能只看 token 数。
- 生产 CUDA dispatch 采用保守双阈值：`num_tokens ≥ 32` 且 `d_model × expert_hidden_dim ≥ 1,048,576` 时进入 expert-major bucketed 路径，否则保留 token-major indexed。当前收益来自权重访问局部性，还不是 grouped GEMM 或 tensor-core kernel；cuBLAS grouped wrapper 仍要求宿主矩阵列表/动态尺寸，不能用它引入隐式 host route 同步。

## Close 回顾

- **完成了什么**：CPU Float32 correctness、独立 Transformers tiny parity、CPU sparse dispatch、真实 checkpoint 可复用的路由驱动 expert streaming，以及无 host routing fallback 的 compact Reactant/XLA CPU 与 RTX 4090 D CUDA 路径；本章仍 Open。
- **验证证据**：MoE 专项 `246 / 246`、默认全套 `5,909 / 5,909`，XLA `3 / 3`、CUDA `27 / 27`；官方资产契约 `110 / 110`，双分片验证器成功路径新增 3 项。128-expert CPU 基准 `4.69×`；官方投影宽度 synthetic BF16 的生产策略保持 1/8/16-token indexed，并在 32/64-token bucketed 达到 `1.48× / 3.28×`，各设备路径均与相应 oracle 对齐。
- **没有完成及原因**：本机没有官方 Qwen3-30B-A3B checkpoint，真实资产 checksum、逐层 parity、峰值内存仍未验证；CUDA indexed 单-token decode 仍慢于 CPU，需要 expert 分桶/grouped GEMM，而 XLA 仍使用物化 route 权重的 portable fallback。
- **最重要的认知变化**：原始 Qwen3 MoE 没有 shared expert；同时，compact route pairs 能保证算术稀疏，却不会自动消除选中权重物化与 gather 带宽成本，生产加速仍需要融合/分桶。
- **是否满足 Close 条件**：否。
- **带到下一阶段的问题**：怎样用 expert 分桶/grouped GEMM 或 tensor-core kernel 改善 CUDA 吞吐；待 checkpoint 下载恢复后，再用现有 streamed prompt/cache decode 完成官方 30B-A3B 真实逐层 parity 与峰值内存实测？
