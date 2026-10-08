# Chapter 47 — Qwen3-VL 长生成 allocation profile 与 decoder workspace 归因

> 所属 Episode：Episode 10 — Qwen3-VL 高效生成
>
> 状态：Open

## Open：核心问题

Chapter 46 已证明 static K/V 不再沿 token 轴增长，但一次 32-token generation
仍累计报告约 `3.65 GB` GPU allocation traffic。这个数字把 cache 初始化、vision
结果消费、prefill、31 次 decode、完整 vocabulary logits 和 host greedy selection
混在一起，也不是峰值显存。当前最需要回答的是：

> 在更长的真实 BF16 generation 中，每 token 分配来自哪些 decoder stage，主要
> workspace 的形状与有界容量是什么？

本章先建立可复现的测量与归因，不预设应该先复用 attention、MLP、RoPE 还是
vocabulary projection，也不把 profiler 的同步时延当作 production latency。

## 预期结果

本章 Close 时，应当可以展示或验证：

1. 冻结单图 workload 的 `32 / 128 / 256` token BF16 static generation，prompt
   长度保持 `76`，最终 cache position 分别为 `107 / 203 / 331`；整个 token prefix
   必须匹配独立 HF Float32/CPU free-running oracle。
2. model load、vision forward、cache init、prefill、被排除的 warmup/cold path、
   steady decode 和 host greedy selection 被分别记账。
3. allocation traffic、allocation count、CPU/GC、逐 token latency、CUDA pool
   used/reserved high-water mark、steady baseline drift 与显式 reclaim 回收量使用不同
   字段，避免概念混写。
4. 最终 decode step 的 CUDA.jl allocator traffic 能按 layer/stage 精确闭合，且
   profiled outer bytes/count 与同 position 的无 hook step 一致；static K/V
   `copyto!` 保持零 GPU allocation。
5. 依据数据选出一个 request-local、有明确 shape/byte 上界的 workspace，交给下一章
   实现复用。

## 固定 workload

| 项目 | 契约 |
| --- | --- |
| 模型 | `Qwen/Qwen3-VL-2B-Instruct`，沿用 Chapter 45/46 frozen revisions 与资产哈希 |
| dtype / device | BF16 / CUDA；报告必须记录实际 GPU、driver、runtime，不能跨设备比较绝对时延 |
| 输入 | 单张确定性 `256×256` RGB image、`Describe.`、batch 1、全一 attention mask |
| prompt | 76 tokens，`rope_delta = -56` |
| generation | greedy，`stop_token_ids=[]`，32 / 128 / 256 tokens |
| 独立 oracle | frozen HF Float32 / CPU / eager / DynamicCache，固定生成 256 tokens、忽略 EOS；只作为 token timeline gate |
| static capacity | `prompt_tokens + generated_tokens - 1`，即 107 / 203 / 331 |
| BF16 K/V payload | 12,271,616 / 23,281,664 / 37,961,728 bytes |
| 样本 | 最长 shape warmup 一次并排除；每个长度三次，奇偶轮换执行顺序 |

Chapter 46 的真实数据来自 RTX 4090 D。若本章在另一块 GPU 上执行，只比较同一
进程内的 correctness、allocation 和归因；不能把跨设备 latency 差异写成优化收益。

## 已落地的 profiling 边界

`_profile_qwen3_vl_text_decode_step_static` 在不改变公共 API 的情况下，让诊断调用方
以 `runner(stage, layer_index, thunk)` 包裹每个阶段。request-level stage 使用 layer
index `0`，decoder block 使用 one-based layer：

- request：`token_embedding`、`mrope_prepare`；
- 每层：`pre_attention_norm`、`qkv_projection`、`qk_norm`、`qk_rope`、
  `kv_write`、`attention`、`attention_output_projection_residual`、
  `post_attention_norm`、`mlp_gate_up_projection`、`mlp_activation`、
  `mlp_down_projection_residual`；
- request：`final_norm`、`vocab_logits`。

默认离线 tiny regression 已证明 profiled/unprofiled logits、K/V、position 和
`rope_delta` 完全相同，四层调用顺序为固定的 48 stages。普通 decode 仍传入
`runner=nothing`，不会把 profiler 变成公共生成契约；profiled prefill 还显式使用
`Int32` 的 `logits_to_keep`，验证下游 profiling 路径复用相同的宽整数合同。

## 独立长轨迹 oracle

`export_qwen3_vl_long_generation_reference.py` 复用 Chapter 44/45 已冻结的
Python、Transformers、Torch build、CPU capability、checkpoint revision 与资产 SHA
门禁，以 Float32/CPU/eager `DynamicCache` 从 raw image 和 prompt 独立 free-run 256
tokens。它不接受 Julia token 做 teacher forcing，也不因 EOS 提前停止。每步只保存
top-2、完整 logits raw SHA、cache/mRoPE 几何和整数 token timeline；不会像 Chapter 45
短 reference 那样保留每一步完整 K/V snapshot，因此避免约 11 GiB 的 O(n²) 主机复制。

```bash
QWEN3_VL_MODEL_DIR=/home/ubuntu/models/modelscope/Qwen/Qwen3-VL-2B-Instruct
QWEN3_VL_ORACLE_PYTHONPATH=/tmp/lifeai-qwen3vl-oracle/lib/python3.10/site-packages:/tmp/lifeai-qwen3vl-uv-cache/archive-v0/SNUjiORDNkYR55Or
QWEN3_VL_LONG_REFERENCE=/tmp/qwen3-vl-decode-f32/long_generation_reference.json

PYTHONPATH="$QWEN3_VL_ORACLE_PYTHONPATH" \
  .venv/bin/python scripts/export_qwen3_vl_long_generation_reference.py \
  "$QWEN3_VL_MODEL_DIR" "$QWEN3_VL_LONG_REFERENCE" \
  --greedy-tokens 256 --checkpoint-lengths 32,128,256

sha256sum "$QWEN3_VL_LONG_REFERENCE"
```

Close 前必须在两个独立进程重建并得到逐字节相同 artifact SHA。benchmark 不接受
reference 自报哈希：运行者必须通过
`LIFEAI_QWEN3_VL_LONG_REFERENCE_SHA256` 显式钉住经复核的外层文件 SHA。Julia
loader 在加载模型前校验 schema/claim、Float32 CPU backend/build、运行前后 exporter
源码 SHA、revision/assets、image/prompt/input IDs、256-step cache/mRoPE 公式、可重算的
Float32 top-2 margin、固定检查点 prefix/28-layer K/V geometry 和前四 token；任一不符
都会 fail closed。

真实 benchmark 入口为：

```bash
LIFEAI_QWEN3_VL_MODEL_DIR=/path/to/Qwen3-VL-2B-Instruct \
LIFEAI_QWEN3_VL_REFERENCE_DIR=/path/to/chapter45-reference \
LIFEAI_QWEN3_VL_LONG_REFERENCE=/path/to/long_generation_reference.json \
LIFEAI_QWEN3_VL_LONG_REFERENCE_SHA256=<reviewed-file-sha256> \
julia --project=. --startup-file=no \
  scripts/benchmark_qwen3_vl_static_long_generation.jl \
  /path/to/Qwen3-VL-2B-Instruct \
  /path/to/chapter45-reference \
  /tmp/qwen3_vl_long_profile.json
```

脚本复用 Chapter 46 的 frozen BF16 preparation，但 `include` verifier 时不会再触发
顶层执行。每个 warmup/sample 的**完整生成 prefix**必须匹配 pinned HF oracle，而不再只
看前四 token。JSON 保存 preparation、源码/Manifest/短 reference/长 oracle/asset 哈希
与被排除的 warmup 指标；正式采样前还执行一次不计入结果的 profiled-path warmup。
输出 schema 显式标记 `closed=false`，在真实结果经复核并提交前不作为历史 acceptance
fixture。可调长度最小为 4，且不能超过 oracle timeline。

## 指标语义

- `CUDA.@timed.gpu_bytes`：CUDA.jl allocator counter 在测量区间内的累计 traffic；
  不是全部 driver/library 分配、当前 live bytes 或物理显存峰值，也不能直接作为可复用
  workspace 的容量。
- `gpu_allocation_count`：同一区间的 GPU allocation 次数。
- `pool_high_watermark.used_bytes/reserved_bytes`：CUDA stream-ordered pool 的 live /
  reserved 高水位。
- `device_free_bytes_before/after`：测量边界快照，不是外部物理显存峰值采样。
- 无 hook 的 steady run 才用于 latency；逐 stage runner 会同步设备，只用于 bytes/count
  归因。
- host greedy selection 单独记录完整 vocabulary 搬运、production-compatible top-2
  `partialsortperm` 与 margin；logits SHA256 在计时区间外计算。不能把这部分时间写成
  decoder kernel latency。
- `memory_drift` 比较 profiled warmup 后与全部 samples 后、均已 full GC 但尚未
  `CUDA.reclaim()` 的 free/used/cached；`reclaim` 只描述显式回收释放量，不冒充 drift。
- attribution 同时按 `(stage, layer)` 和跨层 stage family 聚合，并机械记录当前
  `dominant_stage`；它仍不是 workspace shape/dtype/byte cap 设计。
- oracle comparison 只比较 HF Float32 与 LifeAI BF16 的 greedy token IDs，不声称
  logits/K/V strict parity。若后缀分叉，必须报告 blocker，不能放宽成“两边各自
  deterministic”；脚本会把首个分叉 step、双方 token、HF runner-up/margin 与两条
  prefix 写入请求的输出路径，并标记 `status="correctness_blocker"`。

## 计划

| 工作项 | 所属主线 | 交付物 | 验收方式 | 状态 |
| --- | --- | --- | --- | --- |
| 恢复可导入的 frozen verifier helper | 工程 | Chapter 46 main guard | `include` 无模型加载副作用 | 已完成 |
| decoder stage hook | 高效推理 | internal profiled decode entry | tiny profiled/unprofiled exact、48-stage order | 已完成 |
| 独立 long oracle exporter | 正确性 | streaming HF Float32/CPU 256-token JSON | frozen env/assets、fixed-length free-run、无 K/V snapshot 累积 | 已完成 |
| long oracle loader | 正确性 | pinned SHA + fail-closed schema/timeline contract | synthetic 正/反例默认离线测试 | 已完成 |
| 真实 long oracle 重建 | 正确性 | 两次逐字节相同 JSON 与 reviewed SHA | 完整 256-token timeline、前四 token 延续 Chapter 45 | 已完成 |
| 长生成 benchmark | 高效推理 | `benchmark_qwen3_vl_static_long_generation.jl` | `--help` smoke、schema 自描述 | 已完成骨架 |
| 真实 GPU 32/128/256 运行 | 高效推理 | raw JSON 与环境/源码哈希 | 三次 deterministic samples | warmup 在 step 79 fail closed |
| allocation attribution | 高效推理 | position 107/203/331 stage 明细 | CUDA.jl bytes/count 100% 闭合、与无 hook surface 一致，K/V write 0 bytes | 被 correctness blocker 挡住 |
| workspace 选择 | 模型 / 工程 | dominant stage 与 bounded scratch 设计 | shape、dtype、ownership、byte cap 明确 | 未选择 |

## Close 条件

只有以下条件满足后才能关闭本章：

- 真实 checkpoint 完成 32/128/256-token BF16 static runs，三次 token timeline
  deterministic，前四枚仍为 `[1987, 2169, 375, 265]`。
- 独立 HF Float32/CPU exporter 必须在两个进程得到逐字节相同的 256-token artifact；
  reviewed 外层 SHA 被显式钉住，LifeAI BF16 的 32/128/256 完整 prefix 全部 exact。
- final position、capacity、`rope_delta` 和 56 个 K/V buffer identity 全部通过。
- 冷加载、vision、prefill、decode、host selection 与 reclaim 指标分开报告。
- 三个最终 position 的 profiled step 对 CUDA.jl GPU bytes/count 100% 归因，outer
  bytes/count 与同 position 无 hook step 一致，且 `kv_write` allocation traffic 为零。
- dominant stage 由聚合数据机械选出，并给出 request-local workspace 的最大 shape、
  dtype、byte cap、reset 与并发 ownership。
- 原始结果、源码/资产/reference 哈希和离线 contract test 一起提交。
- 文档明确不声称 BF16 HuggingFace strict parity、zero-allocation whole loop、稳定吞吐、
  batch/padding、multi-image/video 或 sampling 已完成。

## 2026-10-08 真实运行

`/tmp` 里的 Transformers 4.57 overlay 与 Chapter 45 reference 已经不在。仓库
`.venv` 仍是 Transformers `4.51.3`，没有被改写。持久 CPU oracle 放在
`/home/ubuntu/models/lifeai-oracles/qwen3vl-transformers-4.57.0`，版本与
`scripts/qwen3_vl_long_reference_contract.jl` 一致：Python `3.10.12`、NumPy
`1.26.4`、Pillow `11.3.0`、safetensors `0.5.3`、tokenizers `0.22.1`、Jinja2
`3.1.6`、Transformers `4.57.0`、PyTorch `2.7.1+cpu`（git
`e2d141dbde55c2a4370fac5165b0561b6af4798b`，config SHA
`dcb7f1e248794c5c144992643f929d0dc623511210f6fe5615f78cb009d64745`）、
torchvision `0.22.1+cu126`、intra/inter-op threads `24/24`、CPU capability
`AVX2`。导入 torchvision 时仍出现既有的 `libc10_cuda.so` image-extension
warning；NMS workaround 只沿用 exporter 里已有的那一处。该环境重新导出的
Chapter 45 Float32 decode reference 与冻结 SHA 逐字节相同。

| artifact | path | SHA256 |
| --- | --- | --- |
| Chapter 45 tensors | `/home/ubuntu/models/lifeai-references/Qwen3-VL-2B-Instruct/chapter45-decode-f32/reference.safetensors` | `a98812e25efb44c02ab9c06e974ab718724f35f2f1c686e4bdc395d856c03e81` |
| Chapter 45 metadata | `.../chapter45-decode-f32/reference.json` | `569fe3666b65ee2f497327e9ce9931f81652d5bdc32d44dfb9fb774435caccfc` |
| long oracle | `.../chapter47-long-oracle/long_generation_reference.json` | `72461330a8c5a0e74e7489abfdd481223aeb47d6c161fdc12dc0048e82690140` |

两个独立进程写出的 long oracle 经 `cmp` 逐字节相同。前四枚 one-based token 仍是
`[1987, 2169, 375, 265]`，256-token timeline SHA 为
`e4db8b32e2e2a1379b11eac4023d0731b20f314c135c5cfd42bd53e12dfb6f4d`。prompt 仍是
76 tokens，`rope_delta = -56`。

RTX 4090 D（driver `570.153.02`，CUDA.jl `6.2.1` 报告 driver API `13.3.0` /
runtime `12.9.0`，Julia `1.12.6`）上的 BF16 static benchmark 在最长 shape warmup
处 fail closed，没有进入 32/128/256 的三次正式采样，也没有 allocation
attribution。blocker 写在
`.../chapter47-long-oracle/qwen3_vl_long_profile.json`，文件 SHA256 为
`90b10f6cb25489984314bfb02b727557fd25d71a1a1424c08bda98301d00c8d4`，
`status="correctness_blocker"`、`closed=false`、`sample_label="warmup"`。

| 项目 | 值 |
| --- | --- |
| 首个分叉 | generated step `79` |
| 前缀 | 前 `78` 枚与 HF Float32/CPU oracle exact |
| LifeAI | one-based `282` / zero-based `281`，decode 为 ` p` |
| HF top-1 | one-based `2519` / zero-based `2518`，decode 为 ` red` |
| HF runner-up | zero-based `281`，即 LifeAI 的选择 |
| HF top-1 / top-2 / margin | `25.939213 / 25.889927 / 0.04928589` |
| LifeAI timeline SHA | `5a1c88495f544aed31ee27f81ac37cc3fcee6688864398eef8b33b755e5e8f9b` |

step 80 继续分叉（LifeAI one-based `15505`，HF one-based `83`），所以这是时间线
分叉，不是单点标注差异。这条 warmup 的前 78 枚覆盖了 32-token prefix，但正式
三次采样没有执行，不能把这次观察写成 32-token acceptance。blocker schema
不包含 CUDA allocation、pool high-water、latency 或 dominant stage；这些字段
因此保持未测。比较只覆盖 greedy token IDs，不声称 BF16 logits/K/V strict parity、
整条 decode 零分配、稳定吞吐、batch/padding、multi-image/video 或 sampling。

## 当前边界与下一步

Chapter 47 保持 Open，Episode 10 也保持 Open。独立 256-token Float32/CPU oracle
已经冻结，真实 BF16 GPU 运行停在 correctness blocker。下一步先定位 generated
step 79 的 BF16 CUDA greedy 与 HF Float32/CPU timeline 分叉，再恢复
32/128/256 正式采样和 allocation attribution。在完整 prefix exact 之前不选择
workspace，也不把两边各自 deterministic 写成通过。
