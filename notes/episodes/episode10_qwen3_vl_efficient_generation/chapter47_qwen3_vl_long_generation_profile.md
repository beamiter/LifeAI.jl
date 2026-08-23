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
`runner=nothing`，不会把 profiler 变成公共生成契约。

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
| 独立 long oracle exporter | 正确性 | streaming HF Float32/CPU 256-token JSON | frozen env/assets、fixed-length free-run、无 K/V snapshot 累积 | 已完成骨架 |
| long oracle loader | 正确性 | pinned SHA + fail-closed schema/timeline contract | synthetic 正/反例默认离线测试 | 已完成 |
| 真实 long oracle 重建 | 正确性 | 两次逐字节相同 JSON 与 reviewed SHA | 完整 256-token timeline、前四 token 延续 Chapter 45 | 待执行 |
| 长生成 benchmark | 高效推理 | `benchmark_qwen3_vl_static_long_generation.jl` | `--help` smoke、schema 自描述 | 已完成骨架 |
| 真实 GPU 32/128/256 运行 | 高效推理 | raw JSON 与环境/源码哈希 | 三次 deterministic samples | 待执行 |
| allocation attribution | 高效推理 | position 107/203/331 stage 明细 | CUDA.jl bytes/count 100% 闭合、与无 hook surface 一致，K/V write 0 bytes | 待执行 |
| workspace 选择 | 模型 / 工程 | dominant stage 与 bounded scratch 设计 | shape、dtype、ownership、byte cap 明确 | 待数据 |

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

## 当前边界与下一步

当前环境已验证 profiling hook 的离线数值透明性和 benchmark 的参数解析，但没有本章
要求的冻结模型目录与 Chapter 45 reference，因此尚未生成真实长轨迹报告。Chapter 47
保持 Open；独立 oracle 的 exporter/loader 已落地，但真实 256-token artifact、双进程
重建、目标 GPU 运行、attribution closure 和 workspace shape/dtype/byte cap 都仍待
完成。下一步是在冻结 CPU oracle 环境生成并复核长轨迹，再把 reviewed SHA 交给目标
GPU benchmark；若 Float32/BF16 token suffix 分叉，先把分叉记录为 correctness blocker。
