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
   长度保持 `76`，最终 cache position 分别为 `107 / 203 / 331`。
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

真实 benchmark 入口为：

```bash
LIFEAI_QWEN3_VL_MODEL_DIR=/path/to/Qwen3-VL-2B-Instruct \
LIFEAI_QWEN3_VL_REFERENCE_DIR=/path/to/chapter45-reference \
julia --project=. --startup-file=no \
  scripts/benchmark_qwen3_vl_static_long_generation.jl \
  /path/to/Qwen3-VL-2B-Instruct \
  /path/to/chapter45-reference \
  /tmp/qwen3_vl_long_profile.json
```

脚本复用 Chapter 46 的 frozen BF16 preparation，但 `include` verifier 时不会再触发
顶层执行。JSON 保存 preparation、源码/Manifest/reference/asset 哈希与被排除的 warmup
指标；正式采样前还执行一次不计入结果的 profiled-path warmup。输出 schema 显式标记
`closed=false`，在真实结果经复核并提交前不作为历史 acceptance fixture。可调长度最小
为 4，确保冻结的四 token 前缀门禁始终有定义。

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

## 计划

| 工作项 | 所属主线 | 交付物 | 验收方式 | 状态 |
| --- | --- | --- | --- | --- |
| 恢复可导入的 frozen verifier helper | 工程 | Chapter 46 main guard | `include` 无模型加载副作用 | 已完成 |
| decoder stage hook | 高效推理 | internal profiled decode entry | tiny profiled/unprofiled exact、48-stage order | 已完成 |
| 长生成 benchmark | 高效推理 | `benchmark_qwen3_vl_static_long_generation.jl` | `--help` smoke、schema 自描述 | 已完成骨架 |
| 真实 GPU 32/128/256 运行 | 高效推理 | raw JSON 与环境/源码哈希 | 三次 deterministic samples | 待执行 |
| allocation attribution | 高效推理 | position 107/203/331 stage 明细 | CUDA.jl bytes/count 100% 闭合、与无 hook surface 一致，K/V write 0 bytes | 待执行 |
| workspace 选择 | 模型 / 工程 | dominant stage 与 bounded scratch 设计 | shape、dtype、ownership、byte cap 明确 | 待数据 |

## Close 条件

只有以下条件满足后才能关闭本章：

- 真实 checkpoint 完成 32/128/256-token BF16 static runs，三次 token timeline
  deterministic，前四枚仍为 `[1987, 2169, 375, 265]`。
- 第 5–256 token 必须增加独立于当前 static 路径的冻结 correctness oracle；仅靠同实现
  重复 deterministic 或 profiled/unprofiled 一致，不能排除稳定错误。
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
保持 Open；长 suffix 的独立 oracle、真实运行、attribution closure 和 workspace
shape/dtype/byte cap 都仍待完成。下一步是在选定的目标 GPU 上先生成/冻结独立长轨迹
oracle，再运行脚本并据数据决定 Chapter 48 的第一个 workspace reuse 对象。
