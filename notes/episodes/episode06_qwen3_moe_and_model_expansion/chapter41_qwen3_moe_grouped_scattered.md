# Chapter 41 — Qwen3 MoE grouped scattered expert-cache dispatch

> 所属 Episode：Episode 06 — Qwen3 MoE 与模型架构扩展
>
> 状态：Closed
>
> 日期：2026-08-21
>
> 真实资产：`Qwen/Qwen3-30B-A3B@ad44e777bcd18fa416d9da3bd8f70d33ebb85d39`
>
> 设备：NVIDIA GeForce RTX 4090 D，CUDA 12.9，Julia 1.12.6 / 8 threads

## 本章问题

Chapter 28 的 scattered cache 只接入 scalar CUDA kernels。默认的 grouped BF16
WMMA 仍必须先把命中的独立 expert matrices `cat` 成 active 3D tensors；因此
expert read/upload 即使已经受 cache 控制，每次请求仍产生巨量 device-to-device
materialization。Chapter 35 之后留下的最明确 MoE 缺口，是让 grouped WMMA 直接
消费 generation-safe device pointer tables，同时保持原 grouped BF16 舍入口径、
路由、I/O 与生命周期完全不变。

## Close 条件

1. `grouped_experts=true + expert_cache_dispatch=:scattered` 不再被拒绝，CUDA
   WMMA 直接读取分散 BF16 matrices；portable backend 保留 materialized fallback；
2. `m32n8k16` 与 `m16n16k16` 都有 synthetic CUDA exact test，并覆盖非首 expert、
   新 generation/新 pointer、plan 淘汰与 workspace 淘汰；
3. pointer plans 与 grouped workspaces 有界，clear/reconfigure 不会复用 stale
   pointer；
4. 官方 30B 同进程比较 materialized-grouped、grouped-scattered、
   scalar-scattered。前两者必须 logits/routes bitwise exact，且逐次 expert I/O、
   cache hit/miss/eviction exact；
5. 2-token reference 与宽 prefill 都必须给出时延、Julia allocation、logical
   materialization、pointer/workspace bytes 与 GPU memory；
6. scalar-scattered 只作为另一 Float32 activation 数值契约的诊断对照，不把其
   后续路由/I/O差异误写成 grouped correctness 失败。

## 实现

### Pointer-backed WMMA

CUDA extension 新增两组 pointer-table WMMA kernels。每个 route tile 先由 padded
offset 定位 local expert，再从 gate/up/down pointer vector 取矩阵基址。Julia
GPU `LLVMPtr + n` 的 `n` 是 byte offset，因此矩阵内 column-major linear offset
必须乘 `sizeof(BFloat16)`；synthetic test 特意让非首 expert 使用独立 allocations，
避免连续 3D storage 偶然掩盖地址错误。

执行语义与 materialized grouped 路径一致：

1. Float32 token 按 route bucket 写成 padded BF16 token；
2. gate/up 用 BF16 WMMA、Float32 accumulation；
3. SwiGLU 后重新舍入 BF16；
4. down projection 再用 BF16 WMMA、Float32 accumulation；
5. 按原 route 顺序与 Float32 routing weight 合并。

因此 pointer-backed 与 contiguous-3D 路径使用相同 tile、padding 和舍入边界，
真实模型可以要求逐位一致，而不是只设容差。

### 有界状态与失效

- 复用 Chapter 29 的 ordered expert ids + monotonic entry generations pointer plan；
  每层最多 4 份 plan，淘汰/reload 后即使 allocator 重用裸地址也不会命中旧 plan；
- grouped workspace 分开持有 BF16 padded token/hidden、Float32 gate/up/down、inverse
  route 与 final output；不能与 scalar Float32-hidden workspace 混用；
- 相同 `(num_tokens, pair_count, route_tile)` workspace 可 grow-to-fit 后复用，最多
  保留 4 个 token shapes；逻辑 bytes 随替换精确扣减；
- `clear_hf_qwen3_moe_expert_cache!` 与 runtime reconfigure 都丢弃整个 opaque
  state，pointer/workspace bytes 归零；
- cache stats 现在显式报告 `grouped_experts`。`workspace_allocations` 只表示 retained
  workspace 创建/增长，不把每次 route bucketing/layout 临时数组混入此计数。

## Synthetic 与回归验证

- `d_model=32, hidden_dim=32` 覆盖 `m32n8k16`；`hidden_dim=16` 覆盖
  `m16n16k16`。两者对 materialized grouped output 都逐位一致；
- 用不同 BF16 tensor allocations 和不同数值重载同一组 expert，新 generation
  必须 build 新 plan，output 对新 materialized reference exact 且明确不同于旧值；
- 5 个 token shapes 后只保留 4 个 grouped workspaces；6 组 generations 后每层只
  保留 4 plans，重新请求已淘汰的首个 generation 必须 rebuild；
- 原 Chapter 24/28/29/30/35 CUDA 专项保持通过；portable tiny fixture 接受新组合、
  仍走 materialized fallback，runtime configure 可同时切换 grouped mode。

## 真实测量方法

单个 resident session 固定：

- 40,960-token BF16 static KV capacity；
- 8 GiB layer-balanced expert cache、`gc_interval_layers=8`；
- tensor read、8-worker pageable overlap、raw/final host buffer reuse；
- 每个 configuration 先 clear/reconfigure，再 warmup 1 次、计量 3 次；
- workload A 是冻结 HF reference 的 2-token prompt + 1 decode；
- workload B 把同一 token pair 确定性重复到 32 tokens + 1 decode，用于宽 prefill；
- 每次记录 CUDA synchronize 后的 prefill/decode wall time、`@allocated`、
  `/proc/self/io`、GPU allocator snapshot、完整 logits/routes/cache traffic。

复现命令：

```bash
julia --threads=8 --project=. --startup-file=no \
  scripts/benchmark_qwen3_moe_cuda_grouped_scattered.jl \
  MODEL_DIR \
  MODEL_DIR/lifeai-references/chapter24-real-parity/bfloat16 \
  benchmark_results/qwen3_moe_cuda_grouped_scattered/summary.json
```

## 真实结果

### Correctness 与 traffic gate

两个 workload 上，materialized-grouped 与 grouped-scattered 都满足：

- prefill logits bitwise exact，decode logits bitwise exact；
- 48 层 prefill/decode active-expert routes exact；
- 3 个 measured runs 各自重复 exact；
- 每次 measured prefill/decode/total expert bytes read/upload，以及 cache
  hit/miss/eviction 逐项 exact。

2-token grouped-scattered 对冻结 Transformers BF16 reference 的 prefill/decode
max-abs 为 `0.421875 / 0.2890625`，argmax 都一致；这是既有跨框架 BF16 边界，
不是新 pointer path 引入的误差。

scalar-scattered 使用不在 SwiGLU 后舍入 BF16 的 Float32 activation contract，
所以只作诊断：2-token 对 materialized grouped 的 prefill/decode max-abs 为
`0.375 / 0.25`，32-token 为 `0.25 / 0.3125`，四个 argmax 都一致；后续 routes
和 I/O 可以不同，不参加 grouped equality gate。

### 时延与 materialization

| workload | configuration | prefill median | decode median | request median | logical active materialization |
| --- | --- | ---: | ---: | ---: | ---: |
| 2-token | materialized-grouped | `4.053 s` | `1.272 s` | `5.324 s` | `10.560 GB` |
| 2-token | grouped-scattered | `0.258 s` | `0.245 s` | `0.495 s` | `0` |
| 2-token | scalar-scattered | `0.264 s` | `0.276 s` | `0.529 s` | `0` |
| 32-token | materialized-grouped | `16.359 s` | `1.813 s` | `18.155 s` | `16.930 GB` |
| 32-token | grouped-scattered | `2.138 s` | `0.837 s` | `2.975 s` | `0` |
| 32-token | scalar-scattered | `2.049 s` | `0.804 s` | `2.874 s` | `0` |

grouped-scattered 相对同数值契约的 materialized-grouped：

- 2-token prefill/request 加速 `15.707× / 10.745×`；
- 32-token prefill/request 加速 `7.650× / 6.103×`；
- measured hit/miss trace 中分别消除 `10.560 / 16.930 GB` logical active concat；
- retained state 很小：2-token 为 `32,568` pointer bytes + `3,858,528`
  workspace bytes；32-token 为 `85,344` + `11,972,640` bytes；
- warmup 后两种 workload 的 96 次 grouped-scattered dispatch 都是 96 workspace
  reuses、零 retained workspace allocation。

本机同进程 snapshot 中，grouped-scattered 的 minimum GPU free 相对 materialized
在 2-token 多 `234,881,024` bytes（224 MiB），在 32-token 多 `637,534,208`
bytes（608 MiB）。Julia request allocation 中位从 `83,605,632` 降到
`75,927,120` bytes，宽 workload 从 `149,191,552` 降到 `136,900,304`
bytes；主要收益仍是取消 GPU active tensor materialization，而不是宣称剩余
Julia allocation 已全部归因。

## 决策

- **支持并推荐 opt-in**：Ampere+ CUDA、BF16 expert cache 且
  `grouped_experts=true` 时，`expert_cache_dispatch=:scattered` 现在是已验证配置；
- **不改零配置默认**：expert cache 默认预算仍为 0，而 scattered 要求正预算；
  `expert_cache_dispatch` 因此继续默认 `:materialized`，避免改变 CPU/portable 与纯
  streaming 行为。启用正 cache budget 的生产配置应显式选择 `:scattered`；
- 保持 `gc_interval_layers=8` 的长期建议。Chapter 41 没有重新做任意长服务稳定性，
  不用一次快 benchmark 推翻 Chapter 29 的 allocator-tail 证据；
- 不把 scalar 略快的 32-token 数字当成 grouped 回退理由：它是不同数值契约且
  已产生不同后续路由/I/O；需要 grouped BF16 行为时只能比较两条 grouped 路径；
- 40K 本章指实际分配完整 KV capacity，不声称已填满 40K prompt。full-window
  质量/FlashAttention 是独立长上下文专项，不阻塞本章或 Episode Close。

## 已知边界

- offload session 仍是单请求、单 CUDA stream 容器；共享 output workspace 与裸
  pointer plan 没有并发调用语义，不应由多个 task 同时执行同一 session；
- runtime reconfigure 会保留 KV position，因此 grouped 数值模式最好只在下一次
  prefill/request 边界切换；benchmark 的每次 measured request 都先由 prefill reset；
- 4-shape 是 retained object 数量上限，不是独立 byte budget。默认 128-token chunk
  的单个保守上界约 37.6 MiB，四个约 150 MiB；本章实际最大仅 11.97 MiB。若未来
  开放超大 prefill chunks，应再加入 workspace byte cap；
- route bucketing、padded offsets 与 layout 的临时 CuArrays 不计入 retained
  `workspace_bytes`。它们是剩余 Julia/CUDA allocator traffic 的已知 attribution，
  但不影响本章“取消 active expert concat”的结论。

## Episode 方向与 Close

Episode 06 的三个 Close 条件现在全部满足：官方 30B parity 已冻结；active-expert
sparse/offload/cache 有真实性能与内存证据；后续 Qwen 架构方向确定为
**Qwen3-VL**，因为它在复用现有 Qwen3 decoder 的同时补上视觉 encoder、跨模态
projector 与 image-token layout 这组尚未验证的架构缺口。reranker 更适合归入
长期记忆/检索评测，不作为下一模型本体章节。

按当前用户优先级，模型扩展不立即启动 Qwen3-VL；Chapter 41 关闭 Episode 06
后回到 Episode 08，下一章实现环境事件记忆写回与检索，再处理跨 adapter safety
semantics。

## 验证与冻结资产

- 新 grouped pointer CUDA synthetic/lifecycle：`47 / 47`；
- 既有 Episode 06 CUDA 回归：`181 / 181`；
- Chapter 41 真实结果 contract：`60 / 60`；
- 真实 30B：2 workloads × 3 configurations × 1 warmup + 3 measured runs，最终
  verification gate 全部通过；
- 冻结报告：
  `benchmark_results/qwen3_moe_cuda_grouped_scattered/summary.json`；
- 报告绑定 benchmark script、offload implementation 与 CUDA extension 的 SHA256；
  旧 Chapter 25–35 timing 的历史源码 provenance 单独按原运行 commit 恢复，不把
  Chapter 41 新 hash 追写进旧报告。
