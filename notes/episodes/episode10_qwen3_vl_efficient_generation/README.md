# Episode 10 — Qwen3-VL 高效生成

> 状态：Open
>
> 收录章节：Chapter 46–

## 本卷主题

Episode 09 已经关闭单图、batch-1、全一 attention mask 下从 raw image 到
greedy text 的 correctness 闭环，但 Chapter 45 的 dynamic KV cache 每次 decode
都通过 `cat` 重新分配并复制历史 prefix。Episode 10 把问题从“能否正确生成”推进到
“能否在显式资源预算内稳定复用生成状态”。

本卷首先把 Qwen3-VL K/V 改成调用方指定容量的预分配 storage，再逐步处理仍然存在的
attention、projection、RoPE、logits 和高层 generation 临时分配。这里的“static”首先
指 K/V backing storage 的物理 shape 和地址固定，不自动等价于整条执行图静态编译、
CUDA Graph、零分配 decode 或长上下文部署已经完成。

## 章节目录

1. [`Chapter 46 — Qwen3-VL bounded static KV cache 与可复用 generation state`](chapter46_qwen3_vl_static_cache.md)（Closed）
2. [`Chapter 47 — Qwen3-VL 长生成 allocation profile 与 decoder workspace 归因`](chapter47_qwen3_vl_long_generation_profile.md)（Closed）
3. [`Chapter 48 — Qwen3-VL decode mlp_activation 常量分配`](chapter48_qwen3_vl_mlp_activation.md)（Closed）
4. [`Chapter 49 — Qwen3-VL decode qk_rope 常量分配`](chapter49_qwen3_vl_qk_rope.md)（Closed）
5. [`Chapter 50 — Qwen3-VL decode qk_norm 常量分配`](chapter50_qwen3_vl_qk_norm.md)（Closed）
6. [`Chapter 51 — Qwen3-VL decode hidden RMSNorm 常量分配`](chapter51_qwen3_vl_hidden_norm.md)（Closed）
7. [`Chapter 52 — Qwen3-VL decode gate/up projection 常量分配`](chapter52_qwen3_vl_gate_up.md)（Closed）
8. [`Chapter 53 — Qwen3-VL decode hidden residual 常量分配`](chapter53_qwen3_vl_hidden_residual.md)（Closed）
9. [`Chapter 54 — Qwen3-VL decode tied logits 常量分配`](chapter54_qwen3_vl_tied_logits.md)（Closed）
10. [`Chapter 55 — Qwen3-VL decode QKV projection 常量分配`](chapter55_qwen3_vl_qkv.md)（Closed）
11. [`Chapter 56 — Qwen3-VL decode attention softmax 归约`](chapter56_qwen3_vl_attention_softmax.md)（Closed）
12. [`Chapter 57 — Qwen3-VL decode token embedding 常量分配`](chapter57_qwen3_vl_token_embedding.md)（Closed）
13. [`Chapter 58 — Qwen3-VL decode attention batched GEMM 的 64 bytes`](chapter58_qwen3_vl_batched_gemm.md)（Closed）
14. [`Chapter 59 — Qwen3-VL decode gemmEx 的持久 alpha/beta`](chapter59_qwen3_vl_gemm_scalar.md)（Closed）
15. [`Chapter 60 — Qwen3-VL decode hidden RMSNorm partial`](chapter60_qwen3_vl_norm_partial.md)（Closed）
16. [`Chapter 61 — Qwen3-VL decode mRoPE cos/sin 表`](chapter61_qwen3_vl_mrope.md)（Closed）
17. [`Chapter 62 — Qwen3-VL cache 重叠判断不再调用 dataids`](chapter62_qwen3_vl_storage_overlap.md)（Closed）
18. [`Chapter 63 — Qwen3-VL decode RoPE 合成 kernel`](chapter63_qwen3_vl_fused_rope.md)（Closed）
19. [`Chapter 64 — Qwen3-VL decode QK RMSNorm 合成 kernel`](chapter64_qwen3_vl_fused_qk_norm.md)（Closed）
20. [`Chapter 65 — Qwen3-VL decode hidden RMSNorm 元素合成 kernel`](chapter65_qwen3_vl_fused_hidden_norm.md)（Closed）
21. [`Chapter 66 — Qwen3-VL decode hidden residual 合成 kernel`](chapter66_qwen3_vl_fused_residual.md)（Closed）
22. [`Chapter 67 — Qwen3-VL decode MLP SiLU 合成 kernel`](chapter67_qwen3_vl_fused_mlp.md)（Closed）
23. [`Chapter 68 — Qwen3-VL decode attention softmax 合成 kernel`](chapter68_qwen3_vl_fused_attention.md)（Closed）

## 预期能力变化

- **显式资源预算**：低层 static cache 必须给出 `capacity`，按 dtype、device、层数、
  KV heads 和 batch 一次性分配，拒绝隐式使用官方超长 context 上限。
- **稳定请求状态**：prefill/decode 只写有效 prefix，底层 K/V storage identity 和
  logical bytes 不随 token 增长；reset 后可在同一 allocation 上开启新请求。
- **保持 correctness**：static 与 Chapter 45 dynamic/HF oracle 的有效 K/V prefix、
  logits、mRoPE timeline 和 greedy token 必须一致。
- **逐项关闭分配**：把“已消除 K/V `cat`”与“尚存 decoder workspace/临时数组”分开
  记账，再以真实 GPU 长生成证据决定下一章优化对象。

## 当前进展

Chapter 46 已 Closed。`Qwen3VLStaticKVCache` 为每层预分配
`(head_dim, kv_heads, capacity, batch)` K/V，`position` 只标记有效 prefix；prefill
和单 token decode 原地写入，同一 cache 与每层数组 identity 均保持不变。
`reset_qwen3_vl_static_kv_cache!` 默认只重置 logical state，也可用 `clear=true`
显式清零 storage。低层初始化要求显式 `capacity`；高层 `cache=:static` 在调用方未
指定 `static_capacity` 时只推导本次 prompt 与 `max_new_tokens` 所需的精确容量，
不会默认分配官方最大 context。

Chapter 45 的 deterministic HF `DynamicCache` fixture 被继续用作独立 oracle。
static prefill 与两步 decode 的所有层有效 K/V prefix、hidden/logits 和 greedy
`[8,8,8]` 均与 HF/dynamic baseline 一致；capacity `8/9/10`、overflow 原子性、
storage bytes、reset/reuse 与错误契约进入默认离线回归。

真实 RTX 4090 D Float32 verifier 也已通过：capacity 128 的 56 个 K/V buffers 在
prefill、三次 decode 和 reset 中保持 object/device-pointer identity 且互不 alias；
static/dynamic 的有效 prefix 与 logits bitwise 一致，四个 HF reference max-abs 仍为
`3.86238e-5 / 4.00543e-5 / 2.95639e-5 / 2.90871e-5`。32-token 对照中
static 相对 dynamic 少报告 `624,820,224` GPU allocated bytes，但 static 全程仍有约
`3.65 GB` 临时分配；三个 latency 样本只作观察，不是 acceptance gate。
BF16 same-device smoke 的 static/dynamic K/V 与 logits 也 bitwise 一致，32-token
allocated bytes 少 `312,410,112`；相对 HF Float32 oracle 只记录 cross-dtype
boundary，不声称 BF16 strict parity。最终交替顺序三样本的 dynamic/static latency
中位为 `0.849395 / 0.840655 s`，两者接近且对顺序/GC 敏感，因此不作为 gate 或
普遍加速结论。

Chapter 47 已 Open。它先加入数值透明的 internal decode-stage runner，把 request
与 28 个 decoder blocks 拆成 token embedding、mRoPE、Q/K/V projection、QK-Norm、
RoPE、K/V write、attention、O projection/residual、MLP、final norm 和 vocabulary
logits 等阶段；tiny profiled/unprofiled decode 的 logits、K/V 与状态 exact，调用顺序
进入默认离线回归。2026-10-08 已在持久 CPU oracle 上由两个独立进程导出逐字节相同的
256-token Float32/CPU timeline，外层 SHA 为 `72461330…90140`，前四枚仍是
`[1987, 2169, 375, 265]`。RTX 4090 D 的 BF16 static 256-token warmup 在 generated
step 79 fail closed：前 78 枚 exact，LifeAI 选了 zero-based `281`（` p`），HF
Float32 top-1 是 `2518`（` red`），runner-up 正是 `281`，margin `0.04928589`。
正式 32/128/256 三次采样和 allocation attribution 没有执行。随后的同前缀拆分表明
这是跨 dtype 边界，不是 position/mRoPE/cache length 错误：LifeAI Float32 CUDA
79/79 argmax 与 HF 相同，BF16 只在 step 79 把 `25.939213 / 25.889927` 落成
`26 / 25.875` 后选中 runner-up。在 HuggingFace token 的 teacher-forced 前缀上，
Float32 CUDA 256/256 仍一致；BF16 只在 step 79 和 step 95 离开，后者两个候选
都落成 `28.375`。一次非验收、单 repeat 的 BF16 profile 在 position `107 / 203 / 331`
上把最后一步的 CUDA.jl bytes 和 allocation count 与无 hook 表面逐项对齐，
`kv_write` 为 0；当时 dominant stage 都是 `attention`。随后 BF16 单 token decode
把这条随 position 增长的 K/V permute 和 score/softmax 收进按 capacity 预分配、
28 层共用的 scratch。重测后 position `107 / 203 / 331` 的 attention 都是
`7,168` bytes，整步合计都是 `10,563,721` bytes。最大头变成不随 position
增长的 `mlp_activation`（`2,752,512` bytes）。三次 repeat 的关闭采样里，32-token
prefix 与 HF exact，128/256 在 step 70 分叉且三次 timeline 各自 deterministic。
Chapter 47 因此 Closed。这次关闭不声称 BF16 strict parity 或整条 decode 零分配。

Chapter 48 已 Closed。融合 `mlp_activation` 是每层 5 个固定形状数组，合计
`98,304` bytes：3 个 Float32 与 2 个 BF16，公式为
`intermediate_size * (3 * sizeof(Float32) + 2 * sizeof(BFloat16))`。
`intermediate_size = 6144` 时 28 层原来各分配一次，单步 `2,752,512` bytes、
`140` 次分配。BF16 单 token decode 现在把这 5 块 buffer 按 intermediate size
预分配，28 层共用，reset 保留 storage；prefill 和 Float32 不创建它。同进程
SiLU 乘积与原融合表达式逐位一致，预热后的复用调用是 `0` GPU bytes。
2026-10-08 一次 repeat 的重测在
`qwen3_vl_allocation_mlp_scratch.json`，SHA256 为
`4aa2057607d940a12901bb3d71f8498a9fff1c6d548e597d07c8d7c3a22906de`。
position `107 / 203 / 331` 的 `mlp_activation` 都是 `0` bytes，整步合计都是
`7,811,209` bytes、`3,628` 次分配，正好少了原来的 `2,752,512 / 140`。
没有新的线性头。最大常量 stage 变成 `qk_rope` 的 `2,265,088` bytes。
32-token prefix 仍与 HF exact，128/256 仍在 step 70 分叉，margin
`0.02797699`。最终 logits 没有升成 Float32。

Chapter 49 已 Closed。单次 Q 的 `_bf16a_apply_rope` 是 `53,760` bytes、`19`
次分配，单次 K 是 `27,136` bytes、`19` 次分配，合计正好是每层的 `80,896 / 38`。
BF16 单 token decode 把这些 half 切片、Float32 提升和中间 BF16 舍入写进按
query heads 预分配的 scratch，28 层共用，K 使用 head 前缀，reset 保留 storage。
Q 与 K 的输出保持两块，因为它们要同时活到 `kv_write` 和 attention。prefill、
Float32 和多 token 仍走原来的 `_bf16a_apply_rope`，该共享函数没有修改。同进程
结果逐位一致，拆开后的复用调用是 `0` GPU bytes。驻留 scratch 是 `37,376`
bytes。2026-10-08 一次 repeat 的重测在 `qwen3_vl_allocation_qk_rope.json`，
SHA256 为 `d8726e41c1171fe42b823254bb0ccb3fde40baf547c700dd87bda228f0c398d8`。
position `107 / 203 / 331` 的 `qk_rope` 都是 `0` bytes，整步合计都是
`5,546,121` bytes、`2,564` 次分配，正好少了原来的 `2,265,088 / 1,064`。
没有新的线性头。最大常量 stage 变成 `qk_norm` 的 `1,066,240` bytes。
32-token prefix 仍与 HF exact，128/256 仍在 step 70 分叉，margin
`0.02797699`。最终 logits 没有升成 Float32。

Chapter 50 已 Closed。单次 Q RMSNorm 是 `25,216` bytes、`7` 次分配，单次 K 是
`12,864` bytes、`7` 次分配，合计每层 `38,080 / 14`。BF16 单 token decode 用
`sum!(abs2, dest, x)` 和预分配 broadcast 重放同一串舍入，28 层共用，K 使用
head 前缀，Q/K 输出保持两块，reset 保留 storage。prefill 和 Float32 仍走
`_qwen3_vl_text_rmsnorm`。同进程结果逐位一致，复用调用是 `0` GPU bytes。驻留
scratch 是 `27,200` bytes。重测 SHA256 为
`8f415d759aca2cf27a402c484c2c6c2c0a06624a25c64bceb6f74681f25afdad`。position
`107 / 203 / 331` 的 `qk_norm` 都是 `0`，整步都是 `4,479,881` bytes、`2,172`
次分配，正好少了 `1,066,240 / 392`。没有新的线性头。最大的并列常量阶段是
`pre_attention_norm` 和 `post_attention_norm`，各 `918,176` bytes。32-token
prefix 仍与 HF exact，128/256 仍在 step 70 分叉。最终 logits 没有升成 Float32。

## Episode Close 条件

- bounded K/V storage 在真实 Qwen3-VL checkpoint 与目标 GPU 上证明地址、容量、
  dtype/device、有效 prefix 和 token/text correctness。（Chapter 46 主体已满足）
- 对长于 Chapter 45 四 token smoke 的生成轨迹给出 steady allocation、latency、
  峰值显存与 capacity 边界报告，区分冷编译、模型加载、vision prefill 和 decode。
- 识别并有界复用主要 per-token decoder workspace；若暂不消除，必须量化并记录
  剩余分配来源，不能把 fixed K/V storage 写成 whole-loop zero allocation。
- 为 BF16 strict oracle、长上下文和失败恢复建立各自的门禁；batch/padding、
  multi-image/video 与 sampling 若不在本卷处理，继续作为显式非目标保留。

## 为什么仍是 Open

Chapter 46 关闭的是 K/V growth allocation：它消除了 dynamic `cat` 对历史 prefix 的
重复分配与复制，并提供可 reset 的有界 request state。但 attention、线性投影、mRoPE、
MLP、final norm、vocabulary logits 和 generation trace 仍会产生临时对象；当前也没有
CUDA Graph/XLA static executable、并发 request scheduler 或完整 BF16 strict
oracle。Chapter 47 已 Closed：decode attention 的 position 线性分配已收进 scratch，
三次 repeat 的最后一步归因闭合，`kv_write` 为 0。128/256 的 BF16 自由生成仍在
step 70 与 HF Float32 分叉，这是跨 dtype 边界，不是未关闭的 allocation 账。
Chapter 48 已把不随 position 增长的 `mlp_activation` 收成 `98,304` byte 的跨层
scratch，该 stage 在三个 position 上都是 `0` GPU bytes。Chapter 49 已把
`qk_rope` 收成 `37,376` byte 的跨层 scratch，该 stage 在三个 position 上都是
`0` GPU bytes。Chapter 50 已把 `qk_norm` 收成 `27,200` byte 的跨层 scratch，该 stage
也是 `0` GPU bytes。Chapter 51 已把 hidden RMSNorm 的大数组收成 `32,772` byte
的 scratch。`pre_attention_norm` 和 `post_attention_norm` 各剩 `448` bytes，
`final_norm` 剩 `16` bytes，都是 `sum!` 的多 block partial。最大常量头变成 `mlp_gate_up_projection`（`688,576` bytes）。Chapter 52 把 gate
和 up 的输出收进 `24,576` byte scratch，该阶段降到 `448` bytes。Chapter 53
把两处 hidden residual 收进另一块 `24,576` byte scratch，两处都降到 `224`
bytes。整步现在是 `14,017` bytes。最大阶段是 `attention` 剩下的 `7,168`
bytes，每层 `256` bytes。Chapter 56 把 attention softmax 的 `maximum` 和
`sum` 改成写入已有 buffer。该阶段降到 `3,584` bytes，整步是 `10,433` bytes。
最大阶段变成 `token_embedding`（`4,105` bytes）。Chapter 57 把这一列写进
`4,096` byte 的 scratch，该阶段变成 `0`。整步现在是 `6,328` bytes。最大阶段
回到 `attention` 的 `3,584` bytes，来自每层两次 `batched_mul!` 各 `64` bytes。
Chapter 58 试过一次 strided batched GEMM：分配能降到每调用 `8` bytes，但
sequence `331` 有一个元素差 `0.0625`，自由生成分叉回到 step 79。该替换没有
留下。Chapter 59 保留 per-head 顺序，只把同一条 `cublasGemmEx` 的 alpha/beta
换成进程内持久标量。`attention`、`qkv_projection`、`mlp_gate_up_projection`、
两处 residual 和 `vocab_logits` 都降到 `0`。整步当时是 `1,168` bytes。
Chapter 60 复用 `sum!` 自己的多 block partial，归约顺序不变。三处 hidden
norm 都降到 `0`。整步当时是 `256` bytes，只剩 `mrope_prepare`。Chapter 61
把单 token 的 cos/sin 表收进 `256` byte scratch。position `107 / 203 / 331`
的最后一步现在是 `0` GPU bytes。同一次生成只有第一次 decode 按 capacity
创建 scratch；256-token 那次是 `2,018,180` bytes、`56` 次分配，其后每步是
`0`。Chapter 62—67 把 cache 重叠判断和 RoPE、QK norm、hidden norm、residual、
MLP SiLU 的元素广播收成 kernel。GPU 最后一步仍是 `0`。256-token 稳态一步的
CPU 分配从 `7,586,128` bytes 降到 `2,168,720`。Chapter 68 试过把 attention
softmax 的元素步收成 kernel，输出和原来的广播不是逐位一致，所以没有留下。
Episode 10 保持 Open，不能把这些 scratch 写成整条 decode 零分配。
