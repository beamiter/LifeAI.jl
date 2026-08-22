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
2. [`Chapter 47 — Qwen3-VL 长生成 allocation profile 与 decoder workspace 归因`](chapter47_qwen3_vl_long_generation_profile.md)（Open）

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
进入默认离线回归。新的真实 benchmark 固定 76-token 单图 prompt 与
32/128/256-token BF16 static generation，分别记录 allocation traffic、allocation
count、pool high-water mark、host greedy selection 和无 hook latency。真实报告与
第 5–256 token 的独立 correctness oracle 尚未采集，因此本章和 Episode 10 都保持
Open，不提前选择或宣称 workspace 优化对象。

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
CUDA Graph/XLA static executable、并发 request scheduler、长生成 benchmark 或完整
BF16 strict oracle。Chapter 47 已提供真实 allocation profile 的可复现入口，但冻结
32/128/256-token 报告与 workspace 决策尚未完成。因此 Episode 10 保持 Open，不能仅凭
固定 K/V shape 或 profiler 骨架宣称部署优化完成。
