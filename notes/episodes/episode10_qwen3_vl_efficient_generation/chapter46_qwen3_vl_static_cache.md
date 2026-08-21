# Chapter 46 — Qwen3-VL bounded static KV cache 与可复用 generation state

> 所属 Episode：Episode 10 — Qwen3-VL 高效生成
>
> 状态：Closed
>
> 日期：2026-08-21
>
> 真实资产：`Qwen/Qwen3-VL-2B-Instruct`
>
> ModelScope revision：`ae9985b208c074c10cfbe3a61b5cb7268cdc9c53`
>
> Hugging Face revision：`78448d793a7eb2f7a987a1da76d464384aa1becd`
>
> Oracle：Chapter 45 frozen Transformers `4.57.0` `DynamicCache` fixture 与
> LifeAI dynamic cache
>
> 设备：CPU 离线 oracle；NVIDIA GeForce RTX 4090 D Float32 strict 与
> BFloat16 same-device smoke

## Closed：核心问题

Chapter 45 已证明 Qwen3-VL cached prefill、mRoPE decode 与单图 greedy generation
正确，但 dynamic cache 每追加一个 token、每一层都用 `cat` 创建更大的 K/V，并复制
全部历史 prefix。它适合关闭 correctness，不适合作为长生成 storage 策略。

本章回答一个更窄也更工程化的问题：能否让调用方在请求开始前给出物理容量，一次性
分配每层 K/V，在 prefill、decode 和跨请求 reset 中保持 backing arrays 不变，同时
继续复用 Chapter 45 的 HF/dynamic 数值契约？

答案是可以。LifeAI 现在提供 batch-1 的 bounded static Qwen3-VL cache、原地
prefill/decode、显式 reset，以及高层 generation 的 `cache=:static` 选择。tiny
oracle 覆盖 prefill 与两次 decode 的所有层有效 K/V prefix、hidden/logits、满容量
行为和 greedy timeline。这里的结论严格限定为 **K/V storage 固定容量、固定 identity**；
它不是整条 decoder 执行图零分配或静态编译的证明。

## Close 条件

1. 每层 K/V 的物理 layout 固定为
   `(head_dim, num_key_value_heads, capacity, batch)`，只让 `position` 前缀参与
   attention；K 仍保存 QK-Norm+mRoPE 后结果，V 仍保存 projection 后结果。
2. 低层初始化必须显式给出合法 `capacity`，cache dtype/device 与 text embedding
   一致；batch-1、Float32/BFloat16 之外 fail closed。
3. static prefill 与至少两次 decode 的 hidden/logits 和所有层有效 K/V prefix
   与 frozen HF `DynamicCache` fixture、LifeAI dynamic path 一致。
4. prefill/decode 返回同一个 mutable request state，所有 layer K/V array identity
   和总 storage bytes 保持不变；满容量失败不得推进 position 或破坏 prefix。
5. reset 默认只重置 `position/rope_delta`，可选择清零 storage，且 reset 后同一
   allocation 可重新 prefill/decode 并得到相同结果。
6. token-level 和 raw single-image generation 可以显式选择 static cache，容量不足、
   非法 cache mode 和 dynamic/static 参数混用均 fail closed。
7. 文档明确 fixed K/V storage 与 zero-allocation decode、static graph、长上下文、
   BF16 strict parity、batch/padding、multi-image/video 和 sampling 的边界。

## 数据结构与容量契约

每层 storage 为：

```text
(head_dim, num_key_value_heads, capacity, batch)
```

`Qwen3VLStaticKVCache` 是一个 mutable request state，包含：

- `layers`：固定数量的 `Qwen3VLStaticLayerKVCache`；
- `position`：已经物理写入、允许 attention 读取的 token 数；
- `rope_delta`：由 multimodal prompt 固化的请求级 mRoPE 偏移；
- `batch_size`：本章只允许 `1`；
- `capacity`：调用方选择的 K/V 物理 token 容量。

低层构造必须写出容量：

```julia
cache = init_qwen3_vl_static_kv_cache(
    text_parameters;
    capacity=256,
    batch_size=1,
)
```

省略 `capacity` 会直接失败。这样做是刻意的：官方 config 的最大 context 很大，
若低层 API 默认按上限分配，单个请求就可能在调用方没有意识到的情况下占用巨量显存。
高层 generation 则掌握 prompt length 和 `max_new_tokens`，因此
`cache=:static, static_capacity=nothing` 可以安全地推导精确的 processed capacity：

```text
required_capacity = prompt_length + max(0, max_new_tokens - 1)
```

这里减一仍来自 Chapter 45 的 timeline：第一枚输出 token 直接从 prefill logits
选择，只有前 `N-1` 枚生成 token 会作为 decode 输入写入 K/V。调用方也可以显式给出
更大的 `static_capacity`，用于后续 reset/reuse；小于 required capacity 会在执行前拒绝。

## 原地 prefill、decode 与有效前缀

公共低层 API 为：

```julia
prefill, same_cache = hf_qwen3_vl_text_prefill_static(
    text_parameters,
    input_ids,
    rope_layout;
    vision_features,
    cache,
    logits_to_keep=1,
)

logits, same_cache = hf_qwen3_vl_text_decode_step_static(
    text_parameters,
    token,
    cache,
)
```

prefill 将第 `1:prompt_length` 段写入每层 storage，随后设置 physical `position`
和 request-local `rope_delta`。decode 在计算前使用：

```text
mrope_coordinate_0_based = cache.position + cache.rope_delta
write_slot_1_based = cache.position + 1
valid_attention_prefix = 1:write_slot_1_based
```

每层通过 `copyto!` 写一个预先存在的 view，attention 只看到 valid prefix；capacity
后方的 storage 即使保留旧请求字节也不参与计算。写完所有层后才推进 logical
position。容量已满、decode token 非法、prompt 超长、layout/delta 不一致等可预检错误
都在写入前失败。

static 与 dynamic 共用 projection、QK-Norm、mRoPE、attention、residual 和 SwiGLU
数学核心，区别只在 K/V store policy：dynamic 返回增长后的新 arrays，static 写入固定
storage view。这让两条路径的差异收敛到 cache ownership，而不是复制一份 decoder
计算实现。

## reset 与跨请求复用

```julia
reset_qwen3_vl_static_kv_cache!(cache)
reset_qwen3_vl_static_kv_cache!(cache; clear=true)
```

默认 `clear=false` 只把 `position` 和 `rope_delta` 归零。旧 K/V 物理字节仍存在，
但新的 valid prefix 从空开始，因此不会被 attention 读取；这条路径保持 reset 成本与
token 数无关。`clear=true` 额外把所有 K/V buffer 清零，适合需要显式数据卫生的调用方，
代价与完整 capacity 成正比。两种模式都返回同一个 cache object，不替换 layer arrays。

本章没有实现同一 cache 的并发写。cache 是单请求可变状态；服务层若要并发，必须为
每个 in-flight request 提供独立 cache，或在外层完成严格串行化。

## logical bytes 与已消除的复制

static K/V 的 logical bytes 为：

```text
2 × layers × head_dim × num_kv_heads × capacity × batch × sizeof(dtype)
```

官方 Qwen3-VL-2B text decoder 使用 28 layers、head dim 128、8 KV heads、batch 1：

| dtype | 每个 capacity token | capacity 79 |
| --- | ---: | ---: |
| Float32 | `229,376` bytes（`224 KiB`） | `18,120,704` bytes |
| BFloat16 | `114,688` bytes（`112 KiB`） | `9,060,352` bytes |

capacity 79 正好覆盖 Chapter 45 的 76-token prompt 与 4-token generation
（prefill 加三次 decode）。dynamic 路径在每一步只保留当前长度的 logical payload，
但历史 allocations 与 prefix copy 已经发生；static 路径在开始时一次性支付 capacity
storage，并在 decode 中只复制新 token 的 K/V slice。其代价是未使用的 tail 也占物理
容量，所以调用方必须基于请求 budget 选择 capacity，而不是盲目取最大 context。

这些数字只计算 K/V tensor payload，不包含 allocator metadata、模型参数、vision
features、attention scores、Q/K/V projections、MLP intermediates、mRoPE arrays、
vocabulary logits、generation trace 或 CUDA workspace。

## frozen oracle 与离线回归

Chapter 46 不生成一份“static HF cache”伪 reference。HF `DynamicCache` 已经冻结
同一个数学状态；static 是 LifeAI 的 storage policy。测试直接复用 Chapter 45 fixture：

```text
test/episodes/episode09_qwen3_vl_multimodal_perception/
  chapter45_qwen3_vl_dynamic_decode/fixtures/tiny_text_dynamic_decode.json
```

fixture SHA256 为
`7b20111e43aa9efd2aae0be3f4a740ab1fdeaff9bd0a6ffd0bfef49adfeeffd8`，包含 4-layer
tiny model 的 prefill、decode.0、decode.1 hidden/logits 和全部 K/V snapshot。
Chapter 46 的测试从同一输入同时运行 static 与 dynamic：

- prefill cache length `8`、`rope_delta=-2`；
- 两次 decode 后 length `9/10`，三轴坐标 `6/7`；
- 每层 static valid K/V prefix 与 HF fixture、LifeAI dynamic cache 数值一致；
- 三阶段 top-1 均为 LifeAI 1-based token `8`，greedy timeline 为 `[8,8,8]`；
- capacity `8/9/10` 分别覆盖 prompt-only、一次 append 与两次 append 的精确边界；
- 每一步 cache object、K/V array identity 与 total storage bytes 不变；
- overflow、空 cache decode、重复 prefill、非法 token、坏 `rope_delta`、坏 capacity
  和 batch 2 都 fail closed，并检查失败后 logical state/prefix 不被推进；
- reset 后在原 allocation 重跑 prefill/decode，hidden/logits 与 frozen oracle 保持一致；
  `clear=true` 还验证所有 storage 归零但 identity 不变；
- public `generate_hf_qwen3_vl_tokens(...; cache=:static)` 与 dynamic 模式的 ids、
  trace logits、stop reason 和最终 cache timeline 一致。

Chapter 46 默认离线专项最终为 `313 / 313`；其中还包含
`max_new_tokens=typemax(Int)`，确保 prompt 与 decode append 的长度加法在任何
allocation 前 checked/fail closed，而不是整数回绕后尝试错误大小的 storage。
仓库级 close run 为 `9,307 passed + 1 intentional broken`、`0 failure / 0 error`；
唯一 broken 是默认离线环境未设置真实 Qwen3-VL checkpoint 时的显式门禁。

专项可单独运行：

```bash
julia --project=. --startup-file=no -e '
using Test, LifeAI
include("test/episodes/episode10_qwen3_vl_efficient_generation/" *
        "chapter46_qwen3_vl_static_cache/test_qwen3_vl_static_cache.jl")
'
```

该默认总数不把 accelerator 专项或真实 checkpoint verifier 混入离线测试计数。

## RTX 4090 D Float32 真实验收

`scripts/verify_qwen3_vl_static_cache_cuda.jl` 先完整执行 Chapter 45 verifier，再在
同一进程、同一权重/vision features/reference 上对照 dynamic 与 static。它继续
硬编码检查：

- ModelScope revision `ae9985b…9c53` 与 Hugging Face revision `78448d…becd`；
- Float32 metadata SHA256
  `569fe3666b65ee2f497327e9ce9931f81652d5bdc32d44dfb9fb774435caccfc`；
- Float32 tensors SHA256
  `a98812e25efb44c02ab9c06e974ab718724f35f2f1c686e4bdc395d856c03e81`；
- deterministic image、rendered prompt、13-file checkpoint 与 tokenizer ids。

capacity 固定为 128。28 层的 56 个 K/V `CuArray` 均验证 Julia object identity、
device pointer identity、dtype/shape/device、非空指针和两两不 alias；prefill、三次
decode 与 reset 后全部不变。static 的每层有效 K/V prefix 和 dynamic path bitwise
一致，static/dynamic logits 也逐位一致，因此相对 HF Float32 reference 的误差与
Chapter 45 完全相同：

| phase | static cache position | argmax 1-based | max-abs |
| --- | ---: | ---: | ---: |
| prefill | `76` | `1987` | `3.862380981445e-5` |
| decode.0 | `77` | `2169` | `4.005432128906e-5` |
| decode.1 | `78` | `375` | `2.956390380859e-5` |
| decode.2 | `79` | `265` | `2.908706665039e-5` |

四阶段 strict max/mean/relative-L2/cosine gate 全部通过，greedy ids 仍为
`[1987,2169,375,265]`，文本仍为 `This image is a`。capacity 128 的 Float32
K/V payload 固定为 `29,360,128` bytes；四阶段结束时 valid 79-token prefix 为
`18,120,704` bytes，但未使用 tail 仍属于已分配容量。

同一 verifier 另做 32 generated-token dynamic/static 对照，最终 physical cache
position 都是 `107`、生成 ids 完全一致，且前四枚继续命中 frozen HF reference。
预热后各取三个样本，报告中位数：

| metric | dynamic | static | 差值 |
| --- | ---: | ---: | ---: |
| generation seconds | `0.616681` | `0.585846` | `-0.030835` |
| GPU allocated bytes | `4,270,103,300` | `3,645,283,076` | `-624,820,224` |
| GPU allocation count | `69,431` | `67,751` | `-1,680` |

按 K/V payload 计算，31 次 decode 的 dynamic `cat` output 共
`654,180,352` bytes，而 static 只写新 token slices，共 `7,110,656` bytes。
实测 GPU allocation 减少 `624,820,224` bytes，与消除这一 prefix-growth 来源的量级
一致；但 static 全程仍分配约 `3.65 GB` GPU 临时量，直接证明 whole-loop 并非
zero allocation。latency 只有三个交错样本，脚本显式输出
`latency_is_acceptance_gate=false`；`0.586` 对 `0.617 s` 只是本机观察，不被写成
稳定吞吐结论或普遍 speedup。

复现命令：

```bash
QWEN3_VL_MODEL_DIR=/home/ubuntu/models/modelscope/Qwen/Qwen3-VL-2B-Instruct
QWEN3_VL_DECODE_REFERENCE_DIR=/tmp/qwen3-vl-decode-f32

julia --project=. --startup-file=no \
  scripts/verify_qwen3_vl_static_cache_cuda.jl \
  "$QWEN3_VL_MODEL_DIR" "$QWEN3_VL_DECODE_REFERENCE_DIR" cuda float32
```

GPU 位于普通文件沙箱之外；设备隔离下看不到 CUDA 不代表 verifier 失败。验收必须在
获准访问宿主设备的环境运行。

## RTX 4090 D BFloat16 same-device smoke

同一 verifier 的 `bfloat16` 模式从冻结 checkpoint 直接加载 BF16 vision/text
parameters。四阶段 static/dynamic K/V prefix 与 logits 仍然 bitwise 一致，greedy
ids 仍为 `[1987,2169,375,265]`。但当前没有独立冻结的 HF BF16 decode tensors；
脚本只把 Chapter 45 Float32 CPU oracle 当语义锚点，以下是 cross-dtype boundary，
不是 BF16 strict parity：

| phase | max-abs vs HF Float32 | cosine vs HF Float32 |
| --- | ---: | ---: |
| prefill | `0.439221` | `0.999718` |
| decode.0 | `0.517180` | `0.999859` |
| decode.1 | `0.291030` | `0.999933` |
| decode.2 | `0.546978` | `0.999091` |

capacity 128 的 BF16 K/V payload 为 `14,680,064` bytes，valid 79-token prefix 为
`9,060,352` bytes。32-token same-device 对照的 GPU allocated bytes 中位从
dynamic `1,881,657,676` 降到 static `1,569,247,564`，精确少
`312,410,112` bytes；allocation count 同样少 `1,680`。理论 dynamic `cat`
outputs 为 `327,090,176` bytes，static slice writes 为 `3,555,328` bytes。

BF16 的最终三样本 latency 中位数为 dynamic `0.849395 s`、static `0.840655 s`，
两者接近。固定顺序的先导测量曾显示 static 更慢，但在释放返回 cache 并交替 A/B
顺序后没有复现，说明这种短窗口对执行顺序和 GC 敏感。latency 因此不属于
acceptance gate，也不能由 allocation bytes 下降推导必然改善。
复现时把最后一个参数改为 `bfloat16`：

```bash
julia --project=. --startup-file=no \
  scripts/verify_qwen3_vl_static_cache_cuda.jl \
  "$QWEN3_VL_MODEL_DIR" "$QWEN3_VL_DECODE_REFERENCE_DIR" cuda bfloat16
```

## 高层 generation 选择

低层 token API 和 raw single-image API 都接受：

```julia
result = generate_hf_qwen3_vl_tokens(
    text_parameters,
    input_ids,
    rope_layout;
    vision_features,
    max_new_tokens=4,
    cache=:static,
    static_capacity=79,
)
```

`cache=:dynamic` 保持 Chapter 45 默认行为，且拒绝同时传 `static_capacity`；
`cache=:static` 使用本章 preallocated path。返回值新增 `cache_mode`，最终 cache
仍遵循 `prompt_length + generated_count - 1` 的 physical timeline。高层 raw image
入口只切换 text K/V store policy，不重复运行 vision tower，也不改变 processor、chat、
placeholder、mRoPE、greedy/EOS 或 tokenizer 语义。

## 一个真实 Julia API 踩坑：keyword 不能参与 dispatch

初版集成曾尝试保留同名 prefill API，只用 keyword 参数类型区分 cache：

```julia
f(parameters, ids, layout; cache::Qwen3VLKVCache) = ...
f(parameters, ids, layout; cache::Qwen3VLStaticKVCache) = ...
```

这在 Julia 中不是两个可分派的方法。method table 只按 positional arguments
选择方法，keyword 参数及其类型不参与 dispatch；第二个定义会覆盖第一个。实际结果
不是抽象层面的风险，而是预编译直接报告 method overwrite，dynamic 调用随后把
`Qwen3VLKVCache` 传给只接受 static keyword 的实现并抛出 `TypeError`。

最终修复是保留显式命名的 `hf_qwen3_vl_text_prefill_static`，generation 内部 helper
则把 cache 放在 positional argument 上，以 `Qwen3VLKVCache` / `Qwen3VLStaticKVCache`
安全分派。`hf_qwen3_vl_text_decode_step` 本来就把 cache 放在 positional argument，
但为保持 public boundary 清楚，本章同样公开显式 `_static` decode 名称。Chapter 45 +
46 联合回归确认 dynamic 路径没有再被覆盖。

## fixed storage 不等于 zero allocation

本章可以严格声称：

- 不再为 K/V token 轴增长调用 `cat`；
- 不再在每次 decode 复制整个历史 K/V prefix；
- cache 和每层 K/V backing arrays 在 prefill/decode/reset 间保持 identity；
- K/V payload 在 cache 建立后固定为 capacity bytes。

本章不能声称：

- 整个 decode step 零 Julia/CUDA allocation；
- attention、projection、MLP、RoPE 或 logits workspace 已预分配复用；
- 已形成固定 CUDA Graph、Reactant/XLA executable 或 shape-polymorphic compiled loop；
- latency 一定优于 dynamic path 的每个短请求阶段；
- 已完成长上下文、并发、多 batch 或服务级吞吐验收。

static path 仍会构造 token embedding、Q/K/V、attention context、MLP intermediates、
final hidden 和 vocabulary logits；高层 generation 还会保存 trace/ids。固定 K/V 是
消除一个已知 `O(T²)` prefix-copy 来源的必要步骤，不是性能工作的终点。

## 已知边界与下一步

- **输入边界未扩张**：真实高层路径仍是单图、batch 1、全一 attention mask；
  padding、multi-image、video 和 image URL 不在本章。
- **生成策略未扩张**：仍只验收 greedy 与 EOS/length stop；sampling 未进入 static
  oracle。
- **dtype 结论分开**：storage 支持 Float32/BFloat16，当前真实 strict gate 是
  Float32；BF16 verifier 只允许 same-device static/dynamic smoke，不能因 token
  相同就写成跨实现 strict parity。
- **capacity 不是 context policy**：本章只验证 bounded storage 与 fail-closed
  overflow，没有实现 turn truncation、sliding window、paged attention 或 prefix sharing。
- **reset 不是并发调度**：同一 mutable cache 不能被两个请求同时写；服务层 ownership
  与 cancellation/recovery 仍需单独设计。
- **性能边界仍 Open**：下一章应先量化真实长 generation 的 per-token allocation、
  steady latency 与峰值显存，再决定优先复用 attention/linear workspace、引入 fused
  kernel，还是进入 compiled graph。

因此 Chapter 46 Closed：bounded preallocated K/V 的 correctness、ownership、容量和
复用契约已经闭合；Episode 10 保持 Open，等待整条生成循环的分配与长序列证据。
