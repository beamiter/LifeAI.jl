# Chapter 45 — Qwen3-VL dynamic KV cache 与 image-to-text generation

> 所属 Episode：Episode 09 — Qwen3-VL 多模态感知
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
> Oracle：Transformers `4.57.0` / PyTorch `2.7.1+cpu`，eager attention、
> `DynamicCache`、每次调用显式全一 attention mask
>
> 设备：NVIDIA GeForce RTX 4090 D

## Closed：核心问题

Chapter 44 已经把 raw image、content-list chat、vision tower、三轴 mRoPE 和
28-layer decoder prefill 接到 logits，但每次只能重算完整 prompt。本章继续回答：
能否为每个多模态请求保存未扩展的 GQA K/V，在不重复运行 vision tower、main visual
replacement 或 DeepStack injection 的情况下逐 token decode，并让真实 greedy
image-to-text 输出与官方 Transformers reference 一致？

答案是可以。LifeAI 现在具备 batch-1、单图、全一 attention mask 下的 dynamic KV
cached prefill、单 token decode、token-level greedy loop，以及从 raw image/content-list
message 到 completion text 的高层入口。官方 2B Float32 reference 的 prefill 加三次
decode 共四个 logits 阶段在 RTX 4090 D 上全部通过 strict gate，生成文本为
`"This image is a"`。这一最小闭环关闭了 Episode 09。

## Close 条件

1. 请求级 cache 明确保存每层 RoPE 后的 K 和未旋转 V，layout 为
   `(head_dim, kv_heads, tokens, batch)`，不持久化 GQA-expanded heads。
2. 物理 cache position 与请求级 `rope_delta` 分离；decode 的零基三轴坐标严格使用
   `cache.position + cache.rope_delta`。
3. cached prefill 与 Chapter 44 cache-free prefill 数值一致，并只在 prefill 执行
   main visual replacement 和三层 DeepStack injection。
4. deterministic tiny oracle 至少覆盖 prefill 加两次 decode、每层完整 K/V snapshot、
   prefix immutability、final hidden、logits 和 greedy timeline。
5. 官方 2B Float32 CPU oracle 与 RTX 4090 D LifeAI CUDA 路径完成 prefill 加三次
   decode 的逐阶段 logits parity，四个 greedy token 和解码文本一致。
6. raw single-image content-list API 串联 processor、chat、tokenizer、vision、cached
   decoder、EOS/length stop 与 completion text。
7. 文档明确 dynamic `cat`、static cache、BF16 generation、video、multi-image、
   batch/padding 和长上下文边界。

## 不可变 reference

本章继续消费 Chapter 43—44 冻结的同一 13-file checkpoint，不引入第二套权重身份。
`scripts/export_qwen3_vl_decode_reference.py` 有两个独立模式：

- `tiny` 构造确定性 4-layer Float32 text model，完整保存 prefill、两次 decode 的
  final hidden、logits 以及所有层 K/V；
- `real` 从官方 2B checkpoint、确定性 `256×256` RGB image 和 `Describe.` message
  开始，保存 prefill 与三次 decode 的 last hidden/logits。真实 K/V 只冻结每层
  shape 与 raw SHA256，避免复制大体积 cache tensor。

不可变文件 SHA256 为：

| reference | SHA256 |
| --- | --- |
| committed tiny JSON | `7b20111e43aa9efd2aae0be3f4a740ab1fdeaff9bd0a6ffd0bfef49adfeeffd8` |
| real Float32 `reference.json` | `569fe3666b65ee2f497327e9ce9931f81652d5bdc32d44dfb9fb774435caccfc` |
| real Float32 `reference.safetensors` | `a98812e25efb44c02ab9c06e974ab718724f35f2f1c686e4bdc395d856c03e81` |

real reference 内八个 tensor 的 raw SHA256 也被 metadata 冻结：

| phase | final hidden | logits |
| --- | --- | --- |
| prefill | `6df50c39b0a676fc3d5cd35b56c7606c82da6a7899b0f42a85d5abef542ab352` | `0009916e90726f5562a35c11dbeb8b6d56e9f279da85fad30f653a8679f676f5` |
| decode.0 | `9ce5355e91f5fc83e6d96e23b2a0a24d47e73ff8a3d835eb239f57aa568f0986` | `ff6a599357bc4b94147067940d122b420fd81bf59b9d704cb5c0bf7f3a15fbee` |
| decode.1 | `acef6604bf39bba12b2b987643cf9158d2d48f8accf542b88ef239181c383aaa` | `59b1f7c0f0c66cd68f63e9c9c9685ce9e6198f5321dd06adf50dc36adb4ca89d` |
| decode.2 | `5ca77b27409a1f8d0c6fea74efdc5dd3a853dae92736e204ed9c5f421cac6e5d` | `f45640db9b8405865a3adcfee0182d9b9b057fa55f8b1ff3716e85f5e4bf9046` |

确定性 raw image SHA256 为
`ec143579b36852cf212bbb368798479d193a8c6d039942fca89a49fc820dff3f`，官方
rendered prompt SHA256 为
`7a50d10ccb53359de53e3e9b032c39b15fd3abbfed51ec844117c3b93da07271`。
verifier 同时检查 ModelScope/Hugging Face revisions、官方资产 checksum、reference
文件 checksum、tensor names/shapes/dtypes 和 prompt/token ids，metadata 不能通过
自报新 hash 绕过门禁。

## cache layout 与生命周期

`Qwen3VLKVCache` 是请求状态，不属于模型参数，保存：

- `layers`：28 个 `LayerKVCache`；
- `position`：已经物理处理并存入 K/V 的 token 数；
- `rope_delta`：多模态 prompt 导出的请求级坐标压缩量；
- `batch_size`：本章固定为 1。

每层 K/V layout 为：

```text
(head_dim, num_key_value_heads, cached_tokens, batch)
```

HF `DynamicCache` 使用 `(batch, kv_heads, tokens, head_dim)`；oracle 明确冻结
HF→Julia permutation `(4,2,3,1)`。K 在 projection、per-head QK-Norm 和 mRoPE
之后写入，V 在 projection 后直接写入。真实 2B 的 16 Q heads 只在 attention 计算中
按 8 KV heads 分组，不把 repeat 后的 16 heads 存入 cache。

公共核心 API 为：

```julia
cache = init_qwen3_vl_kv_cache(text_parameters; batch_size=1)

prefill, cache = hf_qwen3_vl_text_prefill_cached(
    text_parameters,
    input_ids,
    rope_layout;
    vision_features,
    cache,
    logits_to_keep=1,
)

logits, cache = hf_qwen3_vl_text_decode_step(
    text_parameters,
    token,
    cache,
)
```

cached prefill 必须消费空 cache，并把 `position` 更新为 prompt physical length、把
`rope_delta` 从 `Qwen3VLRopeLayout` 固化进请求。decode 每次只接受一个 token，
沿 token 轴追加一片 K/V，旧 prefix 保持逐位不变。decode 不接收另一份外部
position ids，也不重新注入 visual/DeepStack feature，避免请求状态出现两个真相来源。
prefill 还会独立重算
`maximum(position_ids) + 1 - sequence_length`，即使 `rope_deltas` shape 合法，
只要数值与 prompt 坐标不一致也会 fail closed。

## physical position 与 mRoPE coordinate

多模态 prompt 的物理长度与 mRoPE 坐标并不相等。decode 统一使用：

```text
next_mrope_coordinate_0_based = cache.position + cache.rope_delta
```

tiny prompt 的 physical length 为 8、`rope_delta=-2`，因此两次 decode 输入的
physical positions 是 `8 / 9`，T/H/W 坐标是 `6 / 7`，cache length 依次为
`8 → 9 → 10`。真实 prompt 的 physical length 为 76、`rope_delta=-56`，三次
decode 的 physical positions 是 `76 / 77 / 78`，T/H/W 坐标是
`20 / 21 / 22`，cache length 为 `76 → 77 → 78 → 79`。

这一区分也解释了 generation timeline：第一个输出 token 直接从 prefill logits
选择；若要生成 4 个 token，只需把前三个选择结果依次送入 3 次 decode。第四个 token
来自最后一次 decode logits，尚未作为输入写回，所以最终 cache length 是 79，而不是
80。

## tiny DynamicCache oracle

tiny 模型沿用 Chapter 44 的 hidden `16`、4 layers、2 Q heads / 1 KV head、head dim
`8` 与 `mrope_section=(2,1,1)`。exporter 在每个阶段立即
`detach().clone().cpu().contiguous()`；HF `DynamicCache` 会原地扩展，若只保留
对象引用，早期 snapshot 会被后续 decode 污染。

fixture 完整冻结三阶段：

| phase | cache length | decode T/H/W | greedy id（HF 0-based / LifeAI 1-based） |
| --- | ---: | --- | --- |
| prefill | `8` | — | `7 / 8` |
| decode.0 | `9` | `(6,6,6)` | `7 / 8` |
| decode.1 | `10` | `(7,7,7)` | `7 / 8` |

测试逐层比较所有 K/V、final hidden 与 logits，并证明 `8→9`、`9→10` 时旧 K/V
prefix bit-exact 不变。Chapter 44 的 prefill final hidden/logits SHA256
`f2e5565a…f154` / `9f38d30c…d0d6` 也被复用，防止“加 cache 后 prefill 语义变化”。

Chapter 45 专项共 `331 / 331`：frozen fixture contract `122 / 122`、cached
prefill/two-step decode `170 / 170`、greedy generation timeline `39 / 39`。本章
新增的 cached-prefill case 专门拒绝 shape 合法但与 position ids 不一致的
`rope_delta`。最终隔离 `Pkg.test()` 为 `8,994 passed`、
`1 intentional broken`、`0 failure / 0 error`；唯一 broken 是未提供真实
Qwen3-VL checkpoint 环境变量时的 opt-in 门禁。首次隔离运行还暴露了 Chapter
44/45 fixture 使用的 `Base64` 未列入 test target，补齐该标准库测试依赖后全套通过。

## greedy generation API

`generate_hf_qwen3_vl_tokens` 从已经准备好的 token、rope layout 和 vision features
执行 cached prefill 与 greedy decode；token ids 使用 LifeAI 1-based 约定。返回值
显式包含 prompt/generated/all ids、`:greedy` strategy、`:eos` 或 `:length` stop
reason、逐 step top-two margin、prefill result 和最终 cache。

`generate_hf_qwen3_vl` 再把高层 raw boundary 串起来：

```text
local RGB image/content-list message
  → exact processor/chat/placeholder/tokenizer/mRoPE
  → vision tower + main/DeepStack features
  → cached text prefill
  → single-token greedy decode
  → completion text
```

真实输入仍是确定性 `256×256` image 与 `Describe.`，形成 sequence `76`、image
tokens `64`。官方与 LifeAI 的生成结果为：

| 口径 | 冻结值 |
| --- | --- |
| HF token ids（0-based） | `[1986, 2168, 374, 264]` |
| LifeAI token ids（1-based） | `[1987, 2169, 375, 265]` |
| token pieces | `["This", " image", " is", " a"]` |
| completion | `"This image is a"` |

真实 run 由 `max_new_tokens=4` 截止，没有把长度截止误写成 EOS 命中。高层 raw-image
wrapper 与手工 cached-prefill/decode loop 的 ids、text、prompt、placeholder expansion、
最终 cache position 和 `rope_delta` 全部一致。

## RTX 4090 D Float32 strict 结果

`scripts/verify_qwen3_vl_decode_cuda.jl` 在 `CUDA.allowscalar(false)` 下重新执行 raw
processor、chat/tokenizer、vision、cached prefill、三次 decode 和高层 wrapper。
四阶段以下指标均比较完整 151,936-way last-position logits：

| phase | cache length | mRoPE T/H/W | argmax 1-based | max-abs | mean-abs | relative L2 | cosine |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| prefill | `76` | — | `1987` | `3.862380981445e-5` | `7.502852388181e-6` | `2.491764492934e-6` | `0.9999999999970042` |
| decode.0 | `77` | `(20,20,20)` | `2169` | `4.005432128906e-5` | `5.626603576536e-6` | `1.276814328535e-6` | `0.9999999999992562` |
| decode.1 | `78` | `(21,21,21)` | `375` | `2.956390380859e-5` | `5.176316295721e-6` | `1.219367434425e-6` | `0.9999999999993456` |
| decode.2 | `79` | `(22,22,22)` | `265` | `2.908706665039e-5` | `5.168276760988e-6` | `1.653558204443e-6` | `0.9999999999986956` |

strict gate 为 max-abs `≤1e-4`、mean-abs `≤2e-5`、relative L2 `≤5e-6`、
cosine `≥0.99999999`；四阶段全部通过且 argmax 全部一致。每次 decode 还验证只追加
一个 physical position、28 层 cache shape/dtype/device residency 正确、旧 prefix
bit-exact 不变、`rope_delta=-56` 不变。

Float32 cache 的 logical bytes 为：

```text
2 × 28 layers × 128 head_dim × 8 KV heads × tokens × 4 bytes
= 229,376 bytes / cached token
```

| cache length | logical K/V bytes |
| ---: | ---: |
| `76` | `17,432,576` |
| `77` | `17,661,952` |
| `78` | `17,891,328` |
| `79` | `18,120,704` |

这些数字不含 allocator metadata、旧 dynamic allocations、attention workspace 或模型
参数。最终 `18,120,704` bytes 只表示当前 79-token K/V payload。

本次证据还记录 vision/text load `5.937709 / 27.140332 s`、vision cold forward
`11.718628 s`、cached prefill cold `6.680964 s`、三次 decode
`0.758993 / 0.041506 / 0.015878 s`，高层 generation `3.174205 s`。cold 与首个
decode 含 Julia/CUDA kernel 编译，高层调用又覆盖不同组合边界，因此这些值只证明
真实执行，不作为 steady latency 或 tokens/s benchmark。

## 复现方式

```bash
MODEL_DIR=/home/ubuntu/models/modelscope/Qwen/Qwen3-VL-2B-Instruct
REFERENCE_DIR=/tmp/qwen3-vl-decode-f32
ORACLE_PYTHONPATH=/tmp/lifeai-qwen3vl-oracle/lib/python3.10/site-packages:/tmp/lifeai-qwen3vl-uv-cache/archive-v0/SNUjiORDNkYR55Or

PYTHONPATH="$ORACLE_PYTHONPATH" \
  .venv/bin/python scripts/export_qwen3_vl_decode_reference.py real \
  "$MODEL_DIR" "$REFERENCE_DIR" \
  --dtype float32 --device cpu --greedy-tokens 4

julia --project=. --startup-file=no \
  scripts/verify_qwen3_vl_decode_cuda.jl \
  "$MODEL_DIR" "$REFERENCE_DIR" cuda
```

tiny fixture 可用下列命令重建，并以文件 SHA256 判断是否仍是同一个 oracle：

```bash
PYTHONPATH="$ORACLE_PYTHONPATH" \
  .venv/bin/python scripts/export_qwen3_vl_decode_reference.py tiny \
  /tmp/qwen3-vl-tiny-decode.json
```

reference 保持在仓库外；committed tiny fixture 只包含小型完整 tensors。冻结依赖与
checkpoint 恢复方式继续见 [`notes/local_model_assets.md`](../../local_model_assets.md)。

## 已知边界与决策

- **dynamic `cat` 是 correctness baseline**：每次 decode、每层都沿 token 轴重新
  分配并复制完整 prefix，累计 copy 为 `O(T²)`；CUDA allocator 峰值也会高于当前
  logical K/V bytes。本章没有把它包装成高吞吐或长上下文服务方案。
- **没有 static/preallocated VL cache**：尚无原位 slot write、capacity planning、
  chunked prefill 或 XLA compiled decode；这应在单独的部署章节处理。
- **真实 strict generation 仅 Float32**：默认 BF16 CUDA 高层路径也实跑得到
  `[1987,2169,375,265]` / `"This image is a"`，cache length `79`、
  `rope_delta=-56`；但它只是 smoke，没有独立 BF16 HF cache/decode oracle 或
  逐阶段数值门禁，不能写成 BF16 strict parity。
- **batch/padding 未实现**：本章固定 batch 1、全一 attention mask。padding prefix
  若写入 cache，decode 必须持续屏蔽无效 K/V，不能直接复用本章的 mask-free 单 query。
- **单图边界**：高层 API 明确拒绝 image URL、video 和 multi-image；真实权重也只验收
  一张 `256×256` image。
- **只有 greedy**：sampling、beam search、streaming callback、speculative decode
  均未实现。stop ids/EOS 路径有 tiny 测试，真实四-token run 因长度上限停止。
- **无长上下文与 steady 性能结论**：真实 prompt 为 76 tokens，只生成 4 tokens；
  没有验证 262K context、内存碎片、长序列累计误差或稳定吞吐。

## Close 回顾

- **完成了什么**：把 Chapter 44 的 cache-free logits 扩展成 request-local dynamic
  K/V、cached prefill、单 token decode、greedy timeline 和 raw image-to-text API。
- **验证证据**：tiny fixture SHA256 `7b20111e…ffd8`，专项 `331 / 331`；真实 F32
  metadata/tensor SHA256 为 `569fe366…caccfc` / `a98812e2…03e81`；RTX 4090 D
  四阶段 strict logits gate、四个 argmax 和 completion 全部一致。
- **没有完成及原因**：static cache、BF16 strict oracle、batch/padding、video、多图、
  sampling 和长上下文需要各自独立证据，不能从这条最小 dynamic path 外推。
- **最重要的认知变化**：多模态 decode 有两条位置时间线。physical cache position
  决定 K/V slot，而 `physical + rope_delta` 决定 T/H/W rotation；把二者合并会从第一
  个 decode token 起系统性错位。
- **是否满足 Close 条件**：是。cache layout/lifecycle、位置语义、tiny 全 tensor、
  真实 Float32 CUDA parity、greedy ids/text 和能力边界均已冻结，Episode 09 Closed。
