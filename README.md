# LifeAI.jl

> 构建有生命感的 AGI，并让它走进真实世界。

LifeAI.jl 是一个长期演进的 Julia/Lux 研究与工程项目。项目不止于实现一个语言模型，而是希望逐步构建能够持续感知、记忆、学习、决策和行动的智能系统，并将这些能力用于机器狗、桌面机器人、具身机器人以及其他可能的智能载体。

这里也是一份公开的学习与构建记录：从关键模型组件开始，理解原理、亲手实现、测试验证、分析性能，再逐步组合成更完整的智能体。

## 项目宗旨

LifeAI.jl 沿四条相互连接的主线持续积累：

1. **模型基础组件**：持续学习和实现 Attention、位置编码、Transformer、训练、推理加速等可复用能力。
2. **智能体核心**：逐步加入记忆、规划、工具使用、反思、多模态理解和持续学习能力。
3. **具身闭环**：让模型通过统一接口连接感知与行动，先在可验证的环境中运行，再走向机器狗、桌面机器人和其他实体设备。
4. **学习过程记录**：用 Chapter 推动小步交付，用 Episode 串联完整能力主线，沉淀结果、实验、失败和认知变化。

这里所说的“有生命感”，不是只让模型表现得像某种人格，而是让系统在长期互动中体现出连续性、状态、记忆、主动性、适应性以及与环境真实连接的行动能力。

## 当前状态

**阶段判断：Qwen3 dense family 真实权重 parity 与 BF16 GPU 推理全闭环，Qwen3-Embedding-0.6B 与最小 dense exact semantic memory 也已完成真实 BF16 parity；RTX 4090 D 上的 dense 容量上限仍是已实证生成的 14B mixed RTN，日常部署选择 Qwen3-8B BF16。原始 Qwen3-30B-A3B 现已具备 40K-capacity BF16 GPU resident/offload session、global/layer-balanced device expert cache、scalar 与 grouped WMMA 直接消费分散 BF16 cache matrices 的 generation-safe CUDA dispatch，以及经过 storage I/O 验证的当前层 bounded parallel miss reads。grouped-scattered 对相同 grouped 数值契约逐位一致，2/32-token request 相对 active-3D materialization 加速 `10.745× / 6.103×`。Qwen3-VL-2B-Instruct 已从双 registry 资产契约、raw image fast processor、content-list chat、vision tower、main/DeepStack visual injection、三轴 mRoPE 和完整 decoder prefill，继续闭合到 dynamic KV baseline、bounded preallocated static KV、单 token decode、可 reset request state 与 greedy image-to-text generation；真实 Float32 RTX 4090 D 上 static/dynamic 的四阶段有效 K/V 与 logits bitwise 一致，并保持 Transformers reference 的 token/text strict gate，32-token static 对照减少 `624,820,224` GPU allocated bytes。智能体侧已具备工具、跨请求持久记忆、由环境终态评分并可联合 replay 的 observation/action 闭环，以及 clean successful 环境事件的显式写回、fresh-load exact-spec 检索和无反馈后续执行。**

Chapter 01—147 均已 Closed，Episode 06/07/09 已 Closed，Episode 08/10 已 Open。[`Chapter 46 — Qwen3-VL bounded static KV cache 与可复用 generation state`](notes/episodes/episode10_qwen3_vl_efficient_generation/chapter46_qwen3_vl_static_cache.md) 在 Chapter 45 dynamic correctness baseline 上预分配 `(head_dim, kv_heads, capacity, batch)` storage，prefill/decode/reset 全程保持 cache 与每层 K/V array identity，容量满时 fail closed。低层 `capacity` 必填；高层 `cache=:static` 只推导本次 prompt/generation 所需容量，不会默认分配官方最大 context。tiny oracle 三路全四层有效 K/V、hidden/logits 与 greedy `[8,8,8]` 一致；真实 RTX 4090 D Float32 的 56 个 buffers 保持 object/device-pointer identity，四阶段 static/dynamic K/V 与 logits bitwise 一致。32-token 对照的 GPU allocated bytes 从 `4,270,103,300` 降到 `3,645,283,076`，但后者也证明整条 decode 仍非零分配；Episode 10 因此保持 Open。

[`Chapter 47 — Qwen3-VL 长生成 allocation profile 与 decoder workspace 归因`](notes/episodes/episode10_qwen3_vl_efficient_generation/chapter47_qwen3_vl_long_generation_profile.md) 已建立数值透明的 internal stage runner、HF Float32/CPU 流式 256-token oracle 和真实 BF16 benchmark。2026-10-08 两个独立进程导出的 256-token oracle 逐字节相同，SHA256 为 `72461330a8c5a0e74e7489abfdd481223aeb47d6c161fdc12dc0048e82690140`，前四枚仍是 `[1987, 2169, 375, 265]`。RTX 4090 D 上 BF16 static 256-token warmup 在 generated step 79 与该 oracle 分叉。同前缀拆分里，LifeAI Float32 CUDA 79/79 argmax 与 HF 相同，BF16 把 step 79 的 `25.939213 / 25.889927` 落成 `26 / 25.875` 后选中 runner-up。Teacher-forced 正确前缀上 Float32 是 256/256，BF16 另有 step 95 的 `28.375` 平局。一次非验收 profile 曾把 position `107 / 203 / 331` 的最后一步归因闭合，`kv_write` 为 0，dominant stage 是 `attention`。复用后 attention 在这三个 position 上都是 `7,168` bytes，整步合计都是 `10,563,721` bytes；最大头变成不随 position 增长的 `mlp_activation`。三次 repeat 里 32-token prefix 与 HF exact，128/256 在 step 70 分叉。Chapter 47 于 2026-10-08 Closed。最终 logits 没有升成 Float32，也不声称 BF16 strict parity。

[`Chapter 48 — Qwen3-VL decode mlp_activation 常量分配`](notes/episodes/episode10_qwen3_vl_efficient_generation/chapter48_qwen3_vl_mlp_activation.md) 核对了融合 SiLU 乘积留下的 5 个 `(6144, 1, 1)` 数组：3 个 Float32 与 2 个 BF16，每层 `98,304` bytes，28 层 `2,752,512` bytes、`140` 次分配。BF16 单 token decode 现在按 `intermediate_size` 预分配这块 scratch，28 层共用，reset 保留 storage；prefill 和 Float32 不创建它。同进程乘积与原表达式逐位一致，复用调用是 `0` GPU bytes。2026-10-08 一次 repeat 的重测 SHA256 为 `4aa2057607d940a12901bb3d71f8498a9fff1c6d548e597d07c8d7c3a22906de`。position `107 / 203 / 331` 的 `mlp_activation` 都是 `0`，整步都是 `7,811,209` bytes、`3,628` 次分配，没有新的线性头。最大常量 stage 变成 `qk_rope` 的 `2,265,088` bytes。32-token prefix 仍与 HF exact，128/256 仍在 step 70 分叉。Chapter 48 于 2026-10-08 Closed。Episode 10 保持 Open。最终 logits 没有升成 Float32，也不声称整条 decode 零分配。

[`Chapter 49 — Qwen3-VL decode qk_rope 常量分配`](notes/episodes/episode10_qwen3_vl_efficient_generation/chapter49_qwen3_vl_qk_rope.md) 拆开了单次 `_bf16a_apply_rope`：Q 是 `53,760` bytes、`19` 次分配，K 是 `27,136` bytes、`19` 次分配，合计每层 `80,896 / 38`。BF16 单 token decode 现在把同一串 half 切片、Float32 提升和 BF16 舍入写进按 query heads 预分配的 scratch，28 层共用，K 使用 head 前缀，Q/K 输出保持两块，reset 保留 storage。prefill、Float32 和多 token 仍走原来的共享函数。同进程结果逐位一致，复用调用是 `0` GPU bytes，驻留 scratch 是 `37,376` bytes。2026-10-08 一次 repeat 的重测 SHA256 为 `d8726e41c1171fe42b823254bb0ccb3fde40baf547c700dd87bda228f0c398d8`。position `107 / 203 / 331` 的 `qk_rope` 都是 `0`，整步都是 `5,546,121` bytes、`2,564` 次分配，没有新的线性头。最大常量 stage 变成 `qk_norm` 的 `1,066,240` bytes。32-token prefix 仍与 HF exact，128/256 仍在 step 70 分叉。Chapter 49 于 2026-10-08 Closed。Episode 10 保持 Open。最终 logits 没有升成 Float32，也不声称整条 decode 零分配。

[`Chapter 50 — Qwen3-VL decode qk_norm 常量分配`](notes/episodes/episode10_qwen3_vl_efficient_generation/chapter50_qwen3_vl_qk_norm.md) 拆开了单次 Q/K RMSNorm：Q 是 `25,216` bytes、`7` 次分配，K 是 `12,864` bytes、`7` 次分配，合计每层 `38,080 / 14`。BF16 单 token decode 用 `sum!(abs2, dest, x)` 和预分配 broadcast 重放同一串舍入，28 层共用，K 使用 head 前缀，Q/K 输出保持两块。prefill 和 Float32 仍走原来的 RMSNorm。同进程结果逐位一致，复用调用是 `0` GPU bytes，驻留 scratch 是 `27,200` bytes。2026-10-08 一次 repeat 的重测 SHA256 为 `8f415d759aca2cf27a402c484c2c6c2c0a06624a25c64bceb6f74681f25afdad`。position `107 / 203 / 331` 的 `qk_norm` 都是 `0`，整步都是 `4,479,881` bytes、`2,172` 次分配，没有新的线性头。最大并列常量阶段是 `pre_attention_norm` 和 `post_attention_norm`，各 `918,176` bytes。32-token prefix 仍与 HF exact，128/256 仍在 step 70 分叉。Chapter 50 于 2026-10-08 Closed。Episode 10 保持 Open。最终 logits 没有升成 Float32，也不声称整条 decode 零分配。

[`Chapter 59 — Qwen3-VL decode gemmEx 的持久 alpha/beta`](notes/episodes/episode10_qwen3_vl_efficient_generation/chapter59_qwen3_vl_gemm_scalar.md) 没有改用 Chapter 58 否决的 strided GEMM。per-head 顺序保持不变，同一条 `cublasGemmEx` 改用进程内持久的 device alpha/beta。2026-10-08 一次 repeat 的重测 SHA256 为 `c0bdb3d4fa6427bce41d8f42a95f075ec6c53331c10eed1d00a2aa5014a6298c`。position `107 / 203 / 331` 的最后一步都是 `1,168` bytes、`59` 次分配。`attention`、`qkv_projection`、`mlp_gate_up_projection`、两处 residual 和 `vocab_logits` 都是 `0`。最大并列阶段是 `pre_attention_norm` 和 `post_attention_norm`，各 `448` bytes。32-token prefix 仍与 HF exact，128/256 仍在 step 70 分叉，margin `0.02797699`。Chapter 59 于 2026-10-08 Closed。Episode 10 保持 Open。最终 logits 没有升成 Float32，也不声称整条 decode 零分配。

[`Chapter 60 — Qwen3-VL decode hidden RMSNorm partial`](notes/episodes/episode10_qwen3_vl_efficient_generation/chapter60_qwen3_vl_norm_partial.md) 复用 CUDA `sum!` 在归约长度 `2048` 上的多 block partial。launch configuration 不变，完整 hidden RMSNorm 仍与 `_bf16a_rmsnorm` 逐位一致。2026-10-08 一次 repeat 的重测 SHA256 为 `b9e6d0ab4331d689d448ff8d03b7714d283c0553454f37f89b37abb43276361e`。position `107 / 203 / 331` 的最后一步都是 `256` bytes、`2` 次分配。三处 hidden norm 都是 `0`。只剩 `mrope_prepare`。32-token prefix 仍与 HF exact，128/256 仍在 step 70 分叉，margin `0.02797699`。Chapter 60 于 2026-10-08 Closed。Episode 10 保持 Open。最终 logits 没有升成 Float32，也不声称整条 decode 零分配。

[`Chapter 61 — Qwen3-VL decode mRoPE cos/sin 表`](notes/episodes/episode10_qwen3_vl_efficient_generation/chapter61_qwen3_vl_mrope.md) 把单 token host 频率照原式计算，两张 `(64, 1, 1)` BF16 cos/sin 写进 `256` byte scratch。2026-10-08 一次 repeat 的重测 SHA256 为 `8411b055efea353da282dc295558b5db99b6d088b7814a62673ce54ecc7d46b9`。position `107 / 203 / 331` 的最后一步都是 `0` bytes、`0` 次分配。256-token 生成只有第一次 decode 分配 `2,018,180` bytes、`56` 次，那是按 capacity 创建 scratch。32-token prefix 仍与 HF exact，128/256 仍在 step 70 分叉，margin `0.02797699`。Chapter 61 于 2026-10-08 Closed。Episode 10 保持 Open。最终 logits 没有升成 Float32，也不声称整条 decode 零分配。

[`Chapter 67 — Qwen3-VL decode MLP SiLU 合成 kernel`](notes/episodes/episode10_qwen3_vl_efficient_generation/chapter67_qwen3_vl_fused_mlp.md) 和 Chapter 62—66 一起，把 cache 重叠判断以及 RoPE、QK norm、hidden norm、residual、SiLU 的元素广播收成 kernel。2026-10-09 一次 repeat 的重测 SHA256 为 `ebf007aafe96f34972d45a8429899acdcfd8781468e4f9b24611396ec9ab8b4a`。GPU 最后一步仍是 `0`。256-token 稳态一步的 CPU 分配从 `7,586,128` bytes 降到 `2,168,720`。32-token prefix 仍与 HF exact，128/256 仍在 step 70 分叉，margin `0.02797699`。Chapter 62—67 于 2026-10-09 Closed。Episode 10 保持 Open。最终 logits 没有升成 Float32，也不声称整条 decode 零分配。

[`Chapter 69 — Qwen3-VL decode attention softmax 元素 kernel`](notes/episodes/episode10_qwen3_vl_efficient_generation/chapter69_qwen3_vl_softmax_kernel.md) 把 scale、exp、weight 三条广播收成 kernel，`maximum!` 和 `sum!` 留在原处。sequence `1 / 80 / 330 / 331` 的 2048 个 attention 输出与广播路径逐位一致。2026-10-09 一次 repeat 的重测 SHA256 为 `1d463083013c0076f62032731f14a02129e0eaced160ee09e4cdb87e79a5d638`。GPU 最后一步仍是 `0`。256-token 稳态一步的 CPU 分配从 `2,168,720` 降到 `1,817,040`。32-token prefix 仍与 HF exact，128/256 仍在 step 70 分叉，margin `0.02797699`。Chapter 69 于 2026-10-09 Closed。Episode 10 保持 Open。最终 logits 没有升成 Float32，也不声称整条 decode 零分配。

[`Chapter 70 — Qwen3-VL decode attention permute kernel`](notes/episodes/episode10_qwen3_vl_efficient_generation/chapter70_qwen3_vl_permute_kernel.md) 把 query、K/V、weight 和 context 的置换收成下标相同的拷贝 kernel。sequence `1 / 80 / 330 / 331` 的 attention 输出、query 和整段 K/V 与 `permutedims!` 逐位一致。2026-10-09 一次 repeat 的重测 SHA256 为 `269aa7775f10056a8abe10a70f0180288db4b995b862e07a2f00761903e8244f`。GPU 最后一步仍是 `0`。position `107 / 203 / 331` 的 CPU 都是 `1,422,800` bytes。32-token prefix 仍与 HF exact，128/256 仍在 step 70 分叉，margin `0.02797699`。Chapter 70 于 2026-10-09 Closed。Episode 10 保持 Open。最终 logits 没有升成 Float32，也不声称整条 decode 零分配。

[`Chapter 71 — Qwen3-VL decode QK RMSNorm 的准备`](notes/episodes/episode10_qwen3_vl_efficient_generation/chapter71_qwen3_vl_qk_prepare.md) 把 QK 的升精度收成 kernel，除以 `head_dim` 写进原来的归一化 kernel，`sum!` 改走缓存的同一条归约。query `2,048` 与 key `1,024` 个元素和改前的 GPU 路径逐位一致。2026-10-09 一次 repeat 的重测 SHA256 为 `ab40b3fee9c35b43385c32d5c70dbffff01f44ccfc548b76cd2ceac1bd8d1da6`。GPU 最后一步仍是 `0`。position `107 / 203 / 331` 的 CPU 都是 `1,126,224` bytes。32-token prefix 仍与 HF exact，128/256 仍在 step 70 分叉，margin `0.02797699`。Chapter 71 于 2026-10-09 Closed。Episode 10 保持 Open。最终 logits 没有升成 Float32，也不声称整条 decode 零分配。

[`Chapter 72 — Qwen3-VL decode cublasGemmEx 的启动对象`](notes/episodes/episode10_qwen3_vl_efficient_generation/chapter72_qwen3_vl_gemm_launch.md) 仍调用同一条 `cublasGemmEx`，但 device 地址、handle、alpha/beta 和 BF16 数据类型不再每次重建。稠密调用是 `16` bytes，短序列 head 切片是 `80` bytes，都与原来的 GEMM 逐位一致。2026-10-09 一次 repeat 的重测 SHA256 为 `33a5f3d04f0bdc2cf405358585cec60a26607e0fe259453f331ca9452b36c355`。GPU 最后一步仍是 `0`。256-token 稳态一步的 CPU 是 `873,104` bytes。32-token prefix 仍与 HF exact，128/256 仍在 step 70 分叉，margin `0.02797699`。Chapter 72 于 2026-10-09 Closed。Episode 10 保持 Open。最终 logits 没有升成 Float32，也不声称整条 decode 零分配。

[`Chapter 73 — Qwen3-VL decode hidden RMSNorm 的准备`](notes/episodes/episode10_qwen3_vl_efficient_generation/chapter73_qwen3_vl_hidden_prepare.md) 把 hidden RMSNorm 的升精度收成 kernel，除以 `hidden_size` 写进原来的归一化 kernel。`2,048` 个元素与改前的 GPU 路径逐位一致。2026-10-09 一次 repeat 的重测 SHA256 为 `aa375cd1d1ea25041f1bf50259838af4c0f176253c77b90b4a6d794d34861fa0`。GPU 最后一步仍是 `0`。position `107 / 203 / 331` 的 CPU 都是 `677,920` bytes。32-token prefix 仍与 HF exact，128/256 仍在 step 70 分叉，margin `0.02797699`。Chapter 73 于 2026-10-09 Closed。Episode 10 保持 Open。最终 logits 没有升成 Float32，也不声称整条 decode 零分配。

[`Chapter 74 — Qwen3-VL decode attention maximum`](notes/episodes/episode10_qwen3_vl_efficient_generation/chapter74_qwen3_vl_attention_max.md) 用逐 head 顺序扫描替换 attention 的 `maximum!`。sequence `1 / 80 / 330 / 331` 的 16 个最大值逐位一致。2026-10-09 一次 repeat 的重测 SHA256 为 `030d937ac838355235bed889beac2720c496a1707b6e0db364a06421e6c28d2a`。GPU 最后一步仍是 `0`。256-token 稳态一步的 CPU 是 `617,440` bytes。32-token prefix 仍与 HF exact，128/256 仍在 step 70 分叉，margin `0.02797699`。Chapter 74 于 2026-10-09 Closed。Episode 10 保持 Open。最终 logits 没有升成 Float32，也不声称整条 decode 零分配。

[`Chapter 75 — Qwen3-VL decode per-head GEMM 的指针偏移`](notes/episodes/episode10_qwen3_vl_efficient_generation/chapter75_qwen3_vl_batched_pointer.md) 用 `(index - 1) * stride(3)` 代替每头 GEMM 的 `view`。sequence `1 / 80 / 331` 的 score 和 context 与逐头路径逐位一致。8 头一次从 `768` bytes 降到 `128`。2026-10-09 一次 repeat 的重测 SHA256 为 `8cfa7b7f67eb1f0c241b751d6a047a68c997e1e25db81a7d58143eca6485beae`。GPU 最后一步仍是 `0`。256-token 稳态一步的 CPU 是 `581,600` bytes。32-token prefix 仍与 HF exact，128/256 仍在 step 70 分叉，margin `0.02797699`。Chapter 75 于 2026-10-09 Closed。Episode 10 保持 Open。最终 logits 没有升成 Float32，也不声称整条 decode 零分配。

[`Chapter 76 — Qwen3-VL decode abs2 归约计划的具体类型`](notes/episodes/episode10_qwen3_vl_efficient_generation/chapter76_qwen3_vl_abs2_plan.md) 让缓存的 `sum!(abs2)` 计划带上具体字段类型，kernel 和 launch configuration 不变。query、key 前缀和 hidden 长度 `2048` 都与 `sum!` 逐位一致。2026-10-09 一次 repeat 的重测 SHA256 为 `7aad1e4a7a720fe31136200fd016125e0ed48d002e073b62ed4e16f1060d6fd1`。GPU 最后一步仍是 `0`。position `107 / 203 / 331` 的 CPU 都是 `513,152` bytes。32-token prefix 仍与 HF exact，128/256 仍在 step 70 分叉，margin `0.02797699`。Chapter 76 于 2026-10-09 Closed。Episode 10 保持 Open。最终 logits 没有升成 Float32，也不声称整条 decode 零分配。

[`Chapter 77 — Qwen3-VL decode 单 token K/V 写入`](notes/episodes/episode10_qwen3_vl_efficient_generation/chapter77_qwen3_vl_kv_write.md) 把单 token 的 K/V `copyto!` 换成拷贝 kernel。index `1 / 80 / 331` 的整段 cache 与 `copyto!` 逐位一致。2026-10-09 一次 repeat 的重测 SHA256 为 `20917f6fd9f56e3f2cfa8752c14f857d7d72ad99268c35ac6dc16a17347fc792`。GPU 最后一步仍是 `0`。256-token 稳态一步的 CPU 是 `441,024` bytes。32-token prefix 仍与 HF exact，128/256 仍在 step 70 分叉，margin `0.02797699`。Chapter 77 于 2026-10-09 Closed。Episode 10 保持 Open。最终 logits 没有升成 Float32，也不声称整条 decode 零分配。

[`Chapter 78 — Qwen3-VL decode attention softmax 分母`](notes/episodes/episode10_qwen3_vl_efficient_generation/chapter78_qwen3_vl_attention_sum.md) 在 `sum!` 仍是单 block 时改走同一条 warp 树。sequence `1:331` 的 16 个分母逐位一致。更长的多 block 归约仍走 `sum!`。2026-10-09 一次 repeat 的重测 SHA256 为 `ea2340ed9ab2231f79a1e93490d72c2cd17b5411e51b8a2f2dda00c93aed0290`。GPU 最后一步仍是 `0`。position `107 / 203 / 331` 的 CPU 都是 `424,896` bytes。32-token prefix 仍与 HF exact，128/256 仍在 step 70 分叉，margin `0.02797699`。Chapter 78 于 2026-10-09 Closed。Episode 10 保持 Open。最终 logits 没有升成 Float32，也不声称整条 decode 零分配。

[`Chapter 79 — Qwen3-VL decode QK abs2 的具体引用`](notes/episodes/episode10_qwen3_vl_efficient_generation/chapter79_qwen3_vl_qk_abs2_ref.md) 把 Q/K 的单 block abs2 计划放进具体 `Ref`。query 与 key 都和 `sum!(abs2)` 逐位一致。hidden 的多 block 归约仍走原来的字典。2026-10-09 一次 repeat 的重测 SHA256 为 `10da9cdb7e18588ce9431f3a8dcc9b719a9062ea12d53cb107304f1d4396fcc2`。GPU 最后一步仍是 `0`。position `107 / 203 / 331` 的 CPU 都是 `415,936` bytes。32-token prefix 仍与 HF exact，128/256 仍在 step 70 分叉，margin `0.02797699`。Chapter 79 于 2026-10-09 Closed。Episode 10 保持 Open。最终 logits 没有升成 Float32，也不声称整条 decode 零分配。

[`Chapter 80 — Qwen3-VL decode QK scale 的 reshape`](notes/episodes/episode10_qwen3_vl_efficient_generation/chapter80_qwen3_vl_qk_scale.md) 复用 Q/K 长度 `128` 的 scale reshape。归一化输出与每次现做的 reshape 逐位一致。2026-10-09 一次 repeat 的重测 SHA256 为 `9105965aca4d2c296b5f41b34f4820af00a60ee277153fb38db3dc0ed129fb50`。GPU 最后一步仍是 `0`。position `107 / 203 / 331` 的 CPU 都是 `415,744` bytes。32-token prefix 仍与 HF exact，128/256 仍在 step 70 分叉，margin `0.02797699`。Chapter 80 于 2026-10-09 Closed。Episode 10 保持 Open。最终 logits 没有升成 Float32，也不声称整条 decode 零分配。

[`Chapter 81 — Qwen3-VL decode hidden abs2 的尾部归约`](notes/episodes/episode10_qwen3_vl_efficient_generation/chapter81_qwen3_vl_hidden_tail.md) 把 hidden abs2 的 4 块尾部换成与 `sum!` 相同的 warp 树。整段归约仍和 `sum!(abs2)` 逐位一致。2026-10-09 一次 repeat 的重测 SHA256 为 `73e26bcfc57fea24cd2ba006b1aa51fe2ca5b245bb73f00c32766f6151dd9721`。GPU 最后一步仍是 `0`。position `107 / 203 / 331` 的 CPU 都是 `394,768` bytes。32-token prefix 仍与 HF exact，128/256 仍在 step 70 分叉，margin `0.02797699`。Chapter 81 于 2026-10-09 Closed。Episode 10 保持 Open。最终 logits 没有升成 Float32，也不声称整条 decode 零分配。

[`Chapter 82 — Qwen3-VL decode K/V 的同一次拷贝`](notes/episodes/episode10_qwen3_vl_efficient_generation/chapter82_qwen3_vl_kv_pair.md) 把 K/V 拷进 attention scratch 的两次 kernel，以及单 token 写回 cache 的两次 kernel，各收成一次启动。整段 K/V 与原来的路径逐位一致。2026-10-09 一次 repeat 的重测 SHA256 为 `adbfdda29a4aa59a751cd48f5547822b8d21fd947225efdd5b28225a68e9da6c`。GPU 最后一步仍是 `0`。position `107 / 203 / 331` 的 CPU 都是 `388,496` bytes。32-token prefix 仍与 HF exact，128/256 仍在 step 70 分叉，margin `0.02797699`。Chapter 82 于 2026-10-09 Closed。Episode 10 保持 Open。最终 logits 没有升成 Float32，也不声称整条 decode 零分配。

[`Chapter 83 — Qwen3-VL decode Q/K RoPE 的同一次启动`](notes/episodes/episode10_qwen3_vl_efficient_generation/chapter83_qwen3_vl_rope_pair.md) 把 Q 和 K 的 RoPE 收成一次启动。输出与两次单张量 kernel 逐位一致。2026-10-09 一次 repeat 的重测 SHA256 为 `39f8c6ab8f0320a24c27c7f5af400ea98c45322fe91282be5893fbc96ca80fad`。GPU 最后一步仍是 `0`。position `107 / 203 / 331` 的 CPU 都是 `384,464` bytes。32-token prefix 仍与 HF exact，128/256 仍在 step 70 分叉，margin `0.02797699`。Chapter 83 于 2026-10-09 Closed。Episode 10 保持 Open。最终 logits 没有升成 Float32，也不声称整条 decode 零分配。

[`Chapter 84 — Qwen3-VL decode attention 的准备启动`](notes/episodes/episode10_qwen3_vl_efficient_generation/chapter84_qwen3_vl_attn_prepare.md) 把 query 载入和 K/V 拷贝收成一次启动。sequence `1 / 80 / 331` 与原来的两次 kernel 逐位一致。2026-10-09 一次 repeat 的重测 SHA256 为 `fe36b8825d6582c932c9bf264db617d04a1276bff0210987320c78f001d54a35`。GPU 最后一步仍是 `0`。position `107 / 203 / 331` 的 CPU 都是 `381,328` bytes。32-token prefix 仍与 HF exact，128/256 仍在 step 70 分叉，margin `0.02797699`。Chapter 84 于 2026-10-09 Closed。Episode 10 保持 Open。最终 logits 没有升成 Float32，也不声称整条 decode 零分配。

[`Chapter 85 — Qwen3-VL decode attention weight 的置换`](notes/episodes/episode10_qwen3_vl_efficient_generation/chapter85_qwen3_vl_weight_perm.md) 把 softmax 权重和把它置换到 `(token, group, head)` 收成一次启动。sequence `1 / 80 / 331` 的整段权重和置换结果与原来的两次 kernel 逐位一致。2026-10-09 一次 repeat 的重测 SHA256 为 `922a55bc3533657cc5a09278239cd6e64d20c443147215edd441ab69b58fcda7`。GPU 最后一步仍是 `0`。position `107 / 203 / 331` 的 CPU 都是 `377,744` bytes。32-token prefix 仍与 HF exact，128/256 仍在 step 70 分叉，margin `0.02797699`。Chapter 85 于 2026-10-09 Closed。Episode 10 保持 Open。最终 logits 没有升成 Float32，也不声称整条 decode 零分配。

[`Chapter 86 — Qwen3-VL decode QK RMSNorm 的同一次启动`](notes/episodes/episode10_qwen3_vl_efficient_generation/chapter86_qwen3_vl_qk_fused.md) 把 Q 和 K 的 RMSNorm 按共享 scratch 的顺序收成一次启动。输出和 `mean_square` 与两条单张量路径逐位一致。2026-10-09 一次 repeat 的重测 SHA256 为 `1f6e990c7b7e060edd280755d60ff385279370538e3f81a038e71eeaa78d9f3c`。GPU 最后一步仍是 `0`。position `107 / 203 / 331` 的 CPU 都是 `365,648` bytes。32-token prefix 仍与 HF exact，128/256 仍在 step 70 分叉，margin `0.02797699`。Chapter 86 于 2026-10-09 Closed。Episode 10 保持 Open。最终 logits 没有升成 Float32，也不声称整条 decode 零分配。

[`Chapter 87 — Qwen3-VL decode attention softmax 的同一次启动`](notes/episodes/episode10_qwen3_vl_efficient_generation/chapter87_qwen3_vl_softmax.md) 把 scale、max、exp、sum 和权重置换按原顺序收成一次启动。sequence `1 / 80 / 330 / 331` 与五次 kernel 逐位一致。2026-10-09 一次 repeat 的重测 SHA256 为 `9d41f7461c0387d88d2333f6c9184108380140695af958ca2c6940af7673c331`。GPU 最后一步仍是 `0`。position `107 / 203 / 331` 的 CPU 都是 `344,144` bytes。32-token prefix 仍与 HF exact，128/256 仍在 step 70 分叉，margin `0.02797699`。Chapter 87 于 2026-10-09 Closed。Episode 10 保持 Open。最终 logits 没有升成 Float32，也不声称整条 decode 零分配。

[`Chapter 88 — Qwen3-VL decode mRoPE 频率表`](notes/episodes/episode10_qwen3_vl_efficient_generation/chapter88_qwen3_vl_mrope_tables.md) 复用 inv frequency 和 host 上的 cos/sin 缓冲。coordinate `0 / 1 / 76 / 100 / 331 / 10000` 与原来的频率函数逐位一致。2026-10-09 一次 repeat 的重测 SHA256 为 `9d3a146875250b9dcfa4ded7b7c617620903cd20f21279d71973660b0a9965f5`。GPU 最后一步仍是 `0`。position `107 / 203 / 331` 的 CPU 都是 `342,672` bytes。32-token prefix 仍与 HF exact，128/256 仍在 step 70 分叉，margin `0.02797699`。Chapter 88 于 2026-10-09 Closed。Episode 10 保持 Open。最终 logits 没有升成 Float32，也不声称整条 decode 零分配。

[`Chapter 89 — Qwen3-VL decode 单 token GEMM`](notes/episodes/episode10_qwen3_vl_efficient_generation/chapter89_qwen3_vl_gemm_token.md) 让单 token 的 `cublasGemmEx` 直接使用 `(out, 1, 1)` 和 `(in, 1, 1)` 的 stride。四个形状与 reshape 路径逐位一致。2026-10-09 一次 repeat 的重测 SHA256 为 `4c20846e26ec45e49530b9e5568a7f0f75f15741a84154ac5834030cd1d896a6`。GPU 最后一步仍是 `0`。position `107 / 203 / 331` 的 CPU 都是 `312,208` bytes。32-token prefix 仍与 HF exact，128/256 仍在 step 70 分叉，margin `0.02797699`。Chapter 89 于 2026-10-09 Closed。Episode 10 保持 Open。最终 logits 没有升成 Float32，也不声称整条 decode 零分配。

[`Chapter 90 — Qwen3-VL decode hidden RMSNorm 的两次 kernel`](notes/episodes/episode10_qwen3_vl_efficient_generation/chapter90_qwen3_vl_hidden_fused.md) 在 abs2 计划仍是 4×576 时，把长度 `2048` 的 hidden RMSNorm 收成两次 kernel。四组随机输入的 output 和 `mean_square` 与原来的路径逐位一致。2026-10-09 一次 repeat 的重测 SHA256 为 `5cea2add288724b3126c025c3ca3f0ca05352ea5d6f387288516a292d77f6ccf`。GPU 最后一步仍是 `0`。position `107 / 203 / 331` 的 CPU 都是 `297,952` bytes。32-token prefix 仍与 HF exact，128/256 仍在 step 70 分叉，margin `0.02797699`。Chapter 90 于 2026-10-09 Closed。Episode 10 保持 Open。最终 logits 没有升成 Float32，也不声称整条 decode 零分配。

[`Chapter 91 — Qwen3-VL decode attention softmax 的启动对象`](notes/episodes/episode10_qwen3_vl_efficient_generation/chapter91_qwen3_vl_softmax_launch.md) 把编译好的 softmax kernel 留在具体 `Ref` 里，成功时返回 `Bool`。sequence `1 / 2 / 31 / 32 / 33 / 80 / 330 / 331` 与五次 kernel 逐位一致。2026-10-09 一次 repeat 的重测 SHA256 为 `00534ab94c2fa492f615082b041b007f042cf72e0f907bb2a5f19ec8d0d4ea09`。GPU 最后一步仍是 `0`。position `107 / 203 / 331` 的 CPU 都是 `288,544` bytes。32-token prefix 仍与 HF exact，128/256 仍在 step 70 分叉，margin `0.02797699`。Chapter 91 于 2026-10-09 Closed。Episode 10 保持 Open。最终 logits 没有升成 Float32，也不声称整条 decode 零分配。

[`Chapter 92 — Qwen3-VL decode cublasGemmEx 的 context`](notes/episodes/episode10_qwen3_vl_efficient_generation/chapter92_qwen3_vl_gemm_context.md) 让同一条 `cublasGemmEx` 不再每次查询当前 context。矩阵乘、转置和 8 头 batched mul 与原来的包装逐位一致。2026-10-09 一次 repeat 的重测 SHA256 为 `5ca785fc6469ce8cd9abc963e9a7a362126eb2f01bf3fb490e5bcf272ab708e3`。GPU 最后一步仍是 `0`。position `107 / 203 / 331` 的 CPU 都是 `278,224` bytes。32-token prefix 仍与 HF exact，128/256 仍在 step 70 分叉，margin `0.02797699`。Chapter 92 于 2026-10-09 Closed。Episode 10 保持 Open。最终 logits 没有升成 Float32，也不声称整条 decode 零分配。

[`Chapter 93 — Qwen3-VL decode attention query 的 reshape`](notes/episodes/episode10_qwen3_vl_efficient_generation/chapter93_qwen3_vl_query_matrix.md) 复用共享 scratch 上 `(groups, head_dim, kv_heads)` 的 query 矩阵。sequence `1 / 80 / 331` 与原来的载入加拷贝逐位一致。2026-10-09 一次 repeat 的重测 SHA256 为 `5337c63eef2bf9bfaacdd1929ee173744b7acc6cfac967f87310299341895fca`。GPU 最后一步仍是 `0`。position `107 / 203 / 331` 的 CPU 都是 `275,536` bytes。32-token prefix 仍与 HF exact，128/256 仍在 step 70 分叉，margin `0.02797699`。Chapter 93 于 2026-10-09 Closed。Episode 10 保持 Open。最终 logits 没有升成 Float32，也不声称整条 decode 零分配。

[`Chapter 94 — Qwen3-VL decode hidden RMSNorm 的一次启动`](notes/episodes/episode10_qwen3_vl_efficient_generation/chapter94_qwen3_vl_hidden_single.md) 在 abs2 计划仍是 4×576 时，把长度 `2048` 的 hidden RMSNorm 收成一次启动。六组随机输入的 output 和 `mean_square` 与 `sum!(abs2)` 路径逐位一致。2026-10-09 一次 repeat 的重测 SHA256 为 `9e122b995b02a9c2d85b6b1ae67491aa8ea541adb581160493d00d30f398bd10`。GPU 最后一步仍是 `0`。position `107 / 203 / 331` 的 CPU 都是 `266,416` bytes。32-token prefix 仍与 HF exact，128/256 仍在 step 70 分叉，margin `0.02797699`。Chapter 94 于 2026-10-09 Closed。Episode 10 保持 Open。最终 logits 没有升成 Float32，也不声称整条 decode 零分配。

[`Chapter 95 — Qwen3-VL decode attention softmax 的直接启动`](notes/episodes/episode10_qwen3_vl_efficient_generation/chapter95_qwen3_vl_softmax_raw.md) 把 softmax 参数留在具体盒子里，直接调用 `cuLaunchKernel`。sequence `1 / 2 / 31 / 32 / 33 / 80 / 330 / 331` 与五次 kernel 逐位一致。2026-10-09 一次 repeat 的重测 SHA256 为 `c1f8e35ac1c364dd29d19c2814e1198406e3133fee7cec71d215bdea99463d9b`。GPU 最后一步仍是 `0`。position `107 / 203 / 331` 的 CPU 都是 `260,144` bytes。32-token prefix 仍与 HF exact，128/256 仍在 step 70 分叉，margin `0.02797699`。Chapter 95 于 2026-10-09 Closed。Episode 10 保持 Open。最终 logits 没有升成 Float32，也不声称整条 decode 零分配。

[`Chapter 96 — Qwen3-VL decode attention 准备和写回的直接启动`](notes/episodes/episode10_qwen3_vl_efficient_generation/chapter96_qwen3_vl_attn_raw.md) 把 attention 准备和 context 写回的参数留在具体盒子里，直接调用 `cuLaunchKernel`。sequence `1 / 2 / 31 / 32 / 33 / 80 / 330 / 331` 与原来的路径逐位一致。2026-10-09 一次 repeat 的重测 SHA256 为 `24cd654c35811cc9375573995687bbfc47fb207237a28d2b9b1d7d8f0b616896`。GPU 最后一步仍是 `0`。position `107 / 203 / 331` 的 CPU 都是 `250,288` bytes。32-token prefix 仍与 HF exact，128/256 仍在 step 70 分叉，margin `0.02797699`。Chapter 96 于 2026-10-09 Closed。Episode 10 保持 Open。最终 logits 没有升成 Float32，也不声称整条 decode 零分配。

[`Chapter 97 — Qwen3-VL decode hidden RMSNorm 的直接启动`](notes/episodes/episode10_qwen3_vl_efficient_generation/chapter97_qwen3_vl_hidden_raw.md) 把长度 `2048` 的 hidden RMSNorm 参数留在具体盒子里，直接调用 `cuLaunchKernel`。四组 scale 的 output 和 `mean_square` 与 `sum!(abs2)` 路径逐位一致。2026-10-09 一次 repeat 的重测 SHA256 为 `dc1a6db24f75ade00da6e847ca52996b11d3770537823b755cf2519003ab08db`。GPU 最后一步仍是 `0`。position `107 / 203 / 331` 的 CPU 都是 `240,256` bytes。32-token prefix 仍与 HF exact，128/256 仍在 step 70 分叉，margin `0.02797699`。Chapter 97 于 2026-10-09 Closed。Episode 10 保持 Open。最终 logits 没有升成 Float32，也不声称整条 decode 零分配。

[`Chapter 98 — Qwen3-VL decode residual 加法的直接启动`](notes/episodes/episode10_qwen3_vl_efficient_generation/chapter98_qwen3_vl_residual_raw.md) 把 residual 加法的参数留在具体盒子里，直接调用 `cuLaunchKernel`。两组输入与 `BFloat16(Float32(residual) + Float32(projected))` 逐位一致。2026-10-09 一次 repeat 的重测 SHA256 为 `3db90548a5f9e730f543a5db2dd4885817d8b325d2b198407a090d404c3a9d2b`。GPU 最后一步仍是 `0`。position `107 / 203 / 331` 的 CPU 都是 `231,296` bytes。32-token prefix 仍与 HF exact，128/256 仍在 step 70 分叉，margin `0.02797699`。Chapter 98 于 2026-10-09 Closed。Episode 10 保持 Open。最终 logits 没有升成 Float32，也不声称整条 decode 零分配。

[`Chapter 99 — Qwen3-VL decode QKV 的 reshape`](notes/episodes/episode10_qwen3_vl_efficient_generation/chapter99_qwen3_vl_qkv_matrix.md) 复用共享 scratch 上 query、key、value 的 `(head_dim, heads, 1, 1)` 矩阵。两组输入与每次现做的 reshape 逐位一致。2026-10-09 一次 repeat 的重测 SHA256 为 `e1f0049928c88d1aa76b452ac8ca9400266bbd526717d9218eb572384036f1bf`。GPU 最后一步仍是 `0`。position `107 / 203 / 331` 的 CPU 都是 `223,232` bytes。32-token prefix 仍与 HF exact，128/256 仍在 step 70 分叉，margin `0.02797699`。Chapter 99 于 2026-10-09 Closed。Episode 10 保持 Open。最终 logits 没有升成 Float32，也不声称整条 decode 零分配。

[`Chapter 100 — Qwen3-VL decode QK RMSNorm 的直接启动`](notes/episodes/episode10_qwen3_vl_efficient_generation/chapter100_qwen3_vl_qk_raw.md) 把 QK RMSNorm 的参数留在具体盒子里，直接调用 `cuLaunchKernel`，并把每层 scale 的 reshape 留住。四组 scale 的 query、key 和 `mean_square` 与两条单张量路径逐位一致。2026-10-09 一次 repeat 的重测 SHA256 为 `cc01142c0d717ac61f2e769e90e88ff6f0a6c41a34311fdd33833c0e8ef1e84d`。GPU 最后一步仍是 `0`。position `107 / 203 / 331` 的 CPU 都是 `211,776` bytes。32-token prefix 仍与 HF exact，128/256 仍在 step 70 分叉，margin `0.02797699`。Chapter 100 于 2026-10-09 Closed。Episode 10 保持 Open。最终 logits 没有升成 Float32，也不声称整条 decode 零分配。

[`Chapter 101 — Qwen3-VL decode QK RoPE 的直接启动`](notes/episodes/episode10_qwen3_vl_efficient_generation/chapter101_qwen3_vl_rope_raw.md) 把 Q/K RoPE 的参数留在具体盒子里，直接调用 `cuLaunchKernel`。三组 cos/sin 的输出与两条单张量 kernel 逐位一致。2026-10-09 一次 repeat 的重测 SHA256 为 `4eb335b744abf30c86693f3faf560c4d6d45b6a4908ac878d414d67bc37fb104`。GPU 最后一步仍是 `0`。position `107 / 203 / 331` 的 CPU 都是 `205,952` bytes。32-token prefix 仍与 HF exact，128/256 仍在 step 70 分叉，margin `0.02797699`。Chapter 101 于 2026-10-09 Closed。Episode 10 保持 Open。最终 logits 没有升成 Float32，也不声称整条 decode 零分配。

[`Chapter 102 — Qwen3-VL decode K/V 写入的直接启动`](notes/episodes/episode10_qwen3_vl_efficient_generation/chapter102_qwen3_vl_kv_raw.md) 把单 token K/V 写入的参数留在具体盒子里，直接调用 `cuLaunchKernel`。index `1 / 80 / 331` 的整段 cache 与 `copyto!` 逐位一致。2026-10-09 一次 repeat 的重测 SHA256 为 `545067cbab352dc90671bfa184b221e01c9b51c7cebdd18154a3894459992cc2`。GPU 最后一步仍是 `0`。position `107 / 203 / 331` 的 CPU 都是 `201,024` bytes。32-token prefix 仍与 HF exact，128/256 仍在 step 70 分叉，margin `0.02797699`。Chapter 102 于 2026-10-09 Closed。Episode 10 保持 Open。最终 logits 没有升成 Float32，也不声称整条 decode 零分配。

[`Chapter 103 — Qwen3-VL decode MLP SiLU 的直接启动`](notes/episodes/episode10_qwen3_vl_efficient_generation/chapter103_qwen3_vl_mlp_raw.md) 把 SiLU 乘积的参数留在具体盒子里，直接调用 `cuLaunchKernel`。两组输入与原来的舍入公式逐位一致。2026-10-09 一次 repeat 的重测 SHA256 为 `7b293b12258a3be217777064f9d1b4ad2d3e534c03ff2fb05a952eb0c7d8d576`。GPU 最后一步仍是 `0`。position `107 / 203 / 331` 的 CPU 都是 `196,544` bytes。32-token prefix 仍与 HF exact，128/256 仍在 step 70 分叉，margin `0.02797699`。Chapter 103 于 2026-10-09 Closed。Episode 10 保持 Open。最终 logits 没有升成 Float32，也不声称整条 decode 零分配。

[`Chapter 104 — Qwen3-VL decode stage 不再包闭包`](notes/episodes/episode10_qwen3_vl_efficient_generation/chapter104_qwen3_vl_stage_inline.md) 在没有 profiler 时直接执行 stage 表达式。有 runner 时仍把同一个表达式交给 profiler。2026-10-09 一次 repeat 的重测 SHA256 为 `942e68a633d89d896a587e7c7786f03da1a293ef0da916df20f0cf35e2e9e125`。GPU 最后一步仍是 `0`。position `331` 的 CPU 是 `129,600` bytes，`107 / 203` 是 `129,616` bytes。32-token prefix 仍与 HF exact，128/256 仍在 step 70 分叉，margin `0.02797699`。Chapter 104 于 2026-10-09 Closed。Episode 10 保持 Open。最终 logits 没有升成 Float32，也不声称整条 decode 零分配。

[`Chapter 105 — Qwen3-VL decode 单 token 层走热路径`](notes/episodes/episode10_qwen3_vl_efficient_generation/chapter105_qwen3_vl_decode_block.md) 在没有 profiler 时让 BF16 单 token 层只走热路径。有 profiler 时仍走原来的 stage 函数。专用一层的 hidden、K、V 与原来的 stage 函数逐位一致。2026-10-09 一次 repeat 的重测 SHA256 为 `414590cfcf94ced3dd4d15f83cdd5a392e8bd584d07e2d4f52d6dc516c894135`。GPU 最后一步仍是 `0`。position `331` 的 CPU 是 `22,080` bytes，`107 / 203` 是 `22,096` bytes。32-token prefix 仍与 HF exact，128/256 仍在 step 70 分叉，margin `0.02797699`。Chapter 105 于 2026-10-09 Closed。Episode 10 保持 Open。最终 logits 没有升成 Float32，也不声称整条 decode 零分配。

[`Chapter 106 — Qwen3-VL decode 28 层一次调用`](notes/episodes/episode10_qwen3_vl_efficient_generation/chapter106_qwen3_vl_decode_layers.md) 把 28 层热路径收进一个具体类型的函数，大函数只调用一次。层内计算与 Chapter 105 相同。2026-10-09 一次 repeat 的重测 SHA256 为 `8e06c526e0bd0c6a12572979034dccd8c0a38475d10a525411b5c1dab547bb6b`。GPU 最后一步仍是 `0`。position `331` 的 CPU 是 `16,768` bytes，`107 / 203` 是 `16,784` bytes。32-token prefix 仍与 HF exact，128/256 仍在 step 70 分叉，margin `0.02797699`。Chapter 106 于 2026-10-09 Closed。Episode 10 保持 Open。最终 logits 没有升成 Float32，也不声称整条 decode 零分配。

[`Chapter 107 — Qwen3-VL decode 复用 cache 重叠检查`](notes/episodes/episode10_qwen3_vl_efficient_generation/chapter107_qwen3_vl_seen_storage.md) 把每步两次重建的 storage 列表留在 cache 上，清空后继续做原来的重叠检查。2026-10-09 一次 repeat 的重测 SHA256 为 `0e9e3ec5332667677e006ddda68de743ff39071effd1bdcdae4624ad7eae3273`。GPU 最后一步仍是 `0`。position `331` 的 CPU 是 `13,344` bytes，`107 / 203` 是 `13,360` bytes。32-token prefix 仍与 HF exact，128/256 仍在 step 70 分叉，margin `0.02797699`。Chapter 107 于 2026-10-09 Closed。Episode 10 保持 Open。最终 logits 没有升成 Float32，也不声称整条 decode 零分配。

[`Chapter 108 — Qwen3-VL decode 不再拷贝整份参数`](notes/episodes/episode10_qwen3_vl_efficient_generation/chapter108_qwen3_vl_decode_pack.md) 把 blocks、embedding、final norm 和 spec 留在 cache 上的可变包装里。稳态的 28 层调用传这个包装。2026-10-09 一次 repeat 的重测 SHA256 为 `192477eaa948f93a095264f64205930cbaa87b75f8570e9eee22c25256b4f22b`。GPU 最后一步仍是 `0`。position `331` 的 CPU 是 `10,272` bytes，`107 / 203` 是 `10,288` bytes。32-token prefix 仍与 HF exact，128/256 仍在 step 70 分叉，margin `0.02797699`。Chapter 108 于 2026-10-09 Closed。Episode 10 保持 Open。最终 logits 没有升成 Float32，也不声称整条 decode 零分配。

[`Chapter 109 — Qwen3-VL 长生成计时不再捕获参数`](notes/episodes/episode10_qwen3_vl_efficient_generation/chapter109_qwen3_vl_time_decode.md) 把长生成的 decode 计时放进小函数。参数先作为参数传进去，再计时同一次 decode。2026-10-09 一次 repeat 的重测 SHA256 为 `d82e5e953538c6d1631e09ac9bfe887325439f4055831627371c1378dbfa74aa`。GPU 最后一步仍是 `0`。position `331` 的 CPU 是 `7,200` bytes，`107 / 203` 是 `7,216` bytes。32-token prefix 仍与 HF exact，128/256 仍在 step 70 分叉，margin `0.02797699`。Chapter 109 于 2026-10-09 Closed。Episode 10 保持 Open。最终 logits 没有升成 Float32，也不声称整条 decode 零分配。

[`Chapter 110 — Qwen3-VL decode token embedding 直接拷贝`](notes/episodes/episode10_qwen3_vl_efficient_generation/chapter110_qwen3_vl_embedding_copy.md) 用一次 kernel 抄 embedding 的连续列。CPU 数组仍走原来的广播。2026-10-09 一次 repeat 的重测 SHA256 为 `ddd117e0443de28401032c581f353bfcdf8f983894b3a2a42ec2d01a5bb3655b`。GPU 最后一步仍是 `0`。position `331` 的 CPU 是 `5,344` bytes，`107 / 203` 是 `5,360` bytes。32-token prefix 仍与 HF exact，128/256 仍在 step 70 分叉，margin `0.02797699`。Chapter 110 于 2026-10-09 Closed。Episode 10 保持 Open。最终 logits 没有升成 Float32，也不声称整条 decode 零分配。

[`Chapter 111 — Qwen3-VL decode mRoPE 不再传递整个 spec`](notes/episodes/episode10_qwen3_vl_efficient_generation/chapter111_qwen3_vl_mrope_fields.md) 让 mRoPE 只接收 `head_dim`、`rope_theta` 和 `mrope_section`。频率公式没有变。2026-10-09 一次 repeat 的重测 SHA256 为 `a0cd3dc851bc37e61cf7e6d978514651fbcf51ae340b3e91ee461bc5d1f6778d`。GPU 最后一步仍是 `0`。position `331` 的 CPU 是 `5,248` bytes，`107 / 203` 是 `5,264` bytes。32-token prefix 仍与 HF exact，128/256 仍在 step 70 分叉，margin `0.02797699`。Chapter 111 于 2026-10-09 Closed。Episode 10 保持 Open。最终 logits 没有升成 Float32，也不声称整条 decode 零分配。

[`Chapter 112 — Qwen3-VL decode 复用具体 scratch`](notes/episodes/episode10_qwen3_vl_efficient_generation/chapter112_qwen3_vl_decode_ready.md) 在第一步 BF16 decode 之后把 scratch 留在具体类型的包装里。后面的步不再走 ensure。2026-10-09 一次 repeat 的重测 SHA256 为 `232a29ef3afb41f19c443d3aa2a5006cf13691c8e232afb609ea7b2422ed644d`。GPU 最后一步仍是 `0`。position `331` 的 CPU 是 `4,400` bytes，`107 / 203` 是 `4,416` bytes。32-token prefix 仍与 HF exact，128/256 仍在 step 70 分叉，margin `0.02797699`。Chapter 112 于 2026-10-09 Closed。Episode 10 保持 Open。最终 logits 没有升成 Float32，也不声称整条 decode 零分配。

[`Chapter 113 — Qwen3-VL decode mRoPE 主机拷贝不再查询 context`](notes/episodes/episode10_qwen3_vl_efficient_generation/chapter113_qwen3_vl_mrope_copy.md) 用 `cuMemcpyHtoDAsync` 把 mRoPE 的主机表抄到设备。CPU 数组仍走 `copyto!`。2026-10-09 一次 repeat 的重测 SHA256 为 `c29c0a498bb5eb7f421effdc7212f3205599d968fedc39e13fce7670602def1d`。GPU 最后一步仍是 `0`。position `331` 的 CPU 是 `4,336` bytes，`107 / 203` 是 `4,352` bytes。32-token prefix 仍与 HF exact，128/256 仍在 step 70 分叉，margin `0.02797699`。Chapter 113 于 2026-10-09 Closed。Episode 10 保持 Open。最终 logits 没有升成 Float32，也不声称整条 decode 零分配。

[`Chapter 114 — Qwen3-VL decode tied logits 复用 reshape`](notes/episodes/episode10_qwen3_vl_efficient_generation/chapter114_qwen3_vl_tied_matrix.md) 把 tied logits 的输出和 hidden 收成的两张矩阵留住，再走原来的转置 GEMM。2026-10-09 一次 repeat 的重测 SHA256 为 `791f7e29a744f4402f061e12dc81ab1c6cc01ee386fa227ff7a04c8415387cb3`。GPU 最后一步仍是 `0`。position `331` 的 CPU 是 `4,176` bytes，`107 / 203` 是 `4,192` bytes。32-token prefix 仍与 HF exact，128/256 仍在 step 70 分叉，margin `0.02797699`。Chapter 114 于 2026-10-09 Closed。Episode 10 保持 Open。最终 logits 没有升成 Float32，也不声称整条 decode 零分配。

[`Chapter 115 — Qwen3-VL 稳态循环只进入一次具体包装`](notes/episodes/episode10_qwen3_vl_efficient_generation/chapter115_qwen3_vl_ready_loop.md) 让长生成的第一步装好具体包装，后面的步在已经知道类型的循环里计时。ready 步本身是 `0` bytes。2026-10-09 一次 repeat 的重测 SHA256 为 `4caaf5d2593f99549eb761f7285bd1d9e4911bba253ed76e33739a2c6145bd3c`。GPU 最后一步仍是 `0`。position `107 / 203 / 331` 的 CPU 都是 `4,144` bytes。32-token prefix 仍与 HF exact，128/256 仍在 step 70 分叉，margin `0.02797699`。Chapter 115 于 2026-10-09 Closed。Episode 10 保持 Open。最终 logits 没有升成 Float32，也不声称整条 decode 零分配。

[`Chapter 116 — Qwen3-VL 稳态计时不再包一层 @sync`](notes/episodes/episode10_qwen3_vl_efficient_generation/chapter116_qwen3_vl_ready_timer.md) 把稳态步的 `CUDA.@timed` 换成窗口外同步的 `gc_num` 计数。ready 步本身是 `0` bytes，原先的 `4,144` bytes 是计时器里的 `@sync`。2026-10-09 一次 repeat 的重测 SHA256 为 `36246c5cc9e2d2e47fc1392db105143c6375c87f46109ac803cb65f41699576a`。GPU 最后一步仍是 `0`。position `107 / 203 / 331` 的 CPU 都是 `0` bytes。32-token prefix 仍与 HF exact，128/256 仍在 step 70 分叉，margin `0.02797699`。Chapter 116 于 2026-10-09 Closed。Episode 10 保持 Open。最终 logits 没有升成 Float32，也不声称整条 decode 零分配。

[`Chapter 117 — Qwen3-VL 主机选 token 不再做全长排序`](notes/episodes/episode10_qwen3_vl_efficient_generation/chapter117_qwen3_vl_top_two.md) 用一次线性扫描选出前两名。顺序与 `partialsortperm` 相同，相等时保留更小的下标。2026-10-09 一次 repeat 的重测 SHA256 为 `978dd531c9f64d4f0deda0237ce67d62db75712d312bb6406826da636112a02e`。GPU 最后一步仍是 `0`。position `107 / 203 / 331` 的 CPU 仍是 `0` bytes。256-token 主机选择从 `2,739,400` 降到 `308,232` bytes。32-token prefix 仍与 HF exact，128/256 仍在 step 70 分叉，margin `0.02797699`。Chapter 117 于 2026-10-09 Closed。Episode 10 保持 Open。最终 logits 没有升成 Float32，也不声称整条 decode 零分配。

[`Chapter 118 — Qwen3-VL 主机 logits 复用一块缓冲`](notes/episodes/episode10_qwen3_vl_efficient_generation/chapter118_qwen3_vl_logit_copy.md) 把连续的 BF16 logits 用同步 `cuMemcpyDtoH` 抄进一块复用的主机缓冲。2026-10-09 一次 repeat 的重测 SHA256 为 `1a5ee12eadee76b467864460e202d2a371532eb5637337e1c06056abc882722c`。GPU 最后一步仍是 `0`。position `107 / 203 / 331` 的 CPU 仍是 `0` bytes。256-token 主机选择从 `308,232` 降到 `0` bytes。32-token prefix 仍与 HF exact，128/256 仍在 step 70 分叉，margin `0.02797699`。Chapter 118 于 2026-10-09 Closed。Episode 10 保持 Open。最终 logits 没有升成 Float32，也不声称整条 decode 零分配。

[`Chapter 119 — Qwen3-VL 第一次 decode 不再重拷 blocks`](notes/episodes/episode10_qwen3_vl_efficient_generation/chapter119_qwen3_vl_ready_install.md) 让 ready 沿用 pack 里已经放下的 blocks。2026-10-09 一次 repeat 的重测 SHA256 为 `ec735216c159c84be0712a9bbe46c35df4600d4bf40e5a5eb7c35667f0d1bbc2`。GPU 最后一步仍是 `0`。position `107 / 203 / 331` 的 CPU 仍是 `0` bytes。128/256 的第一次 decode CPU 从 `40,368` 降到 `37,664` bytes。32-token prefix 仍与 HF exact，128/256 仍在 step 70 分叉，margin `0.02797699`。Chapter 119 于 2026-10-09 Closed。Episode 10 保持 Open。最终 logits 没有升成 Float32，也不声称整条 decode 零分配。

[`Chapter 120 — Qwen3-VL attention scratch 计字节不再装箱`](notes/episodes/episode10_qwen3_vl_efficient_generation/chapter120_qwen3_vl_attention_bytes.md) 直接把 13 个字段的 `sizeof` 相加。字节公式没有变。2026-10-09 一次 repeat 的重测 SHA256 为 `16645b1138ca6369ca3a0b386ad53422680ae645765a2cc1addbf27d76abebcf`。GPU 最后一步仍是 `0`。position `107 / 203 / 331` 的 CPU 仍是 `0` bytes。128/256 的第一次 decode CPU 从 `37,664` 降到 `35,904` bytes。32-token prefix 仍与 HF exact，128/256 仍在 step 70 分叉，margin `0.02797699`。Chapter 120 于 2026-10-09 Closed。Episode 10 保持 Open。最终 logits 没有升成 Float32，也不声称整条 decode 零分配。

[`Chapter 121 — Qwen3-VL attention scratch 共用两块存储`](notes/episodes/episode10_qwen3_vl_efficient_generation/chapter121_qwen3_vl_attention_pack.md) 把 attention scratch 的 13 块数组切自两块 BF16/Float32 存储。CPU 路径仍是 `similar`。2026-10-09 一次 repeat 的重测 SHA256 为 `bab6018c177ba1083c6aa1a7e956180d6156b582e0bdd317fa53d65f1f800708`。GPU 最后一步仍是 `0`。position `107 / 203 / 331` 的 CPU 仍是 `0` bytes。128/256 的第一次 decode CPU 从 `35,904` 降到 `34,304` bytes。256-token 第一次 decode 的 GPU 字节仍是 `2,018,180`，分配次数从 `56` 降到 `45`。32-token prefix 仍与 HF exact，128/256 仍在 step 70 分叉，margin `0.02797699`。Chapter 121 于 2026-10-09 Closed。Episode 10 保持 Open。最终 logits 没有升成 Float32，也不声称整条 decode 零分配。

[`Chapter 122 — Qwen3-VL QK RoPE scratch 共用两块存储`](notes/episodes/episode10_qwen3_vl_efficient_generation/chapter122_qwen3_vl_rope_pack.md) 把 QK RoPE scratch 的 12 块数组切自两块 Float32/BF16 存储。CPU 路径仍是 `similar`。2026-10-09 一次 repeat 的重测 SHA256 为 `1a3c49ac4089ae92e1365725899c42fdea28cc98328b0112816e47348944ad97`。GPU 最后一步仍是 `0`。position `107 / 203 / 331` 的 CPU 仍是 `0` bytes。128/256 的第一次 decode CPU 从 `34,304` 降到 `32,864` bytes。256-token 第一次 decode 的 GPU 字节仍是 `2,018,180`，分配次数从 `45` 降到 `35`。32-token prefix 仍与 HF exact，128/256 仍在 step 70 分叉，margin `0.02797699`。Chapter 122 于 2026-10-09 Closed。Episode 10 保持 Open。最终 logits 没有升成 Float32，也不声称整条 decode 零分配。

[`Chapter 123 — Qwen3-VL QK norm scratch 共用两块存储`](notes/episodes/episode10_qwen3_vl_efficient_generation/chapter123_qwen3_vl_qk_norm_pack.md) 把 QK norm scratch 的 7 块数组切自两块 Float32/BF16 存储。CPU 路径仍是 `similar`。2026-10-09 一次 repeat 的重测 SHA256 为 `802f5ab3a2f4499dfff4f5b7de66b639145ae70a950128ec592976e69edf434f`。GPU 最后一步仍是 `0`。position `107 / 203 / 331` 的 CPU 仍是 `0` bytes。128/256 的第一次 decode CPU 从 `32,864` 降到 `32,224` bytes。256-token 第一次 decode 的 GPU 字节仍是 `2,018,180`，分配次数从 `35` 降到 `30`。32-token prefix 仍与 HF exact，128/256 仍在 step 70 分叉，margin `0.02797699`。Chapter 123 于 2026-10-09 Closed。Episode 10 保持 Open。最终 logits 没有升成 Float32，也不声称整条 decode 零分配。

[`Chapter 124 — Qwen3-VL hidden norm scratch 共用两块存储`](notes/episodes/episode10_qwen3_vl_efficient_generation/chapter124_qwen3_vl_hidden_norm_pack.md) 把 hidden norm scratch 的 6 块数组切自两块 Float32/BF16 存储。CPU 路径仍是 `similar`。2026-10-09 一次 repeat 的重测 SHA256 为 `02074e31b3194d537e593cc6289178fb862cd088544710a6d8958db55a75bccc`。GPU 最后一步仍是 `0`。position `107 / 203 / 331` 的 CPU 仍是 `0` bytes。128/256 的第一次 decode CPU 从 `32,224` 降到 `31,744` bytes。256-token 第一次 decode 的 GPU 字节仍是 `2,018,180`，分配次数从 `30` 降到 `26`。32-token prefix 仍与 HF exact，128/256 仍在 step 70 分叉，margin `0.02797699`。Chapter 124 于 2026-10-09 Closed。Episode 10 保持 Open。最终 logits 没有升成 Float32，也不声称整条 decode 零分配。

[`Chapter 125 — Qwen3-VL MLP activation scratch 共用两块存储`](notes/episodes/episode10_qwen3_vl_efficient_generation/chapter125_qwen3_vl_mlp_pack.md) 把 MLP activation scratch 的 5 块数组切自两块 Float32/BF16 存储。CPU 路径仍是 `similar`。2026-10-09 一次 repeat 的重测 SHA256 为 `7ffc03b19d4421bd06c3a98c583664b2765261adeab657554d01bfdfde9353d2`。GPU 最后一步仍是 `0`。position `107 / 203 / 331` 的 CPU 仍是 `0` bytes。128/256 的第一次 decode CPU 从 `31,744` 降到 `31,424` bytes。256-token 第一次 decode 的 GPU 字节仍是 `2,018,180`，分配次数从 `26` 降到 `23`。32-token prefix 仍与 HF exact，128/256 仍在 step 70 分叉，margin `0.02797699`。Chapter 125 于 2026-10-09 Closed。Episode 10 保持 Open。最终 logits 没有升成 Float32，也不声称整条 decode 零分配。

[`Chapter 126 — Qwen3-VL hidden residual scratch 共用两块存储`](notes/episodes/episode10_qwen3_vl_efficient_generation/chapter126_qwen3_vl_residual_pack.md) 把 hidden residual scratch 的 4 块数组切自两块 Float32/BF16 存储。CPU 路径仍是 `similar`。2026-10-09 一次 repeat 的重测 SHA256 为 `6db4aa2978fc8e80b064352f4968d6ccbeb236c6057f852244907eb84cff5336`。GPU 最后一步仍是 `0`。position `107 / 203 / 331` 的 CPU 仍是 `0` bytes。128/256 的第一次 decode CPU 从 `31,424` 降到 `31,264` bytes。256-token 第一次 decode 的 GPU 字节仍是 `2,018,180`，分配次数从 `23` 降到 `21`。32-token prefix 仍与 HF exact，128/256 仍在 step 70 分叉，margin `0.02797699`。Chapter 126 于 2026-10-09 Closed。Episode 10 保持 Open。最终 logits 没有升成 Float32，也不声称整条 decode 零分配。

[`Chapter 127 — Qwen3-VL ready 不再拷贝 blocks`](notes/episodes/episode10_qwen3_vl_efficient_generation/chapter127_qwen3_vl_ready_pack.md) 让 ready 只留着 pack，不再把 blocks 拷进自己。2026-10-09 一次 repeat 的重测 SHA256 为 `e5f49acc926b4b205186f93ed9f42c8e3c21583ffc5f1b4c01085e6f6594ed73`。GPU 最后一步仍是 `0`。position `107 / 203 / 331` 的 CPU 仍是 `0` bytes。128/256 的第一次 decode CPU 从 `31,264` 降到 `25,856` bytes。32-token prefix 仍与 HF exact，128/256 仍在 step 70 分叉，margin `0.02797699`。Chapter 127 于 2026-10-09 Closed。Episode 10 保持 Open。最终 logits 没有升成 Float32，也不声称整条 decode 零分配。

[`Chapter 128 — Qwen3-VL 第一次 decode 不再包一层 @sync`](notes/episodes/episode10_qwen3_vl_efficient_generation/chapter128_qwen3_vl_first_timer.md) 把第一次 decode 的 `CUDA.@timed` 换成窗口外同步的 `gc_num` 计数。多出来的 `4,144` bytes 是计时器里的 `@sync`。2026-10-09 一次 repeat 的重测 SHA256 为 `ee02b9b657eaef91346f255c6c2401ff1ba2a4e92f5756d6f8ef4952e7ce4016`。GPU 最后一步仍是 `0`。position `107 / 203 / 331` 的 CPU 仍是 `0` bytes。128/256 的第一次 decode CPU 从 `25,856` 降到 `21,712` bytes。32-token prefix 仍与 HF exact，128/256 仍在 step 70 分叉，margin `0.02797699`。Chapter 128 于 2026-10-09 Closed。Episode 10 保持 Open。最终 logits 没有升成 Float32，也不声称整条 decode 零分配。

[`Chapter 129 — Qwen3-VL 参数 pack 随 cache 创建`](notes/episodes/episode10_qwen3_vl_efficient_generation/chapter129_qwen3_vl_cache_pack.md) 把 blocks 那份 tuple 的拷贝挪到 static cache 创建时。第一次 decode 直接复用。2026-10-09 一次 repeat 的重测 SHA256 为 `137d69319e472b2faaf15ca5e89b097ebf87d09142d668a64fec64ba12b677c5`。GPU 最后一步仍是 `0`。position `107 / 203 / 331` 的 CPU 仍是 `0` bytes。128/256 的第一次 decode CPU 从 `21,712` 降到 `19,024` bytes。32-token prefix 仍与 HF exact，128/256 仍在 step 70 分叉，margin `0.02797699`。Chapter 129 于 2026-10-09 Closed。Episode 10 保持 Open。最终 logits 没有升成 Float32，也不声称整条 decode 零分配。

[`Chapter 130 — Qwen3-VL 九块 BF16 scratch 共用一块存储`](notes/episodes/episode10_qwen3_vl_efficient_generation/chapter130_qwen3_vl_bf16_tails.md) 把 QKV、gate/up、mRoPE、token embedding 和 tied logits 这 9 块 BF16 数组切自同一块存储。CPU 路径仍是 `similar`。2026-10-09 一次 repeat 的重测 SHA256 为 `31ee2e77b9a8bb327e803e73521791ba66eff6e5abb6e0b6770b89ca7b89c469`。GPU 最后一步仍是 `0`。position `107 / 203 / 331` 的 CPU 仍是 `0` bytes。128/256 的第一次 decode CPU 从 `19,024` 降到 `17,872` bytes。256-token 第一次 decode 的 GPU 字节仍是 `2,018,180`，分配次数从 `21` 降到 `13`。32-token prefix 仍与 HF exact，128/256 仍在 step 70 分叉，margin `0.02797699`。Chapter 130 于 2026-10-09 Closed。Episode 10 保持 Open。最终 logits 没有升成 Float32，也不声称整条 decode 零分配。

[`Chapter 131 — Qwen3-VL decode scratch 共用两块存储`](notes/episodes/episode10_qwen3_vl_efficient_generation/chapter131_qwen3_vl_decode_workspace.md) 把第一次 decode 的 13 块父存储收成一块 BF16 和一块 Float32。CPU 路径仍是 `similar`。2026-10-09 一次 repeat 的重测 SHA256 为 `2bc4e279c7355bd436203dd0982ce47b88b07c837d99911eecab08a0f49b8487`。GPU 最后一步仍是 `0`。position `107 / 203 / 331` 的 CPU 仍是 `0` bytes。128/256 的第一次 decode CPU 从 `17,872` 降到 `14,912` bytes。256-token 第一次 decode 的 GPU 字节仍是 `2,018,180`，分配次数从 `13` 降到 `2`。32-token prefix 仍与 HF exact，128/256 仍在 step 70 分叉，margin `0.02797699`。Chapter 131 于 2026-10-09 Closed。Episode 10 保持 Open。最终 logits 没有升成 Float32，也不声称整条 decode 零分配。

[`Chapter 132 — Qwen3-VL 复用 kernel 参数盒`](notes/episodes/episode10_qwen3_vl_efficient_generation/chapter132_qwen3_vl_param_reuse.md) 在 scratch 换成新数组时改写已有参数盒里的设备指针。2026-10-09 一次 repeat 的重测 SHA256 为 `1fb6be933ff633abbcd6c6126cf3924496534843d9871915d929efd07d97138b`。GPU 最后一步仍是 `0`。position `107 / 203 / 331` 的 CPU 仍是 `0` bytes。128/256 的第一次 decode CPU 从 `14,912` 降到 `9,168` bytes。256-token 第一次 decode 的 GPU 字节仍是 `2,018,180`，分配次数仍是 `2`。32-token prefix 仍与 HF exact，128/256 仍在 step 70 分叉，margin `0.02797699`。Chapter 132 于 2026-10-09 Closed。Episode 10 保持 Open。最终 logits 没有升成 Float32，也不声称整条 decode 零分配。

[`Chapter 133 — Qwen3-VL scratch 切片共用父引用`](notes/episodes/episode10_qwen3_vl_efficient_generation/chapter133_qwen3_vl_shared_ref.md) 让 decode scratch 的切片共用父缓冲的 `DataRef`，两块父缓冲留在 cache 上。CPU 路径仍是 `similar`。2026-10-10 一次 repeat 的重测 SHA256 为 `103dc1d18457d27054ddad1215f1d0b777b98694747e9b426f5101e0cc9230dd`。GPU 最后一步仍是 `0`。position `107 / 203 / 331` 的 CPU 仍是 `0` bytes。32/128/256 的第一次 decode CPU 都从 `9,168` 降到 `7,408` bytes。256-token 第一次 decode 的 GPU 字节仍是 `2,018,180`，分配次数仍是 `2`。32-token prefix 仍与 HF exact，128/256 仍在 step 70 分叉，margin `0.02797699`。Chapter 133 于 2026-10-10 Closed。Episode 10 保持 Open。最终 logits 没有升成 Float32，也不声称整条 decode 零分配。

[`Chapter 134 — Qwen3-VL 复用 reshape 矩阵`](notes/episodes/episode10_qwen3_vl_efficient_generation/chapter134_qwen3_vl_reshape_reuse.md) 在新的 scratch 上改写已有 QKV、tied、query 和 attention 投影矩阵的指针和形状。2026-10-10 一次 repeat 的重测 SHA256 为 `92cce38e05d6123d6f55f4a520cbc4c7df36ce3be53339690296fafa904cb17a`。GPU 最后一步仍是 `0`。position `107 / 203 / 331` 的 CPU 仍是 `0` bytes。32/128/256 的第一次 decode CPU 都从 `7,408` 降到 `6,768` bytes。256-token 第一次 decode 的 GPU 字节仍是 `2,018,180`，分配次数仍是 `2`。32-token prefix 仍与 HF exact，128/256 仍在 step 70 分叉，margin `0.02797699`。Chapter 134 于 2026-10-10 Closed。Episode 10 保持 Open。最终 logits 没有升成 Float32，也不声称整条 decode 零分配。

[`Chapter 135 — Qwen3-VL 复用 scratch 切片`](notes/episodes/episode10_qwen3_vl_efficient_generation/chapter135_qwen3_vl_slice_reuse.md) 让后来的请求复用第一次 decode 的 56 张 scratch 切片，只改指针和形状。2026-10-10 一次 repeat 的重测 SHA256 为 `526066a3cdf57673e5384c36a625c1acd2861a451a5dd47a24c1c5f9af48640b`。GPU 最后一步仍是 `0`。position `107 / 203 / 331` 的 CPU 仍是 `0` bytes。32/128/256 的第一次 decode CPU 都从 `6,768` 降到 `3,264` bytes。256-token 第一次 decode 的 GPU 字节仍是 `2,018,180`，分配次数仍是 `2`。32-token prefix 仍与 HF exact，128/256 仍在 step 70 分叉，margin `0.02797699`。Chapter 135 于 2026-10-10 Closed。Episode 10 保持 Open。最终 logits 没有升成 Float32，也不声称整条 decode 零分配。

[`Chapter 136 — Qwen3-VL 复用请求直接安装 Ready`](notes/episodes/episode10_qwen3_vl_efficient_generation/chapter136_qwen3_vl_shell_ready.md) 让后来的请求把复用的 scratch 直接装进具体类型的 Ready，不再装进 `Any` 字段。2026-10-10 一次 repeat 的重测 SHA256 为 `1cad8aca2a56b0149cd8533d45bb3d3230e1b16529216408bf1c8caa408454bf`。GPU 最后一步仍是 `0`。position `107 / 203 / 331` 的 CPU 仍是 `0` bytes。32/128/256 的第一次 decode CPU 都从 `3,264` 降到 `1,312` bytes。256-token 第一次 decode 的 GPU 字节仍是 `2,018,180`，分配次数仍是 `2`。32-token prefix 仍与 HF exact，128/256 仍在 step 70 分叉，margin `0.02797699`。Chapter 136 于 2026-10-10 Closed。Episode 10 保持 Open。最终 logits 没有升成 Float32，也不声称整条 decode 零分配。

[`Chapter 137 — Qwen3-VL 复用 Ready`](notes/episodes/episode10_qwen3_vl_efficient_generation/chapter137_qwen3_vl_ready_reuse.md) 让后来的请求复用同一个 Ready，只换新容量的 attention scratch。2026-10-10 一次 repeat 的重测 SHA256 为 `72409ccf67d9ceb0848d2a1de2fc82e7aa0a2c654ac81a66d336faa3a19e1365`。GPU 最后一步仍是 `0`。position `107 / 203 / 331` 的 CPU 仍是 `0` bytes。32/128/256 的第一次 decode CPU 都从 `1,312` 降到 `800` bytes。256-token 第一次 decode 的 GPU 字节仍是 `2,018,180`，分配次数仍是 `2`。32-token prefix 仍与 HF exact，128/256 仍在 step 70 分叉，margin `0.02797699`。Chapter 137 于 2026-10-10 Closed。Episode 10 保持 Open。最终 logits 没有升成 Float32，也不声称整条 decode 零分配。

[`Chapter 138 — Qwen3-VL 复用父缓冲`](notes/episodes/episode10_qwen3_vl_efficient_generation/chapter138_qwen3_vl_parent_reuse.md) 让后来的请求在已有父缓冲装得下时不再 `similar` 新的两块。2026-10-10 一次 repeat 的重测 SHA256 为 `1111c98b93bca2cd344bb4cc4468898b8d2a77fb9b5dc4ceebb21e48b80a3ccd`。GPU 最后一步仍是 `0`。position `107 / 203 / 331` 的 CPU 仍是 `0` bytes。32/128/256 的第一次 decode CPU 都从 `800` 降到 `352` bytes。这三次的第一次 decode GPU 字节都降到 `0`，分配次数都从 `2` 降到 `0`。32-token prefix 仍与 HF exact，128/256 仍在 step 70 分叉，margin `0.02797699`。Chapter 138 于 2026-10-10 Closed。Episode 10 保持 Open。最终 logits 没有升成 Float32，也不声称整条 decode 零分配。

[`Chapter 139 — Qwen3-VL attention capacity 原地更新`](notes/episodes/episode10_qwen3_vl_efficient_generation/chapter139_qwen3_vl_attention_capacity.md) 让后来的请求原地改 attention scratch 的 `capacity`，不再新建这个对象。2026-10-10 一次 repeat 的重测 SHA256 为 `d3597accee924dba321e9252a6f9411fbd6838152971d5a9c28302ba1a3fca42`。GPU 最后一步仍是 `0`。position `107 / 203 / 331` 的 CPU 仍是 `0` bytes。32/128/256 的第一次 decode CPU 都从 `352` 降到 `192` bytes。第一次 decode 的 GPU 字节仍是 `0`，分配次数仍是 `0`。32-token prefix 仍与 HF exact，128/256 仍在 step 70 分叉，margin `0.02797699`。Chapter 139 于 2026-10-10 Closed。Episode 10 保持 Open。最终 logits 没有升成 Float32，也不声称整条 decode 零分配。

[`Chapter 140 — Qwen3-VL 复用入口不再装箱 spec`](notes/episodes/episode10_qwen3_vl_efficient_generation/chapter140_qwen3_vl_reuse_entry.md) 把复用路径拆出大函数，并在具体函数里从 pack 读取 text spec。2026-10-10 一次 repeat 的重测 SHA256 为 `65ce9f34941b20837d9aeae43badbf13e7bc03366a1ba0c1a26ce8303bda0bf4`。GPU 最后一步仍是 `0`。position `107 / 203 / 331` 的 CPU 仍是 `0` bytes。32/128/256 的第一次 decode CPU 都从 `192` 降到 `48` bytes。第一次 decode 的 GPU 字节仍是 `0`，分配次数仍是 `0`。32-token prefix 仍与 HF exact，128/256 仍在 step 70 分叉，margin `0.02797699`。Chapter 140 于 2026-10-10 Closed。Episode 10 保持 Open。最终 logits 没有升成 Float32，也不声称整条 decode 零分配。

[`Chapter 141 — Qwen3-VL 静态 Ready 调用`](notes/episodes/episode10_qwen3_vl_efficient_generation/chapter141_qwen3_vl_static_ready.md) 用 BF16 Ready 的具体类型把 steady 调用收成静态调用。2026-10-10 一次 repeat 的重测 SHA256 为 `f9a66705ccb388606f478839ac20770733c0c008905ccdcd53f35ee599362831`。GPU 最后一步仍是 `0`。position `107 / 203 / 331` 的 CPU 仍是 `0` bytes。32/128/256 的第一次 decode CPU 都从 `48` 降到 `32` bytes。第一次 decode 的 GPU 字节仍是 `0`，分配次数仍是 `0`。32-token prefix 仍与 HF exact，128/256 仍在 step 70 分叉，margin `0.02797699`。Chapter 141 于 2026-10-10 Closed。Episode 10 保持 Open。最终 logits 没有升成 Float32，也不声称整条 decode 零分配。

[`Chapter 142 — Qwen3-VL 第一次 decode 直接调用 Ready`](notes/episodes/episode10_qwen3_vl_efficient_generation/chapter142_qwen3_vl_ready_call.md) 让第一次 decode 的计时窗口和稳态步一样，直接调用具体 Ready。2026-10-10 一次 repeat 的重测 SHA256 为 `c838c5f037c16a35820eb45aa8fbe7d93139a519ab7fac1f3946e6947053ccc6`。GPU 最后一步仍是 `0`。position `107 / 203 / 331` 的 CPU 仍是 `0` bytes。32/128/256 的第一次 decode CPU 都从 `32` 降到 `0` bytes。第一次 decode 的 GPU 字节仍是 `0`，分配次数仍是 `0`。32-token prefix 仍与 HF exact，128/256 仍在 step 70 分叉，margin `0.02797699`。Chapter 142 于 2026-10-10 Closed。Episode 10 保持 Open。最终 logits 没有升成 Float32，也不声称整条 decode 零分配。

[`Chapter 143 — Qwen3-VL GPU 的 BF16 转 Float32 直接拷贝`](notes/episodes/episode10_qwen3_vl_efficient_generation/chapter143_qwen3_vl_f32_copy.md) 让 GPU 上的 `_bf16a_f32` 用 `copyto!` 写出和 `1.0f0 .* x` 相同的值。CPU 路径仍是原来的广播。2026-10-10 一次 repeat 的重测 SHA256 为 `f132e3624ccb19b175babe19eafd447c3300356e738edeada965fc11a6a4f0ad`。GPU 最后一步仍是 `0`。position `107 / 203 / 331` 的 CPU 仍是 `0` bytes。32/128/256 的第一次 decode CPU 仍是 `0` bytes。256 的 prefill CPU 从 `8,007,336` 降到 `6,918,616` bytes，prefill GPU 仍是 `858,377,949` bytes、`4240` 次。32-token prefix 仍与 HF exact，128/256 仍在 step 70 分叉，margin `0.02797699`。Chapter 143 于 2026-10-10 Closed。Episode 10 保持 Open。最终 logits 没有升成 Float32，也不声称整条 decode 零分配。

[`Chapter 144 — Qwen3-VL GPU 的 BF16 转 Float32 复用 kernel`](notes/episodes/episode10_qwen3_vl_efficient_generation/chapter144_qwen3_vl_promote_kernel.md) 让连续的 GPU 数组复用同一个转换 kernel，按 `1.0f0 * Float32(x[i])` 写入。不连续的数组仍走 `copyto!`。CPU 路径仍是原来的广播。2026-10-10 一次 repeat 的重测 SHA256 为 `25f498529f6e84e7a8e2902b5e02c79e5fb62519a1472e5e015d8b0fcb9ab0a7`。GPU 最后一步仍是 `0`。position `107 / 203 / 331` 的 CPU 仍是 `0` bytes。32/128/256 的第一次 decode CPU 仍是 `0` bytes。256 的 prefill CPU 从 `6,918,616` 降到 `5,657,880` bytes，prefill GPU 仍是 `858,377,949` bytes、`4240` 次。32-token prefix 仍与 HF exact，128/256 仍在 step 70 分叉，margin `0.02797699`。Chapter 144 于 2026-10-10 Closed。Episode 10 保持 Open。最终 logits 没有升成 Float32，也不声称整条 decode 零分配。

[`Chapter 145 — Qwen3-VL 静态 decode 的完整参数绑定`](notes/episodes/episode10_qwen3_vl_efficient_generation/chapter145_qwen3_vl_parameter_binding.md) 修复共享 embedding 时替换 blocks、final_norm 或 spec 仍沿用旧 pack/Ready 的问题。参数绑定变化会重建派生工作区，相同参数继续保留 Ready/scratch。CPU BF16 的 300 项新增回归与 RTX 4090 D 的 456 项 CUDA 参数绑定回归通过，覆盖 reset/re-prefill、两步 logits 与各层有效 K/V 对照、无效输入不修改状态。真实 2B 的 32/128/256-token 剖面保持稳态 CPU/GPU 零分配；首次 decode 为 10,816 CPU bytes。32-token 与 oracle exact，长生成仍在原来的 step 70 分叉。随后 Chapter 146 继续修复跨 cache 工作区共享。

[`Chapter 146 — Qwen3-VL CUDA 工作区的请求所有权`](notes/episodes/episode10_qwen3_vl_efficient_generation/chapter146_qwen3_vl_workspace_ownership.md) 修复不同 cache 共享可变 Ready/scratch 的问题。26 项 GPU 生命周期回归和 456 项 CUDA 参数绑定回归通过；真实 2B 的 32/128/256-token 轨迹及逐步 logits 哈希与修复前一致。新 cache 的首次 decode 分配两块独立父缓冲，分别为 `1,043,332 / 1,461,124 / 2,018,180` GPU bytes，后续每步仍为 CPU/GPU 零分配。同一 cache 的 reset 继续复用原存储。

[`Chapter 147 — Qwen3-VL 显式复用高层生成 cache`](notes/episodes/episode10_qwen3_vl_efficient_generation/chapter147_qwen3_vl_generation_reuse.md) 支持在 token 与原始图文接口中传入 `cache=existing_cache`。完整输入准备通过后才 reset，失败保留旧请求。421 项 CPU 回归通过；真实 RTX 4090 D 上 image → image → text → image 的 token、逐步 logits 和有效 K/V 与 fresh 对照一致，Ready、工作区和 K/V 存储身份保留，独立 cache 地址不重叠。

[`Chapter 42 — 显式环境事件记忆写回与因果验收`](notes/episodes/episode08_environment_action_loop/chapter42_environment_event_memory.md) 在 ModelScope Qwen3-8B / RTX 4090 D 上让 8 个 full-feedback writer 以 BFS 最短路 `8/8` clean success，显式写出 8 条事件；fresh-load 后 CPU Qwen3-Embedding raw top-1 / exact-spec 都是 `8/8`。三个无中间反馈 reader 臂为 no-memory `0/8`、retrieved `7/8`、等 token 且机械 non-solving 的 mirrored distractor `0/8`，相关记忆相对两组对照的双侧精确 McNemar p 均为 `0.015625`。`grid/08` 是正确检索后的执行失败，不是 retrieval miss；32 条轨迹已在不加载两套模型且不追加 journal 的第二进程完整 replay。环境事件记忆条目已关闭；Episode 08 仍保留跨 adapter timeout / execution failure / e-stop / idempotent action safety 缺口。

[`Chapter 41 — Qwen3 MoE grouped scattered expert-cache dispatch`](notes/episodes/episode06_qwen3_moe_and_model_expansion/chapter41_qwen3_moe_grouped_scattered.md) 在官方 30B 同进程 2-token 与 32-token 三路对照中，让 grouped WMMA 直接读取 generation-safe expert pointer tables；两条 grouped 路径的 prefill/decode logits、逐层 routes 与每次 I/O/cache traffic 全部 exact，并消除 `10.560 / 16.930 GB` logical active concat。它选择的下一架构 Qwen3-VL 已由 Chapter 43 启动并关闭 vision-only 第一阶段。

[`Chapter 40 — 确定性 GridWorld observation/action 与真实反馈闭环`](notes/episodes/episode08_environment_action_loop/chapter40_deterministic_gridworld.md) 首次用环境终态而非模型自述判定任务成功：冻结 8 题的 BFS oracle 与 Qwen3-4B full-feedback 都是 `8/8`、44 个动作、0 个非法动作，模型逐题走出最短路；首轮 prompt 完全相同的 feedback-withheld 臂为 `0/8`、67 个非法/失败动作，配对 McNemar p=`0.0078125`。第二进程重算 148 / 148 prompts、140 / 140 tool outcomes、132 / 132 transitions 与 16 / 16 final states。

[`Chapter 39 — 跨请求、可回放的语义记忆闭环`](notes/episodes/episode07_agent_closed_loop/chapter39_persistent_semantic_memory.md) 已把版本化 append-only journal、fresh-load 严格恢复、exact semantic index 和 retrieval context 接入真实 Qwen3：冻结 12 个 private-fact tasks 上 retrieval top-1 / recall@1 均为 `12/12`，no-memory / retrieved / 等 token 干扰三臂为 `0/12 / 12/12 / 0/12`，两组精确 McNemar p 均为 `0.00048828125`；第二进程 36 / 36 条完整 prompt/context 轨迹 replay 一致。默认专项为 `100 / 100`，加真实 Qwen3-4B tokenizer 为 `112 / 112`。

[`Chapter 38 — 工具到底帮不帮得上忙`](notes/episodes/episode07_agent_closed_loop/chapter38_qwen3_tool_task_success.md) 在同一批 150 道 GSM8K 上做了一次配对对照：无工具基线 `141/150 = .940`，声明计算器后 `137/150 = .913`，再加一句要求使用工具的 system message 仍是 `137/150`。**掉的题几乎全部没有调用工具**（`tool-declared` 掉 6 题、调用过工具的 0 题），所以测到的是「在 prompt 里声明工具」这件事本身对 greedy 解码的扰动，而不是工具的效用；两次下降的精确 McNemar p 为 `0.29` / `0.39`，150 题分辨不了 `2.67` 个点。

[`Chapter 37 — Qwen3 dense 任务质量基线`](notes/episodes/episode07_agent_closed_loop/chapter37_qwen3_task_quality.md) 第一次回答「复现出来的模型答对率是多少」：冻结 MMLU 200 题 + GSM8K 150 题，0.6B/1.7B/4B 的 MMLU loglikelihood 为 `.365 / .395 / .555`，4B 的 GSM8K 为 `141/150 = .940`。同时给出一条本章最重要的负结果——**MMLU loglikelihood 的逐题决策在 BF16 下不可复现**，既不跨 dtype，也不跨设备，甚至不跨同一实现里两条数学等价的写法：协议对齐后与 HuggingFace fp32 参照的一致率为 `193/200 = 96.5%`；同一份权重同一协议，CUDA 给 `73`、CPU 给 `74`、fp32 给 `70`；我们自己 fast 与 general 两条路径之间也有 `6/200` 翻转。accuracy 的极差 2 个点仍远小于 Wilson 区间宽度，标题数字可用，但逐题一致的主张不成立。

[`Chapter 36 — Qwen3 tools chat template HF parity 与 4B 工具调用闭环`](notes/episodes/episode07_agent_closed_loop/chapter36_qwen3_tools_chat_template.md) 用官方 Jinja 模板经 CPython 渲染的外部 reference，把 chat template 的全部分支做到逐字节 parity（30 / 30 case，真实 4B tokenizer 下 token id 亦逐位相同），并修复了 Chapter 08 遗留的两条真实分歧：历史 assistant 的 `<think>` 块未剥离、以及无 user 消息时 `last_query_index` 方向相反。开启 `--thinking` 的多轮聊天从第 2 轮起会命中前者。

[`Chapter 35 — Qwen3 MoE bounded host projection-buffer reuse`](notes/episodes/episode06_qwen3_moe_and_model_expansion/chapter35_qwen3_moe_host_buffer_reuse.md) 在明确 CUDA pageable、pinned 与 CPU identity 的不同 ownership 后，以 72 MiB final-matrix pool 配合 Chapter 34 的 24 MiB raw pool。30B-A3B English32 同进程 A/B 的 cold/revisit Julia allocation 从约 `28.967 GB` 降到 `0.288 GB`（`99.01%`），时延分别加速 `1.315× / 1.296×`，所有输出 exact；CPU 与 pinned async 自动保持无 final pool。

目前已经具备：

- 手写与批量 scaled dot-product attention、因果遮罩和 Multi-Head Attention。
- 支持 legacy interleaved 与 HF rotate_half 配对的 RoPE、learned absolute position、pre-norm TransformerBlock 和 decoder-only GPTModel。
- 可独立切换的 LayerNorm / RMSNorm、GELU / GELU-New / SwiGLU、projection bias / LM-head bias、untied / tied embedding-output projection；legacy 默认保持不变。
- 字符级 Tokenizer、DatasetLoader、next-token loss 和训练循环。
- 无泄漏 train / validation 划分、采用补偿求和的 token-weighted evaluation / perplexity，
  训练/评估指标无法有限表示为 Float32 时显式报错，以及带有限 Float32
  超参数边界的 global gradient norm clipping。
- checkpoint v2、设备无关保存/加载、确定性断点续训和 v1 legacy checkpoint 迁移。
- 基于 Zygote 的常规训练，以及 Reactant/Enzyme 驱动的 XLA 训练路径。
- greedy、temperature、top-k、top-p 文本生成；Qwen3 可严格读取官方 generation config，并支持固定 uniform 流的可重放采样。
- 动态 KV Cache 的 prefill / decode，以及面向 XLA 的固定形状 KV Cache 和编译后增量解码。
- Qwen3 BF16 常驻 session：固定容量 KV、bounded chunked prefill、
  last-token-only vocabulary projection、EOS/采样、chat history token
  预算与最老 turn-pair 裁剪；4K/4090 D profile 已完成真实权重验收。
- Qwen3-8B BF16 XLA single-residency session：safetensors 逐层直接组装
  packed compact tree，只做一次递归 device transfer；固定形状 64-token
  prefill 与单 token decode 共用 4K 静态 KV。4090 D 日常 CLI 当前明确为
  batch-1 greedy，sampling 仍走 CUDA eager 路径。
- Qwen3-8B XLA loopback resident service：严格的 `/healthz` 与
  Ollama-compatible `/api/generate` 子集支持 buffered/NDJSON、UTF-8
  安全 streaming、context/options fail-closed 和 single-flight；轻量
  client 只读取 tokenizer，权重与 compiled executable 常驻 server。
- full / dynamic / static KV Cache correctness matrix，以及 CPU、CUDA GPU、XLA CPU、XLA GPU 四后端 benchmark。
- 原始 Qwen3 MoE 的 Float32 top-k router、逐 expert SwiGLU、full/dynamic/static cache，以及 Float32/native BF16 的路由后 active-expert 流式读取。官方 30B-A3B 的 48 层 prompt/cache-decode parity 已冻结：Float32 top-8 集合 `1,152 / 1,152` 一致、BF16 路由槽位重合 `95.92%`，两种模式最终 argmax 均一致。`HFQwen3MoEOffloadSession` 让 attention/router/norm/LM head 与 40K KV 常驻 GPU，expert 在路由后按层读取并用局部 id 交给 scalar 或 grouped WMMA；最坏工作集硬下限 `7.166 GiB`。device cache 可选 global/layer-balanced LRU，按实际 expert-layer bytes 预算，支持跨请求保留、显式清空和运行时重配；scalar 与 grouped CUDA 都可用 scattered pointer-table dispatch 消除 active tensor 拼接，有界复用 pointer/workspace state，并配置 forced-GC cadence。cache miss 可显式选择 bounded parallel host reads 与 CUDA pinned upload；32-token trace 的 reader 默认上限为 8。严格相邻 batch read 已实现，但真实 workload 证明 coalescing 更慢，默认仍逐 tensor 读取；同 dtype多维 decode、raw payload 与 pageable final host-matrix allocation 已依次消除。
- Qwen3-VL-2B-Instruct 的双 registry immutable revision、13-file checksum、
  strict text/vision config 与完整 625-tensor BF16 shape/payload contract；raw
  PNG/JPEG/RGB decode、官方 CPU UInt8 bicubic、normalize、grid/token count 与
  merge-order patchify；content-list chat、VL tokenizer profile、placeholder
  expansion 和 T/H/W mRoPE；3D patch embedding、position interpolation、vision
  RoPE、24-layer per-temporal-frame packed attention、三个 DeepStack mergers、
  main merger，以及 main visual replacement/DeepStack injection 后的 28-layer
  decoder prefill；请求级 dynamic KV baseline、bounded preallocated static KV、
  cached prefill、单 token decode、reset/reuse 和 raw single-image greedy generation
  已接通。Float32 RTX 4090 D 已完成
  Transformers 4.57.0 的 prefill/decode/token/text strict gate；BF16 static/dynamic
  decode 只完成 same-device bitwise smoke，并单独记录相对 Float32 oracle 的跨 dtype
  边界，不声称 HF BF16 strict parity。
- 六个官方 Qwen3 dense 规格的 immutable revision/config checksum、自动识别、显式 variant 校验、精确参数量；严格的 BF16/F32 safetensors 单文件/分片读取、HF 参数映射与显式 0-based token-id 边界转换。
- Qwen3-Embedding-0.6B 的独立 immutable revision/8-asset checksum、
  151,669-vocabulary/32K contract、embedding tokenizer 尾 token 与
  base-model safetensors namespace；BF16 hidden-state 前向不分配
  vocabulary logits 或保留 KV cache。
- instruction-aware query、变长批 attention mask、last-token pooling、
  L2 normalization 与 1024/512/256/128/64 维 MRL；内存内 dense exact
  cosine semantic memory 可关联 metadata，并有真实 notes 检索示例。
- 版本化 append-only agent memory journal：保存 source text 与 string metadata，
  fresh load 严格校验 schema/sequence/ID/checksum，启动时重建 exact embedding
  index；retrieval query、命中 ID/score、注入内容与 prompt 均带 sha256，可离线 replay。
  冻结私有事实三臂实测为 `0/12 / 12/12 / 0/12`，相关记忆同时胜过无记忆与等 token 干扰；
  clean successful GridWorld episode 还可经显式 policy 写为 canonical event，同 run ID 幂等，
  authoritative spec/source binding 和生产 exact-spec context 均 fail closed。
- Qwen3 weight-only INT8 per-channel / packed groupwise INT4，以及统一的
  `QuantizationPlan`：可按 one-based layer、projection 与独立 LM head
  选择 INT4/INT8/BF16，streamed/in-memory 路径共用策略；真实树统计与
  topology 估算逐 byte 对齐。`ActivationCalibration` 可用独立 token
  逐层采集每个线性输入的 second moment，并以 `:activation_mse` 做
  diagonal activation-aware clipping；这不是完整 AWQ/GPTQ。
- Qwen3-0.6B 逐层 hidden states、full logits、dynamic/static cache decode 的真实 HF reference parity；真实权重测试显式 opt-in，默认测试保持离线。
- Qwen3-0.6B 的 16-step 官方 sampled reference parity、position 40,959 独立 Transformers RoPE fixture，以及 CPU/CUDA/Reactant-XLA GPU 真实推理 benchmark。
- 严格的 Qwen3 HF tokenizer：NFC、目标 regex、ByteLevel、imported BPE、added/special tokens、artifact/checkpoint 与 provenance fingerprint。
- 官方 Qwen3 chat template 的全部分支：`# Tools` 头、assistant `tool_calls`、`tool`
  角色的 `<tool_response>` 合并、think 块拆分与 `last_query_index` 反向扫描，30 / 30
  case 与 CPython + Jinja2 渲染的外部 reference 逐字节相同；`tojson` 按 CPython
  `json.dumps(ensure_ascii=False)` 语义复刻，含 CPython `repr(float)` 与任意精度整数，
  无序 `Dict` 与会窄化数字的 `JSON3.Object` fail closed。tools / tool 角色 /
  `tool_calls` 以官方模板 sha256 为准入条件。
- 不使用 `eval` 的算术求值器与 calculator 工具，以及同批题 A/B 的配对统计（2×2 不一致对 +
  精确 McNemar）；准确率相同的两个臂仍可能有 8/150 题在两个方向上互换，只有配对表能看见。
- 任务质量评测基线：冻结的 MMLU/GSM8K 子集（含许可、上游 row_idx 与 sha256 provenance）、
  loglikelihood 与 generative 两种协议、机械且写死的答案抽取规则，以及强制同时报出
  Wilson 区间、未解析数、截断数与格式合规数的报告口径。
- 最小工具闭环：工具声明与注册、`<tool_call>` 解析与合法性判定、沙箱化内置工具，以及
  「观察 → 决策 → 调用 → 回填 → 再决策」的多 step 循环；每一步记录 prompt sha256 与
  generated ids，可在不加载模型的情况下离线 replay。
- 确定性环境闭环：通用 observation/action/transition 类型、hidden-wall GridWorld、唯一 allowlisted
  `move(direction)` 副作用面、动作预算和环境终态评分；Qwen3-4B full-feedback 8 / 8 最短路完成，
  feedback-withheld 0 / 8，模型 prompt、工具结果和环境状态可联合 replay。Qwen3-8B 的 8 条 clean
  writer 事件在 fresh-load 后 raw/exact-spec `8/8` 命中；无反馈 reader 的 no/retrieved/等 token 干扰
  为 `0/8 / 7/8 / 0/8`，32 条 writer/reader 轨迹可 tokenizer-only replay。
- full/dynamic/static/XLA 的真实 greedy text-generation parity。
- 严格的 GPT-2 config、Float32 safetensors、Conv1D/fused-QKV 映射与 GPT-2 byte-level BPE adapter；冻结 revision/checksum 不匹配时 fail closed。
- GPT-2 124M 的 10 组 tokenizer corpus、embedding、12 层 residual、final hidden、full logits 与 full/dynamic/static 8-step greedy text 均通过 Transformers reference parity。
- GPT-2 124M 的 16/64/256-token CPU/CUDA correctness 与 steady-state benchmark，以及 learned-position/GELU-New XLA 同构 smoke。
- 围绕 Attention、RoPE、prefill/decode 和 KV Cache 的 Pluto 可视化学习笔记。
- 默认测试套件全部通过；Reactant/XLA 专项测试需显式启用。

尚未具备：

- GPT-2 的 WebText 从零训练、论文 zero-shot quality、Medium/Large/XL、cross-attention 与分类 head 未复现；当前完成的是 124M 官方 checkpoint 的 Float32 推理/架构复现。
- Qwen3 MoE 官方 30B-A3B checkpoint 的 40K cache 容量、短/32-token GPU offload、跨请求 device LRU、中英文自然文本 cache-budget sweep、scalar/grouped CUDA scattered cache 与 500-request reuse 生命周期已实跑，但 full-window 40K prefill、长序列生成质量、layer-ahead 异步预取、grouped workspace byte cap、完整 AWQ/GPTQ、
  activation/KV-cache 量化与量化 GEMM；30B 的 48 层 prompt/cache-decode 真实 parity 与 dense 0.6B—32B 真实权重
  parity、native BF16、weight-only INT8/INT4 与 diagonal
  activation-aware clipping 已完成，不能再沿用早期“只验证 0.6B”的边界。
- 通用 Jinja 渲染器：Chapter 36 的实现是按官方模板逐分支手写的 Julia 渲染，并以模板 sha256 作为准入条件，不是通用模板引擎。
- 更完整的质量评估：Chapter 37 只覆盖 0.6B/1.7B/4B 与 8 个 MMLU subject，且全部是 0-shot、
  greedy 单次；8B 及以上、5-shot、多 seed 方差与更大 token 预算下的 generative 上界都未做。
  本章数字是本仓库自己的基线，不能与官方榜单并列。
- ANN/reranker、通用 redaction/摘要/遗忘、并发 writer、规划与反思；Chapter 42 仍是 single-writer
  append-only journal，同 run ID 幂等但不同 run 可重复写同一事实，只接纳 clean successful GridWorld
  episode，不是完整长期记忆系统。
- Qwen3-VL 当前已从 raw PNG/JPEG/RGB image 接到 content-list chat、vision、
  decoder prefill、dynamic KV 与 bounded static KV、增量 decode、可 reset request
  state 和 greedy image-to-text generation；
  尚无 sampled generation 或 video。真实权重验收仍只覆盖单图、batch 1、
  sequence 76 和全一 attention mask。256-token HF Float32/CPU oracle 已冻结，但
  RTX 4090 D 的 BF16 static warmup 在 generated step 79 分叉；同前缀的 Float32
  CUDA 路径 79/79 与 HF argmax 一致，所以这是 BF16 舍入边界，不是 cache 几何错误。
  长生成 allocation profile 因此尚未完成。padding/multi-image、长上下文、整条 decode
  执行图的零分配/静态编译、音频与其他传感器输入也未实现。
- 连续控制、动力学、外部 simulator 与实体机器人 adapter；Chapter 40/42 只有离散确定性 GridWorld，
  生产记忆强制 exact-spec，cross-spec 只用于 diagnostic control。timeout、execution failure、e-stop 与
  idempotent action safety 尚未定义，不等于具备真实设备控制或物理安全。
- 在线或持续学习机制。

更详细的能力盘点、验证范围与建议里程碑见 [`notes/current_status.md`](notes/current_status.md)。

大模型权重与真实 reference 存放在仓库外的持久模型目录，不使用易清理的
`/tmp`，也不提交进 Git；历史资产机器使用 `/home/yj/models/`，当前验证机
使用 `/home/ubuntu/models/modelscope/`。Qwen3 六尺寸（0.6B—32B）、
Qwen3-Embedding-0.6B、Qwen3-VL-2B-Instruct 与 GPT-2 124M 的完整资产和恢复命令见
[`notes/local_model_assets.md`](notes/local_model_assets.md)，六个 Qwen3
dense config 规格见 [`Chapter 11`](notes/episodes/episode03_model_family_and_large_weights/chapter11_qwen3_dense_family.md)，
Qwen3-VL vision-only 边界见 [`Chapter 43`](notes/episodes/episode09_qwen3_vl_multimodal_perception/chapter43_qwen3_vl_vision_architecture.md)，raw processor/chat 与 decoder prefill 边界见 [`Chapter 44`](notes/episodes/episode09_qwen3_vl_multimodal_perception/chapter44_qwen3_vl_multimodal_prefill.md)，dynamic cache/decode 与完整单图 greedy 边界见 [`Chapter 45`](notes/episodes/episode09_qwen3_vl_multimodal_perception/chapter45_qwen3_vl_dynamic_decode.md)，bounded static K/V 与 request-state reuse 边界见 [`Chapter 46`](notes/episodes/episode10_qwen3_vl_efficient_generation/chapter46_qwen3_vl_static_cache.md)，长生成 allocation profile 协议与当前 Open 边界见 [`Chapter 47`](notes/episodes/episode10_qwen3_vl_efficient_generation/chapter47_qwen3_vl_long_generation_profile.md)。

## 演进路线

```text
模型基本组件
    ↓
可训练、可生成、可评估的模型闭环
    ↓
记忆 + 规划 + 工具 + 多模态的智能体核心
    ↓
感知 → 决策 → 行动 → 反馈的具身闭环
    ↓
能长期互动、持续适应的“有生命感”AGI
```

模型组件不会在进入下一阶段后停止建设；它会始终作为底层主线，与智能体和具身实验互相驱动。

## 记录节奏

- **Chapter**：目标驱动的逻辑研发章节，不与自然周绑定。每章先定义核心问题、交付物和 Close 条件；条件满足后立即复盘关闭。
- **Episode**：围绕一条完整能力主线组织多个 Chapter，不与自然月绑定；主题闭环后总结验证证据、关键学习、失败尝试、架构决策和下一卷重点。
- **状态快照**：每次 Chapter Close 后按需更新，只描述仓库此刻真实具备的能力，明确区分“已实现”“已验证”和“尚未开始”。

```text
Open Chapter → 执行与验证 → 满足 Close 条件 → 复盘并更新状态 → Open Next Chapter
```

记录索引与模板见 [`notes/README.md`](notes/README.md)。

## 仓库结构

```text
src/
├── LifeAI.jl      # 按依赖顺序组装实现
├── api.jl         # 按职责分组的公共 API 清单
├── core/          # Attention、RoPE、Transformer、sampling
├── data/          # Tokenizer 与 DatasetLoader
├── io/            # HuggingFace config / safetensors 权重加载
├── models/        # GPT 模型
├── train/         # Zygote / Reactant-XLA 训练
├── generation/    # 共享采样/输入边界、文本生成与 KV Cache
├── agent/         # 工具、持久记忆、环境/action loop
└── eval/          # 任务质量、记忆与环境闭环评测

test/
├── runtests.jl    # 按 Episode / Chapter 分组的统一入口
├── episodes/      # 与 notes/episodes 一一对应的测试与 fixture
└── support/       # 跨 Chapter 复用的测试构造器与资产定位工具
examples/          # 最小训练和生成示例
notebook/          # 可交互的原理与实验记录
notes/
├── README.md      # Episode / Chapter 总目录
├── episodes/      # 按主题卷组织的 Chapter 正文
└── templates/     # Episode / Chapter 模板
```

## 开始使用

安装依赖：

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

运行默认测试：

```bash
julia --project=. -e 'using Pkg; Pkg.test()'
```

在具备对应 XLA 后端的环境中运行专项测试：

```bash
LIFEAI_TEST_XLA=true julia --project=. -e 'using Pkg; Pkg.test()'
```

运行字符级 GPT 训练与生成示例（默认使用 Reactant/XLA GPU）：

```bash
julia --project=. examples/minigpt.jl
```

运行 RMSNorm + SwiGLU + tied embedding 的可恢复训练示例：

```bash
julia --project=. examples/modern_gpt.jl
```

也可以直接构建可独立开关的 modern 配置：

```julia
model = GPTModel(
    vocab_size,
    d_model,
    num_heads,
    num_layers;
    norm_type=:rmsnorm,
    mlp_type=:swiglu,
    tie_embeddings=true,
)
```

RTX 4090 D 上启动 Chapter 20 的 Qwen3-8B BF16 XLA 日常聊天：

```bash
julia --project=. --startup-file=no \
  scripts/run_qwen3_xla_chat.jl \
  /home/ubuntu/models/modelscope/Qwen/Qwen3-8B
```

默认 profile 是 4,096-token 总 context（最多 3,584-token
prompt/history + 512-token output）、batch 1、non-thinking、greedy；
XLA allocator 默认 fraction 为 `0.87` 且不预分配。首次进程仍需约
1.5 分钟编译 prefill/decode executable；Reactant 0.2.275 当前只持久化
autotune cache，不持久化完整 kernel executable。

需要跨请求复用模型时，可启动常驻 HTTP 服务：

```bash
scripts/launch_qwen3_xla_server.sh /path/to/Qwen3-8B
```

启动器默认保持 Julia 的 CPU 编译选项。Linux/x86-64 上，Julia 1.12.6 /
LLVM 18 的 AVX-512-BF16 批量转换可能在 CPU RoPE 表转换阶段报
`Cannot select ... v16bf16 ... insert_subvector`。遇到该错误时，可显式启用：

```bash
scripts/launch_qwen3_xla_server.sh /path/to/Qwen3-8B --cpu-bf16-compat
```

该开关在 Julia 启动前设置 `--cpu-target=native,-avx512bf16`，只调整
CPU 指令生成；模型权重、GPU 计算和 XLA 缓存仍使用 BF16。
`LIFEAI_JULIA_BIN=/path/to/julia` 可选择 Julia 可执行文件。也可直接把
上述 CPU target 参数加到既有 `julia` 启动命令中。

服务会即时输出 host load、参数传输、runtime allocation、prefill compile、
decode compile 的开始及完成耗时。`--quiet-startup` 可关闭这些详细阶段日志，
保留原有启动摘要。库函数默认不输出阶段日志；调用
`load_hf_qwen3_bf16_xla_session` 或 `load_qwen3_xla_http_service` 时，可传入
`on_progress=event -> ...`，接收 `(; stage, status, seconds)`。开始事件的
`seconds` 为 `nothing`，完成事件为阶段耗时；回调抛错会终止加载。

在需要固定 Julia CUDA.jl toolkit 的环境，可将
`configs/deployment/LocalPreferences.cuda12_9.example.toml` 中的配置合并到
仓库根目录的 `LocalPreferences.toml`，再启动新的 Julia 进程。
示例选择 12.9，但不会改变其他用户的默认配置；实际版本应与驱动和可用
artifacts 匹配。该偏好不设置 Reactant_jll 自带 XLA 的 CUDA toolkit。
`LocalPreferences.toml` 是忽略的机器本地文件，请保留已有的其他偏好。

需要 temperature/top-k/top-p sampling 时保留 Chapter 19 CUDA eager 入口：

```bash
julia --project=. --startup-file=no \
  scripts/run_qwen3_cuda_chat.jl \
  /home/ubuntu/models/modelscope/Qwen/Qwen3-8B
```

14B mixed RTN 仍是当前已实证的尺寸上限，但 0.377 tok/s 和不足 1 GiB
的短输入余量不适合作为日常默认。

运行 Chapter 36 的工具调用闭环（RTX 5080 16 GiB 上用 Qwen3-4B 验收）：

```bash
julia --project=. --startup-file=no scripts/run_qwen3_tool_loop.jl \
  /path/to/Qwen3-4B/<revision> \
  --variant qwen3_4b --revision <revision> \
  --out benchmark_results/chapter36/tool_loop_trace_run1.jsonl \
  --summary benchmark_results/chapter36/tool_loop_summary_run1.json
```

默认任务集是冻结的 20 题（8 题算术、6 题目录列举、6 题文件读取），文件系统工具被限制在
`--sandbox` 指定的根内。每个 model turn 写一行 JSONL：prompt sha256、prompt token 数、
generated ids、原始 completion、合法性判定、工具调用与返回、prefill/decode 秒数。
`greedy` 下两次独立进程运行的全部 turn 逐位相同。

也可以直接组装工具并渲染官方 tools prompt：

```julia
registry = default_agent_tools(pwd())
prompt = apply_qwen3_chat_template(
    tokenizer,
    [(role="user", content="1+2?")];
    tools=qwen3_tool_specs(registry),
)
calls = parse_qwen3_tool_calls(completion)
result = invoke_agent_tool(registry, first(calls.calls))
```

`tools`、`tool` 角色与 `tool_calls` 要求 tokenizer 携带官方 Qwen3 `chat_template`
（sha256 `a55ee1b1…74d8`，六个 dense 尺寸一致），其他模板一律 fail closed。

查看或严格选择 Qwen3 dense family member：

```julia
for spec in qwen3_dense_specs()
    println(spec.model_id, ": ", qwen3_dense_parameter_count(spec))
end

bundle = load_hf_qwen3_bundle(
    "/path/to/Qwen3-4B";
    variant=:qwen3_4b,
    revision=qwen3_dense_spec(:qwen3_4b).revision,
    max_seq_len=256,
)
```

`variant` 会校验所有架构 shape，但不会下载文件。0.6B—32B 六个 dense
尺寸均已完成真实权重逐层 parity（8B—32B 使用流式验证）；完整 tokenizer
到 text-generation reference 仍以 0.6B 为主，不能把两种验证口径混写。

运行 Qwen3-Embedding-0.6B 的最小 notes 语义检索：

```bash
julia --threads=8 --project=. --startup-file=no \
  examples/qwen3_embedding_memory.jl \
  /home/ubuntu/models/modelscope/Qwen/Qwen3-Embedding-0.6B
```

也可以直接构建和查询内存索引：

```julia
bundle = load_hf_qwen3_embedding_bundle(
    "/path/to/Qwen3-Embedding-0.6B";
    max_seq_len=256,
)
memory = build_qwen3_semantic_memory(
    bundle,
    ["KV cache 避免重复计算历史 token。", "MRL 支持截取 embedding 前若干维。"];
    dimension=512,
)
hits = search_qwen3_semantic_memory(
    bundle,
    memory,
    "如何减少增量解码的重复计算？";
    top_k=1,
)
```

这是 dense exact、进程内 baseline；不包含持久化、ANN、reranker 或 agent
memory policy。Python oracle 只使用仓库 `.venv`，固定依赖见
`requirements/week22-reference.txt`。

加载冻结的 GPT-2 124M 并执行 greedy generation：

```julia
bundle = load_hf_gpt2_bundle(
    "/home/yj/models/huggingface/openai-community/gpt2/607a30d783dfa663caf39e06633721c8d4cfcd7e";
    revision="607a30d783dfa663caf39e06633721c8d4cfcd7e",
    max_seq_len=256,
)
result = generate_hf_text(
    bundle,
    "The meaning of life is";
    strategy=:greedy,
    cache=:dynamic,
    max_new_tokens=8,
)
```

没有 NVIDIA GPU 时可以尝试 XLA CPU 后端：

```bash
LIFEAI_XLA_BACKEND=cpu julia --project=. examples/minigpt.jl
```

对比 CPU、CUDA GPU、XLA+CPU 和 XLA+GPU 的训练与 KV Cache 推理性能：

```bash
./scripts/benchmark_week03.sh
```

脚本会分别记录首编译/首次执行、post-compile warm-up 和 steady-state 指标，并保留逐 iteration 原始耗时，生成 TSV 原始数据与 Markdown 汇总。默认使用 3 个 warm-up step 和 30 个正式样本，避免一次性 runtime settling 扭曲 p90。配置项和指标口径见 [`Chapter 03 四后端性能对比`](notes/episodes/episode01_transformer_and_training_foundations/chapter03_reproducible_training.md#四后端性能对比)。

其中 XLA+GPU 会额外对比 no-cache、dynamic KV Cache 和 static KV Cache，并报告不同 shape 导致的 executable 数量与 cold compilation 总成本。

运行 Chapter 04 的五配置 CPU 对照，以及 baseline / modern 四后端 benchmark：

```bash
./scripts/benchmark_week04.sh
```

该脚本使用三个固定 seed 汇总短程 validation 结果，并将 cold compile、warm-up 与 steady-state 性能分开记录。实验用于验证组件可归因性和后端兼容性，不代表真实模型质量排名。
