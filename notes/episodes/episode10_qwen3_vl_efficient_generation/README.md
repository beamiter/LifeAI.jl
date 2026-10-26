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
24. [`Chapter 69 — Qwen3-VL decode attention softmax 元素 kernel`](chapter69_qwen3_vl_softmax_kernel.md)（Closed）
25. [`Chapter 70 — Qwen3-VL decode attention permute kernel`](chapter70_qwen3_vl_permute_kernel.md)（Closed）
26. [`Chapter 71 — Qwen3-VL decode QK RMSNorm 的准备`](chapter71_qwen3_vl_qk_prepare.md)（Closed）
27. [`Chapter 72 — Qwen3-VL decode cublasGemmEx 的启动对象`](chapter72_qwen3_vl_gemm_launch.md)（Closed）
28. [`Chapter 73 — Qwen3-VL decode hidden RMSNorm 的准备`](chapter73_qwen3_vl_hidden_prepare.md)（Closed）
29. [`Chapter 74 — Qwen3-VL decode attention maximum`](chapter74_qwen3_vl_attention_max.md)（Closed）
30. [`Chapter 75 — Qwen3-VL decode per-head GEMM 的指针偏移`](chapter75_qwen3_vl_batched_pointer.md)（Closed）
31. [`Chapter 76 — Qwen3-VL decode abs2 归约计划的具体类型`](chapter76_qwen3_vl_abs2_plan.md)（Closed）
32. [`Chapter 77 — Qwen3-VL decode 单 token K/V 写入`](chapter77_qwen3_vl_kv_write.md)（Closed）
33. [`Chapter 78 — Qwen3-VL decode attention softmax 分母`](chapter78_qwen3_vl_attention_sum.md)（Closed）
34. [`Chapter 79 — Qwen3-VL decode QK abs2 的具体引用`](chapter79_qwen3_vl_qk_abs2_ref.md)（Closed）
35. [`Chapter 80 — Qwen3-VL decode QK scale 的 reshape`](chapter80_qwen3_vl_qk_scale.md)（Closed）
36. [`Chapter 81 — Qwen3-VL decode hidden abs2 的尾部归约`](chapter81_qwen3_vl_hidden_tail.md)（Closed）
37. [`Chapter 82 — Qwen3-VL decode K/V 的同一次拷贝`](chapter82_qwen3_vl_kv_pair.md)（Closed）
38. [`Chapter 83 — Qwen3-VL decode Q/K RoPE 的同一次启动`](chapter83_qwen3_vl_rope_pair.md)（Closed）
39. [`Chapter 84 — Qwen3-VL decode attention 的准备启动`](chapter84_qwen3_vl_attn_prepare.md)（Closed）
40. [`Chapter 85 — Qwen3-VL decode attention weight 的置换`](chapter85_qwen3_vl_weight_perm.md)（Closed）
41. [`Chapter 86 — Qwen3-VL decode QK RMSNorm 的同一次启动`](chapter86_qwen3_vl_qk_fused.md)（Closed）
42. [`Chapter 87 — Qwen3-VL decode attention softmax 的同一次启动`](chapter87_qwen3_vl_softmax.md)（Closed）
43. [`Chapter 88 — Qwen3-VL decode mRoPE 频率表`](chapter88_qwen3_vl_mrope_tables.md)（Closed）
44. [`Chapter 89 — Qwen3-VL decode 单 token GEMM`](chapter89_qwen3_vl_gemm_token.md)（Closed）
45. [`Chapter 90 — Qwen3-VL decode hidden RMSNorm 的两次 kernel`](chapter90_qwen3_vl_hidden_fused.md)（Closed）
46. [`Chapter 91 — Qwen3-VL decode attention softmax 的启动对象`](chapter91_qwen3_vl_softmax_launch.md)（Closed）
47. [`Chapter 92 — Qwen3-VL decode cublasGemmEx 的 context`](chapter92_qwen3_vl_gemm_context.md)（Closed）
48. [`Chapter 93 — Qwen3-VL decode attention query 的 reshape`](chapter93_qwen3_vl_query_matrix.md)（Closed）
49. [`Chapter 94 — Qwen3-VL decode hidden RMSNorm 的一次启动`](chapter94_qwen3_vl_hidden_single.md)（Closed）
50. [`Chapter 95 — Qwen3-VL decode attention softmax 的直接启动`](chapter95_qwen3_vl_softmax_raw.md)（Closed）
51. [`Chapter 96 — Qwen3-VL decode attention 准备和写回的直接启动`](chapter96_qwen3_vl_attn_raw.md)（Closed）
52. [`Chapter 97 — Qwen3-VL decode hidden RMSNorm 的直接启动`](chapter97_qwen3_vl_hidden_raw.md)（Closed）
53. [`Chapter 98 — Qwen3-VL decode residual 加法的直接启动`](chapter98_qwen3_vl_residual_raw.md)（Closed）
54. [`Chapter 99 — Qwen3-VL decode QKV 的 reshape`](chapter99_qwen3_vl_qkv_matrix.md)（Closed）
55. [`Chapter 100 — Qwen3-VL decode QK RMSNorm 的直接启动`](chapter100_qwen3_vl_qk_raw.md)（Closed）
56. [`Chapter 101 — Qwen3-VL decode QK RoPE 的直接启动`](chapter101_qwen3_vl_rope_raw.md)（Closed）
57. [`Chapter 102 — Qwen3-VL decode K/V 写入的直接启动`](chapter102_qwen3_vl_kv_raw.md)（Closed）
58. [`Chapter 103 — Qwen3-VL decode MLP SiLU 的直接启动`](chapter103_qwen3_vl_mlp_raw.md)（Closed）
59. [`Chapter 104 — Qwen3-VL decode stage 不再包闭包`](chapter104_qwen3_vl_stage_inline.md)（Closed）
60. [`Chapter 105 — Qwen3-VL decode 单 token 层走热路径`](chapter105_qwen3_vl_decode_block.md)（Closed）
61. [`Chapter 106 — Qwen3-VL decode 28 层一次调用`](chapter106_qwen3_vl_decode_layers.md)（Closed）
62. [`Chapter 107 — Qwen3-VL decode 复用 cache 重叠检查`](chapter107_qwen3_vl_seen_storage.md)（Closed）
63. [`Chapter 108 — Qwen3-VL decode 不再拷贝整份参数`](chapter108_qwen3_vl_decode_pack.md)（Closed）
64. [`Chapter 109 — Qwen3-VL 长生成计时不再捕获参数`](chapter109_qwen3_vl_time_decode.md)（Closed）
65. [`Chapter 110 — Qwen3-VL decode token embedding 直接拷贝`](chapter110_qwen3_vl_embedding_copy.md)（Closed）
66. [`Chapter 111 — Qwen3-VL decode mRoPE 不再传递整个 spec`](chapter111_qwen3_vl_mrope_fields.md)（Closed）
67. [`Chapter 112 — Qwen3-VL decode 复用具体 scratch`](chapter112_qwen3_vl_decode_ready.md)（Closed）
68. [`Chapter 113 — Qwen3-VL decode mRoPE 主机拷贝不再查询 context`](chapter113_qwen3_vl_mrope_copy.md)（Closed）
69. [`Chapter 114 — Qwen3-VL decode tied logits 复用 reshape`](chapter114_qwen3_vl_tied_matrix.md)（Closed）
70. [`Chapter 115 — Qwen3-VL 稳态循环只进入一次具体包装`](chapter115_qwen3_vl_ready_loop.md)（Closed）
71. [`Chapter 116 — Qwen3-VL 稳态计时不再包一层 @sync`](chapter116_qwen3_vl_ready_timer.md)（Closed）
72. [`Chapter 117 — Qwen3-VL 主机选 token 不再做全长排序`](chapter117_qwen3_vl_top_two.md)（Closed）
73. [`Chapter 118 — Qwen3-VL 主机 logits 复用一块缓冲`](chapter118_qwen3_vl_logit_copy.md)（Closed）
74. [`Chapter 119 — Qwen3-VL 第一次 decode 不再重拷 blocks`](chapter119_qwen3_vl_ready_install.md)（Closed）
75. [`Chapter 120 — Qwen3-VL attention scratch 计字节不再装箱`](chapter120_qwen3_vl_attention_bytes.md)（Closed）
76. [`Chapter 121 — Qwen3-VL attention scratch 共用两块存储`](chapter121_qwen3_vl_attention_pack.md)（Closed）
77. [`Chapter 122 — Qwen3-VL QK RoPE scratch 共用两块存储`](chapter122_qwen3_vl_rope_pack.md)（Closed）
78. [`Chapter 123 — Qwen3-VL QK norm scratch 共用两块存储`](chapter123_qwen3_vl_qk_norm_pack.md)（Closed）
79. [`Chapter 124 — Qwen3-VL hidden norm scratch 共用两块存储`](chapter124_qwen3_vl_hidden_norm_pack.md)（Closed）
80. [`Chapter 125 — Qwen3-VL MLP activation scratch 共用两块存储`](chapter125_qwen3_vl_mlp_pack.md)（Closed）
81. [`Chapter 126 — Qwen3-VL hidden residual scratch 共用两块存储`](chapter126_qwen3_vl_residual_pack.md)（Closed）
82. [`Chapter 127 — Qwen3-VL ready 不再拷贝 blocks`](chapter127_qwen3_vl_ready_pack.md)（Closed）
83. [`Chapter 128 — Qwen3-VL 第一次 decode 不再包一层 @sync`](chapter128_qwen3_vl_first_timer.md)（Closed）
84. [`Chapter 129 — Qwen3-VL 参数 pack 随 cache 创建`](chapter129_qwen3_vl_cache_pack.md)（Closed）
85. [`Chapter 130 — Qwen3-VL 九块 BF16 scratch 共用一块存储`](chapter130_qwen3_vl_bf16_tails.md)（Closed）
86. [`Chapter 131 — Qwen3-VL decode scratch 共用两块存储`](chapter131_qwen3_vl_decode_workspace.md)（Closed）
87. [`Chapter 132 — Qwen3-VL 复用 kernel 参数盒`](chapter132_qwen3_vl_param_reuse.md)（Closed）
88. [`Chapter 133 — Qwen3-VL scratch 切片共用父引用`](chapter133_qwen3_vl_shared_ref.md)（Closed）
89. [`Chapter 134 — Qwen3-VL 复用 reshape 矩阵`](chapter134_qwen3_vl_reshape_reuse.md)（Closed）
90. [`Chapter 135 — Qwen3-VL 复用 scratch 切片`](chapter135_qwen3_vl_slice_reuse.md)（Closed）
91. [`Chapter 136 — Qwen3-VL 复用请求直接安装 Ready`](chapter136_qwen3_vl_shell_ready.md)（Closed）
92. [`Chapter 137 — Qwen3-VL 复用 Ready`](chapter137_qwen3_vl_ready_reuse.md)（Closed）
93. [`Chapter 138 — Qwen3-VL 复用父缓冲`](chapter138_qwen3_vl_parent_reuse.md)（Closed）
94. [`Chapter 139 — Qwen3-VL attention capacity 原地更新`](chapter139_qwen3_vl_attention_capacity.md)（Closed）
95. [`Chapter 140 — Qwen3-VL 复用入口不再装箱 spec`](chapter140_qwen3_vl_reuse_entry.md)（Closed）
96. [`Chapter 141 — Qwen3-VL 静态 Ready 调用`](chapter141_qwen3_vl_static_ready.md)（Closed）
97. [`Chapter 142 — Qwen3-VL 第一次 decode 直接调用 Ready`](chapter142_qwen3_vl_ready_call.md)（Closed）
98. [`Chapter 143 — Qwen3-VL GPU 的 BF16 转 Float32 直接拷贝`](chapter143_qwen3_vl_f32_copy.md)（Closed）
99. [`Chapter 144 — Qwen3-VL GPU 的 BF16 转 Float32 复用 kernel`](chapter144_qwen3_vl_promote_kernel.md)（Closed）
100. [`Chapter 145 — Qwen3-VL 静态 decode 的完整参数绑定`](chapter145_qwen3_vl_parameter_binding.md)（Closed）
101. [`Chapter 146 — Qwen3-VL CUDA 工作区的请求所有权`](chapter146_qwen3_vl_workspace_ownership.md)（Closed）
102. [`Chapter 147 — Qwen3-VL 显式复用高层生成 cache`](chapter147_qwen3_vl_generation_reuse.md)（Closed）
103. [`Chapter 148 — Qwen3-VL 可重放采样生成`](chapter148_qwen3_vl_sampling.md)（Closed；CPU/GPU 与 DSW 默认集成 `16,825` 项通过）
104. [`Chapter 149 — Qwen3-VL 多图生成与上下文预算`](chapter149_qwen3_vl_multi_image.md)（Closed；双图 CPU/真实 GPU 验收通过，并修复 BF16 context GEMM 步长差异）

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

Chapter 147 已完成真实 2B GPU 的显式请求 cache 复用；Chapter 148 继续把高层 token/raw 图文接口扩展为可重放采样。temperature/top-k/top-p、固定 uniform、RNG 与 distribution trace 的 CPU 专项 `1345/1345` 已通过。真实 RTX PRO 5000 72GB BF16 的 32/128/256-token sampling 与复用已通过，逐步 logits、分布与有效 K/V exact，独立 oracle 和 top-k=1/greedy 对照通过。[报告](../../../benchmark_results/qwen3_vl_sampling/summary.json) 冻结模型、源码与随机流 provenance。DSW 默认集成 suite `16,825` 项通过，Chapter 148 于 2026-10-15 Closed；Episode 10 继续 Open。默认 greedy 及已记录的 BF16/HF Float32 数值边界不变。


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
  Chapter 149 已完成 raw 双图 CPU/真实 GPU 验收；长多图生成与 video 继续作为显式边界。sampling 已由 Chapter 148 完成独立验收。

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
Chapter 69 把 scale、exp、weight 分成三次 kernel，`maximum!` 和 `sum!` 留在
原处。sequence `1 / 80 / 330 / 331` 的 2048 个 attention 输出与广播路径逐位
一致。GPU 最后一步仍是 `0`。256-token 稳态一步的 CPU 分配从 `2,168,720`
降到 `1,817,040`。Chapter 70 把 query、K/V、weight 和 context 的置换收成
拷贝 kernel。GPU 最后一步仍是 `0`。256-token 稳态一步的 CPU 分配降到
`1,422,800`。Chapter 71 把 QK norm 的升精度和除以 `head_dim` 收进 kernel，
`sum!` 改走缓存的同一条归约。GPU 最后一步仍是 `0`。256-token 稳态一步的
CPU 分配降到 `1,126,224`。Chapter 72 把 `cublasGemmEx` 的 handle、数据类型
和 device 地址收成持久对象。GPU 最后一步仍是 `0`。256-token 稳态一步的
CPU 分配降到 `873,104`。Chapter 73 把 hidden RMSNorm 的升精度和除法收进
kernel。GPU 最后一步仍是 `0`。256-token 稳态一步的 CPU 分配降到
`677,920`。Chapter 74 把 attention 的 `maximum!` 换成逐 head 顺序扫描。
GPU 最后一步仍是 `0`。256-token 稳态一步的 CPU 分配降到 `617,440`。
Chapter 75 用 stride 偏移代替 per-head GEMM 的 `view`。GPU 最后一步仍是
`0`。256-token 稳态一步的 CPU 分配降到 `581,600`。Chapter 76 把 abs2
归约计划收成具体类型，归约本身不变。GPU 最后一步仍是 `0`。256-token 稳态
一步的 CPU 分配降到 `513,152`。Chapter 77 把单 token 的 K/V `copyto!` 换成
拷贝 kernel。GPU 最后一步仍是 `0`。256-token 稳态一步的 CPU 分配降到
`441,024`。Chapter 78 把 attention softmax 分母换成与 `sum!` 相同的单 block
warp 树。GPU 最后一步仍是 `0`。position `107 / 203 / 331` 的 CPU 都是
`424,896`。Chapter 79 把 Q/K 的 abs2 计划放进具体 `Ref`。GPU 最后一步仍是
`0`。position `107 / 203 / 331` 的 CPU 都是 `415,936`。Chapter 80 复用
Q/K scale 的 reshape。GPU 最后一步仍是 `0`。position `107 / 203 / 331`
的 CPU 都是 `415,744`。Chapter 81 把 hidden abs2 的 4 块尾部换成同一条
warp 树。GPU 最后一步仍是 `0`。position `107 / 203 / 331` 的 CPU 都是
`394,768`。Chapter 82 把 K/V 的两次拷贝和两次写回收成各一次启动。
GPU 最后一步仍是 `0`。position `107 / 203 / 331` 的 CPU 都是 `388,496`。
Chapter 83 把 Q/K RoPE 收成一次启动。GPU 最后一步仍是 `0`。position
`107 / 203 / 331` 的 CPU 都是 `384,464`。Chapter 84 把 query 载入和 K/V
拷贝收成一次启动。GPU 最后一步仍是 `0`。position `107 / 203 / 331` 的
CPU 都是 `381,328`。Chapter 85 把 softmax 权重和它的置换收成一次启动。
GPU 最后一步仍是 `0`。position `107 / 203 / 331` 的 CPU 都是 `377,744`。
Chapter 86 把 Q/K RMSNorm 收成一次启动。GPU 最后一步仍是 `0`。position
`107 / 203 / 331` 的 CPU 都是 `365,648`。Chapter 87 把 attention softmax
收成一次启动。GPU 最后一步仍是 `0`。position `107 / 203 / 331` 的 CPU
都是 `344,144`。Chapter 88 复用 mRoPE 的 inv frequency 和 host 缓冲。
GPU 最后一步仍是 `0`。position `107 / 203 / 331` 的 CPU 都是 `342,672`。
Chapter 89 让单 token GEMM 直接使用原来的 stride。GPU 最后一步仍是 `0`。
position `107 / 203 / 331` 的 CPU 都是 `312,208`。Chapter 90 把长度
`2048` 的 hidden RMSNorm 收成两次 kernel。GPU 最后一步仍是 `0`。
position `107 / 203 / 331` 的 CPU 都是 `297,952`。Chapter 91 把 attention
softmax 的 kernel 留在具体 `Ref` 里。GPU 最后一步仍是 `0`。position
`107 / 203 / 331` 的 CPU 都是 `288,544`。Chapter 92 让 `cublasGemmEx`
不再每次查询 context。GPU 最后一步仍是 `0`。position `107 / 203 / 331`
的 CPU 都是 `278,224`。Chapter 93 复用 attention query 的 reshape。
GPU 最后一步仍是 `0`。position `107 / 203 / 331` 的 CPU 都是 `275,536`。
Chapter 94 把长度 `2048` 的 hidden RMSNorm 收成一次启动。GPU 最后一步
仍是 `0`。position `107 / 203 / 331` 的 CPU 都是 `266,416`。Chapter 95
直接启动 attention softmax。GPU 最后一步仍是 `0`。position
`107 / 203 / 331` 的 CPU 都是 `260,144`。Chapter 96 直接启动 attention
的准备和 context 写回。GPU 最后一步仍是 `0`。position `107 / 203 / 331`
的 CPU 都是 `250,288`。Chapter 97 直接启动 hidden RMSNorm。GPU 最后一步
仍是 `0`。position `107 / 203 / 331` 的 CPU 都是 `240,256`。Chapter 98
直接启动 residual 加法。GPU 最后一步仍是 `0`。position `107 / 203 / 331`
的 CPU 都是 `231,296`。Chapter 99 复用 QKV 的 reshape。GPU 最后一步仍是
`0`。position `107 / 203 / 331` 的 CPU 都是 `223,232`。Chapter 100
直接启动 QK RMSNorm。GPU 最后一步仍是 `0`。position `107 / 203 / 331`
的 CPU 都是 `211,776`。Chapter 101 直接启动 QK RoPE。GPU 最后一步仍是
`0`。position `107 / 203 / 331` 的 CPU 都是 `205,952`。Chapter 102
直接启动 K/V 写入。GPU 最后一步仍是 `0`。position `107 / 203 / 331`
的 CPU 都是 `201,024`。Chapter 103 直接启动 MLP SiLU。GPU 最后一步仍是
`0`。position `107 / 203 / 331` 的 CPU 都是 `196,544`。Chapter 104
在没有 profiler 时直接执行 stage。GPU 最后一步仍是 `0`。position
`107 / 203` 的 CPU 是 `129,616`，`331` 是 `129,600`。Chapter 105
在没有 profiler 时让单 token 层只走热路径。GPU 最后一步仍是 `0`。
position `107 / 203` 的 CPU 是 `22,096`，`331` 是 `22,080`。Chapter 106
把 28 层收成一次调用。GPU 最后一步仍是 `0`。position `107 / 203`
的 CPU 是 `16,784`，`331` 是 `16,768`。Chapter 107 复用 cache
重叠检查的数组。GPU 最后一步仍是 `0`。position `107 / 203` 的 CPU 是
`13,360`，`331` 是 `13,344`。Chapter 108 不再在稳态调用里拷贝整份
参数。GPU 最后一步仍是 `0`。position `107 / 203` 的 CPU 是 `10,288`，
`331` 是 `10,272`。Chapter 109 把长生成的 decode 计时放进小函数。
GPU 最后一步仍是 `0`。position `107 / 203` 的 CPU 是 `7,216`，`331`
是 `7,200`。Chapter 110 直接拷贝 token embedding 的一列。GPU 最后一步
仍是 `0`。position `107 / 203` 的 CPU 是 `5,360`，`331` 是 `5,344`。
Chapter 111 让 mRoPE 只接收 `head_dim`、`rope_theta` 和
`mrope_section`。GPU 最后一步仍是 `0`。position `107 / 203` 的 CPU 是
`5,264`，`331` 是 `5,248`。Chapter 112 把 scratch 留在具体类型的
包装里。GPU 最后一步仍是 `0`。position `107 / 203` 的 CPU 是 `4,416`，
`331` 是 `4,400`。Chapter 113 直接把 mRoPE 的主机表抄到设备。GPU
最后一步仍是 `0`。position `107 / 203` 的 CPU 是 `4,352`，`331` 是
`4,336`。Chapter 114 复用 tied logits 的两张矩阵。GPU 最后一步仍是
`0`。position `107 / 203` 的 CPU 是 `4,192`，`331` 是 `4,176`。
Chapter 115 让稳态循环只进入一次具体包装。GPU 最后一步仍是 `0`。
position `107 / 203 / 331` 的 CPU 都是 `4,144`。Chapter 116 让稳态
计时不再包一层 `@sync`。GPU 最后一步仍是 `0`。position
`107 / 203 / 331` 的 CPU 都是 `0`。Chapter 117 用线性扫描代替主机
选择的全长排序。GPU 最后一步仍是 `0`。position `107 / 203 / 331`
的 CPU 仍是 `0`。256-token 主机选择从 `2,739,400` 降到 `308,232`。
Chapter 118 复用一块主机 logits 缓冲。GPU 最后一步仍是 `0`。
position `107 / 203 / 331` 的 CPU 仍是 `0`。256-token 主机选择
从 `308,232` 降到 `0`。Chapter 119 让第一次 decode 沿用 pack 里的
blocks。GPU 最后一步仍是 `0`。position `107 / 203 / 331` 的 CPU
仍是 `0`。128/256 的第一次 decode CPU 从 `40,368` 降到 `37,664`。
Chapter 120 让 attention scratch 的字节核对不再装箱。GPU 最后一步
仍是 `0`。position `107 / 203 / 331` 的 CPU 仍是 `0`。128/256 的
第一次 decode CPU 从 `37,664` 降到 `35,904`。Chapter 121 让
attention scratch 的 13 块数组共用两块存储。GPU 最后一步仍是 `0`。
position `107 / 203 / 331` 的 CPU 仍是 `0`。128/256 的第一次 decode
CPU 从 `35,904` 降到 `34,304`。Chapter 122 让 QK RoPE scratch 的
12 块数组共用两块存储。GPU 最后一步仍是 `0`。position
`107 / 203 / 331` 的 CPU 仍是 `0`。128/256 的第一次 decode CPU 从
`34,304` 降到 `32,864`。Chapter 123 让 QK norm scratch 的 7 块数组
共用两块存储。GPU 最后一步仍是 `0`。position `107 / 203 / 331`
的 CPU 仍是 `0`。128/256 的第一次 decode CPU 从 `32,864` 降到
`32,224`。Chapter 124 让 hidden norm scratch 的 6 块数组共用两块
存储。GPU 最后一步仍是 `0`。position `107 / 203 / 331` 的 CPU 仍是
`0`。128/256 的第一次 decode CPU 从 `32,224` 降到 `31,744`。
Chapter 125 让 MLP activation scratch 的 5 块数组共用两块存储。
GPU 最后一步仍是 `0`。position `107 / 203 / 331` 的 CPU 仍是 `0`。
128/256 的第一次 decode CPU 从 `31,744` 降到 `31,424`。Chapter 126
让 hidden residual scratch 的 4 块数组共用两块存储。GPU 最后一步
仍是 `0`。position `107 / 203 / 331` 的 CPU 仍是 `0`。128/256 的
第一次 decode CPU 从 `31,424` 降到 `31,264`。Chapter 127 让 ready
只留着 pack，不再拷贝 blocks。GPU 最后一步仍是 `0`。position
`107 / 203 / 331` 的 CPU 仍是 `0`。128/256 的第一次 decode CPU 从
`31,264` 降到 `25,856`。Chapter 128 让第一次 decode 的计时不再包
一层 `@sync`。GPU 最后一步仍是 `0`。position `107 / 203 / 331`
的 CPU 仍是 `0`。128/256 的第一次 decode CPU 从 `25,856` 降到
`21,712`。Chapter 129 让参数 pack 随 cache 创建。GPU 最后一步仍是
`0`。position `107 / 203 / 331` 的 CPU 仍是 `0`。128/256 的第一次
decode CPU 从 `21,712` 降到 `19,024`。Chapter 130 让 QKV、gate/up、
mRoPE、token embedding 和 tied logits 这 9 块 BF16 数组共用一块
存储。GPU 最后一步仍是 `0`。position `107 / 203 / 331` 的 CPU 仍是
`0`。128/256 的第一次 decode CPU 从 `19,024` 降到 `17,872`。
Chapter 131 让全部 decode scratch 切自一块 BF16 和一块 Float32。
GPU 最后一步仍是 `0`。position `107 / 203 / 331` 的 CPU 仍是 `0`。
128/256 的第一次 decode CPU 从 `17,872` 降到 `14,912`。
Chapter 132 让新的 scratch 复用已经建好的 kernel 参数盒。GPU
最后一步仍是 `0`。position `107 / 203 / 331` 的 CPU 仍是 `0`。
128/256 的第一次 decode CPU 从 `14,912` 降到 `9,168`。
Chapter 133 让 scratch 切片共用父缓冲的引用。GPU 最后一步仍是
`0`。position `107 / 203 / 331` 的 CPU 仍是 `0`。32/128/256 的第一次
decode CPU 都从 `9,168` 降到 `7,408`。Chapter 134 让 QKV、tied、
query 和 attention 投影复用已经建好的 reshape。GPU 最后一步仍是
`0`。position `107 / 203 / 331` 的 CPU 仍是 `0`。32/128/256 的第一次
decode CPU 都从 `7,408` 降到 `6,768`。Chapter 135 让后来的请求复用
这 56 张 scratch 切片，只改指针和形状。GPU 最后一步仍是 `0`。
position `107 / 203 / 331` 的 CPU 仍是 `0`。32/128/256 的第一次
decode CPU 都从 `6,768` 降到 `3,264`。Chapter 136 让后来的请求直接把
这些 scratch 装进 Ready，不再装进 `Any` 字段。GPU 最后一步仍是 `0`。
position `107 / 203 / 331` 的 CPU 仍是 `0`。32/128/256 的第一次
decode CPU 都从 `3,264` 降到 `1,312`。Chapter 137 让后来的请求复用
同一个 Ready，只换新容量的 attention scratch。GPU 最后一步仍是 `0`。
position `107 / 203 / 331` 的 CPU 仍是 `0`。32/128/256 的第一次
decode CPU 都从 `1,312` 降到 `800`。Chapter 138 让后来的请求在父缓冲
装得下时不再重新分配。GPU 最后一步仍是 `0`。position
`107 / 203 / 331` 的 CPU 仍是 `0`。32/128/256 的第一次 decode CPU
都从 `800` 降到 `352`。Chapter 139 让后来的请求原地改 attention
scratch 的 `capacity`。GPU 最后一步仍是 `0`。position
`107 / 203 / 331` 的 CPU 仍是 `0`。32/128/256 的第一次 decode CPU
都从 `352` 降到 `192`。Chapter 140 让复用入口不再装箱 text spec。
GPU 最后一步仍是 `0`。position `107 / 203 / 331` 的 CPU 仍是 `0`。
32/128/256 的第一次 decode CPU 都从 `192` 降到 `48`。Chapter 141
让 BF16 Ready 的具体类型把 steady 调用收成静态调用。GPU 最后一步仍是
`0`。position `107 / 203 / 331` 的 CPU 仍是 `0`。32/128/256 的第一次
decode CPU 都从 `48` 降到 `32`。Chapter 142 让第一次 decode 的计时窗口
和稳态步一样，直接调用具体 Ready。GPU 最后一步仍是 `0`。position
`107 / 203 / 331` 的 CPU 仍是 `0`。32/128/256 的第一次 decode CPU
都从 `32` 降到 `0`。Chapter 143 让 GPU 上的 BF16 转 Float32 走
`copyto!`。GPU 最后一步仍是 `0`。position `107 / 203 / 331` 的 CPU
仍是 `0`。32/128/256 的第一次 decode CPU 仍是 `0`。256 的 prefill
CPU 从 `8,007,336` 降到 `6,918,616`。Chapter 144 让连续 GPU 数组的
BF16 转 Float32 复用同一个 kernel。GPU 最后一步仍是 `0`。position
`107 / 203 / 331` 的 CPU 仍是 `0`。32/128/256 的第一次 decode CPU
仍是 `0`。256 的 prefill CPU 从 `6,918,616` 降到 `5,657,880`。
Episode 10 保持 Open，不能把这些 scratch 写成整条 decode 零分配。
