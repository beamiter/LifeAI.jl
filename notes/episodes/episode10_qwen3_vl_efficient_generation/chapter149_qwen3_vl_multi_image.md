# Chapter 149 — Qwen3-VL 多图生成与上下文预算

> 所属 Episode：Episode 10 — Qwen3-VL 高效生成
>
> 状态：Closed
>
> 开启：2026-10-26
>
> 关闭：2026-10-26

## 范围

将 `generate_hf_qwen3_vl` 从单图扩展到一个请求内的多张图像。仍为 batch 1、
全一 attention mask，沿用 Chapter 147 的显式 static cache 复用和 Chapter 148 的
greedy/sample 语义。图像可位于同一条 user 消息，也可分布在多条 user 消息中。

```julia
messages = [(role="user", content=Any[
    (type="image", image="first.png"),
    (type="image", image="second.png"),
    (type="text", text="Compare these two images."),
])]
result = generate_hf_qwen3_vl(
    vision_parameters, text_parameters, tokenizer, messages;
    cache=:static, max_new_tokens=32,
)
```

每张图片仍使用已有的 raw image processor。按 chat 消息及 content 顺序拼接
`pixel_values` 和 `image_grid_thw`，再进入已有的 vision forward。vision tower 的
`frame_ranges` 隔离每张图的 attention；main visual embeddings 和三份 DeepStack
特征按相同顺序注入展开后的视觉位置。未新增图像间 attention 或改变 decoder 公式。

返回值新增 `original_image_sizes` 和 `resized_image_sizes`，逐图保存 `(height, width)`。
单图时原来的 `original_image_size` / `resized_image_size` 仍返回 tuple；多图时这两个
旧字段为 `nothing`，调用方使用复数字段。URL/video、padding 和 batched generation
仍不支持。raw 接口要求至少一张图；纯文本继续使用 token 入口。

## 上下文与失败边界

此前 raw 和 token 入口的额外视觉预算检查，把已经展开进 prompt 的 visual tokens
又加了一遍。现在所有图的 grid 都参与视觉计数，而物理 cache 的预算统一为：

```text
length(expanded_prompt_ids) + max(0, max_new_tokens - 1)
```

最后一个输出 token 不会再写入 KV。mRoPE 坐标上限仍单独检查；物理容量通过不代表
mRoPE 自动通过。placeholder/grid 的数量和顺序继续由 mRoPE 构造器严格验证。

所有图像处理、prompt 展开、容量/mRoPE 检查和 vision 准备完成后，才 reset 旧 cache。
第二张图片损坏、角色非法、URL/video、超出容量或非法采样参数均保留旧请求。
零输出请求仍验证完整图像与 prompt，但跳过 vision/decode 并将有效 cache 长度清零。

同轮修复极小合法 temperature 的溢出：普通 Float32 采样路径保持不变；仅在缩放产生
非有限值时，以 Float64 居中后缩放。输入 logits 自身非有限或无法用 Float32 表示时
仍拒绝。fallback 的过滤后分布及选中概率保留 Float64，避免记录 trace 时又将有限值
转成 Float32 无穷；普通路径的 trace 仍为 Float32。该 fallback 避免原本通过
preflight 的配置在 cache reset 后因缩放溢出失败。

## 验证与关闭条件

新增 CPU 回归使用真实 raw processor、chat/tokenizer、mRoPE 和 tiny decoder；用于
入口测试的 vision stub 依赖逐图像素，避免只验证全零特征时掩盖图像错序。另用真实 tiny
vision tower 比较两图合并前向与单图独立前向的拼接，核对 main/DeepStack 特征及分图
attention 隔离；Float32/BFloat16 均通过，按容差比较，不声称跨 shape 逐位一致。覆盖不同尺寸、
同/跨消息、Float32/BFloat16、dynamic/static/reused cache、greedy/固定 uniform 采样、
图像顺序对 prefill logits 的影响、精确容量上限、超一拒绝及失败前状态保护。

运行默认离线 suite：

```bash
julia --project=. test/runtests.jl
```

只读 Julia depot 下可加 `--compiled-modules=existing`，防止预编译尝试写入 depot。
多图专项共 `614` 项断言通过：顺序与尺寸 metadata `35`、Float32/BFloat16
cached/sampling/replay `240`、失败原子性 `176`、单图→多图→单图存储复用 `128`、
精确容量 `23`、真实 tiny vision 隔离 `12`。采样核心及高层 trace 专项另有
`149/149` 通过，其中新增 `76` 项检查缩放溢出与 trace 精度。

多图入口初版的默认离线集成 suite 退出状态为 `0`，共 `17,557` 项通过、`2` 项
opt-in 跳过。随后修复 GPU 步长问题并加入 fixture 后，重新运行 Episode 09/10：
`5,665` 项通过、`1` 项 opt-in 跳过；其中 Chapter 149 为 `632` 项。
默认运行在沙箱内执行，日志中的 CUDA 初始化失败来自未暴露的设备；GPU 多图验收
按用户要求在沙箱外的 RTX 4090 D 上单独执行。

- [x] raw 多图入口、逐图 geometry metadata 与兼容单图字段。
- [x] 视觉 token 不重复计算、各图总量与物理/mRoPE 独立准入。
- [x] CPU 多图生成、cache 复用和失败原子性回归。
- [x] 冻结真实 Qwen3-VL-2B 两张不同几何图像的独立 Transformers oracle。
- [x] RTX 4090 D Float32：HF strict gate 与 raw/token、dynamic/static/reuse 的 logits、有效 KV 和输出。
- [x] BF16 dynamic/static 的 logits 和有效 K/V 逐位一致。

## 真实模型初次验收

按用户要求在沙箱外使用 RTX 4090 D。两张原始图像为 `128×128` 与 `128×192`，
官方 processor 产生 grids `(1,16,16)` / `(1,14,20)`，共 `134` 个视觉 token。
完整 prompt 长度 `150`，`rope_delta=-116`，8-token 请求最终 cache position 为 `157`。
两个独立 CPU Transformers 4.57.0 / PyTorch 2.7.1 进程得到逐字节一致的参考：

- metadata SHA256：`d0ce46eb851c371620897d31e72621610852bed26757e3a12e02ba5ef97b83eb`。
- tensor SHA256：`b9e7a8735c5c2ab36bb89f9dc7079715fe001027a9bc7552ac49ea468e2e226e`。
- zero-based greedy IDs：`[785,1378,5335,525,7009,19516,11,714]`。

Float32 八步 greedy 与 HF 完全相同，full-vocabulary logits 最大误差
`5.817413330078125e-5`，严格沿用 Chapter 46 阈值；greedy/sample 的 raw/token、
dynamic/static/reuse 全部一致。BF16 独立进程重测确认 prefill 和第一次 decode 一致，
第二次 decode 起 dynamic/static logits 有差异，八枚 greedy 和采样 token 暂仍一致；
各路径自己的 raw/token 和 static reuse 保持 exact。
[初次 BF16 诊断](../../../benchmark_results/qwen3_vl_multi_image/initial_bf16_diagnostic.json)
保留了逐步误差与 K/V 差异。初次验收据此失败，没有以相同输出 token 替代 logits/KV 门禁。

## BF16 context GEMM 的容量步长修复

同前缀、同 token 的逐层诊断把首差定位到第二次 decode 的第 3 层 attention：一个
BF16 元素相差 `1.4901161193847656e-8`。固定真实 Q/K/V 后，scores、softmax 和
weights 全部逐位一致，首次差异来自 context GEMM。只改变物理容量就能复现：
`seq=152, capacity=157` 失败，而 capacity `152/160/256` 均逐位一致。

原因是静态 `weight_perm` 把容量作为矩阵 leading dimension，dynamic 使用有效长度；
这改变了 cuBLAS 的对齐条件及累积路径。修复只把 CUDA `weight_perm` 的逻辑形状改为
`(seq, groups, kv_heads)`，父缓冲、offset、地址和对象保持不变。缓存的 softmax kernel
参数在形状变化时同步刷新，避免继续写旧布局；内存统计仍按物理 capacity 计数。
CPU 路径保持原有实现。

[固定输入的视图修复实验](../../../benchmark_results/qwen3_vl_multi_image/context_view_probe.json)
证明 capacity 157 的差异被消除，152/160/256 的对照保持一致。约 55KB 的
[`attention_context_stride.json`](../../../test/episodes/episode10_qwen3_vl_efficient_generation/chapter149_qwen3_vl_multi_image/fixtures/attention_context_stride.json)
保留了出错 head 的 V/W，GPU 回归比较修复结果与当前设备的 dense baseline，不要求所有
GPU/toolkit 都重现旧 cuBLAS 差异。另覆盖 `152→151→152` 的逻辑形状切换、kernel 参数
缓存刷新、缓冲身份和物理字节计数。默认 fixture 检查 `18/18` 通过。

## 修复后的验收与关闭

RTX 4090 D 上，两种 dtype × 两种生成策略 × 六种入口/cache 组合共 `24` 个
8-token 请求全部通过：raw/token 的 dynamic、fresh static 与 reused static；每个
dtype/strategy 内的 token、逐步 logits、采样分布与有效 K/V 哈希完全一致，复用时
K/V 和 workspace 的对象及设备地址不变。Float32 八步仍保持 HF strict gate，最大
logits 误差 `5.817413330078125e-5`。BF16 使用同 dtype 内部对照，不声称跨 HF
Float32 的 logits exact。

专项 GPU 回归 `98/98` 通过：fixture `18`、context/存储/零分配 hook `26`、完整
attention 动态形状与物理字节计数 `54`。其中逻辑形状更新 hook 在预热后测得 `0` CPU
allocated bytes；这不等于重测整条长生成的全部 allocation/performance。

[最终 GPU 报告](../../../benchmark_results/qwen3_vl_multi_image/summary.json) SHA256：
`6afd957cf9b5683fede47d610505773e7910270494a8a38f7dffb13c489d0754`。
报告冻结模型资产、独立 oracle 和生产/验证代码哈希；
[oracle metadata](../../../benchmark_results/qwen3_vl_multi_image/oracle_metadata.json)
保留完整输入、逐步结果与 tensor hashes。原始 tensor 文件可按以下命令重新导出：

```bash
MODEL=/home/ubuntu/models/modelscope/Qwen/Qwen3-VL-2B-Instruct
ORACLE=/home/ubuntu/models/lifeai-oracles/qwen3vl-transformers-4.57.0/bin/python
PYTHONDONTWRITEBYTECODE=1 "$ORACLE" scripts/export_qwen3_vl_multi_image_reference.py \
  "$MODEL" /tmp/lifeai-ch149-oracle --device cpu --dtype float32 --greedy-tokens 8 --threads 8
OPENBLAS_NUM_THREADS=1 julia --compiled-modules=existing --project=. --startup-file=no \
  scripts/verify_qwen3_vl_multi_image_cuda.jl "$MODEL" /tmp/lifeai-ch149-oracle \
  benchmark_results/qwen3_vl_multi_image/summary.json 8 both
LIFEAI_TEST_CUDA=true julia --compiled-modules=existing --project=. --startup-file=no \
  test/episodes/episode10_qwen3_vl_efficient_generation/chapter149_qwen3_vl_multi_image/test_qwen3_vl_attention_context_stride.jl
```

后两条命令需要能访问 GPU 的执行环境。CPU 回归、独立参考和真实 GPU 门禁均已完成，
Chapter 149 于 2026-10-26 Closed。Episode 10 保持 Open：本章真实权重范围是双图、
batch 1、全一 mask 与 8-token 请求；video、padding、长多图生成和完整执行图仍未验收。
本轮改变了静态 attention 的有效布局，旧的 256-token allocation/分叉哈希仍只代表其
原始提交，不能当作修复后实现的性能或长生成结果。
