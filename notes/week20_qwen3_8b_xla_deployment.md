# Week 20 — Qwen3-8B XLA 单驻留 4K 日常部署

> 状态：Closed（2026-07-30）

> 依赖基线：[`Week 19 — Qwen3-8B / RTX 4090 D 日常本地部署`](week19_qwen3_8b_4090d_deployment.md)
> 已把同一冻结模型落成 4K CUDA eager 日常入口，但 3,584-token prefill
> 为 46.85 s、整窗 decode 为 10.25 tok/s。Week 20 不改变模型、revision
> 或 context，只解决 XLA 原树与 packed tree 重复驻留，并建立同口径验收。

## 核心问题

能否在 24 GB 级 RTX 4090 D 上，不同时上传 ordinary 与 packed 两棵
Qwen3-8B BF16 参数树，而是从 safetensors 直接流式组装唯一 compact
packed tree、只做一次递归 device transfer，并用固定形状 XLA
prefill/decode 把 Week 19 的 3,584+512 日常窗口提速，同时守住 CUDA
BF16 greedy parity、连续请求稳定性和至少 2 GiB 的物理显存余量？

## 冻结的 XLA 日常 profile

配置：
[`configs/deployment/qwen3_8b_4090d_bf16_xla_daily.json`](../configs/deployment/qwen3_8b_4090d_bf16_xla_daily.json)

- 模型：`Qwen/Qwen3-8B`
- revision：`b968826d9c46dd6066d109eabc6255188de91218`
- dtype：BF16 权重 + BF16 KV，batch 1
- 总 context：4,096 tokens
- 最大 prompt/history：3,584 tokens
- 最大输出：512 tokens
- 固定 prefill chunk：64 tokens
- decode：greedy；设备端 argmax，每步只取回一个 token
- XLA allocator：`XLA_REACTANT_GPU_MEM_FRACTION=0.87`
- XLA preallocation：`XLA_REACTANT_GPU_PREALLOCATE=false`
- shared profile schema 中的两项 eager reclaim interval 保留为兼容字段；
  XLA 路径不调用 CUDA pool reclaim，显存边界由 BFC fraction 和连续
  物理采样验收
- 物理 GPU 容量门槛：25,000,000,000 bytes
- allocator / workspace 余量门槛：2 GiB
- profile SHA256：
  `0638eecce7864d261770c8af1698575f055cf3149d6e5d70605c7cf35dbb8d01`
- frozen asset manifest SHA256：
  `f4737c1aca92b3cbf046da7861af88fc2d4650552397b7d6f4b7edade5040e91`

两个 XLA 环境变量必须在 `using LifeAI` / `using Reactant` 之前设置；CLI
与 benchmark 均提供上述可覆盖默认值。覆盖 `0.87` 可以用于实验，但不再
等价于本 Week 已关闭的 4090D 显存 profile。

## 单驻留实现

### 流式 compact tree

`load_hf_qwen3_compact_model` 不先构造整棵 ordinary 参数树。它按层读取
safetensors，在 host 上仅短暂保留该层的独立 Q/K/V 与 gate/up，然后
立即打包为：

- 一份 `qkv_weight`；
- 一份 `gate_up_weight`；
- 保持独立的 O/down projection、Q/K norm 与两处 block norm。

因此完整 host tree 已是 XLA kernel 使用的 compact topology。Qwen3-8B
最终为 `8 × 36 + 3 = 291` 个 tensor leaves，逻辑参数
16,381,470,720 bytes（15.256 GiB）。`Reactant.to_rarray` 只对这棵树
递归调用一次；加载指标明确冻结：

- `device_parameter_tree_count = 1`；
- `device_parameter_tree_transfer_count = 1`；
- ordinary 完整参数树没有构造、没有上传；
- 没有第二棵 packed projection tree 上传。

参数上传后的 allocator `bytes_in_use` 为 16,495,444,480 bytes
（15.363 GiB），低于“逻辑参数 + 1 GiB”的驻留门槛。它不是仅凭字段名
宣称单树：离线测试还比较 compact tree 与 ordinary tree 现场打包后的
逐值相同，并对源码中的唯一 transfer 做回归检查。

### 固定窗口与编译 kernel

`HFQwen3BF16XLASession` 常驻唯一参数树、完整 4K BF16 K/V
（603,979,776 bytes，576 MiB）、RoPE 表，以及两个 executable：

- 64-token fixed-shape chunk prefill；
- 1-token static-cache greedy decode。

prompt 在 host 侧向左补齐到 64 的倍数。补位 cache slot 的 key position
设为 `typemax(Int32)`，从 attention 中排除；这使 1—3,584 token 的请求
共享同一 prefill executable，同时不让 pad token 内容影响真实 prompt
logits。窗口规划同时区分：

- logical sequence：真实 prompt + 请求输出；
- physical bucket：左补齐后的 prompt；
- cache position：bucket + generated - 1，最后选中的 token 不必写回。

真实 65-token case 因而进入 128-token bucket、左补 63 slots，并与
CUDA BF16 reference 的 32/32 tokens 一致；3,584+512 整窗最终是
logical sequence 4,096、cache position 4,095。

### 公共 API 与日常 CLI

新增公共 session API：

- `load_hf_qwen3_bf16_xla_session`
- `generate_hf_qwen3_bf16_xla!`
- `reset_hf_qwen3_bf16_xla_session!`

日常入口复用 Week 19 的 frozen asset 校验、Qwen3 chat template、
thinking 开关、system message、最老 user/assistant turn-pair 裁剪、
`/clear`、EOF 处理与 tokenizer bytes 流式输出：

```bash
julia --project=. --startup-file=no \
  scripts/run_qwen3_xla_chat.jl \
  /home/ubuntu/models/modelscope/Qwen/Qwen3-8B
```

一次性 smoke：

```bash
julia --project=. --startup-file=no \
  scripts/run_qwen3_xla_chat.jl \
  /home/ubuntu/models/modelscope/Qwen/Qwen3-8B \
  --prompt "用三点解释 KV Cache" --greedy --max-new-tokens 64
```

当前冻结 profile 只允许 greedy。非 greedy profile 会明确报错，不会把
sampling 静默降级为 greedy；`--seed` 为接口一致性保留，但不会改变
greedy 结果。

## 独立 CUDA BF16 oracle

XLA benchmark 不用自身输出当 reference。先由 Week 19 CUDA eager session
导出 schema 2 oracle：

```bash
julia --project=. --startup-file=no \
  scripts/export_qwen3_8b_greedy_reference.jl \
  /home/ubuntu/models/modelscope/Qwen/Qwen3-8B \
  configs/deployment/qwen3_8b_4090d_bf16_xla_daily.json \
  benchmark_results/week20/qwen3_8b_cuda_greedy_reference.json
```

oracle SHA256 被 benchmark 源码固定为
`83f62afbbb470b695b6990a3b86a8860407a37874354d6b039e1ce19917e2747`；
加载时还逐项核对 schema/source、model/revision、profile/manifest
SHA256、10 个 frozen assets，以及三个 case 的 prompt/output 长度：

| case | prompt | 对比输出 |
| --- | ---: | ---: |
| `single_chunk_64` | 64 | 32 |
| `left_padded_65` | 65（bucket 128） | 32 |
| `full_prompt_3584` | 3,584 | 32 |

最终 XLA 与独立 CUDA BF16 reference 为 **96 / 96 tokens 完全一致**。
这覆盖单 chunk、非整 chunk 左补齐和最大 prompt 三种形状，而不是只验证
一个短 prompt 的首 token。

## 验证结果

### 离线与编译 smoke

- Week 20 离线专项：`105 / 105`。
- 默认完整套件：`5,380 / 5,380`。
- 独立 Reactant CPU compiled prefill smoke：`5 / 5`。
- compact streamed tree 与 ordinary→compact reference 逐值相同；
  packed chunk prefill 与既有 full packed prefill 相同。
- exact 4K/rounded bucket/left-padding mask/overflow 均有离线覆盖。
- CUDA BF16 oracle 三 case：96 / 96 greedy token parity。

### RTX 4090 D 最终验收

最终运行环境为 NVIDIA GeForce RTX 4090 D，物理显存 24,564 MiB；
Reactant `0.2.275`，GPU fraction `0.87`，preallocate `false`。autotune
persistent cache 已启用，kernel persistent cache 未启用。10 个 frozen
文件共 16,392,983,007 bytes，校验耗时 33.580 s。

加载与编译：

| 阶段 | 结果 |
| --- | ---: |
| host compact BF16 load | 108.445 s |
| 唯一参数树 transfer | 3.266 s |
| RoPE / 4K KV runtime allocation | 0.283 s |
| 64-token prefill compile | 71.696 s |
| 1-token decode compile | 25.204 s |
| 总 ready wall | 213.219 s |
| ready 物理 free | 6,909 MiB |

请求与性能：

| 项目 | 结果 |
| --- | ---: |
| 首次 64+32 请求 | 8.762 s wall；prefill 4.249 s；decode 6.99 tok/s |
| 后续 7 次 64-token prefill 中位数 | 0.02733 s |
| 后续 7 次 32-token decode 中位数 | 40.421 tok/s |
| 65-token 左补齐请求 | bucket 128；32/32 parity；0.810 s wall |
| 3,584-token prefill | 1.49764 s，门槛 ≤ 20 s |
| 3,584+512 整窗 | 13.921 s wall |
| 整窗 decode | 41.132 tok/s，门槛 ≥ 25 tok/s |
| 整窗 sequence / cache | 4,096 / 4,095 |
| 8 次复用请求 allocator drift | 234,752 bytes（0.224 MiB） |
| allocator 估算最低 free | 4,509,495,552 bytes（4.200 GiB） |
| 连续物理采样最低 free | 2,412,773,376 bytes（2.247 GiB，2,301 MiB） |

相对 Week 19 同一模型、同一 4K 窗口的 CUDA eager 最终值，XLA 的
3,584-token prefill 从 46.849 s 降到 1.498 s，约 31.3×；整窗 decode
从 10.246 提升到 41.132 tok/s，约 4.0×。cold/ready 成本仍单独保留，
不把 compile 时间混进 steady throughput，也不把 steady 数字写成首次
启动体验。

### 显存口径

最终 benchmark 同时记录 Reactant allocator 和 `nvidia-smi`：

- allocator limit：21,959,278,592 bytes；
- ready allocator in-use：17,100,490,240 bytes；
- 整窗后 allocator peak：17,449,783,040 bytes；
- 请求结束离散物理快照最低 free：2,305 MiB；
- 200 ms 连续外部采样 1,225 点，最大 used 21,769 MiB，最低 free
  2,301 MiB。

最终连续采样 CSV：
`benchmark_results/week20/qwen3_8b_4090d_bf16_xla_daily_nvidia_smi.csv`，
SHA256
`72fd3fe80b56647714604a85499a3a9d8e5833412f7a50864d8ea1aa6588b586`。
这里的 2,301 MiB 是资产校验完成后，覆盖 XLA load、compile、短请求、
padding case 与 3,584+512 整窗的真实外部 trace 最小值，不是只在请求
结束查询一次的 snapshot。

最终 JSON：
`benchmark_results/week20/qwen3_8b_4090d_bf16_xla_daily.json`，SHA256
`075dc76a023a8143213f640bfb354d6e38c10b5747b5ef3e8c7e6baeb1c730dc`；
其中所有 acceptance 字段均为 true，顶层 `closed=true`。

## Close 条件

- [x] 10 个 frozen model assets 与 immutable revision 全部匹配。
- [x] host 直接构造 291-leaf compact tree，只上传一棵参数树一次；参数
  驻留不高于逻辑参数 + 1 GiB。
- [x] 单 chunk、左补齐和 3,584-token 三 case 与独立 CUDA BF16 oracle
  达到 96 / 96 greedy tokens。
- [x] 8 次同 session 请求确定一致，allocator drift 不高于 256 MiB。
- [x] 3,584+512 禁用 EOS 后完整生成，sequence/cache 为 4,096/4,095。
- [x] 3,584-token prefill 不高于 20 s，整窗 decode 不低于 25 tok/s。
- [x] allocator 估算和 200 ms 连续物理采样最低 free 均至少 2 GiB。
- [x] 日常 CLI 保留 asset gate、chat/history、流式 bytes、`/clear` 与 EOF
  行为，非 greedy profile fail closed。
- [x] Week 20 专项、独立 XLA smoke 与默认回归通过：
  `105 / 105`、`5 / 5`、`5,380 / 5,380`。

## 风险与取舍

- 当前日常 XLA profile 是 batch-1 greedy，不宣称 sampling 已完成硬件
  验收。公共 session 保留 sampled 分支，但它会逐步取回完整 logits，
  不属于本 Week 的 closed 性能口径。
- greedy decode 虽在设备端做 argmax，流式 CLI 仍每 token 进行一次
  executable 调用和单 token D2H；尚未把整个 decode loop 放进一个 XLA
  while，也没有 device-side EOS/sampling 闭环。
- 64-token chunk 限定了单次 prefill shape，但 attention 仍是普通实现，
  不是 FlashAttention。4K 是本机日常工程边界，不外推为 Qwen3 原生
  40,960 context 的实证。
- ready 需要 213 s，其中 host load 108 s、compile 约 97 s。当前只持久化
  autotune cache，kernel executable cache 未启用；适合模型常驻后的多次
  请求，不适合每条 prompt 启动一个新进程。
- `XLA_REACTANT_GPU_MEM_FRACTION=0.87` 是显存安全契约的一部分。调高可让
  allocator 扩大 pool，但会直接侵蚀桌面/driver/其他进程余量；调低则需
  重新证明 compile 和整窗不会 OOM。
- benchmark 使用冻结的重复 token probe，目标是 shape、数值、速度与显存
  验收，不是回答质量评测。日常 chat 的语义质量仍来自同一 Qwen3-8B
  BF16 权重，不由该 synthetic probe 证明。

## 实验与过程记录

### 2026-07-30：Open，并从重复驻留改为单树 transfer

- Week 19 已定位 8B packed-XLA 的阻塞点：ordinary tree 与 packed
  projection tree 同时上传，未计完整 KV 就超过 4090D 安全预算。
- 新增 header-indexed streamed tensor 读取的 BF16 target dtype，逐层
  组装 QKV/gate-up，避免先建立完整 ordinary tree。
- compact host tree 与既有 ordinary tree 的 `_bf16a_compact_parameters`
  结果逐值相同；291 个 leaves 与逻辑 bytes 都进入测试。
- 加入固定 64-token prefill、rounded bucket、left-padding key mask、
  static 4K KV 和 allocator snapshots。
- 建立独立 CUDA eager schema 2 oracle，并把 oracle SHA 固定在 XLA
  benchmark 中，防止任意 caller JSON 被当成 correctness reference。

### 2026-07-30：0.95 与 0.89 未通过物理余量门禁

第一轮沿用较激进的 `GPU_MEM_FRACTION=0.95`。它完成单树、整窗和性能
检查，阶段性 JSON 在尚未纳入连续物理门禁的旧 acceptance 下显示
`closed=true`；但 3,215 个 200 ms `nvidia-smi` 样本显示最低 free
只有 **330 MiB**。该结果不能作为日常 profile，因而作废。保留证据：

- JSON SHA256：
  `ac1a609278634d9841972e7b06b5d7cdf5ff8b451032b00c2c8e0d292698cef6`
- CSV SHA256：
  `4593827f03536584879f43056cc6d178112d10c2a49a557f4c78924d69911386`

第二轮把 fraction 降到 `0.89`，1,517 个连续样本最低 free 为
**1,848 MiB**，仍低于冻结的 2 GiB 门槛，因此不 Close。失败 trace
SHA256：
`e217b8f8d9b96fec548843692a2cbc1f934643571369aee3e6f1b391198fd17e`。

这两轮促使 benchmark 把外部 monitor 纳入自身生命周期，并把 trace
SHA、样本数、真实最小 free 与 `physical_minimum_free_passed` 写入最终
JSON；allocator 的内部 estimate 不再能单独关闭本 Week。

### 2026-07-30：0.87 最终关闭

最终 fraction `0.87` 在不牺牲稳态性能的情况下，把 200 ms 连续物理
最低 free 提升到 2,301 MiB。八次短请求、左补齐 parity、3,584+512
整窗、96/96 CUDA parity、参数驻留、allocator drift、内部/外部显存和
性能门槛全部通过；最终 JSON 明确记录 `closed=true`。

## Close 回顾

Week 20 消除了 8B XLA 的关键容量问题：不是再压低模型精度，而是让
safetensors loader 直接产出 kernel 所需的唯一 compact tree，并只上传
一次。Qwen3-8B BF16 因此在同一张 RTX 4090 D 上保留 Week 19 的完整
4K 日常窗口，同时把长 prefill 提升约 31×、decode 提升约 4×，并以
96/96 独立 CUDA token parity 和 2,301 MiB 连续物理最低余量关闭。

最重要的工程结论是：XLA allocator 的逻辑余量不等价于整机物理余量。
0.95 和 0.89 都能完成计算，却不满足日常安全门槛；只有把外部连续采样
变成 Close 条件，`0.87` 才成为可复现的部署默认。下一步若继续优化，应
优先处理 persistent executable cache、单次 device decode loop 与
device-side sampling，而不是扩大 4K context 或重新引入第二棵参数树。
