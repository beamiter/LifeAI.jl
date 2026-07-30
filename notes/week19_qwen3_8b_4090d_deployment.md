# Week 19 — Qwen3-8B / RTX 4090 D 日常本地部署

> 状态：Closed（2026-07-30）

> 依赖基线：Week 17/18 已证明 Qwen3-14B mixed RTN 能在同一张
> RTX 4090 D 上生成，但仅余约 0.91 GiB，且只有 0.377 tok/s。本 Week
> 不把“刚好装下”误写成“适合日常使用”。

## 核心问题

能否在 24 GB 级 RTX 4090 D 上，用 LifeAI 把 Qwen3-8B BF16 从一次性
验证脚本升级成模型只加载一次、具有明确总 context、分块 prefill、固定
KV cache、EOS、采样和多轮历史裁剪的日常本地运行入口？

## 尺寸结论

“最大能运行”和“推荐日常部署”分开回答：

| 选择 | 权重 / 已测显存 | context 余量 | 速度 / 保真 | 定位 |
| --- | ---: | ---: | --- | --- |
| 14B 全 INT8 | tree 14.487 GiB；总 VRAM 23.453 GiB | 只余约 54 MiB | 0.816 tok/s；16/16 | 不安全，不部署 |
| 14B mixed RTN | tree 12.093 GiB；总 VRAM 22.597 GiB | 只余约 0.91 GiB | 0.377 tok/s；16/16 | 当前 LifeAI 已实证尺寸上限 |
| 8B BF16 | GPU 参数 15.281 GiB；4K KV 576 MiB | 3584 prefill 峰值仍余 2.05 GiB | 无量化误差；steady 11.78 tok/s | **日常默认** |
| 4B BF16 | tree 7.492 GiB | 很充足 | 质量低于 8B | XLA / 低延迟备选 |

Qwen3-30B-A3B 是 MoE；“A3B”表示每 token 的活跃参数，不表示只需驻留
3B 权重。LifeAI 当前没有 `qwen3_moe` loader、router/expert dispatch 或
MoE forward，因此不能部署。Qwen3-32B dense 的 INT4 tensor tree 虽有
纸面驻留空间，当前量化线性层会逐层反量化回 BF16，且尚无 4090 D
多-token 成功证据，也不能宣称可部署。

## 冻结的日常 profile

冻结配置：
[`configs/deployment/qwen3_8b_4090d_bf16_daily.json`](../configs/deployment/qwen3_8b_4090d_bf16_daily.json)

- 模型：`Qwen/Qwen3-8B`
- revision：`b968826d9c46dd6066d109eabc6255188de91218`
- dtype：BF16 权重 + BF16 KV，batch 1
- 总 context：4,096 tokens
- 默认最大 prompt/history：3,584 tokens
- 默认最大输出：512 tokens
- prefill chunk：64 tokens
- CUDA pool 回收：每个 prefill chunk 先显式 full GC 再调用
  `CUDA.reclaim()`；每 8 个 decode tokens 先增量 GC 再调用
  `CUDA.reclaim()`，请求结束采用前者。当前 CUDA.jl 的
  `CUDA.reclaim()` 本身还会执行 full GC、同步、清理 task-local library
  state 并 trim pool；这里的“full/增量”只描述它之前的显式 GC pass
- 默认模式：non-thinking，`temperature=0.7 / top_k=20 / top_p=0.8`
- profile SHA256：
  `93e7bde699fad4f0c93153e8d1c0458326c1ba848127cc14758fff066d944e4b`
- frozen asset manifest SHA256：
  `f4737c1aca92b3cbf046da7861af88fc2d4650552397b7d6f4b7edade5040e91`
- 启动前 free-memory 门禁：BF16 参数 + 完整静态 KV + 5 GiB
  allocator/GEMM/attention workspace reserve

这里的 4K 是 LifeAI 运行时总预算，不是模型原生能力声明。prompt、chat
template 和输出共同占用它。选择 4K 而不是直接承诺 32K，是因为当前
attention 仍非 FlashAttention；长 prefill 的临时张量和时间成本比 KV
本身更早成为日常使用瓶颈。

## Context 预算

BF16 KV cache 的逻辑 tensor bytes 为：

```text
layers × 2(K/V) × head_dim × kv_heads × tokens × batch × 2 bytes
```

Qwen3-8B 是 36 层、8 个 KV heads、head dim 128，即每 token 144 KiB：

| 总 context | BF16 KV |
| ---: | ---: |
| 1,024 | 144 MiB |
| 2,048 | 288 MiB |
| 4,096 | 576 MiB |
| 8,192 | 1,152 MiB |

这些值不包含 CUDA allocator、attention/GEMM workspace 和权重。公共
`qwen3_kv_cache_bytes` 只报告可精确计算的逻辑 KV bytes，硬件验收另记
实际 free/used VRAM。

## 已实现

- `Qwen3DeploymentProfile`：严格解析配置；未知字段、moving revision、
  context 溢出均 fail closed。
- 启动前流式校验 config/tokenizer/index 和五个权重分片的 frozen
  size/SHA256；ModelScope `master` 不能绕过 immutable HF revision。
- `HFQwen3BF16Session`：模型、RoPE 与固定容量 K/V 只创建一次，请求间
  只 reset 逻辑位置。
- 通用 session 构造器仍以 chunk 128 为设备无关默认；4090D 日常 CLI 和
  benchmark 必须显式读取本 profile 的 chunk 64，不能绕过 profile。
- 部署 prefill 不再返回 embedding / 每层 residual trace；中间 chunk
  完全跳过 final norm 和 vocabulary projection，只在整个 prompt 的最后
  一个 token 计算首 token logits。
- 64-token 分块 prefill，把一次性平方级 attention workspace 限定在
  `chunk × current_prefix`，同时保持因果语义。
- EOS 提前停止、greedy 与 temperature/top-k/top-p sampling。
- chat template、thinking 开关和多轮 history；超预算时保留 system 与
  最新请求，按最老 user/assistant turn pair 裁剪。
- 单 prompt / 交互 CLI：

```bash
julia --project=. --startup-file=no \
  scripts/run_qwen3_cuda_chat.jl \
  /home/ubuntu/models/modelscope/Qwen/Qwen3-8B
```

一次性确定性 smoke：

```bash
julia --project=. --startup-file=no \
  scripts/run_qwen3_cuda_chat.jl \
  /home/ubuntu/models/modelscope/Qwen/Qwen3-8B \
  --prompt "用三点解释 KV Cache" --greedy --max-new-tokens 64
```

生成 token 会按原始 tokenizer bytes 流式写出。交互模式支持 `/clear` 和
`/exit`；`--system TEXT` 固定 system message，`--seed N` 可重放采样；
`--thinking` 显式打开思考，默认关闭，避免日常短测试的思考 token 吃掉
context 与等待时间。

## 验证结果

### 离线

- Week 19 专项：`80 / 80`。
- 默认完整套件：`5,275 / 5,275`。
- 分块 prefill 的末位置 logits、下一 token decode logits、4-step greedy
  与既有 BF16 accel 路径逐值相同。
- cache reset 后重复请求确定；EOS、overflow、history compaction 均覆盖。
- Week 15 / Week 16 回归：`82 / 82`、`168 / 168`。

### RTX 4090 D

冻结权重的 10 个必需文件（16,392,983,007 bytes）全部通过 size/SHA256
校验。最终 benchmark 采用 200 ms `nvidia-smi` 外部采样，结果为：

| 项目 | 结果 |
| --- | ---: |
| asset 校验 | 33.606 s |
| host BF16 load | 106.214 s |
| GPU upload / session init | 1.976 s / 1.561 s |
| ready free VRAM | 6.914 GiB |
| 1,024-token prefill | 12.815 s，79.91 tok/s |
| 2,048-token prefill | 26.206 s，78.15 tok/s |
| 3,584-token prefill | 46.849 s，76.50 tok/s |
| 3,584 + 512 整窗 | sequence 4,096；cache 4,095；96.949 s wall |
| 整窗 decode | 10.246 tok/s |
| steady 26-token TTFT / decode | 0.765 s / 11.776 tok/s |
| 默认采样 TTFT / decode | 1.588 s / 11.388 tok/s |
| 最低 CUDA free / 物理 free | 2.190 GiB / 2,099 MiB |
| 整窗 decode 最低 CUDA free | 4.190 GiB |
| 请求结束 free VRAM | 6.887 GiB |
| 进程最大 host RSS | 19.89 GiB |

默认采样的真实输出开头为：

```text
KV Cache（Key-Value Cache）是自回归模型（如Transformer）在进行解码时用于加速推理的一个关键优化技术。它之所以能
```

本地原始结果：
`benchmark_results/week19/qwen3_8b_4090d_bf16_daily.json`，SHA256
`5c01dd5e218167778255b71f2b1053e153cbef279b60806e14ab5911fe2fa0c2`。
2,856 个外部显存样本保存在同目录
`qwen3_8b_4090d_bf16_daily_nvidia_smi.csv`，SHA256
`018d873a77e6d19ca6727fb9b2fdd00801e5cd4a6b6de10ddb0ec75be530555d`。

## Close 条件

- [x] 冻结 revision 的 Qwen3-8B 五个分片 checksum 全部匹配。
- [x] RTX 4090 D 能一次加载并连续处理至少三次请求。
- [x] 3,584-token prompt + 512-token 输出禁用 EOS 后完整实跑，生成 512
  tokens、最终缓存 4,095 tokens（最后选中的 token 无需写回 cache）。
- [x] steady 短请求 TTFT 不高于 5 s、decode 不低于 3 tok/s；3,584-token
  steady prefill 不高于 120 s，否则下调默认 context 或改称容量 profile。
- [x] ready、3,584-token prefill 峰值与满预算 decode 均至少保留 1 GiB
  `nvidia-smi` 实测 free VRAM；CUDA 内部门槛设为 2 GiB，给两种口径
  的 allocator/driver 差异留余量。
- [x] 记录 load、prefill、decode tok/s、总/空闲显存和模型输出。
- [x] Week 19 专项及默认测试全部通过。

## 风险与取舍

- 这条路径是 CUDA eager，不是 XLA。当前 8B packed-XLA 会同时驻留原树
  与 packed tree，未计 KV 已超过 4090 D 可用显存；要上 8B XLA，先做
  compact tree 单次 transfer。若本周必须交付 XLA，合理尺寸是 4B。
- 固定 cache 解决反复 `cat` 和 allocator 碎片；它不把普通 attention
  变成 FlashAttention。
- 全量 BF16 shard 在 host 侧组装后再上传，启动期间建议至少保留
  32 GiB 可用系统内存；上传完成后 benchmark 会释放 host 参数树。
- 4K 是默认日常边界。8K 只有在峰值显存和 TTFT 实测后才可作为 stretch，
  不能仅凭 1.125 GiB KV 预算开放。

## 实验与过程记录

### 2026-07-30：Open 与部署路径落地

- GPU：NVIDIA GeForce RTX 4090 D，24,564 MiB，总空闲 23,668 MiB，
  driver 570.153.02。
- 明确 14B mixed RTN 是容量上限而非日常选择；选择 8B BF16 保留模型
  质量与 CUDA tensor-core 路径。
- 删除部署路径中的 full prompt vocabulary projection、完整逐层 trace 和
  动态 cache concatenate；新增静态 cache 与分块 prefill。
- 用本机 frozen Qwen3-0.6B 对同一 CUDA session 做真权重 smoke：
  18-token chat prompt、16-token greedy 正常生成；cold prefill
  `15.473 s`（含首次 CUDA kernel 初始化），decode `11.62 tok/s`，证明
  CuArray prefix view、静态 cache 写入与文本边界可运行。它不替代 8B
  显存/context 验收。
- 跳过中间 chunk vocabulary projection 后，以 `chunk=4` 再跑真实
  0.6B CUDA 多 chunk smoke：6-token prompt + 16-token greedy 完成，
  cache 为 21 tokens（prompt + output - 1），cold prefill `15.950 s`，
  decode `13.59 tok/s`。
- frozen Qwen3-8B 从 ModelScope 镜像下载后，以 Week 13 的 immutable
  Hugging Face revision checksum 契约校验，10/10 文件一致。
- 第一轮 4K 不回收 CUDA pool，3,584 prefill 仅 `2.417 s`、整窗
  decode `18.68 tok/s`，但最低 free 只约 51 MiB，不能作为日常配置。
- 每 4 chunks / 32 decode tokens 回收后仍只有 480 MiB 物理余量；
  chunk 64 + 每 chunk 增量 GC 也只有 1,301 MiB。
- 最终用 chunk 64；prefill 每 chunk 先做显式 full GC 再调用
  `CUDA.reclaim()`，decode 每 8 tokens 先做增量 GC 再调用同一个完整
  reclaim。物理最低 free 提升到 2,099 MiB，代价是 3,584 prefill
  增至 `46.849 s`。该取舍通过全部日常门槛。
- 最终用户入口再做 18-token prompt + 16-token greedy CLI smoke：
  正常输出“`KV Cache 是在大语言模型推理过程中，用于存储已生成 token 的`”，
  cold prefill `16.745 s`、decode `6.54 tok/s`、请求后 free `6.86 GiB`。
- 首次手工交互暴露顶层 loop 的 `history = convert(...)` soft-scope
  歧义；改为 `empty!` / `append!` 原地更新，并处理 stdin EOF。真实
  `/clear → hi → /exit` 回归无 warning/exception，输出正常；对应 3 项
  自动检查使 Week 19 专项增至 `80 / 80`。

## Close 回顾

4090D 的“最大能跑”仍是 14B mixed RTN，但它没有足够上下文余量和交互
速度；真正适合日常验证的是 Qwen3-8B BF16。Week 19 把它冻结为 4K 总
预算，即最多 3,584-token prompt/history 加 512-token output，并完成：

- exact-asset 启动门禁、常驻 session、静态 KV、分块 last-logit prefill；
- EOS、greedy/采样、thinking 开关、多轮历史裁剪和流式 CLI；
- 3,584+512 整窗、连续请求、内部/外部双口径显存和真实文本验收。

4K 不是 Qwen3-8B 的原生 context 上限，而是当前 LifeAI 普通 attention
在 24 GB 卡上的日常工程边界。8K 暂不开放；后续若要提升 context 或
prefill 速度，优先级是 fused/FlashAttention 或 XLA compact-tree
transfer，而不是继续压缩模型质量。
