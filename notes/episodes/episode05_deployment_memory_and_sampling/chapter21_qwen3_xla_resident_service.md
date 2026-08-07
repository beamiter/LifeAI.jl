# Chapter 21 — Qwen3-8B XLA 常驻本地推理服务

> 状态：Closed（2026-07-30）

> 依赖基线：[`Chapter 20 — Qwen3-8B XLA 单驻留 4K 日常部署`](../../episodes/episode05_deployment_memory_and_sampling/chapter20_qwen3_8b_xla_deployment.md)
> 已完成单进程内的 compiled session 复用，但每次重新运行 CLI 仍需重新读取
> 15.27 GiB 权重、上传参数并编译 prefill/decode executable。

## 核心问题

能否把 Chapter 20 的唯一 XLA session 变成只监听本机回环地址的常驻服务，
让多个独立客户端请求复用同一棵参数树、同一组 compiled executable 与静态
KV cache，同时守住请求串行化、4K context 门禁、流式 UTF-8 输出、
CUDA BF16 token parity、吞吐和 2 GiB 物理显存余量？

## 为什么不把跨进程 kernel cache 当作主方案

本机 Reactant `0.2.275` 的公开 persistent compile cache 只向 XLA 配置：

- `xla_gpu_kernel_cache_file`；
- per-fusion autotune cache。

没有公开、稳定的 Julia API 可以序列化并在下一进程恢复完整 PJRT
executable。临时启用 persistent kernel cache 后，缩小 GPU compiled
prefill 的两个独立进程仍分别约 55 秒；kernel cache 文件生成并不等于
应用 executable 可直接恢复。Chapter 21 因而不提交这个实验性 preference，
也不把“缓存已生成”误写成冷启动已经解决。

## 已实现能力

- 单进程只加载一次 Qwen3-8B XLA session，后续 HTTP 请求不再加载或编译。
- 默认 `127.0.0.1:11435`，避免与 Ollama 默认 `11434` 冲突；非回环绑定需
  显式 opt-in。
- `GET /healthz` 提供 ready、加载次数、请求计数和运行中/排队状态。
- `POST /api/generate` 提供严格的 Ollama-compatible 子集，支持 NDJSON
  token streaming 与非流式 JSON，方便复用既有端到端 benchmark client。
- 固定 greedy、batch 1、4K profile；拒绝错误 model、超限 prompt/output、
  不支持的 options 和过大 body，不静默改变请求。
- 用单把锁保护会原地写入的静态 KV cache；并发请求排队但输出不串线，
  请求异常后锁仍可继续服务。
- 提供日常 server launcher、最小客户端命令和可复现实机验收脚本。

## 服务边界与协议

`Qwen3XLAHTTPService` 在构造时调用 loader **恰好一次**，保存返回的
`HFQwen3BF16XLASession`，此后所有请求共享同一参数树、prefill/decode
executable、RoPE 表和静态 KV cache。协议层只依赖四个可注入 callable：
loader、prompt encoder、generator 和 token decoder，因此默认测试无需读取
真实权重或初始化 Reactant GPU。

静态 KV 会原地更新，generation 由一把 `ReentrantLock` single-flight
保护；独立 metrics lock 记录 total/completed/failed/queued/active 与
`max_active`。请求抛错和客户端写失败都通过 `finally` 释放 active 状态及
generation lock，下一请求可以继续执行。

端点是有意收窄的 Ollama-compatible 子集：

- `GET /healthz`：返回 ready、model、context、`load_count` 与请求计数；
- `POST /api/generate`：要求 exact model、`raw=true`、greedy
  `temperature=0`、`repeat_penalty=1`、`repeat_last_n=0` 和 exact
  `num_ctx=4096`；
- 同时支持 buffered JSON 与逐 token NDJSON；
- 拒绝未知字段、sampling、错误 model/media type、空 prompt、超过 1 MiB
  body，以及 padded prompt + output 超过 4K 的请求；
- token 可能只含 UTF-8 多字节序列的一部分，streamer 会累积到完整有效
  UTF-8 后才 flush，不会向客户端写破碎字符。

公共 `serve_qwen3_xla_http!` 默认 `127.0.0.1:11435`。日常 launcher 对
非 loopback 地址再加一道 `--allow-non-loopback` 门禁；这个开关只代表调用
者明确承担网络暴露，不提供 TLS、认证或多租户隔离。

## 日常使用

终端一启动一次常驻服务：

```bash
julia --project=. --startup-file=no \
  scripts/run_qwen3_xla_server.jl \
  /home/ubuntu/models/modelscope/Qwen/Qwen3-8B
```

看到 `ready http://127.0.0.1:11435 load_count=1` 后，在其他终端发请求：

```bash
julia --project=. --startup-file=no \
  scripts/qwen3_xla_client.jl \
  /home/ubuntu/models/modelscope/Qwen/Qwen3-8B \
  --prompt "用三点解释 KV Cache" \
  --max-new-tokens 128
```

client 只读取 tokenizer 资产并在本地套 Qwen3 chat template，不读取 8B
权重。健康检查：

```bash
curl -s http://127.0.0.1:11435/healthz
```

既有 Ollama benchmark client 的 payload 与本端点严格子集一致，可把 URL
改到 LifeAI 服务而无需改 benchmark 协议：

```bash
python3 scripts/benchmark_qwen3_e2e_ollama.py \
  --url http://127.0.0.1:11435/api/generate \
  --model Qwen/Qwen3-8B \
  --prompt "<已经套好 template 的 raw prompt>" \
  --num-predict 128 --num-ctx 4096
```

## RTX 4090 D 最终验收

验收脚本只加载一个 service PID，并令所有 HTTP 请求带
`Connection: close`，保证 10 次复用是独立 TCP connection：

```bash
julia --threads=auto --project=. --startup-file=no \
  scripts/benchmark_qwen3_xla_service.jl \
  /home/ubuntu/models/modelscope/Qwen/Qwen3-8B \
  configs/deployment/qwen3_8b_4090d_bf16_xla_daily.json \
  benchmark_results/week20/qwen3_8b_cuda_greedy_reference.json \
  benchmark_results/week21/qwen3_8b_4090d_bf16_xla_service.json
```

冷启动仍然存在，但只支付一次：

| 阶段 | 秒 |
| --- | ---: |
| 10-file frozen asset SHA256 | 33.053 |
| compact host load | 104.982 |
| 唯一参数树 transfer | 2.880 |
| KV / runtime allocation | 0.288 |
| prefill compile | 70.075 |
| decode compile | 25.382 |
| service constructor | 205.320 |
| asset check 至首个 health response | 242.386 |

启动后不再发生 load/compile。最终 `/healthz` 为 `load_count=1`、
`15 completed / 0 failed`、`active=queued=0`、`max_active=1`，session
identity 在全部请求前后相同。

真实请求结果：

| 项目 | 结果 |
| --- | ---: |
| 9 个 steady 64-token prefill 中位数 | 0.02728 s |
| steady decode 中位数 | 41.351 tok/s |
| 3,584-token parity prefill | 1.4860 s |
| 3,584+512 prefill | 1.4828 s |
| 3,584+512 decode | 41.351 tok/s |
| full-window service / client wall | 13.841 / 13.935 s |
| reusable-request allocator drift | 167,680 bytes |
| 200 ms physical samples | 1,228 |
| 最低物理 free | 2,416,967,680 bytes（2.251 GiB） |

64、65（bucket 128）与 3,584-token 三组 frozen CUDA BF16 oracle 仍为
**96 / 96 tokens**；整窗前 32 token 也一致，最终生成 512 token、
cache position 4,095。

两个客户端同时发起请求时，HTTP task dispatch 在第二个 handler 进入
generation lock 前已经串行化，因此 lock queue 指标只有微秒级，不能拿
“lock 上等待至少 10 ms”当硬件判据。原始记录显示两次服务计算合计
1.5631 s、pair wall 1.6398 s，比值 1.049；结合 `max_active=1` 和离线
direct-handler 并发测试，证明没有重叠写静态 KV。报告以
`acceptance_revision=2` 明确冻结这个修正判据，原始请求耗时、输出和显存
样本均未更改。

最终证据：

```text
benchmark_results/week21/qwen3_8b_4090d_bf16_xla_service.json
SHA256 73c1bf7a0a7dbabbd2ee1b4ae246022e05d2296e11d8e1d665624fb4cc4b6152

benchmark_results/week21/qwen3_8b_4090d_bf16_xla_service_nvidia_smi.csv
SHA256 e006940214ecabb3802dda178faaad994491cfeae2fc2cfd3425a0d71c2d960b
```

## 测试与关闭

- Chapter 21 专项：`109 / 109`；
- 加真实 loopback socket opt-in：`116 / 116`；
- 默认完整套件：`5,489 / 5,489`；
- Reactant CPU compiled prefill smoke：`5 / 5`；
- 真机 acceptance：所有 `_passed` 字段为 true，顶层 `closed=true`。

## Close 条件

1. 离线测试覆盖协议 schema、路由/方法、body/context 门禁、stream 与
   non-stream、loader 只执行一次、并发 single-flight、异常后的锁释放。
2. 一个 server PID 接受至少 10 个跨连接请求，`load_count = 1` 且
   compiled session 身份不变。
3. 64、65、3,584-token 三组冻结 CUDA BF16 oracle 仍为 `96 / 96` token
   完全一致；3,584+512 整窗仍成功。
4. 短请求 steady decode 不低于 35 tok/s，最大 prompt prefill 不高于
   2 秒，整窗 decode 不低于 35 tok/s。
5. 连续 200 ms `nvidia-smi` 采样的最低物理空闲显存不少于 2 GiB；
   多请求后 `/healthz` 仍 ready，服务内 allocator 没有无界漂移。
6. 默认完整测试、Chapter 21 专项与 opt-in XLA smoke 全部通过；使用方法、
   实测边界和 persistent cache 的负结果进入文档后关闭本 Chapter。

以上六项全部满足，Chapter 21 于 2026-07-30 Closed。这里解决的是“常驻
进程后的重复请求冷启动”，不是跨进程 executable 恢复；server 退出后下次
启动仍需重新读取权重和编译。后续若继续做 XLA 性能，优先候选是经完整
EOS/streaming/alias 验证的固定容量 device-side decode block，而不是把
kernel cache 文件包装成完整 executable cache。
