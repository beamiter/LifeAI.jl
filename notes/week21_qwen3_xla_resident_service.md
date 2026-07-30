# Week 21 — Qwen3-8B XLA 常驻本地推理服务

> 状态：Open（2026-07-30）

> 依赖基线：[`Week 20 — Qwen3-8B XLA 单驻留 4K 日常部署`](week20_qwen3_8b_xla_deployment.md)
> 已完成单进程内的 compiled session 复用，但每次重新运行 CLI 仍需重新读取
> 15.27 GiB 权重、上传参数并编译 prefill/decode executable。

## 核心问题

能否把 Week 20 的唯一 XLA session 变成只监听本机回环地址的常驻服务，
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
应用 executable 可直接恢复。Week 21 因而不提交这个实验性 preference，
也不把“缓存已生成”误写成冷启动已经解决。

## 计划能力

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
6. 默认完整测试、Week 21 专项与 opt-in XLA smoke 全部通过；使用方法、
   实测边界和 persistent cache 的负结果进入文档后关闭本 Week。

