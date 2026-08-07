# Episode 05 — 部署、记忆与设备采样

> 状态：Closed
>
> 收录章节：Chapter 19–23

这一卷把模型能力推进到可日常使用的本地系统：8B CUDA/XLA 入口、常驻 HTTP 服务、最小语义记忆，以及编译 executable 内的完整采样策略。

## 章节目录

1. [`Chapter 19 — Qwen3-8B / RTX 4090 D 日常本地部署`](chapter19_qwen3_8b_4090d_deployment.md)
2. [`Chapter 20 — Qwen3-8B XLA 单驻留 4K 日常部署`](chapter20_qwen3_8b_xla_deployment.md)
3. [`Chapter 21 — Qwen3-8B XLA 常驻本地推理服务`](chapter21_qwen3_xla_resident_service.md)
4. [`Chapter 22 — Qwen3-Embedding-0.6B 与最小语义记忆`](chapter22_qwen3_embedding_memory.md)
5. [`Chapter 23 — Qwen3 XLA 设备端采样`](chapter23_qwen3_xla_device_sampling.md)

## 本卷结果

Qwen3-8B 形成 4K XLA single-residency 与 loopback resident service，Qwen3-Embedding-0.6B 提供最小 dense exact semantic memory，设备采样达到 237.23 tok/s 并与宿主策略逐 token 一致。
