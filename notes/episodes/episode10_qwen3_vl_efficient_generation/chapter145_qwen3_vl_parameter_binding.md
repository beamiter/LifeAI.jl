# Chapter 145 — Qwen3-VL 静态 decode 的完整参数绑定

> 所属 Episode：Episode 10 — Qwen3-VL 高效生成
>
> 状态：Closed
>
> 开启：2026-10-13
>
> 关闭：2026-10-13

## 问题与复现

静态 cache 的 pack 和 Ready 只用 `embedding ===` 判断能否复用。保留同一
embedding、替换 `blocks`、`final_norm` 或 text spec 后，prefill 会读取新参数，
decode 却可能继续读取旧参数。`reset_qwen3_vl_static_kv_cache!` 保留 Ready，
因此 reset 后重新 prefill 也会触发。

CPU BF16 tiny 模型已复现：把 `final_norm` 换成全零数组，reset 并重新 prefill
后，新建 cache 的 logits 全零，旧 cache 的 logits 最大绝对值仍是
`0.20605469`。这不是 BF16 与 Float32 的数值差异，而是旧参数被错误复用。

## 修改

- pack 和 Ready 同时核对 embedding、blocks、final_norm 和 spec 的身份。
- 参数绑定变化时，在输入验证之后重建 pack 与派生 scratch/workspace；K/V
  storage 保留。相同参数的普通 decode 与 reset/re-prefill 仍保留 Ready 和
  scratch identity。
- CUDA 的入口、全局 workspace shell 和 Ready 复用也采用相同绑定条件，
  防止新 pack 被旧全局状态覆盖。长生成计时入口在安装 workspace 前同步绑定。

这里检查的是参数来源，不扫描权重内容。它不会修复调用方在非空 cache 上
原地修改权重后留下的旧 K/V；切换权重的请求仍应 reset 并重新 prefill。

## 验证与边界

新增离线测试共 `300/300` 通过。三种替换分别覆盖尚未安装 Ready 的 cache，
以及已有 Ready、reset 后重新 prefill 的 cache。每例比较两步 decode logits
与所有层有效 K/V，结果与新建 cache exact；全零 final_norm 输出全零。
相同参数保留 Ready/scratch，无效 context spec 在修改 K/V、position、
rope_delta 或 Ready 前报错。

关联回归也已通过：Episode 09 为 `1,878` 项通过，真实 2B checkpoint 因未设置
本地模型路径而跳过 `1` 项；Episode 10 为 `1,389/1,389`，包含上述 `300` 项。
具体类型、预热后的 CPU 参数绑定检查为 `0` allocated bytes。
全库既有测试另有 `14,711` 项通过、同一项真实 2B checkpoint 跳过；该次运行在
新用例接入默认 runner 前启动，新增 `300` 项由上述完整 Episode 09/10 专测覆盖。

整库命令为 `OPENBLAS_NUM_THREADS=1 julia --compiled-modules=existing
--startup-file=no --project=. test/runtests.jl`。本环境的 Julia depot 只读，使用
`--compiled-modules=existing` 复用可用缓存，其余模块在内存中加载。

测试入口：
`test/episodes/episode10_qwen3_vl_efficient_generation/chapter145_qwen3_vl_parameter_binding/test_qwen3_vl_parameter_binding.jl`，
已纳入默认 `test/runtests.jl`，复用 Chapter 46/47 的 tiny fixture。

2026-10-13 已在沙箱外的 RTX 4090 D 上补齐验证。独立 CUDA 参数绑定脚本
通过 `456/456`；真实 2B 的一次 repeat、32/128/256-token 剖面 SHA256 为
`8963edea2ca1a8d20cb9a2401affdfa80768ce76e27da8f73f67c3ba99b87fb9`。
三个长度的稳态 decode 都保持 CPU/GPU 零分配，首次 decode 是 `10,816`
CPU bytes、`0` GPU bytes。32-token 与 HF oracle exact，128/256 仍在原来的
step 70 分叉，margin `0.02797699`。这是记录分叉的 allocation profile，
`closed=false`、`acceptance_fixture=false`，不声称 BF16 strict parity。

随后 Chapter 146 继续修复跨 cache 工作区共享；以上分配数据属于它之前的
参数绑定版本。Episode 10 保持 Open。
