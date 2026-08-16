# Episode 07 — 智能体闭环

> 状态：Open
>
> 收录章节：Chapter 36–

这一卷开始兑现 Milestone C：把已经完成 parity 的 Qwen3 从“能正确生成 token”推进到“能在多个 step 之间保持状态、按环境反馈改变下一步动作”。第一步不是设计一个大而全的 agent 框架，而是先把与外部世界交换信息的**协议**做到与官方实现逐字节一致——工具声明、工具调用、工具返回都必须落在 HuggingFace 官方 chat template 的语义里，否则后续所有关于“模型好不好用”的测量都建立在一个已经偏离参照系的 prompt 上。

本卷同时改变了验证的参照物。Episode 02—06 的参照物是“LifeAI 与 HuggingFace 的数值一致”；本卷的第一个参照物是**外部独立实现**：官方 Jinja 模板经 CPython + Jinja2 渲染的结果，与 LifeAI 的 Julia 渲染逐字节比较。

## 章节目录

1. [`Chapter 36 — Qwen3 tools chat template HF parity 与 4B 工具调用闭环`](chapter36_qwen3_tools_chat_template.md)（Closed）

## 预期能力变化

- **模型基本组件**：不新增算子。本卷复用已冻结的 dense BF16 推理路径。
- **智能体核心**：从“无状态单轮 prompt”推进到“工具声明 → 工具调用 → 工具执行 → 结果回填 → 再决策”的多 step 闭环，并要求整条轨迹可 replay、可离线测试。
- **HuggingFace 互操作**：chat template 从 Chapter 08 的 system/user/assistant 子集扩展到官方模板全部分支，包含 `# Tools` 头、`<tool_call>`、`tool` 角色的 `<tool_response>` 合并，以及 `tojson` 的 CPython 语义。
- **工程与测试**：引入不依赖 torch/transformers 的 Python oracle（只需 jinja2 + markupsafe），并把渲染 parity 做成**离线默认测试**——冻结官方模板原文与 reference 输出，纯字符串比较，不需要真实权重。

## Episode Close 条件

- 官方 chat template 的全部分支完成逐字节 parity，并有外部实现作为参照。（Chapter 36 已完成）
- 至少一种真实权重能在多个 step 之间保持状态，并根据工具返回改变下一步动作，轨迹可复现。（Chapter 36 已完成单请求内闭环）
- 跨请求的持久状态与记忆策略、检索接入、以及“任务成功率”而非“协议合法率”的质量测量。（未开始）

## 与 Episode 06 的关系

Episode 06 仍为 Open，但它的剩余工作项（grouped scattered dispatch、full-window 40K prefill、剩余 allocation profiling）都需要 Qwen3-30B-A3B 的 61 GB checkpoint 与 RTX 4090 D 24 GiB；当前工作机是 RTX 5080 16 GiB 且本地没有该 checkpoint。Episode 07 不接管这些工作项，回到那台机器时 Episode 06 原样继续。
