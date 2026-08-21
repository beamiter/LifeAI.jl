# Episode 07 — 智能体闭环

> 状态：Closed
>
> 收录章节：Chapter 36–39

这一卷开始兑现 Milestone C：把已经完成 parity 的 Qwen3 从“能正确生成 token”推进到“能在多个 step 之间保持状态、按环境反馈改变下一步动作”。第一步不是设计一个大而全的 agent 框架，而是先把与外部世界交换信息的**协议**做到与官方实现逐字节一致——工具声明、工具调用、工具返回都必须落在 HuggingFace 官方 chat template 的语义里，否则后续所有关于“模型好不好用”的测量都建立在一个已经偏离参照系的 prompt 上。

本卷同时改变了验证的参照物。Episode 02—06 的参照物是“LifeAI 与 HuggingFace 的数值一致”；本卷的第一个参照物是**外部独立实现**：官方 Jinja 模板经 CPython + Jinja2 渲染的结果，与 LifeAI 的 Julia 渲染逐字节比较。

## 章节目录

1. [`Chapter 36 — Qwen3 tools chat template HF parity 与 4B 工具调用闭环`](chapter36_qwen3_tools_chat_template.md)（Closed）
2. [`Chapter 37 — Qwen3 dense 任务质量基线`](chapter37_qwen3_task_quality.md)（Closed）
3. [`Chapter 38 — 工具到底帮不帮得上忙`](chapter38_qwen3_tool_task_success.md)（Closed）
4. [`Chapter 39 — 跨请求、可回放的语义记忆闭环`](chapter39_persistent_semantic_memory.md)（Closed）

## 预期能力变化

- **模型基本组件**：不新增算子。本卷复用已冻结的 dense BF16 推理路径。
- **智能体核心**：从“无状态单轮 prompt”推进到“工具声明 → 工具调用 → 工具执行 → 结果回填 → 再决策”的多 step 闭环，并要求整条轨迹可 replay、可离线测试。
- **HuggingFace 互操作**：chat template 从 Chapter 08 的 system/user/assistant 子集扩展到官方模板全部分支，包含 `# Tools` 头、`<tool_call>`、`tool` 角色的 `<tool_response>` 合并，以及 `tojson` 的 CPython 语义。
- **工程与测试**：引入不依赖 torch/transformers 的 Python oracle（只需 jinja2 + markupsafe），并把渲染 parity 做成**离线默认测试**——冻结官方模板原文与 reference 输出，纯字符串比较，不需要真实权重。

## Episode Close 条件

- 官方 chat template 的全部分支完成逐字节 parity，并有外部实现作为参照。（Chapter 36 已完成）
- 至少一种真实权重能在多个 step 之间保持状态，并根据工具返回改变下一步动作，轨迹可复现。（Chapter 36 已完成单请求内闭环）
- “任务成功率”而非“协议合法率”的质量测量。（Chapter 38 已完成工具闭环测量）
- 跨请求的持久状态与记忆策略、检索接入，并用 no-memory / retrieved / 等 token 干扰三臂证明
  收益来自相关内容。（Chapter 39：retrieval `12/12`；三臂 `0/12 → 12/12 ← 0/12`，两组
  精确 McNemar p 均为 `0.00048828125`；36 / 36 轨迹跨进程 replay）

## Episode 回顾

Episode 07 已把 Qwen3 从单轮 token generation 推到两个层次的有状态闭环：单请求内能按工具结果继续
决策，跨请求能把 source facts 持久化、重新索引并注入后续请求。协议层以 CPython + Jinja2 的官方
模板渲染为独立 oracle；质量层没有把「格式合法」或「检索命中」冒充任务成功，而是分别测 GSM8K
工具三臂与 synthetic private-fact 记忆三臂。

本卷最重要的两条经验是：给 prompt 增加工具声明，即使模型不调用工具也可能改变 greedy 答案；给
prompt 增加记忆也必须设置等 token 的无关干扰臂。本卷因此把后续 agent 能力的验收标准固定为完整
trace/replay 加任务级因果对照，而不只看协议或中间指标。

下一卷可以在已验证的 memory/tool loop 上选择一个更完整的 policy 问题，例如自动写回与遗忘、
planning/反思，或 observation/action 仿真接口；Chapter 39 的 12 题结果不外推为完整长期记忆系统。

## 与 Episode 06 的关系

Episode 07 关闭时，Episode 06 的 grouped scattered 仍等待 61 GB checkpoint 与
RTX 4090 D；该历史依赖后来已在 Chapter 41 满足。pointer-backed grouped WMMA
完成真实 30B exact/traffic/performance gate 后，Episode 06 已 Closed；full-window
40K 与剩余 allocator attribution 被明确保留为独立专项，不再阻塞 agent/environment
主线。当前工作回到 Episode 08 的环境事件记忆写回。
