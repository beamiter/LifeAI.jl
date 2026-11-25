# Episode 08 — 环境与行动闭环

> 状态：Closed
>
> 收录章节：Chapter 40、42、150

## 本卷主题

Episode 07 已证明模型能在单请求内使用工具、跨请求检索记忆，但工具结果仍只是文本服务返回值。
本卷开始建立具身系统最小公共边界：环境产生结构化 observation，policy 只能提交 allowlisted action，
环境执行后返回新的 state/feedback；任务成功必须由环境终态判定，而不是模型的自然语言自述。

第一章刻意选择确定性 GridWorld。它不模拟真实机器人的动力学，但能把 observation/action、动作预算、
非法动作、终止状态、任务成功和跨进程 replay 变成严格可测的契约。Chapter 42 再把 clean successful
episode 经显式 policy 写入持久 journal，并用后续无反馈 episode 验收检索内容的因果收益；Chapter 150
建立不依赖 GridWorld 的跨 adapter 动作安全契约（超时、执行失败分类、急停与外部互锁、幂等去重与离线审计），
正式闭合本卷。

## 章节目录

1. [`Chapter 40 — 确定性 GridWorld observation/action 与真实反馈闭环`](chapter40_deterministic_gridworld.md)（Closed）
2. [`Chapter 42 — 显式环境事件记忆写回与因果验收`](chapter42_environment_event_memory.md)（Closed）
3. [`Chapter 150 — 跨 Adapter 动作安全契约`](chapter150_adapter_action_safety.md)（Closed）

## 预期能力变化

- **智能体核心**：policy 从调用无状态工具推进到读取环境 observation、提交 action、根据新状态再决策。
- **持久记忆**：clean successful episode 可经显式 policy 写为 canonical environment event；fresh-load
  后 exact-spec 检索并注入后续决策，完整 source binding 可 replay。
- **多模态与具身**：先建立与具体机器人无关的离散环境与跨 adapter 安全边界（超时、失败不确定性分类、急停、
  幂等动作与离线审计）；本卷自身不提供图像输入、动力学或实体设备控制，后续 Qwen3-VL 视觉能力由 Episode 09/10 记录。
- **工程与测试**：每个 state/transition/tool output/prompt 与安全审计记录都可独立重算；环境成功而非回答文本是主指标。

## Episode Close 条件

- [x] 通用 observation/action/transition 类型与一个确定性 simulator，动作白名单和预算 fail closed。
- [x] 真实 Qwen3 权重完成多 step 环境反馈闭环，并以任务终态、非法动作和路径效率验收。
- [x] scripted oracle 与反馈 withheld 对照；完整轨迹可在第二进程不加载模型地 replay。
- [x] 环境事件按显式 policy 写入 Chapter 39 memory journal，并在后续 episode 中检索使用。
- [x] 为超时、执行失败、急停和幂等 action 定义不依赖 GridWorld 的 adapter/safety 契约。

Chapter 40 已建立最小离散环境闭环；Chapter 42 已关闭环境事件写回、fresh-load exact-spec retrieval、
token-matched 三臂因果验收与 tokenizer-only replay；Chapter 150 已闭合不依赖 GridWorld 的跨 adapter
timeout、execution failure、e-stop 与 idempotent action 安全契约及离线审计校验。Episode 08 于 2026-11-25 Closed。
