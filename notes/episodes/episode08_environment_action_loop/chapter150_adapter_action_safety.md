# Chapter 150 — 跨 Adapter 动作安全契约（超时、执行失败、急停与幂等）

> 所属 Episode：Episode 08 — 环境与行动闭环
>
> 状态：Closed
>
> 日期：2026-11-25

## 本章问题

Chapter 40 建立了确定性 GridWorld 的 observation/action/transition 闭环，Chapter 42 闭合了 clean successful
环境事件的显式记忆写回与因果验收；但 Episode 08 仍留着最后一条 Close 条件：

> 为超时、执行失败、急停和幂等 action 定义不依赖 GridWorld 的 adapter/safety 契约。

真实外部模拟器与实体机器人控制面临四类纯内存 GridWorld 不会出现的风险：调用挂起超时（副作用不确定）、
执行失败（调用前失败 vs 下发后状态不确定）、硬件/操作员急停（e-stop 锁存与安全恢复），以及网络/agent
重试导致的重复下发（幂等与载荷冲突）。

## 实现范围

新增不依赖 GridWorld 的跨 adapter 安全层 [`src/agent/safety.jl`](../../../src/agent/safety.jl)，并在
[`src/LifeAI.jl`](../../../src/LifeAI.jl) 与 [`src/api.jl`](../../../src/api.jl) 导出公共契约：

- **`ActionSafetyPolicy`**：不可变且带规范 SHA-256 摘要的安全策略，严格校验 `default_timeout_seconds`、
  `max_timeout_seconds`、`max_consecutive_failures`、`estop_on_timeout`、`estop_on_uncertain_failure`、
  `require_idempotency_key` 与有界 `max_idempotency_entries`。拒绝 `Bool`、非有限值、非正数和越界配置。
- **`SafeActionRequest` & `AgentActionExecutionError`**：请求校验 `idempotency_key` 标识规范与可选超时覆盖，
  计算 `(action.name, action.value)` 的 `payload_sha256`；外部 adapter 可通过 `AgentActionExecutionError`
  显式区分 `:not_executed`（下发前失败）与 `:uncertain`（下发后状态不确定）；未分类异常默认按 `:uncertain`
  fail closed 处理。
- **`SafeEnvironmentAdapter{E<:AbstractAgentEnvironment}`**：可包裹任意 `AbstractAgentEnvironment`（包括
  独立于 GridWorld 的物理/模拟器 adapter 或 `GridWorldEnvironment`），通过 `dispatch_lock` 串行化派发：
  1. **外部互锁与急停（E-Stop）**：每次派发前检查可选 `external_interlock`；支持手动 `trigger_agent_estop!`、
     超时/不确定失败/连续失败熔断自动锁存。急停激活时一切动作与 `reset_agent_environment!` 均 fail closed
     拒绝执行；`clear_agent_estop!` 必须提供非空 `operator` 与 `reason`，且在外部互锁仍触发或超时任务未结束时
     拒绝清除（除非显式 `force=true`）。
  2. **有界超时（Timeout）**：在隔离异步任务下执行底层 `step_agent_environment!` 并通过 `timedwait` 限期返回；
     超时返回 `status=:timed_out`、`side_effect_certainty=:uncertain` 并按策略锁存急停，同时追踪未结束任务，
     防止后续动作与悬挂调用并发污染硬件状态。
  3. **幂等动作（Idempotency）**：同 `idempotency_key` + 同 `payload_sha256` 在已提交或确定未执行时返回
     `:replayed`（带 `replayed_from_sequence`），绝不重复调用底层 adapter；同 key 不同 payload 直接
     `:rejected / :idempotency_conflict`；对 `:uncertain` / `:timed_out` key 重放直接 fail closed 拒绝；
     缓存满时 fail closed 拒绝新 key，不做静默驱逐。
- **工具集成与跨进程审计重放**：提供 `safe_environment_tool`、`safe_gridworld_move_tool` /
  `safe_gridworld_tool_registry`、`safe_adapter_audit_payload` 与 `verify_safe_adapter_audit`。离线校验独立重算
  策略摘要、急停状态摘要、安全事件链、每步 outcome 摘要与篡改检测，无需持有实时 adapter。

## 验证

测试位于 [`test_adapter_action_safety.jl`](../../../test/episodes/episode08_environment_action_loop/chapter150_adapter_action_safety/test_adapter_action_safety.jl)
并已接入 `test/runtests.jl`：

- 独立于 GridWorld 的 `_MockActuatorEnv` 覆盖策略/请求 fail-closed、幂等重放、payload 冲突、有界缓存上限、
  `:not_executed` vs `:uncertain` 失败分类、连续失败熔断、真实阻塞超时锁存与任务收敛前拒绝解除急停、手动急停、
  外部硬件互锁阻断、JSON 往返离线审计校验及篡改拒绝。
- `SafeEnvironmentAdapter{GridWorldEnvironment}` 通过 `safe_gridworld_tool_registry` 验证工具调用去重、目标到达、
  安全 reset 与离线审计。

## 结论与边界

Episode 08 的最后一条 Close 条件已满足，Episode 08 正式 Closed。本契约定义进程内单 adapter 的确定性安全语义与
离线审计；真实设备驱动、硬件看门狗总线和分布式多进程幂等存储留待后续实体设备 adapter 阶段按需接入。
