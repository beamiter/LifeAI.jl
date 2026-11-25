# Chapter 153 — 具身多模态 VLA 安全闭环

> 状态：Closed（2026-11-25）
>
> 所属卷：[Episode 11 — 具身多模态视觉-语言-行动安全闭环](README.md)

## 目标

打通 `Qwen3-VL` 视觉/视频感知 → 结构化 `<tool_call>` 策略 → Chapter 150 `SafeEnvironmentAdapter` 安全执行 → 离线可验证审计与条件记忆写回的端到端具身多模态闭环，并在非 GridWorld 连续空间环境与真实 `Qwen3-VL-2B-Instruct` GPU 权重上完成验收。

## 实现

新增 [`src/agent/embodied_vla.jl`](../../../src/agent/embodied_vla.jl) 并接入 `src/LifeAI.jl` 与 `src/api.jl`：

1. **统一视觉观测协议（`VisualAgentObservation`）**：绑定结构化 `AgentEnvironmentObservation`、当前 `(H, W, 3)` RGB 相机帧、滑动历史帧窗口（供 `:video` 模式使用）以及 canonical SHA-256 摘要。
2. **双视觉环境证明非 GridWorld 绑定**：
   - `render_gridworld_rgb(env; cell_size=16)`：确定性渲染 GridWorld 相机视图。
   - `Kinematic2DSpec` / `Kinematic2DEnvironment` / `render_kinematic2d_rgb`：独立于 GridWorld 的连续 2D 平面机器人/末端执行器环境，支持连续 `(dx, dy)` 驱动、步长限幅、工作空间边界与圆形障碍物防碰撞校验，并通过 `safe_kinematic2d_tool_registry` 复用 Chapter 150 `safe_environment_tool`。
3. **可插拔多模态闭环与离线审计（`run_embodied_vla_episode`、`qwen3_vl_embodied_policy`、`verify_embodied_vla_trace`）**：
   - 支持 `:image`（单帧）与 `:video`（多帧滑动窗口）两种观测模式，每步自动注入确定性 `idempotency_key`，所有动作强制经由 `SafeEnvironmentAdapter` 执行；急停/外部互锁触发立即终止回合且禁止写回 `AgentMemoryStore`（严格继承 clean-success-only 写回不变式）。
   - `verify_embodied_vla_trace` 离线独立复核每步摘要、末帧像素摘要、内嵌 Chapter 150 安全审计链与顶层 trace SHA-256，对任意篡改 fail closed。

## 验证结果（2026-11-25，NVIDIA RTX PRO 5000 72GB Blackwell，Julia 1.12.6 `-O1`）

- **离线 CPU 专项**：[`test_embodied_vla_loop.jl`](../../../test/episodes/episode11_embodied_multimodal_agent/chapter153_embodied_vla_loop/test_embodied_vla_loop.jl) **39/39** 通过，覆盖 `Kinematic2D` 物理与安全边界、`:image` 模式 GridWorld + 记忆写回与篡改拒绝、`:video` 模式连续 `Kinematic2D` 绕障抵达目标，以及外部安全光幕互锁急停阻断与零记忆写回。
- **真实 `Qwen3-VL-2B-Instruct` Blackwell GPU 具身闭环 smoke**（[`scripts/verify_embodied_vla_cuda.jl`](../../../scripts/verify_embodied_vla_cuda.jl)，报告冻结于 [`benchmark_results/embodied_vla/summary.json`](../../../benchmark_results/embodied_vla/summary.json)，SHA-256 `d4fe1bb249f202febe6c99af0da452b3ea7a5b3728036c0d4ad44fd8f2e74cfc`）：
  - 真实 `Qwen3-VL-2B-Instruct`（BFloat16，共享 `capacity=1024` static cache）在 `:video` 模式下实时观察 `SafeEnvironmentAdapter{Kinematic2DEnvironment}` 渲染的 `96×128` RGB 相机画面运行 3 步（`max_new_tokens=48`）：
    - 第 1 步直接输出合法 `<tool_call>`（`drive(dx=1.0, dy=0.0, idempotency_key="vla-gpu-kin2d-step-1")`）并经由 `SafeEnvironmentAdapter` 安全执行（`kin2d_valid_tool_calls=1`）；
    - 第 2 步因 48-token 预算截断 JSON，被严格解析器 fail-closed 拦截而未执行动作；
    - 第 3 步复用了第 1 步的 `idempotency_key`，被 Chapter 150 幂等保护拦截。整条回合诚实记录 `kin2d_success=false`（表明未经微调的零样本 2B VLM 尚不能靠单次提示稳定完成多步连续空间控制），同时全链路审计 `kin2d_trace_verified=true` 通过。
  - 在 GridWorld 渲染视图上，真实 `Qwen3-VL-2B-Instruct` 识别出 `"two bright colored squares"` 与左上角红色方块；确定性策略通过安全视觉闭环完成目标并验证审计链（`gridworld_audited_success=true`，`gridworld_trace_verified=true`）。

## 明确边界

- 本章实测环境为进程内确定性 2D 连续/离散视觉仿真，尚未接入外部物理引擎进程（如 MuJoCo/Isaac Sim）或真实机器人控制总线；零样本 2B 模型连续控制成功率未经任务微调。Episode 11 保持 Open。
