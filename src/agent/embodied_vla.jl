# Episode 11 / Chapter 153: Embodied Multimodal Vision-Language-Action (VLA)
# Closed Loop. Connects visual/video perception (Qwen3-VL), structured tool
# calls, Chapter 150 cross-adapter action safety (timeout / failure certainty /
# e-stop / idempotency), and conditional clean-success semantic memory.

const EMBODIED_VLA_FORMAT_VERSION = 1

"""
    VisualAgentObservation

Canonical multimodal observation combining a structured
`AgentEnvironmentObservation` with an RGB `UInt8` camera frame `(H, W, 3)` and
a bounded sliding frame history for single-image or multi-frame video policies.
"""
struct VisualAgentObservation
    observation::AgentEnvironmentObservation
    frame_rgb::Array{UInt8,3}
    frame_history::Vector{Array{UInt8,3}}
    fps::Float64
    frame_sha256::String
    sha256::String
end

function _visual_frame_sha256(frame_rgb::AbstractArray{UInt8,3})
    size(frame_rgb, 3) == 3 || throw(DimensionMismatch(
        "visual observation frame must have shape (H, W, 3)",
    ))
    h, w, _ = size(frame_rgb)
    h > 0 && w > 0 || throw(ArgumentError("visual observation frame must be non-empty"))
    io = IOBuffer()
    write(io, Int64(h), Int64(w), Int64(3))
    write(io, vec(Array{UInt8,3}(frame_rgb)))
    return _sha256_hex(take!(io))
end

function VisualAgentObservation(
    observation::AgentEnvironmentObservation,
    frame_rgb::AbstractArray{UInt8,3};
    frame_history::AbstractVector{<:AbstractArray{UInt8,3}}=Array{UInt8,3}[Array{UInt8,3}(frame_rgb)],
    fps::Real=2.0,
)
    fps isa Real && !(fps isa Bool) && isfinite(Float64(fps)) && Float64(fps) > 0.0 ||
        throw(ArgumentError("visual observation fps must be a finite positive real"))
    frame = Array{UInt8,3}(frame_rgb)
    fsha = _visual_frame_sha256(frame)
    history = Array{UInt8,3}[Array{UInt8,3}(f) for f in frame_history]
    !isempty(history) || throw(ArgumentError("frame_history must contain at least one frame"))
    for (idx, hframe) in enumerate(history)
        size(hframe) == size(frame) || throw(DimensionMismatch(
            "frame_history[$idx] size $(size(hframe)) differs from frame_rgb size $(size(frame))",
        ))
    end
    history_hashes = String[_visual_frame_sha256(f) for f in history]
    digest_payload = (;
        schema_version=EMBODIED_VLA_FORMAT_VERSION,
        kind="visual_agent_observation",
        task_id=observation.task_id,
        step=observation.step,
        terminal=observation.terminal,
        success=observation.success,
        state_sha256=observation.state_sha256,
        frame_sha256=fsha,
        history_sha256=history_hashes,
        fps=Float64(fps),
    )
    return VisualAgentObservation(
        observation,
        frame,
        history,
        Float64(fps),
        fsha,
        _sha256_hex(String(JSON3.write(digest_payload))),
    )
end

# ---------------------------------------------------------------------------
# Visual renderer for GridWorldEnvironment
# ---------------------------------------------------------------------------

"""
    render_gridworld_rgb(env::GridWorldEnvironment; cell_size=16)

Deterministically render a `GridWorldEnvironment` state as an `(H, W, 3)` RGB
`UInt8` image: light background, dark-gray walls, bright-green goal, and
bright-red agent marker (gold if agent is at goal).
"""
function render_gridworld_rgb(env::GridWorldEnvironment; cell_size::Integer=16)
    cell = _strict_host_int(cell_size, "cell_size")
    cell >= 4 || throw(ArgumentError("cell_size must be at least 4"))
    h = env.spec.height * cell
    w = env.spec.width * cell
    img = fill(UInt8(235), h, w, 3)

    # Draw subtle cell grid lines.
    for gx in 1:env.spec.width
        x0 = (gx - 1) * cell + 1
        img[:, x0, :] .= UInt8(200)
    end
    for gy in 1:env.spec.height
        y0 = (gy - 1) * cell + 1
        img[y0, :, :] .= UInt8(200)
    end

    # Paint walls (dark slate).
    for (wx, wy) in env.spec.walls
        ys = ((wy - 1) * cell + 2):(wy * cell)
        xs = ((wx - 1) * cell + 2):(wx * cell)
        img[ys, xs, 1] .= UInt8(45)
        img[ys, xs, 2] .= UInt8(50)
        img[ys, xs, 3] .= UInt8(60)
    end

    # Paint goal (bright green).
    gx, gy = env.spec.goal
    gys = ((gy - 1) * cell + 3):(gy * cell - 1)
    gxs = ((gx - 1) * cell + 3):(gx * cell - 1)
    img[gys, gxs, 1] .= UInt8(35)
    img[gys, gxs, 2] .= UInt8(205)
    img[gys, gxs, 3] .= UInt8(65)

    # Paint agent (bright red; gold if reached goal).
    ax, ay = env.position
    pad = max(2, cell ÷ 4)
    ays = ((ay - 1) * cell + pad + 1):(ay * cell - pad)
    axs = ((ax - 1) * cell + pad + 1):(ax * cell - pad)
    if env.position == env.spec.goal
        img[ays, axs, 1] .= UInt8(245)
        img[ays, axs, 2] .= UInt8(195)
        img[ays, axs, 3] .= UInt8(25)
    else
        img[ays, axs, 1] .= UInt8(230)
        img[ays, axs, 2] .= UInt8(40)
        img[ays, axs, 3] .= UInt8(35)
    end
    return img
end

# ---------------------------------------------------------------------------
# Non-GridWorld Embodied Environment: Kinematic2DEnvironment
# ---------------------------------------------------------------------------

"""
    Kinematic2DSpec

Immutable geometry and safety bounds for a deterministic 2D continuous-position
mobile base / planar end-effector environment (independent of GridWorld).
"""
struct Kinematic2DSpec
    id::String
    width::Float64
    height::Float64
    start::NTuple{2,Float64}
    goal::NTuple{2,Float64}
    goal_radius::Float64
    obstacle_center::NTuple{2,Float64}
    obstacle_radius::Float64
    max_step::Float64
    max_actions::Int
    sha256::String
end

function _kinematic_pos(val, label::AbstractString)
    (val isa Tuple || val isa AbstractVector) && length(val) == 2 ||
        throw(ArgumentError("$label must be a 2-element (x, y) coordinate"))
    x = Float64(val[1]); y = Float64(val[2])
    !(val[1] isa Bool) && !(val[2] isa Bool) && isfinite(x) && isfinite(y) ||
        throw(ArgumentError("$label coordinates must be finite non-Boolean reals"))
    return (x, y)
end

function Kinematic2DSpec(;
    id::AbstractString,
    width::Real=10.0,
    height::Real=10.0,
    start=(1.5, 5.0),
    goal=(8.5, 5.0),
    goal_radius::Real=1.0,
    obstacle_center=(5.0, 5.0),
    obstacle_radius::Real=1.2,
    max_step::Real=2.0,
    max_actions::Integer=12,
)
    id_val = _agent_environment_id(String(id))
    w = _safety_positive_float64(width, "width")
    h = _safety_positive_float64(height, "height")
    st = _kinematic_pos(start, "start")
    gl = _kinematic_pos(goal, "goal")
    gr = _safety_positive_float64(goal_radius, "goal_radius")
    ob = _kinematic_pos(obstacle_center, "obstacle_center")
    obr = Float64(obstacle_radius)
    !(obstacle_radius isa Bool) && isfinite(obr) && obr >= 0.0 ||
        throw(ArgumentError("obstacle_radius must be a finite non-negative real"))
    ms = _safety_positive_float64(max_step, "max_step")
    ma = _strict_host_int(max_actions, "max_actions")
    ma > 0 || throw(ArgumentError("max_actions must be positive"))
    0.0 <= st[1] <= w && 0.0 <= st[2] <= h || throw(ArgumentError("start lies outside workspace"))
    0.0 <= gl[1] <= w && 0.0 <= gl[2] <= h || throw(ArgumentError("goal lies outside workspace"))
    hypot(st[1] - ob[1], st[2] - ob[2]) > obr || throw(ArgumentError("start collides with obstacle"))
    hypot(gl[1] - ob[1], gl[2] - ob[2]) > obr || throw(ArgumentError("goal collides with obstacle"))
    hypot(st[1] - gl[1], st[2] - gl[2]) > gr || throw(ArgumentError("start already inside goal_radius"))

    payload = (;
        schema_version=EMBODIED_VLA_FORMAT_VERSION,
        environment="kinematic_2d",
        id=id_val,
        width=w,
        height=h,
        start=(; x=st[1], y=st[2]),
        goal=(; x=gl[1], y=gl[2]),
        goal_radius=gr,
        obstacle_center=(; x=ob[1], y=ob[2]),
        obstacle_radius=obr,
        max_step=ms,
        max_actions=ma,
    )
    return Kinematic2DSpec(
        id_val, w, h, st, gl, gr, ob, obr, ms, ma,
        _sha256_hex(String(JSON3.write(payload))),
    )
end

"""
    Kinematic2DEnvironment(spec::Kinematic2DSpec) <: AbstractAgentEnvironment

Deterministic continuous 2D planar robot simulator supporting `"drive"` actions
encoded as `"dx,dy"` (e.g. `"1.5,-0.5"`). Rejects out-of-bounds, oversized-step,
and obstacle-colliding commands without mutating position.
"""
mutable struct Kinematic2DEnvironment <: AbstractAgentEnvironment
    spec::Kinematic2DSpec
    position::NTuple{2,Float64}
    actions::Int
    terminal::Bool
    success::Bool
    transitions::Vector{AgentEnvironmentTransition}
end

Kinematic2DEnvironment(spec::Kinematic2DSpec) = Kinematic2DEnvironment(
    spec, spec.start, 0, false, false, AgentEnvironmentTransition[],
)

function reset_agent_environment!(env::Kinematic2DEnvironment)
    env.position = env.spec.start
    env.actions = 0
    env.terminal = false
    env.success = false
    empty!(env.transitions)
    return env
end

function _kinematic2d_state_sha256(env::Kinematic2DEnvironment)
    payload = (;
        schema_version=EMBODIED_VLA_FORMAT_VERSION,
        spec_sha256=env.spec.sha256,
        x=round(env.position[1]; digits=6),
        y=round(env.position[2]; digits=6),
        actions=env.actions,
        terminal=env.terminal,
        success=env.success,
    )
    return _sha256_hex(String(JSON3.write(payload)))
end

function observe_agent_environment(env::Kinematic2DEnvironment)
    state_sha = _kinematic2d_state_sha256(env)
    dist_goal = hypot(env.position[1] - env.spec.goal[1], env.position[2] - env.spec.goal[2])
    rendered = String(JSON3.write((;
        environment="kinematic_2d",
        task_id=env.spec.id,
        step=env.actions,
        max_actions=env.spec.max_actions,
        position=(; x=round(env.position[1]; digits=4), y=round(env.position[2]; digits=4)),
        goal=(; x=env.spec.goal[1], y=env.spec.goal[2], radius=env.spec.goal_radius),
        obstacle=(; x=env.spec.obstacle_center[1], y=env.spec.obstacle_center[2], radius=env.spec.obstacle_radius),
        distance_to_goal=round(dist_goal; digits=4),
        max_step=env.spec.max_step,
        terminal=env.terminal,
        success=env.success,
    )))
    return AgentEnvironmentObservation(
        env.spec.id,
        env.actions,
        env.terminal,
        env.success,
        state_sha,
        rendered,
        _sha256_hex(rendered),
    )
end

function _parse_kinematic2d_delta(raw::AbstractString)
    parts = split(strip(String(raw)), ',')
    length(parts) == 2 || throw(ArgumentError("drive value must be 'dx,dy'"))
    dx = tryparse(Float64, strip(parts[1]))
    dy = tryparse(Float64, strip(parts[2]))
    (dx !== nothing && dy !== nothing && isfinite(dx) && isfinite(dy)) ||
        throw(ArgumentError("drive dx,dy must be finite real numbers"))
    return (dx, dy)
end

function step_agent_environment!(env::Kinematic2DEnvironment, action::AgentEnvironmentAction)
    before_sha = _kinematic2d_state_sha256(env)
    action.name == "drive" || throw(ArgumentError(
        "Kinematic2DEnvironment only accepts action 'drive'; got $(repr(action.name))",
    ))
    dx, dy = _parse_kinematic2d_delta(action.value)
    accepted = false
    reason = "accepted"

    if env.terminal
        reason = "episode already terminal"
    else
        env.actions += 1
        step_norm = hypot(dx, dy)
        nx = env.position[1] + dx
        ny = env.position[2] + dy
        if step_norm > env.spec.max_step + 1e-9
            reason = "step magnitude exceeds max_step"
        elseif !(0.0 <= nx <= env.spec.width && 0.0 <= ny <= env.spec.height)
            reason = "target outside workspace bounds"
        elseif hypot(nx - env.spec.obstacle_center[1], ny - env.spec.obstacle_center[2]) <= env.spec.obstacle_radius
            reason = "target collides with obstacle"
        else
            env.position = (nx, ny)
            accepted = true
            if hypot(nx - env.spec.goal[1], ny - env.spec.goal[2]) <= env.spec.goal_radius
                env.terminal = true
                env.success = true
                reason = "goal reached"
            end
        end
        if !env.terminal && env.actions >= env.spec.max_actions
            env.terminal = true
            reason = accepted ? "action budget exhausted" : reason * "; action budget exhausted"
        end
    end

    obs = observe_agent_environment(env)
    tx_payload = (;
        schema_version=EMBODIED_VLA_FORMAT_VERSION,
        step=env.actions,
        action=(; name=action.name, value=action.value),
        accepted,
        reason,
        before_state_sha256=before_sha,
        after_state_sha256=obs.state_sha256,
        observation_sha256=obs.rendered_sha256,
    )
    tx = AgentEnvironmentTransition(
        env.actions,
        action,
        accepted,
        reason,
        before_sha,
        obs.state_sha256,
        obs,
        _sha256_hex(String(JSON3.write(tx_payload))),
    )
    push!(env.transitions, tx)
    return tx
end

"""
    render_kinematic2d_rgb(env::Kinematic2DEnvironment; height=96, width=128)

Deterministically render a top-down RGB camera view `(height, width, 3)` of the
2D continuous workspace: pale background, dark-blue obstacle circle, bright-green
goal disk, and bright-red robot disk.
"""
function render_kinematic2d_rgb(
    env::Kinematic2DEnvironment;
    height::Integer=96,
    width::Integer=128,
)
    h = _strict_host_int(height, "height")
    w = _strict_host_int(width, "width")
    h >= 16 && w >= 16 || throw(ArgumentError("render dimensions must be at least 16x16"))
    img = fill(UInt8(240), h, w, 3)

    # Border frame.
    img[1:2, :, :] .= UInt8(120)
    img[(h - 1):h, :, :] .= UInt8(120)
    img[:, 1:2, :] .= UInt8(120)
    img[:, (w - 1):w, :] .= UInt8(120)

    sx = env.spec.width / Float64(w)
    sy = env.spec.height / Float64(h)
    agent_r = max(0.45, min(env.spec.width, env.spec.height) * 0.055)
    ox, oy = env.spec.obstacle_center
    orad = env.spec.obstacle_radius
    gx, gy = env.spec.goal
    grad = env.spec.goal_radius
    ax, ay = env.position

    @inbounds for row in 1:h, col in 1:w
        wx = (col - 0.5) * sx
        wy = (row - 0.5) * sy
        if orad > 0.0 && hypot(wx - ox, wy - oy) <= orad
            img[row, col, 1] = UInt8(40)
            img[row, col, 2] = UInt8(60)
            img[row, col, 3] = UInt8(125)
        elseif hypot(wx - gx, wy - gy) <= grad
            img[row, col, 1] = UInt8(40)
            img[row, col, 2] = UInt8(205)
            img[row, col, 3] = UInt8(70)
        end
        if hypot(wx - ax, wy - ay) <= agent_r
            if env.success
                img[row, col, 1] = UInt8(245)
                img[row, col, 2] = UInt8(195)
                img[row, col, 3] = UInt8(25)
            else
                img[row, col, 1] = UInt8(235)
                img[row, col, 2] = UInt8(45)
                img[row, col, 3] = UInt8(40)
            end
        end
    end
    return img
end

# ---------------------------------------------------------------------------
# Unified Visual Observation Dispatcher
# ---------------------------------------------------------------------------

render_environment_rgb(env::GridWorldEnvironment; kwargs...) =
    render_gridworld_rgb(env; kwargs...)
render_environment_rgb(env::Kinematic2DEnvironment; kwargs...) =
    render_kinematic2d_rgb(env; kwargs...)
render_environment_rgb(adapter::SafeEnvironmentAdapter; kwargs...) =
    lock(adapter.dispatch_lock) do
        render_environment_rgb(adapter.environment; kwargs...)
    end

function observe_visual_environment(
    env::AbstractAgentEnvironment;
    history::AbstractVector{<:AbstractArray{UInt8,3}}=Array{UInt8,3}[],
    max_history::Integer=4,
    fps::Real=2.0,
    render_kwargs...,
)
    max_h = _strict_host_int(max_history, "max_history")
    max_h > 0 || throw(ArgumentError("max_history must be positive"))
    obs = observe_agent_environment(env)
    frame = render_environment_rgb(env; render_kwargs...)
    new_hist = Array{UInt8,3}[Array{UInt8,3}(f) for f in history]
    push!(new_hist, frame)
    if length(new_hist) > max_h
        new_hist = new_hist[(end - max_h + 1):end]
    end
    return VisualAgentObservation(obs, frame; frame_history=new_hist, fps)
end

# ---------------------------------------------------------------------------
# Safe Tool Registry for Kinematic2DEnvironment
# ---------------------------------------------------------------------------

function safe_kinematic2d_tool_registry(adapter::SafeEnvironmentAdapter{Kinematic2DEnvironment})
    tool = safe_environment_tool(
        adapter;
        name="drive",
        description="Move the 2D robot by (dx, dy) within max_step through the safety adapter.",
        properties=(;
            dx=(; type="number", description="Delta x within max_step"),
            dy=(; type="number", description="Delta y within max_step"),
        ),
        required=["dx", "dy"],
        action_from_arguments=function (arguments)
            dx = _tool_argument(arguments, "dx")
            dy = _tool_argument(arguments, "dy")
            (dx isa Real && !(dx isa Bool) && isfinite(Float64(dx))) ||
                throw(ArgumentError("dx must be a finite real number"))
            (dy isa Real && !(dy isa Bool) && isfinite(Float64(dy))) ||
                throw(ArgumentError("dy must be a finite real number"))
            return AgentEnvironmentAction("drive", "$(Float64(dx)),$(Float64(dy))")
        end,
    )
    return ToolRegistry([tool])
end

safe_visual_tool_registry(adapter::SafeEnvironmentAdapter{GridWorldEnvironment}) =
    safe_gridworld_tool_registry(adapter)
safe_visual_tool_registry(adapter::SafeEnvironmentAdapter{Kinematic2DEnvironment}) =
    safe_kinematic2d_tool_registry(adapter)

# ---------------------------------------------------------------------------
# Multimodal Embodied VLA Episode Runner & Offline Verification
# ---------------------------------------------------------------------------

struct EmbodiedVLAStep
    step_index::Int
    media_mode::Symbol
    frame_sha256::String
    observation_sha256::String
    assistant_response::String
    parsed_tool_name::Union{Nothing,String}
    idempotency_key::Union{Nothing,String}
    outcome::Union{Nothing,SafeActionOutcome}
    error::Union{Nothing,String}
    sha256::String
end

struct EmbodiedVLATrace
    schema_version::Int
    episode_id::String
    adapter_id::String
    media_mode::Symbol
    steps::Vector{EmbodiedVLAStep}
    final_observation::VisualAgentObservation
    success::Bool
    terminal::Bool
    estop_active::Bool
    audit_payload::String
    memory_written::Bool
    memory_record_id::Union{Nothing,String}
    sha256::String
end

"""
    qwen3_vl_embodied_policy(vision_parameters, text_parameters, tokenizer;
                             max_new_tokens=64, cache=:static,
                             processor_spec=..., video_processor_spec=...)

Create a real Qwen3-VL multimodal policy closure compatible with
[`run_embodied_vla_episode`](@ref). Reuses a single bounded static cache across
steps when a `Qwen3VLStaticKVCache` is supplied.
"""
function qwen3_vl_embodied_policy(
    vision_parameters,
    text_parameters,
    tokenizer::HFQwen3Tokenizer;
    max_new_tokens::Integer=64,
    cache::Union{Symbol,Qwen3VLStaticKVCache}=:static,
    processor_spec::Qwen3VLProcessorSpec=qwen3_vl_processor_spec(),
    video_processor_spec::Qwen3VLProcessorSpec=qwen3_vl_video_processor_spec(),
    strategy::Symbol=:greedy,
    temperature=nothing,
    top_k=nothing,
    top_p=nothing,
    tools=nothing,
)
    return function (messages, _visual_obs::VisualAgentObservation, _step::Int)
        res = generate_hf_qwen3_vl(
            vision_parameters,
            text_parameters,
            tokenizer,
            messages;
            max_new_tokens,
            tools,
            cache,
            processor_spec,
            video_processor_spec,
            strategy,
            temperature,
            top_k,
            top_p,
        )
        return res.completion
    end
end

"""
    run_embodied_vla_episode(adapter::SafeEnvironmentAdapter, policy;
                             episode_id="vla-ep-01",
                             media_mode=:image,
                             max_steps=16,
                             video_history_frames=4,
                             fps=2.0,
                             system_prompt=...,
                             tool_registry=safe_visual_tool_registry(adapter),
                             memory_store=nothing,
                             write_clean_success_only=true,
                             render_kwargs...)

Run a verifiable multimodal vision-language-action closed loop through a
Chapter 150 [`SafeEnvironmentAdapter`](@ref):
1. Captures a deterministic RGB visual observation (`:image` or `:video` sliding window).
2. Builds Qwen3-VL content-list messages and invokes `policy(messages, visual_obs, step)`.
3. Parses `<tool_call>` JSON strictly via `parse_qwen3_tool_calls`, injects a deterministic
   per-step `idempotency_key` if omitted, and dispatches strictly through `tool_registry`
   / `SafeEnvironmentAdapter`.
4. Halts immediately on terminal state, e-stop latch, or step budget exhaustion.
5. Runs `verify_safe_adapter_audit` and writes to `memory_store` **only** on clean
   successful episodes when `write_clean_success_only=true`.
"""
function run_embodied_vla_episode(
    adapter::SafeEnvironmentAdapter,
    policy;
    episode_id::AbstractString="vla-ep-01",
    media_mode::Symbol=:image,
    max_steps::Integer=16,
    video_history_frames::Integer=4,
    fps::Real=2.0,
    system_prompt::AbstractString="You are an embodied visual agent. Inspect the camera view and emit exactly one valid tool call.",
    tool_registry::ToolRegistry=safe_visual_tool_registry(adapter),
    memory_store::Union{Nothing,AgentMemoryStore}=nothing,
    write_clean_success_only::Bool=true,
    render_kwargs...,
)
    ep_id = _agent_environment_id(String(episode_id))
    media_mode in (:image, :video) || throw(ArgumentError(
        "media_mode must be :image or :video; got $(repr(media_mode))",
    ))
    step_limit = _strict_host_int(max_steps, "max_steps")
    step_limit > 0 || throw(ArgumentError("max_steps must be positive"))
    hist_len = _strict_host_int(video_history_frames, "video_history_frames")
    hist_len >= 2 || throw(ArgumentError("video_history_frames must be at least 2"))

    visual_obs = observe_visual_environment(
        adapter; max_history=hist_len, fps, render_kwargs...,
    )
    steps = EmbodiedVLAStep[]

    for step_idx in 1:step_limit
        visual_obs.observation.terminal && break
        adapter.estop.active && break

        media_item = if media_mode === :video && length(visual_obs.frame_history) >= 2
            (type="video", video=visual_obs.frame_history, fps=visual_obs.fps)
        else
            (type="image", image=visual_obs.frame_rgb)
        end
        messages = Any[
            (role="system", content=String(system_prompt)),
            (role="user", content=Any[
                media_item,
                (type="text", text="Step $(step_idx). State: $(visual_obs.observation.rendered)"),
            ]),
        ]

        raw_response = String(policy(messages, visual_obs, step_idx))
        parsed = parse_qwen3_tool_calls(raw_response)

        tool_name = nothing
        idem_key = nothing
        step_outcome = nothing
        step_error = nothing

        if !isempty(parsed.invalid) || length(parsed.calls) != 1
            step_error = isempty(parsed.invalid) ?
                "expected exactly one tool call; got $(length(parsed.calls))" :
                join(parsed.invalid, "; ")
        else
            call = only(parsed.calls)
            tool_name = call.name
            args = call.arguments
            if !haskey(args, "idempotency_key")
                idem_key = "$(ep_id)-step-$(step_idx)"
                args_pairs = Pair{String,Any}[k => v for (k, v) in pairs(args)]
                push!(args_pairs, "idempotency_key" => idem_key)
                call = Qwen3ToolCall(call.name, OrderedJSONObject(args_pairs), call.raw)
            else
                idem_key = String(args["idempotency_key"])
            end

            prev_audit_len = length(adapter.audit_log)
            tool_res = invoke_agent_tool(tool_registry, call)
            if length(adapter.audit_log) > prev_audit_len
                step_outcome = adapter.audit_log[end]
            end
            if !tool_res.ok
                step_error = something(tool_res.error, "tool invocation failed")
            elseif step_outcome !== nothing && step_outcome.status in (:timed_out, :failed, :estopped, :rejected)
                step_error = something(step_outcome.error, "safe action status=$(step_outcome.status)")
            end
        end

        step_payload = (;
            schema_version=EMBODIED_VLA_FORMAT_VERSION,
            kind="embodied_vla_step",
            episode_id=ep_id,
            step_index=step_idx,
            media_mode=String(media_mode),
            frame_sha256=visual_obs.frame_sha256,
            observation_sha256=visual_obs.sha256,
            assistant_response=raw_response,
            parsed_tool_name=tool_name,
            idempotency_key=idem_key,
            outcome_sha256=step_outcome === nothing ? nothing : step_outcome.sha256,
            error=step_error,
        )
        push!(steps, EmbodiedVLAStep(
            step_idx,
            media_mode,
            visual_obs.frame_sha256,
            visual_obs.sha256,
            raw_response,
            tool_name,
            idem_key,
            step_outcome,
            step_error,
            _sha256_hex(String(JSON3.write(step_payload))),
        ))

        visual_obs = observe_visual_environment(
            adapter;
            history=visual_obs.frame_history,
            max_history=hist_len,
            fps,
            render_kwargs...,
        )
        adapter.estop.active && break
    end

    audit_json = String(JSON3.write(safe_adapter_audit_payload(adapter)))
    verify_safe_adapter_audit(JSON3.read(audit_json))

    clean_success = visual_obs.observation.success &&
        visual_obs.observation.terminal &&
        !adapter.estop.active &&
        all(s -> s.error === nothing && s.outcome !== nothing && s.outcome.status in (:executed, :replayed), steps)

    memory_written = false
    memory_record_id = nothing
    if memory_store !== nothing && (!write_clean_success_only || clean_success)
        rec = append_agent_memory!(
            memory_store,
            "embodied_vla episode $(ep_id) on $(adapter.adapter_id): success=$(clean_success), steps=$(length(steps)), final_state=$(visual_obs.observation.state_sha256)";
            metadata=Dict{String,Any}(
                "episode_id" => ep_id,
                "adapter_id" => adapter.adapter_id,
                "media_mode" => String(media_mode),
                "clean_success" => string(clean_success),
                "final_frame_sha256" => visual_obs.frame_sha256,
            ),
        )
        memory_written = true
        memory_record_id = rec.id
    end

    trace_payload = (;
        schema_version=EMBODIED_VLA_FORMAT_VERSION,
        kind="embodied_vla_trace",
        episode_id=ep_id,
        adapter_id=adapter.adapter_id,
        media_mode=String(media_mode),
        step_hashes=[s.sha256 for s in steps],
        final_observation_sha256=visual_obs.sha256,
        success=visual_obs.observation.success,
        terminal=visual_obs.observation.terminal,
        estop_active=adapter.estop.active,
        audit_sha256=_sha256_hex(audit_json),
        memory_written,
        memory_record_id,
    )
    return EmbodiedVLATrace(
        EMBODIED_VLA_FORMAT_VERSION,
        ep_id,
        adapter.adapter_id,
        media_mode,
        steps,
        visual_obs,
        visual_obs.observation.success,
        visual_obs.observation.terminal,
        adapter.estop.active,
        audit_json,
        memory_written,
        memory_record_id,
        _sha256_hex(String(JSON3.write(trace_payload))),
    )
end

"""
    verify_embodied_vla_trace(trace::EmbodiedVLATrace)

Independently verify an [`EmbodiedVLATrace`](@ref) offline: recomputes every
step digest, verifies the embedded Chapter 150 `SafeEnvironmentAdapter` audit
chain, verifies final observation frame digest, and confirms the top-level
trace SHA-256.
"""
function verify_embodied_vla_trace(trace::EmbodiedVLATrace)
    trace.schema_version == EMBODIED_VLA_FORMAT_VERSION || throw(ArgumentError(
        "unsupported EmbodiedVLATrace schema_version $(trace.schema_version)",
    ))
    verify_safe_adapter_audit(JSON3.read(trace.audit_payload))

    actual_final_frame_sha = _visual_frame_sha256(trace.final_observation.frame_rgb)
    actual_final_frame_sha == trace.final_observation.frame_sha256 || throw(ArgumentError(
        "EmbodiedVLATrace final frame SHA-256 mismatch",
    ))

    for (idx, step) in enumerate(trace.steps)
        step.step_index == idx || throw(ArgumentError(
            "EmbodiedVLATrace step_index mismatch at $idx",
        ))
        step_payload = (;
            schema_version=EMBODIED_VLA_FORMAT_VERSION,
            kind="embodied_vla_step",
            episode_id=trace.episode_id,
            step_index=step.step_index,
            media_mode=String(step.media_mode),
            frame_sha256=step.frame_sha256,
            observation_sha256=step.observation_sha256,
            assistant_response=step.assistant_response,
            parsed_tool_name=step.parsed_tool_name,
            idempotency_key=step.idempotency_key,
            outcome_sha256=step.outcome === nothing ? nothing : step.outcome.sha256,
            error=step.error,
        )
        expected = _sha256_hex(String(JSON3.write(step_payload)))
        expected == step.sha256 || throw(ArgumentError(
            "EmbodiedVLATrace step $idx digest mismatch",
        ))
    end

    trace_payload = (;
        schema_version=EMBODIED_VLA_FORMAT_VERSION,
        kind="embodied_vla_trace",
        episode_id=trace.episode_id,
        adapter_id=trace.adapter_id,
        media_mode=String(trace.media_mode),
        step_hashes=[s.sha256 for s in trace.steps],
        final_observation_sha256=trace.final_observation.sha256,
        success=trace.success,
        terminal=trace.terminal,
        estop_active=trace.estop_active,
        audit_sha256=_sha256_hex(trace.audit_payload),
        memory_written=trace.memory_written,
        memory_record_id=trace.memory_record_id,
    )
    expected_trace = _sha256_hex(String(JSON3.write(trace_payload)))
    expected_trace == trace.sha256 || throw(ArgumentError(
        "EmbodiedVLATrace top-level digest mismatch",
    ))
    return true
end
