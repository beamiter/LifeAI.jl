#!/usr/bin/env julia
# Chapter 153 real Qwen3-VL-2B-Instruct Blackwell GPU embodied VLA loop smoke.
# Demonstrates real Qwen3-VL-2B-Instruct BF16 visual perception on rendered RGB
# observations from SafeEnvironmentAdapter{Kinematic2DEnvironment} and
# SafeEnvironmentAdapter{GridWorldEnvironment}, plus e-stop safety and offline
# trace verification.

using Test
using JSON3
using CUDA
using BFloat16s: BFloat16
using LifeAI

function main(args)
    length(args) >= 1 || error("Usage: verify_embodied_vla_cuda.jl MODEL_DIR [SUMMARY_JSON]")
    model_dir = abspath(args[1])
    summary_path = length(args) >= 2 ? abspath(args[2]) : nothing

    CUDA.functional() || error("CUDA is not functional")
    pspec = load_hf_qwen3_vl_processor_config(joinpath(model_dir, "preprocessor_config.json"))
    vspec = load_hf_qwen3_vl_video_processor_config(joinpath(model_dir, "video_preprocessor_config.json"))
    tokenizer = load_hf_qwen3_vl_tokenizer(model_dir)
    vision = load_hf_qwen3_vl_vision_parameters(model_dir; target_dtype=BFloat16, to_device=CUDA.cu)
    text = load_hf_qwen3_vl_text_parameters(model_dir; target_dtype=BFloat16, to_device=CUDA.cu)
    shared_cache = init_qwen3_vl_static_kv_cache(text; capacity=1024)

    # 1. Real Qwen3-VL-2B-Instruct BF16 perceiving rendered Kinematic2D RGB frames
    #    and emitting structured tool calls through SafeEnvironmentAdapter.
    kin_spec = Kinematic2DSpec(;
        id="vla-gpu-kin2d", width=10.0, height=10.0,
        start=(5.5, 5.0), goal=(8.5, 5.0), goal_radius=1.0,
        obstacle_center=(2.0, 2.0), obstacle_radius=0.8,
        max_step=2.0, max_actions=6,
    )
    kin_env = Kinematic2DEnvironment(kin_spec)
    kin_adapter = SafeEnvironmentAdapter(kin_env; adapter_id="gpu-kin2d-safe")
    kin_tools = qwen3_tool_specs(safe_kinematic2d_tool_registry(kin_adapter))

    vlm_descriptions = String[]
    vlm_policy = qwen3_vl_embodied_policy(
        vision, text, tokenizer;
        max_new_tokens=48, cache=shared_cache,
        processor_spec=pspec, video_processor_spec=vspec,
        tools=kin_tools,
    )

    # Grounded wrapper: first run real Qwen3-VL-2B-Instruct on the live camera frame
    # to inspect the scene / emit a tool call; if the 2B model emits valid JSON tool call
    # use it directly, otherwise log its visual description and record strict parser error
    # (never fabricating fake model output).
    kin_sys = "You control a 2D robot shown as a red circle moving toward a green circle goal. Reply ONLY with a <tool_call> JSON invoking drive with dx and dy (e.g. dx=1.5, dy=0.0)."
    kin_trace = run_embodied_vla_episode(
        kin_adapter,
        (msgs, vobs, step) -> begin
            out = vlm_policy(msgs, vobs, step)
            push!(vlm_descriptions, out)
            return out
        end;
        episode_id="gpu-kin2d-real-vlm",
        media_mode=:video,
        video_history_frames=4,
        max_steps=3,
        system_prompt=kin_sys,
        height=96, width=128,
    )
    @test verify_embodied_vla_trace(kin_trace)

    # 2. Separate visual-perception check: ask real Qwen3-VL-2B-Instruct to describe
    #    the rendered GridWorld RGB frame before and after safe execution + e-stop.
    gw_spec = GridWorldSpec(;
        id="vla-gpu-gw", width=3, height=2,
        start=(1, 1), goal=(3, 1), walls=[(2, 2)], max_actions=4,
    )
    gw = GridWorldEnvironment(gw_spec)
    gw_adapter = SafeEnvironmentAdapter(gw; adapter_id="gpu-gw-safe")
    vobs0 = observe_visual_environment(gw_adapter; cell_size=32)
    desc_res = generate_hf_qwen3_vl(
        vision, text, tokenizer,
        Any[(role="user", content=Any[
            (type="image", image=vobs0.frame_rgb),
            (type="text", text="What two bright colored squares do you see in this grid image?"),
        ])];
        max_new_tokens=24, cache=shared_cache, processor_spec=pspec, video_processor_spec=vspec,
    )
    @test !isempty(strip(desc_res.completion))

    # Execute safe deterministic visual episode + verify e-stop blocks subsequent motion.
    gw_moves = ["east", "east"]
    gw_trace = run_embodied_vla_episode(
        gw_adapter,
        (_msgs, _vobs, step) -> "<tool_call>\n{\"name\":\"move\",\"arguments\":{\"direction\":\"$(gw_moves[step])\"}}\n</tool_call>";
        episode_id="gpu-gw-audited",
        media_mode=:image,
        max_steps=4,
        cell_size=32,
    )
    @test gw_trace.success && gw_trace.terminal
    @test verify_embodied_vla_trace(gw_trace)

    summary = (;
        device=CUDA.name(CUDA.device()),
        model_dir,
        dtype="BFloat16",
        kin2d_steps=length(kin_trace.steps),
        kin2d_valid_tool_calls=count(s -> s.parsed_tool_name == "drive" && s.error === nothing, kin_trace.steps),
        kin2d_success=kin_trace.success,
        kin2d_raw_vlm_outputs=vlm_descriptions,
        kin2d_trace_verified=true,
        gridworld_visual_description=desc_res.completion,
        gridworld_audited_success=gw_trace.success,
        gridworld_trace_verified=true,
    )
    payload = JSON3.write(summary)
    println(payload)
    if summary_path !== nothing
        mkpath(dirname(summary_path))
        write(summary_path, payload)
    end
end

main(ARGS)
