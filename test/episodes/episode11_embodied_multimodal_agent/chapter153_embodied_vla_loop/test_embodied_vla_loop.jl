using Test
using JSON3
using LifeAI

@testset "Chapter 153 — Embodied multimodal VLA closed loop" begin
    @testset "Kinematic2D continuous environment physics, safety, and RGB rendering" begin
        spec = Kinematic2DSpec(;
            id="kin2d-demo-01",
            width=10.0, height=10.0,
            start=(1.5, 5.0), goal=(8.5, 5.0), goal_radius=1.0,
            obstacle_center=(5.0, 5.0), obstacle_radius=1.2,
            max_step=2.0, max_actions=10,
        )
        @test length(spec.sha256) == 64
        # Fail-closed spec validation.
        @test_throws ArgumentError Kinematic2DSpec(; id="bad-1", start=(5.0, 5.0), obstacle_center=(5.0, 5.0), obstacle_radius=1.0)
        @test_throws ArgumentError Kinematic2DSpec(; id="bad-2", start=(8.2, 5.0), goal=(8.5, 5.0), goal_radius=1.0)
        @test_throws ArgumentError Kinematic2DSpec(; id="bad-3", max_step=0.0)
        @test_throws ArgumentError Kinematic2DSpec(; id="bad-4", width=true)

        env = Kinematic2DEnvironment(spec)
        frame0 = render_kinematic2d_rgb(env; height=64, width=80)
        @test size(frame0) == (64, 80, 3)
        @test eltype(frame0) === UInt8

        # Oversized step rejected without moving robot.
        tx_big = step_agent_environment!(env, AgentEnvironmentAction("drive", "3.0,0.0"))
        @test !tx_big.accepted
        @test env.position == (1.5, 5.0)

        # Valid detour around obstacle reaches goal.
        reset_agent_environment!(env)
        for delta in ("1.5,1.2", "1.5,1.0", "1.5,0.0", "1.5,-1.0", "1.2,-1.2")
            tx = step_agent_environment!(env, AgentEnvironmentAction("drive", delta))
            @test tx.accepted
        end
        @test env.terminal && env.success
        frame_done = render_kinematic2d_rgb(env; height=64, width=80)
        @test frame_done != frame0
    end

    @testset "Image-mode VLA loop on SafeEnvironmentAdapter{GridWorldEnvironment} + memory writeback" begin
        spec = GridWorldSpec(;
            id="vla-gw-01", width=3, height=2,
            start=(1, 1), goal=(3, 1), walls=[(2, 2)], max_actions=5,
        )
        gw = GridWorldEnvironment(spec)
        adapter = SafeEnvironmentAdapter(gw; adapter_id="safe-vla-gw")
        tmp = mktempdir()
        store = load_agent_memory_store(joinpath(tmp, "mem.jsonl"); create=true)

        seen_media_types = String[]
        scripted_moves = ["east", "east"]
        policy = (messages, vobs, step) -> begin
            user_content = messages[2].content
            push!(seen_media_types, String(user_content[1].type))
            @test size(vobs.frame_rgb) == (32, 48, 3)
            dir = scripted_moves[step]
            return "<tool_call>\n{\"name\":\"move\",\"arguments\":{\"direction\":\"$dir\"}}\n</tool_call>"
        end

        trace = run_embodied_vla_episode(
            adapter, policy;
            episode_id="ep-gw-image-01",
            media_mode=:image,
            max_steps=5,
            memory_store=store,
            write_clean_success_only=true,
            cell_size=16,
        )
        @test trace.success && trace.terminal && !trace.estop_active
        @test length(trace.steps) == 2
        @test seen_media_types == ["image", "image"]
        @test trace.memory_written && trace.memory_record_id !== nothing
        @test length(store.records) == 1
        @test verify_embodied_vla_trace(trace)

        # Tampering with a step digest fails offline verification.
        bad_step = EmbodiedVLAStep(
            trace.steps[1].step_index, trace.steps[1].media_mode,
            trace.steps[1].frame_sha256, trace.steps[1].observation_sha256,
            "tampered", trace.steps[1].parsed_tool_name,
            trace.steps[1].idempotency_key, trace.steps[1].outcome,
            trace.steps[1].error, trace.steps[1].sha256,
        )
        bad_trace = EmbodiedVLATrace(
            trace.schema_version, trace.episode_id, trace.adapter_id,
            trace.media_mode, [bad_step, trace.steps[2]],
            trace.final_observation, trace.success, trace.terminal,
            trace.estop_active, trace.audit_payload, trace.memory_written,
            trace.memory_record_id, trace.sha256,
        )
        @test_throws ArgumentError verify_embodied_vla_trace(bad_trace)
        rm(tmp; recursive=true)
    end

    @testset "Video-mode VLA loop on non-GridWorld SafeEnvironmentAdapter{Kinematic2DEnvironment}" begin
        spec = Kinematic2DSpec(;
            id="vla-kin2d-01", width=10.0, height=10.0,
            start=(1.5, 5.0), goal=(8.5, 5.0), goal_radius=1.0,
            obstacle_center=(5.0, 5.0), obstacle_radius=1.2,
            max_step=2.0, max_actions=8,
        )
        env = Kinematic2DEnvironment(spec)
        adapter = SafeEnvironmentAdapter(env; adapter_id="safe-vla-kin2d")
        tmp = mktempdir()
        store = load_agent_memory_store(joinpath(tmp, "kin_mem.jsonl"); create=true)

        deltas = [(1.5, 1.2), (1.5, 1.0), (1.5, 0.0), (1.5, -1.0), (1.2, -1.2)]
        media_modes_seen = String[]
        history_lengths = Int[]
        policy = (messages, vobs, step) -> begin
            m = messages[2].content[1]
            push!(media_modes_seen, String(m.type))
            push!(history_lengths, length(vobs.frame_history))
            dx, dy = deltas[step]
            return "<tool_call>\n{\"name\":\"drive\",\"arguments\":{\"dx\":$dx,\"dy\":$dy}}\n</tool_call>"
        end

        trace = run_embodied_vla_episode(
            adapter, policy;
            episode_id="ep-kin2d-video-01",
            media_mode=:video,
            video_history_frames=4,
            fps=2.0,
            max_steps=8,
            memory_store=store,
            height=64, width=64,
        )
        @test trace.success && trace.terminal && !trace.estop_active
        @test length(trace.steps) == 5
        # Step 1 has 1 frame -> image; steps 2..5 have >= 2 frames -> video sliding window capped at 4.
        @test media_modes_seen == ["image", "video", "video", "video", "video"]
        @test history_lengths == [1, 2, 3, 4, 4]
        @test trace.memory_written
        @test verify_embodied_vla_trace(trace)
        rm(tmp; recursive=true)
    end

    @testset "E-stop interlock halts VLA episode and blocks memory writeback" begin
        spec = Kinematic2DSpec(; id="vla-estop-01", max_actions=6)
        env = Kinematic2DEnvironment(spec)
        interlock = Ref(false)
        adapter = SafeEnvironmentAdapter(
            env;
            adapter_id="safe-vla-estop",
            external_interlock=() -> (interlock[], "safety light curtain tripped"),
        )
        tmp = mktempdir()
        store = load_agent_memory_store(joinpath(tmp, "estop_mem.jsonl"); create=true)

        policy = (_messages, _vobs, step) -> begin
            step == 2 && (interlock[] = true)
            return "<tool_call>\n{\"name\":\"drive\",\"arguments\":{\"dx\":1.0,\"dy\":1.0}}\n</tool_call>"
        end

        trace = run_embodied_vla_episode(
            adapter, policy;
            episode_id="ep-estop-01",
            media_mode=:image,
            max_steps=6,
            memory_store=store,
            write_clean_success_only=true,
            height=48, width=48,
        )
        @test !trace.success
        @test trace.estop_active
        @test length(trace.steps) == 2
        @test trace.steps[1].outcome.status === :executed
        @test trace.steps[2].outcome.status === :estopped
        @test !trace.memory_written
        @test isempty(store.records)
        @test verify_embodied_vla_trace(trace)
        rm(tmp; recursive=true)
    end
end
