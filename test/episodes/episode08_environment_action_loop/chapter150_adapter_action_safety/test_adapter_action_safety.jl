using Test
using JSON3
using LifeAI

# Adapter-agnostic mock environment (completely independent of GridWorld) to
# prove timeout, execution failure, e-stop, interlock, and idempotency contracts.
mutable struct _MockActuatorEnv <: AbstractAgentEnvironment
    id::String
    step_count::Int
    calls::Int
    state_value::String
    mode::Symbol # :ok, :slow, :not_executed_error, :uncertain_error, :unexpected_error, :arg_error
    delay_seconds::Float64
    gate::Union{Nothing,Base.Event}
end

_MockActuatorEnv(id="mock-actuator") = _MockActuatorEnv(id, 0, 0, "idle", :ok, 0.0, nothing)

function LifeAI.observe_agent_environment(env::_MockActuatorEnv)
    state_json = JSON3.write((; id=env.id, step=env.step_count, state=env.state_value))
    state_sha = LifeAI._sha256_hex(state_json)
    rendered = JSON3.write((;
        environment="mock_actuator",
        task_id=env.id,
        step=env.step_count,
        state=env.state_value,
        terminal=false,
        success=false,
        state_sha256=state_sha,
    ))
    return AgentEnvironmentObservation(
        env.id,
        env.step_count,
        false,
        false,
        state_sha,
        rendered,
        LifeAI._sha256_hex(rendered),
    )
end

function LifeAI.reset_agent_environment!(env::_MockActuatorEnv)
    env.step_count = 0
    env.calls = 0
    env.state_value = "idle"
    env.mode = :ok
    env.delay_seconds = 0.0
    env.gate = nothing
    return env
end

function LifeAI.step_agent_environment!(env::_MockActuatorEnv, action::AgentEnvironmentAction)
    env.calls += 1
    if env.mode === :slow
        if env.gate !== nothing
            wait(env.gate)
        else
            sleep(env.delay_seconds)
        end
    elseif env.mode === :not_executed_error
        throw(AgentActionExecutionError(
            "link_offline",
            "actuator link down before dispatch";
            side_effect_certainty=:not_executed,
        ))
    elseif env.mode === :uncertain_error
        env.state_value = "possibly_moved"
        throw(AgentActionExecutionError(
            "ack_lost",
            "transport dropped after command send";
            side_effect_certainty=:uncertain,
        ))
    elseif env.mode === :unexpected_error
        error("unhandled driver fault")
    elseif env.mode === :arg_error
        throw(ArgumentError("invalid torque setpoint"))
    end
    before = observe_agent_environment(env)
    env.step_count += 1
    env.state_value = action.name * ":" * action.value
    after = observe_agent_environment(env)
    t_sha = LifeAI._sha256_hex(JSON3.write((;
        step=env.step_count,
        action=(; name=action.name, value=action.value),
        before=before.state_sha256,
        after=after.state_sha256,
    )))
    return AgentEnvironmentTransition(
        env.step_count,
        action,
        true,
        "actuated",
        before.state_sha256,
        after.state_sha256,
        after,
        t_sha,
    )
end

@testset "Chapter 150 — Adapter-agnostic action safety contracts" begin
    @testset "Policy and request validation fail closed" begin
        @test_throws ArgumentError ActionSafetyPolicy(; default_timeout_seconds=0.0)
        @test_throws ArgumentError ActionSafetyPolicy(; default_timeout_seconds=-1.0)
        @test_throws ArgumentError ActionSafetyPolicy(; default_timeout_seconds=Inf)
        @test_throws ArgumentError ActionSafetyPolicy(; default_timeout_seconds=true)
        @test_throws ArgumentError ActionSafetyPolicy(; default_timeout_seconds=10.0, max_timeout_seconds=5.0)
        @test_throws ArgumentError ActionSafetyPolicy(; max_consecutive_failures=0)
        @test_throws ArgumentError ActionSafetyPolicy(; max_consecutive_failures=true)
        @test_throws ArgumentError ActionSafetyPolicy(; max_idempotency_entries=0)
        @test_throws ArgumentError AgentActionExecutionError("", "msg")
        @test_throws ArgumentError AgentActionExecutionError("code", "msg"; side_effect_certainty=:committed)

        act = AgentEnvironmentAction("set_pose", "stand")
        @test_throws ArgumentError SafeActionRequest(act; idempotency_key="")
        @test_throws ArgumentError SafeActionRequest(act; idempotency_key="bad key with space")
        @test_throws ArgumentError SafeActionRequest(act; timeout_seconds=-0.5)
        @test_throws ArgumentError SafeActionRequest(act; timeout_seconds=false)
        @test_throws ArgumentError SafeActionRequest(AgentEnvironmentAction("", "x"); idempotency_key="k1")

        p1 = ActionSafetyPolicy(; default_timeout_seconds=1.5, max_timeout_seconds=5.0)
        p2 = ActionSafetyPolicy(; default_timeout_seconds=1.5, max_timeout_seconds=5.0)
        @test p1.sha256 == p2.sha256
        @test length(p1.sha256) == 64
    end

    @testset "Idempotency replay, conflict rejection, and bounded cache" begin
        env = _MockActuatorEnv("arm-01")
        policy = ActionSafetyPolicy(;
            default_timeout_seconds=1.0,
            max_timeout_seconds=2.0,
            require_idempotency_key=true,
            max_idempotency_entries=2,
        )
        adapter = SafeEnvironmentAdapter(env; adapter_id="arm-safe", policy)

        # Missing required idempotency key fails closed without calling adapter.
        missing = execute_safe_action!(adapter, AgentEnvironmentAction("grip", "close"))
        @test missing.status === :rejected
        @test missing.failure_mode === :precondition
        @test missing.side_effect_certainty === :not_executed
        @test env.calls == 0

        # Exceeding policy max timeout fails closed without calling adapter.
        over_t = execute_safe_action!(
            adapter,
            AgentEnvironmentAction("grip", "close");
            idempotency_key="k-over",
            timeout_seconds=10.0,
        )
        @test over_t.status === :rejected
        @test over_t.failure_mode === :precondition
        @test env.calls == 0

        # First execution succeeds and commits once.
        first_out = execute_safe_action!(
            adapter,
            AgentEnvironmentAction("grip", "close");
            idempotency_key="k1",
        )
        @test first_out.status === :executed
        @test first_out.failure_mode === :none
        @test first_out.side_effect_certainty === :committed
        @test first_out.transition !== nothing
        @test env.calls == 1

        # Exact retry replays canonical outcome WITHOUT re-executing underlying adapter.
        replay_out = execute_safe_action!(
            adapter,
            AgentEnvironmentAction("grip", "close");
            idempotency_key="k1",
        )
        @test replay_out.status === :replayed
        @test replay_out.replayed_from_sequence == first_out.sequence
        @test replay_out.transition == first_out.transition
        @test replay_out.side_effect_certainty === :committed
        @test env.calls == 1

        # Reusing key with a DIFFERENT action payload fails closed.
        conflict = execute_safe_action!(
            adapter,
            AgentEnvironmentAction("grip", "open");
            idempotency_key="k1",
        )
        @test conflict.status === :rejected
        @test conflict.failure_mode === :idempotency_conflict
        @test conflict.side_effect_certainty === :not_executed
        @test conflict.replayed_from_sequence == first_out.sequence
        @test env.calls == 1

        # Fill second cache slot, then third distinct key is rejected (no silent eviction).
        second_out = execute_safe_action!(
            adapter,
            AgentEnvironmentAction("grip", "open");
            idempotency_key="k2",
        )
        @test second_out.status === :executed
        @test env.calls == 2

        overflow = execute_safe_action!(
            adapter,
            AgentEnvironmentAction("grip", "hold");
            idempotency_key="k3",
        )
        @test overflow.status === :rejected
        @test overflow.failure_mode === :precondition
        @test env.calls == 2

        # Audit verifies cleanly via JSON round-trip.
        audit = safe_adapter_audit_payload(adapter)
        reparsed = JSON3.read(JSON3.write(audit))
        verified = verify_safe_adapter_audit(reparsed)
        @test verified.verified_outcomes == length(adapter.audit_log)
        @test verified.audit_sha256 == audit.audit_sha256
    end

    @testset "Execution failures, certainty classification, and consecutive-failure e-stop" begin
        env = _MockActuatorEnv("arm-02")
        policy = ActionSafetyPolicy(;
            default_timeout_seconds=1.0,
            max_consecutive_failures=2,
            estop_on_uncertain_failure=true,
        )
        adapter = SafeEnvironmentAdapter(env; adapter_id="arm-fail", policy)

        # Deterministic :not_executed failure does NOT latch e-stop on first hit.
        env.mode = :not_executed_error
        f1 = execute_safe_action!(
            adapter,
            AgentEnvironmentAction("step", "fwd");
            idempotency_key="f1",
        )
        @test f1.status === :failed
        @test f1.failure_mode === :execution_error
        @test f1.side_effect_certainty === :not_executed
        @test !f1.estop_triggered
        @test !adapter.estop.active

        # Replaying a :not_executed key returns the deterministic failure without re-calling.
        calls_before = env.calls
        f1_replay = execute_safe_action!(
            adapter,
            AgentEnvironmentAction("step", "fwd");
            idempotency_key="f1",
        )
        @test f1_replay.status === :replayed
        @test f1_replay.side_effect_certainty === :not_executed
        @test env.calls == calls_before

        # Second consecutive failure hits max_consecutive_failures=2 -> latches e-stop.
        env.mode = :arg_error
        f2 = execute_safe_action!(
            adapter,
            AgentEnvironmentAction("step", "fwd");
            idempotency_key="f2",
        )
        @test f2.status === :failed
        @test f2.failure_mode === :precondition
        @test f2.estop_triggered
        @test adapter.estop.active
        @test adapter.estop.source == "consecutive_failures"

        # While e-stop is active, even a valid action is blocked fail-closed.
        env.mode = :ok
        blocked = execute_safe_action!(
            adapter,
            AgentEnvironmentAction("step", "fwd");
            idempotency_key="f3",
        )
        @test blocked.status === :estopped
        @test blocked.failure_mode === :estop_active
        @test env.calls == calls_before + 1

        # Reset refuses while e-stop is active.
        @test_throws ArgumentError reset_agent_environment!(adapter)

        # Clear requires non-empty operator & reason.
        @test_throws ArgumentError clear_agent_estop!(adapter; operator="", reason="ok")
        @test_throws ArgumentError clear_agent_estop!(adapter; operator="bgl", reason=" ")
        clear_agent_estop!(adapter; operator="bgl", reason="inspected hardware")
        @test !adapter.estop.active
        @test adapter.estop.clear_count == 1

        # Uncertain failure immediately latches e-stop and forbids key replay.
        env.mode = :uncertain_error
        u1 = execute_safe_action!(
            adapter,
            AgentEnvironmentAction("step", "turn");
            idempotency_key="u1",
        )
        @test u1.status === :failed
        @test u1.failure_mode === :uncertain_side_effect
        @test u1.side_effect_certainty === :uncertain
        @test u1.estop_triggered
        @test adapter.estop.active

        clear_agent_estop!(adapter; operator="bgl", reason="verified state")
        env.mode = :ok
        u1_retry = execute_safe_action!(
            adapter,
            AgentEnvironmentAction("step", "turn");
            idempotency_key="u1",
        )
        @test u1_retry.status === :rejected
        @test u1_retry.failure_mode === :uncertain_side_effect
        @test u1_retry.side_effect_certainty === :uncertain

        # Unexpected exception defaults to :uncertain and latches e-stop.
        env.mode = :unexpected_error
        u2 = execute_safe_action!(
            adapter,
            AgentEnvironmentAction("step", "jump");
            idempotency_key="u2",
        )
        @test u2.status === :failed
        @test u2.side_effect_certainty === :uncertain
        @test u2.estop_triggered
    end

    @testset "Timeout latches e-stop, blocks clear until finished, and verifies offline" begin
        env = _MockActuatorEnv("arm-timeout")
        gate = Base.Event()
        env.mode = :slow
        env.gate = gate
        policy = ActionSafetyPolicy(;
            default_timeout_seconds=0.05,
            max_timeout_seconds=1.0,
            estop_on_timeout=true,
        )
        adapter = SafeEnvironmentAdapter(env; adapter_id="arm-to", policy)

        t_out = execute_safe_action!(
            adapter,
            AgentEnvironmentAction("move_arm", "home");
            idempotency_key="t1",
        )
        @test t_out.status === :timed_out
        @test t_out.failure_mode === :timeout
        @test t_out.side_effect_certainty === :uncertain
        @test t_out.estop_triggered
        @test adapter.estop.active
        @test adapter.estop.source == "timeout"

        # Cannot clear e-stop without force=true while timed-out task is still stuck.
        @test_throws ArgumentError clear_agent_estop!(
            adapter;
            operator="bgl",
            reason="try too early",
        )

        # Release the stalled worker and let it terminate.
        notify(gate)
        @test timedwait(() -> all(istaskdone, adapter.in_flight_tasks), 2.0) === :ok

        clear_agent_estop!(adapter; operator="bgl", reason="worker settled")
        @test !adapter.estop.active

        # Replaying timed-out key fails closed with :uncertain_side_effect.
        t1_replay = execute_safe_action!(
            adapter,
            AgentEnvironmentAction("move_arm", "home");
            idempotency_key="t1",
        )
        @test t1_replay.status === :rejected
        @test t1_replay.failure_mode === :uncertain_side_effect

        # New key works once cleared and healthy.
        env.mode = :ok
        env.gate = nothing
        ok_out = execute_safe_action!(
            adapter,
            AgentEnvironmentAction("move_arm", "home");
            idempotency_key="t2",
        )
        @test ok_out.status === :executed
        @test ok_out.transition !== nothing

        # Tampering with serialized audit fails verification.
        audit = safe_adapter_audit_payload(adapter)
        verified = verify_safe_adapter_audit(JSON3.read(JSON3.write(audit)))
        @test verified.verified_outcomes == length(adapter.audit_log)
        @test verified.verified_events == length(adapter.safety_events)

        tampered = JSON3.read(JSON3.write(audit), Dict{String,Any})
        tampered["outcomes"][1]["status"] = "executed"
        @test_throws ArgumentError verify_safe_adapter_audit(
            JSON3.read(JSON3.write(tampered)),
        )
    end

    @testset "External interlock and manual e-stop override dispatch and clear" begin
        interlock_tripped = Ref(false)
        env = _MockActuatorEnv("arm-interlock")
        adapter = SafeEnvironmentAdapter(
            env;
            adapter_id="arm-il",
            external_interlock=() -> (interlock_tripped[], "bumper contact"),
        )

        # Manual e-stop.
        trigger_agent_estop!(adapter, "operator pressed red button")
        @test adapter.estop.active
        @test adapter.estop.source == "manual"
        b1 = execute_safe_action!(
            adapter,
            AgentEnvironmentAction("move", "east");
            idempotency_key="m1",
        )
        @test b1.status === :estopped
        @test env.calls == 0

        clear_agent_estop!(adapter; operator="bgl", reason="button released")
        @test !adapter.estop.active

        # External interlock trips before dispatch -> latches e-stop and blocks clear.
        interlock_tripped[] = true
        b2 = execute_safe_action!(
            adapter,
            AgentEnvironmentAction("move", "east");
            idempotency_key="m2",
        )
        @test b2.status === :estopped
        @test b2.estop_triggered
        @test adapter.estop.source == "external_interlock"
        @test env.calls == 0
        @test_throws ArgumentError clear_agent_estop!(
            adapter;
            operator="bgl",
            reason="cannot clear while tripped",
        )

        interlock_tripped[] = false
        clear_agent_estop!(adapter; operator="bgl", reason="bumper cleared")
        ok = execute_safe_action!(
            adapter,
            AgentEnvironmentAction("move", "east");
            idempotency_key="m2",
        )
        @test ok.status === :executed
        @test env.calls == 1
    end

    @testset "GridWorld integration through safe tool registry" begin
        spec = GridWorldSpec(;
            id="safe-gw-01",
            width=3,
            height=2,
            start=(1, 1),
            goal=(3, 1),
            walls=[(2, 2)],
            max_actions=5,
        )
        gw = GridWorldEnvironment(spec)
        adapter = SafeEnvironmentAdapter(gw; adapter_id="gw-safe")
        registry = safe_gridworld_tool_registry(adapter)

        # First move via tool registry.
        c1 = Qwen3ToolCall(
            "move",
            parse_qwen3_json("""{"direction":"east","idempotency_key":"step-1"}"""),
            """{"direction":"east","idempotency_key":"step-1"}""",
        )
        r1 = invoke_agent_tool(registry, c1)
        @test r1.ok
        p1 = JSON3.read(r1.output)
        @test p1.status == "executed"
        @test p1.observation.position.x == 2 && p1.observation.position.y == 1
        @test gw.actions == 1

        # Duplicate tool call with same idempotency_key replays without advancing GridWorld.
        r1_dup = invoke_agent_tool(registry, c1)
        @test r1_dup.ok
        p1_dup = JSON3.read(r1_dup.output)
        @test p1_dup.status == "replayed"
        @test p1_dup.replayed_from_sequence == 1
        @test gw.actions == 1

        # Second move reaches goal.
        c2 = Qwen3ToolCall(
            "move",
            parse_qwen3_json("""{"direction":"east","idempotency_key":"step-2"}"""),
            """{"direction":"east","idempotency_key":"step-2"}""",
        )
        r2 = invoke_agent_tool(registry, c2)
        @test r2.ok
        p2 = JSON3.read(r2.output)
        @test p2.status == "executed"
        @test p2.observation.terminal == true
        @test p2.observation.success == true
        @test gw.actions == 2

        # Reset clears idempotency keys and resets GridWorld cleanly.
        reset_agent_environment!(adapter)
        @test gw.actions == 0
        @test isempty(adapter.idempotency_cache)

        audit = safe_adapter_audit_payload(adapter)
        verified = verify_safe_adapter_audit(JSON3.read(JSON3.write(audit)))
        @test verified.verified_outcomes == 3
    end
end
