#!/usr/bin/env julia

using JSON3
using LifeAI
using Lux
using Random: Xoshiro
using Reactant
using Statistics: median

const D_MODEL = 128
const EXPERT_HIDDEN_DIM = 64
const NUM_EXPERTS = 128
const EXPERTS_PER_TOKEN = 8
const SEQUENCE_LENGTH = 64
const BATCH_SIZE = 1
const SAMPLES = 7
const PARAMETER_SEED = 20260811
const INPUT_SEED = 20260812

elapsed_seconds(started) = (time_ns() - started) / 1.0e9

function main(output_path)
    backend = get(ENV, "LIFEAI_XLA_BACKEND", "cpu")
    Reactant.set_default_backend(backend)
    layer = Qwen3SparseMoE(
        D_MODEL,
        EXPERT_HIDDEN_DIM,
        NUM_EXPERTS,
        EXPERTS_PER_TOKEN,
    )
    parameters, state = Lux.setup(Xoshiro(PARAMETER_SEED), layer)
    x = randn(
        Xoshiro(INPUT_SEED),
        Float32,
        D_MODEL,
        SEQUENCE_LENGTH,
        BATCH_SIZE,
    )
    host = qwen3_moe_forward_with_stats(layer, x, parameters)
    dense = qwen3_dense_expert_reference(
        reshape(x, D_MODEL, :),
        host.routing,
        parameters.experts,
    )

    transfer_started = time_ns()
    x_device = Reactant.to_rarray(x)
    router_device = Reactant.to_rarray(parameters.gate.weight)
    gate_device = Reactant.to_rarray(parameters.experts.gate_proj)
    up_device = Reactant.to_rarray(parameters.experts.up_proj)
    down_device = Reactant.to_rarray(parameters.experts.down_proj)
    transfer_seconds = elapsed_seconds(transfer_started)

    kernel = (tokens, router, gate, up, down) -> first(layer(
        tokens,
        (;
            gate=(; weight=router),
            experts=(; gate_proj=gate, up_proj=up, down_proj=down),
        ),
        state,
    ))
    compile_started = time_ns()
    compiled = Reactant.@compile kernel(
        x_device,
        router_device,
        gate_device,
        up_device,
        down_device,
    )
    compile_seconds = elapsed_seconds(compile_started)

    arguments = (
        x_device,
        router_device,
        gate_device,
        up_device,
        down_device,
    )
    Array(compiled(arguments...))
    samples = Float64[]
    actual = nothing
    for _ in 1:SAMPLES
        started = time_ns()
        actual = Array(compiled(arguments...))
        push!(samples, elapsed_seconds(started))
    end

    routed_pairs = EXPERTS_PER_TOKEN * SEQUENCE_LENGTH * BATCH_SIZE
    dense_pairs = NUM_EXPERTS * SEQUENCE_LENGTH * BATCH_SIZE
    result = (;
        schema_version=1,
        benchmark="qwen3_moe_compact_sparse_dispatch_xla",
        environment=(;
            julia_version=string(VERSION),
            backend,
            cpu=Sys.CPU_NAME,
        ),
        shape=(;
            d_model=D_MODEL,
            expert_hidden_dim=EXPERT_HIDDEN_DIM,
            num_experts=NUM_EXPERTS,
            experts_per_token=EXPERTS_PER_TOKEN,
            sequence_length=SEQUENCE_LENGTH,
            batch_size=BATCH_SIZE,
        ),
        dispatch=(;
            compact_route_entries=routed_pairs,
            sparse_token_expert_pairs=routed_pairs,
            dense_token_expert_pairs=dense_pairs,
            pair_reduction=dense_pairs / routed_pairs,
        ),
        results=(;
            transfer_seconds,
            compile_seconds,
            warmup_runs=1,
            samples_seconds=samples,
            steady_median_seconds=median(samples),
            max_abs_vs_dense_oracle=maximum(
                abs.(actual .- reshape(dense, size(actual))),
            ),
        ),
    )
    mkpath(dirname(output_path))
    open(output_path, "w") do io
        write(io, JSON3.write(result))
        write(io, '\n')
    end
    println(JSON3.write(result))
end

output_path = isempty(ARGS) ? joinpath(
    "benchmark_results",
    "qwen3_moe_sparse_dispatch",
    "xla_cpu_reference.json",
) : ARGS[1]
main(output_path)
