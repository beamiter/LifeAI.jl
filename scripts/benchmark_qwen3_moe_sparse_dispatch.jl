#!/usr/bin/env julia

using JSON3
using LifeAI
using LinearAlgebra
using Lux
using Random: Xoshiro
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

function elapsed_samples(f, count)
    return [(@elapsed f()) for _ in 1:count]
end

function main(output_path)
    BLAS.set_num_threads(1)
    layer = Qwen3SparseMoE(
        D_MODEL,
        EXPERT_HIDDEN_DIM,
        NUM_EXPERTS,
        EXPERTS_PER_TOKEN,
    )
    parameters, _ = Lux.setup(Xoshiro(PARAMETER_SEED), layer)
    x = randn(
        Xoshiro(INPUT_SEED),
        Float32,
        D_MODEL,
        SEQUENCE_LENGTH,
        BATCH_SIZE,
    )
    routed = qwen3_moe_forward_with_stats(layer, x, parameters)
    tokens = reshape(x, D_MODEL, :)

    # Warm both paths before collecting steady host timings.
    qwen3_sparse_expert_dispatch(tokens, routed.routing, parameters.experts)
    qwen3_dense_expert_reference(tokens, routed.routing, parameters.experts)
    sparse_samples = elapsed_samples(
        () -> qwen3_sparse_expert_dispatch(
            tokens,
            routed.routing,
            parameters.experts,
        ),
        SAMPLES,
    )
    dense_samples = elapsed_samples(
        () -> qwen3_dense_expert_reference(
            tokens,
            routed.routing,
            parameters.experts,
        ),
        SAMPLES,
    )
    sparse_output, stats = qwen3_sparse_expert_dispatch(
        tokens,
        routed.routing,
        parameters.experts,
    )
    dense_output = qwen3_dense_expert_reference(
        tokens,
        routed.routing,
        parameters.experts,
    )
    sparse_median = median(sparse_samples)
    dense_median = median(dense_samples)
    result = (;
        schema_version=1,
        benchmark="qwen3_moe_sparse_dispatch_cpu_reference",
        environment=(;
            julia_version=string(VERSION),
            cpu=Sys.CPU_NAME,
            blas_threads=BLAS.get_num_threads(),
        ),
        shape=(;
            d_model=D_MODEL,
            expert_hidden_dim=EXPERT_HIDDEN_DIM,
            num_experts=NUM_EXPERTS,
            experts_per_token=EXPERTS_PER_TOKEN,
            sequence_length=SEQUENCE_LENGTH,
            batch_size=BATCH_SIZE,
        ),
        measurement=(;
            warmup_runs=1,
            samples=SAMPLES,
            parameter_seed=PARAMETER_SEED,
            input_seed=INPUT_SEED,
        ),
        dispatch=(;
            active_experts=stats.active_expert_count,
            sparse_token_expert_pairs=stats.routed_token_expert_pairs,
            dense_token_expert_pairs=stats.dense_token_expert_pairs,
            pair_reduction=stats.dense_token_expert_pairs /
                stats.routed_token_expert_pairs,
        ),
        results=(;
            sparse_seconds=sparse_samples,
            dense_seconds=dense_samples,
            sparse_median_seconds=sparse_median,
            dense_median_seconds=dense_median,
            speedup=dense_median / sparse_median,
            max_abs=maximum(abs.(sparse_output .- dense_output)),
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
    "cpu_reference.json",
) : ARGS[1]
main(output_path)
