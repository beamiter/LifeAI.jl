#!/usr/bin/env julia

using BFloat16s: BFloat16
using CUDA
using JSON3
using LifeAI
using Random: Xoshiro
using Statistics: median

const D_MODEL = 2_048
const EXPERT_HIDDEN_DIM = 768
const NUM_EXPERTS = 128
const EXPERTS_PER_TOKEN = 8
const TOKEN_COUNTS = (1, 8, 16, 32, 64)
const SAMPLES = 15
const INPUT_SEED = 20260813
const LIFEAI_CUDA_EXT = Base.get_extension(LifeAI, :LifeAICUDAExt)

elapsed_seconds(started) = (time_ns() - started) / 1.0e9

function synchronize_call(f)
    output = f()
    CUDA.synchronize()
    return output
end

function measured_samples(f)
    samples = Float64[]
    output = nothing
    for _ in 1:SAMPLES
        started = time_ns()
        output = synchronize_call(f)
        push!(samples, elapsed_seconds(started))
    end
    return output, samples
end

function routes(token_count)
    expert_indices = Matrix{Int32}(undef, EXPERTS_PER_TOKEN, token_count)
    for token_index in 1:token_count, slot in 1:EXPERTS_PER_TOKEN
        route_offset = (token_index - 1) * EXPERTS_PER_TOKEN + slot - 1
        expert_indices[slot, token_index] =
            Int32(mod(route_offset, NUM_EXPERTS) + 1)
    end
    routing_weights = fill(
        Float32(1 / EXPERTS_PER_TOKEN),
        EXPERTS_PER_TOKEN,
        token_count,
    )
    return CUDA.cu(expert_indices), CUDA.cu(routing_weights)
end

function benchmark_case(expert_parameters_bf16, expert_parameters_f32, token_count)
    tokens = CUDA.cu(randn(
        Xoshiro(INPUT_SEED + token_count),
        Float32,
        D_MODEL,
        token_count,
    ))
    expert_indices, routing_weights = routes(token_count)
    bf16 = () -> LIFEAI_CUDA_EXT.qwen3_cuda_indexed_sparse_expert_dispatch(
        tokens,
        expert_indices,
        routing_weights,
        expert_parameters_bf16,
    )
    f32 = () -> LIFEAI_CUDA_EXT.qwen3_cuda_indexed_sparse_expert_dispatch(
        tokens,
        expert_indices,
        routing_weights,
        expert_parameters_f32,
    )
    bucketed_bf16 = () ->
        LIFEAI_CUDA_EXT.qwen3_cuda_bucketed_sparse_expert_dispatch(
            tokens,
            expert_indices,
            routing_weights,
            expert_parameters_bf16,
        )

    bf16_cold_started = time_ns()
    bf16_output = synchronize_call(bf16)
    bf16_cold_seconds = elapsed_seconds(bf16_cold_started)
    f32_cold_started = time_ns()
    f32_output = synchronize_call(f32)
    f32_cold_seconds = elapsed_seconds(f32_cold_started)
    bucketed_cold_started = time_ns()
    bucketed_output = synchronize_call(bucketed_bf16)
    bucketed_bf16_cold_seconds = elapsed_seconds(bucketed_cold_started)

    bf16_output, bf16_samples = measured_samples(bf16)
    f32_output, f32_samples = measured_samples(f32)
    bucketed_output, bucketed_samples = measured_samples(bucketed_bf16)
    bf16_host = Array(bf16_output)
    f32_host = Array(f32_output)
    bucketed_host = Array(bucketed_output)
    bf16_median = median(bf16_samples)
    f32_median = median(f32_samples)
    bucketed_median = median(bucketed_samples)

    return (;
        token_count,
        routed_pairs=token_count * EXPERTS_PER_TOKEN,
        production_strategy=
            LIFEAI_CUDA_EXT._qwen3_cuda_use_bucketed_dispatch(
                D_MODEL,
                EXPERT_HIDDEN_DIM,
                token_count,
            ) ? "bucketed" : "indexed",
        bf16_cold_seconds,
        f32_cold_seconds,
        bucketed_bf16_cold_seconds,
        bf16_cuda_seconds=bf16_samples,
        bf16_cuda_median_seconds=bf16_median,
        f32_cuda_seconds=f32_samples,
        f32_cuda_median_seconds=f32_median,
        bucketed_bf16_cuda_seconds=bucketed_samples,
        bucketed_bf16_cuda_median_seconds=bucketed_median,
        bf16_over_f32_speedup=f32_median / bf16_median,
        bucketed_over_indexed_speedup=bf16_median / bucketed_median,
        max_abs_bf16_vs_f32=maximum(abs.(bf16_host .- f32_host)),
        max_abs_bucketed_vs_indexed=maximum(abs.(bucketed_host .- bf16_host)),
    )
end

function main(output_path)
    CUDA.functional() || error("CUDA is not functional")
    CUDA.allowscalar(false)
    parameter_value = BFloat16(0.001)
    expert_parameters_bf16 = (;
        gate_proj=CUDA.fill(
            parameter_value,
            EXPERT_HIDDEN_DIM,
            D_MODEL,
            NUM_EXPERTS,
        ),
        up_proj=CUDA.fill(
            parameter_value,
            EXPERT_HIDDEN_DIM,
            D_MODEL,
            NUM_EXPERTS,
        ),
        down_proj=CUDA.fill(
            parameter_value,
            D_MODEL,
            EXPERT_HIDDEN_DIM,
            NUM_EXPERTS,
        ),
    )
    expert_parameters_f32 = map(
        array -> Float32.(array),
        expert_parameters_bf16,
    )
    CUDA.synchronize()
    cases = [
        benchmark_case(
            expert_parameters_bf16,
            expert_parameters_f32,
            token_count,
        ) for token_count in TOKEN_COUNTS
    ]
    bf16_parameter_bytes = sum(
        array -> sizeof(eltype(array)) * length(array),
        expert_parameters_bf16,
    )
    f32_parameter_bytes = sum(
        array -> sizeof(eltype(array)) * length(array),
        expert_parameters_f32,
    )
    result = (;
        schema_version=1,
        benchmark="qwen3_moe_cuda_bf16_projection_widths",
        environment=(;
            julia_version=string(VERSION),
            device=CUDA.name(CUDA.device()),
            runtime_version=string(CUDA.runtime_version()),
            driver_version=string(CUDA.driver_version()),
        ),
        shape=(;
            d_model=D_MODEL,
            expert_hidden_dim=EXPERT_HIDDEN_DIM,
            num_experts=NUM_EXPERTS,
            experts_per_token=EXPERTS_PER_TOKEN,
            bf16_expert_parameter_bytes=bf16_parameter_bytes,
            f32_expert_parameter_bytes=f32_parameter_bytes,
        ),
        measurement=(; input_seed=INPUT_SEED, samples=SAMPLES),
        cases,
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
    "cuda_4090d_bf16_projection_widths.json",
) : ARGS[1]
main(output_path)
