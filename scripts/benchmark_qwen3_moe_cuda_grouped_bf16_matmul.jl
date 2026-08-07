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
const TOKEN_COUNTS = (16, 32, 64, 128, 256)
const SAMPLES = 15
const INPUT_SEED = 20260827
const LIFEAI_CUDA_EXT = Base.get_extension(LifeAI, :LifeAICUDAExt)

elapsed_seconds(started) = (time_ns() - started) / 1.0e9

function _scalar_grouped_bf16_matmul_kernel!(
    output,
    weights,
    inputs,
    sorted_experts,
    output_dim,
    input_dim,
    pair_count,
)
    linear_index = (CUDA.blockIdx().x - 1) * CUDA.blockDim().x +
        CUDA.threadIdx().x
    if linear_index <= output_dim * pair_count
        output_index = (linear_index - 1) % output_dim + 1
        pair_index = (linear_index - 1) ÷ output_dim + 1
        expert_index = Int(sorted_experts[pair_index])
        accumulator = 0.0f0
        @inbounds for input_index in 1:input_dim
            accumulator = muladd(
                Float32(weights[output_index, input_index, expert_index]),
                Float32(inputs[input_index, pair_index]),
                accumulator,
            )
        end
        @inbounds output[output_index, pair_index] = accumulator
    end
    return
end

function scalar_grouped_bf16_matmul(weights, inputs, sorted_experts)
    output_dim, input_dim, _ = size(weights)
    pair_count = size(inputs, 2)
    output = CUDA.zeros(Float32, output_dim, pair_count)
    threads = 256
    blocks = cld(length(output), threads)
    CUDA.@cuda threads=threads blocks=blocks _scalar_grouped_bf16_matmul_kernel!(
        output,
        weights,
        inputs,
        sorted_experts,
        output_dim,
        input_dim,
        pair_count,
    )
    return output
end

function routes(token_count)
    pair_count = token_count * EXPERTS_PER_TOKEN
    unsorted_experts = Int32[
        mod(pair_index - 1, NUM_EXPERTS) + 1
        for pair_index in 1:pair_count
    ]
    permutation = sortperm(unsorted_experts; alg=Base.Sort.MergeSort)
    sorted_experts = unsorted_experts[permutation]
    expert_counts = Int32[
        count(==(expert_index), sorted_experts)
        for expert_index in 1:NUM_EXPERTS
    ]
    expert_offsets = Vector{Int32}(undef, NUM_EXPERTS + 1)
    expert_offsets[1] = 1
    for expert_index in 1:NUM_EXPERTS
        expert_offsets[expert_index + 1] =
            expert_offsets[expert_index] + expert_counts[expert_index]
    end
    return (
        sorted_experts=CUDA.cu(sorted_experts),
        expert_counts=CUDA.cu(expert_counts),
        expert_offsets=CUDA.cu(expert_offsets),
        host_counts=expert_counts,
    )
end

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

function benchmark_case(weights, token_count)
    route_data = routes(token_count)
    pair_count = token_count * EXPERTS_PER_TOKEN
    inputs = CUDA.cu(BFloat16.(randn(
        Xoshiro(INPUT_SEED + token_count),
        Float32,
        D_MODEL,
        pair_count,
    )))
    scalar = () -> scalar_grouped_bf16_matmul(
        weights,
        inputs,
        route_data.sorted_experts,
    )
    wmma = () -> LIFEAI_CUDA_EXT.qwen3_cuda_grouped_bf16_matmul(
        weights,
        inputs,
        route_data.sorted_experts,
        route_data.expert_counts,
        route_data.expert_offsets,
    )

    scalar_cold_started = time_ns()
    scalar_output = synchronize_call(scalar)
    scalar_cold_seconds = elapsed_seconds(scalar_cold_started)
    wmma_cold_started = time_ns()
    wmma_output = synchronize_call(wmma)
    wmma_cold_seconds = elapsed_seconds(wmma_cold_started)
    scalar_output, scalar_samples = measured_samples(scalar)
    wmma_output, wmma_samples = measured_samples(wmma)
    scalar_median = median(scalar_samples)
    wmma_median = median(wmma_samples)
    padded_pairs = sum(
        count == 0 ? 0 : cld(Int(count), 8) * 8
        for count in route_data.host_counts
    )

    return (;
        token_count,
        pair_count,
        active_experts=count(>(0), route_data.host_counts),
        padded_pairs,
        padding_ratio=padded_pairs / pair_count,
        scalar_cold_seconds,
        wmma_cold_seconds,
        scalar_cuda_seconds=scalar_samples,
        scalar_cuda_median_seconds=scalar_median,
        wmma_cuda_seconds=wmma_samples,
        wmma_cuda_median_seconds=wmma_median,
        wmma_over_scalar_speedup=scalar_median / wmma_median,
        max_abs_wmma_vs_scalar=maximum(abs.(
            Array(wmma_output) .- Array(scalar_output),
        )),
    )
end

function main(output_path)
    CUDA.functional() || error("CUDA is not functional")
    CUDA.allowscalar(false)
    weights = CUDA.fill(
        BFloat16(0.001),
        EXPERT_HIDDEN_DIM,
        D_MODEL,
        NUM_EXPERTS,
    )
    CUDA.synchronize()
    cases = [benchmark_case(weights, token_count) for token_count in TOKEN_COUNTS]
    result = (;
        schema_version=2,
        benchmark="qwen3_moe_cuda_grouped_bf16_matmul",
        environment=(;
            julia_version=string(VERSION),
            device=CUDA.name(CUDA.device()),
            runtime_version=string(CUDA.runtime_version()),
            driver_version=string(CUDA.driver_version()),
        ),
        shape=(;
            input_dim=D_MODEL,
            output_dim=EXPERT_HIDDEN_DIM,
            num_experts=NUM_EXPERTS,
            experts_per_token=EXPERTS_PER_TOKEN,
            bf16_weight_bytes=sizeof(eltype(weights)) * length(weights),
            wmma_tile=(m=32, n=8, k=16),
        ),
        measurement=(;
            input_seed=INPUT_SEED,
            samples=SAMPLES,
            includes_device_padding_pack_and_unpack=true,
            excludes_route_bucketing=true,
        ),
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
    "cuda_4090d_grouped_bf16_matmul.json",
) : ARGS[1]
main(output_path)
