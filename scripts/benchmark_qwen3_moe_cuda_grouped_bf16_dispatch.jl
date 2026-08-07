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
const TOKEN_COUNTS = (32, 64, 128, 256)
const SAMPLES = 50
const INPUT_SEED = 20260832
const LIFEAI_CUDA_EXT = Base.get_extension(LifeAI, :LifeAICUDAExt)

elapsed_seconds(started) = (time_ns() - started) / 1.0e9

function _legacy_pack_expert_major_tokens_kernel!(
    grouped_tokens,
    tokens,
    route_permutation,
    d_model,
    experts_per_token,
    pair_count,
)
    linear_index = (CUDA.blockIdx().x - 1) * CUDA.blockDim().x +
        CUDA.threadIdx().x
    if linear_index <= d_model * pair_count
        model_index = (linear_index - 1) % d_model + 1
        bucket_pair = (linear_index - 1) ÷ d_model + 1
        original_pair = Int(route_permutation[bucket_pair])
        token_index = (original_pair - 1) ÷ experts_per_token + 1
        @inbounds grouped_tokens[model_index, bucket_pair] =
            BFloat16(tokens[model_index, token_index])
    end
    return
end

function routes(token_count)
    expert_indices = Matrix{Int32}(undef, EXPERTS_PER_TOKEN, token_count)
    for token_index in 1:token_count, slot in 1:EXPERTS_PER_TOKEN
        pair_index = (token_index - 1) * EXPERTS_PER_TOKEN + slot
        expert_indices[slot, token_index] =
            Int32(mod(pair_index - 1, NUM_EXPERTS) + 1)
    end
    routing_weights = fill(
        Float32(1 / EXPERTS_PER_TOKEN),
        EXPERTS_PER_TOKEN,
        token_count,
    )
    return CUDA.cu(expert_indices), CUDA.cu(routing_weights)
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

function legacy_grouped_bf16_dispatch(
    tokens,
    expert_indices,
    routing_weights,
    expert_parameters,
)
    num_experts = size(expert_parameters.gate_proj, 3)
    d_model, num_tokens = size(tokens)
    hidden_dim = size(expert_parameters.gate_proj, 1)
    experts_per_token = size(expert_indices, 1)
    pair_count = experts_per_token * num_tokens
    buckets = LIFEAI_CUDA_EXT.qwen3_cuda_bucket_routes(
        expert_indices,
        num_experts,
    )
    grouped_tokens = CUDA.zeros(BFloat16, d_model, pair_count)
    threads = 256
    token_blocks = cld(length(grouped_tokens), threads)
    CUDA.@cuda threads=threads blocks=token_blocks _legacy_pack_expert_major_tokens_kernel!(
        grouped_tokens,
        tokens,
        buckets.route_permutation,
        d_model,
        experts_per_token,
        pair_count,
    )
    gate = LIFEAI_CUDA_EXT.qwen3_cuda_grouped_bf16_matmul(
        expert_parameters.gate_proj,
        grouped_tokens,
        buckets.sorted_experts,
        buckets.expert_counts,
        buckets.expert_offsets,
    )
    up = LIFEAI_CUDA_EXT.qwen3_cuda_grouped_bf16_matmul(
        expert_parameters.up_proj,
        grouped_tokens,
        buckets.sorted_experts,
        buckets.expert_counts,
        buckets.expert_offsets,
    )
    grouped_hidden = BFloat16.((gate ./ (1.0f0 .+ exp.(-gate))) .* up)
    grouped_output = LIFEAI_CUDA_EXT.qwen3_cuda_grouped_bf16_matmul(
        expert_parameters.down_proj,
        grouped_hidden,
        buckets.sorted_experts,
        buckets.expert_counts,
        buckets.expert_offsets,
    )
    routed_output = CUDA.zeros(Float32, d_model, pair_count)
    scatter_blocks = cld(length(routed_output), threads)
    CUDA.@cuda threads=threads blocks=scatter_blocks LIFEAI_CUDA_EXT._qwen3_cuda_scatter_expert_major_output_kernel!(
        routed_output,
        grouped_output,
        buckets.route_permutation,
        d_model,
        pair_count,
    )
    output = CUDA.zeros(Float32, d_model, num_tokens)
    combine_blocks = cld(length(output), threads)
    CUDA.@cuda threads=threads blocks=combine_blocks LIFEAI_CUDA_EXT._qwen3_cuda_combine_routes_kernel!(
        output,
        routed_output,
        routing_weights,
        d_model,
        num_tokens,
        experts_per_token,
    )
    return output
end

function benchmark_case(expert_parameters, token_count)
    tokens = CUDA.cu(randn(
        Xoshiro(INPUT_SEED + token_count),
        Float32,
        D_MODEL,
        token_count,
    ))
    expert_indices, routing_weights = routes(token_count)
    scalar_bucketed = () ->
        LIFEAI_CUDA_EXT.qwen3_cuda_bucketed_sparse_expert_dispatch(
            tokens,
            expert_indices,
            routing_weights,
            expert_parameters,
        )
    grouped_wmma = () ->
        LIFEAI_CUDA_EXT.qwen3_cuda_grouped_bf16_sparse_expert_dispatch(
            tokens,
            expert_indices,
            routing_weights,
            expert_parameters,
        )
    legacy_grouped_wmma = () -> legacy_grouped_bf16_dispatch(
        tokens,
        expert_indices,
        routing_weights,
        expert_parameters,
    )

    scalar_cold_started = time_ns()
    scalar_output = synchronize_call(scalar_bucketed)
    scalar_cold_seconds = elapsed_seconds(scalar_cold_started)
    legacy_cold_started = time_ns()
    legacy_output = synchronize_call(legacy_grouped_wmma)
    legacy_cold_seconds = elapsed_seconds(legacy_cold_started)
    wmma_cold_started = time_ns()
    wmma_output = synchronize_call(grouped_wmma)
    wmma_cold_seconds = elapsed_seconds(wmma_cold_started)
    scalar_output, scalar_samples = measured_samples(scalar_bucketed)
    legacy_output, legacy_samples = measured_samples(legacy_grouped_wmma)
    wmma_output, wmma_samples = measured_samples(grouped_wmma)
    scalar_median = median(scalar_samples)
    legacy_median = median(legacy_samples)
    wmma_median = median(wmma_samples)

    return (;
        token_count,
        pair_count=token_count * EXPERTS_PER_TOKEN,
        scalar_bucketed_cold_seconds=scalar_cold_seconds,
        legacy_grouped_wmma_cold_seconds=legacy_cold_seconds,
        grouped_wmma_cold_seconds=wmma_cold_seconds,
        scalar_bucketed_cuda_seconds=scalar_samples,
        scalar_bucketed_cuda_median_seconds=scalar_median,
        legacy_grouped_wmma_cuda_seconds=legacy_samples,
        legacy_grouped_wmma_cuda_median_seconds=legacy_median,
        grouped_wmma_cuda_seconds=wmma_samples,
        grouped_wmma_cuda_median_seconds=wmma_median,
        grouped_wmma_over_scalar_bucketed_speedup=scalar_median / wmma_median,
        shared_workspace_over_legacy_speedup=legacy_median / wmma_median,
        max_abs_shared_vs_legacy=maximum(abs.(
            Array(wmma_output) .- Array(legacy_output),
        )),
        max_abs_grouped_wmma_vs_scalar_bucketed=maximum(abs.(
            Array(wmma_output) .- Array(scalar_output),
        )),
    )
end

function main(output_path)
    CUDA.functional() || error("CUDA is not functional")
    CUDA.allowscalar(false)
    parameter_value = BFloat16(0.001)
    expert_parameters = (;
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
    CUDA.synchronize()
    cases = [
        benchmark_case(expert_parameters, token_count)
        for token_count in TOKEN_COUNTS
    ]
    result = (;
        schema_version=3,
        benchmark="qwen3_moe_cuda_grouped_bf16_dispatch",
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
            bf16_expert_parameter_bytes=sum(
                parameter -> sizeof(eltype(parameter)) * length(parameter),
                expert_parameters,
            ),
            wmma_tile=(m=32, n=8, k=16),
        ),
        measurement=(;
            input_seed=INPUT_SEED,
            samples=SAMPLES,
            includes_route_bucketing=true,
            includes_complete_swiglu_dispatch=true,
            legacy_rebuilds_layout_and_pack_for_each_projection=true,
            optimized_reuses_layout_and_padded_intermediates=true,
            optimized_combines_directly_from_padded_output=true,
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
    "cuda_4090d_grouped_bf16_dispatch.json",
) : ARGS[1]
main(output_path)
