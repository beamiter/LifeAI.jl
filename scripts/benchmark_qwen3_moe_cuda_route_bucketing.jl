#!/usr/bin/env julia

using CUDA
using JSON3
using LifeAI
using Random: Xoshiro
using Statistics: median

const NUM_EXPERTS = 128
const EXPERTS_PER_TOKEN = 8
const TOKEN_COUNTS = (1, 8, 64, 512)
const SAMPLES = 30
const ROUTE_SEED = 20260814
const LIFEAI_CUDA_EXT = Base.get_extension(LifeAI, :LifeAICUDAExt)

elapsed_seconds(started) = (time_ns() - started) / 1.0e9

function bucket_once(expert_indices)
    buckets = LIFEAI_CUDA_EXT.qwen3_cuda_bucket_routes(
        expert_indices,
        NUM_EXPERTS,
    )
    CUDA.synchronize()
    return buckets
end

function measured_samples(expert_indices)
    samples = Float64[]
    buckets = nothing
    for _ in 1:SAMPLES
        started = time_ns()
        buckets = bucket_once(expert_indices)
        push!(samples, elapsed_seconds(started))
    end
    return buckets, samples
end

function benchmark_case(token_count)
    pair_count = token_count * EXPERTS_PER_TOKEN
    host_indices = rand(
        Xoshiro(ROUTE_SEED + token_count),
        Int32(1):Int32(NUM_EXPERTS),
        EXPERTS_PER_TOKEN,
        token_count,
    )
    expert_indices = CUDA.cu(host_indices)
    cold_started = time_ns()
    buckets = bucket_once(expert_indices)
    cold_seconds = elapsed_seconds(cold_started)
    buckets, samples = measured_samples(expert_indices)
    sorted_experts = Array(buckets.sorted_experts)
    route_permutation = Array(buckets.route_permutation)
    expert_counts = Array(buckets.expert_counts)
    expert_offsets = Array(buckets.expert_offsets)
    expected_permutation = sortperm(vec(host_indices); alg=Base.Sort.MergeSort)
    expected_sorted = vec(host_indices)[expected_permutation]
    sorted_experts == expected_sorted || error("sorted expert mismatch")
    route_permutation == expected_permutation || error("route permutation mismatch")
    sum(expert_counts) == pair_count || error("expert count sum mismatch")
    expert_offsets[1] == 1 || error("first expert offset mismatch")
    expert_offsets[end] == pair_count + 1 || error("last expert offset mismatch")

    output_bytes = (
        pair_count * sizeof(Int32) * 2 +
        NUM_EXPERTS * sizeof(Int32) +
        (NUM_EXPERTS + 1) * sizeof(Int32)
    )
    return (;
        token_count,
        pair_count,
        active_experts=count(>(0), expert_counts),
        cold_seconds,
        cuda_seconds=samples,
        cuda_median_seconds=median(samples),
        output_bytes,
        stable_permutation_verified=true,
        offsets_verified=true,
    )
end

function main(output_path)
    CUDA.functional() || error("CUDA is not functional")
    CUDA.allowscalar(false)
    cases = [benchmark_case(token_count) for token_count in TOKEN_COUNTS]
    result = (;
        schema_version=1,
        benchmark="qwen3_moe_cuda_route_bucketing",
        environment=(;
            julia_version=string(VERSION),
            device=CUDA.name(CUDA.device()),
            runtime_version=string(CUDA.runtime_version()),
            driver_version=string(CUDA.driver_version()),
        ),
        shape=(;
            num_experts=NUM_EXPERTS,
            experts_per_token=EXPERTS_PER_TOKEN,
        ),
        measurement=(; route_seed=ROUTE_SEED, samples=SAMPLES),
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
    "cuda_4090d_route_bucketing.json",
) : ARGS[1]
main(output_path)
