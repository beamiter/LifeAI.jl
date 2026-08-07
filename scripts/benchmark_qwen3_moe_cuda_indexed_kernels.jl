#!/usr/bin/env julia

using CUDA
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
const SEQUENCE_LENGTHS = (1, 64)
const BATCH_SIZE = 1
const SAMPLES = 15
const PARAMETER_SEED = 20260811
const INPUT_SEED = 20260812

elapsed_seconds(started) = (time_ns() - started) / 1.0e9

function indexed_forward(layer, x, parameters, state)
    output, _ = layer(x, parameters, state)
    CUDA.synchronize()
    return output
end

function route_major_forward(layer, x, parameters)
    tokens = reshape(x, layer.d_model, :)
    routed = qwen3_device_topk_routing(
        parameters.gate.weight * tokens,
        layer.experts_per_token;
        normalize=layer.normalize_routing,
    )
    output = LifeAI.qwen3_route_major_expert_dispatch(
        tokens,
        routed.expert_indices,
        routed.routing_weights,
        parameters.experts,
    )
    CUDA.synchronize()
    return reshape(output, size(x))
end

function measured_samples(f)
    samples = Float64[]
    output = nothing
    for _ in 1:SAMPLES
        started = time_ns()
        output = f()
        push!(samples, elapsed_seconds(started))
    end
    return output, samples
end

function benchmark_case(layer, parameters, state, parameters_gpu, sequence_length)
    x = randn(
        Xoshiro(INPUT_SEED + sequence_length),
        Float32,
        D_MODEL,
        sequence_length,
        BATCH_SIZE,
    )
    host = qwen3_moe_forward_with_stats(layer, x, parameters)
    dense = qwen3_dense_expert_reference(
        reshape(x, D_MODEL, :),
        host.routing,
        parameters.experts,
    )
    qwen3_moe_forward_with_stats(layer, x, parameters)
    cpu_sparse_samples = [
        (@elapsed qwen3_moe_forward_with_stats(layer, x, parameters))
        for _ in 1:SAMPLES
    ]
    x_gpu = CUDA.cu(x)

    indexed_cold_started = time_ns()
    indexed_output = indexed_forward(layer, x_gpu, parameters_gpu, state)
    indexed_cold_seconds = elapsed_seconds(indexed_cold_started)
    route_major_cold_started = time_ns()
    route_major_output = route_major_forward(layer, x_gpu, parameters_gpu)
    route_major_cold_seconds = elapsed_seconds(route_major_cold_started)

    indexed_output, indexed_samples = measured_samples(
        () -> indexed_forward(layer, x_gpu, parameters_gpu, state),
    )
    route_major_output, route_major_samples = measured_samples(
        () -> route_major_forward(layer, x_gpu, parameters_gpu),
    )
    indexed = Array(indexed_output)
    route_major = Array(route_major_output)
    indexed_median = median(indexed_samples)
    route_major_median = median(route_major_samples)
    cpu_sparse_median = median(cpu_sparse_samples)

    token_count = sequence_length * BATCH_SIZE
    routed_pairs = EXPERTS_PER_TOKEN * token_count
    selected_weight_bytes = routed_pairs * 3 * D_MODEL * EXPERT_HIDDEN_DIM *
        sizeof(Float32)
    workspace_bytes = qwen3_cuda_indexed_workspace_bytes(
        D_MODEL,
        EXPERT_HIDDEN_DIM,
        token_count,
        EXPERTS_PER_TOKEN,
    )
    return (;
        sequence_length,
        token_count,
        dispatch=(;
            sparse_token_expert_pairs=routed_pairs,
            dense_token_expert_pairs=NUM_EXPERTS * token_count,
            pair_reduction=NUM_EXPERTS / EXPERTS_PER_TOKEN,
            route_major_selected_weight_bytes=selected_weight_bytes,
            indexed_workspace_bytes=workspace_bytes,
            workspace_reduction=selected_weight_bytes / workspace_bytes,
        ),
        results=(;
            indexed_cold_seconds,
            route_major_cold_seconds,
            samples=SAMPLES,
            indexed_cuda_seconds=indexed_samples,
            indexed_cuda_median_seconds=indexed_median,
            route_major_cuda_seconds=route_major_samples,
            route_major_cuda_median_seconds=route_major_median,
            cpu_sparse_seconds=cpu_sparse_samples,
            cpu_sparse_median_seconds=cpu_sparse_median,
            indexed_speedup=route_major_median / indexed_median,
            indexed_cuda_over_cpu=cpu_sparse_median / indexed_median,
            max_abs_indexed_vs_dense=maximum(
                abs.(indexed .- reshape(dense, size(indexed))),
            ),
            max_abs_indexed_vs_route_major=maximum(
                abs.(indexed .- route_major),
            ),
        ),
    )
end

function main(output_path)
    CUDA.functional() || error("CUDA is not functional")
    CUDA.allowscalar(false)
    BLAS.set_num_threads(1)
    layer = Qwen3SparseMoE(
        D_MODEL,
        EXPERT_HIDDEN_DIM,
        NUM_EXPERTS,
        EXPERTS_PER_TOKEN,
    )
    parameters, state = Lux.setup(Xoshiro(PARAMETER_SEED), layer)
    parameters_gpu = CUDA.cu(parameters)
    CUDA.synchronize()
    cases = [
        benchmark_case(layer, parameters, state, parameters_gpu, sequence_length)
        for sequence_length in SEQUENCE_LENGTHS
    ]
    result = (;
        schema_version=1,
        benchmark="qwen3_moe_cuda_indexed_kernels",
        environment=(;
            julia_version=string(VERSION),
            device=CUDA.name(CUDA.device()),
            runtime_version=string(CUDA.runtime_version()),
            driver_version=string(CUDA.driver_version()),
            blas_threads=BLAS.get_num_threads(),
        ),
        shape=(;
            d_model=D_MODEL,
            expert_hidden_dim=EXPERT_HIDDEN_DIM,
            num_experts=NUM_EXPERTS,
            experts_per_token=EXPERTS_PER_TOKEN,
            batch_size=BATCH_SIZE,
        ),
        measurement=(;
            parameter_seed=PARAMETER_SEED,
            input_seed_base=INPUT_SEED,
            warmup_runs=1,
            samples=SAMPLES,
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
    "cuda_4090d_indexed_kernels.json",
) : ARGS[1]
main(output_path)
