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

function cuda_forward(layer, x, parameters, state)
    output, _ = layer(x, parameters, state)
    CUDA.synchronize()
    return output
end

function host_samples(layer, x, parameters)
    qwen3_moe_forward_with_stats(layer, x, parameters)
    return [(@elapsed qwen3_moe_forward_with_stats(layer, x, parameters))
            for _ in 1:SAMPLES]
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
    cpu_samples = host_samples(layer, x, parameters)
    x_gpu = CUDA.cu(x)

    cold_started = time_ns()
    output_gpu = cuda_forward(layer, x_gpu, parameters_gpu, state)
    cold_seconds = elapsed_seconds(cold_started)
    cuda_samples = Float64[]
    for _ in 1:SAMPLES
        started = time_ns()
        output_gpu = cuda_forward(layer, x_gpu, parameters_gpu, state)
        push!(cuda_samples, elapsed_seconds(started))
    end
    output = Array(output_gpu)

    token_count = sequence_length * BATCH_SIZE
    routed_pairs = EXPERTS_PER_TOKEN * token_count
    dense_pairs = NUM_EXPERTS * token_count
    selected_weight_bytes = routed_pairs * 3 * D_MODEL * EXPERT_HIDDEN_DIM *
        sizeof(Float32)
    return (;
        sequence_length,
        token_count,
        dispatch=(;
            compact_route_entries=routed_pairs,
            sparse_token_expert_pairs=routed_pairs,
            dense_token_expert_pairs=dense_pairs,
            pair_reduction=dense_pairs / routed_pairs,
            selected_weight_materialization_bytes=selected_weight_bytes,
        ),
        results=(;
            cold_seconds,
            samples=SAMPLES,
            cuda_seconds=cuda_samples,
            cuda_median_seconds=median(cuda_samples),
            cpu_sparse_seconds=cpu_samples,
            cpu_sparse_median_seconds=median(cpu_samples),
            cpu_over_cuda=median(cpu_samples) / median(cuda_samples),
            max_abs_vs_dense_oracle=maximum(
                abs.(output .- reshape(dense, size(output))),
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
    transfer_started = time_ns()
    parameters_gpu = CUDA.cu(parameters)
    CUDA.synchronize()
    transfer_seconds = elapsed_seconds(transfer_started)

    cases = [
        benchmark_case(layer, parameters, state, parameters_gpu, sequence_length)
        for sequence_length in SEQUENCE_LENGTHS
    ]
    result = (;
        schema_version=1,
        benchmark="qwen3_moe_compact_sparse_dispatch_cuda",
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
            parameter_transfer_seconds=transfer_seconds,
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
    "cuda_4090d_reference.json",
) : ARGS[1]
main(output_path)
