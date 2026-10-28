#!/usr/bin/env julia
# Native FP4 GEMM requires a compatible Blackwell GPU and cuBLASLt.
using LifeAI, CUDA, Test, Random, LinearAlgebra, JSON3

CUDA.functional() || error("A functional CUDA device is required")
CUDA.allowscalar(false)
native_nvfp4_supported() || error("Native NVFP4 GEMM is not supported by this GPU/runtime")

# The reference quantizes both operands. Comparing a W4A4 kernel only with
# W4A16 arithmetic would conflate activation quantization and GEMM errors.
function native_test_reference(weight::NVFP4Weight, input::AbstractMatrix)
    activation = quantize_nvfp4(permutedims(Array(input)))
    return dequantize_nvfp4(weight) * permutedims(dequantize_nvfp4(activation))
end

struct _NativeTestW4A4Reference
    weight::NVFP4Weight
end
LifeAI._flint_linear(weight::_NativeTestW4A4Reference, input) =
    native_test_reference(weight.weight, input)

function native_test_relative_error(actual, reference)
    return norm(actual - reference) / max(norm(reference), eps(Float32))
end

function native_test_calls(bundle)
    return sum(native_nvfp4_stats(weight).native_calls
        for weight in values(bundle.parameters) if weight isa AbstractNativeNVFP4Weight)
end

measurements = Any[]
@testset "Native NVFP4 GEMM arithmetic and padding" begin
    for (index, (m, k, n)) in enumerate(((1, 1, 1), (17, 19, 3),
            (31, 65, 1), (33, 129, 7), (32, 128, 16), (128, 256, 33)))
        rng = Xoshiro(8123 + index)
        weight_host = quantize_nvfp4(randn(rng, Float32, m, k) ./ sqrt(Float32(k)))
        weight_gpu = LifeAI._flint_weight_to_device(CUDA.cu, weight_host)
        prepared = prepare_nvfp4_gemm(weight_gpu)
        @test nvfp4_weight(prepared) === weight_gpu
        input = randn(rng, Float32, k, n)
        expected = native_test_reference(weight_host, input)
        actual = Array(native_nvfp4_gemm(prepared, CUDA.cu(input)))
        @test size(actual) == (m, n)
        @test all(isfinite, actual)
        @test actual ≈ expected atol=3f-4 rtol=5f-4
        stats = native_nvfp4_stats(prepared)
        @test stats.backend == :cublaslt_nvfp4
        @test stats.native_calls >= 1
        @test stats.padded_m >= m && stats.padded_k >= k
        push!(measurements, (kind="gemm", m=m, k=k, n=n,
            max_absolute_error=maximum(abs.(actual - expected)),
            relative_l2_error=native_test_relative_error(actual, expected), stats=stats))
    end
end

@testset "Native NVFP4 reuses a prepared weight across token counts" begin
    rng = Xoshiro(934)
    weight = quantize_nvfp4(randn(rng, Float32, 37, 83))
    prepared = prepare_nvfp4_gemm(LifeAI._flint_weight_to_device(CUDA.cu, weight))
    for n in (1, 2, 17, 65, 1)
        input = randn(rng, Float32, 83, n)
        actual = Array(native_nvfp4_gemm(prepared, CUDA.cu(input)))
        expected = native_test_reference(weight, input)
        @test actual ≈ expected atol=3f-4 rtol=5f-4
    end
    stats = native_nvfp4_stats(prepared)
    @test stats.native_calls >= 5
    @test stats.backend == :cublaslt_nvfp4
    push!(measurements, (kind="reuse", stats=stats))
end

@testset "Native NVFP4 guards, lifecycle and shared-workspace concurrency" begin
    rng = Xoshiro(814)
    host_weight = quantize_nvfp4(randn(rng, Float32, 21, 35))
    prepared = prepare_nvfp4_gemm(LifeAI._flint_weight_to_device(CUDA.cu, host_weight))
    @test_throws DimensionMismatch native_nvfp4_gemm(prepared, CUDA.zeros(Float32, 34, 2))
    @test_throws ArgumentError native_nvfp4_gemm(prepared, CUDA.zeros(Float32, 35, 0))
    host_inputs = (randn(rng, Float32, 35, 1), randn(rng, Float32, 35, 7))
    device_inputs = CUDA.cu.(host_inputs)
    # Distinct Julia tasks use distinct CUDA streams; the backend must serialize
    # descriptor updates and the shared workspace until each call completes.
    tasks = [@async native_nvfp4_gemm(prepared, input) for input in device_inputs]
    outputs = fetch.(tasks)
    for (output, input) in zip(outputs, host_inputs)
        @test Array(output) ≈ native_test_reference(host_weight, input) atol=3f-4 rtol=5f-4
    end
    @test native_nvfp4_stats(prepared).native_calls >= 2
    close(prepared)
    @test native_nvfp4_stats(prepared).closed
    @test_throws ArgumentError native_nvfp4_gemm(prepared, first(device_inputs))
    close(prepared) # Closing an already closed wrapper is harmless.
end

@testset "Native NVFP4 zero operands" begin
    rng = Xoshiro(782)
    for (weight, input) in ((zeros(Float32, 19, 35), randn(rng, Float32, 35, 3)),
            (randn(rng, Float32, 19, 35), zeros(Float32, 35, 3)))
        prepared = prepare_nvfp4_gemm(quantize_nvfp4(CUDA.cu(weight)))
        actual = Array(native_nvfp4_gemm(prepared, CUDA.cu(input)))
        @test size(actual) == (19, 3)
        @test all(iszero, actual)
        @test all(isfinite, actual)
        @test native_nvfp4_stats(prepared).native_calls >= 1
    end
end

@testset "Flint uses native FP4 GEMM in full and cached inference" begin
    fixture = joinpath(@__DIR__, "..", "test", "flint", "fixtures", "tiny")
    host = load_hf_flint_quantized(fixture; scheme=:nvfp4)
    reference = (model=host.model, parameters=Dict{String,Any}(
        name => (weight isa NVFP4Weight ? _NativeTestW4A4Reference(weight) : weight)
        for (name, weight) in host.parameters))
    native = prepare_flint_nvfp4(load_hf_flint_quantized(fixture;
        scheme=:nvfp4, device=CUDA.cu))
    @test any(weight isa AbstractNativeNVFP4Weight for weight in values(native.parameters))
    tokens = [1, 3, 5, 7, 9, 11, 13, 15, 17]
    expected = hf_flint_forward(reference.model, reference.parameters, tokens)
    expected_full = copy(expected)
    actual = Array(hf_flint_forward(native.model, native.parameters, tokens))
    @test all(isfinite, actual)
    @test native_test_relative_error(actual, expected) < 0.015f0
    @test native_test_calls(native) > 0
    push!(measurements, (kind="flint_full",
        max_absolute_error=maximum(abs.(actual-expected)),
        relative_l2_error=native_test_relative_error(actual, expected),
        native_calls=native_test_calls(native)))

    # The per-tensor activation scale can differ between batch shapes. Compare
    # each cached call to the same cached CPU W4A4 computation, not full prefill.
    native_cache = FlintKVCache(native.model)
    reference_cache = FlintKVCache(reference.model)
    for ids in (tokens[1:3], tokens[4:6], tokens[7:9])
        expected = hf_flint_forward(reference.model, reference.parameters, ids; cache=reference_cache)
        actual = Array(hf_flint_forward(native.model, native.parameters, ids; cache=native_cache))
        @test all(isfinite, actual)
        @test native_test_relative_error(actual, expected) < 0.02f0
    end
    generated = generate_flint_tokens(native, tokens; max_new_tokens=3,
        temperature=0, eos_ids=Int[])
    @test length(generated) == 3
    @test all(id -> 1 <= id <= native.model.vocab_size, generated)
    @test native_test_calls(native) > 10
    push!(measurements, (kind="flint_generation", token_ids=generated,
        native_calls=native_test_calls(native)))

    # Native resource objects must not leak into portable artifacts.
    mktempdir() do directory
        destination = joinpath(directory, "native-saved")
        save_flint_quantized(destination, native)
        restored = load_flint_quantized(destination; device=CUDA.cu)
        @test all(!(weight isa AbstractNativeNVFP4Weight) for weight in values(restored.parameters))
        prepared_again = prepare_flint_nvfp4(restored)
        replay = Array(hf_flint_forward(prepared_again.model, prepared_again.parameters, tokens))
        @test all(isfinite, replay)
        @test native_test_relative_error(replay, expected_full) < 0.015f0
        @test native_test_calls(prepared_again) > 0
    end
end

report = (backend="cublaslt_nvfp4", passed=true, julia_version=string(VERSION),
    device=CUDA.name(CUDA.device()), reference="W4A4 with the same NVFP4 quantizers",
    measurements=measurements)
if !isempty(ARGS)
    length(ARGS) == 1 || error("Usage: verify_flint_nvfp4_native.jl [OUTPUT_JSON]")
    destination = abspath(ARGS[1])
    mkpath(dirname(destination))
    open(destination, "w") do io
        JSON3.pretty(io, report)
    end
end
JSON3.write(stdout, report)
println()
