#!/usr/bin/env julia
# Times the complete native path, including activation quantization and padding.
using LifeAI, CUDA, JSON3, Random, Statistics, LinearAlgebra
using BFloat16s: BFloat16

function native_benchmark_options(args)
    options = Dict("samples"=>"10", "warmup"=>"2", "output"=>"")
    full = false
    i = 1
    while i <= length(args)
        if args[i] == "--full"
            full = true
            i += 1
        elseif args[i] == "--quick"
            full = false
            i += 1
        else
            startswith(args[i], "--") && i < length(args) || error("Invalid benchmark argument $(args[i])")
            key = args[i][3:end]
            haskey(options, key) || error("Unknown benchmark option --$key")
            options[key] = args[i+1]
            i += 2
        end
    end
    samples, warmup = parse(Int, options["samples"]), parse(Int, options["warmup"])
    samples > 0 && warmup > 0 || error("Samples and warmup must be positive")
    return (; samples, warmup, full, output=options["output"])
end

function native_benchmark_timing(f, samples, warmup)
    started = time_ns()
    output = f()
    CUDA.synchronize()
    first_timed_call = (time_ns() - started) / 1e9
    for _ in 1:warmup
        f()
        CUDA.synchronize()
    end
    wall_seconds, gpu_seconds = Float64[], Float64[]
    for _ in 1:samples
        CUDA.synchronize()
        started = time_ns()
        elapsed = CUDA.@elapsed f()
        CUDA.synchronize()
        push!(gpu_seconds, elapsed)
        push!(wall_seconds, (time_ns() - started) / 1e9)
    end
    return (first_timed_call_seconds=first_timed_call, median_wall_seconds=median(wall_seconds),
        minimum_wall_seconds=minimum(wall_seconds), median_gpu_seconds=median(gpu_seconds),
        wall_seconds=wall_seconds, gpu_seconds=gpu_seconds)
end

function main(args)
    options = native_benchmark_options(args)
    CUDA.functional() || error("A functional CUDA device is required")
    CUDA.allowscalar(false)
    native_nvfp4_supported() || error("Native NVFP4 GEMM is not supported by this GPU/runtime")
    shapes = options.full ? [(m,k,n) for (m,k) in ((512,2048),(2048,512),(2048,2048),(4096,2048)) for n in (1,32,128)] :
        [(512,2048,1),(2048,512,32),(4096,2048,128)]
    results = Any[]
    for (index, (m, k, n)) in enumerate(shapes)
        rng = Xoshiro(10000+index)
        raw_weight = randn(rng, Float32, m, k) ./ sqrt(Float32(k))
        input = CUDA.cu(randn(rng, Float32, k, n))
        weight = quantize_nvfp4(CUDA.cu(raw_weight))
        prepare_started = time_ns()
        prepared = prepare_nvfp4_gemm(weight)
        CUDA.synchronize()
        preparation_seconds = (time_ns()-prepare_started)/1e9
        dequantized_weight = dequantize_nvfp4(weight)
        bf16_weight, bf16_input = BFloat16.(dequantized_weight), BFloat16.(input)
        activation = quantize_nvfp4(permutedims(input))
        w4a4_reference = dequantized_weight * permutedims(dequantize_nvfp4(activation))
        native = () -> native_nvfp4_gemm(prepared, input)
        reconstruct_f32 = () -> dequantize_nvfp4(weight) * input
        cached_f32 = () -> dequantized_weight * input
        cached_bf16 = () -> bf16_weight * bf16_input
        actual = native()
        CUDA.synchronize()
        reference_host, actual_host = Array(w4a4_reference), Array(actual)
        max_error = maximum(abs.(actual_host-reference_host))
        relative_error = norm(actual_host-reference_host)/max(norm(reference_host),eps(Float32))
        isfinite(relative_error) && relative_error < 0.002f0 || error("Native arithmetic mismatch for $((m,k,n)): $relative_error")
        timings = (native_w4a4=native_benchmark_timing(native,options.samples,options.warmup),
            dequantize_weight_then_f32_w4a16=native_benchmark_timing(reconstruct_f32,options.samples,options.warmup),
            cached_f32_weight_w4a16=native_benchmark_timing(cached_f32,options.samples,options.warmup),
            cached_bf16_weight_w4a16=native_benchmark_timing(cached_bf16,options.samples,options.warmup))
        stats = native_nvfp4_stats(prepared)
        stats.backend == :cublaslt_nvfp4 && stats.native_calls > 0 || error("Native backend was not used")
        speedup = timings.dequantize_weight_then_f32_w4a16.median_wall_seconds /
            timings.native_w4a4.median_wall_seconds
        push!(results,(m=m,k=k,n=n,preparation_seconds=preparation_seconds,
            stored_weight_bytes=quantized_parameter_bytes(weight),
            reference_max_absolute_error=max_error,reference_relative_l2_error=relative_error,
            speedup_vs_weight_dequantization_f32=speedup,timings=timings,stats=stats))
        println(stderr,"native NVFP4 shape=",(m,k,n)," wall seconds=",timings.native_w4a4.median_wall_seconds,
            " speedup vs weight dequantization=",speedup)
    end
    report = (schema_version=1,backend="cublaslt_nvfp4",device=CUDA.name(CUDA.device()),
        julia_version=string(VERSION),samples=options.samples,warmup=options.warmup,
        mode=options.full ? "full" : "quick", results=results,
        scope="Synthetic GEMMs; correctness priming precedes timing; first timed call is reported separately from warm samples; native includes activation quantization, padding and output crop.",
        comparison="Native arithmetic is W4A4; old dequantized and cached baselines use W4A16 and are not numerically identical.")
    if !isempty(options.output)
        destination=abspath(options.output)
        mkpath(dirname(destination))
        open(destination,"w") do io
            JSON3.pretty(io,report)
        end
    end
    JSON3.write(stdout,report)
    println()
end
main(ARGS)
