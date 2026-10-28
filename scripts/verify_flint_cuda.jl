#!/usr/bin/env julia
using LifeAI, Test, CUDA, LinearAlgebra
CUDA.functional() || error("A functional CUDA device is required")
CUDA.allowscalar(false)
fixture = joinpath(@__DIR__, "..", "test", "flint", "fixtures", "tiny")
cpu = load_hf_flint_model(fixture)
tokens = [1, 3, 5, 7, 9, 11, 13, 15, 17]
@testset "Flint CUDA float and packed weight inference" begin
    for scheme in (:float, :int8, :int4, :nvfp4)
        group = scheme === :nvfp4 ? 16 : 4
        host = scheme == :float ? cpu : load_hf_flint_quantized(fixture; scheme, group)
        gpu = scheme == :float ? load_hf_flint_model(fixture; device=CUDA.cu) :
            load_hf_flint_quantized(fixture; scheme, group, device=CUDA.cu)
        expected = hf_flint_forward(host.model, host.parameters, tokens)
        actual = Array(hf_flint_forward(gpu.model, gpu.parameters, tokens))
        @test actual ≈ expected atol=5f-5 rtol=5f-5
        cache = FlintKVCache(gpu.model)
        prefill = hf_flint_forward(gpu.model, gpu.parameters, tokens[1:5]; cache)
        decoded = hf_flint_forward(gpu.model, gpu.parameters, tokens[6:end]; cache)
        @test Array(hcat(prefill, decoded)) ≈ expected atol=5f-5 rtol=5f-5
        @test generate_flint_tokens(gpu, tokens; max_new_tokens=3, temperature=0, eos_ids=Int[]) ==
            generate_flint_tokens(host, tokens; max_new_tokens=3, temperature=0, eos_ids=Int[])
        println("scheme=", scheme, " maximum_abs_error=", maximum(abs.(actual - expected)),
            " parameter_bytes=", quantized_parameter_bytes(gpu.parameters))
        CUDA.synchronize()
    end
    # Quantization itself also works on-device and preserves encoded bytes.
    values = reshape(Float32.(sin.(1:333)), 9, 37)
    host_nvfp4 = quantize_nvfp4(values)
    device_nvfp4 = quantize_nvfp4(CUDA.cu(values))
    @test Array(device_nvfp4.packed) == host_nvfp4.packed
    @test Array(device_nvfp4.block_scales) == host_nvfp4.block_scales
    @test device_nvfp4.global_scale == host_nvfp4.global_scale
    @test Array(dequantize_nvfp4(device_nvfp4)) ≈ dequantize_nvfp4(host_nvfp4)
end
