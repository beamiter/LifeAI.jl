using Test, LifeAI, CUDA, Random, LinearAlgebra

CUDA.functional() || error("A functional CUDA device is required")
CUDA.allowscalar(false)
native_nvfp4_supported() || error("Native NVFP4 decode requires Blackwell")
const _FP4DecodeExt = Base.get_extension(LifeAI, :LifeAICUDAExt)

function fp4_decode_reference(weight, input)
    activation = quantize_nvfp4(permutedims(input))
    return dequantize_nvfp4(weight) * permutedims(dequantize_nvfp4(activation))
end

@testset "Fused native NVFP4 decode quantizer matches portable bytes" begin
    rng = Xoshiro(2026102801)
    for (m,k) in ((1,1),(17,19),(31,65),(512,2048),(2048,512),(2048,6144))
        host_weight = quantize_nvfp4(randn(rng,Float32,m,k) ./ sqrt(Float32(k)))
        prepared = prepare_nvfp4_gemm(LifeAI._flint_weight_to_device(CUDA.cu,host_weight))
        for input in (randn(rng,Float32,k,1),zeros(Float32,k,1),
                fill(nextfloat(0f0),k,1), fill(0.25f0,k,1))
            device = CUDA.cu(input)
            actual = Array(native_nvfp4_gemm(prepared,device;check_finite=false))
            reference = fp4_decode_reference(host_weight,input)
            @test actual ≈ reference atol=4f-4 rtol=7f-4
            context = _FP4DecodeExt._native_fp4_context()
            buffers = context.decode_buffers[prepared.padded_k]
            expected_activation = quantize_nvfp4(permutedims(input))
            expected_gpu = LifeAI._flint_weight_to_device(CUDA.cu,expected_activation)
            packed, scales = _FP4DecodeExt._native_fp4_repack(expected_gpu,1,prepared.padded_k)
            @test Array(buffers.packed) == Array(packed)
            @test Array(buffers.scales) == Array(scales)
            scalars = Array(buffers.scalars)
            @test scalars[1] == prepared.weight.global_scale * expected_activation.global_scale
            @test scalars[3] == expected_activation.global_scale
            @test scalars[4] == 0f0
        end
        close(prepared)
    end
end

@testset "Asynchronous native decode reuses scratch without corrupting outputs" begin
    rng = Xoshiro(2026102802)
    hosts = [quantize_nvfp4(randn(rng,Float32,128,256)) for _ in 1:3]
    weights = [prepare_nvfp4_gemm(LifeAI._flint_weight_to_device(CUDA.cu,w)) for w in hosts]
    inputs = [randn(rng,Float32,256,1) for _ in 1:24]
    devices = CUDA.cu.(inputs)
    outputs = [native_nvfp4_gemm(weights[mod1(i,3)],devices[i];check_finite=false) for i in eachindex(devices)]
    for i in eachindex(outputs)
        @test Array(outputs[i]) ≈ fp4_decode_reference(hosts[mod1(i,3)],inputs[i]) atol=5f-4 rtol=7f-4
    end
    context = _FP4DecodeExt._native_fp4_context()
    @test length(context.decode_buffers) <= 8
    @test all(native_nvfp4_stats(w).native_calls == 8 for w in weights)

    # Alternate tasks/streams while they share a prepared descriptor/workspace.
    tasks = [@async begin
        result = native_nvfp4_gemm(weights[1],devices[i];check_finite=false)
        yield()
        Array(result)
    end for i in 1:8]
    for (i,task) in enumerate(tasks)
        @test fetch(task) ≈ fp4_decode_reference(hosts[1],inputs[i]) atol=5f-4 rtol=7f-4
    end
    for n in (2,1,7,1)
        input=randn(rng,Float32,256,n)
        @test Array(native_nvfp4_gemm(weights[1],CUDA.cu(input))) ≈ fp4_decode_reference(hosts[1],input) atol=5f-4 rtol=7f-4
    end
    for value in (Inf32,NaN32,-Inf32)
        @test_throws ArgumentError native_nvfp4_gemm(weights[1],CUDA.fill(value,256,1))
    end
    close.(weights)
end
