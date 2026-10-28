using Test, LifeAI, CUDA, Random, LinearAlgebra

CUDA.functional() || error("Token-scaled FP4 tests require CUDA")
native_nvfp4_supported() || error("Token-scaled FP4 tests require Blackwell")
CUDA.allowscalar(false)
const _FP4TokenExt=Base.get_extension(LifeAI,:LifeAICUDAExt)

function fp4_token_reference(weight,input)
    activations=hcat((vec(dequantize_nvfp4(quantize_nvfp4(
        permutedims(input[:,token:token])))) for token in axes(input,2))...)
    return dequantize_nvfp4(weight)*activations
end

struct _FP4TokenCPUReference
    weight::NVFP4Weight
end
LifeAI._flint_linear(weight::_FP4TokenCPUReference,input)=
    fp4_token_reference(weight.weight,Array(input))

@testset "Native FP4 token scales match independent portable quantizers" begin
    rng=Xoshiro(2026102837)
    for (m,k,n) in ((1,1,2),(17,19,3),(31,65,129),(512,2048,17),(128,256,513))
        host=quantize_nvfp4(randn(rng,Float32,m,k)./sqrt(Float32(k)))
        prepared=prepare_nvfp4_gemm(LifeAI._flint_weight_to_device(CUDA.cu,host);
            activation_scaling=:token,portable_device=:cpu)
        for input in (randn(rng,Float32,k,n),zeros(Float32,k,n),
                fill(nextfloat(0f0),k,n))
            device=CUDA.cu(input)
            actual=Array(native_nvfp4_gemm(prepared,device;check_finite=false))
            expected=fp4_token_reference(host,input)
            @test actual≈expected atol=4f-4 rtol=8f-4
            @test all(isfinite,actual)
            packed,scales,globals=_FP4TokenExt._native_fp4_token_activation(
                device,prepared.padded_k;check_finite=false)
            parts=[quantize_nvfp4(permutedims(input[:,token:token])) for token in 1:n]
            portable=NVFP4Weight(vcat((part.packed for part in parts)...),
                vcat((part.block_scales for part in parts)...),1f0,k)
            expected_packed,expected_scales=_FP4TokenExt._native_fp4_repack(
                LifeAI._flint_weight_to_device(CUDA.cu,portable),n,prepared.padded_k)
            @test Array(packed)==Array(expected_packed)
            @test Array(scales)==Array(expected_scales)
            @test Array(globals)==[part.global_scale for part in parts]
        end
        @test native_nvfp4_stats(prepared).activation_scaling===:token
        @test native_nvfp4_stats(prepared).portable_device===:cpu
        close(prepared)
    end
end

@testset "Flint token activation mode matches CPU oracle across prefill chunks" begin
    fixture=joinpath(@__DIR__,"flint","fixtures","tiny")
    host=load_hf_flint_quantized(fixture;scheme=:nvfp4)
    reference=(model=host.model,parameters=Dict{String,Any}(
        name=>(weight isa NVFP4Weight ? _FP4TokenCPUReference(weight) : weight)
        for (name,weight) in host.parameters))
    native=prepare_flint_nvfp4(load_hf_flint_quantized(fixture;scheme=:nvfp4,device=CUDA.cu);
        activation_scaling=:token,portable_device=:cpu)
    tokens=[1,3,5,7,9,11,13,15,17]
    expected=hf_flint_forward(reference.model,reference.parameters,tokens)
    full=Array(hf_flint_forward(native.model,native.parameters,tokens))
    @test full≈expected atol=3f-4 rtol=5f-4
    for chunk in (1,2,4,9)
        cache=FlintKVCache(native.model)
        parts=[Array(hf_flint_forward(native.model,native.parameters,
            tokens[first:min(end,first+chunk-1)];cache)) for first in 1:chunk:length(tokens)]
        @test hcat(parts...)≈expected atol=3f-4 rtol=5f-4
        @test hcat(parts...)≈full atol=3f-4 rtol=5f-4
        @test cache.tokens==length(tokens)
    end
    foreach(weight->weight isa AbstractNativeNVFP4Weight && close(weight),values(native.parameters))
end

@testset "Token-scaled FP4 is independent of batch chunks and other tokens" begin
    rng=Xoshiro(2026102838)
    host=quantize_nvfp4(randn(rng,Float32,37,83))
    source=LifeAI._flint_weight_to_device(CUDA.cu,host)
    token=prepare_nvfp4_gemm(source;activation_scaling=:token)
    global_weight=prepare_nvfp4_gemm(source)
    @test native_nvfp4_stats(global_weight).activation_scaling===:global
    input=randn(rng,Float32,83,139)
    input .*= reshape(Float32.(10.0 .^ range(-5,5;length=139)),1,:)
    whole=Array(native_nvfp4_gemm(token,CUDA.cu(input)))
    expected=fp4_token_reference(host,input)
    @test whole≈expected atol=5f-4 rtol=8f-4
    for chunk in (1,3,17,64,128)
        parts=[Array(native_nvfp4_gemm(token,CUDA.cu(input[:,first:min(end,first+chunk-1)])))
            for first in 1:chunk:size(input,2)]
        @test hcat(parts...)≈whole atol=5f-4 rtol=8f-4
    end

    # A huge second token must not erase an ordinary first token's blocks.
    ordinary=randn(rng,Float32,83,2)
    outlier=copy(ordinary);outlier[:,2].*=1f12
    baseline=Array(native_nvfp4_gemm(token,CUDA.cu(ordinary)))
    isolated=Array(native_nvfp4_gemm(token,CUDA.cu(outlier)))
    contaminated=Array(native_nvfp4_gemm(global_weight,CUDA.cu(outlier)))
    @test isolated[:,1]==baseline[:,1]
    @test isolated[:,1]≈fp4_token_reference(host,outlier)[:,1] atol=4f-4 rtol=8f-4
    @test norm(contaminated[:,1]-isolated[:,1])>0.5f0*norm(isolated[:,1])
    single=CUDA.cu(ordinary[:,1:1])
    @test Array(native_nvfp4_gemm(token,single))==Array(native_nvfp4_gemm(global_weight,single))
    group=LifeAI.native_nvfp4_gemm_multi((token,global_weight),CUDA.cu(ordinary))
    @test Array(group[1])≈baseline atol=4f-4 rtol=8f-4
    @test Array(group[2])==Array(native_nvfp4_gemm(global_weight,CUDA.cu(ordinary)))
    @test_throws ArgumentError prepare_nvfp4_gemm(source;activation_scaling=:row)
    for invalid in (Inf32,-Inf32,NaN32)
        bad=copy(ordinary);bad[13,2]=invalid
        @test_throws ArgumentError native_nvfp4_gemm(token,CUDA.cu(bad))
        unchecked=Array(native_nvfp4_gemm(token,CUDA.cu(bad);check_finite=false))
        @test all(isnan,unchecked[:,2])
        @test unchecked[:,1]≈baseline[:,1] atol=4f-4 rtol=8f-4
    end
    close(token);close(global_weight)
end
