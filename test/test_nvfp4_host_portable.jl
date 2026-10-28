using Test, LifeAI, CUDA, Random, LinearAlgebra

CUDA.functional() || error("Host portable FP4 tests require CUDA")
native_nvfp4_supported() || error("Host portable FP4 tests require native FP4 support")
CUDA.allowscalar(false)

@testset "Prepared FP4 can retain exact portable weights on the host" begin
    rng=Xoshiro(20261028042)
    for (rows,columns) in ((17,19),(512,2048),(2048,512))
        host=quantize_nvfp4(randn(rng,Float32,rows,columns))
        source=LifeAI._flint_weight_to_device(CUDA.cu,host)
        ordinary=prepare_nvfp4_gemm(source)
        offloaded=prepare_nvfp4_gemm(source;portable_device=:cpu)
        portable=nvfp4_weight(offloaded)
        @test nvfp4_weight(ordinary) === source
        @test portable.packed isa Matrix{UInt8}
        @test portable.block_scales isa Matrix{UInt8}
        @test portable.packed == host.packed
        @test portable.block_scales == host.block_scales
        @test portable.global_scale == host.global_scale
        @test size(offloaded)==size(source)
        for count in (1,3,17,1)
            x=CUDA.cu(randn(rng,Float32,columns,count))
            a=Array(native_nvfp4_gemm(ordinary,x;check_finite=false))
            b=Array(native_nvfp4_gemm(offloaded,x;check_finite=false))
            @test a==b
        end
        stats=native_nvfp4_stats(offloaded)
        @test stats.portable_device===:cpu
        @test stats.gpu_weight_bytes==stats.prepared_weight_bytes
        @test native_nvfp4_stats(ordinary).gpu_weight_bytes==
            stats.prepared_weight_bytes+stats.portable_weight_bytes
        @test_throws ArgumentError prepare_nvfp4_gemm(source;portable_device=:unknown)
        close(ordinary);close(offloaded)
    end
end

@testset "Host-portable Flint inference and artifact roundtrip" begin
    fixture=joinpath(@__DIR__,"flint","fixtures","tiny")
    raw=load_hf_flint_quantized(fixture;scheme=:nvfp4,device=CUDA.cu)
    ordinary=prepare_flint_nvfp4(raw)
    offloaded=prepare_flint_nvfp4(raw;portable_device=:cpu)
    @test all(!(weight isa AbstractNativeNVFP4Weight) ||
        nvfp4_weight(weight).packed isa Matrix{UInt8} for weight in values(offloaded.parameters))
    @test flint_native_fp4_stats(offloaded).host_portable_weight_bytes>0
    @test flint_native_fp4_stats(offloaded).gpu_weight_bytes==
        flint_native_fp4_stats(offloaded).prepared_weight_bytes
    tokens=[1,3,5,7,9]
    expected=Array(hf_flint_forward(ordinary.model,ordinary.parameters,tokens))
    actual=Array(hf_flint_forward(offloaded.model,offloaded.parameters,tokens))
    @test actual==expected
    @test generate_flint_tokens(offloaded,tokens;temperature=0,max_new_tokens=3,eos_ids=Int[])==
        generate_flint_tokens(ordinary,tokens;temperature=0,max_new_tokens=3,eos_ids=Int[])
    # A service can drop its owned source dictionary after preparation.
    # Both portable serialization and floating parameters must remain usable.
    empty!(raw.parameters)
    @test Array(hf_flint_forward(offloaded.model,offloaded.parameters,tokens))==actual
    mktempdir() do directory
        artifact=joinpath(directory,"host-portable")
        save_flint_quantized(artifact,offloaded)
        restored=prepare_flint_nvfp4(load_flint_quantized(artifact;device=CUDA.cu);portable_device=:cpu)
        @test Array(hf_flint_forward(restored.model,restored.parameters,tokens))==actual
    end
end
