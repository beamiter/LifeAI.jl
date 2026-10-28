using Test, LifeAI, CUDA, Random, LinearAlgebra

CUDA.functional() || error("A functional CUDA device is required")
CUDA.allowscalar(false)
native_nvfp4_supported() || error("Native NVFP4 multi-projection requires Blackwell")
const _FP4MultiExt = Base.get_extension(LifeAI, :LifeAICUDAExt)

function fp4_multi_reference(weight, input)
    activation = quantize_nvfp4(permutedims(input))
    return dequantize_nvfp4(weight) * permutedims(dequantize_nvfp4(activation))
end

@testset "Native FP4 shared activation groups up to sixteen projections" begin
    rng = Xoshiro(2026102803)
    for k in (1,19,65,2048), count in (2,3,4,6,14,16)
        hosts = ntuple(i -> quantize_nvfp4(randn(rng,Float32,16i+1,k) .* (0.03f0*10f0^mod(i-1,3))),count)
        weights = map(w -> prepare_nvfp4_gemm(LifeAI._flint_weight_to_device(CUDA.cu,w)),hosts)
        for input in (randn(rng,Float32,k,1),zeros(Float32,k,1),fill(nextfloat(0f0),k,1))
            device = CUDA.cu(input)
            values = LifeAI.native_nvfp4_gemm_multi(weights,device;check_finite=false)
            @test values isa Tuple && length(values) == count
            for i in 1:count
                actual = Array(values[i])
                single = Array(native_nvfp4_gemm(weights[i],device;check_finite=false))
                @test actual == single
                @test actual ≈ fp4_multi_reference(hosts[i],input) atol=4f-4 rtol=8f-4
                @test size(actual) == (size(hosts[i],1),1)
            end
            context = _FP4MultiExt._native_fp4_context()
            buffers = context.decode_buffers[weights[1].padded_k]
            portable = quantize_nvfp4(permutedims(input))
            @test Array(buffers.multi_alpha)[1:count] == [w.global_scale*portable.global_scale for w in hosts]
        end
        @test all(native_nvfp4_stats(w).native_calls == 6 for w in weights)
        close.(weights)
    end
end

@testset "Shared projection outputs survive scratch reuse and stream handoff" begin
    rng = Xoshiro(2026102804)
    hosts = ntuple(i -> quantize_nvfp4(randn(rng,Float32,64i,256)),3)
    weights = map(w -> prepare_nvfp4_gemm(LifeAI._flint_weight_to_device(CUDA.cu,w)),hosts)
    inputs = [randn(rng,Float32,256,1) for _ in 1:8]
    devices = CUDA.cu.(inputs)
    outputs = [LifeAI.native_nvfp4_gemm_multi(weights,x;check_finite=false) for x in devices]
    for t in eachindex(outputs), i in 1:3
        @test Array(outputs[t][i]) ≈ fp4_multi_reference(hosts[i],inputs[t]) atol=5f-4 rtol=8f-4
    end
    tasks = [@async begin
        result = LifeAI.native_nvfp4_gemm_multi(weights,devices[t];check_finite=false)
        yield()
        map(Array,result)
    end for t in 1:4]
    for t in eachindex(tasks)
        result=fetch(tasks[t])
        for i in 1:3
            @test result[i] ≈ fp4_multi_reference(hosts[i],inputs[t]) atol=5f-4 rtol=8f-4
        end
    end

    # Prefill can vary without throwing away the hot single-token descriptor.
    decode_plans=map(w -> w.decode_plan,weights)
    for n in (3,1,7,1)
        input=randn(rng,Float32,256,n)
        results=LifeAI.native_nvfp4_gemm_multi(weights,CUDA.cu(input))
        for i in 1:3
            @test Array(results[i]) ≈ fp4_multi_reference(hosts[i],input) atol=5f-4 rtol=8f-4
            @test weights[i].decode_plan === decode_plans[i]
            @test native_nvfp4_stats(weights[i]).cached_shapes == 2
        end
    end
    single=LifeAI.native_nvfp4_gemm_multi((weights[1],),devices[1])
    @test length(single) == 1
    @test Array(single[1]) ≈ fp4_multi_reference(hosts[1],inputs[1]) atol=5f-4 rtol=8f-4

    before=map(w -> native_nvfp4_stats(w).native_calls,weights)
    @test_throws ArgumentError LifeAI.native_nvfp4_gemm_multi((),devices[1])
    @test_throws ArgumentError LifeAI.native_nvfp4_gemm_multi((nothing,),devices[1])
    @test_throws ArgumentError LifeAI.native_nvfp4_gemm_multi(ntuple(i -> weights[mod1(i,3)],17),devices[1])
    @test_throws DimensionMismatch LifeAI.native_nvfp4_gemm_multi(weights,CUDA.zeros(Float32,257,1))
    @test_throws ArgumentError LifeAI.native_nvfp4_gemm_multi(weights,CUDA.zeros(Float32,256,0))
    bad=prepare_nvfp4_gemm(quantize_nvfp4(CUDA.zeros(Float32,64,255)))
    @test_throws DimensionMismatch LifeAI.native_nvfp4_gemm_multi((weights[1],bad),devices[1])
    @test map(w -> native_nvfp4_stats(w).native_calls,weights) == before
    close(bad)
    for value in (Inf32,-Inf32,NaN32)
        @test_throws ArgumentError LifeAI.native_nvfp4_gemm_multi(weights,CUDA.fill(value,256,1))
    end
    close(weights[3])
    @test_throws ArgumentError LifeAI.native_nvfp4_gemm_multi(weights,devices[1])
    close.(weights)
    @test all(w -> w.plan === nothing && w.decode_plan === nothing,weights)
end
