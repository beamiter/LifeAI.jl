using Test, LifeAI, CUDA, Random, LinearAlgebra

CUDA.functional() || error("Flint long attention tests require a functional CUDA device")
CUDA.allowscalar(false)
const _FlintLongAttentionExt = Base.get_extension(LifeAI,:LifeAICUDAExt)

# Independent Float64 oracle: explicitly select causal/window keys and include
# the learned sink in the denominator, without any split-K or tiled layout.
function flint_long_attention_reference(query,key,value,sinks,first_key,query_start,window)
    d,heads,queries=size(query)
    result=zeros(Float32,size(query))
    scale=Float64(inv(sqrt(Float32(d))))
    for head in 1:heads, token in 1:queries
        kvhead=(head-1)÷(heads÷size(key,2))+1
        position=query_start+token-1
        indices=[i for i in axes(key,3) if first_key+i-1<=position &&
            (window==0 || first_key+i-1>position-window)]
        isempty(indices) && continue
        scores=[sum(Float64(query[row,head,token])*Float64(key[row,kvhead,i])
            for row in 1:d)*scale for i in indices]
        maximum_score=max(maximum(scores),Float64(sinks[head]))
        weights=exp.(scores.-maximum_score)
        denominator=sum(weights)+exp(Float64(sinks[head])-maximum_score)
        for row in 1:d
            result[row,head,token]=Float32(sum(Float64(value[row,kvhead,i])*weights[j]
                for (j,i) in enumerate(indices))/denominator)
        end
    end
    return result
end

@testset "Flint split-K attention matches independent causal/sink reference" begin
    rng=Xoshiro(20261028081)
    cases=[(4,4,2,1,1,0,0,0),(8,8,2,3,7,11,15,0),
        (16,8,2,17,73,100,156,17),(128,32,4,1,513,1000,1512,513),
        (16,8,2,65,129,31,95,0),(8,4,4,129,257,0,128,33),
        (8,4,2,3,0,0,0,0),(8,4,2,2,4,0,-5,0),
        (8,4,2,33,17,50,20,0),(8,4,2,17,4,100,80,3),
        (16,4,1,4,25,700,719,1),
        (16,4,1,3,65537,0,65534,0)]
    for (d,heads,kvheads,queries,keys,first_key,query_start,window) in cases
        q=randn(rng,Float32,d,heads,queries)
        k=randn(rng,Float32,d,kvheads,keys)
        v=randn(rng,Float32,d,kvheads,keys)
        sinks=randn(rng,Float32,heads).*3f0
        expected=flint_long_attention_reference(q,k,v,sinks,first_key,query_start,window)
        actual=Array(LifeAI._flint_attention_long(CUDA.cu(q),CUDA.cu(k),CUDA.cu(v),
            CUDA.cu(sinks),first_key,query_start,window))
        @test size(actual)==size(q)
        @test all(isfinite,actual)
        @test actual≈expected atol=8f-6 rtol=6f-5
        if keys>0
            parent_k=CUDA.cu(cat(zeros(Float32,d,kvheads,3),k,zeros(Float32,d,kvheads,8);dims=3))
            parent_v=CUDA.cu(cat(zeros(Float32,d,kvheads,3),v,zeros(Float32,d,kvheads,8);dims=3))
            actual=Array(LifeAI._flint_attention_long(CUDA.cu(q),
                view(parent_k,:,:,4:keys+3),view(parent_v,:,:,4:keys+3),CUDA.cu(sinks),
                first_key,query_start,window))
            @test actual≈expected atol=8f-6 rtol=6f-5
        end
    end

    # Highly positive/negative finite scores and learned sinks must remain
    # stable rather than overflow when partitions are merged.
    for sink in (-1000f0,0f0,1000f0)
        q=fill(25f0,8,4,2);k=fill(25f0,8,2,301)
        v=randn(rng,Float32,8,2,301);sinks=fill(sink,4)
        expected=flint_long_attention_reference(q,k,v,sinks,0,299,0)
        actual=Array(LifeAI._flint_attention_long(CUDA.cu(q),CUDA.cu(k),CUDA.cu(v),CUDA.cu(sinks),0,299,0))
        @test all(isfinite,actual)
        @test actual≈expected atol=1f-5 rtol=8f-5
    end
end

@testset "Flint FP32 GEMM prefill matches split-K across tile boundaries" begin
    rng=Xoshiro(91820)
    for (queries,keys,first_key,query_start,window) in
            ((128,8193,37,8102,0),(513,8193,37,7717,0),
             (777,9001,1000,9224,513),(129,257,131,200,33))
        q=CUDA.cu(randn(rng,Float32,16,8,queries))
        k=CUDA.cu(randn(rng,Float32,16,2,keys))
        v=CUDA.cu(randn(rng,Float32,16,2,keys))
        sinks=CUDA.cu(randn(rng,Float32,8))
        expected=Array(_FlintLongAttentionExt._flint_attention_splitk(q,k,v,sinks,
            first_key,query_start,window))
        actual=Array(LifeAI._flint_attention_long(q,k,v,sinks,first_key,query_start,window))
        @test actual≈expected atol=2f-5 rtol=2f-4
    end
    for queries in (16,128,512,4096)
        plan=_FlintLongAttentionExt._flint_gemm_attention_plan(128,32,4,queries,262144)
        @test plan.scratch_bytes<=1024^3
        @test 1<=plan.query_batch<=512
        @test 1<=plan.key_tile<=8192
    end
end

@testset "Flint long attention scratch and guards" begin
    for queries in (1,16,64,128,4096), window in (0,513)
        plan=_FlintLongAttentionExt._flint_long_attention_plan(128,32,queries,262144,window)
        @test 1<=plan.query_batch<=64
        @test 1<=plan.splits<=128
        @test plan.scratch_bytes<=256*1024^2
    end
    q=CUDA.zeros(Float32,8,4,1);k=CUDA.zeros(Float32,8,2,17);v=copy(k);s=CUDA.zeros(Float32,4)
    @test_throws DimensionMismatch LifeAI._flint_attention_long(q,k,CUDA.zeros(Float32,8,2,16),s,0,16,0)
    @test_throws DimensionMismatch LifeAI._flint_attention_long(q,CUDA.zeros(Float32,8,3,17),CUDA.zeros(Float32,8,3,17),s,0,16,0)
    @test_throws DimensionMismatch LifeAI._flint_attention_long(q,k,v,CUDA.zeros(Float32,3),0,16,0)
    @test_throws ArgumentError LifeAI._flint_attention_long(q,k,v,s,0,16,-1)
    @test_throws ArgumentError LifeAI._flint_attention_long(q,k,v,s,0,Int64(typemax(Int32))+1,0)
end

if get(ENV,"LIFEAI_FLINT_256K_ATTENTION_TESTS","false")=="true"
    @testset "Flint full 256K attention uses bounded scratch" begin
        d,heads,kvheads,keys,queries=128,32,4,262144,16
        q=CUDA.ones(Float32,d,heads,queries)
        k=CUDA.zeros(Float32,d,kvheads,keys)
        v=CUDA.ones(Float32,d,kvheads,keys)
        sinks=CUDA.cu(Float32.(range(-2,2;length=heads)))
        host_sinks=Array(sinks)
        result=Array(LifeAI._flint_attention_long(q,k,v,sinks,0,keys-queries,0))
        expected=Array{Float32}(undef,d,heads,queries)
        for head in 1:heads, token in 1:queries
            count=keys-queries+token
            expected[:,head,token].=Float32(count/(count+exp(Float64(host_sinks[head]))))
        end
        @test all(isfinite,result)
        @test result≈expected atol=3f-5 rtol=3f-5
        plan=_FlintLongAttentionExt._flint_gemm_attention_plan(d,heads,kvheads,queries,keys)
        @test plan.scratch_bytes<=1024^3
        @test plan.scratch_bytes<sizeof(k)
    end

    # Random long KV exercises all 32 GEMM key tiles. The previous complete
    # GPU attention path is an independent oracle for indexing and online
    # softmax, including representative and deliberately sharp score ranges.
    function flint_validate_random_256k_attention()
        rng=Xoshiro(294728301)
        d,heads,kvheads,keys,queries=128,32,4,262144,16
        parent_k=CUDA.cu(randn(rng,Float32,d,kvheads,keys+11))
        parent_v=CUDA.cu(randn(rng,Float32,d,kvheads,keys+11))
        k=view(parent_k,:,:,5:keys+4)
        v=view(parent_v,:,:,5:keys+4)
        sinks=CUDA.cu(Float32.(range(-12,8;length=heads)))
        mask=CUDA.cu([i<=keys-queries+j ? 0f0 : -Inf32
            for i in 1:keys,j in 1:queries])
        @testset "Flint random 256K GEMM and split-K match old GPU attention" begin
            for (qscale,kscale,first_position) in
                    ((0.25f0,1f0,0),(1f0,1f0,4096),(3f0,3f0,73111))
                q=CUDA.cu(randn(rng,Float32,d,heads,queries).*qscale)
                key=kscale==1f0 ? k : k.*kscale
                query_start=first_position+keys-queries
                expected=Array(LifeAI._flint_attention_context(q,key,v,sinks,mask))
                gemm=Array(LifeAI._flint_attention_long(q,key,v,sinks,
                    first_position,query_start,0))
                split=Array(_FlintLongAttentionExt._flint_attention_splitk(q,key,v,sinks,
                    first_position,query_start,0))
                @test all(isfinite,expected)
                @test all(isfinite,gemm)
                @test all(isfinite,split)
                @test gemm≈expected atol=3f-5 rtol=3f-4
                @test split≈expected atol=3f-5 rtol=3f-4
                @test gemm≈split atol=3f-5 rtol=3f-4
                q=key=nothing
                GC.gc(true)
                CUDA.reclaim()
            end
        end
    end
    flint_validate_random_256k_attention()
end
