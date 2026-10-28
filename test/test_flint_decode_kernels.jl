using Test, LifeAI, CUDA, Random

CUDA.functional() || error("Flint fused CUDA kernel tests require a functional GPU")
CUDA.allowscalar(false)
const _FlintDecodeKernelExt = Base.get_extension(LifeAI,:LifeAICUDAExt)

function flint_kernel_norm_reference(input, weight, epsilon)
    return input .* inv.(sqrt.(sum(abs2,input;dims=1)./size(input,1) .+ Float32(epsilon))) .*
        reshape(weight,:,ntuple(_->1,ndims(input)-1)...)
end

function flint_kernel_rope_reference(input, positions, theta)
    d=size(input,1);half=d÷2
    frequencies=Float32(theta) .^ (-Float32.(0:half-1) .* (2f0/d))
    angles=reshape(frequencies,half,1) .* reshape(Float32.(positions),1,:)
    cosine,sine=reshape(cos.(angles),half,1,:),reshape(sin.(angles),half,1,:)
    a,b=input[1:half,:,:],input[half+1:end,:,:]
    return cat(a.*cosine.-b.*sine,b.*cosine.+a.*sine;dims=1)
end

@testset "Fused Flint RMSNorm across token and head dimensions" begin
    rng=Xoshiro(20261028031)
    for shape in ((1,1),(17,3),(128,32,1),(128,4,16),(2048,1),(4096,1),(2048,16))
        input=randn(rng,Float32,shape)
        weight=randn(rng,Float32,shape[1])
        expected=flint_kernel_norm_reference(input,weight,1f-5)
        device,scale=CUDA.cu(input),CUDA.cu(weight)
        actual=Array(LifeAI._flint_norm(device,scale,1f-5))
        old=Array(invoke(LifeAI._flint_norm,Tuple{Any,Any,Any},device,scale,1f-5))
        @test size(actual)==shape
        @test actual≈expected atol=3f-6 rtol=3f-6
        @test actual≈old atol=3f-6 rtol=3f-6
        @test Array(device)==input
        @test Array(scale)==weight
    end
    @test all(iszero,Array(LifeAI._flint_norm(CUDA.zeros(Float32,2048,1),CUDA.ones(Float32,2048),1f-5)))
    @test_throws DimensionMismatch LifeAI._flint_norm(CUDA.zeros(Float32,17,1),CUDA.ones(Float32,16),1f-5)
    @test size(LifeAI._flint_norm(CUDA.zeros(Float32,17,0),CUDA.ones(Float32,17),1f-5))==(17,0)
end

@testset "Cached fused Flint RoPE matches CPU trigonometric windows" begin
    rng=Xoshiro(20261028032)
    for (shape,position) in (((4,4,1),0),((128,32,1),1),((128,4,1),2047),
            ((128,32,1),2048),((128,32,1),8191),((128,4,1),262143),
            ((256,8,1),8000),((128,32,16),64),((128,4,16),2040))
        input=randn(rng,Float32,shape)
        positions=position:position+shape[3]-1
        expected=flint_kernel_rope_reference(input,positions,1000000f0)
        device=CUDA.cu(input)
        actual=Array(LifeAI._flint_rope(device,positions,1000000f0))
        old=Array(invoke(LifeAI._flint_rope,Tuple{Any,Any,Any},device,positions,1000000f0))
        @test size(actual)==shape
        @test actual≈expected atol=1f-6 rtol=1f-6
        @test actual==old
        @test Array(device)==input
    end
    # Exercise eviction after more windows than the bounded per-context cache.
    input=CUDA.cu(randn(rng,Float32,128,4,1))
    for window in 0:10
        positions=(window*2048):(window*2048)
        actual=LifeAI._flint_rope(input,positions,1000000f0)
        @test Array(actual)≈flint_kernel_rope_reference(Array(input),positions,1000000f0) atol=1f-6 rtol=1f-6
    end
    context=_FlintDecodeKernelExt._flint_decode_kernel_context()
    @test length(context.rope_tables)<=8
    @test length(context.rope_order)==length(context.rope_tables)
    tasks=[@async begin
        positions=(i*2048):(i*2048)
        Array(LifeAI._flint_rope(input,positions,1000000f0))
    end for i in 0:9]
    for (i,task) in enumerate(tasks)
        positions=((i-1)*2048):((i-1)*2048)
        @test fetch(task)≈flint_kernel_rope_reference(Array(input),positions,1000000f0) atol=1f-6 rtol=1f-6
    end
end

@testset "Flint GPU greedy sampling preserves finite checks and Julia ties" begin
    rng=Xoshiro(20261028033)
    for count in (1,17,1024,150144,1000000)
        input=randn(rng,Float32,count)
        expected=LifeAI._sample_token(input,Xoshiro(1);temperature=0)
        actual=LifeAI._flint_sample_token(CUDA.cu(reshape(input,:,1)),Xoshiro(1);temperature=0)
        @test actual==expected
    end
    for input in (Float32[-0.0,0.0],Float32[0.0,-0.0],zeros(Float32,1025),
            Float32[1,3,3,2],Float32[-3,-2,-2,-5])
        @test LifeAI._flint_sample_token(CUDA.cu(reshape(input,:,1)),rng;temperature=0)==argmax(input)
    end
    for value in (Inf32,-Inf32,NaN32), where_index in (1,256,150144)
        input=zeros(Float32,150144,1);input[where_index]=value
        @test_throws ArgumentError LifeAI._flint_sample_token(CUDA.cu(input),rng;temperature=0)
    end
    device=CUDA.zeros(Float32,17,1)
    for temperature in (true,false,-1f0,NaN32,Inf32)
        @test_throws ArgumentError LifeAI._flint_sample_token(device,rng;temperature)
    end
    @test_throws ArgumentError LifeAI._flint_sample_token(CUDA.zeros(Float32,0,1),rng;temperature=0)
    @test_throws ArgumentError LifeAI._flint_sample_token(CUDA.zeros(Float32,1000001,1),rng;temperature=0)
    host=Float32[0.4,2.1,-1.5,0.8]
    @test LifeAI._flint_sample_token(CUDA.cu(reshape(host,:,1)),Xoshiro(22);temperature=1,top_k=3,top_p=0.9)==
        LifeAI._sample_token(host,Xoshiro(22);temperature=1,top_k=3,top_p=0.9)
    @test LifeAI._flint_sample_token(CUDA.cu(reshape(host,:,1)),rng;temperature=1,top_k=3,top_p=0.9,sample_uniform=0.5)==
        LifeAI._sample_token(host,rng;temperature=1,top_k=3,top_p=0.9,sample_uniform=0.5)
    hosts=[randn(rng,Float32,150144) for _ in 1:8]
    tasks=[@async LifeAI._flint_sample_token(CUDA.cu(reshape(host,:,1)),Xoshiro(0);temperature=0) for host in hosts]
    @test fetch.(tasks)==argmax.(hosts)
end
