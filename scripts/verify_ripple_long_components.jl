using LifeAI,CUDA,NCCL,JSON3,Test,Random,LinearAlgebra
using BFloat16s:BFloat16
CUDA.allowscalar(false);BLAS.set_num_threads(1)
include(joinpath(@__DIR__,"..","test","ripple","test_ripple_tensor_parallel.jl"))
include(joinpath(@__DIR__,"..","test","ripple","test_ripple_projection.jl"))
include(joinpath(@__DIR__,"..","test","ripple","test_ripple_prefix.jl"))
include(joinpath(@__DIR__,"..","test","ripple","test_ripple_rope.jl"))
include(joinpath(@__DIR__,"..","test","ripple","test_ripple_jobs.jl"))
@testset "Ripple long-context CUDA components" begin
    devices=collect(CUDA.devices())[1:1]
    ripple_tp_projection_checks(devices)
    ripple_tp_prefix_checks(n->ripple_nccl_backend(devices[1:n]);world_sizes=(1,))
    reference=isempty(ARGS) ? RIPPLE_ROPE_FIXTURE : ARGS[1]
    ripple_rope_reference_checks(;device=CUDA.cu,reference_path=reference)
    ext=Base.get_extension(LifeAI,:LifeAICUDAExt)
    plan=ext._flint_gemm_attention_plan(128,10,1,2048,1048576;max_query_batch=2048)
    @test plan.query_batch==2048
    @test plan.scratch_bytes<1024^3
    Random.seed!(1048576)
    q=CUDA.CuArray(randn(Float32,128,10,2049));k=CUDA.CuArray(randn(Float32,128,1,8193));v=CUDA.CuArray(randn(Float32,128,1,8193));s=CUDA.fill(-Inf32,10)
    baseline=Array(LifeAI._flint_attention_long(q,k,v,s,0,6144,0;max_query_batch=512))
    larger=Array(LifeAI._flint_attention_long(q,k,v,s,0,6144,0;max_query_batch=2048))
    @test larger≈baseline atol=3f-5 rtol=3f-5
    @test all(isfinite,larger)
end
