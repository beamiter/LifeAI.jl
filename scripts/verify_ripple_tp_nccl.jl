#!/usr/bin/env julia
using LifeAI,CUDA,NCCL,Test
CUDA.functional() || error("CUDA unavailable")
CUDA.allowscalar(false)
include(joinpath(@__DIR__,"..","test","ripple","test_ripple_tensor_parallel.jl"))
count=isempty(ARGS) ? length(CUDA.devices()) : parse(Int,ARGS[1])
devices=collect(CUDA.devices())[1:count]
@testset "Ripple physical NCCL TP$count vs official reference" begin
    ripple_tp_numerical_checks(n->ripple_nccl_backend(devices[1:n]);world_sizes=(count,))
end
