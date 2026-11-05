#!/usr/bin/env julia
using CUDA, Test, LifeAI
CUDA.functional() || error("CUDA unavailable")
CUDA.allowscalar(false)
include(joinpath(@__DIR__, "..", "test", "ripple", "test_ripple.jl"))
include(joinpath(@__DIR__, "..", "test", "ripple", "test_ripple_offload.jl"))
include(joinpath(@__DIR__, "..", "test", "ripple", "test_ripple_device_reader.jl"))
include(joinpath(@__DIR__, "..", "test", "ripple", "test_ripple_int4_gemv.jl"))
@testset "Ripple CUDA vs official PyTorch including cache eviction and packed weights" begin
    ripple_numerical_checks(CUDA.cu)
    ripple_offload_checks(CUDA.cu)
    ripple_device_reader_checks(CUDA.cu)
    ripple_int4_gemv_checks(CUDA.cu)
end
