#!/usr/bin/env julia
# CUPTI activity trace including the actual native cuBLASLt kernel name.
using LifeAI, CUDA, Random, JSON3
CUDA.functional() && native_nvfp4_supported() || error("Native NVFP4 CUDA support is required")
CUDA.allowscalar(false)
rng = Xoshiro(20261028)
weight = quantize_nvfp4(CUDA.cu(randn(rng, Float32, 512, 2048)))
prepared = prepare_nvfp4_gemm(weight)
x = CUDA.cu(randn(rng, Float32, 2048, 128))
for _ in 1:3
    native_nvfp4_gemm(prepared, x)
end
CUDA.synchronize()
trace = CUDA.@profile trace=true raw=true native_nvfp4_gemm(prepared, x)
println(trace)
kernel_names = unique(String.(filter(name -> occursin("gemm", lowercase(name)) &&
    occursin("e2m1", lowercase(name)), trace.device.name)))
isempty(kernel_names) && error("CUPTI trace did not contain an E2M1 GEMM kernel")
JSON3.write(stdout, (native_kernels=kernel_names, stats=native_nvfp4_stats(prepared)))
println()
