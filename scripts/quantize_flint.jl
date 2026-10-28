#!/usr/bin/env julia
# julia --project=. scripts/quantize_flint.jl HF_DIR OUTPUT_DIR [int4|int8|nvfp4] [GROUP] [maxabs|mse]
using LifeAI
using JSON3
length(ARGS) in 2:5 || error("Usage: quantize_flint.jl HF_DIR OUTPUT_DIR [int4|int8|nvfp4] [GROUP (NVFP4:16; otherwise:128)] [maxabs|mse]")
scheme = length(ARGS) >= 3 ? Symbol(ARGS[3]) : :int4
group = length(ARGS) >= 4 ? parse(Int, ARGS[4]) : (scheme === :nvfp4 ? 16 : 128)
calibration = length(ARGS) >= 5 ? Symbol(ARGS[5]) : :maxabs
started = time()
bundle = load_hf_flint_quantized(ARGS[1]; scheme, group, calibration)
output = save_flint_quantized(ARGS[2], bundle)
JSON3.write(stdout, (output=output, scheme=scheme, group=group, calibration=calibration,
    parameter_bytes=quantized_parameter_bytes(bundle.parameters), seconds=time()-started))
println()
