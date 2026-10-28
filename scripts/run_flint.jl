#!/usr/bin/env julia
# julia --project=. scripts/run_flint.jl MODEL_DIR [--prompt TEXT] [--device cpu|cuda]
using LifeAI
using JSON3
using Random

# Import at top level before main's world age, only when CUDA is requested.
if any(i -> ARGS[i] == "--device" && i < length(ARGS) && ARGS[i+1] == "cuda", eachindex(ARGS))
    using CUDA
end

function main(args)
    isempty(args) && error("Usage: run_flint.jl MODEL_DIR [--prompt TEXT] [--device cpu|cuda] [--scheme float|int8|int4|nvfp4] [--nvfp4-backend auto|native|reference] [--group N] [--max-new-tokens 128] [--temperature 1.0] [--seed 42] [--save-quantized DIR] [--raw]")
    path = args[1]
    options = Dict("prompt" => "请用中文简要介绍你自己。", "device" => "cpu", "scheme" => "int4", "nvfp4-backend" => "auto", "group" => "", "max-new-tokens" => "128", "temperature" => "1.0", "seed" => "42")
    raw = false
    i = 2
    while i <= length(args)
        if args[i] == "--raw"
            raw = true
            i += 1
        else
            startswith(args[i], "--") && i < length(args) || error("Invalid CLI argument: $(args[i])")
            key = args[i][3:end]
            key in (keys(options)..., "save-quantized") || error("Unknown option --$key")
            options[key] = args[i+1]
            i += 2
        end
    end
    options["device"] in ("cpu", "cuda") || error("--device must be cpu or cuda")
    options["nvfp4-backend"] in ("auto", "native", "reference") || error("--nvfp4-backend must be auto, native, or reference")
    options["nvfp4-backend"] == "native" && options["device"] != "cuda" && error("native NVFP4 GEMM requires --device cuda")
    scheme = Symbol(options["scheme"])
    scheme in (:float, :int8, :int4, :nvfp4) || error("--scheme must be float, int8, int4, or nvfp4")
    count = parse(Int, options["max-new-tokens"])
    temperature = parse(Float32, options["temperature"])
    seed = parse(Int, options["seed"])
    group = isempty(options["group"]) ? (scheme === :nvfp4 ? 16 : 128) : parse(Int, options["group"])
    count >= 0 || error("--max-new-tokens must be nonnegative")
    isfinite(temperature) && temperature >= 0 || error("--temperature must be finite and nonnegative")
    group > 0 || error("--group must be positive")
    scheme != :int4 || iseven(group) || error("INT4 --group must be even")
    scheme != :nvfp4 || group == 16 || error("NVFP4 --group must be 16")
    raw && isempty(options["prompt"]) && error("--raw prompt must not be empty")
    device = identity
    if options["device"] == "cuda"
        CUDA.functional() || error("CUDA is not functional")
        CUDA.allowscalar(false)
        device = CUDA.cu
    end
    started = time()
    bundle = if isfile(joinpath(path, "manifest.json"))
        load_flint_quantized(path; device)
    elseif scheme == :float
        load_hf_flint_model(path; device)
    else
        load_hf_flint_quantized(path; scheme, group, device)
    end
    has_nvfp4 = any(weight -> weight isa NVFP4Weight, values(bundle.parameters))
    use_native = options["nvfp4-backend"] == "native" ||
        (options["nvfp4-backend"] == "auto" && options["device"] == "cuda" && has_nvfp4)
    if use_native
        has_nvfp4 || error("native NVFP4 GEMM requires NVFP4 weights")
        bundle = prepare_flint_nvfp4(bundle)
    end
    load_seconds = time() - started
    if haskey(options, "save-quantized")
        save_flint_quantized(options["save-quantized"], bundle)
    end
    tokenizer = load_hf_flint_tokenizer(path)
    prompt = raw ? options["prompt"] : [(role="user", content=options["prompt"])]
    started = time()
    result = generate_flint(bundle, tokenizer, prompt;
        max_new_tokens=count, temperature, rng=Xoshiro(seed))
    backend_stats = flint_native_fp4_stats(bundle)
    JSON3.write(stdout, (text=result.text, token_ids=result.token_ids,
        generated_tokens=length(result.token_ids), load_seconds=load_seconds,
        generation_seconds=time()-started, parameter_bytes=quantized_parameter_bytes(bundle.parameters),
        device=options["device"], nvfp4_backend=backend_stats.backend,
        native_fp4_calls=backend_stats.native_calls))
    println()
end
main(ARGS)
