#!/usr/bin/env julia
using LifeAI, JSON3, Random
using BFloat16s: BFloat16
if any(i -> ARGS[i] == "--device" && i < length(ARGS) && ARGS[i+1] == "cuda", eachindex(ARGS))
    using CUDA
end

function ripple_main(args)
    isempty(args) && error("Usage: run_ripple.jl MODEL_DIR [--prompt TEXT] [--device cpu|cuda] [--scheme float|int8|int4|nvfp4] [--dtype float32|bf16] [--group N] [--max-new-tokens N] [--prefill-chunk-size N] [--temperature T] [--seed N] [--raw]")
    path = first(args)
    options = Dict("prompt" => "写一个 Julia 二分查找函数。", "device" => "cpu", "scheme" => "float",
        "dtype" => "bf16", "group" => "", "max-new-tokens" => "128", "prefill-chunk-size" => "1024",
        "temperature" => "0", "seed" => "42", "resident-gib" => "", "quantized-output" => "")
    raw = false
    streamed = false
    i = 2
    while i <= length(args)
        if args[i] == "--raw"
            raw = true
            i += 1
        elseif args[i] == "--streamed"
            streamed = true
            i += 1
        else
            startswith(args[i], "--") && i < length(args) || error("invalid Ripple CLI argument")
            key = args[i][3:end]
            haskey(options, key) || error("unknown option --$key")
            options[key] = args[i+1]
            i += 2
        end
    end
    options["device"] in ("cpu", "cuda") || error("invalid --device")
    options["dtype"] in ("float32", "bf16") || error("invalid --dtype")
    scheme = Symbol(options["scheme"])
    scheme in (:float, :int8, :int4, :nvfp4) || error("invalid --scheme")
    count = parse(Int, options["max-new-tokens"])
    chunk = parse(Int, options["prefill-chunk-size"])
    temperature = parse(Float32, options["temperature"])
    count >= 0 && chunk > 0 && isfinite(temperature) && temperature >= 0 || error("invalid generation options")
    group = isempty(options["group"]) ? nothing : parse(Int, options["group"])
    device = identity
    if options["device"] == "cuda"
        CUDA.functional() || error("CUDA unavailable")
        CUDA.allowscalar(false)
        device = CUDA.cu
    end
    tokenizer = load_hf_ripple_tokenizer(path)
    started = time()
    dtype = options["dtype"] == "bf16" ? BFloat16 : Float32
    budget = isempty(options["resident-gib"]) ? nothing : round(Int,parse(Float64,options["resident-gib"])*2.0^30)
    if !isempty(options["quantized-output"])
        quantize_ripple_checkpoint(path,options["quantized-output"];device,scheme,group,dtype,
            on_progress=(name,index,total)->(index%20==0 && println(stderr,"Quantized $index/$total: $name")))
        path = options["quantized-output"]
    end
    bundle = if isfile(joinpath(path,"manifest.json"))
        load_ripple_quantized(path;device,resident_bytes=budget)
    elseif streamed
        scheme===:float || error("--streamed requires --scheme float")
        load_hf_ripple_streamed(path;device,dtype)
    else
        budget===nothing || error("--resident-gib requires a saved Ripple artifact; use --quantized-output first")
        load_hf_ripple_model(path; device, scheme, group,dtype,
            quantize_on_device=options["device"] == "cuda")
    end
    load_seconds = time()-started
    prompt = raw ? options["prompt"] : [(role="user", content=options["prompt"])]
    started = time()
    output = generate_ripple(bundle, tokenizer, prompt; max_new_tokens=count, temperature,
        prefill_chunk_size=chunk, rng=Xoshiro(parse(Int, options["seed"])))
    println(JSON3.write((text=output.text, token_ids=output.token_ids, generated_tokens=length(output.token_ids),
        device=options["device"], scheme=bundle.quantization, load_seconds=load_seconds,
        generation_seconds=time()-started,checksum_backend=hasproperty(bundle,:checksum_backend) ? bundle.checksum_backend : :none,offload=ripple_offload_stats(bundle))))
end
if abspath(PROGRAM_FILE) == @__FILE__
    ripple_main(ARGS)
end
