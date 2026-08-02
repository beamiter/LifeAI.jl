#!/usr/bin/env julia

# Week 23 acceptance: the device-resident sampling policy must pick the same
# tokens as the host policy when both replay the same uniform sequence, and it
# must remove the per-token logits transfer.
#
#   julia --project=. --startup-file=no \
#     scripts/verify_qwen3_xla_device_sampling.jl MODEL_DIR OUTPUT_JSON

ENV["XLA_REACTANT_GPU_MEM_FRACTION"] =
    get(ENV, "XLA_REACTANT_GPU_MEM_FRACTION", "0.85")
ENV["XLA_REACTANT_GPU_PREALLOCATE"] =
    get(ENV, "XLA_REACTANT_GPU_PREALLOCATE", "false")

using LifeAI
using JSON3
using Random: MersenneTwister
using Reactant

function main(args)
    length(args) >= 2 || error(
        "usage: verify_qwen3_xla_device_sampling.jl MODEL_DIR OUTPUT_JSON",
    )
    model_dir = args[1]
    output_path = args[2]
    backend = get(ENV, "LIFEAI_WEEK23_BACKEND", "gpu")
    context_tokens = parse(Int, get(ENV, "LIFEAI_WEEK23_CONTEXT", "512"))
    max_new_tokens = parse(Int, get(ENV, "LIFEAI_WEEK23_TOKENS", "48"))
    Reactant.set_default_backend(backend)

    prompt = "用一句话说明为什么采样解码要把策略放进编译好的执行体。"
    rng = MersenneTwister(20260801)
    uniforms = Float32[rand(rng, Float32) for _ in 1:max_new_tokens]

    reports = Dict{Symbol,Any}()
    tokens = Dict{Symbol,Vector{Int}}()
    for strategy in (:device_sample, :sample)
        session = load_hf_qwen3_bf16_xla_session(
            model_dir;
            context_tokens,
            prefill_chunk_tokens=64,
            strategy,
        )
        rendered = apply_qwen3_chat_template(
            session.tokenizer,
            [(role="user", content=prompt)];
            add_generation_prompt=true,
            enable_thinking=false,
        )
        prompt_ids = encode(
            session.tokenizer,
            rendered;
            add_special_tokens=false,
        )
        # One warm run so the reported decode rate is steady state.
        warm = generate_hf_qwen3_bf16_xla!(
            session,
            prompt_ids;
            max_new_tokens,
            temperature=0.6,
            top_k=20,
            top_p=0.95,
            sample_uniforms=uniforms,
        )
        result = generate_hf_qwen3_bf16_xla!(
            session,
            prompt_ids;
            max_new_tokens,
            temperature=0.6,
            top_k=20,
            top_p=0.95,
            sample_uniforms=uniforms,
        )
        tokens[strategy] = result.generated_ids
        reports[strategy] = Dict(
            "strategy" => String(strategy),
            "prompt_tokens" => length(prompt_ids),
            "generated_tokens" => length(result.generated_ids),
            "warm_generated_tokens" => length(warm.generated_ids),
            "replay_is_deterministic" =>
                warm.generated_ids == result.generated_ids,
            "prefill_seconds" => result.prefill_seconds,
            "decode_seconds" => result.decode_seconds,
            "tokens_per_second" => result.tokens_per_second,
            "prefill_compile_seconds" =>
                session.load_metrics.prefill_compile_seconds,
            "decode_compile_seconds" =>
                session.load_metrics.decode_compile_seconds,
            "sample_top_k" => session.sample_top_k,
            "completion" => result.completion,
            "generated_ids" => result.generated_ids,
        )
        session = nothing
        GC.gc(true)
    end

    device_ids = tokens[:device_sample]
    host_ids = tokens[:sample]
    shared = min(length(device_ids), length(host_ids))
    agreement = count(i -> device_ids[i] == host_ids[i], 1:shared)
    first_divergence = findfirst(i -> device_ids[i] != host_ids[i], 1:shared)
    speedup = reports[:sample]["tokens_per_second"] <= 0 ? nothing :
        reports[:device_sample]["tokens_per_second"] /
        reports[:sample]["tokens_per_second"]

    report = Dict(
        "model_dir" => model_dir,
        "backend" => backend,
        "context_tokens" => context_tokens,
        "requested_tokens" => max_new_tokens,
        "temperature" => 0.6,
        "top_k" => 20,
        "top_p" => 0.95,
        "uniforms" => uniforms,
        "device_sample" => reports[:device_sample],
        "host_sample" => reports[:sample],
        "token_agreement" => "$(agreement) / $(shared)",
        "identical_tokens" =>
            length(device_ids) == length(host_ids) && agreement == shared,
        "first_divergence_step" =>
            first_divergence === nothing ? nothing : first_divergence,
        "decode_speedup" => speedup,
    )
    report["closed"] =
        report["identical_tokens"] === true &&
        reports[:device_sample]["replay_is_deterministic"] === true &&
        speedup !== nothing && speedup > 1

    mkpath(dirname(output_path))
    open(output_path, "w") do io
        JSON3.pretty(io, report)
    end
    println(JSON3.write(report))
    return report["closed"] === true ? 0 : 1
end

exit(main(ARGS))
