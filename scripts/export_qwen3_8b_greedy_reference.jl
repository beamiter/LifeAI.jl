#!/usr/bin/env julia

using BFloat16s: BFloat16
using CUDA
using JSON3
using LifeAI
using LuxCUDA
using SHA: sha256

length(ARGS) in (2, 3) || error(
    "usage: julia --project=. scripts/export_qwen3_8b_greedy_reference.jl " *
    "MODEL_DIR PROFILE_JSON [OUTPUT_JSON]",
)
model_dir, profile_path = ARGS[1:2]
output_path = length(ARGS) == 3 ?
    ARGS[3] :
    joinpath(
        @__DIR__,
        "..",
        "benchmark_results",
        "week20",
        "qwen3_8b_cuda_greedy_reference.json",
    )
profile = load_qwen3_deployment_profile(profile_path)
profile.variant === :qwen3_8b || error("reference profile must select qwen3_8b")
asset_manifest_path = isabspath(profile.asset_manifest) ?
    profile.asset_manifest :
    joinpath(dirname(abspath(profile_path)), profile.asset_manifest)

println(stderr, "verifying frozen Qwen3-8B assets…")
assets = verify_qwen3_deployment_assets(
    model_dir,
    asset_manifest_path;
    model_id=profile.model_id,
    revision=profile.revision,
)
CUDA.functional() || error("CUDA.jl is not functional")
println(stderr, "loading eager BF16 reference model…")
loaded = load_hf_qwen3_bundle(
    model_dir;
    max_seq_len=profile.context_tokens,
    weight_dtype=BFloat16,
    revision=profile.revision,
    variant=profile.variant,
)
parameters = CUDA.cu(loaded.parameters)
session = init_hf_qwen3_bf16_session(
    merge(loaded, (; parameters));
    context_tokens=profile.context_tokens,
    prefill_chunk_tokens=profile.prefill_chunk_tokens,
)
loaded = nothing
GC.gc(true)
CUDA.reclaim()

probe_piece = encode(
    session.tokenizer,
    " LifeAI XLA reference probe.";
    add_special_tokens=false,
)
isempty(probe_piece) && error("reference probe encoded to no tokens")

reference_prompt(count) = [
    probe_piece[mod1(index, length(probe_piece))]
    for index in 1:count
]

function run_reference_case(name, prompt_tokens, output_tokens)
    prompt_ids = reference_prompt(prompt_tokens)
    generated = Ref(0)
    function reclaim_prefill(_)
        CUDA.synchronize()
        CUDA.reclaim()
        return nothing
    end
    function reclaim_decode(_)
        generated[] += 1
        if mod(generated[], profile.decode_reclaim_interval_tokens) == 0
            CUDA.synchronize()
            CUDA.reclaim()
        end
        return nothing
    end
    started = time_ns()
    result = generate_hf_qwen3_bf16!(
        session,
        prompt_ids;
        max_new_tokens=output_tokens,
        strategy=:greedy,
        stop_token_ids=Int[],
        on_prefill_chunk=reclaim_prefill,
        on_token=reclaim_decode,
    )
    CUDA.synchronize()
    seconds = (time_ns() - started) / 1.0e9
    length(result.generated_ids) == output_tokens ||
        error("$name did not produce $output_tokens tokens")
    println(stderr, "$name completed in $(round(seconds; digits=3))s")
    return (;
        name,
        prompt_tokens,
        generated_tokens=length(result.generated_ids),
        prompt_ids_0_based=prompt_ids .- 1,
        generated_ids_0_based=result.generated_ids .- 1,
    )
end

println(stderr, "exporting CUDA parity cases (64, 65, and 3584 tokens)…")
cases = [
    run_reference_case("single_chunk_64", 64, 32),
    run_reference_case("left_padded_65", 65, 32),
    run_reference_case(
        "full_prompt_3584",
        profile.max_prompt_tokens,
        32,
    ),
]

report = (;
    schema_version=2,
    source="LifeAI eager BF16 CUDA parity reference",
    model_id=profile.model_id,
    revision=profile.revision,
    profile_sha256=bytes2hex(sha256(read(profile_path))),
    asset_manifest_sha256=bytes2hex(sha256(read(asset_manifest_path))),
    verified_asset_files=length(assets.files),
    prompt_description="repeated raw tokenizer probe",
    cases,
)
mkpath(dirname(abspath(output_path)))
open(output_path, "w") do io
    JSON3.pretty(io, report)
    write(io, '\n')
end
println(stderr, "wrote $(abspath(output_path))")
