#!/usr/bin/env julia

# Same-prefix Float32 versus BF16 diagnosis for the Chapter 47 step-79
# greedy fork. This script does not change decode math and does not run the
# 32/128/256 allocation profile.

const _DIAG_MODEL_ENV = "LIFEAI_QWEN3_VL_MODEL_DIR"
const _DIAG_REFERENCE_ENV = "LIFEAI_QWEN3_VL_REFERENCE_DIR"
const _DIAG_LONG_REFERENCE_ENV = "LIFEAI_QWEN3_VL_LONG_REFERENCE"
const _DIAG_LONG_REFERENCE_SHA_ENV = "LIFEAI_QWEN3_VL_LONG_REFERENCE_SHA256"
const _DIAG_OUTPUT_ENV = "LIFEAI_QWEN3_VL_STEP79_OUTPUT"
const _DIAG_DEFAULT_GENERATED_STEPS = 79
const _DIAG_PROMPT_TOKENS = 76
const _DIAG_ROPE_DELTA = -56

function _diag_usage(io::IO=stdout)
    println(io, "usage: julia --project=. --startup-file=no \\")
    println(io, "  scripts/diagnose_qwen3_vl_step79_dtype_split.jl \\")
    println(io, "  [MODEL_DIR [REFERENCE_DIR [OUTPUT_JSON]]]")
    println(io)
    println(io, "Compares Float32 and BF16 static greedy decisions with the")
    println(io, "pinned HF Float32 long oracle. This is not an acceptance run.")
    println(io)
    println(io, "  LIFEAI_QWEN3_VL_DIAG_STEPS          default $_DIAG_DEFAULT_GENERATED_STEPS")
    println(io, "  LIFEAI_QWEN3_VL_DIAG_TEACHER_FORCE  default false; feed HF tokens")
    println(io)
    println(io, "  MODEL_DIR:     \$", _DIAG_MODEL_ENV)
    println(io, "  REFERENCE_DIR: \$", _DIAG_REFERENCE_ENV)
    println(io, "  OUTPUT_JSON:   \$", _DIAG_OUTPUT_ENV)
    println(io, "  LONG_ORACLE:   \$", _DIAG_LONG_REFERENCE_ENV)
    println(io, "  ", _DIAG_LONG_REFERENCE_SHA_ENV, "   required pinned file SHA-256")
end

if any(argument -> argument in ("-h", "--help"), ARGS)
    _diag_usage()
    exit()
end
length(ARGS) <= 3 || begin
    _diag_usage(stderr)
    error("expected at most three positional arguments")
end

using BFloat16s: BFloat16
using CUDA
using JSON3
using LifeAI
using SHA: sha256

include(joinpath(@__DIR__, "verify_qwen3_vl_static_cache_cuda.jl"))
include(joinpath(@__DIR__, "qwen3_vl_long_reference_contract.jl"))

function _diag_path(args, index, environment, label; directory::Bool)
    value = if length(args) >= index
        args[index]
    else
        get(ENV, environment, "")
    end
    isempty(value) && error("pass $label or set $environment")
    path = abspath(value)
    if directory
        isdir(path) || error("$label does not exist: $path")
    end
    return path
end

function _diag_select(logits)
    values = vec(Array(@view(logits[:, end, 1])))
    top_ids = partialsortperm(values, 1:2; rev=true)
    token = first(top_ids)
    second = top_ids[2]
    top_logit = Float32(values[token])
    second_logit = Float32(values[second])
    return (;
        token_id_1_based=token,
        token_id_0_based=token - 1,
        top2_token_id_1_based=second,
        top2_token_id_0_based=second - 1,
        top_logit_f32=top_logit,
        top2_logit_f32=second_logit,
        margin_f32=top_logit - second_logit,
    )
end

function _diag_decision(step, selection, cache, hf)
    prompt_tokens = _DIAG_PROMPT_TOKENS
    expected_position = prompt_tokens + step - 1
    expected_rope_delta = _DIAG_ROPE_DELTA
    implied_mrope = step == 1 ? nothing : (cache.position - 1) + cache.rope_delta
    expected_mrope = step == 1 ? nothing :
        (prompt_tokens + step - 2) + expected_rope_delta
    geometry_matches = cache.position == expected_position &&
        cache.rope_delta == expected_rope_delta &&
        implied_mrope == expected_mrope
    return (;
        generated_step=step,
        lifeai_token_id_1_based=selection.token_id_1_based,
        lifeai_token_id_0_based=selection.token_id_0_based,
        lifeai_top2_token_id_0_based=selection.top2_token_id_0_based,
        lifeai_top_logit_f32=selection.top_logit_f32,
        lifeai_top2_logit_f32=selection.top2_logit_f32,
        lifeai_margin_f32=selection.margin_f32,
        hf_token_id_0_based=hf.top1_token_id_0_based,
        hf_top2_token_id_0_based=hf.top2_token_id_0_based,
        hf_top_logit_f32=hf.top1_logit_f32,
        hf_top2_logit_f32=hf.top2_logit_f32,
        hf_margin_f32=hf.margin_f32,
        argmax_matches_hf=selection.token_id_0_based == hf.top1_token_id_0_based,
        lifeai_top1_is_hf_runner_up=
            selection.token_id_0_based == hf.top2_token_id_0_based,
        cache_position=cache.position,
        rope_delta=cache.rope_delta,
        implied_mrope_coordinate=implied_mrope,
        expected_cache_position=expected_position,
        expected_rope_delta,
        expected_mrope_coordinate=expected_mrope,
        geometry_matches,
    )
end

function _diag_generate(
    prepared,
    long_reference,
    generated_steps;
    teacher_force::Bool=false,
)
    prompt_tokens = length(prepared.input_ids)
    prompt_tokens == _DIAG_PROMPT_TOKENS || error(
        "diagnostic prompt length changed",
    )
    only(prepared.rope_layout.rope_deltas) == _DIAG_ROPE_DELTA || error(
        "diagnostic rope_delta changed",
    )
    generated_steps <= long_reference.generated_token_count || error(
        "diagnostic asks for more tokens than the pinned oracle",
    )
    capacity = prompt_tokens + generated_steps - 1
    cache = init_qwen3_vl_static_kv_cache(
        prepared.text_parameters;
        capacity,
        batch_size=1,
    )
    prefill, returned = hf_qwen3_vl_text_prefill_static(
        prepared.text_parameters,
        prepared.input_ids,
        prepared.rope_layout;
        vision_features=prepared.features,
        cache,
        logits_to_keep=1,
        capture_input_embeddings=false,
        capture_final_hidden=false,
    )
    returned === cache || error("static prefill replaced the request cache")
    decisions = NamedTuple[]
    selection = _diag_select(prefill.logits)
    push!(decisions, _diag_decision(
        1,
        selection,
        cache,
        long_reference.top_two[1],
    ))
    token = teacher_force ? long_reference.generated_ids_1_based[1] :
        selection.token_id_1_based
    for step in 2:generated_steps
        logits, returned = hf_qwen3_vl_text_decode_step_static(
            prepared.text_parameters,
            token,
            cache,
        )
        returned === cache || error("static decode replaced the request cache")
        selection = _diag_select(logits)
        push!(decisions, _diag_decision(
            step,
            selection,
            cache,
            long_reference.top_two[step],
        ))
        token = teacher_force ? long_reference.generated_ids_1_based[step] :
            selection.token_id_1_based
    end
    CUDA.synchronize()
    mismatch = findfirst(decision -> !decision.argmax_matches_hf, decisions)
    return (;
        generated_steps,
        capacity,
        final_position=cache.position,
        rope_delta=cache.rope_delta,
        teacher_force,
        first_divergence_step=mismatch === nothing ? nothing :
            decisions[mismatch].generated_step,
        mismatch_count=count(decision -> !decision.argmax_matches_hf, decisions),
        geometry_matches=all(decision -> decision.geometry_matches, decisions),
        decisions,
    )
end

function _diag_release!()
    GC.gc(true)
    CUDA.reclaim()
    return nothing
end

function _diag_classify(float32_run, bfloat16_run; generated_steps::Int, teacher_force::Bool)
    float32_step = float32_run.first_divergence_step
    bfloat16_step = bfloat16_run.first_divergence_step
    geometry_violation = !float32_run.geometry_matches ||
        !bfloat16_run.geometry_matches
    geometry_violation && return "geometry_violation"
    float32_step !== nothing && return "float32_also_diverges"
    bfloat16_step === nothing && return teacher_force ?
        "teacher_forced_both_match" : "both_match_through_$generated_steps"
    if !teacher_force && generated_steps == 79 && bfloat16_step == 79
        decision = bfloat16_run.decisions[79]
        if decision.lifeai_token_id_0_based == 281 &&
            decision.hf_token_id_0_based == 2518 &&
            decision.lifeai_top1_is_hf_runner_up
            return "cross_dtype_boundary"
        end
    end
    return teacher_force ? "teacher_forced_bfloat16_mismatches" :
        "bfloat16_free_run_mismatches"
end

function _diag_runtime_options()
    generated_steps = tryparse(
        Int,
        get(ENV, "LIFEAI_QWEN3_VL_DIAG_STEPS", string(_DIAG_DEFAULT_GENERATED_STEPS)),
    )
    generated_steps === nothing && error(
        "LIFEAI_QWEN3_VL_DIAG_STEPS must be an integer",
    )
    4 <= generated_steps <= 256 || error(
        "LIFEAI_QWEN3_VL_DIAG_STEPS must be from 4 through 256",
    )
    teacher_raw = lowercase(get(ENV, "LIFEAI_QWEN3_VL_DIAG_TEACHER_FORCE", "false"))
    teacher_force = if teacher_raw in ("1", "true", "yes")
        true
    elseif teacher_raw in ("0", "false", "no", "")
        false
    else
        error("LIFEAI_QWEN3_VL_DIAG_TEACHER_FORCE must be true or false")
    end
    return (; generated_steps, teacher_force)
end

model_dir = _diag_path(ARGS, 1, _DIAG_MODEL_ENV, "MODEL_DIR"; directory=true)
reference_dir = _diag_path(
    ARGS,
    2,
    _DIAG_REFERENCE_ENV,
    "REFERENCE_DIR";
    directory=true,
)
output_path = abspath(length(ARGS) >= 3 ? ARGS[3] : get(ENV, _DIAG_OUTPUT_ENV, ""))
isempty(output_path) && error("pass OUTPUT_JSON or set $_DIAG_OUTPUT_ENV")
long_reference_path = abspath(get(ENV, _DIAG_LONG_REFERENCE_ENV, ""))
isempty(long_reference_path) && error("set $_DIAG_LONG_REFERENCE_ENV")
long_reference_sha256 = get(ENV, _DIAG_LONG_REFERENCE_SHA_ENV, "")
isempty(long_reference_sha256) && error(
    "set $_DIAG_LONG_REFERENCE_SHA_ENV to the independently reviewed oracle hash",
)
runtime_options = _diag_runtime_options()
generated_steps = runtime_options.generated_steps
teacher_force = runtime_options.teacher_force

long_reference = _ch47_load_long_generation_reference(
    long_reference_path,
    long_reference_sha256;
    model_id=_STATIC_EXPECTED_MODEL_ID,
    modelscope_revision=_STATIC_EXPECTED_MODELSCOPE_REVISION,
    huggingface_revision=_STATIC_EXPECTED_HF_REVISION,
    asset_sha256=_STATIC_EXPECTED_ASSET_SHA256,
    required_lengths=(generated_steps,),
    expected_image_sha256=_STATIC_EXPECTED_IMAGE_SHA256,
    expected_rendered_prompt_sha256=_STATIC_EXPECTED_RENDERED_PROMPT_SHA256,
    expected_chapter45_reference_sha256=_STATIC_EXPECTED_REFERENCE_SHA256,
    expected_chapter45_metadata_sha256=_STATIC_EXPECTED_METADATA_SHA256,
    expected_source_sha256=Dict(
        "export_qwen3_vl_long_generation_reference.py" =>
            _static_file_sha256(joinpath(
                @__DIR__,
                "export_qwen3_vl_long_generation_reference.py",
            )),
        "export_qwen3_vl_decode_reference.py" =>
            _static_file_sha256(joinpath(
                @__DIR__,
                "export_qwen3_vl_decode_reference.py",
            )),
        "export_qwen3_vl_prefill_reference.py" =>
            _static_file_sha256(joinpath(
                @__DIR__,
                "export_qwen3_vl_prefill_reference.py",
            )),
    ),
)
long_reference.generated_ids_1_based[1:4] ==
    _STATIC_EXPECTED_GREEDY_IDS_1_BASED || error(
        "HF long oracle differs from the frozen Chapter 45 four-token prefix",
    )

CUDA.functional() || error("CUDA.jl is not functional on this machine")
CUDA.allowscalar(false)

function _diag_dtype_run(compute_dtype)
    prepared = _static_prepare_bfloat16(
        model_dir,
        reference_dir;
        compute_dtype,
    )
    report = _diag_generate(
        prepared,
        long_reference,
        generated_steps;
        teacher_force,
    )
    prepared = nothing
    _diag_release!()
    return report
end

bfloat16_run = _diag_dtype_run(BFloat16)
float32_run = _diag_dtype_run(Float32)
classification = _diag_classify(
    float32_run,
    bfloat16_run;
    generated_steps,
    teacher_force,
)
repo_root = normpath(joinpath(@__DIR__, ".."))
report = (;
    schema_version=1,
    closed=false,
    chapter=47,
    status=teacher_force ? "teacher_forced_dtype_split" : "step79_dtype_split",
    acceptance_fixture=false,
    decode_math_changed=false,
    generated_steps,
    teacher_force,
    classification,
    geometry_violation=classification == "geometry_violation",
    environment=(;
        julia=string(VERSION),
        cuda_jl=string(pkgversion(CUDA)),
        cuda_driver=string(CUDA.driver_version()),
        cuda_runtime=string(CUDA.runtime_version()),
        device=CUDA.name(CUDA.device()),
    ),
    long_reference=(;
        file_sha256=long_reference.file_sha256,
        token_timeline_u32le_sha256=long_reference.token_timeline_u32le_sha256,
        claim=long_reference.claim,
    ),
    bfloat16=bfloat16_run,
    float32=float32_run,
    repository_head=readchomp(`git -C $repo_root rev-parse HEAD`),
)
mkpath(dirname(output_path))
open(output_path, "w") do io
    JSON3.pretty(io, report)
    println(io)
end
println(JSON3.write((;
    output_path,
    classification,
    closed=false,
    bfloat16_first_divergence=bfloat16_run.first_divergence_step,
    bfloat16_mismatch_count=bfloat16_run.mismatch_count,
    float32_first_divergence=float32_run.first_divergence_step,
    float32_mismatch_count=float32_run.mismatch_count,
    geometry_matches=(;
        bfloat16=bfloat16_run.geometry_matches,
        float32=float32_run.geometry_matches,
    ),
)))
