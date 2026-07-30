#!/usr/bin/env julia

using Reactant
using BFloat16s: BFloat16
using Dates: now
using JSON3
using LifeAI
using SHA: sha256
using Statistics: median

2 <= length(ARGS) <= 3 || error(
    "usage: julia --project=. scripts/benchmark_qwen3_xla_e2e.jl " *
    "MODEL_DIR OUTPUT_JSON [SAMPLES]",
)

model_dir = abspath(ARGS[1])
output_path = abspath(ARGS[2])
samples = length(ARGS) == 3 ? parse(Int, ARGS[3]) : 5
samples > 0 || error("SAMPLES must be positive")
max_new_tokens = parse(Int, get(ENV, "MAX_NEW_TOKENS", "32"))
max_new_tokens > 1 || error("MAX_NEW_TOKENS must be greater than one")
parse_bool(value::AbstractString) =
    lowercase(strip(value)) in ("1", "true", "yes", "on")
file_sha256(path::AbstractString) = bytes2hex(sha256(read(path)))
tensor_bytes(x::AbstractArray) = length(x) * sizeof(eltype(x))
tensor_bytes(x::NamedTuple) = sum(tensor_bytes, values(x); init=0)
tensor_bytes(x::Tuple) = sum(tensor_bytes, values(x); init=0)
tensor_bytes(x) = 0
tensor_leaves(x::AbstractArray) = 1
tensor_leaves(x::NamedTuple) = sum(tensor_leaves, values(x); init=0)
tensor_leaves(x::Tuple) = sum(tensor_leaves, values(x); init=0)
tensor_leaves(x) = 0
function optional_bool_env(name::AbstractString)
    raw = lowercase(strip(get(ENV, name, "default")))
    raw == "default" && return nothing
    raw in ("1", "true", "yes", "on") && return true
    raw in ("0", "false", "no", "off") && return false
    error("$name must be true, false, or default")
end
enable_while_command_buffer =
    parse_bool(get(ENV, "XLA_COMMAND_BUFFER_WHILE", "false"))
unroll_command_buffer_loops =
    parse_bool(get(ENV, "XLA_COMMAND_BUFFER_UNROLL_LOOPS", "false"))
triton_gemm_any_override = optional_bool_env("XLA_TRITON_GEMM_ANY")
triton_gemm_override = optional_bool_env("XLA_TRITON_GEMM")
cublaslt_override = optional_bool_env("XLA_CUBLASLT")
xla_dump_dir = strip(get(ENV, "XLA_DUMP_DIR", ""))
canonical_prompt = get(
    ENV,
    "PROMPT",
    "<|im_start|>user\n从1开始数：<|im_end|>\n" *
    "<|im_start|>assistant\n<think>\n\n</think>\n\n",
)

json_get(object, key::AbstractString, default=nothing) =
    haskey(object, key) ? object[key] : default

function load_prompt_plan()
    raw_path = strip(get(ENV, "PROMPTS_JSON", ""))
    isempty(raw_path) && return (
        phase_aware=false,
        source=(
            kind=haskey(ENV, "PROMPT") ? "environment" : "builtin",
            path=nothing,
            description=nothing,
        ),
        cases=[(
            name="canonical",
            phase=nothing,
            prompt=canonical_prompt,
            expected_prompt_tokens=nothing,
        )],
    )

    path = abspath(raw_path)
    isfile(path) || error("PROMPTS_JSON does not exist: $path")
    object = JSON3.read(read(path, String))
    entries = if object isa AbstractVector
        object
    elseif haskey(object, "cases")
        object["cases"]
    elseif haskey(object, "prompts")
        object["prompts"]
    else
        error("PROMPTS_JSON object must contain `cases` or `prompts`")
    end
    isempty(entries) && error("PROMPTS_JSON contains no cases")

    cases = map(enumerate(entries)) do (index, entry)
        entry isa AbstractString && error(
            "phase-aware PROMPTS_JSON case $index must be an object",
        )
        name = String(json_get(entry, "name", "case_$index"))
        isempty(name) && error("PROMPTS_JSON case $index has an empty name")
        raw_phase = json_get(entry, "phase", nothing)
        raw_phase === nothing && error(
            "phase-aware PROMPTS_JSON case $(repr(name)) is missing `phase`",
        )
        phase = String(raw_phase)
        haskey(entry, "rendered_prompt") || error(
            "phase-aware PROMPTS_JSON case $(repr(name)) is missing " *
            "`rendered_prompt`",
        )
        prompt = String(entry["rendered_prompt"])
        isempty(prompt) && error(
            "phase-aware PROMPTS_JSON case $(repr(name)) has an empty prompt",
        )
        expected_values = [
            Int(entry[key])
            for key in (
                "expected_prompt_tokens",
                "prompt_tokens",
                "token_count",
            )
            if haskey(entry, key) && entry[key] !== nothing
        ]
        length(unique(expected_values)) <= 1 || error(
            "phase-aware PROMPTS_JSON case $(repr(name)) has conflicting " *
            "token counts",
        )
        expected_prompt_tokens =
            isempty(expected_values) ? nothing : only(unique(expected_values))
        expected_prompt_tokens === nothing ||
            expected_prompt_tokens > 0 ||
            error(
                "phase-aware PROMPTS_JSON case $(repr(name)) has a " *
                "non-positive prompt token count",
            )
        return (; name, phase, prompt, expected_prompt_tokens)
    end

    names = [case.name for case in cases]
    length(unique(names)) == length(names) ||
        error("phase-aware PROMPTS_JSON case names must be unique")
    phases = [case.phase for case in cases]
    allowed_phases = Set(("cold", "warmup", "measured", "semantic_smoke"))
    all(phase -> phase in allowed_phases, phases) || error(
        "phase must be cold, warmup, measured, or semantic_smoke",
    )
    count(==("cold"), phases) == 1 || error(
        "phase-aware PROMPTS_JSON requires exactly one cold case",
    )
    first(phases) == "cold" ||
        error("the cold case must be first in phase-aware PROMPTS_JSON")
    any(==("measured"), phases) || error(
        "phase-aware PROMPTS_JSON requires at least one measured case",
    )
    first_measured = findfirst(==("measured"), phases)
    any(
        index -> phases[index] == "warmup" && index > first_measured,
        eachindex(phases),
    ) && error(
        "warmup cases must precede measured cases in phase-aware PROMPTS_JSON",
    )
    any(
        index ->
            phases[index] == "semantic_smoke" && index < first_measured,
        eachindex(phases),
    ) && error(
        "semantic_smoke cases must follow measured cases in phase-aware " *
        "PROMPTS_JSON",
    )

    description = if object isa AbstractVector
        nothing
    else
        raw_description = json_get(object, "description", nothing)
        raw_description === nothing ? nothing : String(raw_description)
    end
    return (
        phase_aware=true,
        source=(kind="json", path, description),
        cases,
    )
end

Reactant.set_default_backend("gpu")
tokenizer = load_hf_qwen3_tokenizer(model_dir)
prompt_plan = load_prompt_plan()
prepared_cases = map(prompt_plan.cases) do case
    ids = encode(tokenizer, case.prompt; add_special_tokens=false)
    isempty(ids) && error(
        "prompt case $(repr(case.name)) encoded to no tokens",
    )
    case.expected_prompt_tokens === nothing ||
        length(ids) == case.expected_prompt_tokens ||
        error(
            "prompt case $(repr(case.name)) encoded to $(length(ids)) " *
            "tokens; expected $(case.expected_prompt_tokens)",
        )
    merge(case, (; prompt_ids=ids))
end

performance_cases = if prompt_plan.phase_aware
    filter(
        case -> case.phase in ("cold", "warmup", "measured"),
        prepared_cases,
    )
else
    prepared_cases
end
performance_prompt_tokens = length(first(performance_cases).prompt_ids)
all(
    case -> length(case.prompt_ids) == performance_prompt_tokens,
    performance_cases,
) || error(
    "all cold/warmup/measured cases must have the same token length for one " *
    "XLA compilation; got " *
    join(
        (
            "$(case.name)=$(length(case.prompt_ids))"
            for case in performance_cases
        ),
        ", ",
    ),
)
max_seq_len = performance_prompt_tokens + max_new_tokens - 1

load_start = time_ns()
loaded = load_hf_qwen3_model(
    model_dir;
    max_seq_len,
    weight_dtype=BFloat16,
    variant=:qwen3_0_6b,
)
load_seconds = (time_ns() - load_start) / 1.0e9
model = loaded.model
rope = first(values(model.blocks.layers)).attn.rope

pack_start = time_ns()
packed_decode_projections_host =
    LifeAI._bf16a_pack_decode_projections(loaded.parameters)
pack_seconds = (time_ns() - pack_start) / 1.0e9

transfer_start = time_ns()
parameters = Reactant.to_rarray(loaded.parameters)
packed_decode_projections =
    Reactant.to_rarray(packed_decode_projections_host)
packed_decode_parameters = LifeAI._bf16a_compact_decode_parameters(
    parameters,
    packed_decode_projections,
)
cos_table = Reactant.to_rarray(BFloat16.(rope.cos_cache))
sin_table = Reactant.to_rarray(BFloat16.(rope.sin_cache))
mask = Reactant.to_rarray(
    LifeAI._bf16a_causal_mask(
        performance_prompt_tokens,
        performance_prompt_tokens,
    ),
)
key_positions = Reactant.to_rarray(Int32.(collect(1:max_seq_len)))
key_caches = Tuple(
    Reactant.to_rarray(
        zeros(BFloat16, model.head_dim, model.num_kv_heads, max_seq_len, 1),
    )
    for _ in 1:model.num_layers
)
value_caches = Tuple(
    Reactant.to_rarray(
        zeros(BFloat16, model.head_dim, model.num_kv_heads, max_seq_len, 1),
    )
    for _ in 1:model.num_layers
)
compile_token_matrix = Reactant.to_rarray(
    reshape(copy(first(performance_cases).prompt_ids), :, 1),
)
transfer_seconds = (time_ns() - transfer_start) / 1.0e9

function prefill_greedy(ps, tokens, kc, vc, cos_t, sin_t, mask_t)
    logits = LifeAI._bf16a_static_prefill(
        model, ps, tokens, kc, vc, cos_t, sin_t, mask_t,
    )
    last_logits = logits[:, end:end, 1:1]
    next_token = argmax(vec(LifeAI._bf16a_f32(last_logits)))
    return tokens[1:1, 1] .* 0 .+ next_token
end

function decode_greedy_segment(
    packed,
    token,
    kc,
    vc,
    position,
    cos_t,
    sin_t,
    kp,
    generated,
)
    return LifeAI._bf16a_static_generate_greedy_packed!(
        model,
        packed,
        token,
        kc,
        vc,
        position,
        cos_t,
        sin_t,
        kp,
        generated,
    )
end

prefill_compile_start = time_ns()
compiled_prefill = @compile prefill_greedy(
    packed_decode_parameters,
    compile_token_matrix,
    key_caches,
    value_caches,
    cos_table,
    sin_table,
    mask,
)
prefill_compile_seconds = (time_ns() - prefill_compile_start) / 1.0e9

token_buffer = Reactant.to_rarray(fill(1, 1))
position_buffer = Reactant.to_rarray(fill(Int32(performance_prompt_tokens), 1))
generated_buffer = Reactant.to_rarray(zeros(Int, max_new_tokens))
default_command_buffers =
    Reactant.XLA.get_default_debug_options().xla_gpu_enable_command_buffer
command_buffers = if enable_while_command_buffer
    command_buffer_type =
        Reactant.Proto.xla.var"DebugOptions.CommandBufferCmdType"
    unique(vcat(default_command_buffers, command_buffer_type.WHILE))
else
    default_command_buffers
end
decode_xla_debug_options = (;
    xla_gpu_enable_command_buffer=command_buffers,
    xla_gpu_command_buffer_unroll_loops=
        unroll_command_buffer_loops,
)
triton_gemm_any_override === nothing || (
    decode_xla_debug_options = merge(
        decode_xla_debug_options,
        (; xla_gpu_triton_gemm_any=triton_gemm_any_override),
    )
)
triton_gemm_override === nothing || (
    decode_xla_debug_options = merge(
        decode_xla_debug_options,
        (; xla_gpu_enable_triton_gemm=triton_gemm_override),
    )
)
cublaslt_override === nothing || (
    decode_xla_debug_options = merge(
        decode_xla_debug_options,
        (; xla_gpu_enable_cublaslt=cublaslt_override),
    )
)
isempty(xla_dump_dir) || (
    decode_xla_debug_options = merge(
        decode_xla_debug_options,
        (;
            xla_dump_to=abspath(xla_dump_dir),
            xla_dump_hlo_as_text=true,
            xla_dump_buffer_assignment_analysis=true,
        ),
    )
)
decode_compile_options = Reactant.CompileOptions(;
    xla_debug_options=decode_xla_debug_options,
)
decode_compile_start = time_ns()
compiled_decode = @compile compile_options=decode_compile_options decode_greedy_segment(
    packed_decode_parameters,
    token_buffer,
    key_caches,
    value_caches,
    position_buffer,
    cos_table,
    sin_table,
    key_positions,
    generated_buffer,
)
decode_compile_seconds = (time_ns() - decode_compile_start) / 1.0e9

function run_once(case, sample_index)
    request_start = time_ns()
    tokenization_start = time_ns()
    ids = encode(tokenizer, case.prompt; add_special_tokens=false)
    tokenization_seconds = (time_ns() - tokenization_start) / 1.0e9
    ids == case.prompt_ids || error(
        "tokenizer output changed during benchmark for case " *
        repr(case.name),
    )

    prompt_h2d_start = time_ns()
    token_matrix = Reactant.to_rarray(reshape(ids, :, 1))
    prompt_h2d_seconds = (time_ns() - prompt_h2d_start) / 1.0e9
    prefill_start = time_ns()
    token_state = compiled_prefill(
        packed_decode_parameters,
        token_matrix,
        key_caches,
        value_caches,
        cos_table,
        sin_table,
        mask,
    )
    first_token = Array(token_state)[1]
    prefill_seconds = (time_ns() - prefill_start) / 1.0e9
    internal_first_token_ready_seconds =
        (time_ns() - request_start) / 1.0e9
    decode_state_h2d_start = time_ns()
    request_position = Reactant.to_rarray(fill(Int32(length(ids)), 1))
    request_generated = Reactant.to_rarray(zeros(Int, max_new_tokens))
    decode_state_h2d_seconds =
        (time_ns() - decode_state_h2d_start) / 1.0e9
    position_state = request_position
    decode_start = time_ns()
    generated_state = compiled_decode(
        packed_decode_parameters,
        token_state,
        key_caches,
        value_caches,
        position_state,
        cos_table,
        sin_table,
        key_positions,
        request_generated,
    )
    generated = Int.(Array(generated_state))
    bulk_decode_seconds = (time_ns() - decode_start) / 1.0e9

    text_decode_start = time_ns()
    completion = decode(
        tokenizer,
        generated;
        errors=:replace,
        skip_special_tokens=true,
    )
    text_decode_seconds = (time_ns() - text_decode_start) / 1.0e9
    request_seconds = (time_ns() - request_start) / 1.0e9
    return (;
        status="executed",
        name=case.name,
        phase=case.phase,
        included_in_measured_summary=case.phase == "measured" ||
            (!prompt_plan.phase_aware && sample_index > 0),
        sample_index,
        prompt=case.prompt,
        rendered_prompt=case.prompt,
        prompt_token_ids_0_based=ids .- 1,
        expected_prompt_tokens=case.expected_prompt_tokens,
        prompt_tokens=length(ids),
        first_generated_token_id_0_based=first_token - 1,
        generated_tokens=length(generated),
        generated_token_ids_0_based=generated .- 1,
        completion,
        timings=(;
            tokenization_seconds,
            prompt_h2d_seconds,
            prefill_seconds,
            internal_first_token_ready_seconds,
            decode_state_h2d_seconds,
            bulk_decode_seconds,
            text_decode_seconds,
            request_seconds,
        ),
        rates=(;
            prefill_tokens_per_second=length(ids) / prefill_seconds,
            bulk_post_first_tokens_per_second=
                (max_new_tokens - 1) / bulk_decode_seconds,
            aggregate_compute_tokens_per_second=
                max_new_tokens / (prefill_seconds + bulk_decode_seconds),
            completion_tokens_per_second=max_new_tokens / request_seconds,
        ),
        decode_execution=(;
            delivery="bulk_remainder",
            streaming=false,
            graph_control_flow="stablehlo_while",
            pjrt_invocations=1,
            loop_iterations=max_new_tokens - 1,
            intermediate_token_d2h=0,
            final_generated_vector_d2h=1,
        ),
    )
end

function skipped_semantic_result(case, sample_index)
    return (;
        status="skipped",
        skip_reason=(
            "semantic_smoke has $(length(case.prompt_ids)) prompt tokens, but " *
            "the compiled performance shape requires " *
            "$performance_prompt_tokens; semantic verification is covered " *
            "by the canonical single-prompt benchmark result"
        ),
        name=case.name,
        phase=case.phase,
        included_in_measured_summary=false,
        sample_index,
        prompt=case.prompt,
        rendered_prompt=case.prompt,
        prompt_token_ids_0_based=case.prompt_ids .- 1,
        expected_prompt_tokens=case.expected_prompt_tokens,
        prompt_tokens=length(case.prompt_ids),
        generated_tokens=0,
        generated_token_ids_0_based=Int[],
        completion=nothing,
        timings=nothing,
        rates=nothing,
    )
end

function median_timings(measured)
    isempty(measured) && error("there are no measured samples")
    return (;
        tokenization_seconds=median(
            sample.timings.tokenization_seconds for sample in measured
        ),
        prompt_h2d_seconds=median(
            sample.timings.prompt_h2d_seconds for sample in measured
        ),
        prefill_seconds=median(
            sample.timings.prefill_seconds for sample in measured
        ),
        internal_first_token_ready_seconds=median(
            sample.timings.internal_first_token_ready_seconds
            for sample in measured
        ),
        decode_state_h2d_seconds=median(
            sample.timings.decode_state_h2d_seconds for sample in measured
        ),
        bulk_decode_seconds=median(
            sample.timings.bulk_decode_seconds for sample in measured
        ),
        text_decode_seconds=median(
            sample.timings.text_decode_seconds for sample in measured
        ),
        request_seconds=median(
            sample.timings.request_seconds for sample in measured
        ),
        prefill_tokens_per_second=median(
            sample.rates.prefill_tokens_per_second for sample in measured
        ),
        bulk_post_first_tokens_per_second=median(
            sample.rates.bulk_post_first_tokens_per_second
            for sample in measured
        ),
        aggregate_compute_tokens_per_second=median(
            sample.rates.aggregate_compute_tokens_per_second
            for sample in measured
        ),
        completion_tokens_per_second=median(
            sample.rates.completion_tokens_per_second for sample in measured
        ),
    )
end

case_results, warmup, measured = if prompt_plan.phase_aware
    results = map(enumerate(prepared_cases)) do (index, case)
        if case.phase == "semantic_smoke" &&
                length(case.prompt_ids) != performance_prompt_tokens
            skipped_semantic_result(case, index)
        else
            run_once(case, index)
        end
    end
    (
        results,
        filter(
            result ->
                result.status == "executed" && result.phase == "warmup",
            results,
        ),
        filter(
            result ->
                result.status == "executed" && result.phase == "measured",
            results,
        ),
    )
else
    canonical_case = only(prepared_cases)
    implicit_warmup = run_once(canonical_case, 0)
    implicit_measured = [
        run_once(canonical_case, index) for index in 1:samples
    ]
    (
        vcat([implicit_warmup], implicit_measured),
        implicit_warmup,
        implicit_measured,
    )
end

median_result = median_timings(measured)
measured_summary = (;
    aggregation="median over measured requests only",
    included_phase=prompt_plan.phase_aware ? "measured" : "implicit_measured",
    sample_count=length(measured),
    case_names=[sample.name for sample in measured],
    median=median_result,
)

phase_plan = if prompt_plan.phase_aware
    (;
        enabled=true,
        source=prompt_plan.source,
        scheduling="one request per case in JSON order after XLA compilation",
        cold_definition=(
            "first post-compilation request; load, transfer, and XLA compile " *
            "are reported separately"
        ),
        case_order=[case.name for case in prepared_cases],
        planned_cases=[
            (;
                case.name,
                case.phase,
                prompt_tokens=length(case.prompt_ids),
            )
            for case in prepared_cases
        ],
        phase_counts=(;
            cold=count(case -> case.phase == "cold", prepared_cases),
            warmup=count(case -> case.phase == "warmup", prepared_cases),
            measured=count(case -> case.phase == "measured", prepared_cases),
            semantic_smoke=count(
                case -> case.phase == "semantic_smoke",
                prepared_cases,
            ),
        ),
        performance_phases=["cold", "warmup", "measured"],
        performance_prompt_tokens,
        performance_cases_share_compiled_shape=true,
        samples_argument=samples,
        samples_argument_used=false,
        measured_case_names=[sample.name for sample in measured],
        excluded_from_measured_summary=[
            result.name for result in case_results
            if !result.included_in_measured_summary
        ],
        executed_case_names=[
            result.name for result in case_results
            if result.status == "executed"
        ],
        skipped_case_names=[
            result.name for result in case_results
            if result.status == "skipped"
        ],
        semantic_smoke_policy=(
            "execute when token length matches the compiled performance " *
            "shape; otherwise record a skip and use the canonical " *
            "single-prompt result for semantic verification"
        ),
    )
else
    (;
        enabled=false,
        source=prompt_plan.source,
        scheduling="one implicit warmup followed by SAMPLES measured requests",
        case_order=["canonical"],
        phase_counts=(;
            cold=0,
            warmup=1,
            measured=samples,
            semantic_smoke=0,
        ),
        performance_prompt_tokens,
        samples_argument=samples,
        samples_argument_used=true,
        measured_case_names=fill("canonical", samples),
        excluded_from_measured_summary=["canonical_warmup"],
    )
end

common_report = (;
    schema_version=3,
    recorded_at=string(now()),
    benchmark="lifeai_qwen3_bf16_xla_gpu_e2e",
    artifacts=(;
        benchmark_script=(;
            path=relpath(abspath(@__FILE__), abspath(joinpath(@__DIR__, ".."))),
            sha256=file_sha256(abspath(@__FILE__)),
        ),
        optimized_sources=[
            (;
                path=path,
                sha256=file_sha256(
                    abspath(joinpath(@__DIR__, "..", path)),
                ),
            )
            for path in ("src/models/bf16_accel.jl", "src/models/bf16_xla.jl")
        ],
    ),
    runtime=(;
        julia_version=string(VERSION),
        reactant_version=string(pkgversion(Reactant)),
        backend="xla_gpu",
    ),
    model=(;
        source=loaded.source,
        revision=basename(model_dir),
        dtype="bfloat16",
    ),
    workload=(;
        batch_size=1,
        prompt_tokens=performance_prompt_tokens,
        output_tokens=max_new_tokens,
        max_seq_len,
        decoding="greedy_fixed_length",
        eos_early_stop=false,
    ),
    backend="xla_gpu",
    dtype="bfloat16",
    prompt_tokens=performance_prompt_tokens,
    max_new_tokens,
    load_seconds,
    pack_seconds,
    transfer_seconds,
    compile=(;
        prefill_compile_seconds,
        decode_compile_seconds,
        total_compile_seconds=
            prefill_compile_seconds + decode_compile_seconds,
    ),
    initialization_seconds=(
        load_seconds + pack_seconds + transfer_seconds +
        prefill_compile_seconds + decode_compile_seconds
    ),
    parameter_memory=(;
        original_parameter_bytes=tensor_bytes(loaded.parameters),
        original_parameter_tensor_count=tensor_leaves(loaded.parameters),
        packed_projection_bytes=tensor_bytes(
            packed_decode_projections_host,
        ),
        packed_projection_tensor_count=tensor_leaves(
            packed_decode_projections_host,
        ),
        compact_decode_logical_bytes=tensor_bytes(
            packed_decode_parameters,
        ),
        compact_decode_parameter_tensor_count=tensor_leaves(
            packed_decode_parameters,
        ),
        compact_tree_reuses_original_device_buffers=true,
    ),
    xla_options=(;
        while_command_buffer=enable_while_command_buffer,
        command_buffer_unroll_loops=unroll_command_buffer_loops,
        command_buffer_commands=string.(command_buffers),
        triton_gemm_any_override,
        triton_gemm_override,
        cublaslt_override,
        dump_dir=isempty(xla_dump_dir) ? nothing : abspath(xla_dump_dir),
    ),
    execution=(;
        prefill=(;
            logits_projection="last_prompt_token_only",
            packed_qkv=true,
            packed_gate_up=true,
            compact_parameter_tree=true,
            internal_first_token_d2h=1,
            first_token_exposed_to_caller=false,
        ),
        decode=(;
            delivery="bulk_remainder",
            streaming=false,
            graph_control_flow="stablehlo_while",
            pjrt_invocations=1,
            loop_iterations=max_new_tokens - 1,
            intermediate_token_d2h=0,
            final_generated_vector_d2h=1,
            packed_qkv=true,
            packed_gate_up=true,
            compact_parameter_tree=true,
            grouped_query_attention_without_kv_head_expansion=true,
            effective_while_command_buffer=any(
                ==("WHILE"),
                string.(command_buffers),
            ),
        ),
    ),
    timing_contract=(;
        internal_first_token_ready=(
            "request start through the benchmark's internal first-token D2H; " *
            "includes tokenization, prompt H2D, and prefill, but the token is " *
            "not yielded to the caller"
        ),
        prompt_h2d=(
            "Reactant.to_rarray of the runtime prompt token matrix, measured " *
            "inside every request"
        ),
        decode_state_h2d=(
            "Reactant.to_rarray of fresh position and generated-token state; " *
            "runs after internal first-token timing and prevents donated " *
            "state from leaking between requests"
        ),
        bulk_decode=(
            "all post-first-token steps execute in one PJRT invocation " *
            "containing a StableHLO while; there is no intermediate token " *
            "D2H, and one generated-token vector is copied to host at the end"
        ),
        request=(
            "in-process wall time until the complete 32-token result and " *
            "decoded text are available"
        ),
    ),
    phase_plan,
    case_results,
    samples=measured,
    measured_summary,
    median=median_result,
)

report = if prompt_plan.phase_aware
    merge(common_report, (;
        prompt=nothing,
        prompts_json=prompt_plan.source.path,
        warmup_cases=warmup,
    ))
else
    merge(common_report, (;
        prompt=canonical_prompt,
        prompts_json=nothing,
        warmup,
    ))
end

mkpath(dirname(output_path))
open(output_path, "w") do io
    JSON3.pretty(io, report)
    write(io, '\n')
end
println("wrote $output_path")
