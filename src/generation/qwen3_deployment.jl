using BFloat16s: BFloat16
using JSON3
using MLDataDevices: cpu_device
using Random: AbstractRNG, default_rng
using SHA: sha256

const _QWEN3_DEPLOYMENT_PROFILE_FIELDS = Set((
    "schema_version",
    "name",
    "model_id",
    "revision",
    "variant",
    "weight_dtype",
    "context_tokens",
    "max_prompt_tokens",
    "max_new_tokens",
    "prefill_chunk_tokens",
    "prefill_reclaim_interval_chunks",
    "decode_reclaim_interval_tokens",
    "enable_thinking",
    "strategy",
    "temperature",
    "top_k",
    "top_p",
    "minimum_gpu_bytes",
    "workspace_reserve_bytes",
    "asset_manifest",
))

"""
    Qwen3DeploymentProfile

Frozen, fail-closed settings for one local Qwen3 deployment. Context limits are
total token limits: the rendered prompt plus requested output must fit inside
`context_tokens`.
"""
struct Qwen3DeploymentProfile
    schema_version::Int
    name::String
    model_id::String
    revision::String
    variant::Symbol
    weight_dtype::Symbol
    context_tokens::Int
    max_prompt_tokens::Int
    max_new_tokens::Int
    prefill_chunk_tokens::Int
    prefill_reclaim_interval_chunks::Int
    decode_reclaim_interval_tokens::Int
    enable_thinking::Bool
    strategy::Symbol
    temperature::Float32
    top_k::Int
    top_p::Float32
    minimum_gpu_bytes::Int
    workspace_reserve_bytes::Int
    asset_manifest::String
end

function _qwen3_profile_value(object, key::String)
    haskey(object, key) || throw(ArgumentError(
        "Qwen3 deployment profile is missing required field $(repr(key))",
    ))
    return object[key]
end

function _qwen3_json_object(path::AbstractString, label::AbstractString)
    value = try
        JSON3.read(read(path, String))
    catch error
        throw(ArgumentError(
            "invalid $label JSON in $path: $(sprint(showerror, error))",
        ))
    end
    value isa JSON3.Object || throw(ArgumentError("$label root must be an object"))
    return value
end

function _qwen3_required_string(object, key::String)
    value = _qwen3_profile_value(object, key)
    value isa AbstractString || throw(ArgumentError("$key must be a string"))
    return String(value)
end

function _qwen3_required_integer(object, key::String)
    value = _qwen3_profile_value(object, key)
    value isa Integer && !(value isa Bool) || throw(ArgumentError(
        "$key must be an integer",
    ))
    return Int(value)
end

function _qwen3_required_real(object, key::String)
    value = _qwen3_profile_value(object, key)
    value isa Real && !(value isa Bool) || throw(ArgumentError(
        "$key must be numeric",
    ))
    return value
end

function _qwen3_required_bool(object, key::String)
    value = _qwen3_profile_value(object, key)
    value isa Bool || throw(ArgumentError("$key must be a boolean"))
    return Bool(value)
end

"""
    load_qwen3_deployment_profile(path)

Load and validate a version-1 Qwen3 deployment JSON file. Unknown fields and
values that disagree with the frozen dense-family registry are rejected.
"""
function load_qwen3_deployment_profile(path::AbstractString)
    isfile(path) || throw(ArgumentError("deployment profile does not exist: $path"))
    object = _qwen3_json_object(path, "Qwen3 deployment profile")
    unknown = sort!(collect(setdiff(
        Set(String.(collect(keys(object)))),
        _QWEN3_DEPLOYMENT_PROFILE_FIELDS,
    )))
    isempty(unknown) || throw(ArgumentError(
        "unknown Qwen3 deployment profile field(s): $(join(unknown, ", "))",
    ))

    profile = Qwen3DeploymentProfile(
        _qwen3_required_integer(object, "schema_version"),
        _qwen3_required_string(object, "name"),
        _qwen3_required_string(object, "model_id"),
        _qwen3_required_string(object, "revision"),
        Symbol(_qwen3_required_string(object, "variant")),
        Symbol(_qwen3_required_string(object, "weight_dtype")),
        _qwen3_required_integer(object, "context_tokens"),
        _qwen3_required_integer(object, "max_prompt_tokens"),
        _qwen3_required_integer(object, "max_new_tokens"),
        _qwen3_required_integer(object, "prefill_chunk_tokens"),
        _qwen3_required_integer(object, "prefill_reclaim_interval_chunks"),
        _qwen3_required_integer(object, "decode_reclaim_interval_tokens"),
        _qwen3_required_bool(object, "enable_thinking"),
        Symbol(_qwen3_required_string(object, "strategy")),
        Float32(_qwen3_required_real(object, "temperature")),
        _qwen3_required_integer(object, "top_k"),
        Float32(_qwen3_required_real(object, "top_p")),
        _qwen3_required_integer(object, "minimum_gpu_bytes"),
        _qwen3_required_integer(object, "workspace_reserve_bytes"),
        _qwen3_required_string(object, "asset_manifest"),
    )

    profile.schema_version == 1 || throw(ArgumentError(
        "unsupported Qwen3 deployment profile schema_version $(profile.schema_version)",
    ))
    !isempty(profile.name) || throw(ArgumentError("deployment profile name must not be empty"))
    profile.weight_dtype === :bf16 || throw(ArgumentError(
        "the Week 19 deployment runtime currently requires weight_dtype=bf16",
    ))
    profile.context_tokens > 0 || throw(ArgumentError("context_tokens must be positive"))
    profile.max_prompt_tokens > 0 || throw(ArgumentError("max_prompt_tokens must be positive"))
    profile.max_new_tokens > 0 || throw(ArgumentError("max_new_tokens must be positive"))
    profile.prefill_chunk_tokens > 0 || throw(ArgumentError(
        "prefill_chunk_tokens must be positive",
    ))
    profile.prefill_chunk_tokens <= profile.context_tokens || throw(ArgumentError(
        "prefill_chunk_tokens exceeds context_tokens",
    ))
    profile.prefill_reclaim_interval_chunks > 0 || throw(ArgumentError(
        "prefill_reclaim_interval_chunks must be positive",
    ))
    profile.decode_reclaim_interval_tokens > 0 || throw(ArgumentError(
        "decode_reclaim_interval_tokens must be positive",
    ))
    profile.max_prompt_tokens <= profile.context_tokens || throw(ArgumentError(
        "max_prompt_tokens exceeds context_tokens",
    ))
    profile.max_new_tokens <= profile.context_tokens || throw(ArgumentError(
        "max_new_tokens exceeds context_tokens",
    ))
    profile.max_prompt_tokens <= profile.context_tokens - profile.max_new_tokens ||
        throw(ArgumentError(
            "max_prompt_tokens plus max_new_tokens exceeds context_tokens",
        ))
    profile.strategy in (:greedy, :sample) || throw(ArgumentError(
        "strategy must be greedy or sample",
    ))
    isfinite(profile.temperature) && profile.temperature > 0 || throw(ArgumentError(
        "temperature must be finite and positive",
    ))
    profile.top_k > 0 || throw(ArgumentError("top_k must be positive"))
    isfinite(profile.top_p) && 0 < profile.top_p <= 1 || throw(ArgumentError(
        "top_p must be finite and in (0, 1]",
    ))
    profile.minimum_gpu_bytes > 0 || throw(ArgumentError(
        "minimum_gpu_bytes must be positive",
    ))
    profile.workspace_reserve_bytes >= 0 || throw(ArgumentError(
        "workspace_reserve_bytes must be non-negative",
    ))
    !isempty(profile.asset_manifest) || throw(ArgumentError(
        "asset_manifest must not be empty",
    ))

    spec = qwen3_dense_spec(profile.variant)
    profile.model_id == spec.model_id || throw(ArgumentError(
        "profile model_id does not match the frozen $(profile.variant) spec",
    ))
    profile.revision == spec.revision || throw(ArgumentError(
        "profile revision does not match the frozen $(profile.variant) spec",
    ))
    profile.context_tokens <= spec.max_position_embeddings || throw(ArgumentError(
        "profile context exceeds the checkpoint's native position limit",
    ))
    return profile
end

const _QWEN3_ASSET_MANIFEST_FIELDS = Set((
    "schema_version",
    "model_id",
    "revision",
    "files",
))
const _QWEN3_ASSET_FILE_FIELDS = Set(("name", "size", "sha256"))

function _qwen3_sha256_file(path::AbstractString)
    return open(path, "r") do io
        bytes2hex(sha256(io))
    end
end

"""
    verify_qwen3_deployment_assets(model_dir, manifest_path;
                                   model_id=nothing, revision=nothing)

Stream and verify every required local model asset against a frozen manifest
before allocating model memory. Missing/extra manifest fields, unsafe relative
paths, size mismatches, and SHA256 mismatches fail closed.
"""
function verify_qwen3_deployment_assets(
    model_dir::AbstractString,
    manifest_path::AbstractString;
    model_id=nothing,
    revision=nothing,
)
    isdir(model_dir) || throw(ArgumentError("model directory does not exist: $model_dir"))
    isfile(manifest_path) || throw(ArgumentError(
        "asset manifest does not exist: $manifest_path",
    ))
    object = _qwen3_json_object(manifest_path, "Qwen3 asset manifest")
    unknown = sort!(collect(setdiff(
        Set(String.(collect(keys(object)))),
        _QWEN3_ASSET_MANIFEST_FIELDS,
    )))
    isempty(unknown) || throw(ArgumentError(
        "unknown Qwen3 asset manifest field(s): $(join(unknown, ", "))",
    ))
    schema_version = _qwen3_required_integer(object, "schema_version")
    schema_version == 1 || throw(ArgumentError(
        "unsupported Qwen3 asset manifest schema_version $schema_version",
    ))
    manifest_model_id = _qwen3_required_string(object, "model_id")
    manifest_revision = _qwen3_required_string(object, "revision")
    model_id === nothing || String(model_id) == manifest_model_id || throw(ArgumentError(
        "asset manifest model_id does not match the deployment profile",
    ))
    revision === nothing || String(revision) == manifest_revision || throw(ArgumentError(
        "asset manifest revision does not match the deployment profile",
    ))
    files = _qwen3_profile_value(object, "files")
    files isa JSON3.Array || throw(ArgumentError("asset manifest files must be an array"))
    isempty(files) && throw(ArgumentError("asset manifest files must not be empty"))

    seen = Set{String}()
    verified = NamedTuple[]
    for entry in files
        entry isa JSON3.Object || throw(ArgumentError(
            "each asset manifest file entry must be an object",
        ))
        entry_unknown = sort!(collect(setdiff(
            Set(String.(collect(keys(entry)))),
            _QWEN3_ASSET_FILE_FIELDS,
        )))
        isempty(entry_unknown) || throw(ArgumentError(
            "unknown asset file field(s): $(join(entry_unknown, ", "))",
        ))
        name = _qwen3_required_string(entry, "name")
        basename(name) == name && !isempty(name) || throw(ArgumentError(
            "asset file names must be non-empty basenames",
        ))
        name in seen && throw(ArgumentError("duplicate asset manifest file: $name"))
        push!(seen, name)
        expected_size = _qwen3_required_integer(entry, "size")
        expected_size >= 0 || throw(ArgumentError("asset size must be non-negative"))
        expected_sha256 = lowercase(_qwen3_required_string(entry, "sha256"))
        occursin(r"^[0-9a-f]{64}$", expected_sha256) || throw(ArgumentError(
            "asset sha256 must contain exactly 64 lowercase hex digits",
        ))
        path = joinpath(model_dir, name)
        isfile(path) || throw(ArgumentError("required model asset is missing: $path"))
        actual_size = filesize(path)
        actual_size == expected_size || throw(ArgumentError(
            "model asset size mismatch for $name: expected $expected_size, got $actual_size",
        ))
        actual_sha256 = _qwen3_sha256_file(path)
        actual_sha256 == expected_sha256 || throw(ArgumentError(
            "model asset SHA256 mismatch for $name",
        ))
        push!(verified, (;
            name,
            size=actual_size,
            sha256=actual_sha256,
        ))
    end

    # `load_safetensors(model_dir)` gives the single-file checkpoint precedence
    # over the sharded index. Reject any unverified loader candidate so that an
    # old or injected file cannot replace the exact checkpoint just validated.
    for loader_candidate in ("model.safetensors", "model.safetensors.index.json")
        candidate_path = joinpath(model_dir, loader_candidate)
        if isfile(candidate_path) && !(loader_candidate in seen)
            throw(ArgumentError(
                "unverified HuggingFace loader candidate is present: $candidate_path",
            ))
        end
    end

    return (;
        model_id=manifest_model_id,
        revision=manifest_revision,
        files=Tuple(verified),
        total_bytes=sum(file.size for file in verified),
    )
end

"""
    qwen3_kv_cache_bytes(model_or_spec, context_tokens;
                         batch_size=1, dtype_bytes=2)

Exact logical K/V tensor bytes for a dense Qwen3 cache:
`layers * K/V * head_dim * kv_heads * tokens * batch * dtype_bytes`.
Allocator metadata and attention/GEMM workspaces are deliberately excluded.
"""
function qwen3_kv_cache_bytes(
    model_or_spec,
    context_tokens::Integer;
    batch_size::Integer=1,
    dtype_bytes::Integer=2,
)
    context_tokens >= 0 || throw(ArgumentError("context_tokens must be non-negative"))
    batch_size > 0 || throw(ArgumentError("batch_size must be positive"))
    dtype_bytes > 0 || throw(ArgumentError("dtype_bytes must be positive"))
    factors = try
        (
            Int(model_or_spec.num_layers),
            2,
            Int(model_or_spec.head_dim),
            Int(model_or_spec.num_kv_heads),
            Int(context_tokens),
            Int(batch_size),
            Int(dtype_bytes),
        )
    catch err
        err isa InexactError || rethrow()
        throw(ArgumentError("KV cache dimensions do not fit in Int"))
    end
    bytes = 1
    try
        for factor in factors
            bytes = Base.checked_mul(bytes, factor)
        end
    catch err
        err isa OverflowError || rethrow()
        throw(ArgumentError("KV cache byte count overflows Int"))
    end
    return bytes
end

"""
    HFQwen3BF16Session

Reusable batch-1 runtime for native-BF16 or weight-only-quantized Qwen3
parameters consumed by the LifeAI BF16 accelerator kernels. RoPE tables and a
fixed-capacity KV cache are allocated once. Each request resets only the
logical cache position.
"""
mutable struct HFQwen3BF16Session{M,P,T,G,C,S}
    model::M
    parameters::P
    tokenizer::T
    generation_config::G
    cos_table::C
    sin_table::S
    caches::Vector{Any}
    position::Int
    context_tokens::Int
    prefill_chunk_tokens::Int
end

function _qwen3_session_cache(reference, model::GPTModel, context_tokens::Int)
    shape = (
        model.head_dim,
        model.num_kv_heads,
        context_tokens,
        1,
    )
    return Any[
        BF16AStaticLayerCache(
            similar(reference, BFloat16, shape),
            similar(reference, BFloat16, shape),
        )
        for _ in 1:model.num_layers
    ]
end

"""
    init_hf_qwen3_bf16_session(bundle; context_tokens, prefill_chunk_tokens=128)

Create a reusable runtime from a loaded bundle whose parameter tree already
lives on the desired device. The static cache is allocated eagerly so an
insufficient-memory failure happens before the first request.
"""
function init_hf_qwen3_bf16_session(
    bundle;
    context_tokens::Integer=bundle.model.max_seq_len,
    prefill_chunk_tokens::Integer=128,
)
    model = bundle.model
    parameters = bundle.parameters
    tokenizer = bundle.tokenizer
    _qwen3_validate_semantics(model)
    eltype(parameters.token_embedding.weight) === BFloat16 || throw(ArgumentError(
        "Qwen3 BF16 sessions require a BFloat16 embedding/compute tree",
    ))
    context = Int(context_tokens)
    chunk = Int(prefill_chunk_tokens)
    0 < context <= model.max_seq_len || throw(ArgumentError(
        "context_tokens must be in 1:model.max_seq_len",
    ))
    0 < chunk <= context || throw(ArgumentError(
        "prefill_chunk_tokens must be in 1:context_tokens",
    ))
    vocab_size(tokenizer) <= model.vocab_size || throw(ArgumentError(
        "tokenizer vocabulary exceeds the model vocabulary",
    ))

    rope = first(values(model.blocks.layers)).attn.rope
    to_device = _bf16a_device_mover(parameters.token_embedding.weight)
    cos_table = to_device(BFloat16.(rope.cos_cache[:, 1:context]))
    sin_table = to_device(BFloat16.(rope.sin_cache[:, 1:context]))
    caches = _qwen3_session_cache(
        parameters.token_embedding.weight,
        model,
        context,
    )
    generation_config = hasproperty(bundle, :generation_config) ?
        bundle.generation_config : hf_generation_config(tokenizer)
    return HFQwen3BF16Session(
        model,
        parameters,
        tokenizer,
        generation_config,
        cos_table,
        sin_table,
        caches,
        0,
        context,
        chunk,
    )
end

"""
    load_hf_qwen3_bf16_session(model_dir; to_device=identity, ...)

Load a local checkpoint and construct a reusable BF16 session. This API does
not establish checkpoint provenance; callers that require frozen assets must
run `verify_qwen3_deployment_assets` first.
`to_device=CUDA.cu` selects CUDA without making CUDA a hard dependency of this
module's public API.
"""
function load_hf_qwen3_bf16_session(
    model_dir::AbstractString;
    context_tokens::Integer=4096,
    prefill_chunk_tokens::Integer=128,
    revision::AbstractString="",
    variant=nothing,
    to_device=identity,
)
    loaded = load_hf_qwen3_bundle(
        model_dir;
        max_seq_len=Int(context_tokens),
        weight_dtype=BFloat16,
        revision,
        variant,
    )
    parameters = to_device(loaded.parameters)
    device_bundle = merge(loaded, (; parameters))
    return init_hf_qwen3_bf16_session(
        device_bundle;
        context_tokens,
        prefill_chunk_tokens,
    )
end

"""
    reset_hf_qwen3_bf16_session!(session)

Forget the logical request prefix. Static buffers are intentionally retained
and overwritten on the next request.
"""
function reset_hf_qwen3_bf16_session!(session::HFQwen3BF16Session)
    session.position = 0
    return session
end

function _qwen3_session_token_vector(session, tokens)
    values = Int.(vec(collect(tokens)))
    isempty(values) && throw(ArgumentError("prompt must contain at least one token"))
    _validate_generation_ids(values, session.model.vocab_size)
    return values
end

"""
    prefill_hf_qwen3_bf16!(session, prompt_tokens)

Reset and prefill a request in bounded chunks. Only the final prompt position
is projected to vocabulary logits and no parity trace is retained.
"""
function prefill_hf_qwen3_bf16!(
    session::HFQwen3BF16Session,
    prompt_tokens,
    ;
    on_chunk=nothing,
)
    tokens = _qwen3_session_token_vector(session, prompt_tokens)
    length(tokens) <= session.context_tokens || throw(ArgumentError(
        "prompt exceeds the session context_tokens limit",
    ))
    reset_hf_qwen3_bf16_session!(session)
    logits = nothing
    for first_index in 1:session.prefill_chunk_tokens:length(tokens)
        last_index = min(
            first_index + session.prefill_chunk_tokens - 1,
            length(tokens),
        )
        chunk = reshape(tokens[first_index:last_index], :, 1)
        chunk_tokens = size(chunk, 1)
        key_tokens = session.position + chunk_tokens
        mask = _bf16a_device_mover(session.parameters.token_embedding.weight)(
            _bf16a_causal_mask(chunk_tokens, key_tokens),
        )
        result = _bf16a_forward_pass(
            session.model,
            session.parameters,
            chunk,
            session.caches,
            session.cos_table,
            session.sin_table,
            mask;
            start_pos=session.position + 1,
            project_last_token_only=true,
            project_logits=last_index == length(tokens),
            capture_trace=false,
        )
        result.logits === nothing || (logits = result.logits)
        session.position = key_tokens
        on_chunk === nothing || on_chunk(session.position)
    end
    return logits
end

"""
    decode_hf_qwen3_bf16!(session, token)

Append one LifeAI 1-based token id to the preallocated cache and return logits
for that position.
"""
function decode_hf_qwen3_bf16!(
    session::HFQwen3BF16Session,
    token::Integer,
)
    session.position > 0 || throw(ArgumentError(
        "call prefill_hf_qwen3_bf16! before decode",
    ))
    session.position < session.context_tokens || throw(ArgumentError(
        "Qwen3 session context is full",
    ))
    token_id = Int(token)
    _validate_generation_ids([token_id], session.model.vocab_size)
    result = _bf16a_forward_pass(
        session.model,
        session.parameters,
        reshape([token_id], 1, 1),
        session.caches,
        session.cos_table,
        session.sin_table,
        nothing;
        start_pos=session.position + 1,
        project_last_token_only=true,
        capture_trace=false,
    )
    session.position += 1
    return result.logits
end

function _qwen3_session_strategy(
    session::HFQwen3BF16Session,
    strategy::Symbol,
)
    strategy in (:greedy, :sample, :config) || throw(ArgumentError(
        "strategy must be :greedy, :sample, or :config",
    ))
    return strategy === :config ?
        (session.generation_config.do_sample ? :sample : :greedy) : strategy
end

function _qwen3_session_choice(
    logits,
    host,
    strategy::Symbol,
    rng::AbstractRNG;
    temperature,
    top_k,
    top_p,
    sample_uniform=nothing,
)
    return strategy === :greedy ?
        _hf_greedy_choice(logits, host, false) :
        _hf_sample_choice(
            logits,
            host,
            rng;
            temperature,
            top_k,
            top_p,
            sample_uniform,
        )
end

"""
    generate_hf_qwen3_bf16!(session, prompt_tokens; kwargs...)

Generate token ids from a reusable accelerator session with EOS stopping,
greedy or official temperature/top-k/top-p sampling, and per-phase timings.
"""
function generate_hf_qwen3_bf16!(
    session::HFQwen3BF16Session,
    prompt_tokens;
    max_new_tokens::Integer=512,
    strategy::Symbol=:config,
    temperature=nothing,
    top_k=nothing,
    top_p=nothing,
    rng::AbstractRNG=default_rng(),
    stop_token_ids=nothing,
    on_token=nothing,
    on_prefill_chunk=nothing,
)
    prompt_ids = _qwen3_session_token_vector(session, prompt_tokens)
    requested = Int(max_new_tokens)
    requested >= 0 || throw(ArgumentError("max_new_tokens must be non-negative"))
    requested <= session.context_tokens || throw(ArgumentError(
        "requested output exceeds session context_tokens",
    ))
    length(prompt_ids) <= session.context_tokens - requested || throw(ArgumentError(
        "prompt plus requested output exceeds session context_tokens",
    ))
    resolved_strategy = _qwen3_session_strategy(session, strategy)
    resolved_temperature = temperature === nothing ?
        session.generation_config.temperature : temperature
    resolved_top_k = top_k === nothing ? session.generation_config.top_k : top_k
    resolved_top_p = top_p === nothing ? session.generation_config.top_p : top_p
    if resolved_strategy === :sample
        resolved_temperature isa Real && isfinite(resolved_temperature) &&
            resolved_temperature > 0 || throw(ArgumentError(
                "temperature must be finite and positive",
            ))
        resolved_top_k isa Integer && resolved_top_k > 0 || throw(ArgumentError(
            "top_k must be positive",
        ))
        resolved_top_p isa Real && isfinite(resolved_top_p) &&
            0 < resolved_top_p <= 1 || throw(ArgumentError(
                "top_p must be finite and in (0, 1]",
            ))
    end
    stops = stop_token_ids === nothing ?
        Set(session.tokenizer.eos_ids) : Set(Int.(stop_token_ids))
    all(id -> 1 <= id <= session.model.vocab_size, stops) || throw(ArgumentError(
        "stop token id is outside the model vocabulary",
    ))

    if requested == 0
        reset_hf_qwen3_bf16_session!(session)
        return (;
            prompt_ids,
            generated_ids=Int[],
            token_ids=copy(prompt_ids),
            completion="",
            stop_reason=:length,
            strategy=resolved_strategy,
            prefill_seconds=0.0,
            decode_seconds=0.0,
            tokens_per_second=0.0,
            trace=(),
        )
    end

    prefill_started = time_ns()
    logits = prefill_hf_qwen3_bf16!(
        session,
        prompt_ids;
        on_chunk=on_prefill_chunk,
    )
    host = cpu_device()
    generated_ids = Int[]
    trace = NamedTuple[]
    stop_reason = :length
    token_id, token_trace = _qwen3_session_choice(
        logits,
        host,
        resolved_strategy,
        rng;
        temperature=resolved_temperature,
        top_k=resolved_top_k,
        top_p=resolved_top_p,
    )
    prefill_seconds = (time_ns() - prefill_started) / 1.0e9
    push!(generated_ids, token_id)
    push!(trace, merge((; step=1), token_trace))
    on_token === nothing || on_token(token_id)
    if token_id in stops
        stop_reason = :eos
    end

    decode_started = time_ns()
    if stop_reason !== :eos
        for step in 2:requested
            logits = decode_hf_qwen3_bf16!(session, token_id)
            token_id, token_trace = _qwen3_session_choice(
                logits,
                host,
                resolved_strategy,
                rng;
                temperature=resolved_temperature,
                top_k=resolved_top_k,
                top_p=resolved_top_p,
            )
            push!(generated_ids, token_id)
            push!(trace, merge((; step), token_trace))
            on_token === nothing || on_token(token_id)
            if token_id in stops
                stop_reason = :eos
                break
            end
        end
    end
    decode_seconds = (time_ns() - decode_started) / 1.0e9
    decoded_tokens = max(0, length(generated_ids) - 1)
    return (;
        prompt_ids,
        generated_ids,
        token_ids=vcat(prompt_ids, generated_ids),
        completion=decode(
            session.tokenizer,
            generated_ids;
            errors=:replace,
            skip_special_tokens=true,
        ),
        stop_reason,
        strategy=resolved_strategy,
        prefill_seconds,
        decode_seconds,
        tokens_per_second=decoded_tokens == 0 || decode_seconds == 0 ?
            0.0 : decoded_tokens / decode_seconds,
        trace=Tuple(trace),
    )
end

function _qwen3_render_chat(session::HFQwen3BF16Session, messages, enable_thinking::Bool)
    prompt = apply_qwen3_chat_template(
        session.tokenizer,
        messages;
        add_generation_prompt=true,
        enable_thinking,
    )
    ids = encode(session.tokenizer, prompt; add_special_tokens=false)
    return prompt, ids
end

"""
    fit_qwen3_chat_context(session, messages; max_prompt_tokens,
                           enable_thinking=false)

Render chat history and, when necessary, drop the oldest non-system turn pairs
until it fits. The leading system message and newest message are never dropped.
"""
function fit_qwen3_chat_context(
    session::HFQwen3BF16Session,
    messages;
    max_prompt_tokens::Integer,
    enable_thinking::Bool=false,
)
    limit = Int(max_prompt_tokens)
    0 < limit <= session.context_tokens || throw(ArgumentError(
        "max_prompt_tokens must be in 1:session.context_tokens",
    ))
    retained = collect(messages)
    isempty(retained) && throw(ArgumentError("chat messages must not be empty"))
    dropped = 0
    while true
        prompt, ids = _qwen3_render_chat(session, retained, enable_thinking)
        length(ids) <= limit && return (;
            messages=retained,
            prompt,
            prompt_ids=ids,
            dropped_messages=dropped,
        )
        protected_prefix = String(_hf_message_value(
            first(retained),
            :role;
            required=true,
        )) == "system" ? 1 : 0
        first_droppable = protected_prefix + 1
        first_droppable < length(retained) || throw(ArgumentError(
            "newest chat message cannot fit within max_prompt_tokens",
        ))
        remove_count = 1
        role = String(_hf_message_value(
            retained[first_droppable],
            :role;
            required=true,
        ))
        if role == "user" && first_droppable + 1 < length(retained)
            next_role = String(_hf_message_value(
                retained[first_droppable + 1],
                :role;
                required=true,
            ))
            next_role == "assistant" && (remove_count = 2)
        end
        deleteat!(retained, first_droppable:(first_droppable + remove_count - 1))
        dropped += remove_count
    end
end

"""
    generate_hf_text!(session, input; chat=true, ...)

Text/chat convenience wrapper around `generate_hf_qwen3_bf16!`. A string in
chat mode is treated as one user message; a message collection can carry
system and retained user/assistant turns.
"""
function generate_hf_text!(
    session::HFQwen3BF16Session,
    input;
    chat::Bool=true,
    enable_thinking::Bool=false,
    max_new_tokens::Integer=512,
    max_prompt_tokens::Integer=session.context_tokens - Int(max_new_tokens),
    kwargs...,
)
    if chat
        messages = input isa AbstractString ?
            [(role="user", content=String(input))] : input
        fitted = fit_qwen3_chat_context(
            session,
            messages;
            max_prompt_tokens,
            enable_thinking,
        )
        generated = generate_hf_qwen3_bf16!(
            session,
            fitted.prompt_ids;
            max_new_tokens,
            kwargs...,
        )
        return merge(generated, (;
            prompt=fitted.prompt,
            messages=fitted.messages,
            dropped_messages=fitted.dropped_messages,
        ))
    end
    input isa AbstractString || throw(ArgumentError(
        "raw text generation requires a string input",
    ))
    prompt = String(input)
    prompt_ids = encode(session.tokenizer, prompt; add_special_tokens=false)
    length(prompt_ids) <= max_prompt_tokens || throw(ArgumentError(
        "raw prompt exceeds max_prompt_tokens",
    ))
    generated = generate_hf_qwen3_bf16!(
        session,
        prompt_ids;
        max_new_tokens,
        kwargs...,
    )
    return merge(generated, (; prompt, dropped_messages=0))
end
