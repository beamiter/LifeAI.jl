using JSON3
using SHA: sha256

const _CH47_LONG_REFERENCE_SCHEMA_VERSION = 1
const _CH47_LONG_REFERENCE_ORACLE =
    "qwen3_vl_2b_256_describe_hf_dynamic_cache_long_greedy"
const _CH47_LONG_REFERENCE_CLAIM =
    "float32_cpu_greedy_token_timeline_only"
const _CH47_LONG_REFERENCE_TERMINATION = "fixed_length_ignore_eos"
const _CH47_LONG_REFERENCE_TOKEN_COUNT = 256
const _CH47_LONG_REFERENCE_CHECKPOINT_LENGTHS = (32, 128, 256)
const _CH47_LONG_REFERENCE_ENVIRONMENT = Dict(
    "python" => "3.10.12",
    "numpy" => "1.26.4",
    "pillow" => "11.3.0",
    "safetensors" => "0.5.3",
    "tokenizers" => "0.22.1",
    "jinja2" => "3.1.6",
    "transformers" => "4.57.0",
    "torch" => "2.7.1+cpu",
    "torchvision" => "0.22.1+cu126",
    "torch_git_revision" =>
        "e2d141dbde55c2a4370fac5165b0561b6af4798b",
    "torch_config_sha256" =>
        "dcb7f1e248794c5c144992643f929d0dc623511210f6fe5615f78cb009d64745",
    "cpu_capability" => "AVX2",
)

function _ch47_long_require(condition::Bool, message::AbstractString)
    condition || throw(ArgumentError(message))
    return nothing
end

function _ch47_long_file_sha256(path::AbstractString)
    return open(path, "r") do io
        bytes2hex(sha256(io))
    end
end

function _ch47_long_token_timeline_sha256(token_ids_0_based)
    io = IOBuffer()
    for raw_id in token_ids_0_based
        token_id = Int(raw_id)
        _ch47_long_require(
            0 <= token_id <= typemax(UInt32),
            "long-reference token id is outside UInt32",
        )
        write(io, htol(UInt32(token_id)))
    end
    return bytes2hex(sha256(take!(io)))
end

function _ch47_long_text_sha256(value::AbstractString)
    return bytes2hex(sha256(codeunits(value)))
end

function _ch47_long_string_dict(value, label::AbstractString)
    result = Dict{String,String}()
    for (key, item) in pairs(value)
        name = String(key)
        haskey(result, name) && throw(ArgumentError(
            "$label contains duplicate key $name",
        ))
        result[name] = String(item)
    end
    return result
end

function _ch47_long_int_vector(value, label::AbstractString)
    result = Int[]
    for item in value
        item isa Integer || throw(ArgumentError(
            "$label must contain only integers",
        ))
        push!(result, Int(item))
    end
    return result
end

function _ch47_long_sha256(value, label::AbstractString)
    digest = lowercase(String(value))
    occursin(r"^[0-9a-f]{64}$", digest) || throw(ArgumentError(
        "$label must be a lowercase SHA-256 digest",
    ))
    return digest
end

function _ch47_long_bool(value, label::AbstractString)
    value isa Bool || throw(ArgumentError("$label must be a JSON boolean"))
    return value
end

function _ch47_validate_long_decisions(
    decisions,
    generated_ids_0_based::Vector{Int},
    prompt_tokens::Int,
    rope_delta::Int,
    vocab_size::Int,
)
    length(decisions) == length(generated_ids_0_based) || throw(ArgumentError(
        "long-reference decision count differs from token count",
    ))
    for (step, decision) in enumerate(decisions)
        Int(decision["generated_step"]) == step || throw(ArgumentError(
            "long-reference generated_step is not contiguous at step $step",
        ))
        expected_name = step == 1 ? "prefill" : "decode.$(step - 2)"
        String(decision["phase"]) == expected_name || throw(ArgumentError(
            "long-reference phase changed at step $step",
        ))
        expected_cache_length = prompt_tokens + step - 1
        Int(decision["cache_length"]) == expected_cache_length ||
            throw(ArgumentError(
                "long-reference cache length changed at step $step",
            ))
        mask_shape = _ch47_long_int_vector(
            decision["attention_mask_shape"],
            "long-reference attention mask shape",
        )
        mask_shape == [1, expected_cache_length] || throw(ArgumentError(
            "long-reference attention mask shape changed at step $step",
        ))
        top_two = decision["top_two"]
        Int(top_two["top1_token_id_0_based"]) ==
            generated_ids_0_based[step] || throw(ArgumentError(
                "long-reference top-1 token changed at step $step",
            ))
        top2_id = Int(top_two["top2_token_id_0_based"])
        0 <= top2_id < vocab_size || throw(ArgumentError(
            "long-reference top-2 token is outside the vocabulary at step $step",
        ))
        top2_id != generated_ids_0_based[step] || throw(ArgumentError(
            "long-reference top-1 and top-2 tokens coincide at step $step",
        ))
        top1 = Float64(top_two["top1_logit_f32"])
        top2 = Float64(top_two["top2_logit_f32"])
        margin = Float64(top_two["margin_f32"])
        all(isfinite, (top1, top2, margin)) || throw(ArgumentError(
            "long-reference top-2 metrics are non-finite at step $step",
        ))
        top1 > top2 && margin > 0 || throw(ArgumentError(
            "long-reference top-2 margin is not strictly positive at step $step",
        ))
        top1_f32 = Float32(top1)
        top2_f32 = Float32(top2)
        margin_f32 = Float32(margin)
        Float64(top1_f32) == top1 && Float64(top2_f32) == top2 &&
            Float64(margin_f32) == margin || throw(ArgumentError(
                "long-reference top-2 metrics are not exact Float32 values at step $step",
            ))
        margin_f32 == top1_f32 - top2_f32 || throw(ArgumentError(
            "long-reference top-2 margin is inconsistent at step $step",
        ))
        logits_shape = _ch47_long_int_vector(
            decision["logits_shape"],
            "long-reference logits shape",
        )
        logits_shape == [1, 1, vocab_size] || throw(ArgumentError(
            "long-reference logits shape changed at step $step",
        ))
        String(decision["logits_dtype"]) == "torch.float32" ||
            throw(ArgumentError(
                "long-reference logits dtype changed at step $step",
            ))
        _ch47_long_sha256(
            decision["logits_raw_sha256"],
            "long-reference logits hash at step $step",
        )

        if step == 1
            decision["input_token_id_0_based"] === nothing ||
                throw(ArgumentError(
                    "long-reference prefill unexpectedly consumes a decode token",
                ))
            decision["physical_cache_position_0_based"] === nothing ||
                throw(ArgumentError(
                    "long-reference prefill unexpectedly has a cache write position",
                ))
            isempty(decision["mrope_position_ids_thw_0_based"]) ||
                throw(ArgumentError(
                    "long-reference prefill unexpectedly has one-token mRoPE ids",
                ))
        else
            physical_position = prompt_tokens + step - 2
            Int(decision["input_token_id_0_based"]) ==
                generated_ids_0_based[step - 1] || throw(ArgumentError(
                    "long-reference consumed token changed at step $step",
                ))
            Int(decision["physical_cache_position_0_based"]) ==
                physical_position || throw(ArgumentError(
                    "long-reference cache position changed at step $step",
                ))
            positions = _ch47_long_int_vector(
                decision["mrope_position_ids_thw_0_based"],
                "long-reference mRoPE positions",
            )
            positions == fill(physical_position + rope_delta, 3) ||
                throw(ArgumentError(
                    "long-reference mRoPE coordinate changed at step $step",
                ))
        end
    end
    return nothing
end

function _ch47_validate_long_cache_checkpoints(
    raw_checkpoints,
    prompt_tokens::Int;
    num_hidden_layers::Int,
    num_key_value_heads::Int,
    head_dim::Int,
)
    checkpoints = Dict{String,Any}(
        String(key) => value for (key, value) in pairs(raw_checkpoints)
    )
    Set(keys(checkpoints)) ==
        Set(string.(_CH47_LONG_REFERENCE_CHECKPOINT_LENGTHS)) ||
        throw(ArgumentError(
            "long-reference cache checkpoint lengths changed",
        ))
    for generated_tokens in _CH47_LONG_REFERENCE_CHECKPOINT_LENGTHS
        geometry = checkpoints[string(generated_tokens)]
        cache_length = prompt_tokens + generated_tokens - 1
        Int(geometry["length"]) == cache_length || throw(ArgumentError(
            "long-reference cache checkpoint length changed at $generated_tokens tokens",
        ))
        Int(geometry["layer_count"]) == num_hidden_layers ||
            throw(ArgumentError(
                "long-reference cache layer count changed at $generated_tokens tokens",
            ))
        expected_shape = [1, num_key_value_heads, cache_length, head_dim]
        for kind in ("key_shapes_hf", "value_shapes_hf")
            shapes = collect(geometry[kind])
            length(shapes) == num_hidden_layers || throw(ArgumentError(
                "long-reference $kind layer count changed at $generated_tokens tokens",
            ))
            for (layer, raw_shape) in enumerate(shapes)
                shape = _ch47_long_int_vector(
                    raw_shape,
                    "long-reference $kind shape",
                )
                shape == expected_shape || throw(ArgumentError(
                    "long-reference $kind changed at $generated_tokens tokens, layer $layer",
                ))
            end
        end
    end
    return nothing
end

function _ch47_load_long_generation_reference(
    path::AbstractString,
    expected_file_sha256::AbstractString;
    model_id::AbstractString,
    modelscope_revision::AbstractString,
    huggingface_revision::AbstractString,
    asset_sha256,
    required_lengths,
    expected_dtype::AbstractString="float32",
    expected_device_type::AbstractString="cpu",
    expected_device::AbstractString="cpu",
    expected_prompt_tokens::Int=76,
    expected_rope_delta::Int=-56,
    expected_vocab_size::Int=151_936,
    expected_num_hidden_layers::Int=28,
    expected_num_key_value_heads::Int=8,
    expected_head_dim::Int=128,
    expected_prompt::AbstractString="Describe.",
    expected_image_sha256=nothing,
    expected_rendered_prompt_sha256=nothing,
    expected_chapter45_reference_sha256=nothing,
    expected_chapter45_metadata_sha256=nothing,
    expected_source_sha256=nothing,
)
    resolved_path = abspath(path)
    isfile(resolved_path) || throw(ArgumentError(
        "long-generation reference does not exist: $resolved_path",
    ))
    expected_digest = _ch47_long_sha256(
        expected_file_sha256,
        "expected long-reference file hash",
    )
    reference_bytes = read(resolved_path)
    actual_digest = bytes2hex(sha256(reference_bytes))
    actual_digest == expected_digest || throw(ArgumentError(
        "long-generation reference SHA-256 is not the explicitly pinned value",
    ))
    metadata = JSON3.read(String(reference_bytes))
    Int(metadata["schema_version"]) == _CH47_LONG_REFERENCE_SCHEMA_VERSION ||
        throw(ArgumentError("long-reference schema version changed"))
    String(metadata["oracle"]) == _CH47_LONG_REFERENCE_ORACLE ||
        throw(ArgumentError("long-reference oracle identity changed"))
    String(metadata["claim"]) == _CH47_LONG_REFERENCE_CLAIM ||
        throw(ArgumentError("long-reference claim changed"))
    String(metadata["model_id"]) == model_id ||
        throw(ArgumentError("long-reference model id changed"))
    String(metadata["modelscope_revision"]) == modelscope_revision ||
        throw(ArgumentError("long-reference ModelScope revision changed"))
    String(metadata["huggingface_revision"]) == huggingface_revision ||
        throw(ArgumentError("long-reference Hugging Face revision changed"))
    String(metadata["compute_dtype"]) == expected_dtype ||
        throw(ArgumentError("long-reference compute dtype changed"))
    String(metadata["compute_device_type"]) == expected_device_type ||
        throw(ArgumentError("long-reference device type changed"))
    String(metadata["compute_device"]) == expected_device ||
        throw(ArgumentError("long-reference compute device changed"))
    String(metadata["attention_implementation"]) == "eager" ||
        throw(ArgumentError("long-reference attention implementation changed"))
    String(metadata["attention_mask_contract"]) ==
        "explicit_all_ones_every_call" || throw(ArgumentError(
            "long-reference attention-mask contract changed",
        ))
    String(metadata["cache_contract"]) ==
        "streaming_hf_dynamic_cache_geometry_no_kv_snapshots" ||
        throw(ArgumentError("long-reference cache contract changed"))
    for (name, expected) in _CH47_LONG_REFERENCE_ENVIRONMENT
        String(metadata[name]) == expected || throw(ArgumentError(
            "long-reference frozen environment changed: $name",
        ))
    end
    Int(metadata["torch_num_threads"]) == 24 || throw(ArgumentError(
        "long-reference Torch thread count changed",
    ))
    Int(metadata["torch_num_interop_threads"]) == 24 || throw(ArgumentError(
        "long-reference Torch inter-op thread count changed",
    ))
    Int(metadata["torch_seed"]) == 0 || throw(ArgumentError(
        "long-reference Torch seed changed",
    ))
    _ch47_long_bool(
        metadata["deterministic_algorithms"],
        "long-reference deterministic_algorithms",
    ) || throw(ArgumentError(
        "long-reference deterministic algorithms are disabled",
    ))
    !_ch47_long_bool(
        metadata["deterministic_algorithms_warn_only"],
        "long-reference deterministic_algorithms_warn_only",
    ) ||
        throw(ArgumentError(
            "long-reference deterministic algorithms are warn-only",
        ))
    String(metadata["float32_matmul_precision"]) == "highest" ||
        throw(ArgumentError(
            "long-reference Float32 matmul precision changed",
        ))
    _ch47_long_bool(
        metadata["mkldnn_available"],
        "long-reference mkldnn_available",
    ) && _ch47_long_bool(
        metadata["mkldnn_enabled"],
        "long-reference mkldnn_enabled",
    ) ||
        throw(ArgumentError("long-reference MKLDNN backend changed"))
    !_ch47_long_bool(
        metadata["mkldnn_deterministic"],
        "long-reference mkldnn_deterministic",
    ) || throw(ArgumentError(
        "long-reference MKLDNN deterministic setting changed",
    ))
    _ch47_long_bool(
        metadata["mkl_available"],
        "long-reference mkl_available",
    ) && _ch47_long_bool(
        metadata["openmp_available"],
        "long-reference openmp_available",
    ) ||
        throw(ArgumentError("long-reference CPU math backend changed"))
    String(metadata["cuda_device"]) == "" || throw(ArgumentError(
        "long-reference unexpectedly records a CUDA device",
    ))
    _ch47_long_bool(metadata["greedy"], "long-reference greedy") ||
        throw(ArgumentError(
        "long-reference generation is not greedy",
        ))
    String(metadata["termination_contract"]) ==
        _CH47_LONG_REFERENCE_TERMINATION || throw(ArgumentError(
            "long-reference termination contract changed",
        ))
    isempty(metadata["stop_token_ids_0_based"]) || throw(ArgumentError(
        "long-reference stop-token contract changed",
    ))
    Int(metadata["prompt_tokens"]) == expected_prompt_tokens ||
        throw(ArgumentError("long-reference prompt length changed"))
    Int(metadata["rope_delta"]) == expected_rope_delta ||
        throw(ArgumentError("long-reference rope delta changed"))
    String(metadata["prompt"]) == expected_prompt || throw(ArgumentError(
        "long-reference prompt changed",
    ))
    rendered_prompt = String(metadata["rendered_prompt"])
    image_sha256 = _ch47_long_sha256(
        metadata["image_sha256"],
        "long-reference image hash",
    )
    if expected_image_sha256 !== nothing
        image_sha256 == lowercase(String(expected_image_sha256)) ||
            throw(ArgumentError("long-reference image hash changed"))
    end
    rendered_prompt_sha256 = _ch47_long_sha256(
        metadata["rendered_prompt_sha256"],
        "long-reference rendered prompt hash",
    )
    _ch47_long_text_sha256(rendered_prompt) == rendered_prompt_sha256 ||
        throw(ArgumentError(
            "long-reference rendered prompt does not match its hash",
        ))
    if expected_rendered_prompt_sha256 !== nothing
        rendered_prompt_sha256 ==
            lowercase(String(expected_rendered_prompt_sha256)) ||
            throw(ArgumentError("long-reference rendered prompt hash changed"))
    end
    _ch47_long_int_vector(
        metadata["image_shape_hwc"],
        "long-reference image shape",
    ) == [256, 256, 3] || throw(ArgumentError(
        "long-reference image geometry changed",
    ))
    grid = collect(metadata["grid_thw"])
    length(grid) == 1 && _ch47_long_int_vector(
        only(grid),
        "long-reference image grid",
    ) == [1, 16, 16] || throw(ArgumentError(
        "long-reference image grid changed",
    ))
    Int(metadata["image_token_count"]) == 64 || throw(ArgumentError(
        "long-reference image token count changed",
    ))
    actual_assets = _ch47_long_string_dict(
        metadata["asset_sha256"],
        "long-reference asset hashes",
    )
    expected_assets = Dict(
        String(key) => String(value) for (key, value) in pairs(asset_sha256)
    )
    for (name, digest) in actual_assets
        _ch47_long_sha256(digest, "long-reference checkpoint hash $name")
    end
    actual_assets == expected_assets || throw(ArgumentError(
        "long-reference checkpoint asset hashes changed",
    ))
    actual_sources = _ch47_long_string_dict(
        metadata["source_sha256"],
        "long-reference source hashes",
    )
    for (name, digest) in actual_sources
        _ch47_long_sha256(digest, "long-reference source hash $name")
    end
    if expected_source_sha256 !== nothing
        expected_sources = Dict(
            String(key) => lowercase(String(value))
            for (key, value) in pairs(expected_source_sha256)
        )
        actual_sources == expected_sources || throw(ArgumentError(
            "long-reference exporter source hashes changed",
        ))
    end
    chapter45_hashes = _ch47_long_string_dict(
        metadata["chapter45_reference_sha256"],
        "long-reference Chapter 45 hashes",
    )
    if expected_chapter45_reference_sha256 !== nothing
        get(chapter45_hashes, "reference.safetensors", "") ==
            lowercase(String(expected_chapter45_reference_sha256)) ||
            throw(ArgumentError(
                "long-reference Chapter 45 tensor hash changed",
            ))
    end
    if expected_chapter45_metadata_sha256 !== nothing
        get(chapter45_hashes, "reference.json", "") ==
            lowercase(String(expected_chapter45_metadata_sha256)) ||
            throw(ArgumentError(
                "long-reference Chapter 45 metadata hash changed",
            ))
    end

    input_ids_0_based = _ch47_long_int_vector(
        metadata["input_ids_0_based"],
        "long-reference prompt token ids",
    )
    length(input_ids_0_based) == expected_prompt_tokens || throw(ArgumentError(
        "long-reference prompt token count changed",
    ))
    all(id -> 0 <= id < expected_vocab_size, input_ids_0_based) ||
        throw(ArgumentError("long-reference prompt token is outside the vocabulary"))
    input_ids_digest = _ch47_long_token_timeline_sha256(input_ids_0_based)
    input_ids_digest == _ch47_long_sha256(
        metadata["input_ids_u32le_sha256"],
        "long-reference prompt token hash",
    ) || throw(ArgumentError("long-reference prompt token hash changed"))

    generated_ids_0_based = _ch47_long_int_vector(
        metadata["generated_token_ids_0_based"],
        "long-reference generated token ids",
    )
    isempty(generated_ids_0_based) && throw(ArgumentError(
        "long-reference token timeline is empty",
    ))
    all(id -> 0 <= id < expected_vocab_size, generated_ids_0_based) ||
        throw(ArgumentError("long-reference token id is outside the vocabulary"))
    Int(metadata["generated_token_count"]) == length(generated_ids_0_based) ||
        throw(ArgumentError("long-reference generated token count changed"))
    length(generated_ids_0_based) == _CH47_LONG_REFERENCE_TOKEN_COUNT ||
        throw(ArgumentError("long-reference must contain exactly 256 tokens"))
    timeline_digest = _ch47_long_token_timeline_sha256(generated_ids_0_based)
    timeline_digest == _ch47_long_sha256(
        metadata["token_timeline_u32le_sha256"],
        "long-reference token timeline hash",
    ) || throw(ArgumentError("long-reference token timeline hash changed"))

    required = Int.(collect(required_lengths))
    isempty(required) && throw(ArgumentError(
        "long-reference validation requires at least one target length",
    ))
    length(unique(required)) == length(required) || throw(ArgumentError(
        "long-reference target lengths must not contain duplicates",
    ))
    sort!(required)
    first(required) >= 4 || throw(ArgumentError(
        "long-reference target lengths must be at least four tokens",
    ))
    last(required) <= length(generated_ids_0_based) || throw(ArgumentError(
        "long-reference token timeline is shorter than the requested workload",
    ))
    prefix_hashes = _ch47_long_string_dict(
        metadata["checkpoint_prefix_u32le_sha256"],
        "long-reference checkpoint hashes",
    )
    Set(keys(prefix_hashes)) ==
        Set(string.(_CH47_LONG_REFERENCE_CHECKPOINT_LENGTHS)) ||
        throw(ArgumentError(
            "long-reference checkpoint lengths changed",
        ))
    for token_count in _CH47_LONG_REFERENCE_CHECKPOINT_LENGTHS
        key = string(token_count)
        expected_prefix = _ch47_long_token_timeline_sha256(
            @view(generated_ids_0_based[1:token_count]),
        )
        prefix_hashes[key] == expected_prefix || throw(ArgumentError(
            "long-reference $token_count-token checkpoint hash changed",
        ))
    end
    _ch47_validate_long_cache_checkpoints(
        metadata["checkpoint_cache_geometry"],
        expected_prompt_tokens;
        num_hidden_layers=expected_num_hidden_layers,
        num_key_value_heads=expected_num_key_value_heads,
        head_dim=expected_head_dim,
    )
    _ch47_validate_long_decisions(
        metadata["decisions"],
        generated_ids_0_based,
        expected_prompt_tokens,
        expected_rope_delta,
        expected_vocab_size,
    )
    top_two = [
        (
            top1_token_id_0_based=
                Int(decision["top_two"]["top1_token_id_0_based"]),
            top2_token_id_0_based=
                Int(decision["top_two"]["top2_token_id_0_based"]),
            top1_logit_f32=
                Float32(decision["top_two"]["top1_logit_f32"]),
            top2_logit_f32=
                Float32(decision["top_two"]["top2_logit_f32"]),
            margin_f32=Float32(decision["top_two"]["margin_f32"]),
        ) for decision in metadata["decisions"]
    ]

    return (;
        path=resolved_path,
        file_sha256=actual_digest,
        token_timeline_u32le_sha256=timeline_digest,
        generated_ids_0_based,
        generated_ids_1_based=generated_ids_0_based .+ 1,
        input_ids_0_based,
        input_ids_u32le_sha256=input_ids_digest,
        generated_token_count=length(generated_ids_0_based),
        top_two,
        checkpoint_prefix_u32le_sha256=prefix_hashes,
        chapter45_reference_sha256=chapter45_hashes,
        source_sha256=actual_sources,
        compute_device=String(metadata["compute_device"]),
        claim=String(metadata["claim"]),
        cuda_device=String(metadata["cuda_device"]),
        transformers=String(metadata["transformers"]),
        torch=String(metadata["torch"]),
        generated_text=String(metadata["generated_text"]),
    )
end

function _ch47_long_first_divergence(actual_ids_1_based, reference)
    actual = Int.(collect(actual_ids_1_based))
    token_count = length(actual)
    token_count <= reference.generated_token_count || throw(ArgumentError(
        "LifeAI token timeline is longer than the pinned HF oracle",
    ))
    mismatch = findfirst(
        index -> actual[index] != reference.generated_ids_1_based[index],
        eachindex(actual),
    )
    mismatch === nothing && return nothing
    top_two = reference.top_two[mismatch]
    return (;
        generated_step=mismatch,
        lifeai_token_id_1_based=actual[mismatch],
        lifeai_token_id_0_based=actual[mismatch] - 1,
        hf_token_id_1_based=reference.generated_ids_1_based[mismatch],
        hf_token_id_0_based=reference.generated_ids_0_based[mismatch],
        hf_top2_token_id_0_based=top_two.top2_token_id_0_based,
        hf_top1_logit_f32=top_two.top1_logit_f32,
        hf_top2_logit_f32=top_two.top2_logit_f32,
        hf_margin_f32=top_two.margin_f32,
    )
end
