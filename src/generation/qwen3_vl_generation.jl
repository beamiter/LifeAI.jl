using MLDataDevices: cpu_device, get_device

function _qwen3_vl_generation_stop_ids(parameters, stop_token_ids)
    raw = if stop_token_ids === nothing
        if hasproperty(parameters, :checkpoint)
            checkpoint = parameters.checkpoint
            (checkpoint.eos_token_id + 1, checkpoint.bos_token_id + 1)
        else
            ()
        end
    else
        stop_token_ids
    end
    stops = Set(Int.(collect(raw)))
    all(id -> 1 <= id <= parameters.spec.vocab_size, stops) || throw(
        ArgumentError("Qwen3-VL stop token id is outside the vocabulary"),
    )
    return stops
end

function _qwen3_vl_generation_limits(
    spec::Qwen3VLTextSpec,
    prompt_length::Int,
    max_new_tokens::Int,
)
    max_new_tokens >= 0 || throw(ArgumentError(
        "max_new_tokens must be non-negative",
    ))
    prompt_length > 0 || throw(ArgumentError(
        "Qwen3-VL prompt must contain at least one token",
    ))
    decode_appends = max(0, max_new_tokens - 1)
    processed = try
        Base.Checked.checked_add(prompt_length, decode_appends)
    catch error
        error isa OverflowError || rethrow()
        throw(ArgumentError("Qwen3-VL generation length exceeds Int capacity"))
    end
    processed <= spec.max_position_embeddings || throw(ArgumentError(
        "Qwen3-VL prompt plus generated context exceeds max_position_embeddings",
    ))
    return processed
end

function _qwen3_vl_generation_cache(
    text_parameters,
    required_capacity::Int,
    cache_mode::Symbol,
    static_capacity,
)
    if cache_mode === :dynamic
        static_capacity === nothing || throw(ArgumentError(
            "static_capacity is only valid when cache=:static",
        ))
        return init_qwen3_vl_kv_cache(text_parameters; batch_size=1)
    elseif cache_mode === :static
        capacity = if static_capacity === nothing
            required_capacity
        else
            static_capacity isa Integer && !(static_capacity isa Bool) || throw(
                ArgumentError("static_capacity must be an integer"),
            )
            Int(static_capacity)
        end
        capacity >= required_capacity || throw(ArgumentError(
            "static_capacity is smaller than the processed generation context",
        ))
        return init_qwen3_vl_static_kv_cache(
            text_parameters;
            capacity,
            batch_size=1,
        )
    end
    throw(ArgumentError("Qwen3-VL cache must be :dynamic or :static"))
end

function _qwen3_vl_generation_prefill(
    text_parameters,
    tokens,
    rope_layout,
    vision_features,
    cache::Qwen3VLKVCache,
)
    return hf_qwen3_vl_text_prefill_cached(
        text_parameters,
        tokens,
        rope_layout;
        vision_features,
        cache,
        logits_to_keep=1,
    )
end

function _qwen3_vl_generation_prefill(
    text_parameters,
    tokens,
    rope_layout,
    vision_features,
    cache::Qwen3VLStaticKVCache,
)
    return hf_qwen3_vl_text_prefill_static(
        text_parameters,
        tokens,
        rope_layout;
        vision_features,
        cache,
        logits_to_keep=1,
    )
end

_qwen3_vl_generation_decode(text_parameters, token, cache::Qwen3VLKVCache) =
    hf_qwen3_vl_text_decode_step(text_parameters, token, cache)

_qwen3_vl_generation_decode(
    text_parameters,
    token,
    cache::Qwen3VLStaticKVCache,
) = hf_qwen3_vl_text_decode_step_static(text_parameters, token, cache)

"""
    generate_hf_qwen3_vl_tokens(text_parameters, input_ids, rope_layout;
                                vision_features=nothing,
                                max_new_tokens=32, stop_token_ids=nothing,
                                capture_logits=false, cache=:dynamic,
                                static_capacity=nothing)

Greedily generate one Qwen3-VL sequence from already-tokenized multimodal
prefill inputs. The first output token is selected from the final prefill
logits. Only that selected token is then appended through incremental decode
to obtain the next token, avoiding an extra prompt-token decode and the usual
one-token cache offset bug.

Token ids use LifeAI's one-based convention. The returned cache therefore
contains `prompt_length + generated_count - 1` valid tokens unless generation
stops before a decode is needed. `cache=:static` selects preallocated storage;
`static_capacity` defaults to the exact processed context. The returned static
cache can be reset and reused through the low-level static-cache API.
Generation intentionally supports batch size one and greedy selection only.
"""
function generate_hf_qwen3_vl_tokens(
    text_parameters,
    input_ids,
    rope_layout::Qwen3VLRopeLayout;
    vision_features=nothing,
    max_new_tokens::Int=32,
    stop_token_ids=nothing,
    capture_logits::Bool=false,
    cache::Symbol=:dynamic,
    static_capacity=nothing,
)
    tokens = _qwen3_vl_token_matrix(input_ids)
    prompt_length, batch_size = size(tokens)
    batch_size == 1 || throw(ArgumentError(
        "Qwen3-VL generation currently supports batch size one",
    ))
    required_capacity = _qwen3_vl_generation_limits(
        text_parameters.spec,
        prompt_length,
        max_new_tokens,
    )
    all(id -> 1 <= id <= text_parameters.spec.vocab_size, tokens) || throw(
        ArgumentError("Qwen3-VL input_ids contain an out-of-vocabulary id"),
    )
    stops = _qwen3_vl_generation_stop_ids(text_parameters, stop_token_ids)
    prompt_ids = vec(copy(tokens))
    generated_ids = Int[]
    trace = NamedTuple[]
    cache_state = _qwen3_vl_generation_cache(
        text_parameters,
        required_capacity,
        cache,
        static_capacity,
    )
    stop_reason = :length

    if max_new_tokens == 0
        return (;
            prompt_ids,
            generated_ids,
            token_ids=copy(prompt_ids),
            stop_reason,
            strategy=:greedy,
            trace=Tuple(trace),
            prefill=nothing,
            cache=cache_state,
            cache_mode=cache,
        )
    end

    prefill_result, cache_state = _qwen3_vl_generation_prefill(
        text_parameters,
        tokens,
        rope_layout,
        vision_features,
        cache_state,
    )
    logits = prefill_result.logits
    host = cpu_device()

    for step in 1:max_new_tokens
        token_id, token_trace = _hf_greedy_choice(
            logits,
            host,
            capture_logits,
        )
        push!(generated_ids, token_id)
        push!(trace, merge((; step), token_trace))
        if token_id in stops
            stop_reason = :eos
            break
        end
        if step < max_new_tokens
            logits, cache_state = _qwen3_vl_generation_decode(
                text_parameters,
                token_id,
                cache_state,
            )
        end
    end

    return (;
        prompt_ids,
        generated_ids,
        token_ids=vcat(prompt_ids, generated_ids),
        stop_reason,
        strategy=:greedy,
        trace=Tuple(trace),
        prefill=prefill_result,
        cache=cache_state,
        cache_mode=cache,
    )
end

function _qwen3_vl_generation_image(messages)
    images = Any[]
    for message in messages
        role = _hf_message_value(message, :role; required=true)
        raw_content = _hf_message_value(message, :content; default=nothing)
        raw_content isa AbstractString && continue
        items = _qwen3_vl_content_list(raw_content, "Qwen3-VL generation")
        for item in items
            content_type = _qwen3_vl_content_value(item, :type; default=nothing)
            has_video = content_type == "video" ||
                _qwen3_vl_content_has(item, :video)
            has_video && throw(ArgumentError(
                "Qwen3-VL video generation is not implemented",
            ))
            has_image = content_type == "image" ||
                _qwen3_vl_content_has(item, :image) ||
                _qwen3_vl_content_has(item, :image_url)
            has_image || continue
            String(role) == "user" || throw(ArgumentError(
                "Qwen3-VL generation only accepts image content in user messages",
            ))
            _qwen3_vl_content_has(item, :image_url) && throw(ArgumentError(
                "Qwen3-VL image_url loading is not implemented; pass an image path " *
                "or UInt8 array in the image field",
            ))
            _qwen3_vl_content_has(item, :image) || throw(ArgumentError(
                "Qwen3-VL image content requires an image payload",
            ))
            image = _qwen3_vl_content_value(item, :image; default=nothing)
            image === nothing && throw(ArgumentError(
                "Qwen3-VL image payload must not be null",
            ))
            push!(images, image)
        end
    end
    length(images) == 1 || throw(ArgumentError(
        "Qwen3-VL generation requires exactly one image; got $(length(images))",
    ))
    return only(images)
end

function _qwen3_vl_generation_process_image(image, processor_spec)
    if image isa AbstractString
        return qwen3_vl_process_image(image; spec=processor_spec)
    elseif image isa AbstractArray{UInt8,3}
        layout = if size(image, 3) == 3
            :hwc
        elseif size(image, 1) == 3
            :chw
        else
            throw(DimensionMismatch(
                "Qwen3-VL UInt8 image arrays must have three RGB channels",
            ))
        end
        return qwen3_vl_process_image(image; layout, spec=processor_spec)
    end
    throw(ArgumentError(
        "Qwen3-VL image payload must be a local path or a three-dimensional " *
        "UInt8 array",
    ))
end

"""
    generate_hf_qwen3_vl(vision_parameters, text_parameters, tokenizer,
                         messages; max_new_tokens=32, ...)

Run the raw single-image Qwen3-VL chat path: exact image processing,
content-list rendering, per-grid placeholder expansion, tokenizer encoding,
mRoPE layout, vision tower, cached multimodal prefill, and greedy incremental
decode. Image content must carry either a local path or an HWC/CHW UInt8 array
in its `image` field. URL, video, multi-image, padding, and batched generation
remain explicit non-goals.
"""
function generate_hf_qwen3_vl(
    vision_parameters,
    text_parameters,
    tokenizer::HFQwen3Tokenizer,
    messages;
    max_new_tokens::Int=32,
    stop_token_ids=nothing,
    tools=nothing,
    add_vision_id::Bool=false,
    capture_logits::Bool=false,
    decode_errors::Symbol=:replace,
    skip_special_tokens::Bool=true,
    processor_spec::Qwen3VLProcessorSpec=qwen3_vl_processor_spec(),
    cache::Symbol=:dynamic,
    static_capacity=nothing,
)
    tokenizer.profile === :qwen3_vl_generation || throw(ArgumentError(
        "generate_hf_qwen3_vl requires a Qwen3-VL tokenizer",
    ))
    vision_parameters.spec.out_hidden_size == text_parameters.spec.hidden_size ||
        throw(DimensionMismatch(
            "Qwen3-VL vision output width does not match text hidden size",
        ))
    vision_dtype = eltype(vision_parameters.patch_weight)
    vision_dtype == eltype(text_parameters.embedding) || throw(ArgumentError(
        "Qwen3-VL vision and text parameter dtypes must match",
    ))
    get_device(vision_parameters.patch_weight) ==
        get_device(text_parameters.embedding) || throw(ArgumentError(
            "Qwen3-VL vision and text parameters must reside on the same device",
        ))

    message_list = collect(Any, messages)
    image = _qwen3_vl_generation_image(message_list)
    processed = _qwen3_vl_generation_process_image(image, processor_spec)
    prompt = apply_qwen3_vl_chat_template(
        tokenizer,
        message_list;
        tools,
        add_generation_prompt=true,
        add_vision_id,
    )
    expanded_prompt = qwen3_vl_expand_image_placeholders(
        prompt,
        processed.grid_thw;
        spec=processor_spec,
    )
    prompt_ids = encode(tokenizer, expanded_prompt; add_special_tokens=false)
    rope_layout = qwen3_vl_rope_layout(
        prompt_ids,
        processed.grid_thw;
        checkpoint=text_parameters.checkpoint,
    )

    to_device = _bf16a_device_mover(vision_parameters.patch_weight)
    pixels = to_device(vision_dtype.(processed.pixel_values))
    vision_input = Qwen3VLVisionInput(
        pixels,
        processed.grid_thw;
        spec=vision_parameters.spec,
    )
    vision_features = hf_qwen3_vl_vision_forward(
        vision_parameters,
        vision_input,
    )
    resolved_stops = stop_token_ids === nothing ?
        tokenizer.eos_ids : stop_token_ids
    generated = generate_hf_qwen3_vl_tokens(
        text_parameters,
        prompt_ids,
        rope_layout;
        vision_features,
        max_new_tokens,
        stop_token_ids=resolved_stops,
        capture_logits,
        cache,
        static_capacity,
    )
    completion = decode(
        tokenizer,
        generated.generated_ids;
        errors=decode_errors,
        skip_special_tokens,
    )
    text = decode(
        tokenizer,
        generated.token_ids;
        errors=decode_errors,
        skip_special_tokens,
    )
    return merge(generated, (;
        prompt,
        expanded_prompt,
        completion,
        text,
        image_grid_thw=copy(processed.grid_thw),
        original_image_size=processed.original_size,
        resized_image_size=processed.resized_size,
        rope_layout,
    ))
end
