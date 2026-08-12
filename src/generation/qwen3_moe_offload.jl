using BFloat16s: BFloat16

# Qwen3 MoE deployment keeps the comparatively small attention/router tree,
# final norm and LM head resident on the accelerator. Expert matrices remain
# in safetensors and are uploaded one active layer at a time. The implementation
# is device-generic; the CUDA extension supplies the grouped BF16 WMMA method.

function _qwen3_moe_tree_bytes(x::AbstractArray)
    return Base.checked_mul(length(x), sizeof(eltype(x)))
end
_qwen3_moe_tree_bytes(x::NamedTuple) =
    sum(_qwen3_moe_tree_bytes, values(x); init=0)
_qwen3_moe_tree_bytes(x::Tuple) =
    sum(_qwen3_moe_tree_bytes, values(x); init=0)
_qwen3_moe_tree_bytes(x) = 0

"""
    qwen3_moe_offload_plan(model, context_tokens; ...)

Estimate the irreducible BF16 device payload for a Qwen3 MoE offload session.
The estimate includes resident attention/router/norm/LM-head parameters, a
static KV cache and at most `max_active_experts` experts for one layer. It does
not include allocator slack, attention scores or grouped-dispatch workspace.
"""
function qwen3_moe_offload_plan(
    model::GPTModel,
    context_tokens::Integer;
    batch_size::Integer=1,
    max_active_experts::Integer=model.num_experts,
    dtype_bytes::Integer=sizeof(BFloat16),
)
    _qwen3_validate_moe_semantics(model)
    context = Int(context_tokens)
    batch = Int(batch_size)
    active = Int(max_active_experts)
    bytes = Int(dtype_bytes)
    0 < context <= model.max_seq_len || throw(ArgumentError(
        "context_tokens must be in 1:model.max_seq_len",
    ))
    batch > 0 || throw(ArgumentError("batch_size must be positive"))
    1 <= active <= model.num_experts || throw(ArgumentError(
        "max_active_experts must be in 1:model.num_experts",
    ))
    bytes > 0 || throw(ArgumentError("dtype_bytes must be positive"))

    d_model = model.d_model
    query_dim = model.num_heads * model.head_dim
    kv_dim = model.num_kv_heads * model.head_dim
    attention_elements =
        query_dim * d_model +
        2 * kv_dim * d_model +
        d_model * query_dim
    layer_resident_elements =
        attention_elements +
        model.num_experts * d_model +
        2 * d_model +
        2 * model.head_dim
    resident_parameter_elements =
        model.num_layers * layer_resident_elements +
        d_model +
        model.vocab_size * d_model
    active_expert_elements =
        active * 3 * d_model * model.mlp_hidden_dim
    resident_parameter_bytes = Base.checked_mul(
        resident_parameter_elements,
        bytes,
    )
    active_expert_layer_bytes = Base.checked_mul(active_expert_elements, bytes)
    kv_cache_bytes = qwen3_kv_cache_bytes(
        model,
        context;
        batch_size=batch,
        dtype_bytes=bytes,
    )
    working_set_floor_bytes = Base.checked_add(
        Base.checked_add(resident_parameter_bytes, active_expert_layer_bytes),
        kv_cache_bytes,
    )
    return (;
        context_tokens=context,
        batch_size=batch,
        max_active_experts=active,
        dtype_bytes=bytes,
        resident_parameter_bytes,
        active_expert_layer_bytes,
        kv_cache_bytes,
        working_set_floor_bytes,
    )
end

function _qwen3_local_expert_routes(
    expert_indices::AbstractMatrix{<:Integer},
    num_experts::Int,
)
    isempty(expert_indices) && throw(ArgumentError(
        "expert route table must not be empty",
    ))
    active_experts = sort!(unique!(Int.(vec(collect(expert_indices)))))
    all(expert -> 1 <= expert <= num_experts, active_experts) ||
        throw(ArgumentError("expert route index is outside 1:num_experts"))
    global_to_local = zeros(Int32, num_experts)
    for (local_index, global_expert) in enumerate(active_experts)
        global_to_local[global_expert] = Int32(local_index)
    end
    local_indices = reshape(
        global_to_local[Int.(vec(collect(expert_indices)))],
        size(expert_indices),
    )
    return (; active_experts, local_indices)
end

# Portable fallback used by non-CUDA devices. LifeAICUDAExt specializes this
# hook to the grouped BF16 WMMA implementation.
function _qwen3_grouped_bf16_expert_dispatch(
    tokens,
    expert_indices,
    routing_weights,
    expert_parameters,
)
    return qwen3_device_sparse_expert_dispatch(
        tokens,
        expert_indices,
        routing_weights,
        expert_parameters,
    )
end

mutable struct HFQwen3MoEOffloadSession
    model
    config
    reader::HFSafetensorsReader
    tensors
    resident_blocks::Vector{Any}
    final_scale
    logits_weight
    cos_table
    sin_table
    caches::Vector{Any}
    position::Int
    context_tokens::Int
    prefill_chunk_tokens::Int
    grouped_experts::Bool
    to_device
    source::String
    resident_parameter_bytes::Int
    expert_bytes_read::Int
end

function _qwen3_moe_load_active_experts(
    session::HFQwen3MoEOffloadSession,
    active_experts::Vector{Int},
    layer::Int,
)
    model = session.model
    hidden_dim = model.mlp_hidden_dim
    d_model = model.d_model
    prefix = "model.layers.$layer.mlp.experts"
    gate_values = map(active_experts) do expert
        reshape(_expect_tensor(
            session.tensors,
            "$prefix.$(expert - 1).gate_proj.weight",
            (hidden_dim, d_model),
        ), hidden_dim, d_model, 1)
    end
    up_values = map(active_experts) do expert
        reshape(_expect_tensor(
            session.tensors,
            "$prefix.$(expert - 1).up_proj.weight",
            (hidden_dim, d_model),
        ), hidden_dim, d_model, 1)
    end
    down_values = map(active_experts) do expert
        reshape(_expect_tensor(
            session.tensors,
            "$prefix.$(expert - 1).down_proj.weight",
            (d_model, hidden_dim),
        ), d_model, hidden_dim, 1)
    end
    host_parameters = (;
        gate_proj=cat(gate_values...; dims=3),
        up_proj=cat(up_values...; dims=3),
        down_proj=cat(down_values...; dims=3),
    )
    session.expert_bytes_read = Base.checked_add(
        session.expert_bytes_read,
        _qwen3_moe_tree_bytes(host_parameters),
    )
    parameters = session.to_device(host_parameters)
    host_parameters = nothing
    gate_values = nothing
    up_values = nothing
    down_values = nothing
    GC.gc(false)
    return parameters
end

function _qwen3_moe_offload_block(
    session::HFQwen3MoEOffloadSession,
    x,
    layer_index::Int,
    mask,
)
    model = session.model
    ps = session.resident_blocks[layer_index]
    num_tokens, batch_size = size(x, 2), size(x, 3)
    head_dim = model.head_dim
    query_dim = model.num_heads * head_dim
    cache_position = session.position
    positions = (cache_position + 1):(cache_position + num_tokens)

    normed = _bf16a_rmsnorm(x, ps.norm1.scale, model.norm_epsilon)
    queries = reshape(
        _bf16a_linear(ps.attn.q_proj.weight, normed),
        head_dim, model.num_heads, num_tokens, batch_size,
    )
    keys = reshape(
        _bf16a_linear(ps.attn.k_proj.weight, normed),
        head_dim, model.num_kv_heads, num_tokens, batch_size,
    )
    values = reshape(
        _bf16a_linear(ps.attn.v_proj.weight, normed),
        head_dim, model.num_kv_heads, num_tokens, batch_size,
    )
    queries = _bf16a_rmsnorm(
        queries,
        ps.attn.q_norm.scale,
        model.qk_norm_epsilon,
    )
    keys = _bf16a_rmsnorm(
        keys,
        ps.attn.k_norm.scale,
        model.qk_norm_epsilon,
    )
    cos_slice = session.cos_table[:, positions]
    sin_slice = session.sin_table[:, positions]
    queries = _bf16a_apply_rope(queries, cos_slice, sin_slice)
    keys = _bf16a_apply_rope(keys, cos_slice, sin_slice)
    all_keys, all_values, updated_cache = _bf16a_append_cache(
        session.caches[layer_index],
        keys,
        values,
        cache_position,
    )
    context = _bf16a_attention(
        queries,
        all_keys,
        all_values;
        scaling=1.0f0 / sqrt(Float32(head_dim)),
        mask,
    )
    attention_output = _bf16a_linear(
        ps.attn.o_proj.weight,
        reshape(context, query_dim, num_tokens, batch_size),
    )
    residual = BFloat16.(_bf16a_f32(x) .+ _bf16a_f32(attention_output))
    normed2 = _bf16a_rmsnorm(residual, ps.norm2.scale, model.norm_epsilon)
    router_logits = reshape(
        _bf16a_linear(ps.gate.weight, normed2),
        model.num_experts,
        :,
    )
    routed = qwen3_device_topk_routing(
        router_logits,
        model.experts_per_token;
        normalize=model.normalize_routing,
    )

    host_routes = Array(routed.expert_indices)
    remapped = _qwen3_local_expert_routes(host_routes, model.num_experts)
    local_indices = session.to_device(remapped.local_indices)
    expert_parameters = _qwen3_moe_load_active_experts(
        session,
        remapped.active_experts,
        layer_index - 1,
    )
    expert_tokens = reshape(_bf16a_f32(normed2), model.d_model, :)
    expert_output = if session.grouped_experts
        _qwen3_grouped_bf16_expert_dispatch(
            expert_tokens,
            local_indices,
            routed.routing_weights,
            expert_parameters,
        )
    else
        qwen3_device_sparse_expert_dispatch(
            expert_tokens,
            local_indices,
            routed.routing_weights,
            expert_parameters,
        )
    end
    output = BFloat16.(
        _bf16a_f32(residual) .+
        reshape(expert_output, model.d_model, num_tokens, batch_size)
    )
    expert_parameters = nothing
    GC.gc(false)
    return output, updated_cache, remapped.active_experts
end

function _qwen3_moe_offload_chunk!(
    session::HFQwen3MoEOffloadSession,
    token_matrix::Matrix{Int},
    ;
    project_logits::Bool=true,
)
    seq_len, batch_size = size(token_matrix)
    batch_size == 1 || throw(ArgumentError(
        "Qwen3 MoE offload sessions currently require batch_size == 1",
    ))
    session.position + seq_len <= session.context_tokens || throw(ArgumentError(
        "tokens exceed the offload session context capacity",
    ))
    host_embedding = _read_embedding_rows(
        session.reader,
        "model.embed_tokens.weight",
        token_matrix,
        session.model.d_model,
        session.model.vocab_size;
        target_dtype=BFloat16,
    )
    x = session.to_device(host_embedding)
    host_embedding = nothing
    key_tokens = session.position + seq_len
    mask = session.to_device(_bf16a_causal_mask(seq_len, key_tokens))
    active_experts = Vector{Vector{Int}}(undef, session.model.num_layers)
    for layer_index in 1:session.model.num_layers
        x, session.caches[layer_index], active_experts[layer_index] =
            _qwen3_moe_offload_block(
                session,
                x,
                layer_index,
                mask,
            )
    end
    logits = if project_logits
        last_hidden = _bf16a_rmsnorm(
            x[:, end:end, :],
            session.final_scale,
            session.model.norm_epsilon,
        )
        Array(_bf16a_linear(session.logits_weight, last_hidden))
    else
        nothing
    end
    session.position = key_tokens
    return (; logits, active_experts=Tuple(active_experts))
end

"""
    load_hf_qwen3_moe_offload_session(model_dir; to_device=identity, ...)

Construct a BF16 Qwen3 MoE session whose attention/router/norm/LM-head tensors
and static KV cache stay on `to_device`, while active expert tensors stream one
layer at a time. Passing `to_device=CUDA.cu` activates CUDA without coupling
the core package API to CUDA. `grouped_experts=true` selects the CUDA grouped
BF16 tensor-core extension when available.
"""
function load_hf_qwen3_moe_offload_session(
    model_dir::AbstractString;
    context_tokens::Integer=4096,
    prefill_chunk_tokens::Integer=128,
    grouped_experts::Bool=true,
    to_device=identity,
    on_resident_layer=nothing,
)
    isdir(model_dir) || throw(ArgumentError(
        "model directory does not exist: $model_dir",
    ))
    context = Int(context_tokens)
    chunk = Int(prefill_chunk_tokens)
    config = load_hf_qwen3_moe_config(
        joinpath(model_dir, "config.json");
        max_seq_len=context,
    )
    model = GPTModel(config)
    _qwen3_validate_moe_semantics(model)
    0 < chunk <= context || throw(ArgumentError(
        "prefill_chunk_tokens must be in 1:context_tokens",
    ))
    reader = open_safetensors_reader(model_dir)
    _qwen3_validate_moe_tensor_names(
        model,
        Set(String.(collect(keys(reader)))),
    )
    tensors = _StreamedTensors(reader, BFloat16)

    resident_blocks = Vector{Any}(undef, model.num_layers)
    resident_parameter_bytes = 0
    for layer_index in 1:model.num_layers
        host_parameters = _streamed_qwen3_moe_block_parameters(
            model,
            tensors,
            layer_index - 1,
        )
        resident_parameter_bytes = Base.checked_add(
            resident_parameter_bytes,
            _qwen3_moe_tree_bytes(host_parameters),
        )
        resident_blocks[layer_index] = to_device(host_parameters)
        host_parameters = nothing
        GC.gc(false)
        on_resident_layer === nothing || on_resident_layer(
            layer_index,
            model.num_layers,
        )
    end
    final_scale_host = reshape(_expect_tensor(
        tensors,
        "model.norm.weight",
        (model.d_model,),
    ), model.d_model, 1, 1)
    logits_weight_host = _expect_tensor(
        tensors,
        "lm_head.weight",
        (model.vocab_size, model.d_model),
    )
    resident_parameter_bytes = Base.checked_add(
        resident_parameter_bytes,
        Base.checked_add(
            _qwen3_moe_tree_bytes(final_scale_host),
            _qwen3_moe_tree_bytes(logits_weight_host),
        ),
    )
    final_scale = to_device(final_scale_host)
    logits_weight = to_device(logits_weight_host)
    final_scale_host = nothing
    logits_weight_host = nothing
    rope = first(values(model.blocks.layers)).attn.rope
    cos_table = to_device(BFloat16.(rope.cos_cache[:, 1:context]))
    sin_table = to_device(BFloat16.(rope.sin_cache[:, 1:context]))
    caches = _qwen3_session_cache(final_scale, model, context)
    GC.gc()

    return HFQwen3MoEOffloadSession(
        model,
        config,
        reader,
        tensors,
        resident_blocks,
        final_scale,
        logits_weight,
        cos_table,
        sin_table,
        caches,
        0,
        context,
        chunk,
        grouped_experts,
        to_device,
        abspath(model_dir),
        resident_parameter_bytes,
        0,
    )
end

"""Reset the logical prefix while retaining resident tensors and static KV buffers."""
function reset_hf_qwen3_moe_offload_session!(
    session::HFQwen3MoEOffloadSession,
)
    session.position = 0
    session.expert_bytes_read = 0
    return session
end

"""
    prefill_hf_qwen3_moe_offload!(session, prompt_tokens)

Reset and prefill a batch-one prompt in bounded chunks. The result contains
host logits for the final prompt token plus per-chunk active expert ids and
the exact expert payload bytes read from safetensors.
"""
function prefill_hf_qwen3_moe_offload!(
    session::HFQwen3MoEOffloadSession,
    prompt_tokens,
)
    tokens = Int.(vec(collect(prompt_tokens)))
    isempty(tokens) && throw(ArgumentError("prompt must contain at least one token"))
    _validate_generation_ids(tokens, session.model.vocab_size)
    length(tokens) <= session.context_tokens || throw(ArgumentError(
        "prompt exceeds the offload session context capacity",
    ))
    reset_hf_qwen3_moe_offload_session!(session)
    chunks = Any[]
    result = nothing
    for first_index in 1:session.prefill_chunk_tokens:length(tokens)
        last_index = min(
            first_index + session.prefill_chunk_tokens - 1,
            length(tokens),
        )
        before = session.expert_bytes_read
        result = _qwen3_moe_offload_chunk!(
            session,
            reshape(tokens[first_index:last_index], :, 1),
            ;
            project_logits=last_index == length(tokens),
        )
        push!(chunks, (;
            first_token=first_index,
            last_token=last_index,
            active_experts=result.active_experts,
            expert_bytes_read=session.expert_bytes_read - before,
        ))
    end
    return (;
        logits=result.logits,
        position=session.position,
        chunks=Tuple(chunks),
        expert_bytes_read=session.expert_bytes_read,
    )
end

"""Decode one batch-one token using the session's resident static KV cache."""
function decode_hf_qwen3_moe_offload!(
    session::HFQwen3MoEOffloadSession,
    token,
)
    session.position > 0 || throw(ArgumentError(
        "prefill must run before Qwen3 MoE offload decode",
    ))
    values = Int.(vec(collect(token isa Integer ? [token] : token)))
    length(values) == 1 || throw(ArgumentError(
        "Qwen3 MoE offload decode requires exactly one token",
    ))
    _validate_generation_ids(values, session.model.vocab_size)
    before = session.expert_bytes_read
    result = _qwen3_moe_offload_chunk!(session, reshape(values, 1, 1))
    return (;
        logits=result.logits,
        position=session.position,
        active_experts=result.active_experts,
        expert_bytes_read=session.expert_bytes_read - before,
    )
end
