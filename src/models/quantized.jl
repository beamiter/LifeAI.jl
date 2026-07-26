using BFloat16s: BFloat16
import Adapt

# Week 16: RTN weight-only quantization. Linear weights (including the LM
# head) are stored as INT8 with one symmetric scale per output channel, or as
# packed INT4 with one symmetric scale per output channel per input group.
# Embeddings and norm scales stay BF16. Compute is unchanged: weights are
# dequantized back to BF16 right before the existing `_bf16a_linear`, so
# quantization only changes weight residency, never the compute contract.

"""
    Int8ChannelWeight(q, scale)

Symmetric per-output-channel INT8 weight: `w ≈ Float32(q) .* scale` with
`q::Matrix{Int8}` of shape `(out, in)` and `scale::Vector{Float32}` per row.
"""
struct Int8ChannelWeight{Q,S}
    q::Q
    scale::S
end
Adapt.@adapt_structure Int8ChannelWeight

"""
    Int4GroupWeight(packed, scale, group, in_dim)

Symmetric group-wise INT4 weight. Adjacent input columns are packed two per
byte (`packed::Matrix{UInt8}` of shape `(out, in ÷ 2)`, low nibble first,
values offset by +8); `scale` has shape `(out, in ÷ group)`.
"""
struct Int4GroupWeight{Q,S}
    packed::Q
    scale::S
    group::Int
    in_dim::Int
end
Adapt.Adapt.adapt_structure(to, w::Int4GroupWeight) = Int4GroupWeight(
    Adapt.adapt(to, w.packed),
    Adapt.adapt(to, w.scale),
    w.group,
    w.in_dim,
)

function _quantize_int8_channel(weight::AbstractMatrix)
    w = Float32.(weight)
    scale = vec(maximum(abs, w; dims=2)) ./ 127.0f0
    scale = max.(scale, 1.0f-12)
    q = Int8.(clamp.(round.(w ./ scale), -127.0f0, 127.0f0))
    return Int8ChannelWeight(q, scale)
end

function _quantize_int4_group(weight::AbstractMatrix; group::Int=128)
    out_dim, in_dim = size(weight)
    in_dim % group == 0 || throw(ArgumentError(
        "input dimension $in_dim is not divisible by group $group",
    ))
    iseven(group) || throw(ArgumentError("group size must be even"))
    w = Float32.(weight)
    groups = in_dim ÷ group
    grouped = reshape(w, out_dim, group, groups)
    scale = reshape(maximum(abs, grouped; dims=2), out_dim, groups) ./ 7.0f0
    scale = max.(scale, 1.0f-12)
    q = clamp.(
        round.(grouped ./ reshape(scale, out_dim, 1, groups)),
        -7.0f0,
        7.0f0,
    )
    shifted = UInt8.(reshape(q, out_dim, in_dim) .+ 8.0f0)
    low = shifted[:, 1:2:end]
    high = shifted[:, 2:2:end]
    packed = low .| (high .<< 4)
    return Int4GroupWeight(packed, scale, group, in_dim)
end

_dequantize_bf16(weight::AbstractMatrix) = weight

function _dequantize_bf16(weight::Int8ChannelWeight)
    return BFloat16.(Float32.(weight.q) .* reshape(weight.scale, :, 1))
end

function _dequantize_bf16(weight::Int4GroupWeight)
    out_dim = size(weight.packed, 1)
    # Packed column c holds input columns (2c-1, 2c); both live in the same
    # group, so one gathered scale column serves the pair. Each half is fused
    # into a single broadcast so no full-width Float32 intermediate exists.
    packed_group_index = ((2 .* (1:(weight.in_dim ÷ 2)) .- 2) .÷ weight.group) .+ 1
    scale_half = weight.scale[:, packed_group_index]
    low = BFloat16.((Float32.(weight.packed .& 0x0f) .- 8.0f0) .* scale_half)
    high = BFloat16.((Float32.(weight.packed .>> 4) .- 8.0f0) .* scale_half)
    stacked = cat(
        reshape(low, out_dim, 1, weight.in_dim ÷ 2),
        reshape(high, out_dim, 1, weight.in_dim ÷ 2);
        dims=2,
    )
    return reshape(stacked, out_dim, weight.in_dim)
end

_quant_row_slice(weight::Int8ChannelWeight, rows) =
    Int8ChannelWeight(weight.q[rows, :], weight.scale[rows])
_quant_row_slice(weight::Int4GroupWeight, rows) = Int4GroupWeight(
    weight.packed[rows, :], weight.scale[rows, :], weight.group, weight.in_dim,
)
_quant_out_dim(weight::Int8ChannelWeight) = size(weight.q, 1)
_quant_out_dim(weight::Int4GroupWeight) = size(weight.packed, 1)

# Dequantize and multiply in output-row chunks: a monolithic BF16 lm_head
# (1.5 GiB for 14B) plus its dequant transients is exactly what pushed the
# GPU memory pool over the edge.
const _QUANT_LINEAR_CHUNK_ROWS = 8_192

function _bf16a_linear_quantized(weight, x::AbstractArray{<:Any,3})
    out_dim = _quant_out_dim(weight)
    out_dim <= _QUANT_LINEAR_CHUNK_ROWS &&
        return _bf16a_linear(_dequantize_bf16(weight), x)
    in_dim, num_tokens, batch_size = size(x)
    chunks = map(1:_QUANT_LINEAR_CHUNK_ROWS:out_dim) do row_start
        rows = row_start:min(row_start + _QUANT_LINEAR_CHUNK_ROWS - 1, out_dim)
        _bf16a_linear(_dequantize_bf16(_quant_row_slice(weight, rows)), x)
    end
    return cat(chunks...; dims=1)
end

_bf16a_linear(weight::Int8ChannelWeight, x::AbstractArray{<:Any,3}) =
    _bf16a_linear_quantized(weight, x)
_bf16a_linear(weight::Int4GroupWeight, x::AbstractArray{<:Any,3}) =
    _bf16a_linear_quantized(weight, x)

function _quantize_linear(weight, scheme::Symbol, group::Int)
    scheme === :int8 && return _quantize_int8_channel(weight)
    scheme === :int4 && return _quantize_int4_group(weight; group)
    throw(ArgumentError("unsupported quantization scheme $(repr(scheme))"))
end

"""
    load_hf_qwen3_quantized(
        model_dir;
        max_seq_len=64,
        scheme=:int8,
        group=128,
        variant=nothing,
    )

Stream a HuggingFace Qwen3 checkpoint from disk and quantize it layer by
layer, so the full BF16 tree is never resident: peak host memory is the
quantized tree plus one layer of transients. Embedding and norm scales stay
BF16; every linear weight (including an untied LM head) becomes INT8
per-channel or packed INT4 group-wise.
"""
function load_hf_qwen3_quantized(
    model_dir::AbstractString;
    max_seq_len=64,
    scheme::Symbol=:int8,
    group::Int=128,
    int8_projections::Tuple{Vararg{Symbol}}=(),
    variant=nothing,
)
    isdir(model_dir) || throw(ArgumentError("model directory does not exist: $model_dir"))
    config = load_hf_qwen3_config(
        joinpath(model_dir, "config.json");
        max_seq_len,
        variant,
    )
    model = GPTModel(config)
    _qwen3_validate_semantics(model)
    reader = open_safetensors_reader(model_dir)
    _qwen3_validate_tensor_names(model, Set(String.(collect(keys(reader)))))
    tensors = _StreamedTensors(reader)

    embedding_hf = _expect_tensor(
        tensors,
        "model.embed_tokens.weight",
        (model.vocab_size, model.d_model),
    )
    token_embedding = (; weight=BFloat16.(permutedims(embedding_hf, (2, 1))))
    embedding_hf = nothing
    GC.gc(false)

    _proj_scheme(name) = name in int8_projections ? :int8 : scheme
    d_model = model.d_model
    q_dim = model.num_heads * model.head_dim
    kv_dim = model.num_kv_heads * model.head_dim
    hidden_dim = model.mlp_hidden_dim
    # 逐投影读入→量化→释放：整块 Float32 常驻曾把 14B 混合精度加载的
    # 宿主峰值推过 22 GiB 触发 OOM。
    function _quantized_projection(prefix, name, shape, proj)
        weight = _expect_tensor(tensors, "$prefix.$name", shape)
        quantized = _quantize_linear(weight, _proj_scheme(proj), group)
        weight = nothing
        GC.gc(false)
        return (; weight=quantized)
    end
    block_values = Vector{Any}(undef, model.num_layers)
    for index in 1:model.num_layers
        prefix = "model.layers.$(index - 1)"
        norm_scale(name, dims) = BFloat16.(reshape(
            _expect_tensor(tensors, "$prefix.$name", (dims,)),
            ntuple(i -> i == 1 ? dims : 1, 3)...,
        ))
        block_values[index] = (;
            norm1=(; scale=norm_scale("input_layernorm.weight", d_model)),
            attn=(;
                q_proj=_quantized_projection(prefix, "self_attn.q_proj.weight", (q_dim, d_model), :q_proj),
                k_proj=_quantized_projection(prefix, "self_attn.k_proj.weight", (kv_dim, d_model), :k_proj),
                v_proj=_quantized_projection(prefix, "self_attn.v_proj.weight", (kv_dim, d_model), :v_proj),
                o_proj=_quantized_projection(prefix, "self_attn.o_proj.weight", (d_model, q_dim), :o_proj),
                q_norm=(; scale=BFloat16.(_expect_tensor(
                    tensors, "$prefix.self_attn.q_norm.weight", (model.head_dim,),
                ))),
                k_norm=(; scale=BFloat16.(_expect_tensor(
                    tensors, "$prefix.self_attn.k_norm.weight", (model.head_dim,),
                ))),
            ),
            norm2=(; scale=norm_scale("post_attention_layernorm.weight", d_model)),
            mlp=(;
                gate_proj=_quantized_projection(prefix, "mlp.gate_proj.weight", (hidden_dim, d_model), :gate_proj),
                up_proj=_quantized_projection(prefix, "mlp.up_proj.weight", (hidden_dim, d_model), :up_proj),
                down_proj=_quantized_projection(prefix, "mlp.down_proj.weight", (d_model, hidden_dim), :down_proj),
            ),
        )
    end
    block_names = Tuple(Symbol("layer_$layer") for layer in 1:model.num_layers)
    blocks = NamedTuple{block_names}(Tuple(block_values))

    final_norm = (; scale=BFloat16.(reshape(_expect_tensor(
        tensors,
        "model.norm.weight",
        (model.d_model,),
    ), model.d_model, 1, 1)))
    lm_head = if model.tie_embeddings
        (;)
    else
        head = _quantize_linear(
            _expect_tensor(
                tensors,
                "lm_head.weight",
                (model.vocab_size, model.d_model),
            ),
            scheme,
            group,
        )
        GC.gc(false)
        (; weight=head)
    end
    dense_spec = config.qwen3_variant === nothing ?
        nothing : qwen3_dense_spec(config.qwen3_variant)
    return (;
        model,
        parameters=(; token_embedding, blocks, final_norm, lm_head),
        config,
        variant=dense_spec,
        scheme,
        group,
        source=abspath(model_dir),
    )
end

"""
    quantize_bf16_parameters(ps; scheme=:int8, group=128)

Quantize every linear weight (attention/MLP projections and an untied LM
head) of a BF16 Qwen3 parameter tree with round-to-nearest. Embedding and
norm scales stay BF16. The result feeds `hf_qwen3_bf16_accel_forward`
unchanged and can be moved to the GPU with `CUDA.cu`.
"""
function quantize_bf16_parameters(ps; scheme::Symbol=:int8, group::Int=128)
    eltype(ps.token_embedding.weight) === BFloat16 || throw(ArgumentError(
        "quantize_bf16_parameters expects a BFloat16 parameter tree",
    ))
    blocks = map(values(ps.blocks)) do block
        (;
            norm1=block.norm1,
            attn=(;
                q_proj=(; weight=_quantize_linear(block.attn.q_proj.weight, scheme, group)),
                k_proj=(; weight=_quantize_linear(block.attn.k_proj.weight, scheme, group)),
                v_proj=(; weight=_quantize_linear(block.attn.v_proj.weight, scheme, group)),
                o_proj=(; weight=_quantize_linear(block.attn.o_proj.weight, scheme, group)),
                q_norm=block.attn.q_norm,
                k_norm=block.attn.k_norm,
            ),
            norm2=block.norm2,
            mlp=(;
                gate_proj=(; weight=_quantize_linear(block.mlp.gate_proj.weight, scheme, group)),
                up_proj=(; weight=_quantize_linear(block.mlp.up_proj.weight, scheme, group)),
                down_proj=(; weight=_quantize_linear(block.mlp.down_proj.weight, scheme, group)),
            ),
        )
    end
    lm_head = if haskey(ps, :lm_head) && haskey(ps.lm_head, :weight)
        (; weight=_quantize_linear(ps.lm_head.weight, scheme, group))
    else
        ps.lm_head
    end
    return (;
        token_embedding=ps.token_embedding,
        blocks=NamedTuple{keys(ps.blocks)}(blocks),
        final_norm=ps.final_norm,
        lm_head,
    )
end
