using JSON3
using Lux
using Random: Xoshiro

const _SAFETENSORS_MAX_HEADER_BYTES = 100_000_000
const _SAFETENSORS_DTYPES = Dict("BF16" => 2, "F32" => 4)

"""
    Qwen3DenseSpec

Frozen architecture and provenance metadata for one official Qwen3 dense
checkpoint family member. `max_position_embeddings` is the checkpoint's native
RoPE limit; a loader may still choose a smaller runtime `max_seq_len`.
"""
struct Qwen3DenseSpec
    variant::Symbol
    model_id::String
    revision::String
    config_sha256::String
    vocab_size::Int
    d_model::Int
    mlp_hidden_dim::Int
    num_layers::Int
    num_heads::Int
    num_kv_heads::Int
    head_dim::Int
    rms_norm_epsilon::Float32
    rope_theta::Float32
    max_position_embeddings::Int
    tie_embeddings::Bool
end

const _QWEN3_DENSE_SPECS = (
    Qwen3DenseSpec(
        :qwen3_0_6b,
        "Qwen/Qwen3-0.6B",
        "c1899de289a04d12100db370d81485cdf75e47ca",
        "660db3b73d788119c04535e48cf9be5f55bc3100841a718637ae695b442f27dd",
        151_936,
        1_024,
        3_072,
        28,
        16,
        8,
        128,
        1.0f-6,
        1.0f6,
        40_960,
        true,
    ),
    Qwen3DenseSpec(
        :qwen3_1_7b,
        "Qwen/Qwen3-1.7B",
        "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e",
        "1ddb5b89ebc90dcb417a45c213d818577e65976454d29385c8f6140771d95197",
        151_936,
        2_048,
        6_144,
        28,
        16,
        8,
        128,
        1.0f-6,
        1.0f6,
        40_960,
        true,
    ),
    Qwen3DenseSpec(
        :qwen3_4b,
        "Qwen/Qwen3-4B",
        "1cfa9a7208912126459214e8b04321603b3df60c",
        "8ba006f74fecfaaeb392872a60f4a480e7ec9860153d2e1b769ec81f9a147f8a",
        151_936,
        2_560,
        9_728,
        36,
        32,
        8,
        128,
        1.0f-6,
        1.0f6,
        40_960,
        true,
    ),
    Qwen3DenseSpec(
        :qwen3_8b,
        "Qwen/Qwen3-8B",
        "b968826d9c46dd6066d109eabc6255188de91218",
        "f7c4eadfbbf522470667b797a3c89be2524832d2d599797248dc304fff447c30",
        151_936,
        4_096,
        12_288,
        36,
        32,
        8,
        128,
        1.0f-6,
        1.0f6,
        40_960,
        false,
    ),
    Qwen3DenseSpec(
        :qwen3_14b,
        "Qwen/Qwen3-14B",
        "40c069824f4251a91eefaf281ebe4c544efd3e18",
        "e73c3664ca09b10a673fef0c22e8a6b456201d49bd4713c9691f775720e8857a",
        151_936,
        5_120,
        17_408,
        40,
        40,
        8,
        128,
        1.0f-6,
        1.0f6,
        40_960,
        false,
    ),
    Qwen3DenseSpec(
        :qwen3_32b,
        "Qwen/Qwen3-32B",
        "9216db5781bf21249d130ec9da846c4624c16137",
        "97e295b63283935788fac5e4f8860862a56d4089538cafc93f0431f2ebe483bb",
        151_936,
        5_120,
        25_600,
        64,
        64,
        8,
        128,
        1.0f-6,
        1.0f6,
        40_960,
        false,
    ),
)

"""Return the frozen Week 11 specifications for all six official Qwen3 dense sizes."""
qwen3_dense_specs() = _QWEN3_DENSE_SPECS

function _qwen3_variant_key(value)
    if value isa Symbol
        any(spec -> spec.variant === value, _QWEN3_DENSE_SPECS) && return value
    end
    text = lowercase(String(value))
    text = replace(text, "qwen/" => "", "qwen3-" => "", "." => "_", "-" => "_")
    startswith(text, "qwen3_") && (text = text[7:end])
    return Symbol("qwen3_" * text)
end

"""
    qwen3_dense_spec(variant)

Look up an official dense-family specification by canonical symbol, model id,
or short size such as `"4B"`.
"""
function qwen3_dense_spec(variant::Union{Symbol,AbstractString})
    key = _qwen3_variant_key(variant)
    for spec in _QWEN3_DENSE_SPECS
        spec.variant === key && return spec
    end
    supported = join((String(spec.variant) for spec in _QWEN3_DENSE_SPECS), ", ")
    throw(ArgumentError(
        "unknown Qwen3 dense variant $(repr(variant)); supported variants: $supported",
    ))
end

function _qwen3_dense_spec(
    vocab_size,
    d_model,
    mlp_hidden_dim,
    num_layers,
    num_heads,
    num_kv_heads,
    head_dim,
    rms_norm_epsilon,
    rope_theta,
    max_position_embeddings,
    tie_embeddings,
)
    for spec in _QWEN3_DENSE_SPECS
        values = (
            spec.vocab_size,
            spec.d_model,
            spec.mlp_hidden_dim,
            spec.num_layers,
            spec.num_heads,
            spec.num_kv_heads,
            spec.head_dim,
            spec.rms_norm_epsilon,
            spec.rope_theta,
            spec.max_position_embeddings,
            spec.tie_embeddings,
        )
        values == (
            vocab_size,
            d_model,
            mlp_hidden_dim,
            num_layers,
            num_heads,
            num_kv_heads,
            head_dim,
            Float32(rms_norm_epsilon),
            Float32(rope_theta),
            max_position_embeddings,
            tie_embeddings,
        ) && return spec
    end
    return nothing
end

"""
    qwen3_dense_parameter_count(spec)

Return the exact number of trainable scalar parameters implied by an official
Qwen3 dense specification, including an untied LM head when present.
"""
function qwen3_dense_parameter_count(spec::Qwen3DenseSpec)
    query_dim = spec.num_heads * spec.head_dim
    kv_dim = spec.num_kv_heads * spec.head_dim
    embedding = spec.vocab_size * spec.d_model
    attention = 2 * query_dim * spec.d_model +
        2 * kv_dim * spec.d_model +
        2 * spec.head_dim
    mlp = 3 * spec.d_model * spec.mlp_hidden_dim
    norms = 2 * spec.d_model
    final_norm = spec.d_model
    lm_head = spec.tie_embeddings ? 0 : embedding
    return embedding +
        spec.num_layers * (attention + mlp + norms) +
        final_norm +
        lm_head
end

function _json_object(path::AbstractString)
    isfile(path) || throw(ArgumentError("JSON file does not exist: $path"))
    value = try
        JSON3.read(read(path, String))
    catch err
        throw(ArgumentError("invalid JSON in $path: $(sprint(showerror, err))"))
    end
    value isa JSON3.Object || throw(ArgumentError("JSON root must be an object: $path"))
    return value
end

function _json_required(object, name::AbstractString, path::AbstractString)
    haskey(object, name) || throw(ArgumentError("missing required field `$name` in $path"))
    return object[name]
end

function _required_int(object, name::AbstractString, path::AbstractString)
    value = _json_required(object, name, path)
    value isa Integer || throw(ArgumentError("`$name` must be an integer in $path"))
    value > 0 || throw(ArgumentError("`$name` must be positive in $path"))
    return Int(value)
end

function _required_bool(object, name::AbstractString, path::AbstractString)
    value = _json_required(object, name, path)
    value isa Bool || throw(ArgumentError("`$name` must be a boolean in $path"))
    return value
end

"""
    load_hf_qwen3_config(path; max_seq_len=nothing, variant=nothing)

Parse and validate a HuggingFace Qwen3 dense `config.json`, returning the
version-stable `NamedTuple` accepted by `GPTModel(config)`. When `variant` is
provided, all architecture-defining fields must exactly match that frozen
official dense-family member. Without it, compatible custom dense configs
remain supported and `qwen3_variant` is `nothing`.
"""
function load_hf_qwen3_config(
    path::AbstractString;
    max_seq_len=nothing,
    variant=nothing,
)
    config = _json_object(path)

    model_type = _json_required(config, "model_type", path)
    model_type == "qwen3" || throw(ArgumentError(
        "unsupported HuggingFace model_type $(repr(model_type)); expected `qwen3`",
    ))

    if haskey(config, "architectures")
        architectures = config["architectures"]
        architectures isa JSON3.Array || throw(ArgumentError(
            "`architectures` must be an array in $path",
        ))
        "Qwen3ForCausalLM" in architectures || throw(ArgumentError(
            "config does not declare `Qwen3ForCausalLM` in $path",
        ))
    end

    hidden_act = _json_required(config, "hidden_act", path)
    hidden_act == "silu" || throw(ArgumentError(
        "unsupported Qwen3 hidden_act $(repr(hidden_act)); expected `silu`",
    ))

    attention_bias = _required_bool(config, "attention_bias", path)
    attention_bias && throw(ArgumentError("Qwen3 attention bias is not supported"))

    attention_dropout = _json_required(config, "attention_dropout", path)
    attention_dropout isa Real || throw(ArgumentError(
        "`attention_dropout` must be numeric in $path",
    ))
    iszero(attention_dropout) || throw(ArgumentError(
        "non-zero attention dropout is not supported",
    ))

    use_sliding_window = haskey(config, "use_sliding_window") ?
        config["use_sliding_window"] : false
    use_sliding_window isa Bool || throw(ArgumentError(
        "`use_sliding_window` must be a boolean in $path",
    ))
    use_sliding_window && throw(ArgumentError("sliding-window attention is not supported"))
    if haskey(config, "sliding_window")
        sliding_window = config["sliding_window"]
        sliding_window === nothing || throw(ArgumentError(
            "sliding-window attention is not supported",
        ))
    end
    if haskey(config, "rope_scaling")
        config["rope_scaling"] === nothing || throw(ArgumentError(
            "RoPE scaling is not supported",
        ))
    end

    vocab_size = _required_int(config, "vocab_size", path)
    d_model = _required_int(config, "hidden_size", path)
    num_layers = _required_int(config, "num_hidden_layers", path)
    num_heads = _required_int(config, "num_attention_heads", path)
    num_kv_heads = _required_int(config, "num_key_value_heads", path)
    head_dim = _required_int(config, "head_dim", path)
    mlp_hidden_dim = _required_int(config, "intermediate_size", path)
    max_positions = _required_int(config, "max_position_embeddings", path)
    num_heads % num_kv_heads == 0 || throw(ArgumentError(
        "num_attention_heads must be divisible by num_key_value_heads",
    ))
    iseven(head_dim) || throw(ArgumentError("Qwen3 head_dim must be even for RoPE"))

    resolved_max_seq_len = max_seq_len === nothing ? max_positions : Int(max_seq_len)
    1 <= resolved_max_seq_len <= max_positions || throw(ArgumentError(
        "max_seq_len must be in 1:$max_positions; got $resolved_max_seq_len",
    ))

    rms_norm_eps = _json_required(config, "rms_norm_eps", path)
    rms_norm_eps isa Real && rms_norm_eps > 0 || throw(ArgumentError(
        "`rms_norm_eps` must be positive in $path",
    ))
    rope_theta = _json_required(config, "rope_theta", path)
    rope_theta isa Real && rope_theta > 0 || throw(ArgumentError(
        "`rope_theta` must be positive in $path",
    ))
    tie_embeddings = _required_bool(config, "tie_word_embeddings", path)
    dense_spec = _qwen3_dense_spec(
        vocab_size,
        d_model,
        mlp_hidden_dim,
        num_layers,
        num_heads,
        num_kv_heads,
        head_dim,
        rms_norm_eps,
        rope_theta,
        max_positions,
        tie_embeddings,
    )
    if variant !== nothing
        expected = qwen3_dense_spec(variant)
        dense_spec === expected || throw(ArgumentError(
            "Qwen3 config in $path does not match requested variant " *
            "$(expected.variant)",
        ))
    end

    return (;
        vocab_size,
        d_model,
        num_heads,
        num_kv_heads,
        num_layers,
        head_dim,
        mlp_hidden_dim,
        use_bias=false,
        is_causal=true,
        use_rope=true,
        use_qk_norm=true,
        qk_norm_epsilon=Float32(rms_norm_eps),
        max_seq_len=resolved_max_seq_len,
        rope_theta=Float32(rope_theta),
        rope_style=:rotate_half,
        norm_epsilon=Float32(rms_norm_eps),
        norm_type=:rmsnorm,
        mlp_type=:swiglu,
        tie_embeddings,
        source_max_seq_len=max_positions,
        qwen3_variant=dense_spec === nothing ? nothing : dense_spec.variant,
    )
end

function _read_le_uint64(io)
    bytes = read(io, 8)
    length(bytes) == 8 || throw(ArgumentError("safetensors file is shorter than 8 bytes"))
    value = zero(UInt64)
    @inbounds for i in 1:8
        value |= UInt64(bytes[i]) << (8 * (i - 1))
    end
    return value
end

function _checked_element_count(shape::Vector{Int}, name::AbstractString)
    count = 1
    for dim in shape
        dim >= 0 || throw(ArgumentError("negative dimension in safetensors tensor `$name`"))
        count = try
            Base.Checked.checked_mul(count, dim)
        catch
            throw(ArgumentError("shape product overflows for safetensors tensor `$name`"))
        end
    end
    return count
end

function _semantic_array(values::Vector{Float32}, shape::Vector{Int})
    isempty(shape) && return reshape(values, ())
    length(shape) == 1 && return reshape(values, shape[1])
    stored = reshape(values, Tuple(reverse(shape)))
    return permutedims(stored, Tuple(reverse(1:length(shape))))
end

function _decode_safetensors_values(
    raw::Vector{UInt8},
    dtype::String,
    shape::Vector{Int},
)
    values = if dtype == "F32"
        copy(reinterpret(Float32, raw))
    elseif dtype == "BF16"
        bits = reinterpret(UInt16, raw)
        [reinterpret(Float32, UInt32(value) << 16) for value in bits]
    else
        throw(ArgumentError("unsupported safetensors dtype `$dtype`"))
    end
    return _semantic_array(values, shape)
end

function _safetensors_entries(path::AbstractString)
    file_bytes = filesize(path)
    file_bytes >= 8 || throw(ArgumentError("safetensors file is shorter than 8 bytes: $path"))

    return open(path, "r") do io
        header_length_u64 = _read_le_uint64(io)
        header_length_u64 <= UInt64(typemax(Int)) || throw(ArgumentError(
            "safetensors header length does not fit in Int: $path",
        ))
        header_length = Int(header_length_u64)
        2 <= header_length <= _SAFETENSORS_MAX_HEADER_BYTES || throw(ArgumentError(
            "invalid safetensors header length $header_length in $path",
        ))
        8 + header_length <= file_bytes || throw(ArgumentError(
            "safetensors header exceeds file size: $path",
        ))

        header_bytes = read(io, header_length)
        length(header_bytes) == header_length || throw(ArgumentError(
            "truncated safetensors header: $path",
        ))
        header = try
            JSON3.read(String(header_bytes))
        catch err
            throw(ArgumentError(
                "invalid safetensors JSON header in $path: $(sprint(showerror, err))",
            ))
        end
        header isa JSON3.Object || throw(ArgumentError(
            "safetensors header root must be an object: $path",
        ))

        entries = NamedTuple[]
        seen_names = Set{String}()
        for raw_name in keys(header)
            name = String(raw_name)
            name == "__metadata__" && continue
            name in seen_names && throw(ArgumentError(
                "duplicate safetensors tensor name `$name` in $path",
            ))
            push!(seen_names, name)
            info = header[raw_name]
            info isa JSON3.Object || throw(ArgumentError(
                "metadata for safetensors tensor `$name` must be an object",
            ))

            dtype_raw = _json_required(info, "dtype", path)
            dtype_raw isa AbstractString || throw(ArgumentError(
                "dtype for safetensors tensor `$name` must be a string",
            ))
            dtype = String(dtype_raw)
            haskey(_SAFETENSORS_DTYPES, dtype) || throw(ArgumentError(
                "unsupported safetensors dtype `$dtype` for tensor `$name`",
            ))

            shape_raw = _json_required(info, "shape", path)
            shape_raw isa JSON3.Array || throw(ArgumentError(
                "shape for safetensors tensor `$name` must be an array",
            ))
            all(dim -> dim isa Integer, shape_raw) || throw(ArgumentError(
                "shape for safetensors tensor `$name` must contain integers",
            ))
            shape = Int.(collect(shape_raw))
            element_count = _checked_element_count(shape, name)

            offsets = _json_required(info, "data_offsets", path)
            offsets isa JSON3.Array && length(offsets) == 2 || throw(ArgumentError(
                "data_offsets for safetensors tensor `$name` must have length two",
            ))
            all(offset -> offset isa Integer, offsets) || throw(ArgumentError(
                "data_offsets for safetensors tensor `$name` must contain integers",
            ))
            data_start, data_stop = Int(offsets[1]), Int(offsets[2])
            0 <= data_start <= data_stop || throw(ArgumentError(
                "invalid data_offsets for safetensors tensor `$name`",
            ))
            expected_bytes = try
                Base.Checked.checked_mul(element_count, _SAFETENSORS_DTYPES[dtype])
            catch
                throw(ArgumentError("byte length overflows for safetensors tensor `$name`"))
            end
            data_stop - data_start == expected_bytes || throw(ArgumentError(
                "byte length does not match dtype/shape for safetensors tensor `$name`",
            ))
            push!(entries, (; name, dtype, shape, data_start, data_stop))
        end

        data_bytes = file_bytes - 8 - header_length
        cursor = 0
        for entry in sort(entries; by=entry -> (entry.data_start, entry.data_stop))
            entry.data_start == cursor || throw(ArgumentError(
                "safetensors data offsets contain a gap or overlap before `$(entry.name)`",
            ))
            entry.data_stop <= data_bytes || throw(ArgumentError(
                "safetensors tensor `$(entry.name)` exceeds file size",
            ))
            cursor = entry.data_stop
        end
        cursor == data_bytes || throw(ArgumentError(
            "safetensors data section contains unindexed bytes: $path",
        ))
        return entries, 8 + header_length
    end
end

function _load_safetensors_file(
    path::AbstractString;
    target_dtype::Type=Float32,
)
    target_dtype === Float32 || throw(ArgumentError(
        "Week 07 safetensors loading only supports target_dtype=Float32",
    ))
    isfile(path) || throw(ArgumentError("safetensors file does not exist: $path"))
    entries, data_base = _safetensors_entries(path)
    tensors = Dict{String,Any}()

    open(path, "r") do io
        for entry in entries
            seek(io, data_base + entry.data_start)
            raw = read(io, entry.data_stop - entry.data_start)
            length(raw) == entry.data_stop - entry.data_start || throw(ArgumentError(
                "truncated data for safetensors tensor `$(entry.name)`",
            ))
            tensors[entry.name] = _decode_safetensors_values(
                raw,
                entry.dtype,
                entry.shape,
            )
        end
    end
    return tensors
end

function _safe_shard_path(root::AbstractString, shard::AbstractString)
    isabspath(shard) && throw(ArgumentError("safetensors shard path must be relative"))
    path = normpath(joinpath(root, shard))
    relative = relpath(path, root)
    (relative == ".." || startswith(relative, "..$(Base.Filesystem.path_separator)")) &&
        throw(ArgumentError("safetensors shard path escapes the model directory"))
    return path
end

function _load_safetensors_index(
    path::AbstractString;
    target_dtype::Type=Float32,
)
    index = _json_object(path)
    weight_map_raw = _json_required(index, "weight_map", path)
    weight_map_raw isa JSON3.Object || throw(ArgumentError(
        "safetensors index `weight_map` must be an object: $path",
    ))
    weight_map = Dict{String,String}()
    for raw_name in keys(weight_map_raw)
        name = String(raw_name)
        haskey(weight_map, name) && throw(ArgumentError(
            "duplicate tensor `$name` in safetensors index",
        ))
        shard_raw = weight_map_raw[raw_name]
        shard_raw isa AbstractString || throw(ArgumentError(
            "safetensors shard name for `$name` must be a string",
        ))
        weight_map[name] = String(shard_raw)
    end
    isempty(weight_map) && throw(ArgumentError("safetensors index weight_map is empty"))

    root = dirname(abspath(path))
    merged = Dict{String,Any}()
    for shard in sort!(unique!(collect(values(weight_map))))
        shard_path = _safe_shard_path(root, shard)
        shard_tensors = _load_safetensors_file(shard_path; target_dtype)
        for (name, tensor) in shard_tensors
            get(weight_map, name, nothing) == shard || throw(ArgumentError(
                "tensor `$name` is stored in `$shard` but the index assigns a different shard",
            ))
            haskey(merged, name) && throw(ArgumentError(
                "duplicate tensor `$name` across safetensors shards",
            ))
            merged[name] = tensor
        end
    end
    Set(keys(merged)) == Set(keys(weight_map)) || throw(ArgumentError(
        "safetensors index contains missing or unindexed tensors",
    ))
    return merged
end

"""
    load_safetensors(path; target_dtype=Float32)

Strictly load BF16/F32 tensors from one `.safetensors` file, a
`model.safetensors.index.json`, or a model directory containing either form.
Arrays are returned with their declared semantic shape in a string-keyed Dict.
"""
function load_safetensors(
    path::AbstractString;
    target_dtype::Type=Float32,
)
    resolved = if isdir(path)
        single = joinpath(path, "model.safetensors")
        index = joinpath(path, "model.safetensors.index.json")
        if isfile(single)
            single
        elseif isfile(index)
            index
        else
            throw(ArgumentError(
                "model directory contains neither model.safetensors nor model.safetensors.index.json: $path",
            ))
        end
    else
        path
    end

    endswith(resolved, ".index.json") && return _load_safetensors_index(
        resolved;
        target_dtype,
    )
    return _load_safetensors_file(resolved; target_dtype)
end

function _expect_tensor(
    tensors::AbstractDict,
    name::String,
    expected_shape::Tuple,
)
    haskey(tensors, name) || throw(ArgumentError("missing HuggingFace tensor `$name`"))
    tensor = tensors[name]
    tensor isa AbstractArray || throw(ArgumentError("HuggingFace tensor `$name` is not an array"))
    size(tensor) == expected_shape || throw(DimensionMismatch(
        "HuggingFace tensor `$name` has shape $(size(tensor)); expected $expected_shape",
    ))
    eltype(tensor) === Float32 || throw(ArgumentError(
        "HuggingFace tensor `$name` must contain Float32 values after loading",
    ))
    return tensor
end

function _qwen3_expected_tensor_names(model::GPTModel)
    names = Set(["model.embed_tokens.weight", "model.norm.weight"])
    for layer in 0:(model.num_layers - 1)
        prefix = "model.layers.$layer"
        union!(names, [
            "$prefix.input_layernorm.weight",
            "$prefix.self_attn.q_proj.weight",
            "$prefix.self_attn.k_proj.weight",
            "$prefix.self_attn.v_proj.weight",
            "$prefix.self_attn.o_proj.weight",
            "$prefix.self_attn.q_norm.weight",
            "$prefix.self_attn.k_norm.weight",
            "$prefix.post_attention_layernorm.weight",
            "$prefix.mlp.gate_proj.weight",
            "$prefix.mlp.up_proj.weight",
            "$prefix.mlp.down_proj.weight",
        ])
    end
    model.tie_embeddings || push!(names, "lm_head.weight")
    return names
end

function _format_tensor_names(names)
    values = sort!(collect(names))
    length(values) <= 8 && return join(values, ", ")
    return join(values[1:8], ", ") * ", … ($(length(values)) total)"
end

"""
    load_hf_qwen3_parameters(model, tensors)

Map a complete, Float32 HuggingFace Qwen3 state dict into the Lux parameter
tree. Missing and unexpected tensors are rejected before any tree is returned.
"""
function load_hf_qwen3_parameters(
    model::GPTModel,
    tensors::AbstractDict,
)
    model.norm_type === :rmsnorm || throw(ArgumentError("Qwen3 requires RMSNorm"))
    model.mlp_type === :swiglu || throw(ArgumentError("Qwen3 requires SwiGLU"))
    model.use_qk_norm || throw(ArgumentError("Qwen3 requires QK-Norm"))
    model.use_rope && model.rope_style === :rotate_half || throw(ArgumentError(
        "Qwen3 requires rotate_half RoPE",
    ))
    model.use_bias && throw(ArgumentError("Qwen3 weight loading requires bias-free projections"))

    expected_names = _qwen3_expected_tensor_names(model)
    actual_names = Set(String.(collect(keys(tensors))))
    allowed_names = model.tie_embeddings ?
        union(expected_names, Set(["lm_head.weight"])) : expected_names
    missing = setdiff(expected_names, actual_names)
    unexpected = setdiff(actual_names, allowed_names)
    isempty(missing) || throw(ArgumentError(
        "missing HuggingFace tensors: $(_format_tensor_names(missing))",
    ))
    isempty(unexpected) || throw(ArgumentError(
        "unexpected HuggingFace tensors: $(_format_tensor_names(unexpected))",
    ))

    d_model = model.d_model
    q_dim = model.num_heads * model.head_dim
    kv_dim = model.num_kv_heads * model.head_dim
    hidden_dim = model.mlp_hidden_dim

    embedding_hf = _expect_tensor(
        tensors,
        "model.embed_tokens.weight",
        (model.vocab_size, d_model),
    )
    if model.tie_embeddings && haskey(tensors, "lm_head.weight")
        tied_head = _expect_tensor(
            tensors,
            "lm_head.weight",
            (model.vocab_size, d_model),
        )
        tied_head == embedding_hf || throw(ArgumentError(
            "tied Qwen3 lm_head.weight does not equal model.embed_tokens.weight",
        ))
    end
    token_embedding = (; weight=permutedims(embedding_hf, (2, 1)))

    block_values = ntuple(model.num_layers) do julia_layer
        layer = julia_layer - 1
        prefix = "model.layers.$layer"
        norm1 = (; scale=reshape(_expect_tensor(
            tensors,
            "$prefix.input_layernorm.weight",
            (d_model,),
        ), d_model, 1, 1))
        attn = (;
            q_proj=(; weight=_expect_tensor(
                tensors,
                "$prefix.self_attn.q_proj.weight",
                (q_dim, d_model),
            )),
            k_proj=(; weight=_expect_tensor(
                tensors,
                "$prefix.self_attn.k_proj.weight",
                (kv_dim, d_model),
            )),
            v_proj=(; weight=_expect_tensor(
                tensors,
                "$prefix.self_attn.v_proj.weight",
                (kv_dim, d_model),
            )),
            o_proj=(; weight=_expect_tensor(
                tensors,
                "$prefix.self_attn.o_proj.weight",
                (d_model, q_dim),
            )),
            q_norm=(; scale=_expect_tensor(
                tensors,
                "$prefix.self_attn.q_norm.weight",
                (model.head_dim,),
            )),
            k_norm=(; scale=_expect_tensor(
                tensors,
                "$prefix.self_attn.k_norm.weight",
                (model.head_dim,),
            )),
        )
        norm2 = (; scale=reshape(_expect_tensor(
            tensors,
            "$prefix.post_attention_layernorm.weight",
            (d_model,),
        ), d_model, 1, 1))
        mlp = (;
            gate_proj=(; weight=_expect_tensor(
                tensors,
                "$prefix.mlp.gate_proj.weight",
                (hidden_dim, d_model),
            )),
            up_proj=(; weight=_expect_tensor(
                tensors,
                "$prefix.mlp.up_proj.weight",
                (hidden_dim, d_model),
            )),
            down_proj=(; weight=_expect_tensor(
                tensors,
                "$prefix.mlp.down_proj.weight",
                (d_model, hidden_dim),
            )),
        )
        return (; norm1, attn, norm2, mlp)
    end
    block_names = Tuple(Symbol("layer_$layer") for layer in 1:model.num_layers)
    blocks = NamedTuple{block_names}(block_values)

    final_norm = (; scale=reshape(_expect_tensor(
        tensors,
        "model.norm.weight",
        (d_model,),
    ), d_model, 1, 1))
    lm_head = if model.tie_embeddings
        (;)
    else
        (; weight=_expect_tensor(
            tensors,
            "lm_head.weight",
            (model.vocab_size, d_model),
        ))
    end
    return (; token_embedding, blocks, final_norm, lm_head)
end

"""
    hf_token_ids(ids; vocab_size=nothing)

Convert HuggingFace's 0-based token ids to LifeAI's public 1-based token ids.
The array shape is preserved.
"""
function hf_token_ids(ids::AbstractArray{T}; vocab_size=nothing) where {T<:Integer}
    all(id -> id >= 0, ids) || throw(ArgumentError("HuggingFace token ids must be non-negative"))
    if vocab_size !== nothing
        resolved_vocab_size = Int(vocab_size)
        resolved_vocab_size > 0 || throw(ArgumentError("vocab_size must be positive"))
        all(id -> id < resolved_vocab_size, ids) || throw(ArgumentError(
            "HuggingFace token id is outside 0:$(resolved_vocab_size - 1)",
        ))
    end
    return Int.(ids) .+ 1
end

"""
    load_hf_qwen3_model(
        model_dir;
        max_seq_len=2048,
        weight_dtype=Float32,
        variant=nothing,
    )

Load a local HuggingFace Qwen3 dense model directory into a `GPTModel`, Lux
parameter tree, and Lux state tree. This function never downloads files.
Pass `variant` to require an exact official Week 11 dense-family shape.
"""
function load_hf_qwen3_model(
    model_dir::AbstractString;
    max_seq_len=2048,
    weight_dtype::Type=Float32,
    variant=nothing,
)
    isdir(model_dir) || throw(ArgumentError("model directory does not exist: $model_dir"))
    config = load_hf_qwen3_config(
        joinpath(model_dir, "config.json");
        max_seq_len,
        variant,
    )
    model = GPTModel(config)
    tensors = load_safetensors(model_dir; target_dtype=weight_dtype)
    parameters = load_hf_qwen3_parameters(model, tensors)
    # Parameters reuse every linear/norm array, while the tied HF checkpoint may
    # also contain two large source matrices that were validated and copied into
    # the LifeAI embedding layout. Drop the state-dict container before states
    # are constructed so those source-only arrays can be reclaimed.
    empty!(tensors)
    GC.gc(false)
    states = Lux.initialstates(Xoshiro(0), model)
    dense_spec = config.qwen3_variant === nothing ?
        nothing : qwen3_dense_spec(config.qwen3_variant)
    return (;
        model,
        parameters,
        states,
        config,
        variant=dense_spec,
        source=abspath(model_dir),
    )
end

"""
    load_hf_qwen3_bundle(
        model_dir;
        max_seq_len=2048,
        revision="",
        variant=nothing,
        ...,
    )

Load a local Qwen3 model and its exact HuggingFace tokenizer files as one
text-generation bundle. The tokenizer may define fewer ids than the padded
embedding vocabulary, but every defined tokenizer id must fit the model.
"""
function load_hf_qwen3_bundle(
    model_dir::AbstractString;
    max_seq_len=2048,
    weight_dtype::Type=Float32,
    revision::AbstractString="",
    variant=nothing,
)
    tokenizer = load_hf_qwen3_tokenizer(model_dir; revision)
    loaded = load_hf_qwen3_model(
        model_dir;
        max_seq_len,
        weight_dtype,
        variant,
    )
    vocab_size(tokenizer) <= loaded.model.vocab_size || throw(ArgumentError(
        "tokenizer vocabulary exceeds the Qwen3 model embedding vocabulary",
    ))
    generation_config = hf_generation_config(tokenizer)
    return merge(loaded, (; tokenizer, generation_config))
end

"""
    hf_qwen3_forward_trace(model, tokens, ps, st)

Run an eager Qwen3 forward pass while returning the embedding output, every
block residual output, final normalized hidden state, and logits. Intended for
offline HuggingFace parity fixtures.
"""
function hf_qwen3_forward_trace(
    model::GPTModel,
    tokens,
    ps,
    st::NamedTuple,
)
    @assert ndims(tokens) == 2 "`tokens` must have shape (seq_len, batch)"
    _validate_token_ids(tokens, model.vocab_size)
    x, st_embedding = model.token_embedding(
        tokens,
        ps.token_embedding,
        st.token_embedding,
    )
    x = _add_position_embedding(model, x, ps, 1)
    embedding = x
    block_outputs = Any[]
    state_values = Any[]
    block_names = keys(model.blocks.layers)
    for name in block_names
        block = getproperty(model.blocks.layers, name)
        x, st_block = block(
            x,
            getproperty(ps.blocks, name),
            getproperty(st.blocks, name),
        )
        push!(block_outputs, x)
        push!(state_values, st_block)
    end
    st_blocks = NamedTuple{Tuple(block_names)}(Tuple(state_values))
    final_hidden, st_final = model.final_norm(x, ps.final_norm, st.final_norm)
    logits, st_lm = _project_logits(model, final_hidden, ps, st.lm_head)
    states = (;
        token_embedding=st_embedding,
        blocks=st_blocks,
        final_norm=st_final,
        lm_head=st_lm,
    )
    return (;
        embedding,
        blocks=Tuple(block_outputs),
        final_hidden,
        logits,
        states,
    )
end
