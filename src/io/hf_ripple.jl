"""Ricursive Ripple metadata. Config special-token IDs are zero-based; inference IDs are one-based."""
struct RippleConfig
    vocab_size::Int
    hidden_size::Int
    intermediate_size::Int
    num_hidden_layers::Int
    num_attention_heads::Int
    num_key_value_heads::Int
    head_dim::Int
    max_position_embeddings::Int
    rms_norm_eps::Float32
    rope_theta::Float64
    sliding_window::Int
    layer_types::Vector{String}
    use_gated_attention::Bool
    gate_full_precision::Bool
    use_valres::Bool
    ffn_clamp_threshold::Union{Nothing,Float32}
    embed_scale::Union{Nothing,Float32}
    tie_word_embeddings::Bool
    bos_token_id::Int
    eos_token_id::Vector{Int}
    pad_token_id::Int
end

function RippleConfig(; vocab_size=100352, hidden_size=5120, intermediate_size=40960,
    num_hidden_layers=80, num_attention_heads=80, num_key_value_heads=8, head_dim=128,
    max_position_embeddings=2097152, rms_norm_eps=1f-6, rope_theta=10000.0,
    sliding_window=1024, layer_types=nothing, use_gated_attention=true,
    gate_full_precision=true, use_valres=true, ffn_clamp_threshold=20f0,
    embed_scale=sqrt(Float64(hidden_size)), tie_word_embeddings=false,
    bos_token_id=100257, eos_token_id=[100265, 100257], pad_token_id=100277)
    names = (:vocab_size, :hidden_size, :intermediate_size, :num_hidden_layers,
        :num_attention_heads, :num_key_value_heads, :head_dim, :max_position_embeddings,
        :sliding_window)
    values = (vocab_size, hidden_size, intermediate_size, num_hidden_layers,
        num_attention_heads, num_key_value_heads, head_dim, max_position_embeddings,
        sliding_window)
    sizes = map(zip(names, values)) do (name, value)
        n = _hf_strict_host_int(value, "Ripple $name")
        n > 0 || throw(ArgumentError("Ripple $name must be positive"))
        n
    end
    vocab_size, hidden_size, intermediate_size, num_hidden_layers,
        num_attention_heads, num_key_value_heads, head_dim, max_position_embeddings,
        sliding_window = sizes
    num_attention_heads % num_key_value_heads == 0 ||
        throw(ArgumentError("Ripple query heads must be divisible by KV heads"))
    iseven(head_dim) || throw(ArgumentError("Ripple head_dim must be even"))
    for (name, value) in ((:rms_norm_eps, rms_norm_eps), (:rope_theta, rope_theta),
            (:ffn_clamp_threshold, ffn_clamp_threshold), (:embed_scale, embed_scale))
        value === nothing && name in (:ffn_clamp_threshold, :embed_scale) && continue
        value isa Real && !(value isa Bool) && isfinite(value) && value > 0 ||
            throw(ArgumentError("Ripple $name must be finite and positive, or an optional value must be nothing"))
        name === :rope_theta || (isfinite(Float32(value)) && Float32(value) > 0) ||
            throw(ArgumentError("Ripple $name must be representable in Float32"))
    end
    all(x -> x isa Bool, (use_gated_attention, gate_full_precision, use_valres, tie_word_embeddings)) ||
        throw(ArgumentError("Ripple boolean options must be Bool"))
    types = layer_types === nothing ?
        [i % 5 == 0 ? "full_attention" : "sliding_attention" for i in 1:num_hidden_layers] : String.(layer_types)
    length(types) == num_hidden_layers && all(t -> t in ("full_attention", "sliding_attention"), types) ||
        throw(ArgumentError("Ripple layer_types must specify one supported type per layer"))
    eos = eos_token_id isa Integer ? [eos_token_id] : collect(eos_token_id)
    eos = [_hf_zero_based_token_id(id, "Ripple eos_token_id") for id in eos]
    bos = _hf_zero_based_token_id(bos_token_id, "Ripple bos_token_id")
    pad = _hf_zero_based_token_id(pad_token_id, "Ripple pad_token_id")
    !isempty(eos) && all(id -> id < vocab_size, [bos; pad; eos]) ||
        throw(ArgumentError("Ripple special token IDs must be in the vocabulary"))
    return RippleConfig(sizes[1:8]..., Float32(rms_norm_eps), Float64(rope_theta),
        sliding_window, types, use_gated_attention, gate_full_precision, use_valres,
        ffn_clamp_threshold === nothing ? nothing : Float32(ffn_clamp_threshold),
        embed_scale === nothing ? nothing : Float32(embed_scale), tie_word_embeddings, bos, eos, pad)
end

function ripple_config_dict(config::RippleConfig)
    result = Dict{String,Any}(String(k) => getfield(config, k) for k in fieldnames(RippleConfig))
    merge!(result, Dict("model_type" => "ricursive", "architectures" => ["RicursiveForCausalLM"],
        "hidden_act" => "silu", "hybrid_layer_pattern" => Int.(config.layer_types .== "sliding_attention"),
        "is_hybrid_swa" => length(unique(config.layer_types)) == 2))
    return result
end

"""Read the released Ripple architecture, rejecting unsupported quantized checkpoints and RoPE variants."""
function load_hf_ripple_config(path::AbstractString)
    file = isdir(path) ? joinpath(path, "config.json") : path
    object = _hf_json(read(file, String), "Ripple config.json")
    _hf_reject_duplicate_fields(object, "Ripple config.json")
    get(object, "model_type", nothing) == "ricursive" || throw(ArgumentError("expected model_type=ricursive"))
    get(object, "hidden_act", "silu") == "silu" || throw(ArgumentError("Ripple requires silu"))
    get(object, "attention_bias", false) === false || throw(ArgumentError("Ripple attention bias is unsupported"))
    get(object, "rope_scaling", nothing) === nothing || throw(ArgumentError("Ripple scaled RoPE is unsupported"))
    get(object, "quantization_config", nothing) === nothing ||
        throw(ArgumentError("load the BF16 Ripple checkpoint; upstream packed NVFP4 checkpoints are not supported by this reader"))
    kwargs = Dict{Symbol,Any}(k => object[String(k)] for k in fieldnames(RippleConfig) if haskey(object, String(k)))
    # These optional keys use the upstream class defaults when absent. In particular,
    # a folded embedding must not accidentally get scaled for a second time.
    for (key, value) in ((:hidden_size, 3072), (:intermediate_size, 24576),
            (:num_hidden_layers, 50), (:num_attention_heads, 48), (:max_position_embeddings, 2048))
        get!(kwargs, key, value)
    end
    get!(kwargs, :embed_scale, nothing)
    get!(kwargs, :ffn_clamp_threshold, nothing)
    get!(kwargs, :use_valres, false)
    config = RippleConfig(; kwargs...)
    return config
end

"""Required tensors in Hugging Face (out, in) layout, including all four PeriNorm scales."""
function ripple_expected_tensor_shapes(c::RippleConfig)
    shapes = Dict{String,Tuple}("model.embed_tokens.weight" => (c.vocab_size, c.hidden_size),
        "model.norm.weight" => (c.hidden_size,))
    c.tie_word_embeddings || (shapes["lm_head.weight"] = (c.vocab_size, c.hidden_size))
    for layer in 0:c.num_hidden_layers-1
        p = "model.layers.$layer"
        for norm in ("input_layernorm", "post_attention_layernorm", "pre_feedforward_layernorm", "post_feedforward_layernorm")
            shapes["$p.$norm.weight"] = (c.hidden_size,)
        end
        for (name, width) in (("q_proj", c.num_attention_heads*c.head_dim),
                ("k_proj", c.num_key_value_heads*c.head_dim), ("v_proj", c.num_key_value_heads*c.head_dim))
            shapes["$p.self_attn.$name.weight"] = (width, c.hidden_size)
        end
        shapes["$p.self_attn.o_proj.weight"] = (c.hidden_size, c.num_attention_heads*c.head_dim)
        shapes["$p.self_attn.q_norm.weight"] = (c.head_dim,)
        shapes["$p.self_attn.k_norm.weight"] = (c.head_dim,)
        c.use_gated_attention && (shapes["$p.self_attn.g_proj.weight"] = (c.num_attention_heads*c.head_dim, c.hidden_size))
        c.use_valres && (shapes["$p.self_attn.value_res_lambda"] = (1,))
        _flint_mlp_shapes!(shapes, "$p.mlp", c.hidden_size, c.intermediate_size)
    end
    return shapes
end

function verify_ripple_checkpoint(reader::HFSafetensorsReader, config::RippleConfig)
    shapes = ripple_expected_tensor_shapes(config)
    for (name, shape) in shapes
        haskey(reader, name) || throw(ArgumentError("missing Ripple tensor `$name`"))
        location = reader.locations[name]
        Tuple(location.shape) == shape || throw(DimensionMismatch("Ripple tensor `$name` has shape $(location.shape), expected $shape"))
        location.dtype in ("F32", "BF16", "F16") || throw(ArgumentError("Ripple tensor `$name` must be floating point"))
    end
    return shapes
end

_ripple_quantizable(name) = _flint_quantizable(name) || endswith(name, ".g_proj.weight")

"""
    load_hf_ripple_model(path; dtype=Float32, device=identity, scheme=:float,
                         group=nothing, quantize_on_device=false, on_progress=nothing)

Stream an unquantized Ripple safetensors checkpoint one tensor at a time.
Optional LifeAI weight-only INT8/INT4/NVFP4 quantization keeps embeddings,
norms and value-residual parameters floating point. This does not import the
vendor's packed NVFP4 format. Float32 activations are used for inference.
"""
function load_hf_ripple_model(path::AbstractString; dtype::Type=Float32, device=identity,
        scheme::Symbol=:float, group::Union{Nothing,Integer}=nothing,
        quantize_on_device::Bool=false, on_progress=nothing)
    dtype in (Float32, BFloat16) || throw(ArgumentError("Ripple dtype must be Float32 or BFloat16"))
    scheme in (:float, :int8, :int4, :nvfp4) || throw(ArgumentError("unsupported Ripple weight scheme"))
    spec = scheme === :float ? nothing : _flint_quantization_spec(scheme, group, :maxabs)
    model = load_hf_ripple_config(path)
    reader = open_safetensors_reader(path)
    shapes = verify_ripple_checkpoint(reader, model)
    if scheme === :int4
        all(name -> !_ripple_quantizable(name) || shapes[name][2] % spec.group == 0, keys(shapes)) ||
            throw(ArgumentError("Ripple INT4 projection input widths must be divisible by group"))
    end
    parameters = Dict{String,Any}()
    for (index, name) in enumerate(sort!(collect(keys(shapes))))
        quantize = spec !== nothing && _ripple_quantizable(name)
        target_dtype = length(shapes[name]) == 1 ? Float32 : dtype
        quantize && quantize_on_device && reader.locations[name].dtype == "BF16" && (target_dtype = BFloat16)
        weight = read_safetensors_tensor(reader, name; target_dtype)
        parameters[name] = if quantize
            input = quantize_on_device ? device(weight) : weight
            packed = scheme === :nvfp4 ? quantize_nvfp4(input) : _quantize_linear(input, spec)
            quantize_on_device ? packed : _flint_weight_to_device(device, packed)
        else
            device(weight)
        end
        on_progress === nothing || on_progress(name, index, length(shapes))
    end
    return (; model, parameters, source=abspath(path), quantization=scheme)
end
