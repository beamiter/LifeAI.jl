"""Architecture metadata for RecursiveSI/Flint. Config token IDs use HuggingFace's zero-based convention."""
struct FlintConfig
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
    mlp_only_layers::Vector{Int}
    decoder_sparse_step::Int
    moe_intermediate_size::Int
    shared_expert_intermediate_size::Int
    num_experts::Int
    num_experts_per_tok::Int
    norm_topk_prob::Bool
    tie_word_embeddings::Bool
    bos_token_id::Int
    eos_token_id::Vector{Int}
    pad_token_id::Int
end

function FlintConfig(;
    vocab_size=150144, hidden_size=2048, intermediate_size=6144,
    num_hidden_layers=49, num_attention_heads=32, num_key_value_heads=4,
    head_dim=128, max_position_embeddings=262144, rms_norm_eps=1f-6,
    rope_theta=10000.0, sliding_window=513,
    layer_types=nothing, mlp_only_layers=[0, 1], decoder_sparse_step=1,
    moe_intermediate_size=512, shared_expert_intermediate_size=1024,
    num_experts=256, num_experts_per_tok=6, norm_topk_prob=true,
    tie_word_embeddings=false, bos_token_id=150000,
    eos_token_id=[150001, 150003, 150005, 150018], pad_token_id=150001,
)
    positive = (vocab_size, hidden_size, intermediate_size, num_hidden_layers,
        num_attention_heads, num_key_value_heads, head_dim,
        max_position_embeddings, sliding_window, decoder_sparse_step,
        moe_intermediate_size, shared_expert_intermediate_size,
        num_experts, num_experts_per_tok)
    names = ("vocab_size", "hidden_size", "intermediate_size", "num_hidden_layers",
        "num_attention_heads", "num_key_value_heads", "head_dim",
        "max_position_embeddings", "sliding_window", "decoder_sparse_step",
        "moe_intermediate_size", "shared_expert_intermediate_size",
        "num_experts", "num_experts_per_tok")
    values = map(zip(names, positive)) do (name, value)
        n = _hf_strict_host_int(value, name)
        n > 0 || throw(ArgumentError("Flint $name must be positive"))
        n
    end
    (vocab_size, hidden_size, intermediate_size, num_hidden_layers,
        num_attention_heads, num_key_value_heads, head_dim,
        max_position_embeddings, sliding_window, decoder_sparse_step,
        moe_intermediate_size, shared_expert_intermediate_size,
        num_experts, num_experts_per_tok) = values
    num_attention_heads % num_key_value_heads == 0 || throw(ArgumentError("Flint query heads must be divisible by KV heads"))
    iseven(head_dim) || throw(ArgumentError("Flint head_dim must be even for RoPE"))
    num_experts_per_tok <= num_experts || throw(ArgumentError("Flint top-k exceeds num_experts"))
    rms_norm_eps isa Real && !(rms_norm_eps isa Bool) && isfinite(rms_norm_eps) && rms_norm_eps > 0 ||
        throw(ArgumentError("Flint rms_norm_eps must be finite and positive"))
    rope_theta isa Real && !(rope_theta isa Bool) && isfinite(rope_theta) && rope_theta > 0 ||
        throw(ArgumentError("Flint rope_theta must be finite and positive"))
    norm_topk_prob isa Bool && tie_word_embeddings isa Bool || throw(ArgumentError("Flint boolean options must be Bool"))
    types = layer_types === nothing ?
        [i < 2 || i % 4 == 1 || i == num_hidden_layers - 1 ? "full_attention" : "sliding_attention" for i in 0:num_hidden_layers-1] :
        String.(layer_types)
    length(types) == num_hidden_layers || throw(ArgumentError("Flint layer_types length must match num_hidden_layers"))
    all(t -> t in ("full_attention", "sliding_attention"), types) || throw(ArgumentError("unsupported Flint attention layer type"))
    dense = [_hf_strict_host_int(i, "mlp_only_layers entry") for i in mlp_only_layers]
    length(unique(dense)) == length(dense) || throw(ArgumentError("Flint mlp_only_layers must be unique"))
    all(i -> 0 <= i < num_hidden_layers, dense) || throw(ArgumentError("Flint mlp_only_layers entry out of range"))
    eos = eos_token_id isa Integer ? [eos_token_id] : collect(eos_token_id)
    eos = [_hf_zero_based_token_id(i, "eos_token_id") for i in eos]
    isempty(eos) && throw(ArgumentError("Flint eos_token_id must not be empty"))
    bos = _hf_zero_based_token_id(bos_token_id, "bos_token_id")
    pad = _hf_zero_based_token_id(pad_token_id, "pad_token_id")
    all(i -> i < vocab_size, [bos; pad; eos]) || throw(ArgumentError("Flint special token ID exceeds vocabulary"))
    return FlintConfig(values[1:8]..., Float32(rms_norm_eps), Float64(rope_theta),
        sliding_window, types, dense, decoder_sparse_step, moe_intermediate_size,
        shared_expert_intermediate_size, num_experts, num_experts_per_tok,
        norm_topk_prob, tie_word_embeddings, bos, eos, pad)
end

"""Whether the one-based transformer layer uses routed and shared experts."""
flint_is_sparse(config::FlintConfig, layer::Int) =
    !((layer - 1) in config.mlp_only_layers) && layer % config.decoder_sparse_step == 0

function flint_config_dict(config::FlintConfig)
    result = Dict{String,Any}(String(field) => getfield(config, field) for field in fieldnames(FlintConfig))
    merge!(result, Dict("model_type" => "flint", "architectures" => ["FlintForCausalLM"],
        "attention_bias" => false, "attention_dropout" => 0.0, "hidden_act" => "silu",
        "router_score_function" => "sigmoid", "softmax_type" => "learnable",
        "no_rope_layer_indices" => findall(==("full_attention"), config.layer_types) .- 1))
    return result
end

"""Read a Flint config.json, rejecting architecture variants that this implementation cannot represent."""
function load_hf_flint_config(path::AbstractString)
    file = isdir(path) ? joinpath(path, "config.json") : path
    object = _hf_json(read(file, String), "Flint config.json")
    _hf_reject_duplicate_fields(object, "Flint config.json")
    get(object, "model_type", nothing) == "flint" || throw(ArgumentError("expected model_type=flint"))
    for (key, expected) in (("hidden_act", "silu"), ("attention_bias", false),
            ("router_score_function", "sigmoid"), ("softmax_type", "learnable"))
        get(object, key, expected) == expected || throw(ArgumentError("unsupported Flint $key"))
    end
    get(object, "attention_dropout", 0.0) == 0 || throw(ArgumentError("Flint inference requires attention_dropout=0"))
    kwargs = Dict{Symbol,Any}()
    for key in fieldnames(FlintConfig)
        haskey(object, String(key)) && (kwargs[key] = object[String(key)])
    end
    if haskey(object, "rope_parameters")
        rope = object["rope_parameters"]
        get(rope, "rope_type", "default") == "default" || throw(ArgumentError("unsupported Flint scaled RoPE"))
        kwargs[:rope_theta] = get(rope, "rope_theta", get(kwargs, :rope_theta, 10000.0))
    end
    haskey(object, "rope_scaling") && object["rope_scaling"] !== nothing &&
        throw(ArgumentError("unsupported Flint rope_scaling"))
    config = FlintConfig(; kwargs...)
    if haskey(object, "no_rope_layer_indices")
        expected = findall(==("full_attention"), config.layer_types) .- 1
        sort!(Int.(object["no_rope_layer_indices"])) == expected ||
            throw(ArgumentError("Flint no_rope_layer_indices disagrees with layer_types"))
    end
    return config
end

"""Required checkpoint tensors and their HuggingFace (out, in) dimensions. MTP tensors are optional and unused."""
function flint_expected_tensor_shapes(config::FlintConfig)
    c = config
    shapes = Dict{String,Tuple}("model.embed_tokens.weight" => (c.vocab_size, c.hidden_size),
        "model.norm.weight" => (c.hidden_size,))
    c.tie_word_embeddings || (shapes["lm_head.weight"] = (c.vocab_size, c.hidden_size))
    for layer in 1:c.num_hidden_layers
        prefix = "model.layers.$(layer - 1)"
        shapes["$prefix.input_layernorm.weight"] = (c.hidden_size,)
        shapes["$prefix.post_attention_layernorm.weight"] = (c.hidden_size,)
        shapes["$prefix.self_attn.q_proj.weight"] = (c.num_attention_heads * c.head_dim, c.hidden_size)
        shapes["$prefix.self_attn.k_proj.weight"] = (c.num_key_value_heads * c.head_dim, c.hidden_size)
        shapes["$prefix.self_attn.v_proj.weight"] = (c.num_key_value_heads * c.head_dim, c.hidden_size)
        shapes["$prefix.self_attn.o_proj.weight"] = (c.hidden_size, c.num_attention_heads * c.head_dim)
        shapes["$prefix.self_attn.q_norm.weight"] = (c.head_dim,)
        shapes["$prefix.self_attn.k_norm.weight"] = (c.head_dim,)
        shapes["$prefix.self_attn.softmax_offset"] = (c.num_attention_heads,)
        if flint_is_sparse(c, layer)
            shapes["$prefix.mlp.gate.weight"] = (c.num_experts, c.hidden_size)
            for expert in 0:c.num_experts-1
                _flint_mlp_shapes!(shapes, "$prefix.mlp.experts.$expert", c.hidden_size, c.moe_intermediate_size)
            end
            _flint_mlp_shapes!(shapes, "$prefix.mlp.shared_expert", c.hidden_size, c.shared_expert_intermediate_size)
        else
            _flint_mlp_shapes!(shapes, "$prefix.mlp", c.hidden_size, c.intermediate_size)
        end
    end
    return shapes
end

function _flint_mlp_shapes!(shapes, prefix, hidden, intermediate)
    shapes["$prefix.gate_proj.weight"] = (intermediate, hidden)
    shapes["$prefix.up_proj.weight"] = (intermediate, hidden)
    shapes["$prefix.down_proj.weight"] = (hidden, intermediate)
    return shapes
end

function verify_flint_checkpoint(reader::HFSafetensorsReader, config::FlintConfig)
    shapes = flint_expected_tensor_shapes(config)
    for (name, shape) in shapes
        haskey(reader, name) || throw(ArgumentError("missing Flint checkpoint tensor `$name`"))
        Tuple(reader.locations[name].shape) == shape || throw(DimensionMismatch("Flint tensor `$name` has shape $(Tuple(reader.locations[name].shape)), expected $shape"))
    end
    return shapes
end

"""Load Flint tensors without constructing a second complete floating-point copy. `device` maps each tensor, e.g. `CUDA.cu`."""
function load_hf_flint_model(path::AbstractString; dtype::Type=Float32, device=identity)
    dtype in (Float32, BFloat16) || throw(ArgumentError("Flint loading supports Float32 or BFloat16"))
    model = load_hf_flint_config(path)
    reader = open_safetensors_reader(path)
    shapes = verify_flint_checkpoint(reader, model)
    parameters = Dict{String,Any}()
    for name in sort!(collect(keys(shapes)))
        target = endswith(name, ".mlp.gate.weight") || endswith(name, ".softmax_offset") ? Float32 : dtype
        parameters[name] = device(read_safetensors_tensor(reader, name; target_dtype=target))
    end
    return (; model, parameters, source=abspath(path))
end
