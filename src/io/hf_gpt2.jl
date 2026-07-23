using Lux
using Random: Xoshiro

const _GPT2_CONFIG_FIELDS = Set([
    "activation_function",
    "architectures",
    "attn_pdrop",
    "bos_token_id",
    "embd_pdrop",
    "eos_token_id",
    "initializer_range",
    "layer_norm_epsilon",
    "model_type",
    "n_ctx",
    "n_embd",
    "n_head",
    "n_layer",
    "n_positions",
    "resid_pdrop",
    "summary_activation",
    "summary_first_dropout",
    "summary_proj_to_labels",
    "summary_type",
    "summary_use_proj",
    "task_specific_params",
    "vocab_size",
])
const _GPT2_FROZEN_REVISION = "607a30d783dfa663caf39e06633721c8d4cfcd7e"
const _GPT2_FROZEN_CHECKSUMS = (;
    config_json="0daed7749b4f02b8f76240d5444551d7b08712dab4d0adb8239c56ba823bb7b4",
    generation_config_json="ed0b32ac72c0f5f44a719abb2d7786ea5146c871f83717b7f2018065954de02b",
    tokenizer_json="8414cab924d8b9b33013f0d221c5862f365ee9be39c5c2bfae8a5a9e970478a6",
    tokenizer_config_json="5e04eb606e3a1583530a42e36c2a6b6615c86f34fe77e44d9ddeb43ff940931f",
    vocab_json="196139668be63f3b5d6574427317ae82f612a97c5d1cdaf36ed2256dbf636783",
    merges_txt="1ce1664773c50f3e0cc8842619a93edc4624525b728b188a9e0be33b7726adc5",
    model_safetensors="248dfc3911869ec493c76e65bf2fcf7f615828b0254c12b473182f0f81d3a707",
)

function _gpt2_required_real(config, name, path; positive=false)
    value = _json_required(config, name, path)
    value isa Real && isfinite(value) || throw(ArgumentError(
        "`$name` must be a finite number in $path",
    ))
    positive && value <= 0 && throw(ArgumentError("`$name` must be positive in $path"))
    return value
end

"""
    load_hf_gpt2_config(path; max_seq_len=nothing)

Strictly validate the frozen HuggingFace GPT-2 causal-LM configuration and
return a `GPTModel` constructor contract. Dropout values are recorded but are
inactive in the inference-only adapter.
"""
function load_hf_gpt2_config(path::AbstractString; max_seq_len=nothing)
    config = _json_object(path)
    actual_fields = Set(String.(collect(keys(config))))
    actual_fields == _GPT2_CONFIG_FIELDS || throw(ArgumentError(
        "GPT-2 config fields differ; missing=$(_format_tensor_names(setdiff(_GPT2_CONFIG_FIELDS, actual_fields))) " *
        "unexpected=$(_format_tensor_names(setdiff(actual_fields, _GPT2_CONFIG_FIELDS)))",
    ))
    _json_required(config, "model_type", path) == "gpt2" ||
        throw(ArgumentError("expected HuggingFace model_type `gpt2`"))
    architectures = _json_required(config, "architectures", path)
    architectures isa JSON3.Array && String.(collect(architectures)) == ["GPT2LMHeadModel"] ||
        throw(ArgumentError("GPT-2 config must declare only `GPT2LMHeadModel`"))
    _json_required(config, "activation_function", path) == "gelu_new" ||
        throw(ArgumentError("GPT-2 activation_function must be `gelu_new`"))

    vocab_size = _required_int(config, "vocab_size", path)
    d_model = _required_int(config, "n_embd", path)
    num_heads = _required_int(config, "n_head", path)
    num_layers = _required_int(config, "n_layer", path)
    source_max_seq_len = _required_int(config, "n_positions", path)
    _required_int(config, "n_ctx", path) == source_max_seq_len ||
        throw(ArgumentError("GPT-2 n_ctx and n_positions must match"))
    d_model % num_heads == 0 ||
        throw(ArgumentError("GPT-2 n_embd must be divisible by n_head"))
    resolved_max_seq_len = max_seq_len === nothing ? source_max_seq_len : Int(max_seq_len)
    1 <= resolved_max_seq_len <= source_max_seq_len || throw(ArgumentError(
        "max_seq_len must be in 1:$source_max_seq_len",
    ))

    epsilon = _gpt2_required_real(config, "layer_norm_epsilon", path; positive=true)
    initializer_range = _gpt2_required_real(config, "initializer_range", path; positive=true)
    dropouts = map(name -> begin
        value = _gpt2_required_real(config, name, path)
        0 <= value < 1 || throw(ArgumentError("`$name` must be in [0, 1)"))
        Float32(value)
    end, ("attn_pdrop", "embd_pdrop", "resid_pdrop"))
    bos = _json_required(config, "bos_token_id", path)
    eos = _json_required(config, "eos_token_id", path)
    bos isa Integer && eos isa Integer && bos == eos && bos == vocab_size - 1 ||
        throw(ArgumentError("GPT-2 BOS/EOS must both be the final vocabulary id"))

    # Validate the frozen causal-LM metadata instead of silently accepting a
    # sequence-classification variant.
    _json_required(config, "summary_activation", path) === nothing ||
        throw(ArgumentError("unsupported GPT-2 summary_activation"))
    _json_required(config, "summary_first_dropout", path) isa Real ||
        throw(ArgumentError("summary_first_dropout must be numeric"))
    _json_required(config, "summary_proj_to_labels", path) === true ||
        throw(ArgumentError("summary_proj_to_labels must be true"))
    _json_required(config, "summary_type", path) == "cls_index" ||
        throw(ArgumentError("summary_type must be `cls_index`"))
    _json_required(config, "summary_use_proj", path) === true ||
        throw(ArgumentError("summary_use_proj must be true"))
    _json_required(config, "task_specific_params", path) isa JSON3.Object ||
        throw(ArgumentError("task_specific_params must be an object"))

    return (;
        vocab_size,
        d_model,
        num_heads,
        num_kv_heads=num_heads,
        num_layers,
        head_dim=d_model ÷ num_heads,
        mlp_hidden_dim=4 * d_model,
        use_bias=true,
        lm_head_bias=false,
        is_causal=true,
        use_rope=false,
        position_embedding_type=:learned_absolute,
        use_qk_norm=false,
        qk_norm_epsilon=1.0f-6,
        max_seq_len=resolved_max_seq_len,
        source_max_seq_len,
        rope_theta=10000.0f0,
        rope_style=:interleaved,
        norm_epsilon=Float32(epsilon),
        norm_type=:layernorm,
        mlp_type=:gelu_new,
        tie_embeddings=true,
        initializer_range=Float32(initializer_range),
        inference_dropouts=dropouts,
        bos_hf_id=Int(bos),
        eos_hf_id=Int(eos),
    )
end

function _gpt2_expected_tensor_names(model::GPTModel)
    names = Set(["wte.weight", "wpe.weight", "ln_f.weight", "ln_f.bias"])
    for layer in 0:(model.num_layers - 1)
        prefix = "h.$layer"
        union!(names, [
            "$prefix.attn.bias",
            "$prefix.attn.c_attn.weight",
            "$prefix.attn.c_attn.bias",
            "$prefix.attn.c_proj.weight",
            "$prefix.attn.c_proj.bias",
            "$prefix.ln_1.weight",
            "$prefix.ln_1.bias",
            "$prefix.ln_2.weight",
            "$prefix.ln_2.bias",
            "$prefix.mlp.c_fc.weight",
            "$prefix.mlp.c_fc.bias",
            "$prefix.mlp.c_proj.weight",
            "$prefix.mlp.c_proj.bias",
        ])
    end
    return names
end

function _gpt2_norm_parameters(tensors, prefix, d_model)
    return (;
        bias=reshape(_expect_tensor(tensors, "$prefix.bias", (d_model,)), d_model, 1, 1),
        scale=reshape(_expect_tensor(tensors, "$prefix.weight", (d_model,)), d_model, 1, 1),
    )
end

function _gpt2_validate_causal_buffer(buffer, source_max_seq_len, name)
    size(buffer) == (1, 1, source_max_seq_len, source_max_seq_len) ||
        throw(DimensionMismatch("$name has an invalid causal-mask shape"))
    @inbounds for key in 1:source_max_seq_len, query in 1:source_max_seq_len
        expected = key <= query ? 1.0f0 : 0.0f0
        buffer[1, 1, query, key] == expected || throw(ArgumentError(
            "$name is not an exact lower-triangular causal buffer",
        ))
    end
    return nothing
end

"""
    load_hf_gpt2_parameters(model, tensors; source_max_seq_len=model.max_seq_len)

Map a complete HuggingFace GPT-2 state dict into LifeAI's shared parameter
tree. HF Conv1D `(in, out)` tensors are explicitly split/transposed into
independent Lux Dense `(out, in)` Q/K/V projections.
"""
function load_hf_gpt2_parameters(
    model::GPTModel,
    tensors::AbstractDict;
    source_max_seq_len::Int=model.max_seq_len,
)
    model.position_embedding_type === :learned_absolute && !model.use_rope ||
        throw(ArgumentError("GPT-2 requires learned absolute positions without RoPE"))
    model.norm_type === :layernorm || throw(ArgumentError("GPT-2 requires LayerNorm"))
    model.mlp_type === :gelu_new || throw(ArgumentError("GPT-2 requires GELU-New"))
    model.use_bias || throw(ArgumentError("GPT-2 projections require bias"))
    !model.lm_head_bias || throw(ArgumentError("GPT-2 LM head must not have bias"))
    model.tie_embeddings || throw(ArgumentError("GPT-2 requires tied embeddings"))
    model.num_kv_heads == model.num_heads || throw(ArgumentError("GPT-2 requires full MHA"))
    source_max_seq_len >= model.max_seq_len ||
        throw(ArgumentError("source position table is shorter than model.max_seq_len"))

    expected = _gpt2_expected_tensor_names(model)
    actual = Set(String.(collect(keys(tensors))))
    allowed = union(expected, Set(["lm_head.weight"]))
    missing = setdiff(expected, actual)
    unexpected = setdiff(actual, allowed)
    isempty(missing) || throw(ArgumentError(
        "missing HuggingFace tensors: $(_format_tensor_names(missing))",
    ))
    isempty(unexpected) || throw(ArgumentError(
        "unexpected HuggingFace tensors: $(_format_tensor_names(unexpected))",
    ))

    d_model = model.d_model
    hidden = model.mlp_hidden_dim
    embedding_hf = _expect_tensor(
        tensors,
        "wte.weight",
        (model.vocab_size, d_model),
    )
    if haskey(tensors, "lm_head.weight")
        tied = _expect_tensor(tensors, "lm_head.weight", (model.vocab_size, d_model))
        tied == embedding_hf || throw(ArgumentError(
            "GPT-2 lm_head.weight does not equal wte.weight",
        ))
    end
    position_hf = _expect_tensor(
        tensors,
        "wpe.weight",
        (source_max_seq_len, d_model),
    )
    token_embedding = (; weight=permutedims(embedding_hf, (2, 1)))
    position_embedding = (; weight=permutedims(
        @view(position_hf[1:model.max_seq_len, :]),
        (2, 1),
    ))

    block_values = ntuple(model.num_layers) do julia_layer
        layer = julia_layer - 1
        prefix = "h.$layer"
        causal = _expect_tensor(
            tensors,
            "$prefix.attn.bias",
            (1, 1, source_max_seq_len, source_max_seq_len),
        )
        _gpt2_validate_causal_buffer(causal, source_max_seq_len, "$prefix.attn.bias")
        fused_weight = _expect_tensor(
            tensors,
            "$prefix.attn.c_attn.weight",
            (d_model, 3 * d_model),
        )
        fused_bias = _expect_tensor(
            tensors,
            "$prefix.attn.c_attn.bias",
            (3 * d_model,),
        )
        attn = (;
            q_proj=(;
                weight=permutedims(@view(fused_weight[:, 1:d_model]), (2, 1)),
                bias=copy(@view(fused_bias[1:d_model])),
            ),
            k_proj=(;
                weight=permutedims(@view(fused_weight[:, d_model + 1:2 * d_model]), (2, 1)),
                bias=copy(@view(fused_bias[d_model + 1:2 * d_model])),
            ),
            v_proj=(;
                weight=permutedims(@view(fused_weight[:, 2 * d_model + 1:3 * d_model]), (2, 1)),
                bias=copy(@view(fused_bias[2 * d_model + 1:3 * d_model])),
            ),
            o_proj=(;
                weight=permutedims(_expect_tensor(
                    tensors,
                    "$prefix.attn.c_proj.weight",
                    (d_model, d_model),
                ), (2, 1)),
                bias=_expect_tensor(tensors, "$prefix.attn.c_proj.bias", (d_model,)),
            ),
        )
        mlp = (;
            layer_1=(;
                weight=permutedims(_expect_tensor(
                    tensors,
                    "$prefix.mlp.c_fc.weight",
                    (d_model, hidden),
                ), (2, 1)),
                bias=_expect_tensor(tensors, "$prefix.mlp.c_fc.bias", (hidden,)),
            ),
            layer_2=(;
                weight=permutedims(_expect_tensor(
                    tensors,
                    "$prefix.mlp.c_proj.weight",
                    (hidden, d_model),
                ), (2, 1)),
                bias=_expect_tensor(tensors, "$prefix.mlp.c_proj.bias", (d_model,)),
            ),
        )
        return (;
            norm1=_gpt2_norm_parameters(tensors, "$prefix.ln_1", d_model),
            attn,
            norm2=_gpt2_norm_parameters(tensors, "$prefix.ln_2", d_model),
            mlp,
        )
    end
    block_names = Tuple(Symbol("layer_$layer") for layer in 1:model.num_layers)
    blocks = NamedTuple{block_names}(block_values)
    final_norm = _gpt2_norm_parameters(tensors, "ln_f", d_model)
    return (;
        token_embedding,
        blocks,
        final_norm,
        lm_head=(;),
        position_embedding,
    )
end

function _gpt2_file_checksums(model_dir)
    files = (
        "config.json",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
        "merges.txt",
        "model.safetensors",
    )
    return NamedTuple{Tuple(Symbol.(replace.(files, "." => "_")))}(
        Tuple(_sha256_hex(read(joinpath(model_dir, file))) for file in files),
    )
end

"""
    load_hf_gpt2_model(model_dir; max_seq_len=nothing)

Load a local GPT-2 checkpoint without downloading. The returned parameters use
Float32 inference semantics and the source causal buffers are validated but not
retained.
"""
function load_hf_gpt2_model(
    model_dir::AbstractString;
    max_seq_len=nothing,
    weight_dtype::Type=Float32,
)
    isdir(model_dir) || throw(ArgumentError("model directory does not exist: $model_dir"))
    config = load_hf_gpt2_config(joinpath(model_dir, "config.json"); max_seq_len)
    model = GPTModel(config)
    tensors = load_safetensors(model_dir; target_dtype=weight_dtype)
    parameters = load_hf_gpt2_parameters(
        model,
        tensors;
        source_max_seq_len=config.source_max_seq_len,
    )
    empty!(tensors)
    GC.gc(false)
    states = Lux.initialstates(Xoshiro(0), model)
    return (;
        model,
        parameters,
        states,
        config,
        source=abspath(model_dir),
        checksums=_gpt2_file_checksums(model_dir),
    )
end

"""
    load_hf_gpt2_bundle(model_dir; revision, max_seq_len=nothing)

Load the exact local GPT-2 model and tokenizer as a text-generation bundle.
The immutable source revision is mandatory and is returned with file checksums.
"""
function load_hf_gpt2_bundle(
    model_dir::AbstractString;
    revision::AbstractString,
    max_seq_len=nothing,
    weight_dtype::Type=Float32,
)
    isempty(revision) && throw(ArgumentError("an immutable GPT-2 revision is required"))
    revision == _GPT2_FROZEN_REVISION || throw(ArgumentError(
        "unsupported GPT-2 revision $(repr(revision)); expected $_GPT2_FROZEN_REVISION",
    ))
    tokenizer = load_hf_gpt2_tokenizer(model_dir; revision)
    loaded = load_hf_gpt2_model(model_dir; max_seq_len, weight_dtype)
    loaded.checksums == _GPT2_FROZEN_CHECKSUMS || throw(ArgumentError(
        "GPT-2 files do not match the frozen Week 10 checksums",
    ))
    vocab_size(tokenizer) == loaded.model.vocab_size ||
        throw(ArgumentError("GPT-2 tokenizer and model vocabularies differ"))
    tokenizer.model_max_length == loaded.config.source_max_seq_len ||
        throw(ArgumentError("GPT-2 tokenizer/model context lengths differ"))
    return merge(loaded, (;
        tokenizer,
        generation_config=hf_generation_config(tokenizer),
        revision=String(revision),
    ))
end

"""
    hf_gpt2_forward_trace(model, tokens, ps, st)

Return combined token/position embeddings, every residual block output, final
hidden state and logits for an eager GPT-2 parity run.
"""
function hf_gpt2_forward_trace(model::GPTModel, tokens, ps, st::NamedTuple)
    model.position_embedding_type === :learned_absolute ||
        throw(ArgumentError("hf_gpt2_forward_trace requires learned positions"))
    @assert ndims(tokens) == 2 "`tokens` must have shape (seq_len, batch)"
    _validate_token_ids(tokens, model.vocab_size)
    x, st_embedding = model.token_embedding(tokens, ps.token_embedding, st.token_embedding)
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
