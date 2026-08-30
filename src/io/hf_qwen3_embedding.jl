using BFloat16s: BFloat16
using Lux
using Random: Xoshiro
using SHA

"""
    Qwen3EmbeddingSpec

Frozen architecture and provenance for the official
`Qwen/Qwen3-Embedding-0.6B` checkpoint used by Week 22. The checkpoint is
Qwen3-shaped, but its vocabulary and native context differ from the ordinary
Qwen3-0.6B causal-LM checkpoint and therefore have a separate contract.
"""
struct Qwen3EmbeddingSpec
    variant::Symbol
    model_id::String
    revision::String
    config_sha256::String
    tokenizer_sha256::String
    tokenizer_config_sha256::String
    generation_config_sha256::String
    modules_sha256::String
    sentence_transformers_config_sha256::String
    pooling_config_sha256::String
    model_sha256::String
    vocab_size::Int
    d_model::Int
    mlp_hidden_dim::Int
    num_layers::Int
    num_heads::Int
    num_kv_heads::Int
    head_dim::Int
    max_position_embeddings::Int
    minimum_dimension::Int

    function Qwen3EmbeddingSpec(
        variant,
        model_id,
        revision,
        config_sha256,
        tokenizer_sha256,
        tokenizer_config_sha256,
        generation_config_sha256,
        modules_sha256,
        sentence_transformers_config_sha256,
        pooling_config_sha256,
        model_sha256,
        vocab_size,
        d_model,
        mlp_hidden_dim,
        num_layers,
        num_heads,
        num_kv_heads,
        head_dim,
        max_position_embeddings,
        minimum_dimension,
    )
        variant isa Symbol || throw(ArgumentError(
            "Qwen3 embedding variant must be a Symbol",
        ))
        prefix = "Qwen3 embedding"
        resolved_d_model = _qwen3_spec_positive_int(
            d_model,
            "$prefix d_model",
        )
        resolved_num_heads = _qwen3_spec_positive_int(
            num_heads,
            "$prefix num_heads",
        )
        resolved_num_kv_heads = _qwen3_spec_positive_int(
            num_kv_heads,
            "$prefix num_kv_heads",
        )
        resolved_head_dim = _qwen3_spec_positive_int(
            head_dim,
            "$prefix head_dim",
        )
        _qwen3_spec_attention_geometry(
            prefix,
            resolved_num_heads,
            resolved_num_kv_heads,
            resolved_head_dim,
        )
        resolved_minimum_dimension = _qwen3_spec_positive_int(
            minimum_dimension,
            "$prefix minimum_dimension",
        )
        resolved_minimum_dimension <= resolved_d_model || throw(ArgumentError(
            "$prefix minimum_dimension must not exceed d_model",
        ))
        return new(
            variant,
            _qwen3_spec_string(model_id, "$prefix model_id"),
            _qwen3_spec_string(revision, "$prefix revision"),
            _qwen3_spec_string(config_sha256, "$prefix config_sha256"),
            _qwen3_spec_string(tokenizer_sha256, "$prefix tokenizer_sha256"),
            _qwen3_spec_string(
                tokenizer_config_sha256,
                "$prefix tokenizer_config_sha256",
            ),
            _qwen3_spec_string(
                generation_config_sha256,
                "$prefix generation_config_sha256",
            ),
            _qwen3_spec_string(modules_sha256, "$prefix modules_sha256"),
            _qwen3_spec_string(
                sentence_transformers_config_sha256,
                "$prefix sentence_transformers_config_sha256",
            ),
            _qwen3_spec_string(
                pooling_config_sha256,
                "$prefix pooling_config_sha256",
            ),
            _qwen3_spec_string(model_sha256, "$prefix model_sha256"),
            _qwen3_spec_positive_int(vocab_size, "$prefix vocab_size"),
            resolved_d_model,
            _qwen3_spec_positive_int(mlp_hidden_dim, "$prefix mlp_hidden_dim"),
            _qwen3_spec_positive_int(num_layers, "$prefix num_layers"),
            resolved_num_heads,
            resolved_num_kv_heads,
            resolved_head_dim,
            _qwen3_spec_positive_int(
                max_position_embeddings,
                "$prefix max_position_embeddings",
            ),
            resolved_minimum_dimension,
        )
    end
end

const _QWEN3_EMBEDDING_SPEC = Qwen3EmbeddingSpec(
    :qwen3_embedding_0_6b,
    "Qwen/Qwen3-Embedding-0.6B",
    "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3",
    "b5bf1f51fc45be473a54718cef92448d90a1be001bf9b9a44b8c7f10a19feaa9",
    "def76fb086971c7867b829c23a26261e38d9d74e02139253b38aeb9df8b4b50a",
    "253153d0738ceb4c668d2eff957714dd2bea0b56de772a9fdccd96cbf517e6a0",
    "28396d421a2108acce96383f6a7de78008f7f1b17f807958f3c14c51dbfb65fb",
    "84e40c8e006c9b1d6c122e02cba9b02458120b5fb0c87b746c41e0207cf642cf",
    "10667c72ddb772627bf1780cb7f86af8e2ae0032b8c243c731172064105c6961",
    "37bf193fa101f19101bfad9c31d3eb0f786e247b7b1e5cb7f007d730eed1ddbd",
    "0437e45c94563b09e13cb7a64478fc406947a93cb34a7e05870fc8dcd48e23fd",
    151_669,
    1_024,
    3_072,
    28,
    16,
    8,
    128,
    32_768,
    32,
)

"""Return the immutable Week 22 Qwen3-Embedding-0.6B specification."""
qwen3_embedding_spec() = _QWEN3_EMBEDDING_SPEC

"""Return the exact scalar parameter count implied by the frozen checkpoint."""
function qwen3_embedding_parameter_count(
    spec::Qwen3EmbeddingSpec=qwen3_embedding_spec(),
)
    d_model = BigInt(spec.d_model)
    query_dim = BigInt(spec.num_heads) * spec.head_dim
    kv_dim = BigInt(spec.num_kv_heads) * spec.head_dim
    embedding = BigInt(spec.vocab_size) * d_model
    attention = 2 * query_dim * d_model +
        2 * kv_dim * d_model +
        2 * BigInt(spec.head_dim)
    mlp = 3 * d_model * spec.mlp_hidden_dim
    norms = 2 * d_model
    final_norm = d_model
    count = embedding +
        spec.num_layers * (attention + mlp + norms) +
        final_norm
    return _qwen3_parameter_count_int(count, "Qwen3 embedding parameter count")
end

_qwen3_embedding_sha256_file(path::AbstractString) =
    open(path, "r") do io
        bytes2hex(sha256(io))
    end

function _qwen3_embedding_config_matches(config, spec::Qwen3EmbeddingSpec)
    return config.vocab_size == spec.vocab_size &&
        config.d_model == spec.d_model &&
        config.mlp_hidden_dim == spec.mlp_hidden_dim &&
        config.num_layers == spec.num_layers &&
        config.num_heads == spec.num_heads &&
        config.num_kv_heads == spec.num_kv_heads &&
        config.head_dim == spec.head_dim &&
        config.source_max_seq_len == spec.max_position_embeddings &&
        config.tie_embeddings
end

function _qwen3_embedding_requested_max_seq_len(value)
    requested = _qwen3_requested_max_seq_len(value)
    requested === nothing && return nothing
    maximum = qwen3_embedding_spec().max_position_embeddings
    requested <= maximum || throw(ArgumentError(
        "max_seq_len must be in 1:$maximum; got $requested",
    ))
    return requested
end

"""
    load_hf_qwen3_embedding_config(path; max_seq_len=8192)

Parse a Qwen3 config and require the exact official Week 22 embedding
checkpoint contract, including its frozen config checksum.
"""
function load_hf_qwen3_embedding_config(
    path::AbstractString;
    max_seq_len=8_192,
)
    requested_max_seq_len = _qwen3_embedding_requested_max_seq_len(max_seq_len)
    isfile(path) || throw(ArgumentError(
        "required Qwen3 embedding config does not exist: $path",
    ))
    spec = qwen3_embedding_spec()
    actual_sha256 = _qwen3_embedding_sha256_file(path)
    actual_sha256 == spec.config_sha256 || throw(ArgumentError(
        "Qwen3 embedding config checksum mismatch: expected " *
        "$(spec.config_sha256), computed $actual_sha256",
    ))
    config = load_hf_qwen3_config(path; max_seq_len=requested_max_seq_len)
    _qwen3_embedding_config_matches(config, spec) || throw(ArgumentError(
        "config does not match the frozen $(spec.variant) architecture",
    ))
    config.qwen3_variant === nothing || throw(ArgumentError(
        "embedding config must not alias an ordinary Qwen3 dense variant",
    ))
    return merge(config, (; qwen3_embedding_variant=spec.variant))
end

function _qwen3_embedding_expected_assets(spec::Qwen3EmbeddingSpec)
    return (
        "config.json" => spec.config_sha256,
        "tokenizer.json" => spec.tokenizer_sha256,
        "tokenizer_config.json" => spec.tokenizer_config_sha256,
        "generation_config.json" => spec.generation_config_sha256,
        "modules.json" => spec.modules_sha256,
        "config_sentence_transformers.json" =>
            spec.sentence_transformers_config_sha256,
        joinpath("1_Pooling", "config.json") => spec.pooling_config_sha256,
        "model.safetensors" => spec.model_sha256,
    )
end

"""
    verify_qwen3_embedding_assets(model_dir)

Hash every frozen Week 22 model/tokenizer/SentenceTransformers asset. This is
an explicit provenance operation because hashing the full weight file adds a
second sequential read to model startup.
"""
function verify_qwen3_embedding_assets(model_dir::AbstractString)
    isdir(model_dir) || throw(ArgumentError(
        "model directory does not exist: $model_dir",
    ))
    verified = NamedTuple[]
    for (relative, expected) in _qwen3_embedding_expected_assets(
        qwen3_embedding_spec(),
    )
        isempty(expected) && throw(ArgumentError(
            "Qwen3 embedding asset manifest is incomplete for $relative",
        ))
        path = joinpath(model_dir, relative)
        isfile(path) || throw(ArgumentError(
            "required Qwen3 embedding asset does not exist: $path",
        ))
        actual = _qwen3_embedding_sha256_file(path)
        actual == expected || throw(ArgumentError(
            "Qwen3 embedding asset checksum mismatch for $relative: " *
            "expected $expected, computed $actual",
        ))
        push!(verified, (;
            path=relative,
            bytes=filesize(path),
            sha256=actual,
        ))
    end
    return Tuple(verified)
end

function _qwen3_embedding_base_state_dict(tensors::AbstractDict)
    mapped = Dict{String,Any}()
    for (raw_name, tensor) in tensors
        name = String(raw_name)
        (
            name == "embed_tokens.weight" ||
            name == "norm.weight" ||
            startswith(name, "layers.")
        ) || throw(ArgumentError(
            "unsupported Qwen3 embedding tensor namespace `$name`",
        ))
        mapped_name = "model.$name"
        haskey(mapped, mapped_name) && throw(ArgumentError(
            "duplicate Qwen3 embedding tensor `$mapped_name`",
        ))
        mapped[mapped_name] = tensor
    end
    return mapped
end

"""
    load_hf_qwen3_embedding_model(
        model_dir;
        max_seq_len=8192,
        weight_dtype=BFloat16,
    )

Load the frozen Qwen3-Embedding-0.6B model without treating it as one of the
six ordinary dense causal-LM variants.
"""
function load_hf_qwen3_embedding_model(
    model_dir::AbstractString;
    max_seq_len=8_192,
    weight_dtype::Type=BFloat16,
)
    requested_max_seq_len = _qwen3_embedding_requested_max_seq_len(max_seq_len)
    requested_weight_dtype = _qwen3_weight_dtype(weight_dtype)
    isdir(model_dir) || throw(ArgumentError(
        "model directory does not exist: $model_dir",
    ))
    config = load_hf_qwen3_embedding_config(
        joinpath(model_dir, "config.json");
        max_seq_len=requested_max_seq_len,
    )
    model = GPTModel(config)
    source_tensors = load_safetensors(
        model_dir;
        target_dtype=requested_weight_dtype,
    )
    tensors = _qwen3_embedding_base_state_dict(source_tensors)
    parameters = load_hf_qwen3_parameters(model, tensors)
    empty!(source_tensors)
    empty!(tensors)
    GC.gc(false)
    states = Lux.initialstates(Xoshiro(0), model)
    return (;
        model,
        parameters,
        states,
        config,
        variant=qwen3_embedding_spec(),
        source=abspath(model_dir),
    )
end

"""
    load_hf_qwen3_embedding_bundle(
        model_dir;
        max_seq_len=8192,
        weight_dtype=BFloat16,
        revision=qwen3_embedding_spec().revision,
    )

Load the frozen embedding model and its embedding-profile tokenizer. Small
configuration assets are checked on every load; call
`verify_qwen3_embedding_assets` when the full weight checksum is required.
"""
function load_hf_qwen3_embedding_bundle(
    model_dir::AbstractString;
    max_seq_len=8_192,
    weight_dtype::Type=BFloat16,
    revision::AbstractString=qwen3_embedding_spec().revision,
)
    spec = qwen3_embedding_spec()
    String(revision) == spec.revision || throw(ArgumentError(
        "unsupported Qwen3 embedding revision $(repr(revision)); expected " *
        spec.revision,
    ))
    requested_max_seq_len = _qwen3_embedding_requested_max_seq_len(max_seq_len)
    requested_weight_dtype = _qwen3_weight_dtype(weight_dtype)
    tokenizer = load_hf_qwen3_embedding_tokenizer(model_dir; revision)
    tokenizer.profile === :embedding || error(
        "internal Qwen3 embedding tokenizer profile mismatch",
    )
    checksums = (
        tokenizer.tokenizer_sha256,
        tokenizer.tokenizer_config_sha256,
        tokenizer.generation_config_sha256,
    )
    expected = (
        spec.tokenizer_sha256,
        spec.tokenizer_config_sha256,
        spec.generation_config_sha256,
    )
    checksums == expected || throw(ArgumentError(
        "Qwen3 embedding tokenizer assets do not match frozen revision " *
        spec.revision,
    ))
    loaded = load_hf_qwen3_embedding_model(
        model_dir;
        max_seq_len=requested_max_seq_len,
        weight_dtype=requested_weight_dtype,
    )
    vocab_size(tokenizer) == loaded.model.vocab_size || throw(ArgumentError(
        "embedding tokenizer vocabulary must exactly match the model vocabulary",
    ))
    return merge(loaded, (; tokenizer))
end
