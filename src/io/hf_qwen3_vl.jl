using JSON3
using SHA: sha256

"""One immutable file in the frozen Qwen3-VL checkpoint."""
struct Qwen3VLAssetSpec
    name::String
    bytes::Int
    sha256::String
end

"""Frozen language-tower architecture for Qwen3-VL-2B-Instruct."""
struct Qwen3VLTextSpec
    vocab_size::Int
    hidden_size::Int
    intermediate_size::Int
    num_hidden_layers::Int
    num_attention_heads::Int
    num_key_value_heads::Int
    head_dim::Int
    rms_norm_eps::Float64
    rope_theta::Float64
    max_position_embeddings::Int
    mrope_interleaved::Bool
    mrope_section::NTuple{3,Int}
    tie_word_embeddings::Bool
    hidden_act::String
end

"""Frozen vision-tower and merger architecture for Qwen3-VL-2B-Instruct."""
struct Qwen3VLVisionSpec
    depth::Int
    hidden_size::Int
    intermediate_size::Int
    num_heads::Int
    in_channels::Int
    patch_size::Int
    temporal_patch_size::Int
    spatial_merge_size::Int
    out_hidden_size::Int
    num_position_embeddings::Int
    deepstack_visual_indexes::NTuple{3,Int}
    hidden_act::String
end

"""
    Qwen3VLCheckpointSpec

Immutable provenance and architecture contract for the official
`Qwen/Qwen3-VL-2B-Instruct` checkpoint. The two revision fields deliberately
name their registries: neither may be replaced with a moving `main` branch.
"""
struct Qwen3VLCheckpointSpec
    variant::Symbol
    model_id::String
    modelscope_revision::String
    hf_revision::String
    assets::Tuple
    tensor_count::Int
    tensor_bytes::Int
    parameter_count::Int
    image_token_id::Int
    video_token_id::Int
    vision_start_token_id::Int
    vision_end_token_id::Int
    bos_token_id::Int
    eos_token_id::Int
    text::Qwen3VLTextSpec
    vision::Qwen3VLVisionSpec
end

const _QWEN3_VL_CHECKPOINT_SPEC = Qwen3VLCheckpointSpec(
    :qwen3_vl_2b_instruct,
    "Qwen/Qwen3-VL-2B-Instruct",
    "ae9985b208c074c10cfbe3a61b5cb7268cdc9c53",
    "78448d793a7eb2f7a987a1da76d464384aa1becd",
    (
        Qwen3VLAssetSpec(
            ".gitattributes",
            2_227,
            "74f188c115dbc9614c048d24500099ee572b14830cacdfded13c86aa4fb9c7c7",
        ),
        Qwen3VLAssetSpec(
            "README.md",
            7_136,
            "5fc5be1ca9a3910399bd6239ee5086ab5d82a2a59c5d2b00e887a8835cc110e4",
        ),
        Qwen3VLAssetSpec(
            "chat_template.json",
            5_502,
            "6f8a6a55027e3da5160105556cda5dd69f6423f1c32645f6730d32de7773d0c4",
        ),
        Qwen3VLAssetSpec(
            "config.json",
            1_505,
            "bec4b3d446efa05807365c9e1cec03ac590836879d02f3a6da879971154bdd3b",
        ),
        Qwen3VLAssetSpec(
            "configuration.json",
            51,
            "2d4464e2ead06bc9bc718c781309ad1e7baded626d66e8dcdc8b469ba185faf0",
        ),
        Qwen3VLAssetSpec(
            "generation_config.json",
            269,
            "1e241830b48b397cb0900101421df5450baddc7adf01e5fc86b5615865f3bae4",
        ),
        Qwen3VLAssetSpec(
            "merges.txt",
            1_671_839,
            "599bab54075088774b1733fde865d5bd747cbcc7a547c5bc12610e874e26f5e3",
        ),
        Qwen3VLAssetSpec(
            "preprocessor_config.json",
            390,
            "27225450ac9c6529872ee1924fcb0962ff5634834f817040f444118116f4e516",
        ),
        Qwen3VLAssetSpec(
            "tokenizer_config.json",
            10_868,
            "c2da771801886ad9ae98181793ffd3dfb7f1af30f6f7c6a4e15d7dbba52e2399",
        ),
        Qwen3VLAssetSpec(
            "tokenizer.json",
            7_032_403,
            "a5d85b6dcc535e6b93115a9ef287e6132fdbf30270da6218194ba742261173c7",
        ),
        Qwen3VLAssetSpec(
            "video_preprocessor_config.json",
            385,
            "7768af27c1fafa9cc9011c1dc20067e03f8915e03b63504550e11d5066986d13",
        ),
        Qwen3VLAssetSpec(
            "vocab.json",
            2_776_833,
            "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910",
        ),
        Qwen3VLAssetSpec(
            "model.safetensors",
            4_255_140_312,
            "7de1838c87a5349b016c26a1c3f7d2bc400a3d485f95ef39a7059ffd734977a0",
        ),
    ),
    625,
    4_255_064_064,
    2_127_532_032,
    151_655,
    151_656,
    151_652,
    151_653,
    151_643,
    151_645,
    Qwen3VLTextSpec(
        151_936,
        2_048,
        6_144,
        28,
        16,
        8,
        128,
        1.0e-6,
        5_000_000.0,
        262_144,
        true,
        (24, 20, 20),
        true,
        "silu",
    ),
    Qwen3VLVisionSpec(
        24,
        1_024,
        4_096,
        16,
        3,
        16,
        2,
        2,
        2_048,
        2_304,
        (5, 11, 17),
        "gelu_pytorch_tanh",
    ),
)

"""Return the immutable official Qwen3-VL-2B-Instruct contract."""
qwen3_vl_checkpoint_spec() = _QWEN3_VL_CHECKPOINT_SPEC

# Frozen identities of the external Transformers 4.57.0 / Torch 2.7.1+cpu
# vision references. The reference files live outside the repository, so these
# code-reviewed hashes are the trust anchor rather than their adjacent JSON.
const _QWEN3_VL_VISION_REFERENCE_SHA256 = (
    float32="480d988d9f679c8090f8c80c8e5cd007e5a41c47e6bb5cc7ad2f16541cbe5f88",
    bfloat16="ecd904b8a110169c73c9814d23d43eabcc5a2593d0a746bbbda8bb9c308b36b8",
)

function _qwen3_vl_vision_reference_sha256(compute_dtype)
    name = String(compute_dtype)
    name == "float32" && return _QWEN3_VL_VISION_REFERENCE_SHA256.float32
    name == "bfloat16" && return _QWEN3_VL_VISION_REFERENCE_SHA256.bfloat16
    throw(ArgumentError(
        "unsupported Qwen3-VL vision reference dtype: $name",
    ))
end

function _qwen3_vl_text_parameter_count(text::Qwen3VLTextSpec)
    query_dim = text.num_attention_heads * text.head_dim
    kv_dim = text.num_key_value_heads * text.head_dim
    embedding = text.vocab_size * text.hidden_size
    attention = 2 * query_dim * text.hidden_size +
        2 * kv_dim * text.hidden_size +
        2 * text.head_dim
    mlp = 3 * text.hidden_size * text.intermediate_size
    norms = 2 * text.hidden_size
    return embedding +
        text.num_hidden_layers * (attention + mlp + norms) +
        text.hidden_size
end

function _qwen3_vl_vision_parameter_count(vision::Qwen3VLVisionSpec)
    patch = vision.hidden_size * vision.in_channels *
        vision.temporal_patch_size * vision.patch_size^2 + vision.hidden_size
    positions = vision.num_position_embeddings * vision.hidden_size
    block = 4 * vision.hidden_size +
        3 * vision.hidden_size^2 + 3 * vision.hidden_size +
        vision.hidden_size^2 + vision.hidden_size +
        2 * vision.hidden_size * vision.intermediate_size +
        vision.intermediate_size + vision.hidden_size
    merged_size = vision.hidden_size * vision.spatial_merge_size^2
    merger = 2 * vision.hidden_size +
        vision.intermediate_size * merged_size + vision.intermediate_size +
        vision.out_hidden_size * vision.intermediate_size +
        vision.out_hidden_size
    deep_merger = 2 * merged_size +
        vision.intermediate_size * merged_size + vision.intermediate_size +
        vision.out_hidden_size * vision.intermediate_size +
        vision.out_hidden_size
    return patch + positions + vision.depth * block + merger +
        length(vision.deepstack_visual_indexes) * deep_merger
end

"""Return the exact scalar count implied by the frozen multimodal checkpoint."""
function qwen3_vl_parameter_count(
    spec::Qwen3VLCheckpointSpec=qwen3_vl_checkpoint_spec(),
)
    count = _qwen3_vl_text_parameter_count(spec.text) +
        _qwen3_vl_vision_parameter_count(spec.vision)
    count == spec.parameter_count || throw(ArgumentError(
        "Qwen3-VL specification parameter count is internally inconsistent: " *
        "expected $(spec.parameter_count), computed $count",
    ))
    return count
end

function _qwen3_vl_exact_keys(object, expected, context::AbstractString)
    object isa JSON3.Object || throw(ArgumentError(
        "$context must be a JSON object",
    ))
    actual = Set(String(key) for key in keys(object))
    wanted = Set(String(key) for key in expected)
    actual == wanted || begin
        missing = sort!(collect(setdiff(wanted, actual)))
        unexpected = sort!(collect(setdiff(actual, wanted)))
        throw(ArgumentError(
            "$context fields do not match the frozen Qwen3-VL contract; " *
            "missing=$(repr(missing)), unexpected=$(repr(unexpected))",
        ))
    end
    return object
end

function _qwen3_vl_int(object, name::AbstractString, context::AbstractString)
    value = _json_required(object, name, context)
    value isa Integer && !(value isa Bool) || throw(ArgumentError(
        "`$name` must be an integer in $context",
    ))
    return Int(value)
end

function _qwen3_vl_number(object, name::AbstractString, context::AbstractString)
    value = _json_required(object, name, context)
    value isa Real && !(value isa Bool) || throw(ArgumentError(
        "`$name` must be numeric in $context",
    ))
    isfinite(value) || throw(ArgumentError(
        "`$name` must be finite in $context",
    ))
    return Float64(value)
end

function _qwen3_vl_expect(object, name, expected, context)
    actual = _json_required(object, name, context)
    actual == expected || throw(ArgumentError(
        "unsupported `$name` in $context: expected $(repr(expected)), " *
        "got $(repr(actual))",
    ))
    return actual
end

function _qwen3_vl_expect_int(object, name, expected, context)
    actual = _qwen3_vl_int(object, name, context)
    actual == expected || throw(ArgumentError(
        "unsupported `$name` in $context: expected $expected, got $actual",
    ))
    return actual
end

function _qwen3_vl_expect_number(object, name, expected, context)
    actual = _qwen3_vl_number(object, name, context)
    actual == expected || throw(ArgumentError(
        "unsupported `$name` in $context: expected $expected, got $actual",
    ))
    return actual
end

function _qwen3_vl_int_tuple(value, length_expected::Int, context::AbstractString)
    value isa JSON3.Array || throw(ArgumentError(
        "$context must be an array",
    ))
    length(value) == length_expected || throw(ArgumentError(
        "$context must contain exactly $length_expected integers",
    ))
    converted = map(value) do item
        item isa Integer && !(item isa Bool) || throw(ArgumentError(
            "$context must contain only integers",
        ))
        Int(item)
    end
    return Tuple(converted)
end

const _QWEN3_VL_ROOT_FIELDS = (
    "architectures",
    "image_token_id",
    "model_type",
    "text_config",
    "tie_word_embeddings",
    "transformers_version",
    "video_token_id",
    "vision_config",
    "vision_end_token_id",
    "vision_start_token_id",
)

const _QWEN3_VL_TEXT_FIELDS = (
    "attention_bias",
    "attention_dropout",
    "bos_token_id",
    "dtype",
    "eos_token_id",
    "head_dim",
    "hidden_act",
    "hidden_size",
    "initializer_range",
    "intermediate_size",
    "max_position_embeddings",
    "model_type",
    "num_attention_heads",
    "num_hidden_layers",
    "num_key_value_heads",
    "rms_norm_eps",
    "rope_scaling",
    "rope_theta",
    "tie_word_embeddings",
    "use_cache",
    "vocab_size",
)

const _QWEN3_VL_ROPE_FIELDS = (
    "mrope_interleaved",
    "mrope_section",
    "rope_type",
)

const _QWEN3_VL_VISION_FIELDS = (
    "deepstack_visual_indexes",
    "depth",
    "hidden_act",
    "hidden_size",
    "in_channels",
    "initializer_range",
    "intermediate_size",
    "model_type",
    "num_heads",
    "num_position_embeddings",
    "out_hidden_size",
    "patch_size",
    "spatial_merge_size",
    "temporal_patch_size",
)

"""
    load_hf_qwen3_vl_config(path; max_seq_len=nothing)

Parse the nested Qwen3-VL configuration and require the frozen official
2B-Instruct architecture. `max_seq_len` may reduce, but never extend, the
native text context. The historical nested vision `model_type="qwen3_vl"` is
accepted only for this exact official architecture; newer
`model_type="qwen3_vl_vision"` metadata is accepted under the same constraints.
"""
function load_hf_qwen3_vl_config(
    path::AbstractString;
    max_seq_len=nothing,
)
    spec = qwen3_vl_checkpoint_spec()
    config = _json_object(path)
    _qwen3_vl_exact_keys(config, _QWEN3_VL_ROOT_FIELDS, path)
    _qwen3_vl_expect(config, "model_type", "qwen3_vl", path)

    architectures = _json_required(config, "architectures", path)
    architectures isa JSON3.Array || throw(ArgumentError(
        "`architectures` must be an array in $path",
    ))
    length(architectures) == 1 &&
        architectures[1] == "Qwen3VLForConditionalGeneration" ||
        throw(ArgumentError(
            "$path must declare exactly `Qwen3VLForConditionalGeneration`",
        ))
    _qwen3_vl_expect(config, "tie_word_embeddings", true, path)
    _qwen3_vl_expect(
        config,
        "transformers_version",
        "4.57.0.dev0",
        path,
    )
    _qwen3_vl_expect_int(
        config,
        "image_token_id",
        spec.image_token_id,
        path,
    )
    _qwen3_vl_expect_int(
        config,
        "video_token_id",
        spec.video_token_id,
        path,
    )
    _qwen3_vl_expect_int(
        config,
        "vision_start_token_id",
        spec.vision_start_token_id,
        path,
    )
    _qwen3_vl_expect_int(
        config,
        "vision_end_token_id",
        spec.vision_end_token_id,
        path,
    )

    text_path = "$path:text_config"
    text = _json_required(config, "text_config", path)
    _qwen3_vl_exact_keys(text, _QWEN3_VL_TEXT_FIELDS, text_path)
    text_spec = spec.text
    _qwen3_vl_expect(text, "model_type", "qwen3_vl_text", text_path)
    _qwen3_vl_expect(text, "dtype", "bfloat16", text_path)
    _qwen3_vl_expect(text, "hidden_act", text_spec.hidden_act, text_path)
    _qwen3_vl_expect(text, "attention_bias", false, text_path)
    _qwen3_vl_expect_number(text, "attention_dropout", 0.0, text_path)
    _qwen3_vl_expect_number(text, "initializer_range", 0.02, text_path)
    _qwen3_vl_expect(text, "use_cache", true, text_path)
    _qwen3_vl_expect(
        text,
        "tie_word_embeddings",
        text_spec.tie_word_embeddings,
        text_path,
    )
    for (name, expected) in (
        "vocab_size" => text_spec.vocab_size,
        "hidden_size" => text_spec.hidden_size,
        "intermediate_size" => text_spec.intermediate_size,
        "num_hidden_layers" => text_spec.num_hidden_layers,
        "num_attention_heads" => text_spec.num_attention_heads,
        "num_key_value_heads" => text_spec.num_key_value_heads,
        "head_dim" => text_spec.head_dim,
        "max_position_embeddings" => text_spec.max_position_embeddings,
        "bos_token_id" => spec.bos_token_id,
        "eos_token_id" => spec.eos_token_id,
    )
        _qwen3_vl_expect_int(text, name, expected, text_path)
    end
    _qwen3_vl_expect_number(
        text,
        "rms_norm_eps",
        text_spec.rms_norm_eps,
        text_path,
    )
    _qwen3_vl_expect_number(
        text,
        "rope_theta",
        text_spec.rope_theta,
        text_path,
    )
    text_spec.num_attention_heads % text_spec.num_key_value_heads == 0 ||
        error("invalid frozen Qwen3-VL grouped-query attention contract")
    text_spec.num_attention_heads * text_spec.head_dim ==
        text_spec.hidden_size || error(
            "invalid frozen Qwen3-VL text hidden/head dimension contract",
        )

    rope_path = "$text_path:rope_scaling"
    rope = _json_required(text, "rope_scaling", text_path)
    _qwen3_vl_exact_keys(rope, _QWEN3_VL_ROPE_FIELDS, rope_path)
    _qwen3_vl_expect(rope, "rope_type", "default", rope_path)
    _qwen3_vl_expect(
        rope,
        "mrope_interleaved",
        text_spec.mrope_interleaved,
        rope_path,
    )
    mrope_section = _qwen3_vl_int_tuple(
        _json_required(rope, "mrope_section", rope_path),
        3,
        "$rope_path:mrope_section",
    )
    mrope_section == text_spec.mrope_section || throw(ArgumentError(
        "unsupported mrope_section in $rope_path: expected " *
        "$(text_spec.mrope_section), got $mrope_section",
    ))
    sum(mrope_section) == text_spec.head_dim ÷ 2 || throw(ArgumentError(
        "mrope_section must partition half of head_dim",
    ))

    vision_path = "$path:vision_config"
    vision = _json_required(config, "vision_config", path)
    _qwen3_vl_exact_keys(vision, _QWEN3_VL_VISION_FIELDS, vision_path)
    vision_spec = spec.vision
    vision_model_type = _json_required(vision, "model_type", vision_path)
    vision_model_type in ("qwen3_vl", "qwen3_vl_vision") ||
        throw(ArgumentError(
            "unsupported nested vision model_type " *
            "$(repr(vision_model_type)) in $vision_path",
        ))
    _qwen3_vl_expect(
        vision,
        "hidden_act",
        vision_spec.hidden_act,
        vision_path,
    )
    _qwen3_vl_expect_number(
        vision,
        "initializer_range",
        0.02,
        vision_path,
    )
    for (name, expected) in (
        "depth" => vision_spec.depth,
        "hidden_size" => vision_spec.hidden_size,
        "intermediate_size" => vision_spec.intermediate_size,
        "num_heads" => vision_spec.num_heads,
        "in_channels" => vision_spec.in_channels,
        "patch_size" => vision_spec.patch_size,
        "temporal_patch_size" => vision_spec.temporal_patch_size,
        "spatial_merge_size" => vision_spec.spatial_merge_size,
        "out_hidden_size" => vision_spec.out_hidden_size,
        "num_position_embeddings" => vision_spec.num_position_embeddings,
    )
        _qwen3_vl_expect_int(vision, name, expected, vision_path)
    end
    deepstack = _qwen3_vl_int_tuple(
        _json_required(
            vision,
            "deepstack_visual_indexes",
            vision_path,
        ),
        length(vision_spec.deepstack_visual_indexes),
        "$vision_path:deepstack_visual_indexes",
    )
    deepstack == vision_spec.deepstack_visual_indexes || throw(ArgumentError(
        "unsupported deepstack_visual_indexes in $vision_path: expected " *
        "$(vision_spec.deepstack_visual_indexes), got $deepstack",
    ))
    issorted(deepstack) && length(unique(deepstack)) == length(deepstack) &&
        all(index -> 0 <= index < vision_spec.depth, deepstack) ||
        throw(ArgumentError(
            "deepstack_visual_indexes must be unique, sorted, zero-based " *
            "vision block indexes",
        ))
    vision_spec.hidden_size % vision_spec.num_heads == 0 || throw(
        ArgumentError("vision hidden_size must be divisible by num_heads"),
    )
    vision_spec.out_hidden_size == text_spec.hidden_size || throw(
        ArgumentError(
            "vision out_hidden_size must equal the text hidden_size",
        ),
    )
    vision_spec.hidden_size * vision_spec.spatial_merge_size^2 ==
        vision_spec.intermediate_size || throw(ArgumentError(
            "spatial merger input must equal vision intermediate_size",
        ))

    resolved_max_seq_len = if max_seq_len === nothing
        text_spec.max_position_embeddings
    else
        max_seq_len isa Integer && !(max_seq_len isa Bool) ||
            throw(ArgumentError("max_seq_len must be an integer or nothing"))
        Int(max_seq_len)
    end
    1 <= resolved_max_seq_len <= text_spec.max_position_embeddings ||
        throw(ArgumentError(
            "max_seq_len must be in 1:$(text_spec.max_position_embeddings); " *
            "got $resolved_max_seq_len",
        ))

    return (;
        variant=spec.variant,
        model_type=:qwen3_vl,
        text=text_spec,
        vision=vision_spec,
        max_seq_len=resolved_max_seq_len,
        source_max_seq_len=text_spec.max_position_embeddings,
        image_token_id=spec.image_token_id,
        video_token_id=spec.video_token_id,
        vision_start_token_id=spec.vision_start_token_id,
        vision_end_token_id=spec.vision_end_token_id,
        bos_token_id=spec.bos_token_id,
        eos_token_id=spec.eos_token_id,
        tie_word_embeddings=text_spec.tie_word_embeddings,
    )
end

function _qwen3_vl_add_shape!(shapes, name::String, shape::Tuple)
    haskey(shapes, name) && error("duplicate Qwen3-VL tensor oracle name: $name")
    all(dimension -> dimension > 0, shape) || error(
        "non-positive Qwen3-VL tensor oracle dimension for $name: $shape",
    )
    shapes[name] = shape
    return shapes
end

"""
    qwen3_vl_expected_tensor_shapes([spec])

Return the exact Hugging Face safetensors name-to-shape oracle for the frozen
checkpoint. Shapes retain Hugging Face storage order, before any Julia/Lux
parameter conversion.
"""
function qwen3_vl_expected_tensor_shapes(
    spec::Qwen3VLCheckpointSpec=qwen3_vl_checkpoint_spec(),
)
    shapes = Dict{String,Tuple}()
    text = spec.text
    text_prefix = "model.language_model"
    _qwen3_vl_add_shape!(
        shapes,
        "$text_prefix.embed_tokens.weight",
        (text.vocab_size, text.hidden_size),
    )
    query_dim = text.num_attention_heads * text.head_dim
    kv_dim = text.num_key_value_heads * text.head_dim
    for layer in 0:(text.num_hidden_layers - 1)
        prefix = "$text_prefix.layers.$layer"
        _qwen3_vl_add_shape!(
            shapes,
            "$prefix.input_layernorm.weight",
            (text.hidden_size,),
        )
        _qwen3_vl_add_shape!(
            shapes,
            "$prefix.post_attention_layernorm.weight",
            (text.hidden_size,),
        )
        _qwen3_vl_add_shape!(
            shapes,
            "$prefix.self_attn.q_proj.weight",
            (query_dim, text.hidden_size),
        )
        for projection in ("k_proj", "v_proj")
            _qwen3_vl_add_shape!(
                shapes,
                "$prefix.self_attn.$projection.weight",
                (kv_dim, text.hidden_size),
            )
        end
        _qwen3_vl_add_shape!(
            shapes,
            "$prefix.self_attn.o_proj.weight",
            (text.hidden_size, query_dim),
        )
        for norm in ("q_norm", "k_norm")
            _qwen3_vl_add_shape!(
                shapes,
                "$prefix.self_attn.$norm.weight",
                (text.head_dim,),
            )
        end
        for projection in ("gate_proj", "up_proj")
            _qwen3_vl_add_shape!(
                shapes,
                "$prefix.mlp.$projection.weight",
                (text.intermediate_size, text.hidden_size),
            )
        end
        _qwen3_vl_add_shape!(
            shapes,
            "$prefix.mlp.down_proj.weight",
            (text.hidden_size, text.intermediate_size),
        )
    end
    _qwen3_vl_add_shape!(
        shapes,
        "$text_prefix.norm.weight",
        (text.hidden_size,),
    )
    text.tie_word_embeddings || _qwen3_vl_add_shape!(
        shapes,
        "lm_head.weight",
        (text.vocab_size, text.hidden_size),
    )

    vision = spec.vision
    vision_prefix = "model.visual"
    _qwen3_vl_add_shape!(
        shapes,
        "$vision_prefix.patch_embed.proj.weight",
        (
            vision.hidden_size,
            vision.in_channels,
            vision.temporal_patch_size,
            vision.patch_size,
            vision.patch_size,
        ),
    )
    _qwen3_vl_add_shape!(
        shapes,
        "$vision_prefix.patch_embed.proj.bias",
        (vision.hidden_size,),
    )
    _qwen3_vl_add_shape!(
        shapes,
        "$vision_prefix.pos_embed.weight",
        (vision.num_position_embeddings, vision.hidden_size),
    )
    for block in 0:(vision.depth - 1)
        prefix = "$vision_prefix.blocks.$block"
        for norm in ("norm1", "norm2"), parameter in ("weight", "bias")
            _qwen3_vl_add_shape!(
                shapes,
                "$prefix.$norm.$parameter",
                (vision.hidden_size,),
            )
        end
        _qwen3_vl_add_shape!(
            shapes,
            "$prefix.attn.qkv.weight",
            (3 * vision.hidden_size, vision.hidden_size),
        )
        _qwen3_vl_add_shape!(
            shapes,
            "$prefix.attn.qkv.bias",
            (3 * vision.hidden_size,),
        )
        _qwen3_vl_add_shape!(
            shapes,
            "$prefix.attn.proj.weight",
            (vision.hidden_size, vision.hidden_size),
        )
        _qwen3_vl_add_shape!(
            shapes,
            "$prefix.attn.proj.bias",
            (vision.hidden_size,),
        )
        _qwen3_vl_add_shape!(
            shapes,
            "$prefix.mlp.linear_fc1.weight",
            (vision.intermediate_size, vision.hidden_size),
        )
        _qwen3_vl_add_shape!(
            shapes,
            "$prefix.mlp.linear_fc1.bias",
            (vision.intermediate_size,),
        )
        _qwen3_vl_add_shape!(
            shapes,
            "$prefix.mlp.linear_fc2.weight",
            (vision.hidden_size, vision.intermediate_size),
        )
        _qwen3_vl_add_shape!(
            shapes,
            "$prefix.mlp.linear_fc2.bias",
            (vision.hidden_size,),
        )
    end

    merged_size = vision.hidden_size * vision.spatial_merge_size^2
    merger_prefix = "$vision_prefix.merger"
    for parameter in ("weight", "bias")
        _qwen3_vl_add_shape!(
            shapes,
            "$merger_prefix.norm.$parameter",
            (vision.hidden_size,),
        )
    end
    _qwen3_vl_add_shape!(
        shapes,
        "$merger_prefix.linear_fc1.weight",
        (vision.intermediate_size, merged_size),
    )
    _qwen3_vl_add_shape!(
        shapes,
        "$merger_prefix.linear_fc1.bias",
        (vision.intermediate_size,),
    )
    _qwen3_vl_add_shape!(
        shapes,
        "$merger_prefix.linear_fc2.weight",
        (vision.out_hidden_size, vision.intermediate_size),
    )
    _qwen3_vl_add_shape!(
        shapes,
        "$merger_prefix.linear_fc2.bias",
        (vision.out_hidden_size,),
    )

    for merger in 0:(length(vision.deepstack_visual_indexes) - 1)
        prefix = "$vision_prefix.deepstack_merger_list.$merger"
        for parameter in ("weight", "bias")
            _qwen3_vl_add_shape!(
                shapes,
                "$prefix.norm.$parameter",
                (merged_size,),
            )
        end
        _qwen3_vl_add_shape!(
            shapes,
            "$prefix.linear_fc1.weight",
            (vision.intermediate_size, merged_size),
        )
        _qwen3_vl_add_shape!(
            shapes,
            "$prefix.linear_fc1.bias",
            (vision.intermediate_size,),
        )
        _qwen3_vl_add_shape!(
            shapes,
            "$prefix.linear_fc2.weight",
            (vision.out_hidden_size, vision.intermediate_size),
        )
        _qwen3_vl_add_shape!(
            shapes,
            "$prefix.linear_fc2.bias",
            (vision.out_hidden_size,),
        )
    end

    length(shapes) == spec.tensor_count || error(
        "Qwen3-VL tensor oracle has $(length(shapes)) tensors; " *
        "expected $(spec.tensor_count)",
    )
    parameters = sum(prod(shape) for shape in values(shapes))
    parameters == spec.parameter_count || error(
        "Qwen3-VL tensor oracle has $parameters parameters; " *
        "expected $(spec.parameter_count)",
    )
    return shapes
end

_qwen3_vl_sha256_file(path::AbstractString) = open(path, "r") do io
    bytes2hex(sha256(io))
end

"""
    verify_qwen3_vl_checkpoint(model_dir; max_seq_len=nothing)

Verify every frozen asset and the complete safetensors header contract before
the checkpoint may be used. This deliberately performs a full 4.25 GB SHA-256
read; callers wanting only header inspection should use
`qwen3_vl_expected_tensor_shapes` with `open_safetensors_reader` explicitly.
"""
function verify_qwen3_vl_checkpoint(
    model_dir::AbstractString;
    max_seq_len=nothing,
)
    isdir(model_dir) || throw(ArgumentError(
        "Qwen3-VL model directory does not exist: $model_dir",
    ))
    spec = qwen3_vl_checkpoint_spec()
    length(spec.assets) == 13 || error(
        "Qwen3-VL asset manifest must contain exactly 13 files",
    )
    verified_assets = NamedTuple[]
    for asset in spec.assets
        isempty(asset.sha256) && error(
            "Qwen3-VL asset manifest has an empty hash for $(asset.name)",
        )
        path = joinpath(model_dir, asset.name)
        isfile(path) || throw(ArgumentError(
            "required Qwen3-VL asset does not exist: $path",
        ))
        actual_bytes = filesize(path)
        actual_bytes == asset.bytes || throw(ArgumentError(
            "Qwen3-VL asset size mismatch for $(asset.name): expected " *
            "$(asset.bytes), got $actual_bytes",
        ))
        actual_sha256 = _qwen3_vl_sha256_file(path)
        actual_sha256 == asset.sha256 || throw(ArgumentError(
            "Qwen3-VL asset checksum mismatch for $(asset.name): expected " *
            "$(asset.sha256), computed $actual_sha256",
        ))
        push!(verified_assets, (;
            name=asset.name,
            bytes=actual_bytes,
            sha256=actual_sha256,
        ))
    end

    config = load_hf_qwen3_vl_config(
        joinpath(model_dir, "config.json");
        max_seq_len,
    )
    expected = qwen3_vl_expected_tensor_shapes(spec)
    reader = open_safetensors_reader(joinpath(model_dir, "model.safetensors"))
    actual_names = Set(String(name) for name in keys(reader))
    expected_names = Set(keys(expected))
    actual_names == expected_names || begin
        missing = sort!(collect(setdiff(expected_names, actual_names)))
        unexpected = sort!(collect(setdiff(actual_names, expected_names)))
        throw(ArgumentError(
            "Qwen3-VL safetensors names do not match the frozen contract; " *
            "missing=$(repr(missing)), unexpected=$(repr(unexpected))",
        ))
    end
    length(reader.locations) == spec.tensor_count || throw(ArgumentError(
        "Qwen3-VL safetensors tensor count mismatch: expected " *
        "$(spec.tensor_count), got $(length(reader.locations))",
    ))

    payload_bytes = 0
    parameter_count = 0
    weight_path = abspath(joinpath(model_dir, "model.safetensors"))
    for name in keys(expected)
        location = reader.locations[name]
        location.path == weight_path || throw(ArgumentError(
            "Qwen3-VL tensor `$name` resolved outside the frozen weight file",
        ))
        location.dtype == "BF16" || throw(ArgumentError(
            "Qwen3-VL tensor `$name` must be BF16, got $(location.dtype)",
        ))
        actual_shape = Tuple(location.shape)
        expected_shape = expected[name]
        actual_shape == expected_shape || throw(ArgumentError(
            "Qwen3-VL tensor `$name` has shape $actual_shape; expected " *
            "$expected_shape",
        ))
        tensor_parameters = prod(expected_shape)
        tensor_bytes = location.data_stop - location.data_start
        tensor_bytes == 2 * tensor_parameters || throw(ArgumentError(
            "Qwen3-VL tensor `$name` payload is inconsistent with BF16 shape",
        ))
        parameter_count += tensor_parameters
        payload_bytes += tensor_bytes
    end
    parameter_count == spec.parameter_count || throw(ArgumentError(
        "Qwen3-VL safetensors parameter count mismatch: expected " *
        "$(spec.parameter_count), got $parameter_count",
    ))
    payload_bytes == spec.tensor_bytes || throw(ArgumentError(
        "Qwen3-VL safetensors payload mismatch: expected " *
        "$(spec.tensor_bytes) bytes, got $payload_bytes",
    ))
    qwen3_vl_parameter_count(spec) == parameter_count || error(
        "Qwen3-VL architecture, tensor oracle, and payload disagree",
    )

    return (;
        spec,
        variant=spec.variant,
        model_id=spec.model_id,
        modelscope_revision=spec.modelscope_revision,
        hf_revision=spec.hf_revision,
        assets=Tuple(verified_assets),
        tensor_count=length(reader.locations),
        parameter_count,
        tensor_bytes=payload_bytes,
        config,
        source=abspath(model_dir),
    )
end
