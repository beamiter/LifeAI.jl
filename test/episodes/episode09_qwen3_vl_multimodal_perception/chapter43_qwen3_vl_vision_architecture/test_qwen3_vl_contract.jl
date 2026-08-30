using JSON3
using SHA: sha256
using Test
using LifeAI: Qwen3VLAssetSpec,
    Qwen3VLCheckpointSpec,
    Qwen3VLTextSpec,
    Qwen3VLVisionSpec,
    _qwen3_vl_text_parameter_count,
    load_hf_qwen3_vl_config,
    _qwen3_vl_vision_reference_sha256,
    _qwen3_vl_vision_parameter_count,
    load_hf_qwen3_vl_processor_config,
    qwen3_vl_checkpoint_spec,
    qwen3_vl_expected_tensor_shapes,
    qwen3_vl_parameter_count,
    qwen3_vl_processor_spec,
    verify_qwen3_vl_checkpoint

const _CH43_VL_FIXTURES = joinpath(@__DIR__, "fixtures")
const _CH43_VL_CONFIG = joinpath(_CH43_VL_FIXTURES, "config.json")
const _CH43_VL_PREPROCESSOR =
    joinpath(_CH43_VL_FIXTURES, "preprocessor_config.json")

_ch43_sha256(path) = bytes2hex(sha256(read(path)))

function _ch43_load_mutated_config(mutator)
    document = JSON3.read(read(_CH43_VL_CONFIG, String), Dict{String,Any})
    mutator(document)
    return mktemp() do path, io
        JSON3.write(io, document)
        close(io)
        load_hf_qwen3_vl_config(path)
    end
end

function _ch43_load_mutated_processor(mutator)
    document = JSON3.read(
        read(_CH43_VL_PREPROCESSOR, String),
        Dict{String,Any},
    )
    mutator(document)
    return mktemp() do path, io
        JSON3.write(io, document)
        close(io)
        load_hf_qwen3_vl_processor_config(path)
    end
end

function _ch43_captured_error(thunk)
    try
        thunk()
    catch error
        return error
    end
    error("expected Qwen3-VL call to fail")
end

function _ch43_raw_processor_error(payload::AbstractString)
    return mktemp() do path, io
        write(io, payload)
        close(io)
        failure = _ch43_captured_error() do
            load_hf_qwen3_vl_processor_config(path)
        end
        return (; failure, path=String(path))
    end
end

@testset "Qwen3-VL asset specifications are strict" begin
    spec = Qwen3VLAssetSpec(
        SubString("xmodel.safetensors", 2),
        big(7),
        SubString("xhash", 2),
    )
    @test spec.name === "model.safetensors"
    @test spec.bytes === 7
    @test spec.sha256 === "hash"

    too_large = big(typemax(Int)) + 1
    for (arguments, message) in (
        ((:model, 7, "hash"), "name must be a string"),
        (("model", 7, :hash), "sha256 must be a string"),
        (("model", true, "hash"), "bytes must be an integer"),
        (("model", 7.0, "hash"), "bytes must be an integer"),
        (("model", -1, "hash"), "bytes must be non-negative"),
        (
            ("model", too_large, "hash"),
            "bytes is outside the host integer range",
        ),
    )
        failure = _ch43_captured_error() do
            Qwen3VLAssetSpec(arguments...)
        end
        @test failure isa ArgumentError
        @test sprint(showerror, failure) == "ArgumentError: Qwen3-VL asset $message"
    end
end

@testset "Qwen3-VL text specifications are strict" begin
    valid = (
        big(32),
        Int32(16),
        UInt8(32),
        big(0),
        Int16(2),
        big(1),
        Int128(8),
        Float32(1.0e-6),
        Int32(10_000),
        big(64),
        true,
        (Int32(2), big(1), UInt8(1)),
        true,
        SubString("xsilu", 2),
    )
    spec = Qwen3VLTextSpec(valid...)
    @test spec.vocab_size === 32
    @test spec.num_hidden_layers === 0
    @test spec.rms_norm_eps isa Float64
    @test spec.rope_theta === 10_000.0
    @test spec.mrope_section === (2, 1, 1)
    @test spec.hidden_act === "silu"

    integer_fields = (
        1 => "vocab_size",
        2 => "hidden_size",
        3 => "intermediate_size",
        4 => "num_hidden_layers",
        5 => "num_attention_heads",
        6 => "num_key_value_heads",
        7 => "head_dim",
        10 => "max_position_embeddings",
    )
    for (index, label) in integer_fields
        failure = _ch43_captured_error() do
            Qwen3VLTextSpec(Base.setindex(valid, true, index)...)
        end
        @test failure isa ArgumentError
        @test sprint(showerror, failure) ==
            "ArgumentError: Qwen3-VL text $label must be an integer"

        invalid = index == 4 ? -1 : 0
        qualifier = index == 4 ? "non-negative" : "positive"
        failure = _ch43_captured_error() do
            Qwen3VLTextSpec(Base.setindex(valid, invalid, index)...)
        end
        @test failure isa ArgumentError
        @test sprint(showerror, failure) ==
            "ArgumentError: Qwen3-VL text $label must be $qualifier"
    end

    too_large = big(typemax(Int)) + 1
    for (index, value, message) in (
        (1, 1.0, "vocab_size must be an integer"),
        (1, too_large, "vocab_size is outside the host integer range"),
        (11, 1, "mrope_interleaved must be a Bool"),
        (13, 1, "tie_word_embeddings must be a Bool"),
        (14, :silu, "hidden_act must be a string"),
    )
        failure = _ch43_captured_error() do
            Qwen3VLTextSpec(Base.setindex(valid, value, index)...)
        end
        @test failure isa ArgumentError
        @test sprint(showerror, failure) == "ArgumentError: Qwen3-VL text $message"
    end

    for index in (8, 9)
        label = index == 8 ? "rms_norm_eps" : "rope_theta"
        for (value, message) in (
            (true, "must be a real number"),
            ("1", "must be a real number"),
            (0, "must be positive and finite"),
            (NaN, "must be positive and finite"),
            (Inf, "must be positive and finite"),
            (big(10)^1_000, "must be positive and finite"),
        )
            failure = _ch43_captured_error() do
                Qwen3VLTextSpec(Base.setindex(valid, value, index)...)
            end
            @test failure isa ArgumentError
            @test sprint(showerror, failure) ==
                "ArgumentError: Qwen3-VL text $label $message"
        end
    end

    for (section, message) in (
        ([2, 1, 1], "must be a tuple of three integers"),
        ((2, 1), "must be a tuple of three integers"),
        ((true, 1, 1), "[1] must be an integer"),
        ((-1, 1, 1), "[1] must be non-negative"),
        ((too_large, 1, 1), "[1] is outside the host integer range"),
    )
        failure = _ch43_captured_error() do
            Qwen3VLTextSpec(Base.setindex(valid, section, 12)...)
        end
        @test failure isa ArgumentError
        separator = startswith(message, "[") ? "" : " "
        @test sprint(showerror, failure) ==
            "ArgumentError: Qwen3-VL text mrope_section$separator$message"
    end

    invalid_gqa = Base.setindex(valid, 3, 5)
    invalid_gqa = Base.setindex(invalid_gqa, 2, 6)
    gqa_failure = _ch43_captured_error() do
        Qwen3VLTextSpec(invalid_gqa...)
    end
    @test gqa_failure isa ArgumentError
    @test sprint(showerror, gqa_failure) ==
        "ArgumentError: Qwen3-VL text num_heads must be divisible by num_kv_heads"

    query_overflow = Base.setindex(valid, typemax(Int), 5)
    query_overflow = Base.setindex(query_overflow, 2, 7)
    query_failure = _ch43_captured_error() do
        Qwen3VLTextSpec(query_overflow...)
    end
    @test query_failure isa ArgumentError
    @test sprint(showerror, query_failure) ==
        "ArgumentError: Qwen3-VL text query projection width exceeds the host integer range"

    kv_overflow = Base.setindex(valid, 1, 5)
    kv_overflow = Base.setindex(kv_overflow, typemax(Int), 6)
    kv_overflow = Base.setindex(kv_overflow, 2, 7)
    kv_failure = _ch43_captured_error() do
        Qwen3VLTextSpec(kv_overflow...)
    end
    @test kv_failure isa ArgumentError
    @test sprint(showerror, kv_failure) ==
        "ArgumentError: Qwen3-VL text key/value projection width exceeds the host integer range"
end

@testset "Qwen3-VL vision specifications are strict" begin
    valid = (
        big(0),
        Int32(16),
        UInt8(32),
        big(2),
        Int16(3),
        big(4),
        Int128(2),
        UInt8(2),
        big(16),
        Int32(64),
        (Int32(0), big(1), UInt8(2)),
        SubString("xgelu", 2),
    )
    spec = Qwen3VLVisionSpec(valid...)
    @test spec.depth === 0
    @test spec.hidden_size === 16
    @test spec.spatial_merge_size === 2
    @test spec.deepstack_visual_indexes === (0, 1, 2)
    @test spec.hidden_act === "gelu"

    integer_fields = (
        1 => "depth",
        2 => "hidden_size",
        3 => "intermediate_size",
        4 => "num_heads",
        5 => "in_channels",
        6 => "patch_size",
        7 => "temporal_patch_size",
        8 => "spatial_merge_size",
        9 => "out_hidden_size",
        10 => "num_position_embeddings",
    )
    for (index, label) in integer_fields
        failure = _ch43_captured_error() do
            Qwen3VLVisionSpec(Base.setindex(valid, true, index)...)
        end
        @test failure isa ArgumentError
        @test sprint(showerror, failure) ==
            "ArgumentError: Qwen3-VL vision $label must be an integer"

        invalid = index == 1 ? -1 : 0
        qualifier = index == 1 ? "non-negative" : "positive"
        failure = _ch43_captured_error() do
            Qwen3VLVisionSpec(Base.setindex(valid, invalid, index)...)
        end
        @test failure isa ArgumentError
        @test sprint(showerror, failure) ==
            "ArgumentError: Qwen3-VL vision $label must be $qualifier"
    end

    too_large = big(typemax(Int)) + 1
    for (index, value, message) in (
        (2, 1.0, "hidden_size must be an integer"),
        (2, too_large, "hidden_size is outside the host integer range"),
        (12, :gelu, "hidden_act must be a string"),
    )
        failure = _ch43_captured_error() do
            Qwen3VLVisionSpec(Base.setindex(valid, value, index)...)
        end
        @test failure isa ArgumentError
        @test sprint(showerror, failure) == "ArgumentError: Qwen3-VL vision $message"
    end

    for (indexes, message) in (
        ([0, 1, 2], "must be a tuple of three integers"),
        ((0, 1), "must be a tuple of three integers"),
        ((true, 1, 2), "[1] must be an integer"),
        ((-1, 1, 2), "[1] must be non-negative"),
        ((too_large, 1, 2), "[1] is outside the host integer range"),
    )
        failure = _ch43_captured_error() do
            Qwen3VLVisionSpec(Base.setindex(valid, indexes, 11)...)
        end
        @test failure isa ArgumentError
        separator = startswith(message, "[") ? "" : " "
        @test sprint(showerror, failure) ==
            "ArgumentError: Qwen3-VL vision " *
            "deepstack_visual_indexes$separator$message"
    end

    merged_width_failure = _ch43_captured_error() do
        Qwen3VLVisionSpec(
            Base.setindex(valid, typemax(Int), 8)...,
        )
    end
    @test merged_width_failure isa ArgumentError
    @test sprint(showerror, merged_width_failure) ==
        "ArgumentError: Qwen3-VL vision merged width exceeds the host integer range"

    qkv_width_values = Base.setindex(valid, 1, 8)
    qkv_width_values = Base.setindex(qkv_width_values, 1, 4)
    qkv_width_values = Base.setindex(
        qkv_width_values,
        typemax(Int) ÷ 3 + 1,
        2,
    )
    qkv_width_failure = _ch43_captured_error() do
        Qwen3VLVisionSpec(qkv_width_values...)
    end
    @test qkv_width_failure isa ArgumentError
    @test sprint(showerror, qkv_width_failure) ==
        "ArgumentError: Qwen3-VL vision QKV width exceeds the host integer range"

    head_geometry_failure = _ch43_captured_error() do
        Qwen3VLVisionSpec(Base.setindex(valid, 15, 2)...)
    end
    @test head_geometry_failure isa ArgumentError
    @test sprint(showerror, head_geometry_failure) ==
        "ArgumentError: Qwen3-VL vision hidden_size must be divisible by num_heads"
end

@testset "Qwen3-VL checkpoint specifications are strict" begin
    base = qwen3_vl_checkpoint_spec()
    strings = ntuple(_ -> SubString("xvalue", 2), 3)
    valid = (
        :fixture,
        strings...,
        (),
        ntuple(_ -> big(0), 9)...,
        base.text,
        base.vision,
    )
    spec = Qwen3VLCheckpointSpec(valid...)
    @test spec.variant === :fixture
    @test spec.model_id === "value"
    @test spec.modelscope_revision === "value"
    @test spec.hf_revision === "value"
    @test spec.tensor_count === 0
    @test spec.eos_token_id === 0
    @test isempty(spec.assets)

    integer_fields = (
        6 => "tensor_count",
        7 => "tensor_bytes",
        8 => "parameter_count",
        9 => "image_token_id",
        10 => "video_token_id",
        11 => "vision_start_token_id",
        12 => "vision_end_token_id",
        13 => "bos_token_id",
        14 => "eos_token_id",
    )
    for (index, label) in integer_fields
        for (value, message) in (
            (true, "must be an integer"),
            (-1, "must be non-negative"),
        )
            failure = _ch43_captured_error() do
                Qwen3VLCheckpointSpec(Base.setindex(valid, value, index)...)
            end
            @test failure isa ArgumentError
            @test sprint(showerror, failure) ==
                "ArgumentError: Qwen3-VL checkpoint $label $message"
        end
    end

    too_large = big(typemax(Int)) + 1
    for (index, value, message) in (
        (6, 1.0, "tensor_count must be an integer"),
        (6, too_large, "tensor_count is outside the host integer range"),
        (1, "fixture", "variant must be a Symbol"),
        (2, :model, "model_id must be a string"),
        (5, [], "assets must be a tuple"),
        (
            5,
            ((; name="asset", bytes=0, sha256="hash"),),
            "assets must contain Qwen3VLAssetSpec values",
        ),
        (
            5,
            (
                Qwen3VLAssetSpec("duplicate", 1, "first"),
                Qwen3VLAssetSpec("duplicate", 2, "second"),
            ),
            "asset names must be unique",
        ),
        (15, (;), "text must be a Qwen3VLTextSpec"),
        (16, (;), "vision must be a Qwen3VLVisionSpec"),
    )
        failure = _ch43_captured_error() do
            Qwen3VLCheckpointSpec(Base.setindex(valid, value, index)...)
        end
        @test failure isa ArgumentError
        @test sprint(showerror, failure) ==
            "ArgumentError: Qwen3-VL checkpoint $message"
    end
end

@testset "Qwen3-VL frozen checkpoint and tensor contract" begin
    spec = qwen3_vl_checkpoint_spec()
    @test spec.variant == :qwen3_vl_2b_instruct
    @test spec.model_id == "Qwen/Qwen3-VL-2B-Instruct"
    @test spec.modelscope_revision ==
        "ae9985b208c074c10cfbe3a61b5cb7268cdc9c53"
    @test spec.hf_revision ==
        "78448d793a7eb2f7a987a1da76d464384aa1becd"
    @test length(spec.assets) == 13
    @test sum(asset.bytes for asset in spec.assets) == 4_266_649_720
    @test all(asset -> occursin(r"^[0-9a-f]{64}$", asset.sha256), spec.assets)
    @test Tuple(asset.name for asset in spec.assets) == (
        ".gitattributes",
        "README.md",
        "chat_template.json",
        "config.json",
        "configuration.json",
        "generation_config.json",
        "merges.txt",
        "preprocessor_config.json",
        "tokenizer_config.json",
        "tokenizer.json",
        "video_preprocessor_config.json",
        "vocab.json",
        "model.safetensors",
    )

    assets = Dict(asset.name => asset for asset in spec.assets)
    @test assets["config.json"].bytes == filesize(_CH43_VL_CONFIG) == 1_505
    @test assets["config.json"].sha256 == _ch43_sha256(_CH43_VL_CONFIG) ==
        "bec4b3d446efa05807365c9e1cec03ac590836879d02f3a6da879971154bdd3b"
    @test assets["preprocessor_config.json"].bytes ==
        filesize(_CH43_VL_PREPROCESSOR) == 390
    @test assets["preprocessor_config.json"].sha256 ==
        _ch43_sha256(_CH43_VL_PREPROCESSOR) ==
        "27225450ac9c6529872ee1924fcb0962ff5634834f817040f444118116f4e516"
    @test assets["model.safetensors"].bytes == 4_255_140_312
    @test assets["model.safetensors"].sha256 ==
        "7de1838c87a5349b016c26a1c3f7d2bc400a3d485f95ef39a7059ffd734977a0"

    @test spec.tensor_count == 625
    @test spec.parameter_count == 2_127_532_032
    @test spec.tensor_bytes == 4_255_064_064
    @test qwen3_vl_parameter_count(spec) == spec.parameter_count
    @test _qwen3_vl_text_parameter_count(spec.text) == 1_720_574_976
    @test _qwen3_vl_vision_parameter_count(spec.vision) == 406_957_056
    shapes = qwen3_vl_expected_tensor_shapes(spec)
    @test length(shapes) == spec.tensor_count
    @test sum(prod(shape) for shape in values(shapes)) == spec.parameter_count
    @test 2 * sum(prod(shape) for shape in values(shapes)) == spec.tensor_bytes
    @test shapes["model.language_model.embed_tokens.weight"] == (151_936, 2_048)
    @test shapes["model.visual.patch_embed.proj.weight"] == (1_024, 3, 2, 16, 16)
    @test shapes["model.visual.blocks.23.mlp.linear_fc2.weight"] == (1_024, 4_096)
    @test shapes["model.visual.merger.norm.weight"] == (1_024,)
    @test shapes["model.visual.merger.linear_fc1.weight"] == (4_096, 4_096)
    @test shapes["model.visual.deepstack_merger_list.2.norm.weight"] == (4_096,)
    @test shapes["model.visual.deepstack_merger_list.2.linear_fc2.weight"] ==
        (2_048, 4_096)

    untied_text_fields = map(fieldnames(Qwen3VLTextSpec)) do name
        name === :tie_word_embeddings && return false
        return getfield(spec.text, name)
    end
    untied_text = Qwen3VLTextSpec(untied_text_fields...)
    untied_spec_fields = map(fieldnames(Qwen3VLCheckpointSpec)) do name
        name === :tensor_count && return spec.tensor_count + 1
        name === :tensor_bytes && return spec.tensor_bytes + 622_329_856
        name === :parameter_count && return spec.parameter_count + 311_164_928
        name === :text && return untied_text
        return getfield(spec, name)
    end
    untied_spec = Qwen3VLCheckpointSpec(untied_spec_fields...)
    untied_shapes = qwen3_vl_expected_tensor_shapes(untied_spec)
    @test untied_shapes["lm_head.weight"] == (151_936, 2_048)
    @test qwen3_vl_parameter_count(untied_spec) == 2_438_696_960
    @test sum(prod(shape) for shape in values(untied_shapes)) ==
        qwen3_vl_parameter_count(untied_spec)

    text_overflow = Qwen3VLTextSpec(
        typemax(Int),
        2,
        1,
        1,
        1,
        1,
        1,
        1.0e-6,
        1.0e4,
        1,
        true,
        (1, 1, 1),
        true,
        "silu",
    )
    @test_throws ArgumentError _qwen3_vl_text_parameter_count(text_overflow)

    vision_overflow = Qwen3VLVisionSpec(
        1,
        1,
        typemax(Int),
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        (0, 0, 0),
        "gelu",
    )
    @test_throws ArgumentError _qwen3_vl_vision_parameter_count(vision_overflow)

    half = typemax(Int) ÷ 2
    half_text = Qwen3VLTextSpec(
        half,
        1,
        1,
        1,
        1,
        1,
        2,
        1.0e-6,
        1.0e4,
        1,
        true,
        (1, 0, 0),
        true,
        "silu",
    )
    half_vision = Qwen3VLVisionSpec(
        3,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        half,
        (0, 1, 2),
        "gelu",
    )
    @test _qwen3_vl_text_parameter_count(half_text) == half + 18
    @test _qwen3_vl_vision_parameter_count(half_vision) == half + 74
    total_overflow_spec = Qwen3VLCheckpointSpec(
        :overflow,
        "overflow",
        "overflow",
        "overflow",
        (),
        0,
        0,
        0,
        1,
        1,
        1,
        1,
        1,
        1,
        half_text,
        half_vision,
    )
    @test_throws ArgumentError qwen3_vl_parameter_count(total_overflow_spec)

    maximum_dimension = typemax(Int)
    product_text = Qwen3VLTextSpec(
        1,
        1,
        1,
        0,
        1,
        1,
        2,
        1.0e-6,
        1.0e4,
        1,
        true,
        (1, 0, 0),
        true,
        "silu",
    )
    product_vision = Qwen3VLVisionSpec(
        3,
        1,
        1,
        1,
        maximum_dimension,
        1,
        maximum_dimension,
        1,
        1,
        1,
        (0, 1, 2),
        "gelu",
    )
    product_spec = Qwen3VLCheckpointSpec(
        :overflow,
        "overflow",
        "overflow",
        "overflow",
        (),
        65,
        0,
        77,
        1,
        1,
        1,
        1,
        1,
        1,
        product_text,
        product_vision,
    )
    @test_throws ArgumentError qwen3_vl_expected_tensor_shapes(product_spec)

    aggregate_text = Qwen3VLTextSpec(
        maximum_dimension,
        1,
        1,
        0,
        1,
        1,
        2,
        1.0e-6,
        1.0e4,
        1,
        true,
        (1, 0, 0),
        false,
        "silu",
    )
    aggregate_vision = Qwen3VLVisionSpec(
        3,
        1,
        1,
        1,
        maximum_dimension,
        1,
        1,
        1,
        1,
        maximum_dimension,
        (0, 1, 2),
        "gelu",
    )
    aggregate_spec = Qwen3VLCheckpointSpec(
        :overflow,
        "overflow",
        "overflow",
        "overflow",
        (),
        66,
        0,
        70,
        1,
        1,
        1,
        1,
        1,
        1,
        aggregate_text,
        aggregate_vision,
    )
    @test_throws ArgumentError qwen3_vl_expected_tensor_shapes(aggregate_spec)

    half_byte_text = Qwen3VLTextSpec(
        half,
        1,
        1,
        0,
        1,
        1,
        2,
        1.0e-6,
        1.0e4,
        1,
        true,
        (1, 0, 0),
        true,
        "silu",
    )
    half_byte_vision = Qwen3VLVisionSpec(
        0,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        (0, 0, 0),
        "gelu",
    )
    byte_overflow_spec = Qwen3VLCheckpointSpec(
        :overflow,
        "overflow",
        "overflow",
        "overflow",
        (),
        29,
        0,
        half + 28,
        1,
        1,
        1,
        1,
        1,
        1,
        half_byte_text,
        half_byte_vision,
    )
    @test_throws ArgumentError qwen3_vl_expected_tensor_shapes(byte_overflow_spec)

    @test spec.text.mrope_interleaved
    @test spec.text.mrope_section == (24, 20, 20)
    @test sum(spec.text.mrope_section) == spec.text.head_dim ÷ 2
    @test spec.vision.deepstack_visual_indexes == (5, 11, 17)
    @test spec.vision.hidden_size * spec.vision.spatial_merge_size^2 ==
        spec.vision.intermediate_size
    @test spec.vision.out_hidden_size == spec.text.hidden_size
end

@testset "Qwen3-VL frozen external vision oracle identities" begin
    @test _qwen3_vl_vision_reference_sha256("float32") ==
        "480d988d9f679c8090f8c80c8e5cd007e5a41c47e6bb5cc7ad2f16541cbe5f88"
    @test _qwen3_vl_vision_reference_sha256(:bfloat16) ==
        "ecd904b8a110169c73c9814d23d43eabcc5a2593d0a746bbbda8bb9c308b36b8"
    @test_throws ArgumentError _qwen3_vl_vision_reference_sha256("float16")
end

@testset "Qwen3-VL context requests fail before checkpoint I/O" begin
    mktempdir() do directory
        config_path = joinpath(directory, "config.json")
        too_large = big(typemax(Int)) + 1
        for (value, message) in (
            true => "max_seq_len must be an integer",
            0 => "max_seq_len must be positive",
            too_large => "max_seq_len is outside the host integer range",
            262_145 => "max_seq_len must be in 1:262144; got 262145",
        )
            for request in (
                () -> load_hf_qwen3_vl_config(
                    config_path;
                    max_seq_len=value,
                ),
                () -> verify_qwen3_vl_checkpoint(
                    directory;
                    max_seq_len=value,
                ),
            )
                failure = _ch43_captured_error(request)
                @test failure isa ArgumentError
                @test sprint(showerror, failure) == "ArgumentError: $message"
            end
        end

        config_io_failure = _ch43_captured_error() do
            load_hf_qwen3_vl_config(
                config_path;
                max_seq_len=Int32(4_096),
            )
        end
        @test config_io_failure isa ArgumentError
        @test occursin(
            "JSON file does not exist",
            sprint(showerror, config_io_failure),
        )

        asset_io_failure = _ch43_captured_error() do
            verify_qwen3_vl_checkpoint(
                directory;
                max_seq_len=big(4_096),
            )
        end
        @test asset_io_failure isa ArgumentError
        @test occursin(
            "required Qwen3-VL asset does not exist",
            sprint(showerror, asset_io_failure),
        )
    end
end

@testset "Qwen3-VL strict nested config" begin
    spec = qwen3_vl_checkpoint_spec()
    config = load_hf_qwen3_vl_config(_CH43_VL_CONFIG)
    @test config.variant == spec.variant
    @test config.model_type == :qwen3_vl
    @test config.text == spec.text
    @test config.vision == spec.vision
    @test config.max_seq_len == 262_144
    @test config.source_max_seq_len == 262_144
    @test load_hf_qwen3_vl_config(_CH43_VL_CONFIG; max_seq_len=4_096).max_seq_len ==
        4_096
    @test load_hf_qwen3_vl_config(
        _CH43_VL_CONFIG;
        max_seq_len=Int32(4_096),
    ).max_seq_len == 4_096
    @test_throws ArgumentError load_hf_qwen3_vl_config(
        _CH43_VL_CONFIG;
        max_seq_len=0,
    )
    @test_throws ArgumentError load_hf_qwen3_vl_config(
        _CH43_VL_CONFIG;
        max_seq_len=262_145,
    )
    @test_throws ArgumentError load_hf_qwen3_vl_config(
        _CH43_VL_CONFIG;
        max_seq_len=true,
    )
    @test_throws ArgumentError load_hf_qwen3_vl_config(
        _CH43_VL_CONFIG;
        max_seq_len=4_096.0,
    )
    @test_throws ArgumentError load_hf_qwen3_vl_config(
        _CH43_VL_CONFIG;
        max_seq_len=big(typemax(Int)) + 1,
    )

    @test_throws ArgumentError _ch43_load_mutated_config() do document
        document["model_type"] = "qwen3"
    end
    @test_throws ArgumentError _ch43_load_mutated_config() do document
        document["unexpected"] = 1
    end
    @test_throws ArgumentError _ch43_load_mutated_config() do document
        document["architectures"] = Any["Qwen3VLModel"]
    end
    @test_throws ArgumentError _ch43_load_mutated_config() do document
        document["image_token_id"] = true
    end
    @test_throws ArgumentError _ch43_load_mutated_config() do document
        document["image_token_id"] = big(typemax(Int)) + 1
    end
    for path_to_field in (
        ("tie_word_embeddings",),
        ("text_config", "attention_bias"),
        ("text_config", "use_cache"),
        ("text_config", "tie_word_embeddings"),
        ("text_config", "rope_scaling", "mrope_interleaved"),
    )
        @test_throws ArgumentError _ch43_load_mutated_config() do document
            target = document
            for name in path_to_field[1:(end - 1)]
                target = target[name]
            end
            target[path_to_field[end]] = path_to_field[end] == "attention_bias" ? 0 : 1
        end
    end
    @test_throws ArgumentError _ch43_load_mutated_config() do document
        document["transformers_version"] = "4.57.0"
    end
    @test_throws ArgumentError _ch43_load_mutated_config() do document
        document["text_config"]["hidden_size"] = 2_049
    end
    @test_throws ArgumentError _ch43_load_mutated_config() do document
        document["text_config"]["rope_scaling"]["mrope_interleaved"] = false
    end
    @test_throws ArgumentError _ch43_load_mutated_config() do document
        document["text_config"]["rope_scaling"]["mrope_section"] = Any[24, 20, 19]
    end
    @test_throws ArgumentError _ch43_load_mutated_config() do document
        document["vision_config"]["depth"] = 23
    end
    @test_throws ArgumentError _ch43_load_mutated_config() do document
        document["vision_config"]["deepstack_visual_indexes"] = Any[5, 11, 18]
    end
    @test_throws ArgumentError _ch43_load_mutated_config() do document
        delete!(document["vision_config"], "patch_size")
    end
end

@testset "Qwen3-VL strict processor config and checksum" begin
    frozen = qwen3_vl_processor_spec()
    @test load_hf_qwen3_vl_processor_config(_CH43_VL_PREPROCESSOR) == frozen
    @test frozen.preprocessor_config_sha256 == _ch43_sha256(_CH43_VL_PREPROCESSOR)
    @test frozen.processor_class == "Qwen3VLProcessor"
    @test frozen.image_processor_type == "Qwen2VLImageProcessorFast"
    @test frozen.min_pixels == 65_536
    @test frozen.max_pixels == 16_777_216
    @test (frozen.patch_size, frozen.temporal_patch_size, frozen.merge_size) ==
        (16, 2, 2)
    @test frozen.image_mean == frozen.image_std == (0.5f0, 0.5f0, 0.5f0)

    @test_throws ArgumentError _ch43_load_mutated_processor() do document
        document["patch_size"] = 14
    end
    @test_throws ArgumentError _ch43_load_mutated_processor() do document
        document["size"]["longest_edge"] = 16_777_215
    end

    processor_json = read(_CH43_VL_PREPROCESSOR, String)
    unknown_document = JSON3.read(processor_json, Dict{String,Any})
    unknown_document["z_unexpected"] = false
    unknown_document["a_unexpected"] = true
    unknown = _ch43_raw_processor_error(JSON3.write(unknown_document))
    @test unknown.failure isa ArgumentError
    @test sprint(showerror, unknown.failure) ==
        "ArgumentError: invalid fields in `preprocessor config` in " *
        "$(unknown.path) (unexpected: a_unexpected, z_unexpected)"

    delete!(unknown_document, "processor_class")
    delete!(unknown_document, "image_processor_type")
    missing_unknown = _ch43_raw_processor_error(JSON3.write(unknown_document))
    @test missing_unknown.failure isa ArgumentError
    @test sprint(showerror, missing_unknown.failure) ==
        "ArgumentError: invalid fields in `preprocessor config` in " *
        "$(missing_unknown.path) (missing: image_processor_type, processor_class; " *
        "unexpected: a_unexpected, z_unexpected)"

    duplicate_patch_size = replace(
        processor_json,
        "\"patch_size\": 16," =>
            "\"patch_size\": 8,\n    \"patch_size\": 16,";
        count=1,
    )
    duplicate_root = _ch43_raw_processor_error(duplicate_patch_size)
    @test duplicate_root.failure isa ArgumentError
    @test sprint(showerror, duplicate_root.failure) ==
        "ArgumentError: invalid fields in `preprocessor config` in " *
        "$(duplicate_root.path) (duplicate: patch_size)"

    duplicate_shortest_edge = replace(
        processor_json,
        "\"shortest_edge\": 65536" =>
            "\"shortest_edge\": 1,\n        \"shortest_edge\": 65536";
        count=1,
    )
    duplicate_size = _ch43_raw_processor_error(duplicate_shortest_edge)
    @test duplicate_size.failure isa ArgumentError
    @test sprint(showerror, duplicate_size.failure) ==
        "ArgumentError: invalid fields in `size` in $(duplicate_size.path) " *
        "(duplicate: shortest_edge)"

    # Duplicate bytes are rejected structurally before the frozen SHA is checked.
    @test !occursin(
        "checksum mismatch",
        sprint(showerror, duplicate_root.failure),
    )
    @test !occursin(
        "checksum mismatch",
        sprint(showerror, duplicate_size.failure),
    )

    @test_throws ArgumentError _ch43_load_mutated_processor() do document
        document["unexpected"] = false
    end

    # Semantically identical JSON with different bytes must fail the frozen SHA.
    @test_throws ArgumentError mktemp() do path, io
        write(io, read(_CH43_VL_PREPROCESSOR, String), '\n')
        close(io)
        load_hf_qwen3_vl_processor_config(path)
    end
end
