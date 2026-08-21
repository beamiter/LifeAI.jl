using JSON3
using SHA: sha256
using Test
using LifeAI: load_hf_qwen3_vl_config,
    _qwen3_vl_vision_reference_sha256,
    load_hf_qwen3_vl_processor_config,
    qwen3_vl_checkpoint_spec,
    qwen3_vl_expected_tensor_shapes,
    qwen3_vl_parameter_count,
    qwen3_vl_processor_spec

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
