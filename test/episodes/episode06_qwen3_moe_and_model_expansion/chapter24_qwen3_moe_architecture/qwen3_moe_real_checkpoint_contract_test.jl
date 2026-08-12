using Test
using JSON3
using SHA: sha256
using LifeAI:
    load_hf_qwen3_moe_config,
    qwen3_moe_checkpoint_spec,
    qwen3_moe_parameter_count,
    verify_qwen3_moe_checkpoint

const _QWEN3_MOE_REAL_CONTRACT_DIR = joinpath(
    @__DIR__,
    "fixtures",
    "qwen3_moe_real_checkpoint",
)

@testset "Qwen3-30B-A3B immutable checkpoint contract" begin
    spec = qwen3_moe_checkpoint_spec()
    manifest = JSON3.read(read(joinpath(
        _QWEN3_MOE_REAL_CONTRACT_DIR,
        "assets.json",
    ), String))
    config_path = joinpath(_QWEN3_MOE_REAL_CONTRACT_DIR, "config.json")
    config = load_hf_qwen3_moe_config(config_path)

    @test Int(manifest.schema_version) == 1
    @test String(manifest.model_id) == spec.model_id
    @test String(manifest.revision) == spec.revision
    @test spec.variant === :qwen3_30b_a3b
    @test spec.revision == "ad44e777bcd18fa416d9da3bd8f70d33ebb85d39"
    @test bytes2hex(sha256(read(config_path))) ==
        String(manifest.config.fixture_sha256)
    @test String(manifest.config.sha256) == spec.config_sha256
    @test String(manifest.index.sha256) == spec.index_sha256

    @test config.qwen3_model_type === :moe
    @test config.vocab_size == spec.vocab_size == 151_936
    @test config.d_model == spec.d_model == 2_048
    @test config.dense_mlp_hidden_dim == spec.dense_mlp_hidden_dim == 6_144
    @test config.mlp_hidden_dim == spec.moe_hidden_dim == 768
    @test config.num_layers == spec.num_layers == 48
    @test config.num_heads == spec.num_heads == 32
    @test config.num_kv_heads == spec.num_kv_heads == 4
    @test config.head_dim == spec.head_dim == 128
    @test config.num_experts == spec.num_experts == 128
    @test config.experts_per_token == spec.experts_per_token == 8
    @test config.normalize_routing
    @test !config.tie_embeddings
    @test config.source_max_seq_len == spec.max_position_embeddings == 40_960

    @test qwen3_moe_parameter_count() == 30_532_122_624
    @test qwen3_moe_parameter_count() * 2 == spec.tensor_bytes
    @test Int(manifest.index.tensor_count) == spec.index_tensor_count == 18_867
    @test Int(manifest.index.tensor_bytes) == spec.tensor_bytes
    @test Int(manifest.shard_payload_bytes) == spec.shard_payload_bytes
    @test length(spec.shards) == length(manifest.shards) == 16
    @test sum(shard.bytes for shard in spec.shards) == spec.shard_payload_bytes

    for (shard, asset) in zip(spec.shards, manifest.shards)
        @test shard.filename == String(asset.path)
        @test occursin(r"^model-\d{5}-of-00016\.safetensors$", shard.filename)
        @test shard.bytes == Int(asset.bytes)
        @test shard.sha256 == String(asset.sha256)
        @test length(shard.sha256) == 64
    end

    @test_throws ArgumentError verify_qwen3_moe_checkpoint(
        _QWEN3_MOE_REAL_CONTRACT_DIR,
    )
end

const _QWEN3_MOE_REAL_MODEL_ENV = "LIFEAI_QWEN3_30B_A3B_MODEL_DIR"
if haskey(ENV, _QWEN3_MOE_REAL_MODEL_ENV)
    @testset "Qwen3-30B-A3B local asset integrity" begin
        report = verify_qwen3_moe_checkpoint(
            ENV[_QWEN3_MOE_REAL_MODEL_ENV];
            verify_shard_checksums=true,
        )
        @test report.spec.revision == qwen3_moe_checkpoint_spec().revision
        @test report.tensor_count == 18_867
        @test report.shard_checksums_verified
        @test length(report.shards) == 16
    end
else
    @info "Skipping Qwen3-30B-A3B local asset integrity; set $_QWEN3_MOE_REAL_MODEL_ENV"
end
