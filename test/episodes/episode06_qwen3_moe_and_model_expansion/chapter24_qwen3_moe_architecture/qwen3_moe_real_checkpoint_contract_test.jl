using Test
using JSON3
using SHA: sha256
using LifeAI:
    Qwen3MoECheckpointSpec,
    Qwen3MoEShardSpec,
    load_hf_qwen3_moe_config,
    qwen3_moe_checkpoint_spec,
    qwen3_moe_parameter_count,
    verify_qwen3_moe_checkpoint

const _QWEN3_MOE_REAL_CONTRACT_DIR = joinpath(
    @__DIR__,
    "fixtures",
    "qwen3_moe_real_checkpoint",
)

function _qwen3_moe_contract_captured_error(thunk)
    try
        thunk()
    catch error
        return error
    end
    error("expected Qwen3 MoE checkpoint specification to fail")
end

function _qwen3_moe_contract_geometry_arguments(
    base;
    d_model=base[10],
    num_heads=base[14],
    num_kv_heads=base[15],
    head_dim=base[16],
    num_experts=base[17],
    experts_per_token=base[18],
)
    values = Base.setindex(base, d_model, 10)
    values = Base.setindex(values, num_heads, 14)
    values = Base.setindex(values, num_kv_heads, 15)
    values = Base.setindex(values, head_dim, 16)
    values = Base.setindex(values, num_experts, 17)
    return Base.setindex(values, experts_per_token, 18)
end

function _qwen3_moe_contract_copy(
    spec;
    index_sha256=spec.index_sha256,
    index_tensor_count=spec.index_tensor_count,
    shard_payload_bytes=spec.shard_payload_bytes,
    num_layers=spec.num_layers,
    num_experts=spec.num_experts,
    shards=spec.shards,
)
    return Qwen3MoECheckpointSpec(
        spec.variant,
        spec.model_id,
        spec.revision,
        spec.config_sha256,
        index_sha256,
        index_tensor_count,
        spec.tensor_bytes,
        shard_payload_bytes,
        spec.vocab_size,
        spec.d_model,
        spec.dense_mlp_hidden_dim,
        spec.moe_hidden_dim,
        num_layers,
        spec.num_heads,
        spec.num_kv_heads,
        spec.head_dim,
        num_experts,
        spec.experts_per_token,
        spec.max_position_embeddings,
        shards,
    )
end

@testset "Qwen3 MoE checkpoint specifications are strict" begin
    strings = ntuple(_ -> SubString("xvalue", 2), 4)
    valid = (
        :fixture,
        strings...,
        big(0),
        Int32(0),
        Int16(0),
        ntuple(_ -> big(1), 11)...,
        (),
    )
    spec = Qwen3MoECheckpointSpec(valid...)
    @test spec.variant === :fixture
    @test spec.model_id === "value"
    @test spec.index_tensor_count === 0
    @test spec.max_position_embeddings === 1
    @test isempty(spec.shards)

    for (arguments, label) in (
        (("", 1, "hash"), "filename"),
        (("shard", 1, ""), "sha256"),
    )
        failure = _qwen3_moe_contract_captured_error() do
            Qwen3MoEShardSpec(arguments...)
        end
        @test failure isa ArgumentError
        @test sprint(showerror, failure) ==
            "ArgumentError: Qwen3 MoE shard $label must not be empty"
    end

    for (index, label) in (
        (2, "model_id"),
        (3, "revision"),
        (4, "config_sha256"),
        (5, "index_sha256"),
    )
        failure = _qwen3_moe_contract_captured_error() do
            Qwen3MoECheckpointSpec(Base.setindex(valid, "", index)...)
        end
        @test failure isa ArgumentError
        @test sprint(showerror, failure) ==
            "ArgumentError: Qwen3 MoE checkpoint $label must not be empty"
    end

    integer_fields = (
        6 => "index_tensor_count",
        7 => "tensor_bytes",
        8 => "shard_payload_bytes",
        9 => "vocab_size",
        10 => "d_model",
        11 => "dense_mlp_hidden_dim",
        12 => "moe_hidden_dim",
        13 => "num_layers",
        14 => "num_heads",
        15 => "num_kv_heads",
        16 => "head_dim",
        17 => "num_experts",
        18 => "experts_per_token",
        19 => "max_position_embeddings",
    )
    for (index, label) in integer_fields
        failure = _qwen3_moe_contract_captured_error() do
            Qwen3MoECheckpointSpec(Base.setindex(valid, true, index)...)
        end
        @test failure isa ArgumentError
        @test sprint(showerror, failure) ==
            "ArgumentError: Qwen3 MoE checkpoint $label must be an integer"
    end

    for (index, label) in integer_fields
        failure = _qwen3_moe_contract_captured_error() do
            Qwen3MoECheckpointSpec(Base.setindex(valid, -1, index)...)
        end
        qualifier = index <= 8 ? "non-negative" : "positive"
        @test failure isa ArgumentError
        @test sprint(showerror, failure) ==
            "ArgumentError: Qwen3 MoE checkpoint $label must be $qualifier"
    end

    too_large = big(typemax(Int)) + 1
    for (index, value, message) in (
        (7, 1.0, "tensor_bytes must be an integer"),
        (9, 1.0, "vocab_size must be an integer"),
        (7, too_large, "tensor_bytes is outside the host integer range"),
        (9, too_large, "vocab_size is outside the host integer range"),
    )
        failure = _qwen3_moe_contract_captured_error() do
            Qwen3MoECheckpointSpec(Base.setindex(valid, value, index)...)
        end
        @test failure isa ArgumentError
        @test sprint(showerror, failure) ==
            "ArgumentError: Qwen3 MoE checkpoint $message"
    end

    for (index, value, message) in (
        (1, "fixture", "variant must be a Symbol"),
        (2, :model, "model_id must be a string"),
        (20, [], "shards must be a tuple"),
        (
            20,
            ((; filename="model", bytes=0, sha256="hash"),),
            "shards must contain Qwen3MoEShardSpec values",
        ),
        (
            20,
            (
                Qwen3MoEShardSpec("duplicate", 1, "first"),
                Qwen3MoEShardSpec("duplicate", 2, "second"),
            ),
            "shard filenames must be unique",
        ),
    )
        failure = _qwen3_moe_contract_captured_error() do
            Qwen3MoECheckpointSpec(Base.setindex(valid, value, index)...)
        end
        @test failure isa ArgumentError
        @test sprint(showerror, failure) ==
            "ArgumentError: Qwen3 MoE checkpoint $message"
    end

    wide = Qwen3MoECheckpointSpec(_qwen3_moe_contract_geometry_arguments(
        valid;
        d_model=3,
        num_heads=2,
        num_kv_heads=1,
        head_dim=2,
    )...)
    @test wide.num_heads * wide.head_dim == 4
    @test wide.d_model == 3

    geometry_failures = (
        (
            _qwen3_moe_contract_geometry_arguments(
                valid;
                num_heads=typemax(Int),
                num_kv_heads=1,
                head_dim=2,
            ),
            "query projection width exceeds the host integer range",
        ),
        (
            _qwen3_moe_contract_geometry_arguments(
                valid;
                num_heads=1,
                num_kv_heads=typemax(Int),
                head_dim=2,
            ),
            "key/value projection width exceeds the host integer range",
        ),
        (
            _qwen3_moe_contract_geometry_arguments(
                valid;
                num_heads=3,
                num_kv_heads=2,
                head_dim=2,
            ),
            "num_heads must be divisible by num_kv_heads",
        ),
        (
            _qwen3_moe_contract_geometry_arguments(
                valid;
                num_experts=1,
                experts_per_token=2,
            ),
            "experts_per_token must not exceed num_experts",
        ),
    )
    for (arguments, message) in geometry_failures
        failure = _qwen3_moe_contract_captured_error() do
            Qwen3MoECheckpointSpec(arguments...)
        end
        @test failure isa ArgumentError
        @test sprint(showerror, failure) ==
            "ArgumentError: Qwen3 MoE checkpoint $message"
    end
end

@testset "Qwen3 MoE index tensor counts are exact and preflighted" begin
    base = qwen3_moe_checkpoint_spec()
    cases = (
        (
            _qwen3_moe_contract_copy(base; index_tensor_count=0),
            "frozen Qwen3 MoE architecture does not match index tensor count",
        ),
        (
            _qwen3_moe_contract_copy(
                base;
                index_tensor_count=0,
                num_layers=typemax(Int),
                num_experts=typemax(Int),
            ),
            "Qwen3 MoE index tensor count exceeds the host integer range",
        ),
    )
    mktempdir() do directory
        for (spec, message) in cases
            failure = _qwen3_moe_contract_captured_error() do
                verify_qwen3_moe_checkpoint(
                    directory;
                    spec,
                    verify_shard_checksums=false,
                )
            end
            @test failure isa ArgumentError
            @test sprint(showerror, failure) == "ArgumentError: $message"
            @test !occursin("config.json", sprint(showerror, failure))
        end
    end
end

@testset "Qwen3 MoE shard byte totals are bound at construction" begin
    base = qwen3_moe_checkpoint_spec()
    cases = (
        (
            () -> _qwen3_moe_contract_copy(
                base;
                shard_payload_bytes=0,
                shards=(
                    Qwen3MoEShardSpec("first", typemax(Int), "hash"),
                    Qwen3MoEShardSpec("second", 1, "hash"),
                ),
            ),
            "Qwen3 MoE checkpoint shard payload byte count exceeds " *
            "the host integer range",
        ),
        (
            () -> _qwen3_moe_contract_copy(
                base;
                shard_payload_bytes=2,
                shards=(Qwen3MoEShardSpec("only", 1, "hash"),),
            ),
            "Qwen3 MoE checkpoint shard_payload_bytes must equal " *
            "sum(shard.bytes)",
        ),
    )
    for (build, message) in cases
        failure = _qwen3_moe_contract_captured_error(build)
        @test failure isa ArgumentError
        @test sprint(showerror, failure) == "ArgumentError: $message"
    end
end

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
    overflow_spec = Qwen3MoECheckpointSpec(
        :overflow,
        "overflow",
        "overflow",
        "overflow",
        "overflow",
        0,
        0,
        0,
        typemax(Int),
        2,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        (),
    )
    @test_throws ArgumentError qwen3_moe_parameter_count(overflow_spec)
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
