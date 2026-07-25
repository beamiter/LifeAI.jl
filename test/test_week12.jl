using Test
using JSON3
using Lux
using LifeAI:
    decode_step,
    hf_qwen3_forward_trace,
    hf_token_ids,
    init_kv_cache,
    init_static_kv_cache,
    load_hf_qwen3_model,
    load_safetensors,
    prefill,
    qwen3_dense_parameter_count,
    qwen3_dense_spec

const _WEEK12_ASSETS_PATH = joinpath(
    @__DIR__,
    "fixtures",
    "week12_qwen3_dense_real_weights",
    "assets.json",
)

const _WEEK11_SPECS_PATH_FOR_WEEK12 = joinpath(
    @__DIR__,
    "fixtures",
    "week11_qwen3_dense_family",
    "specs.json",
)

_week12_is_sha256(value) = occursin(r"^[0-9a-f]{64}$", String(value))

@testset "Qwen3 real-weight asset contract" begin
    fixture = JSON3.read(read(_WEEK12_ASSETS_PATH, String))
    week11 = JSON3.read(read(_WEEK11_SPECS_PATH_FOR_WEEK12, String))
    week11_by_variant = Dict(
        String(entry["variant"]) => entry for entry in week11["variants"]
    )

    @test Int.(collect(fixture["token_ids_0_based"])) ==
        [1, 9707, 13, 151643, 100, 42, 151645, 2]
    @test Int(fixture["decode_token_id_0_based"]) == 17

    tolerances = fixture["tolerances"]
    for key in (
        "embedding_max_abs",
        "blocks_max_abs",
        "final_hidden_max_abs",
        "logits_max_abs",
        "decode_max_abs",
    )
        @test Float64(tolerances[key]) > 0.0
    end

    models = fixture["models"]
    @test length(models) == 2
    @test [String(entry["variant"]) for entry in models] ==
        ["qwen3_1_7b", "qwen3_4b"]

    for entry in models
        variant = String(entry["variant"])
        reference = week11_by_variant[variant]

        @test String(entry["model_id"]) == String(reference["model_id"])
        @test String(entry["revision"]) == String(reference["revision"])
        @test String(entry["config_sha256"]) ==
            String(reference["config_sha256"])
        @test Int(entry["parameter_count"]) ==
            Int(reference["parameter_count"])
        @test Int(entry["parameter_count"]) ==
            qwen3_dense_parameter_count(qwen3_dense_spec(Symbol(variant)))

        files = Dict(
            String(file["name"]) => file for file in entry["files"]
        )
        @test haskey(files, "config.json")
        @test haskey(files, "model.safetensors.index.json")
        shards = String.(collect(entry["weight_shards"]))
        @test length(shards) >= 2
        for shard in shards
            @test occursin(r"^model-\d{5}-of-\d{5}\.safetensors$", shard)
            @test haskey(files, shard)
        end
        for (name, file) in files
            @test _week12_is_sha256(file["sha256"])
            @test Int(file["size"]) > 0
        end
        @test Int(files["config.json"]["size"]) == 726

        parity = entry["parity"]
        @test Float64(parity["embedding_max_abs"]) <=
            Float64(tolerances["embedding_max_abs"])
        @test Float64(parity["blocks_max_abs"]) <=
            Float64(tolerances["blocks_max_abs"])
        @test Float64(parity["final_hidden_max_abs"]) <=
            Float64(tolerances["final_hidden_max_abs"])
        @test Float64(parity["logits_max_abs"]) <=
            Float64(tolerances["logits_max_abs"])
        @test Float64(parity["dynamic_decode_max_abs"]) <=
            Float64(tolerances["decode_max_abs"])
        @test Float64(parity["static_decode_max_abs"]) <=
            Float64(tolerances["decode_max_abs"])
        @test Bool(parity["argmax_equal"])
        @test !isempty(String(parity["transformers_version"]))
        @test !isempty(String(parity["torch_version"]))
    end
end

function _week12_integration(variant::Symbol, model_dir::AbstractString)
    fixture = JSON3.read(read(_WEEK12_ASSETS_PATH, String))
    entry = only(
        e for e in fixture["models"] if Symbol(String(e["variant"])) == variant
    )
    reference_dir = joinpath(model_dir, "lifeai-references", "week12-parity")
    isfile(joinpath(reference_dir, "reference.json")) || error(
        "missing $reference_dir/reference.json; " *
        "run scripts/export_qwen3_reference.py first",
    )
    metadata = JSON3.read(read(joinpath(reference_dir, "reference.json"), String))
    reference = load_safetensors(joinpath(reference_dir, "reference.safetensors"))
    @test String(metadata["revision"]) == String(entry["revision"])
    @test Int.(collect(metadata["token_ids_0_based"])) ==
        Int.(collect(fixture["token_ids_0_based"]))

    for file in entry["files"]
        path = joinpath(model_dir, String(file["name"]))
        @test isfile(path)
        @test filesize(path) == Int(file["size"])
    end
    @test isfile(joinpath(model_dir, "model.safetensors.index.json"))
    @test !isfile(joinpath(model_dir, "model.safetensors"))

    loaded = load_hf_qwen3_model(model_dir; max_seq_len=64)
    @test loaded.variant !== nothing
    @test loaded.variant.variant == variant
    @test Lux.parameterlength(loaded.model) == Int(entry["parameter_count"])

    tokens = reshape(hf_token_ids(
        Int.(collect(metadata["token_ids_0_based"]));
        vocab_size=loaded.model.vocab_size,
    ), :, 1)
    trace = hf_qwen3_forward_trace(
        loaded.model,
        tokens,
        loaded.parameters,
        loaded.states,
    )
    hf_layout(array) = permutedims(array, (3, 2, 1))
    tolerances = fixture["tolerances"]
    embedding_tol = Float32(Float64(tolerances["embedding_max_abs"]))
    blocks_tol = Float32(Float64(tolerances["blocks_max_abs"]))
    final_tol = Float32(Float64(tolerances["final_hidden_max_abs"]))
    logits_tol = Float32(Float64(tolerances["logits_max_abs"]))
    decode_tol = Float32(Float64(tolerances["decode_max_abs"]))

    @test maximum(abs.(trace.embedding .- hf_layout(reference["embedding"]))) <=
        embedding_tol
    for layer in 0:(loaded.model.num_layers - 1)
        @test maximum(abs.(
            trace.blocks[layer + 1] .- hf_layout(reference["block.$layer"]),
        )) <= blocks_tol
    end
    @test maximum(abs.(
        trace.final_hidden .- hf_layout(reference["final_hidden"]),
    )) <= final_tol
    @test maximum(abs.(trace.logits .- hf_layout(reference["logits"]))) <=
        logits_tol
    @test argmax(vec(trace.logits)) ==
        argmax(vec(hf_layout(reference["logits"])))

    decode_token = hf_token_ids(
        [Int(metadata["decode_token_id_0_based"])];
        vocab_size=loaded.model.vocab_size,
    )
    expected_decode = hf_layout(reference["decode_logits"])

    cache = init_kv_cache(loaded.model; batch_size=1)
    _, cache, state = prefill(
        loaded.model,
        loaded.parameters,
        loaded.states,
        tokens,
        cache,
    )
    dynamic_logits, _, _ = decode_step(
        loaded.model,
        loaded.parameters,
        state,
        decode_token,
        cache,
    )
    @test maximum(abs.(dynamic_logits .- expected_decode)) <= decode_tol
    @test argmax(vec(dynamic_logits)) == argmax(vec(expected_decode))

    static_cache = init_static_kv_cache(loaded.model; batch_size=1)
    _, static_cache, static_state = prefill(
        loaded.model,
        loaded.parameters,
        loaded.states,
        tokens,
        static_cache,
    )
    static_logits, _, _ = decode_step(
        loaded.model,
        loaded.parameters,
        static_state,
        decode_token,
        static_cache,
    )
    @test maximum(abs.(static_logits .- expected_decode)) <= decode_tol
    @test argmax(vec(static_logits)) == argmax(vec(expected_decode))
    # 释放当前大模型，避免两个 integration 同时驻留超出 30 GiB RAM。
    loaded = nothing
    trace = nothing
    GC.gc()
    return nothing
end

if haskey(ENV, "LIFEAI_QWEN3_1_7B_MODEL_DIR")
    @testset "Qwen3-1.7B HuggingFace integration" begin
        _week12_integration(:qwen3_1_7b, ENV["LIFEAI_QWEN3_1_7B_MODEL_DIR"])
    end
end

if haskey(ENV, "LIFEAI_QWEN3_4B_MODEL_DIR")
    @testset "Qwen3-4B HuggingFace integration" begin
        _week12_integration(:qwen3_4b, ENV["LIFEAI_QWEN3_4B_MODEL_DIR"])
    end
end
