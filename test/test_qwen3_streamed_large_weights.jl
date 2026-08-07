using Test
using JSON3
using Lux
using LifeAI:
    GPTModel,
    decode_step,
    hf_qwen3_forward_trace,
    hf_token_ids,
    init_kv_cache,
    load_hf_qwen3_model,
    load_safetensors,
    open_safetensors_reader,
    prefill,
    qwen3_dense_parameter_count,
    qwen3_dense_spec,
    read_safetensors_tensor,
    stream_hf_qwen3_forward

isdefined(@__MODULE__, :qwen3_reference_directory) ||
    include("repository_test_assets.jl")

const _QWEN3_STREAMING_ASSETS_PATH = joinpath(
    @__DIR__,
    "fixtures",
    "qwen3_streamed_large_weights",
    "assets.json",
)

const _QWEN3_FAMILY_SPECS_PATH_FOR_QWEN3_STREAMING = joinpath(
    @__DIR__,
    "fixtures",
    "qwen3_dense_family",
    "specs.json",
)

function _qwen3_streaming_row_major_values(array)
    values = Float32.(array)
    ndims(values) <= 1 && return vec(values)
    return vec(permutedims(values, Tuple(reverse(1:ndims(values)))))
end

function _qwen3_streaming_tensor_bytes(array, dtype::String)
    values = _qwen3_streaming_row_major_values(array)
    if dtype == "F32"
        return collect(reinterpret(UInt8, values))
    elseif dtype == "BF16"
        bits = UInt16[UInt16(reinterpret(UInt32, value) >> 16) for value in values]
        return collect(reinterpret(UInt8, bits))
    end
    error("unsupported test dtype $dtype")
end

function _qwen3_streaming_write_safetensors(path, specs)
    header = Dict{String,Any}()
    data = UInt8[]
    offset = 0
    for spec in specs
        bytes = _qwen3_streaming_tensor_bytes(spec.array, spec.dtype)
        next_offset = offset + length(bytes)
        header[spec.name] = Dict(
            "dtype" => spec.dtype,
            "shape" => collect(size(spec.array)),
            "data_offsets" => [offset, next_offset],
        )
        append!(data, bytes)
        offset = next_offset
    end
    header_text = JSON3.write(header)
    padding = mod(-ncodeunits(header_text), 8)
    padded_header = header_text * repeat(" ", padding)
    open(path, "w") do io
        write(io, UInt64(ncodeunits(padded_header)))
        write(io, codeunits(padded_header))
        write(io, data)
    end
    return path
end

function _qwen3_streaming_config(; kwargs...)
    return merge(
        Dict{String,Any}(
            "architectures" => ["Qwen3ForCausalLM"],
            "attention_bias" => false,
            "attention_dropout" => 0.0,
            "head_dim" => 4,
            "hidden_act" => "silu",
            "hidden_size" => 8,
            "intermediate_size" => 12,
            "max_position_embeddings" => 32,
            "model_type" => "qwen3",
            "num_attention_heads" => 4,
            "num_hidden_layers" => 3,
            "num_key_value_heads" => 2,
            "rms_norm_eps" => 1.0e-6,
            "rope_scaling" => nothing,
            "rope_theta" => 1_000_000,
            "sliding_window" => nothing,
            "tie_word_embeddings" => false,
            "torch_dtype" => "bfloat16",
            "use_sliding_window" => false,
            "vocab_size" => 19,
        ),
        Dict{String,Any}(String(key) => value for (key, value) in pairs(kwargs)),
    )
end

function _qwen3_streaming_values(shape; norm=false, seed=0)
    count = prod(shape)
    values = Float32[
        (norm ? 1.0f0 : 0.0f0) + Float32(mod(index + seed, 7) - 3) / 64.0f0
        for index in 1:count
    ]
    return reshape(values, shape)
end

function _qwen3_streaming_qwen_tensors(model::GPTModel)
    tensors = Dict{String,Any}()
    tensors["model.embed_tokens.weight"] = _qwen3_streaming_values(
        (model.vocab_size, model.d_model);
        seed=1,
    )
    q_dim = model.num_heads * model.head_dim
    kv_dim = model.num_kv_heads * model.head_dim
    for layer in 0:(model.num_layers - 1)
        prefix = "model.layers.$layer"
        tensors["$prefix.input_layernorm.weight"] =
            _qwen3_streaming_values((model.d_model,); norm=true, seed=10 + layer)
        tensors["$prefix.self_attn.q_proj.weight"] =
            _qwen3_streaming_values((q_dim, model.d_model); seed=20 + layer)
        tensors["$prefix.self_attn.k_proj.weight"] =
            _qwen3_streaming_values((kv_dim, model.d_model); seed=30 + layer)
        tensors["$prefix.self_attn.v_proj.weight"] =
            _qwen3_streaming_values((kv_dim, model.d_model); seed=40 + layer)
        tensors["$prefix.self_attn.o_proj.weight"] =
            _qwen3_streaming_values((model.d_model, q_dim); seed=50 + layer)
        tensors["$prefix.self_attn.q_norm.weight"] =
            _qwen3_streaming_values((model.head_dim,); norm=true, seed=60 + layer)
        tensors["$prefix.self_attn.k_norm.weight"] =
            _qwen3_streaming_values((model.head_dim,); norm=true, seed=70 + layer)
        tensors["$prefix.post_attention_layernorm.weight"] =
            _qwen3_streaming_values((model.d_model,); norm=true, seed=80 + layer)
        tensors["$prefix.mlp.gate_proj.weight"] =
            _qwen3_streaming_values((model.mlp_hidden_dim, model.d_model); seed=90 + layer)
        tensors["$prefix.mlp.up_proj.weight"] =
            _qwen3_streaming_values((model.mlp_hidden_dim, model.d_model); seed=100 + layer)
        tensors["$prefix.mlp.down_proj.weight"] =
            _qwen3_streaming_values((model.d_model, model.mlp_hidden_dim); seed=110 + layer)
    end
    tensors["model.norm.weight"] = _qwen3_streaming_values(
        (model.d_model,);
        norm=true,
        seed=120,
    )
    model.tie_embeddings || (tensors["lm_head.weight"] = _qwen3_streaming_values(
        (model.vocab_size, model.d_model);
        seed=130,
    ))
    return tensors
end

function _qwen3_streaming_specs(tensors, names; dtype="BF16")
    return [(; name, dtype, array=tensors[name]) for name in names]
end

function _qwen3_streaming_write_sharded(directory, tensors; dtype="BF16")
    names = sort!(collect(keys(tensors)))
    midpoint = cld(length(names), 2)
    shards = Dict(
        "model-00001-of-00002.safetensors" => names[1:midpoint],
        "model-00002-of-00002.safetensors" => names[(midpoint + 1):end],
    )
    weight_map = Dict{String,String}()
    for (shard, shard_names) in shards
        _qwen3_streaming_write_safetensors(
            joinpath(directory, shard),
            _qwen3_streaming_specs(tensors, shard_names; dtype),
        )
        for name in shard_names
            weight_map[name] = shard
        end
    end
    index_path = joinpath(directory, "model.safetensors.index.json")
    write(index_path, JSON3.write(Dict("weight_map" => weight_map)))
    return index_path, weight_map
end

function _qwen3_streaming_fixture_dir(directory; tie=false, dtype="BF16")
    write(
        joinpath(directory, "config.json"),
        JSON3.write(_qwen3_streaming_config(; tie_word_embeddings=tie)),
    )
    config = LifeAI.load_hf_qwen3_config(
        joinpath(directory, "config.json");
        max_seq_len=16,
    )
    model = GPTModel(config)
    tensors = _qwen3_streaming_qwen_tensors(model)
    _qwen3_streaming_write_sharded(directory, tensors; dtype)
    return model, tensors
end

@testset "streamed safetensors reader strictness" begin
    mktempdir() do directory
        model, tensors = _qwen3_streaming_fixture_dir(directory)
        reader = open_safetensors_reader(directory)
        @test Set(String.(collect(keys(reader)))) == Set(keys(tensors))
        for name in ("model.embed_tokens.weight", "model.layers.1.mlp.up_proj.weight")
            streamed_tensor = read_safetensors_tensor(reader, name)
            expected = Float32.(
                reinterpret.(
                    Float32,
                    UInt32.(UInt16.(reinterpret(UInt32, Float32.(tensors[name])) .>> 16)) .<< 16,
                )
            )
            @test streamed_tensor == expected
        end
        @test_throws ArgumentError read_safetensors_tensor(reader, "not.a.tensor")

        merged = load_safetensors(directory)
        for name in keys(tensors)
            @test read_safetensors_tensor(reader, name) == merged[name]
        end
    end

    mktempdir() do directory
        _qwen3_streaming_fixture_dir(directory)
        index_path = joinpath(directory, "model.safetensors.index.json")
        index = JSON3.read(read(index_path, String), Dict{String,Any})
        index["weight_map"]["model.norm.weight"] = "model-00003-of-00002.safetensors"
        write(index_path, JSON3.write(index))
        @test_throws ArgumentError open_safetensors_reader(directory)
    end

    mktempdir() do directory
        _qwen3_streaming_fixture_dir(directory)
        index_path = joinpath(directory, "model.safetensors.index.json")
        index = JSON3.read(read(index_path, String), Dict{String,Any})
        moved = "model.norm.weight"
        original = index["weight_map"][moved]
        other = original == "model-00001-of-00002.safetensors" ?
            "model-00002-of-00002.safetensors" : "model-00001-of-00002.safetensors"
        index["weight_map"][moved] = other
        write(index_path, JSON3.write(index))
        @test_throws ArgumentError open_safetensors_reader(directory)
    end

    mktempdir() do directory
        _qwen3_streaming_fixture_dir(directory)
        index_path = joinpath(directory, "model.safetensors.index.json")
        index = JSON3.read(read(index_path, String), Dict{String,Any})
        delete!(index["weight_map"], "model.norm.weight")
        write(index_path, JSON3.write(index))
        @test_throws ArgumentError open_safetensors_reader(directory)
    end

    mktempdir() do directory
        @test_throws ArgumentError open_safetensors_reader(directory)
    end
end

function _qwen3_streaming_streamed_equals_in_memory(directory, model)
    tokens = reshape(hf_token_ids([0, 4, 7, 2, 11]; vocab_size=model.vocab_size), :, 1)
    decode_token = hf_token_ids([3]; vocab_size=model.vocab_size)

    streamed = stream_hf_qwen3_forward(
        directory,
        tokens;
        decode_token,
        max_seq_len=16,
    )
    @test streamed.variant === nothing

    loaded = load_hf_qwen3_model(directory; max_seq_len=16)
    trace = hf_qwen3_forward_trace(
        loaded.model,
        tokens,
        loaded.parameters,
        loaded.states,
    )
    cache = init_kv_cache(loaded.model; batch_size=1)
    _, cache, state = prefill(
        loaded.model,
        loaded.parameters,
        loaded.states,
        tokens,
        cache,
    )
    decode_logits, _, _ = decode_step(
        loaded.model,
        loaded.parameters,
        state,
        decode_token,
        cache,
    )

    @test streamed.embedding == trace.embedding
    @test length(streamed.blocks) == length(trace.blocks)
    for index in 1:length(trace.blocks)
        @test streamed.blocks[index] == trace.blocks[index]
    end
    @test streamed.final_hidden == trace.final_hidden
    @test streamed.logits == trace.logits
    @test streamed.decode_logits == decode_logits
    return nothing
end

@testset "streamed forward is bitwise identical to in-memory" begin
    mktempdir() do directory
        model, _ = _qwen3_streaming_fixture_dir(directory; tie=false, dtype="BF16")
        @test !model.tie_embeddings
        @test model.num_heads * model.head_dim > model.d_model
        _qwen3_streaming_streamed_equals_in_memory(directory, model)
    end

    mktempdir() do directory
        model, _ = _qwen3_streaming_fixture_dir(directory; tie=true, dtype="F32")
        @test model.tie_embeddings
        _qwen3_streaming_streamed_equals_in_memory(directory, model)
    end

    mktempdir() do directory
        model, tensors = _qwen3_streaming_fixture_dir(directory; tie=true)
        tensors["lm_head.weight"] = _qwen3_streaming_values(
            (model.vocab_size, model.d_model);
            seed=131,
        )
        _qwen3_streaming_write_sharded(directory, tensors)
        tokens = reshape(hf_token_ids([0, 1]; vocab_size=model.vocab_size), :, 1)
        @test_throws ArgumentError stream_hf_qwen3_forward(
            directory,
            tokens;
            max_seq_len=16,
        )
    end

    mktempdir() do directory
        model, tensors = _qwen3_streaming_fixture_dir(directory)
        delete!(tensors, "model.layers.2.mlp.down_proj.weight")
        _qwen3_streaming_write_sharded(directory, tensors)
        tokens = reshape(hf_token_ids([0, 1]; vocab_size=model.vocab_size), :, 1)
        @test_throws ArgumentError stream_hf_qwen3_forward(
            directory,
            tokens;
            max_seq_len=16,
        )
    end
end

@testset "Qwen3 streamed large-weight asset contract" begin
    fixture = JSON3.read(read(_QWEN3_STREAMING_ASSETS_PATH, String))
    qwen3_family = JSON3.read(read(_QWEN3_FAMILY_SPECS_PATH_FOR_QWEN3_STREAMING, String))
    qwen3_family_by_variant = Dict(
        String(entry["variant"]) => entry for entry in qwen3_family["variants"]
    )

    @test Int.(collect(fixture["token_ids_0_based"])) ==
        [1, 9707, 13, 151643, 100, 42, 151645, 2]
    @test Int(fixture["decode_token_id_0_based"]) == 17

    tolerances = fixture["tolerances"]
    for key in (
        "embedding_max_abs",
        "blocks_scaled_max_abs",
        "blocks_max_abs_ceiling",
        "final_hidden_max_abs",
        "logits_max_abs",
        "decode_max_abs",
    )
        @test Float64(tolerances[key]) > 0.0
    end

    models = fixture["models"]
    @test [String(entry["variant"]) for entry in models] ==
        ["qwen3_8b", "qwen3_14b", "qwen3_32b"]

    for entry in models
        variant = String(entry["variant"])
        reference = qwen3_family_by_variant[variant]
        spec = qwen3_dense_spec(Symbol(variant))
        @test !spec.tie_embeddings

        @test String(entry["model_id"]) == String(reference["model_id"])
        @test String(entry["revision"]) == String(reference["revision"])
        @test String(entry["config_sha256"]) ==
            String(reference["config_sha256"])
        @test Int(entry["parameter_count"]) ==
            Int(reference["parameter_count"])
        @test Int(entry["parameter_count"]) ==
            qwen3_dense_parameter_count(spec)

        files = Dict(String(file["name"]) => file for file in entry["files"])
        @test haskey(files, "config.json")
        @test haskey(files, "model.safetensors.index.json")
        shards = String.(collect(entry["weight_shards"]))
        @test length(shards) >= 5
        for shard in shards
            @test occursin(r"^model-\d{5}-of-\d{5}\.safetensors$", shard)
            @test haskey(files, shard)
        end
        for (name, file) in files
            @test occursin(r"^[0-9a-f]{64}$", String(file["sha256"]))
            @test Int(file["size"]) > 0
        end

        parity = entry["parity"]
        @test Float64(parity["embedding_max_abs"]) <=
            Float64(tolerances["embedding_max_abs"])
        @test Float64(parity["blocks_scaled_max_abs"]) <=
            Float64(tolerances["blocks_scaled_max_abs"])
        @test Float64(parity["blocks_max_abs"]) <=
            Float64(tolerances["blocks_max_abs_ceiling"])
        @test Float64(parity["final_hidden_max_abs"]) <=
            Float64(tolerances["final_hidden_max_abs"])
        @test Float64(parity["logits_max_abs"]) <=
            Float64(tolerances["logits_max_abs"])
        @test Float64(parity["dynamic_decode_max_abs"]) <=
            Float64(tolerances["decode_max_abs"])
        @test Bool(parity["argmax_equal"])
        @test Float64(parity["streamed_maxrss_gib"]) <
            Float64(entry["float32_parameter_gib"])
        @test !isempty(String(parity["transformers_version"]))
        @test !isempty(String(parity["torch_version"]))
    end
end

function _qwen3_streaming_integration(variant::Symbol, model_dir::AbstractString)
    fixture = JSON3.read(read(_QWEN3_STREAMING_ASSETS_PATH, String))
    entry = only(
        e for e in fixture["models"] if Symbol(String(e["variant"])) == variant
    )
    reference_dir = qwen3_reference_directory(
        model_dir;
        revision=String(entry["revision"]),
    )
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

    tokens = reshape(
        hf_token_ids(Int.(collect(metadata["token_ids_0_based"])); vocab_size=151936),
        :,
        1,
    )
    decode_token = hf_token_ids(
        [Int(metadata["decode_token_id_0_based"])];
        vocab_size=151936,
    )
    streamed = stream_hf_qwen3_forward(
        model_dir,
        tokens;
        decode_token,
        max_seq_len=64,
    )
    @test streamed.variant !== nothing
    @test streamed.variant.variant == variant
    @test !streamed.model.tie_embeddings
    @test Lux.parameterlength(streamed.model) == Int(entry["parameter_count"])

    hf_layout(array) = permutedims(array, (3, 2, 1))
    tolerances = fixture["tolerances"]
    embedding_tol = Float32(Float64(tolerances["embedding_max_abs"]))
    blocks_scaled_tol = Float32(Float64(tolerances["blocks_scaled_max_abs"]))
    blocks_ceiling = Float32(Float64(tolerances["blocks_max_abs_ceiling"]))
    final_tol = Float32(Float64(tolerances["final_hidden_max_abs"]))
    logits_tol = Float32(Float64(tolerances["logits_max_abs"]))
    decode_tol = Float32(Float64(tolerances["decode_max_abs"]))

    @test maximum(abs.(streamed.embedding .- hf_layout(reference["embedding"]))) <=
        embedding_tol
    # 中层 hidden state 含万级激活 outlier；绝对差随 reduction order 漂移
    # 数个 ulp（Pkg.test 的 --check-bounds 即可改变它），因此 block 输出
    # 用尺度感知容差：max-abs 差相对该层 reference 最大激活幅值衡量。
    for layer in 0:(streamed.model.num_layers - 1)
        reference_block = hf_layout(reference["block.$layer"])
        block_difference = maximum(abs.(streamed.blocks[layer + 1] .- reference_block))
        block_scale = max(1.0f0, maximum(abs.(reference_block)))
        @test block_difference <= blocks_scaled_tol * block_scale
        @test block_difference <= blocks_ceiling
    end
    @test maximum(abs.(
        streamed.final_hidden .- hf_layout(reference["final_hidden"]),
    )) <= final_tol
    @test maximum(abs.(streamed.logits .- hf_layout(reference["logits"]))) <=
        logits_tol
    @test argmax(vec(streamed.logits)) ==
        argmax(vec(hf_layout(reference["logits"])))
    expected_decode = hf_layout(reference["decode_logits"])
    @test maximum(abs.(streamed.decode_logits .- expected_decode)) <= decode_tol
    @test argmax(vec(streamed.decode_logits)) == argmax(vec(expected_decode))
    GC.gc()
    return nothing
end

for (variant, env_name) in (
    (:qwen3_8b, "LIFEAI_QWEN3_8B_MODEL_DIR"),
    (:qwen3_14b, "LIFEAI_QWEN3_14B_MODEL_DIR"),
    (:qwen3_32b, "LIFEAI_QWEN3_32B_MODEL_DIR"),
)
    if haskey(ENV, env_name)
        @testset "Qwen3 $(variant) streamed HuggingFace integration" begin
            _qwen3_streaming_integration(variant, ENV[env_name])
        end
    end
end
