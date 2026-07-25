using Test
using BFloat16s: BFloat16
using JSON3
using Lux
using LifeAI:
    GPTModel,
    hf_qwen3_bf16_forward,
    hf_qwen3_forward_trace,
    hf_token_ids,
    load_hf_qwen3_model,
    load_safetensors,
    qwen3_dense_parameter_count,
    qwen3_dense_spec

const _WEEK14_ASSETS_PATH = joinpath(
    @__DIR__,
    "fixtures",
    "week14_qwen3_bf16_compute",
    "assets.json",
)

const _WEEK11_SPECS_PATH_FOR_WEEK14 = joinpath(
    @__DIR__,
    "fixtures",
    "week11_qwen3_dense_family",
    "specs.json",
)

function _week14_row_major_values(array)
    values = Float32.(array)
    ndims(values) <= 1 && return vec(values)
    return vec(permutedims(values, Tuple(reverse(1:ndims(values)))))
end

function _week14_tensor_bytes(array, dtype::String)
    values = _week14_row_major_values(array)
    if dtype == "F32"
        return collect(reinterpret(UInt8, values))
    elseif dtype == "BF16"
        bits = UInt16[UInt16(reinterpret(UInt32, value) >> 16) for value in values]
        return collect(reinterpret(UInt8, bits))
    end
    error("unsupported test dtype $dtype")
end

function _week14_write_safetensors(path, specs)
    header = Dict{String,Any}()
    data = UInt8[]
    offset = 0
    for spec in specs
        bytes = _week14_tensor_bytes(spec.array, spec.dtype)
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

function _week14_config(; kwargs...)
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

function _week14_values(shape; norm=false, seed=0)
    count = prod(shape)
    values = Float32[
        (norm ? 1.0f0 : 0.0f0) + Float32(mod(index + seed, 7) - 3) / 64.0f0
        for index in 1:count
    ]
    return reshape(values, shape)
end

function _week14_qwen_tensors(model::GPTModel)
    tensors = Dict{String,Any}()
    tensors["model.embed_tokens.weight"] = _week14_values(
        (model.vocab_size, model.d_model);
        seed=1,
    )
    q_dim = model.num_heads * model.head_dim
    kv_dim = model.num_kv_heads * model.head_dim
    for layer in 0:(model.num_layers - 1)
        prefix = "model.layers.$layer"
        tensors["$prefix.input_layernorm.weight"] =
            _week14_values((model.d_model,); norm=true, seed=10 + layer)
        tensors["$prefix.self_attn.q_proj.weight"] =
            _week14_values((q_dim, model.d_model); seed=20 + layer)
        tensors["$prefix.self_attn.k_proj.weight"] =
            _week14_values((kv_dim, model.d_model); seed=30 + layer)
        tensors["$prefix.self_attn.v_proj.weight"] =
            _week14_values((kv_dim, model.d_model); seed=40 + layer)
        tensors["$prefix.self_attn.o_proj.weight"] =
            _week14_values((model.d_model, q_dim); seed=50 + layer)
        tensors["$prefix.self_attn.q_norm.weight"] =
            _week14_values((model.head_dim,); norm=true, seed=60 + layer)
        tensors["$prefix.self_attn.k_norm.weight"] =
            _week14_values((model.head_dim,); norm=true, seed=70 + layer)
        tensors["$prefix.post_attention_layernorm.weight"] =
            _week14_values((model.d_model,); norm=true, seed=80 + layer)
        tensors["$prefix.mlp.gate_proj.weight"] =
            _week14_values((model.mlp_hidden_dim, model.d_model); seed=90 + layer)
        tensors["$prefix.mlp.up_proj.weight"] =
            _week14_values((model.mlp_hidden_dim, model.d_model); seed=100 + layer)
        tensors["$prefix.mlp.down_proj.weight"] =
            _week14_values((model.d_model, model.mlp_hidden_dim); seed=110 + layer)
    end
    tensors["model.norm.weight"] = _week14_values(
        (model.d_model,);
        norm=true,
        seed=120,
    )
    model.tie_embeddings || (tensors["lm_head.weight"] = _week14_values(
        (model.vocab_size, model.d_model);
        seed=130,
    ))
    return tensors
end

function _week14_fixture_dir(directory; tie=false)
    write(
        joinpath(directory, "config.json"),
        JSON3.write(_week14_config(; tie_word_embeddings=tie)),
    )
    config = LifeAI.load_hf_qwen3_config(
        joinpath(directory, "config.json");
        max_seq_len=16,
    )
    model = GPTModel(config)
    tensors = _week14_qwen_tensors(model)
    specs = [
        (; name, dtype="BF16", array=tensors[name])
        for name in sort!(collect(keys(tensors)))
    ]
    _week14_write_safetensors(joinpath(directory, "model.safetensors"), specs)
    return model, tensors
end

@testset "BF16 weight loading is bit-preserving" begin
    mktempdir() do directory
        _, tensors = _week14_fixture_dir(directory)
        loaded_f32 = load_hf_qwen3_model(directory; max_seq_len=16)
        loaded_bf16 = load_hf_qwen3_model(
            directory;
            max_seq_len=16,
            weight_dtype=BFloat16,
        )
        @test eltype(loaded_f32.parameters.token_embedding.weight) === Float32
        @test eltype(loaded_bf16.parameters.token_embedding.weight) === BFloat16
        # BF16 storage adopted bit-exactly: upconverting the BF16 tree must
        # reproduce the Float32 tree (which decodes BF16 storage exactly).
        @test Float32.(loaded_bf16.parameters.token_embedding.weight) ==
            loaded_f32.parameters.token_embedding.weight
        first_block_bf16 = loaded_bf16.parameters.blocks.layer_1
        first_block_f32 = loaded_f32.parameters.blocks.layer_1
        @test Float32.(first_block_bf16.attn.q_proj.weight) ==
            first_block_f32.attn.q_proj.weight
        @test Float32.(first_block_bf16.mlp.down_proj.weight) ==
            first_block_f32.mlp.down_proj.weight
        @test Base.summarysize(loaded_bf16.parameters) <
            0.6 * Base.summarysize(loaded_f32.parameters)

        tensors_bf16 = load_safetensors(directory; target_dtype=BFloat16)
        @test eltype(tensors_bf16["model.norm.weight"]) === BFloat16
        @test_throws ArgumentError load_safetensors(
            directory;
            target_dtype=Float64,
        )
    end
end

@testset "BF16 mixed-precision operator semantics" begin
    weight = BFloat16.(_week14_values((3, 4); seed=7))
    x = BFloat16.(reshape(_week14_values((4, 2); seed=3), 4, 2, 1))
    y = LifeAI._bf16_linear(weight, x)
    @test eltype(y) === BFloat16
    @test size(y) == (3, 2, 1)
    expected = BFloat16.(
        Float32.(weight) * reshape(Float32.(x), 4, 2),
    )
    @test vec(y) == vec(expected)

    scale = BFloat16.(_week14_values((5,); norm=true, seed=2))
    h = BFloat16.(reshape(_week14_values((5, 2); seed=9), 5, 2, 1))
    normed = LifeAI._bf16_rmsnorm(h, scale, 1.0f-6)
    hf = Float32.(h)
    manual_norm = BFloat16.(
        hf ./ sqrt.(sum(abs2, hf; dims=1) ./ 5.0f0 .+ 1.0f-6),
    )
    manual = BFloat16.(
        reshape(Float32.(scale), 5, 1, 1) .* Float32.(manual_norm),
    )
    @test eltype(normed) === BFloat16
    @test normed == manual
end

@testset "BF16 forward path on synthetic model" begin
    mktempdir() do directory
        model, _ = _week14_fixture_dir(directory; tie=false)
        loaded = load_hf_qwen3_model(
            directory;
            max_seq_len=16,
            weight_dtype=BFloat16,
        )
        tokens = reshape(hf_token_ids([0, 4, 7, 2, 11]; vocab_size=model.vocab_size), :, 1)
        decode_token = hf_token_ids([3]; vocab_size=model.vocab_size)

        first = hf_qwen3_bf16_forward(
            loaded.model, loaded.parameters, tokens;
            decode_token, greedy_steps=4,
        )
        second = hf_qwen3_bf16_forward(
            loaded.model, loaded.parameters, tokens;
            decode_token, greedy_steps=4,
        )
        @test eltype(first.logits) === BFloat16
        @test first.logits == second.logits
        @test first.decode_logits == second.decode_logits
        @test first.greedy_tokens == second.greedy_tokens
        @test length(first.greedy_tokens) == 4

        # A cached decode step must reproduce the last position of the full
        # BF16 forward over prompt + token: identical BF16 operations run on
        # identical values in the same order.
        extended = vcat(vec(tokens), decode_token)
        full = hf_qwen3_bf16_forward(
            loaded.model, loaded.parameters, reshape(extended, :, 1),
        )
        @test first.decode_logits[:, 1, 1] == full.logits[:, end, 1]

        # BF16 deviates from Float32 but stays in BF16 magnitude.
        loaded_f32 = load_hf_qwen3_model(directory; max_seq_len=16)
        trace_f32 = hf_qwen3_forward_trace(
            loaded_f32.model,
            tokens,
            loaded_f32.parameters,
            loaded_f32.states,
        )
        difference = maximum(abs.(Float32.(first.logits) .- trace_f32.logits))
        @test difference > 0
        @test difference <= 0.1f0

        @test_throws ArgumentError hf_qwen3_bf16_forward(
            loaded_f32.model, loaded_f32.parameters, tokens,
        )
        @test_throws ArgumentError hf_qwen3_bf16_forward(
            loaded.model, loaded.parameters, hcat(vec(tokens), vec(tokens));
            greedy_steps=2,
        )
        @test_throws ArgumentError hf_qwen3_bf16_forward(
            loaded.model, loaded.parameters, tokens;
            greedy_steps=100,
        )
    end
end

@testset "Qwen3 BF16 compute asset contract" begin
    fixture = JSON3.read(read(_WEEK14_ASSETS_PATH, String))
    week11 = JSON3.read(read(_WEEK11_SPECS_PATH_FOR_WEEK14, String))
    week11_by_variant = Dict(
        String(entry["variant"]) => entry for entry in week11["variants"]
    )

    @test Int.(collect(fixture["token_ids_0_based"])) ==
        [1, 9707, 13, 151643, 100, 42, 151645, 2]
    @test Int(fixture["decode_token_id_0_based"]) == 17
    @test Int(fixture["greedy_steps"]) == 16

    tolerances = fixture["tolerances"]
    for key in (
        "embedding_max_abs",
        "blocks_scaled_max_abs",
        "final_hidden_max_abs",
        "logits_max_abs",
        "decode_max_abs",
    )
        @test Float64(tolerances[key]) > 0.0
    end

    models = fixture["models"]
    @test [String(entry["variant"]) for entry in models] ==
        ["qwen3_0_6b", "qwen3_1_7b", "qwen3_4b", "qwen3_8b"]

    for entry in models
        variant = String(entry["variant"])
        reference = week11_by_variant[variant]
        @test String(entry["revision"]) == String(reference["revision"])
        @test Int(entry["parameter_count"]) ==
            qwen3_dense_parameter_count(qwen3_dense_spec(Symbol(variant)))
        @test length(collect(entry["greedy_token_ids_0_based"])) == 16

        parity = entry["parity"]
        @test Float64(parity["embedding_max_abs"]) <=
            Float64(tolerances["embedding_max_abs"])
        @test Float64(parity["blocks_scaled_max_abs"]) <=
            Float64(tolerances["blocks_scaled_max_abs"])
        @test Float64(parity["final_hidden_max_abs"]) <=
            Float64(tolerances["final_hidden_max_abs"])
        @test Float64(parity["logits_max_abs"]) <=
            Float64(tolerances["logits_max_abs"])
        @test Float64(parity["dynamic_decode_max_abs"]) <=
            Float64(tolerances["decode_max_abs"])
        @test Bool(parity["argmax_equal"])
        @test Bool(parity["greedy_match"])
        @test Float64(parity["bf16_tree_gib"]) <
            0.6 * Float64(entry["float32_tree_gib"])
    end
end

function _week14_integration(variant::Symbol, model_dir::AbstractString)
    fixture = JSON3.read(read(_WEEK14_ASSETS_PATH, String))
    entry = only(
        e for e in fixture["models"] if Symbol(String(e["variant"])) == variant
    )
    reference_dir = joinpath(model_dir, "lifeai-references", "week14-bf16-parity")
    isfile(joinpath(reference_dir, "reference.json")) || error(
        "missing $reference_dir/reference.json; run " *
        "scripts/export_qwen3_reference.py --compute-dtype bfloat16 first",
    )
    metadata = JSON3.read(read(joinpath(reference_dir, "reference.json"), String))
    @test String(metadata["compute_dtype"]) == "bfloat16"
    @test String(metadata["revision"]) == String(entry["revision"])
    reference = load_safetensors(joinpath(reference_dir, "reference.safetensors"))

    tokens = reshape(
        hf_token_ids(Int.(collect(metadata["token_ids_0_based"])); vocab_size=151936),
        :,
        1,
    )
    decode_token = hf_token_ids(
        [Int(metadata["decode_token_id_0_based"])];
        vocab_size=151936,
    )
    expected_greedy = Int.(collect(metadata["greedy_token_ids_0_based"])) .+ 1

    loaded = load_hf_qwen3_model(
        model_dir;
        max_seq_len=64,
        weight_dtype=BFloat16,
    )
    @test loaded.variant !== nothing
    @test loaded.variant.variant == variant
    @test eltype(loaded.parameters.token_embedding.weight) === BFloat16

    result = hf_qwen3_bf16_forward(
        loaded.model,
        loaded.parameters,
        tokens;
        decode_token,
        greedy_steps=length(expected_greedy),
    )
    @test result.greedy_tokens == expected_greedy

    hf_layout(array) = permutedims(array, (3, 2, 1))
    tolerances = fixture["tolerances"]
    blocks_scaled_tol = Float32(Float64(tolerances["blocks_scaled_max_abs"]))

    @test maximum(abs.(
        Float32.(result.embedding) .- Float32.(hf_layout(reference["embedding"])),
    )) <= Float32(Float64(tolerances["embedding_max_abs"]))
    for layer in 0:(loaded.model.num_layers - 1)
        reference_block = Float32.(hf_layout(reference["block.$layer"]))
        block_difference = maximum(abs.(
            Float32.(result.blocks[layer + 1]) .- reference_block,
        ))
        block_scale = max(1.0f0, maximum(abs.(reference_block)))
        @test block_difference <= blocks_scaled_tol * block_scale
    end
    @test maximum(abs.(
        Float32.(result.final_hidden) .-
        Float32.(hf_layout(reference["final_hidden"])),
    )) <= Float32(Float64(tolerances["final_hidden_max_abs"]))
    reference_logits = Float32.(hf_layout(reference["logits"]))
    @test maximum(abs.(Float32.(result.logits) .- reference_logits)) <=
        Float32(Float64(tolerances["logits_max_abs"]))
    @test argmax(vec(Float32.(result.logits))) == argmax(vec(reference_logits))
    reference_decode = Float32.(hf_layout(reference["decode_logits"]))
    @test maximum(abs.(Float32.(result.decode_logits) .- reference_decode)) <=
        Float32(Float64(tolerances["decode_max_abs"]))
    @test argmax(vec(Float32.(result.decode_logits))) ==
        argmax(vec(reference_decode))
    GC.gc()
    return nothing
end

for (variant, env_name) in (
    (:qwen3_0_6b, "LIFEAI_QWEN3_BF16_0_6B_MODEL_DIR"),
    (:qwen3_1_7b, "LIFEAI_QWEN3_BF16_1_7B_MODEL_DIR"),
    (:qwen3_4b, "LIFEAI_QWEN3_BF16_4B_MODEL_DIR"),
    (:qwen3_8b, "LIFEAI_QWEN3_BF16_8B_MODEL_DIR"),
)
    if haskey(ENV, env_name)
        @testset "Qwen3 $(variant) BF16 HuggingFace integration" begin
            _week14_integration(variant, ENV[env_name])
        end
    end
end
