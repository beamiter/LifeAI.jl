using Test
using BFloat16s: BFloat16
using JSON3
using Lux
using NNlib: batched_mul
using LifeAI:
    GPTModel,
    hf_qwen3_bf16_accel_forward,
    hf_qwen3_bf16_forward,
    hf_token_ids,
    load_hf_qwen3_model,
    load_safetensors,
    qwen3_dense_parameter_count,
    qwen3_dense_spec

isdefined(@__MODULE__, :qwen3_reference_directory) ||
    include(joinpath(@__DIR__, "..", "..", "..", "support", "repository_test_assets.jl"))

const _QWEN3_ACCELERATION_ASSETS_PATH = joinpath(
    @__DIR__,
    "fixtures",
    "qwen3_bf16_acceleration",
    "assets.json",
)

function _qwen3_acceleration_row_major_values(array)
    values = Float32.(array)
    ndims(values) <= 1 && return vec(values)
    return vec(permutedims(values, Tuple(reverse(1:ndims(values)))))
end

function _qwen3_acceleration_tensor_bytes(array)
    values = _qwen3_acceleration_row_major_values(array)
    bits = UInt16[UInt16(reinterpret(UInt32, value) >> 16) for value in values]
    return collect(reinterpret(UInt8, bits))
end

function _qwen3_acceleration_write_safetensors(path, tensors)
    header = Dict{String,Any}()
    data = UInt8[]
    offset = 0
    for name in sort!(collect(keys(tensors)))
        bytes = _qwen3_acceleration_tensor_bytes(tensors[name])
        next_offset = offset + length(bytes)
        header[name] = Dict(
            "dtype" => "BF16",
            "shape" => collect(size(tensors[name])),
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

function _qwen3_acceleration_config(; kwargs...)
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

function _qwen3_acceleration_values(shape; norm=false, seed=0)
    count = prod(shape)
    values = Float32[
        (norm ? 1.0f0 : 0.0f0) + Float32(mod(index + seed, 7) - 3) / 64.0f0
        for index in 1:count
    ]
    return reshape(values, shape)
end

function _qwen3_acceleration_fixture_dir(directory; tie=false)
    write(
        joinpath(directory, "config.json"),
        JSON3.write(_qwen3_acceleration_config(; tie_word_embeddings=tie)),
    )
    config = LifeAI.load_hf_qwen3_config(
        joinpath(directory, "config.json");
        max_seq_len=16,
    )
    model = GPTModel(config)
    tensors = Dict{String,Any}()
    tensors["model.embed_tokens.weight"] = _qwen3_acceleration_values(
        (model.vocab_size, model.d_model);
        seed=1,
    )
    q_dim = model.num_heads * model.head_dim
    kv_dim = model.num_kv_heads * model.head_dim
    for layer in 0:(model.num_layers - 1)
        prefix = "model.layers.$layer"
        tensors["$prefix.input_layernorm.weight"] =
            _qwen3_acceleration_values((model.d_model,); norm=true, seed=10 + layer)
        tensors["$prefix.self_attn.q_proj.weight"] =
            _qwen3_acceleration_values((q_dim, model.d_model); seed=20 + layer)
        tensors["$prefix.self_attn.k_proj.weight"] =
            _qwen3_acceleration_values((kv_dim, model.d_model); seed=30 + layer)
        tensors["$prefix.self_attn.v_proj.weight"] =
            _qwen3_acceleration_values((kv_dim, model.d_model); seed=40 + layer)
        tensors["$prefix.self_attn.o_proj.weight"] =
            _qwen3_acceleration_values((model.d_model, q_dim); seed=50 + layer)
        tensors["$prefix.self_attn.q_norm.weight"] =
            _qwen3_acceleration_values((model.head_dim,); norm=true, seed=60 + layer)
        tensors["$prefix.self_attn.k_norm.weight"] =
            _qwen3_acceleration_values((model.head_dim,); norm=true, seed=70 + layer)
        tensors["$prefix.post_attention_layernorm.weight"] =
            _qwen3_acceleration_values((model.d_model,); norm=true, seed=80 + layer)
        tensors["$prefix.mlp.gate_proj.weight"] =
            _qwen3_acceleration_values((model.mlp_hidden_dim, model.d_model); seed=90 + layer)
        tensors["$prefix.mlp.up_proj.weight"] =
            _qwen3_acceleration_values((model.mlp_hidden_dim, model.d_model); seed=100 + layer)
        tensors["$prefix.mlp.down_proj.weight"] =
            _qwen3_acceleration_values((model.d_model, model.mlp_hidden_dim); seed=110 + layer)
    end
    tensors["model.norm.weight"] = _qwen3_acceleration_values(
        (model.d_model,);
        norm=true,
        seed=120,
    )
    model.tie_embeddings || (tensors["lm_head.weight"] = _qwen3_acceleration_values(
        (model.vocab_size, model.d_model);
        seed=130,
    ))
    _qwen3_acceleration_write_safetensors(joinpath(directory, "model.safetensors"), tensors)
    return model
end

@testset "BF16 batched matmul keeps F32 accumulation on CPU" begin
    a = BFloat16.(_qwen3_acceleration_values((3, 4, 2); seed=5))
    b = BFloat16.(_qwen3_acceleration_values((4, 2, 2); seed=8))
    c = LifeAI._bf16a_batched_mul(a, b)
    @test eltype(c) === BFloat16
    expected = BFloat16.(batched_mul(Float32.(a), Float32.(b)))
    @test c == expected
end

function _qwen3_acceleration_expanded_attention(
    queries,
    keys,
    values;
    scaling::Float32,
    mask,
)
    head_dim, num_heads, query_tokens, batch_size = size(queries)
    _, num_kv_heads, key_tokens, _ = size(keys)
    groups = num_heads ÷ num_kv_heads
    head_map = ((0:(num_heads - 1)) .÷ groups) .+ 1
    keys_full = keys[:, head_map, :, :]
    values_full = values[:, head_map, :, :]
    q3 = reshape(
        permutedims(queries, (3, 1, 2, 4)),
        query_tokens, head_dim, num_heads * batch_size,
    )
    k3 = reshape(
        permutedims(keys_full, (1, 3, 2, 4)),
        head_dim, key_tokens, num_heads * batch_size,
    )
    scores = LifeAI._bf16a_batched_mul(q3, k3)
    scaled = BFloat16.(LifeAI._bf16a_f32(scores) .* scaling)
    if mask !== nothing
        scaled = BFloat16.(
            LifeAI._bf16a_f32(scaled) .+
            LifeAI._bf16a_f32(reshape(
                mask,
                query_tokens,
                key_tokens,
                1,
            ))
        )
    end
    scores_f = LifeAI._bf16a_f32(scaled)
    exponents = exp.(scores_f .- maximum(scores_f; dims=2))
    weights = BFloat16.(exponents ./ sum(exponents; dims=2))
    v3 = reshape(
        permutedims(values_full, (1, 3, 2, 4)),
        head_dim, key_tokens, num_heads * batch_size,
    )
    context = LifeAI._bf16a_batched_mul(
        v3,
        permutedims(weights, (2, 1, 3)),
    )
    return permutedims(
        reshape(context, head_dim, query_tokens, num_heads, batch_size),
        (1, 3, 2, 4),
    )
end

function _qwen3_acceleration_expanded_static_attention(
    queries,
    keys,
    values,
    key_positions,
    valid_length;
    scaling::Float32,
)
    head_dim, num_heads, query_tokens, batch_size = size(queries)
    _, num_kv_heads, key_tokens, _ = size(keys)
    groups = num_heads ÷ num_kv_heads
    head_map = ((0:(num_heads - 1)) .÷ groups) .+ 1
    keys_full = keys[:, head_map, :, :]
    values_full = values[:, head_map, :, :]
    q3 = reshape(
        permutedims(queries, (3, 1, 2, 4)),
        query_tokens, head_dim, num_heads * batch_size,
    )
    k3 = reshape(
        permutedims(keys_full, (1, 3, 2, 4)),
        head_dim, key_tokens, num_heads * batch_size,
    )
    scores = LifeAI._bf16a_batched_mul(q3, k3)
    scaled = LifeAI._bf16a_f32(
        BFloat16.(LifeAI._bf16a_f32(scores) .* scaling),
    )
    visible = reshape(key_positions, 1, key_tokens, 1) .<= valid_length
    masked = ifelse.(visible, scaled, LifeAI._BF16_MASK_MIN)
    exponents = exp.(masked .- maximum(masked; dims=2))
    weights = BFloat16.(exponents ./ sum(exponents; dims=2))
    v3 = reshape(
        permutedims(values_full, (1, 3, 2, 4)),
        head_dim, key_tokens, num_heads * batch_size,
    )
    context = LifeAI._bf16a_batched_mul(
        v3,
        permutedims(weights, (2, 1, 3)),
    )
    return permutedims(
        reshape(context, head_dim, query_tokens, num_heads, batch_size),
        (1, 3, 2, 4),
    )
end

@testset "grouped GQA matches expanded-head BF16 semantics" begin
    queries = BFloat16.(_qwen3_acceleration_values((4, 6, 3, 2); seed=3))
    keys = BFloat16.(_qwen3_acceleration_values((4, 2, 5, 2); seed=11))
    values = BFloat16.(_qwen3_acceleration_values((4, 2, 5, 2); seed=19))
    scaling = 1.0f0 / sqrt(4.0f0)
    for mask in (nothing, LifeAI._bf16a_causal_mask(3, 5))
        expected = _qwen3_acceleration_expanded_attention(
            queries,
            keys,
            values;
            scaling,
            mask,
        )
        actual = LifeAI._bf16a_attention(
            queries,
            keys,
            values;
            scaling,
            mask,
        )
        @test actual == expected
    end

    static_queries = queries[:, :, 1:1, :]
    key_positions = Int32.(collect(1:5))
    expected = _qwen3_acceleration_expanded_static_attention(
        static_queries,
        keys,
        values,
        key_positions,
        Int32(3);
        scaling,
    )
    actual = LifeAI._bf16a_static_attention(
        static_queries,
        keys,
        values,
        key_positions,
        Int32(3);
        scaling,
    )
    @test actual == expected
end

@testset "vectorized BF16 path is bitwise identical to the loop path" begin
    for tie in (false, true)
        mktempdir() do directory
            model = _qwen3_acceleration_fixture_dir(directory; tie)
            loaded = load_hf_qwen3_model(
                directory;
                max_seq_len=16,
                weight_dtype=BFloat16,
            )
            tokens = reshape(
                hf_token_ids([0, 4, 7, 2, 11]; vocab_size=model.vocab_size),
                :,
                1,
            )
            decode_token = hf_token_ids([3]; vocab_size=model.vocab_size)
            accel = hf_qwen3_bf16_accel_forward(
                loaded.model, loaded.parameters, tokens;
                decode_token, greedy_steps=4,
            )
            loop = hf_qwen3_bf16_forward(
                loaded.model, loaded.parameters, tokens;
                decode_token, greedy_steps=4,
            )
            @test accel.embedding == loop.embedding
            for index in 1:model.num_layers
                @test accel.blocks[index] == loop.blocks[index]
            end
            @test accel.final_hidden == loop.final_hidden
            @test accel.logits == loop.logits
            @test accel.decode_logits == loop.decode_logits
            @test accel.greedy_tokens == loop.greedy_tokens

            two_batch_tokens = hcat(
                vec(tokens),
                hf_token_ids(
                    [6, 1, 8, 5, 9];
                    vocab_size=model.vocab_size,
                ),
            )
            rope = first(values(loaded.model.blocks.layers)).attn.rope
            cos_table = BFloat16.(rope.cos_cache)
            sin_table = BFloat16.(rope.sin_cache)
            mask = LifeAI._bf16a_causal_mask(
                size(two_batch_tokens, 1),
                size(two_batch_tokens, 1),
            )
            full_caches = Vector{Any}(undef, loaded.model.num_layers)
            last_caches = Vector{Any}(undef, loaded.model.num_layers)
            fill!(full_caches, (nothing, nothing))
            fill!(last_caches, (nothing, nothing))
            full = LifeAI._bf16a_forward_pass(
                loaded.model,
                loaded.parameters,
                two_batch_tokens,
                full_caches,
                cos_table,
                sin_table,
                mask;
                start_pos=1,
            )
            last = LifeAI._bf16a_forward_pass(
                loaded.model,
                loaded.parameters,
                two_batch_tokens,
                last_caches,
                cos_table,
                sin_table,
                mask;
                start_pos=1,
                project_last_token_only=true,
            )
            @test size(last.logits) ==
                (loaded.model.vocab_size, 1, 2)
            @test last.logits == full.logits[:, end:end, :]
            @test last.embedding == full.embedding
            @test last.final_hidden == full.final_hidden
            @test last.block_outputs == full.block_outputs
            @test last_caches == full_caches

            static_tokens = two_batch_tokens[:, 1:1]
            static_key_caches = [
                zeros(
                    BFloat16,
                    loaded.model.head_dim,
                    loaded.model.num_kv_heads,
                    loaded.model.max_seq_len,
                    1,
                )
                for _ in 1:loaded.model.num_layers
            ]
            static_value_caches = deepcopy(static_key_caches)
            static_logits = LifeAI._bf16a_static_prefill(
                loaded.model,
                loaded.parameters,
                static_tokens,
                static_key_caches,
                static_value_caches,
                cos_table,
                sin_table,
                mask,
            )
            @test size(static_logits) ==
                (loaded.model.vocab_size, 1, 1)
            @test static_logits == full.logits[:, end:end, 1:1]

            @test_throws ArgumentError hf_qwen3_bf16_accel_forward(
                loaded.model,
                load_hf_qwen3_model(directory; max_seq_len=16).parameters,
                tokens,
            )
        end
    end
end

@testset "Qwen3 BF16 accelerated asset contract" begin
    fixture = JSON3.read(read(_QWEN3_ACCELERATION_ASSETS_PATH, String))
    @test Int.(collect(fixture["token_ids_0_based"])) ==
        [1, 9707, 13, 151643, 100, 42, 151645, 2]
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

    cuda_models = fixture["cuda_models"]
    @test [String(entry["variant"]) for entry in cuda_models] ==
        ["qwen3_0_6b", "qwen3_1_7b", "qwen3_4b"]
    for entry in cuda_models
        spec = qwen3_dense_spec(Symbol(String(entry["variant"])))
        @test Int(entry["parameter_count"]) == qwen3_dense_parameter_count(spec)
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
        @test Float64(parity["warm_tokens_per_second"]) > 1.0
        @test Float64(parity["gpu_tree_gib"]) < 16.0
    end

    xla = fixture["xla_prefill"]
    @test String(xla["variant"]) == "qwen3_0_6b"
    @test Float64(xla["logits_max_abs"]) <=
        Float64(tolerances["logits_max_abs"])
    @test Bool(xla["logits_argmax_equal"])
    @test Bool(xla["greedy_first_token_equal"])
    @test Float64(xla["steady_exec_seconds"]) <
        Float64(xla["compile_seconds"])
end

const _QWEN3_ACCELERATION_CUDA_ENV_NAMES = (
    (:qwen3_0_6b, "LIFEAI_QWEN3_BF16_CUDA_0_6B_MODEL_DIR"),
    (:qwen3_1_7b, "LIFEAI_QWEN3_BF16_CUDA_1_7B_MODEL_DIR"),
    (:qwen3_4b, "LIFEAI_QWEN3_BF16_CUDA_4B_MODEL_DIR"),
)

if any(haskey(ENV, name) for (_, name) in _QWEN3_ACCELERATION_CUDA_ENV_NAMES)

using CUDA

function _qwen3_acceleration_cuda_integration(variant::Symbol, model_dir::AbstractString)
    CUDA.functional() || error(
        "CUDA opt-in integration requested but CUDA is not functional",
    )
    fixture = JSON3.read(read(_QWEN3_ACCELERATION_ASSETS_PATH, String))
    reference_dir = qwen3_reference_directory(
        model_dir;
        compute_dtype="bfloat16",
    )
    metadata = JSON3.read(read(joinpath(reference_dir, "reference.json"), String))
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

    loaded = load_hf_qwen3_model(model_dir; max_seq_len=64, weight_dtype=BFloat16)
    @test loaded.variant !== nothing
    @test loaded.variant.variant == variant
    ps_gpu = CUDA.cu(loaded.parameters)
    result = hf_qwen3_bf16_accel_forward(
        loaded.model,
        ps_gpu,
        tokens;
        decode_token,
        greedy_steps=length(expected_greedy),
    )
    @test result.greedy_tokens == expected_greedy

    hf_layout(array) = permutedims(array, (3, 2, 1))
    tolerances = fixture["tolerances"]
    to_host(x) = Float32.(Array(x))
    reference_logits = Float32.(hf_layout(reference["logits"]))
    @test maximum(abs.(to_host(result.logits) .- reference_logits)) <=
        Float32(Float64(tolerances["logits_max_abs"]))
    @test argmax(vec(to_host(result.logits))) == argmax(vec(reference_logits))
    reference_decode = Float32.(hf_layout(reference["decode_logits"]))
    @test maximum(abs.(to_host(result.decode_logits) .- reference_decode)) <=
        Float32(Float64(tolerances["decode_max_abs"]))
    blocks_scaled_tol = Float32(Float64(tolerances["blocks_scaled_max_abs"]))
    for layer in 0:(loaded.model.num_layers - 1)
        reference_block = Float32.(hf_layout(reference["block.$layer"]))
        difference = maximum(abs.(to_host(result.blocks[layer + 1]) .- reference_block))
        @test difference <= blocks_scaled_tol * max(1.0f0, maximum(abs.(reference_block)))
    end
    GC.gc()
    return nothing
end

for (variant, env_name) in _QWEN3_ACCELERATION_CUDA_ENV_NAMES
    if haskey(ENV, env_name)
        @testset "Qwen3 $(variant) BF16 CUDA integration" begin
            _qwen3_acceleration_cuda_integration(variant, ENV[env_name])
        end
    end
end

end # CUDA opt-in gate
