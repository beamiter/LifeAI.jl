using BFloat16s: BFloat16
using JSON3
import LifeAI
using LifeAI: GPTModel

function _qwen3_tiny_model_fixture_dir(
    directory;
    tie=false,
    vocab_size::Integer=19,
    max_seq_len::Integer=16,
)
    config_json = Dict{String,Any}(
        "architectures" => ["Qwen3ForCausalLM"],
        "attention_bias" => false,
        "attention_dropout" => 0.0,
        "head_dim" => 4,
        "hidden_act" => "silu",
        "hidden_size" => 8,
        "intermediate_size" => 12,
        "max_position_embeddings" => max(32, Int(max_seq_len)),
        "model_type" => "qwen3",
        "num_attention_heads" => 4,
        "num_hidden_layers" => 2,
        "num_key_value_heads" => 2,
        "rms_norm_eps" => 1.0e-6,
        "rope_scaling" => nothing,
        "rope_theta" => 1_000_000,
        "sliding_window" => nothing,
        "tie_word_embeddings" => tie,
        "torch_dtype" => "bfloat16",
        "use_sliding_window" => false,
        "vocab_size" => Int(vocab_size),
    )
    write(joinpath(directory, "config.json"), JSON3.write(config_json))
    config = LifeAI.load_hf_qwen3_config(
        joinpath(directory, "config.json");
        max_seq_len=Int(max_seq_len),
    )
    model = GPTModel(config)
    tensors = Dict{String,Any}()
    values_for(shape, seed; norm=false) = reshape(
        Float32[
            (norm ? 1.0f0 : 0.0f0) +
                Float32(mod(i + seed, 7) - 3) / 64.0f0
            for i in 1:prod(shape)
        ],
        shape,
    )
    tensors["model.embed_tokens.weight"] =
        values_for((model.vocab_size, model.d_model), 1)
    q_dim = model.num_heads * model.head_dim
    kv_dim = model.num_kv_heads * model.head_dim
    for layer in 0:(model.num_layers - 1)
        prefix = "model.layers.$layer"
        tensors["$prefix.input_layernorm.weight"] =
            values_for((model.d_model,), 10 + layer; norm=true)
        tensors["$prefix.self_attn.q_proj.weight"] =
            values_for((q_dim, model.d_model), 20 + layer)
        tensors["$prefix.self_attn.k_proj.weight"] =
            values_for((kv_dim, model.d_model), 30 + layer)
        tensors["$prefix.self_attn.v_proj.weight"] =
            values_for((kv_dim, model.d_model), 40 + layer)
        tensors["$prefix.self_attn.o_proj.weight"] =
            values_for((model.d_model, q_dim), 50 + layer)
        tensors["$prefix.self_attn.q_norm.weight"] =
            values_for((model.head_dim,), 60 + layer; norm=true)
        tensors["$prefix.self_attn.k_norm.weight"] =
            values_for((model.head_dim,), 70 + layer; norm=true)
        tensors["$prefix.post_attention_layernorm.weight"] =
            values_for((model.d_model,), 80 + layer; norm=true)
        tensors["$prefix.mlp.gate_proj.weight"] =
            values_for((model.mlp_hidden_dim, model.d_model), 90 + layer)
        tensors["$prefix.mlp.up_proj.weight"] =
            values_for((model.mlp_hidden_dim, model.d_model), 100 + layer)
        tensors["$prefix.mlp.down_proj.weight"] =
            values_for((model.d_model, model.mlp_hidden_dim), 110 + layer)
    end
    tensors["model.norm.weight"] =
        values_for((model.d_model,), 120; norm=true)
    tie || (
        tensors["lm_head.weight"] =
            values_for((model.vocab_size, model.d_model), 130)
    )

    header = Dict{String,Any}()
    data = UInt8[]
    offset = 0
    for name in sort!(collect(keys(tensors)))
        values = Float32.(tensors[name])
        flat = ndims(values) <= 1 ?
            vec(values) :
            vec(permutedims(values, Tuple(reverse(1:ndims(values)))))
        bits = UInt16[
            UInt16(reinterpret(UInt32, value) >> 16) for value in flat
        ]
        bytes = collect(reinterpret(UInt8, bits))
        header[name] = Dict(
            "dtype" => "BF16",
            "shape" => collect(size(values)),
            "data_offsets" => [offset, offset + length(bytes)],
        )
        append!(data, bytes)
        offset += length(bytes)
    end
    header_text = JSON3.write(header)
    padding = mod(-ncodeunits(header_text), 8)
    padded = header_text * repeat(" ", padding)
    open(joinpath(directory, "model.safetensors"), "w") do io
        write(io, UInt64(ncodeunits(padded)))
        write(io, codeunits(padded))
        write(io, data)
    end
    return model
end
