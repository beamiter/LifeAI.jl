using Test
using BFloat16s: BFloat16
import LifeAI
using LifeAI: Qwen3VLRopeLayout,
    Qwen3VLTextSpec,
    hf_qwen3_vl_text_decode_step,
    hf_qwen3_vl_text_decode_step_static,
    hf_qwen3_vl_text_prefill,
    hf_qwen3_vl_text_prefill_cached,
    hf_qwen3_vl_text_prefill_static,
    init_qwen3_vl_kv_cache,
    init_qwen3_vl_static_kv_cache

function _ch45_attention_width_parameters(
    num_heads::Int,
    num_kv_heads::Int,
    head_dim::Int,
)
    hidden_size = 16
    query_width = num_heads * head_dim
    spec = Qwen3VLTextSpec(
        32,
        hidden_size,
        24,
        1,
        num_heads,
        num_kv_heads,
        head_dim,
        1.0e-6,
        10_000.0,
        32,
        true,
        (head_dim ÷ 2, 0, 0),
        true,
        "silu",
    )
    matrix(rows, columns, offset) = reshape(Float32[
        0.025f0 * sin(0.13f0 * Float32(offset + index))
        for index in 1:(rows * columns)
    ], rows, columns)
    vector(length, offset) = Float32[
        1.0f0 + 0.01f0 * sin(0.17f0 * Float32(offset + index))
        for index in 1:length
    ]
    embedding = matrix(hidden_size, spec.vocab_size, 10)
    block = (;
        norm1=vector(hidden_size, 100),
        # Zero Q/K gives a simple, stable uniform-causal attention oracle while
        # still exercising the complete GQA reshape and output projection path.
        q_weight=zeros(Float32, query_width, hidden_size),
        k_weight=zeros(Float32, head_dim * num_kv_heads, hidden_size),
        v_weight=matrix(head_dim * num_kv_heads, hidden_size, 300),
        o_weight=matrix(hidden_size, query_width, 400),
        q_norm=vector(head_dim, 500),
        k_norm=vector(head_dim, 600),
        norm2=vector(hidden_size, 700),
        # Keep the independent oracle compact: this block's MLP contributes
        # exactly zero, leaving residual + attention output to check directly.
        gate_weight=zeros(Float32, spec.intermediate_size, hidden_size),
        up_weight=zeros(Float32, spec.intermediate_size, hidden_size),
        down_weight=zeros(Float32, hidden_size, spec.intermediate_size),
    )
    return (;
        embedding,
        blocks=(block,),
        final_norm=vector(hidden_size, 1_100),
        spec,
    )
end

function _ch45_uniform_causal_block_reference(parameters, tokens)
    spec = parameters.spec
    block = only(parameters.blocks)
    x = reshape(parameters.embedding[:, tokens], spec.hidden_size, length(tokens))
    mean_square = sum(abs2, x; dims=1) ./ Float32(spec.hidden_size)
    normed = x ./ sqrt.(mean_square .+ Float32(spec.rms_norm_eps)) .*
        reshape(block.norm1, :, 1)
    values = reshape(
        block.v_weight * normed,
        spec.head_dim,
        spec.num_key_value_heads,
        length(tokens),
    )
    groups = spec.num_attention_heads ÷ spec.num_key_value_heads
    context = Array{Float32}(
        undef,
        spec.head_dim,
        spec.num_attention_heads,
        length(tokens),
    )
    for token in eachindex(tokens), kv_head in 1:spec.num_key_value_heads
        prefix_mean = vec(sum(values[:, kv_head, 1:token]; dims=2)) ./
            Float32(token)
        first_query_head = (kv_head - 1) * groups + 1
        for query_head in first_query_head:(first_query_head + groups - 1)
            context[:, query_head, token] = prefix_mean
        end
    end
    flattened = reshape(
        context,
        spec.num_attention_heads * spec.head_dim,
        length(tokens),
    )
    return reshape(x + block.o_weight * flattened, spec.hidden_size, length(tokens), 1)
end

function _ch45_text_layout(length::Int)
    positions = reshape(repeat(collect(0:(length - 1)); inner=3), 3, length, 1)
    return Qwen3VLRopeLayout(
        positions,
        zeros(Int, 1, 1),
        falses(length, 1),
        trues(length, 1),
    )
end

function _ch45_attention_width_dtype(parameters, ::Type{T}) where {T}
    convert_array(value) = value isa AbstractArray ? T.(value) : value
    blocks = map(parameters.blocks) do block
        map(convert_array, block)
    end
    return (;
        embedding=T.(parameters.embedding),
        blocks,
        final_norm=T.(parameters.final_norm),
        spec=parameters.spec,
    )
end

@testset "Chapter 45 — attention projection width is independent of hidden size" begin
    # 2*4=8 is narrower and 4*8=32 is wider than hidden_size=16. The wider
    # case uses two distinct KV heads; both cases have a GQA ratio of two.
    for (num_heads, num_kv_heads, head_dim) in ((2, 1, 4), (4, 2, 8))
        f32_parameters = _ch45_attention_width_parameters(
            num_heads,
            num_kv_heads,
            head_dim,
        )
        for dtype in (Float32, BFloat16)
            parameters = _ch45_attention_width_dtype(f32_parameters, dtype)
            tolerance = dtype === Float32 ? 3.0f-6 : 3.0f-2
            prompt = [2, 5, 7]
            layout = _ch45_text_layout(length(prompt))

            full = hf_qwen3_vl_text_prefill(
                parameters,
                prompt,
                layout;
                logits_to_keep=0,
                capture_layers=(0,),
            )
            if dtype === Float32
                expected_block = _ch45_uniform_causal_block_reference(
                    parameters,
                    prompt,
                )
                @test full.block_outputs[0] ≈ expected_block atol=2.0f-6 rtol=2.0f-6
            end

            dynamic_prefill, dynamic_cache = hf_qwen3_vl_text_prefill_cached(
                parameters,
                prompt,
                layout;
                cache=init_qwen3_vl_kv_cache(parameters),
                logits_to_keep=0,
            )
            static_cache = init_qwen3_vl_static_kv_cache(parameters; capacity=8)
            static_prefill, static_cache = hf_qwen3_vl_text_prefill_static(
                parameters,
                prompt,
                layout;
                cache=static_cache,
                logits_to_keep=0,
            )
            @test dynamic_prefill.logits ≈ full.logits atol=tolerance rtol=tolerance
            @test static_prefill.logits ≈ full.logits atol=tolerance rtol=tolerance

            tokens = copy(prompt)
            for token in (11, 13)
                dynamic_logits, dynamic_cache = hf_qwen3_vl_text_decode_step(
                    parameters,
                    token,
                    dynamic_cache,
                )
                static_logits, static_cache = hf_qwen3_vl_text_decode_step_static(
                    parameters,
                    token,
                    static_cache,
                )
                push!(tokens, token)
                recomputed = hf_qwen3_vl_text_prefill(
                    parameters,
                    tokens,
                    _ch45_text_layout(length(tokens));
                    logits_to_keep=1,
                )
                @test dynamic_logits ≈ recomputed.logits atol=tolerance rtol=tolerance
                @test static_logits ≈ recomputed.logits atol=tolerance rtol=tolerance
            end
            @test dynamic_cache.position == length(tokens)
            @test static_cache.position == length(tokens)
        end
    end
end
