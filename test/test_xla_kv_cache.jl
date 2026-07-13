using Test
using Random
using Lux
using LifeAI:
    GPTModel,
    XLAKVDecoder,
    xla_decode_step!,
    xla_prefill!

@testset "Reactant/XLA fixed-shape KV decoding" begin
    rng = Xoshiro(20260717)
    model = GPTModel(17, 16, 2, 2; max_seq_len=8, use_rope=true)
    ps, st = Lux.setup(rng, model)
    decoder = XLAKVDecoder(
        model,
        ps,
        st;
        batch_size=1,
        xla_backend="cpu",
    )

    prompt = reshape([1, 3, 5], 3, 1)
    reference_logits, _ = model(prompt, ps, st)
    logits, _, _ = xla_prefill!(decoder, prompt)

    @test isapprox(
        Array(logits),
        reference_logits;
        atol=1.0f-5,
        rtol=1.0f-4,
    )
    @test decoder.host_position == 3
    @test length(decoder.prefill_thunks) == 1

    context = vec(prompt)
    cache_shapes = map(layer -> size(layer.keys), decoder.cache.layers)
    compiled_decode = nothing

    for token in (7, 9, 11)
        logits, _, _ = xla_decode_step!(decoder, token)
        if compiled_decode === nothing
            compiled_decode = decoder.decode_thunk
        else
            @test decoder.decode_thunk === compiled_decode
        end
        push!(context, token)
        reference_logits, _ = model(reshape(context, :, 1), ps, st)

        @test isapprox(
            vec(Array(logits)[:, 1, 1]),
            vec(reference_logits[:, end, 1]);
            atol=1.0f-5,
            rtol=1.0f-4,
        )
        @test map(layer -> size(layer.keys), decoder.cache.layers) == cache_shapes
    end

    @test decoder.decode_thunk === compiled_decode
    @test decoder.host_position == 6

    # Matching prompt shapes reuse the existing prefill executable.
    xla_prefill!(decoder, reshape([2, 4, 6], 3, 1))
    @test length(decoder.prefill_thunks) == 1
end
