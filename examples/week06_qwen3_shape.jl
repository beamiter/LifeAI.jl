# Week 06 qwen3_shape end-to-end example.
#
# A scaled-down configuration that is structurally isomorphic to Qwen3 dense:
# GQA (4 query heads / 2 KV heads), QK-Norm, head_dim decoupled from d_model,
# RMSNorm, SwiGLU, no attention bias, RoPE theta = 1e6, tied embeddings.
#
# The full loop: fit tokenizer -> train -> validate -> save -> load -> resume
# -> cached generation. Run with:
#
#     julia --project=. examples/week06_qwen3_shape.jl

using Lux
using Optimisers
using Random
using LifeAI
using LifeAI:
    generate_cached,
    gpt_config,
    init_train_state,
    kv_cache_correctness,
    load_checkpoint,
    resume_gpt!,
    save_checkpoint,
    train_gpt!,
    train_validation_loaders

function main()
    corpus = """
    千里之行，始于足下。合抱之木，生于毫末。
    九层之台，起于累土。天下大事，必作于细。
    工欲善其事，必先利其器。学而不思则罔，思而不学则殆。
    知人者智，自知者明。胜人者有力，自胜者强。
    上善若水，水善利万物而不争。知足者富，强行者有志。
    温故而知新，可以为师矣。三人行，必有我师焉。
    择其善者而从之，其不善者而改之。敏而好学，不耻下问。
    """

    data = train_validation_loaders(
        corpus;
        seq_len=16,
        batch_size=2,
        stride=4,
        validation_fraction=0.2,
    )
    tokenizer = data.tokenizer

    # Scaled-down Qwen3-isomorphic configuration. Real Qwen3-0.6B uses
    # d_model=1024, 16Q/8KV heads, head_dim=128, 28 layers — the shape rules
    # (head_dim independent of d_model ÷ num_heads, grouped KV, QK-norm)
    # are identical here.
    model = GPTModel(
        vocab_size(tokenizer),
        64,             # d_model
        4,              # query heads
        3;              # layers
        num_kv_heads=2, # GQA: 2 groups
        head_dim=32,    # decoupled: 4 * 32 = 128 != 64
        use_qk_norm=true,
        qk_norm_epsilon=1.0f-6,
        use_rope=true,
        rope_theta=1.0f6,
        norm_type=:rmsnorm,
        norm_epsilon=1.0f-6,
        mlp_type=:swiglu,
        use_bias=false,
        tie_embeddings=true,
        max_seq_len=128,
    )

    config = gpt_config(model)
    println("== qwen3_shape configuration ==")
    for key in (:num_heads, :num_kv_heads, :head_dim, :use_qk_norm, :rope_theta)
        println("  $key = $(getfield(config, key))")
    end

    trainer = TrainerGPT(; optimizer=Adam(3.0f-3), max_grad_norm=1.0f0)
    state = init_train_state(MersenneTwister(20260721), model, trainer)
    train_loader = data.train
    validation_loader = data.validation

    println("\n== train ==")
    state, losses = train_gpt!(
        trainer,
        state,
        train_loader;
        epochs=8,
        validation_loader,
        evaluate_every=length(train_loader),
    )
    metrics, _ = evaluate_gpt(model, state.parameters, state.states, validation_loader)
    println("  first loss: $(round(losses[1]; digits=4))")
    println("  final validation loss: $(round(metrics.loss; digits=4))")
    println("  final validation ppl:  $(round(metrics.perplexity; digits=4))")

    println("\n== cache correctness (full vs dynamic vs static) ==")
    rng = MersenneTwister(1)
    prompt = rand(rng, 1:vocab_size(tokenizer), 6)
    decode_tokens = rand(rng, 1:vocab_size(tokenizer), 5)
    report = kv_cache_correctness(
        model, state.parameters, state.states, prompt, decode_tokens,
    )
    println("  passed: $(report.passed)")

    println("\n== save / load / resume ==")
    dir = mktempdir()
    path = joinpath(dir, "qwen3_shape.checkpoint")
    save_checkpoint(
        path, model, tokenizer, trainer, state;
        progress=(; epoch=8, batch=length(train_loader)),
    )
    restored = load_checkpoint(path)
    println("  restored num_kv_heads = $(restored.model.num_kv_heads)")
    println("  restored use_qk_norm  = $(restored.model.use_qk_norm)")
    restored_state, _ = resume_gpt!(restored, train_loader; epochs=9)
    println("  resumed to step $(restored_state.step)")

    println("\n== cached generation ==")
    text, _ = generate_cached(
        restored.model,
        restored_state.parameters,
        restored_state.states,
        restored.tokenizer,
        "千里";
        max_new_tokens=24,
        temperature=0.8f0,
        top_k=8,
        rng=MersenneTwister(2),
    )
    println("  ", text)

    return nothing
end

main()
