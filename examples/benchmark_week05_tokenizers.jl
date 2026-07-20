using Dates
using LifeAI
using Lux
using Printf
using Random
using Statistics

const WEEK05_PROFILES = (:character, :byte, :byte_bpe)

env_int(name, default) = parse(Int, get(ENV, name, string(default)))
env_float(name, default) = parse(Float64, get(ENV, name, string(default)))

function benchmark_seeds()
    raw = replace(get(ENV, "LIFEAI_WEEK05_SEEDS", "20260720,20260721,20260722"), ',' => ' ')
    seeds = parse.(Int, split(raw))
    length(seeds) >= 3 || error("LIFEAI_WEEK05_SEEDS must contain at least three seeds")
    return seeds
end

function benchmark_config()
    config = (;
        d_model=env_int("LIFEAI_WEEK05_EMBED_DIM", 32),
        num_heads=env_int("LIFEAI_WEEK05_NUM_HEADS", 4),
        num_layers=env_int("LIFEAI_WEEK05_NUM_LAYERS", 2),
        seq_len=env_int("LIFEAI_WEEK05_SEQ_LEN", 16),
        batch_size=env_int("LIFEAI_WEEK05_BATCH_SIZE", 2),
        stride=env_int("LIFEAI_WEEK05_STRIDE", 8),
        train_steps=env_int("LIFEAI_WEEK05_TRAIN_STEPS", 8),
        learning_rate=env_float("LIFEAI_WEEK05_LEARNING_RATE", 2.0e-3),
        bpe_vocab_size=env_int("LIFEAI_WEEK05_BPE_VOCAB_SIZE", 320),
        tokenizer_repetitions=env_int("LIFEAI_WEEK05_TOKENIZER_REPETITIONS", 20),
        seeds=benchmark_seeds(),
    )
    config.d_model > 0 || error("embedding dimension must be positive")
    config.num_heads > 0 || error("head count must be positive")
    config.d_model % config.num_heads == 0 || error("embedding dimension must divide heads")
    config.num_layers > 0 || error("layer count must be positive")
    config.seq_len > 0 || error("sequence length must be positive")
    config.batch_size > 0 || error("batch size must be positive")
    config.stride > 0 || error("stride must be positive")
    config.train_steps > 0 || error("training steps must be positive")
    config.learning_rate > 0 || error("learning rate must be positive")
    config.tokenizer_repetitions > 0 || error("tokenizer repetitions must be positive")
    return config
end

function fixture_documents()
    path = normpath(joinpath(@__DIR__, "..", "data", "fixtures", "week05_chinese.toml"))
    return load_text_documents(path)
end

function profile_data(profile::Symbol, documents, config)
    profile in WEEK05_PROFILES || error("unknown Week 05 profile: $profile")
    return build_document_dataset(
        documents;
        tokenizer_type=profile,
        normalization=:none,
        add_unk=true,
        vocab_size=config.bpe_vocab_size,
        min_frequency=2,
        validation_size=1,
        split_seed=20260720,
        seq_len=config.seq_len,
        batch_size=config.batch_size,
        stride=config.stride,
        drop_last=true,
    )
end

function batch_target_bytes(loader::DocumentDatasetLoader, batch_index::Int)
    loader.byte_lengths === nothing && error("loader has no byte-length metadata")
    1 <= batch_index <= length(loader) || throw(BoundsError(loader, batch_index))
    first_sample = (batch_index - 1) * loader.batch_size + 1
    last_sample = first_sample + loader.batch_size - 1
    total = 0
    for sample_index in first_sample:last_sample
        document_index, start = loader.starts[sample_index]
        lengths = loader.byte_lengths[document_index]
        total += sum(@view(lengths[(start + 1):(start + loader.seq_len)]))
    end
    return total
end

function context_coverage_bytes(data, seq_len::Int)
    values = Float64[]
    for record in vcat(collect(data.encoded.train), collect(data.encoded.validation))
        count = min(seq_len, length(record.byte_lengths))
        count == 0 || push!(values, sum(@view(record.byte_lengths[1:count])))
    end
    isempty(values) && error("no encoded document can measure context coverage")
    return mean(values)
end

function tokenizer_throughput(tokenizer, documents, repetitions::Int)
    texts = [document.text for document in documents]
    encoded = [encode(tokenizer, text) for text in texts]
    for (text, ids) in zip(texts, encoded)
        # Byte tokenizers must be lossless on unseen validation text. The legacy
        # character baseline deliberately maps unseen characters to <unk>, so it
        # is measured without pretending that its validation round-trip is exact.
        if !(tokenizer isa Tokenizer)
            decode(tokenizer, ids) == normalize_text(text, tokenizer_config(tokenizer).normalization) ||
                error("byte tokenizer round-trip failed before benchmarking")
        end
    end
    total_bytes = repetitions * sum(ncodeunits, texts)

    encode_sink = 0
    GC.gc()
    encode_seconds = @elapsed begin
        for _ in 1:repetitions
            for text in texts
                encode_sink += length(encode(tokenizer, text))
            end
        end
    end

    decode_sink = 0
    GC.gc()
    decode_seconds = @elapsed begin
        for _ in 1:repetitions
            for ids in encoded
                decode_sink += ncodeunits(decode(tokenizer, ids))
            end
        end
    end
    encode_sink > 0 || error("tokenizer encode benchmark produced no tokens")
    decode_sink > 0 || error("tokenizer decode benchmark produced no bytes")
    return (;
        encode_bytes_per_second=total_bytes / max(encode_seconds, eps(Float64)),
        decode_bytes_per_second=total_bytes / max(decode_seconds, eps(Float64)),
    )
end

function run_profile_seed(profile, data, documents, config, seed)
    model = GPTModel(
        vocab_size(data.tokenizer),
        config.d_model,
        config.num_heads,
        config.num_layers;
        max_seq_len=config.seq_len,
        use_rope=true,
        norm_type=:rmsnorm,
        mlp_type=:swiglu,
        tie_embeddings=true,
    )
    trainer = TrainerGPT(
        backend=:zygote,
        device=Lux.cpu_device(),
        learning_rate=config.learning_rate,
        return_gradients=false,
        max_grad_norm=1.0f0,
    )

    # Warm the architecture, then recreate the exact seeded state for measurement.
    warm_state = init_train_state(Xoshiro(seed), model, trainer)
    warm_state, _, _ = train_step!(trainer, warm_state, data.train[1])
    train_state = init_train_state(Xoshiro(seed), model, trainer)
    initial_metrics, _ = evaluate_gpt(
        model,
        train_state.parameters,
        train_state.states,
        data.validation,
    )

    total_tokens = 0
    total_bytes = 0
    step_seconds = Float64[]
    final_train_loss = NaN32
    for step in 1:config.train_steps
        batch_index = mod1(step, length(data.train))
        batch = data.train[batch_index]
        started = time_ns()
        train_state, final_train_loss, _ = train_step!(
            trainer,
            train_state,
            batch,
        )
        push!(step_seconds, Float64(time_ns() - started) / 1.0e9)
        total_tokens += length(batch[2])
        total_bytes += batch_target_bytes(data.train, batch_index)
    end

    final_metrics, _ = evaluate_gpt(
        model,
        train_state.parameters,
        train_state.states,
        data.validation,
    )
    checkpoint_bytes = mktempdir() do directory
        path = joinpath(directory, "week05.checkpoint")
        save_checkpoint(
            path,
            model,
            data.tokenizer,
            trainer,
            train_state;
            progress=(; epoch=1, batch=config.train_steps),
            train_config=(; profile, seed, steps=config.train_steps),
        )
        filesize(path)
    end
    throughput = tokenizer_throughput(
        data.tokenizer,
        documents,
        config.tokenizer_repetitions,
    )
    statistics = tokenizer_statistics(
        data.tokenizer,
        [document.text for document in documents],
    )
    elapsed = sum(step_seconds)

    return (;
        profile=String(profile),
        seed,
        tokenizer_fingerprint=tokenizer_fingerprint(data.tokenizer),
        split_fingerprint=data.split.fingerprint,
        vocabulary_size=vocab_size(data.tokenizer),
        parameter_count=Lux.parameterlength(train_state.parameters),
        checkpoint_bytes,
        seq_len=config.seq_len,
        context_bytes=context_coverage_bytes(data, config.seq_len),
        corpus_tokens=statistics.tokens,
        corpus_bytes=statistics.bytes,
        tokens_per_byte=statistics.tokens_per_byte,
        encode_bytes_per_second=throughput.encode_bytes_per_second,
        decode_bytes_per_second=throughput.decode_bytes_per_second,
        initial_validation_loss=Float64(initial_metrics.loss),
        initial_bits_per_byte=Float64(initial_metrics.bits_per_byte),
        final_train_loss=Float64(final_train_loss),
        final_validation_loss=Float64(final_metrics.loss),
        final_perplexity=Float64(final_metrics.perplexity),
        final_bits_per_byte=Float64(final_metrics.bits_per_byte),
        training_tokens_per_second=total_tokens / elapsed,
        training_raw_bytes_per_second=total_bytes / elapsed,
        training_p50_ms=median(step_seconds) * 1.0e3,
        training_p90_ms=quantile(step_seconds, 0.90) * 1.0e3,
    )
end

function write_tsv(path, rows)
    mkpath(dirname(abspath(path)))
    open(path, "w") do io
        header = keys(first(rows))
        println(io, join(string.(header), '\t'))
        for row in rows
            println(io, join(string.(values(row)), '\t'))
        end
    end
    return abspath(path)
end

function profile_summary(rows, profile)
    selected = [row for row in rows if row.profile == String(profile)]
    isempty(selected) && error("no rows for profile $profile")
    bpb = [row.final_bits_per_byte for row in selected]
    token_rate = [row.training_tokens_per_second for row in selected]
    byte_rate = [row.training_raw_bytes_per_second for row in selected]
    encode_rate = [row.encode_bytes_per_second for row in selected]
    decode_rate = [row.decode_bytes_per_second for row in selected]
    return (;
        profile=String(profile),
        vocabulary_size=only(unique([row.vocabulary_size for row in selected])),
        parameter_count=only(unique([row.parameter_count for row in selected])),
        checkpoint_kib=mean(row.checkpoint_bytes for row in selected) / 1024,
        tokens_per_byte=mean(row.tokens_per_byte for row in selected),
        context_bytes=mean(row.context_bytes for row in selected),
        final_bpb_mean=mean(bpb),
        final_bpb_min=minimum(bpb),
        final_bpb_max=maximum(bpb),
        train_tokens_per_second=median(token_rate),
        train_bytes_per_second=median(byte_rate),
        encode_bytes_per_second=median(encode_rate),
        decode_bytes_per_second=median(decode_rate),
    )
end

function write_summary(path, rows, config)
    summaries = [profile_summary(rows, profile) for profile in WEEK05_PROFILES]
    open(path, "w") do io
        println(io, "# Week 05 tokenizer comparison")
        println(io)
        println(io, "Generated: ", Dates.format(now(), dateformat"yyyy-mm-dd HH:MM:SS"))
        println(io)
        println(
            io,
            "Fixed model: d_model=$(config.d_model), heads=$(config.num_heads), " *
            "layers=$(config.num_layers), seq_len=$(config.seq_len), batch=$(config.batch_size), " *
            "steps=$(config.train_steps), seeds=$(join(config.seeds, ',')); " *
            "RMSNorm + SwiGLU + tied output projection.",
        )
        println(io)
        println(io, "Per-token perplexity is intentionally not ranked across tokenizers; final BPB is the shared validation unit.")
        println(io)
        println(io, "| Profile | Vocab | Parameters | Checkpoint KiB | Tokens/byte | Context bytes / $(config.seq_len) tokens | Final BPB mean [range] | Train tokens/s | Train raw bytes/s | Encode bytes/s | Decode bytes/s |")
        println(io, "| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: |")
        for summary in summaries
            @printf(
                io,
                "| %s | %d | %d | %.1f | %.4f | %.1f | %.4f [%.4f, %.4f] | %.1f | %.1f | %.1f | %.1f |\n",
                summary.profile,
                summary.vocabulary_size,
                summary.parameter_count,
                summary.checkpoint_kib,
                summary.tokens_per_byte,
                summary.context_bytes,
                summary.final_bpb_mean,
                summary.final_bpb_min,
                summary.final_bpb_max,
                summary.train_tokens_per_second,
                summary.train_bytes_per_second,
                summary.encode_bytes_per_second,
                summary.decode_bytes_per_second,
            )
        end
    end
    return abspath(path)
end

function main(args)
    output_directory = isempty(args) ?
        joinpath(@__DIR__, "..", "benchmark_results", "week05") : args[1]
    config = benchmark_config()
    documents = fixture_documents()
    rows = NamedTuple[]

    for profile in WEEK05_PROFILES
        data = profile_data(profile, documents, config)
        for seed in config.seeds
            println("[week05] profile=$profile seed=$seed")
            row = run_profile_seed(profile, data, documents, config, seed)
            push!(rows, row)
            @printf(
                "[week05] profile=%s seed=%d bpb=%.4f tokens/s=%.1f raw-bytes/s=%.1f\n",
                row.profile,
                row.seed,
                row.final_bits_per_byte,
                row.training_tokens_per_second,
                row.training_raw_bytes_per_second,
            )
        end
    end

    mkpath(output_directory)
    raw_path = write_tsv(joinpath(output_directory, "cpu_matrix.tsv"), rows)
    summary_path = write_summary(joinpath(output_directory, "summary.md"), rows, config)
    println("[week05] raw: $raw_path")
    println("[week05] summary: $summary_path")
    println(read(summary_path, String))
end

abspath(PROGRAM_FILE) == @__FILE__ && main(ARGS)
