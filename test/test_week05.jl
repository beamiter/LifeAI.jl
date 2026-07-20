using Test
using Random
using Lux
using Serialization
using LifeAI:
    AbstractTokenizer,
    ByteBPETokenizer,
    ByteTokenizer,
    DatasetLoader,
    DocumentDatasetLoader,
    GPTModel,
    TextDocument,
    Tokenizer,
    TrainerGPT,
    bits_per_byte,
    build_document_dataset,
    decode,
    decode_bytes,
    encode,
    evaluate_gpt,
    fit_byte_bpe,
    fit_tokenizer,
    generate,
    generate_cached,
    init_train_state,
    load_checkpoint,
    load_dataset_artifact,
    load_tokenizer,
    normalize_text,
    resume_gpt!,
    save_checkpoint,
    save_dataset_artifact,
    save_tokenizer,
    special_token_id,
    split_documents,
    target_byte_count,
    tokenizer_config,
    tokenizer_fingerprint,
    tokenizer_statistics,
    train_step!,
    vocab_size

_week05_document(id, text) = TextDocument(
    id,
    text;
    source_id="week05-fixture",
    source_location="repository fixture",
    license="CC0-1.0",
)

function _week05_tree_isapprox(left, right; atol=1.0f-6, rtol=1.0f-5)
    typeof(left) == typeof(right) || return false
    if left === nothing
        return true
    elseif left isa Number
        return isapprox(left, right; atol, rtol)
    elseif left isa AbstractArray
        return size(left) == size(right) && isapprox(left, right; atol, rtol)
    elseif left isa NamedTuple
        keys(left) == keys(right) || return false
        return all(
            _week05_tree_isapprox(a, b; atol, rtol) for
            (a, b) in zip(values(left), values(right))
        )
    elseif left isa Tuple
        length(left) == length(right) || return false
        return all(
            _week05_tree_isapprox(a, b; atol, rtol) for
            (a, b) in zip(left, right)
        )
    elseif isstructtype(typeof(left)) && fieldcount(typeof(left)) > 0
        return all(
            _week05_tree_isapprox(
                getfield(left, index),
                getfield(right, index);
                atol,
                rtol,
            ) for index in 1:fieldcount(typeof(left))
        )
    end
    return isequal(left, right)
end

@testset "Week 05 tokenizer interfaces and artifacts" begin
    legacy_text = "你好，世界。\n你好，GPT！"
    legacy = fit_tokenizer(legacy_text; add_unk=true)
    @test legacy isa AbstractTokenizer
    @test decode(legacy, encode(legacy, legacy_text)) == legacy_text
    @test tokenizer_config(legacy).type == :character

    byte = ByteTokenizer()
    texts = [
        "生命感来自连续的观察、记忆和行动。",
        "emoji: 🐕🤖✨",
        "组合字符: e\u0301",
        "混合语言 Julia + 中文 + 123",
    ]
    for text in texts
        ids = encode(byte, text)
        @test all(id -> 1 <= id <= vocab_size(byte), ids)
        @test decode_bytes(byte, ids) == Vector{UInt8}(codeunits(text))
        @test decode(byte, ids) == text
        with_boundaries = encode(byte, text; add_special_tokens=true)
        @test first(with_boundaries) == special_token_id(byte, :bos)
        @test last(with_boundaries) == special_token_id(byte, :eos)
        @test decode(byte, with_boundaries; skip_special_tokens=true) == text
    end
    @test vocab_size(byte) == 259
    @test_throws ArgumentError decode(byte, [256]; errors=:strict)
    @test occursin('�', decode(byte, [256]; errors=:replace))

    nfc = ByteTokenizer(normalization=:nfc)
    decomposed = "e\u0301"
    @test decode(nfc, encode(nfc, decomposed)) == normalize_text(decomposed, :nfc) == "é"

    training_texts = [
        repeat("生命感来自记忆与行动。", 30),
        repeat("小机器人观察环境并保存经验。", 30),
    ]
    bpe_a = fit_byte_bpe(training_texts; vocab_size=288, min_frequency=2)
    bpe_b = fit_byte_bpe(reverse(training_texts); vocab_size=288, min_frequency=2)
    @test bpe_a isa ByteBPETokenizer
    @test bpe_a.merges == bpe_b.merges
    @test tokenizer_fingerprint(bpe_a) == tokenizer_fingerprint(bpe_b)
    @test decode(bpe_a, encode(bpe_a, join(training_texts))) == join(training_texts)
    @test length(encode(bpe_a, first(training_texts))) <
          length(encode(byte, first(training_texts)))

    statistics = tokenizer_statistics(bpe_a, training_texts)
    @test statistics.documents == 2
    @test statistics.bytes > 0
    @test statistics.tokens > 0
    @test statistics.tokens_per_byte > 0

    mktempdir() do directory
        for tokenizer in (legacy, byte, bpe_a)
            path = joinpath(directory, "$(tokenizer_config(tokenizer).type).toml")
            @test save_tokenizer(path, tokenizer) == abspath(path)
            restored = load_tokenizer(path)
            @test typeof(restored) == typeof(tokenizer)
            @test tokenizer_fingerprint(restored) == tokenizer_fingerprint(tokenizer)
            probe = tokenizer isa Tokenizer ? legacy_text : first(training_texts)
            @test encode(restored, probe) == encode(tokenizer, probe)
        end

        bpe_path = joinpath(directory, "byte_bpe.toml")
        artifact = read(bpe_path, String)
        fingerprint = tokenizer_fingerprint(bpe_a)
        write(bpe_path, replace(artifact, fingerprint => repeat("0", length(fingerprint))))
        @test_throws ArgumentError load_tokenizer(bpe_path)
    end
end

@testset "Week 05 document split, train-only BPE, and boundary-safe loader" begin
    seed = 20260720
    placeholders = [_week05_document("doc-$index", "placeholder") for index in 1:4]
    preliminary = split_documents(placeholders; validation_size=1, seed)
    validation_id = only(preliminary.validation).id
    documents = TextDocument[
        _week05_document(
            placeholder.id,
            placeholder.id == validation_id ?
                repeat("ZZ validation-only pair. ", 80) :
                repeat("生命感来自连续观察、记忆、反馈与行动。", 80),
        ) for placeholder in placeholders
    ]

    split_a = split_documents(documents; validation_size=1, seed)
    split_b = split_documents(reverse(documents); validation_size=1, seed)
    @test [document.id for document in split_a.train] ==
          [document.id for document in split_b.train]
    @test [document.id for document in split_a.validation] == [validation_id]
    @test split_a.fingerprint == split_b.fingerprint
    @test isempty(intersect(
        Set(document.id for document in split_a.train),
        Set(document.id for document in split_a.validation),
    ))

    data = build_document_dataset(
        documents;
        tokenizer_type=:byte_bpe,
        vocab_size=276,
        min_frequency=2,
        validation_size=1,
        split_seed=seed,
        seq_len=8,
        batch_size=2,
        stride=4,
        drop_last=false,
    )
    @test data.tokenizer isa ByteBPETokenizer
    @test [document.id for document in data.split.validation] == [validation_id]
    @test !any(bytes -> bytes == UInt8[0x5a, 0x5a], data.tokenizer.token_bytes)
    @test data.train isa DocumentDatasetLoader
    @test data.validation isa DocumentDatasetLoader
    @test target_byte_count(data.train) > 0
    @test target_byte_count(data.validation) > 0

    for loader in (data.train, data.validation)
        for (document_index, start) in loader.starts
            @test start + loader.seq_len <= length(loader.token_documents[document_index])
        end
    end

    flat = DatasetLoader(
        ByteTokenizer(),
        repeat("你好", 12);
        seq_len=4,
        batch_size=2,
        stride=2,
        drop_last=false,
    )
    @test length(flat) > 0

    explicit = DocumentDatasetLoader(
        [[1, 2, 3, 4], [10, 11, 12, 13]];
        seq_len=2,
        batch_size=1,
        stride=1,
        drop_last=false,
    )
    samples = [vec(first(explicit[index])) for index in 1:length(explicit)]
    @test all(sample -> all(<(10), sample) || all(>=(10), sample), samples)

    mktempdir() do directory
        artifact = save_dataset_artifact(
            directory,
            data;
            name="week05-chinese-fixture",
            version="1",
        )
        restored = load_dataset_artifact(
            directory;
            seq_len=8,
            batch_size=2,
            stride=4,
            drop_last=false,
        )
        @test restored.fingerprint == artifact.fingerprint
        @test tokenizer_fingerprint(restored.tokenizer) ==
              tokenizer_fingerprint(data.tokenizer)
        @test restored.train[1] == data.train[1]
        @test restored.validation[1] == data.validation[1]

        open(artifact.train_path, "a") do io
            print(io, "tamper")
        end
        @test_throws ArgumentError load_dataset_artifact(
            directory;
            seq_len=8,
            batch_size=2,
            stride=4,
            drop_last=false,
        )
    end
end

@testset "Week 05 byte-normalized evaluation, checkpoint, and generation" begin
    documents = [
        _week05_document("train-a", repeat("生命感来自连续观察与行动。", 60)),
        _week05_document("train-b", repeat("机器人保存经验并根据反馈调整。", 60)),
        _week05_document("validation", repeat("记忆让长期互动保持连续。", 60)),
    ]
    data = build_document_dataset(
        documents;
        tokenizer_type=:byte_bpe,
        vocab_size=276,
        min_frequency=2,
        validation_size=1,
        split_seed=17,
        seq_len=8,
        batch_size=2,
        stride=4,
        drop_last=true,
    )
    model = GPTModel(
        vocab_size(data.tokenizer),
        16,
        2,
        1;
        max_seq_len=16,
        use_rope=true,
        norm_type=:rmsnorm,
        mlp_type=:swiglu,
        tie_embeddings=true,
    )
    trainer = TrainerGPT(learning_rate=1.0f-3, max_grad_norm=1.0f0)
    state = init_train_state(Xoshiro(505), model, trainer)

    metrics, _ = evaluate_gpt(
        model,
        state.parameters,
        state.states,
        data.validation,
    )
    @test metrics.bytes == target_byte_count(data.validation)
    @test metrics.bits_per_byte ≈ bits_per_byte(metrics.total_nll, metrics.bytes) atol=1.0f-6
    @test metrics.nll_per_byte ≈ metrics.total_nll / metrics.bytes atol=1.0f-6
    @test metrics.tokens_per_byte ≈ metrics.tokens / metrics.bytes atol=1.0f-6

    state, loss, gradients = train_step!(trainer, state, data.train[1])
    @test isfinite(loss)
    @test state.step == 1
    @test gradients !== nothing
    x, _ = data.train[1]
    logits_before, _ = model(x, state.parameters, state.states)

    mktempdir() do directory
        path = joinpath(directory, "week05.checkpoint")
        save_checkpoint(
            path,
            model,
            data.tokenizer,
            trainer,
            state;
            progress=(; epoch=1, batch=1),
            metadata=(; week=5, dataset_fingerprint=data.split.fingerprint),
        )
        checkpoint = load_checkpoint(path; backend=:zygote)
        @test checkpoint.tokenizer isa ByteBPETokenizer
        @test tokenizer_fingerprint(checkpoint.tokenizer) ==
              tokenizer_fingerprint(data.tokenizer)
        @test checkpoint.metadata.week == 5
        logits_after, _ = checkpoint.model(
            x,
            checkpoint.train_state.parameters,
            checkpoint.train_state.states,
        )
        @test logits_after ≈ logits_before atol=1.0f-6 rtol=1.0f-5
        @test _week05_tree_isapprox(
            checkpoint.train_state.optimizer_state,
            state.optimizer_state,
        )

        prompt = "生命"
        generated_text, _ = generate(
            checkpoint.model,
            checkpoint.train_state.parameters,
            checkpoint.train_state.states,
            checkpoint.tokenizer,
            prompt;
            max_new_tokens=2,
            temperature=0,
            rng=Xoshiro(506),
        )
        @test !isempty(generated_text)
        cached_text, _ = generate_cached(
            checkpoint.model,
            checkpoint.train_state.parameters,
            checkpoint.train_state.states,
            checkpoint.tokenizer,
            prompt;
            max_new_tokens=2,
            temperature=0,
            rng=Xoshiro(507),
        )
        @test !isempty(cached_text)

        resumed, losses = resume_gpt!(
            checkpoint,
            data.train;
            epochs=1,
            max_steps=1,
        )
        @test resumed.step == state.step + 1
        @test length(losses) == 1
        @test isfinite(only(losses))
    end
end

@testset "Week 05 legacy v2 character checkpoint payload remains loadable" begin
    tokenizer = fit_tokenizer(repeat("abc", 12); add_unk=true)
    loader = DatasetLoader(
        tokenizer,
        repeat("abc", 12);
        seq_len=4,
        batch_size=2,
        stride=2,
    )
    model = GPTModel(vocab_size(tokenizer), 8, 2, 1; max_seq_len=4)
    trainer = TrainerGPT(learning_rate=1.0f-3)
    state = init_train_state(Xoshiro(601), model, trainer)

    mktempdir() do directory
        current_path = joinpath(directory, "current-v2.checkpoint")
        save_checkpoint(current_path, model, tokenizer, trainer, state)
        payload = open(current_path, "r") do io
            deserialize(io)
        end
        old_tokenizer_payload = (;
            type=:character,
            id_to_token=payload.tokenizer.id_to_token,
            unk_id=payload.tokenizer.unk_id,
        )
        old_v2_path = joinpath(directory, "old-v2.checkpoint")
        open(old_v2_path, "w") do io
            serialize(io, merge(payload, (; tokenizer=old_tokenizer_payload)))
        end

        restored = load_checkpoint(old_v2_path; backend=:zygote)
        @test restored.source_format_version == 2
        @test restored.tokenizer isa Tokenizer
        @test restored.tokenizer.id_to_token == tokenizer.id_to_token
        @test restored.tokenizer.unk_id == tokenizer.unk_id
        x, _ = loader[1]
        expected, _ = model(x, state.parameters, state.states)
        actual, _ = restored.model(
            x,
            restored.train_state.parameters,
            restored.train_state.states,
        )
        @test actual ≈ expected atol=1.0f-6 rtol=1.0f-5
    end
end
