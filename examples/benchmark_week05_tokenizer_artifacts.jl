using LifeAI
using Printf

const PROFILES = (:character, :byte, :byte_bpe)

function build_profile(profile, documents)
    return build_document_dataset(
        documents;
        tokenizer_type=profile,
        normalization=:none,
        add_unk=true,
        vocab_size=320,
        min_frequency=2,
        validation_size=1,
        split_seed=20260720,
        seq_len=16,
        batch_size=2,
        stride=8,
        drop_last=true,
    )
end

function tokenizer_artifact_bytes(tokenizer)
    return mktempdir() do directory
        path = joinpath(directory, "tokenizer.toml")
        save_tokenizer(path, tokenizer)
        filesize(path)
    end
end

function validation_unknown_statistics(data)
    total = sum(length(record.token_ids) for record in data.encoded.validation)
    unknown = if data.tokenizer isa Tokenizer && data.tokenizer.unk_id !== nothing
        sum(
            count(==(data.tokenizer.unk_id), record.token_ids) for
            record in data.encoded.validation
        )
    else
        0
    end
    return (;
        total,
        unknown,
        rate=total == 0 ? 0.0 : unknown / total,
    )
end

function full_fixture_is_lossless(tokenizer, texts)
    return all(
        decode(tokenizer, encode(tokenizer, text)) ==
        normalize_text(text, tokenizer_config(tokenizer).normalization) for text in texts
    )
end

function main(args)
    output_directory = isempty(args) ?
        joinpath(@__DIR__, "..", "benchmark_results", "week05") : args[1]
    mkpath(output_directory)
    fixture_path = normpath(joinpath(
        @__DIR__,
        "..",
        "data",
        "fixtures",
        "week05_chinese.toml",
    ))
    documents = load_text_documents(fixture_path)
    texts = [document.text for document in documents]
    rows = NamedTuple[]

    for profile in PROFILES
        data = build_profile(profile, documents)
        statistics = tokenizer_statistics(data.tokenizer, texts)
        unknown = validation_unknown_statistics(data)
        push!(
            rows,
            (;
                profile=String(profile),
                vocabulary_size=vocab_size(data.tokenizer),
                tokenizer_artifact_bytes=tokenizer_artifact_bytes(data.tokenizer),
                documents=statistics.documents,
                corpus_tokens=statistics.tokens,
                corpus_bytes=statistics.bytes,
                corpus_unicode_scalars=statistics.characters,
                tokens_per_byte=statistics.tokens_per_byte,
                tokens_per_unicode_scalar=statistics.tokens_per_character,
                bytes_per_token=statistics.bytes_per_token,
                validation_tokens=unknown.total,
                validation_unknown_tokens=unknown.unknown,
                validation_unknown_rate=unknown.rate,
                full_fixture_lossless=full_fixture_is_lossless(data.tokenizer, texts),
                tokenizer_fingerprint=tokenizer_fingerprint(data.tokenizer),
                split_fingerprint=data.split.fingerprint,
            ),
        )
    end

    tsv_path = joinpath(output_directory, "tokenizer_matrix.tsv")
    open(tsv_path, "w") do io
        println(io, join(string.(keys(first(rows))), '\t'))
        for row in rows
            println(io, join(string.(values(row)), '\t'))
        end
    end

    markdown_path = joinpath(output_directory, "tokenizer_matrix.md")
    open(markdown_path, "w") do io
        println(io, "# Week 05 tokenizer-only matrix")
        println(io)
        println(io, "All profiles use the same four-document fixture and deterministic split. Byte/BPE are lossless on unseen UTF-8; character BPB must be read together with its validation unknown rate.")
        println(io)
        println(io, "| Profile | Vocab | Artifact KiB | Corpus tokens | Corpus bytes | Unicode scalars | Tokens/byte | Tokens/scalar | Bytes/token | Validation unk % | Lossless fixture |")
        println(io, "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |")
        for row in rows
            @printf(
                io,
                "| %s | %d | %.2f | %d | %d | %d | %.4f | %.4f | %.4f | %.3f | %s |\n",
                row.profile,
                row.vocabulary_size,
                row.tokenizer_artifact_bytes / 1024,
                row.corpus_tokens,
                row.corpus_bytes,
                row.corpus_unicode_scalars,
                row.tokens_per_byte,
                row.tokens_per_unicode_scalar,
                row.bytes_per_token,
                100 * row.validation_unknown_rate,
                row.full_fixture_lossless,
            )
        end
    end

    println(read(markdown_path, String))
    println("[week05] tokenizer raw: ", abspath(tsv_path))
    println("[week05] tokenizer summary: ", abspath(markdown_path))
end

abspath(PROGRAM_FILE) == @__FILE__ && main(ARGS)
