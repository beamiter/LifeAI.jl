using SHA: sha256
using TOML

const DATASET_ARTIFACT_VERSION = 1
const ENCODED_SPLIT_FORMAT = "LIFEAI_TOKENS_V1"

"""A versionable text document with explicit source and license metadata."""
struct TextDocument
    id::String
    text::String
    source_id::String
    source_location::String
    license::String
    raw_sha256::String
end

function TextDocument(
    id::AbstractString,
    text::AbstractString;
    source_id::AbstractString,
    license::AbstractString,
    source_location::AbstractString="",
    raw_sha256=nothing,
)
    isempty(id) && throw(ArgumentError("document id must not be empty"))
    isempty(source_id) && throw(ArgumentError("document source_id must not be empty"))
    isempty(license) && throw(ArgumentError("document license must not be empty"))
    computed_checksum = _sha256_hex(text)
    if raw_sha256 !== nothing && lowercase(String(raw_sha256)) != computed_checksum
        throw(ArgumentError(
            "raw checksum mismatch for document $(repr(id)): " *
            "expected $(lowercase(String(raw_sha256))), computed $computed_checksum",
        ))
    end
    return TextDocument(
        String(id),
        String(text),
        String(source_id),
        String(source_location),
        String(license),
        computed_checksum,
    )
end

"""Load `[[documents]]` entries from a TOML fixture or corpus source file."""
function load_text_documents(path::AbstractString)
    isfile(path) || throw(ArgumentError("document file does not exist: $path"))
    payload = TOML.parsefile(path)
    entries = get(payload, "documents", nothing)
    entries isa AbstractVector || throw(ArgumentError(
        "document TOML must contain one or more [[documents]] entries",
    ))
    documents = TextDocument[]
    for entry in entries
        push!(
            documents,
            TextDocument(
                entry["id"],
                entry["text"];
                source_id=entry["source_id"],
                source_location=get(entry, "source_location", ""),
                license=entry["license"],
                raw_sha256=get(entry, "raw_sha256", nothing),
            ),
        )
    end
    isempty(documents) && throw(ArgumentError("document file contains no documents"))
    return documents
end

function _validate_documents(documents::Vector{TextDocument})
    length(documents) >= 2 || throw(ArgumentError(
        "at least two documents are required for train/validation splitting",
    ))
    ids = [document.id for document in documents]
    length(unique(ids)) == length(ids) || throw(ArgumentError(
        "document ids must be unique",
    ))
    all(document -> !isempty(document.license), documents) || throw(ArgumentError(
        "every document must have explicit license metadata",
    ))
    return documents
end

function _document_split_fingerprint(
    train::Vector{TextDocument},
    validation::Vector{TextDocument},
    seed::Int,
)
    output = IOBuffer()
    println(output, "method=sha256_rank")
    println(output, "seed=", seed)
    for document in sort(train; by=document -> document.id)
        println(output, "train=", document.id, ":", document.raw_sha256)
    end
    for document in sort(validation; by=document -> document.id)
        println(output, "validation=", document.id, ":", document.raw_sha256)
    end
    return bytes2hex(sha256(take!(output)))
end

"""
    split_documents(documents; validation_fraction=0.1, validation_size=nothing, seed=0)

Deterministically assign complete documents with a SHA-256 rank. The result is
independent of input order, and no document can occur in both splits.
"""
function split_documents(
    documents;
    validation_fraction::Real=0.1,
    validation_size=nothing,
    seed::Integer=0,
)
    collected = _validate_documents(TextDocument[documents...])
    count = _validation_count(
        length(collected);
        validation_fraction,
        validation_size,
    )
    ranked = sort(
        collected;
        by=document -> (_sha256_hex("$(Int(seed))\0$(document.id)"), document.id),
    )
    validation_ids = Set(document.id for document in ranked[1:count])
    train = sort(
        [document for document in collected if !(document.id in validation_ids)];
        by=document -> document.id,
    )
    validation = sort(
        [document for document in collected if document.id in validation_ids];
        by=document -> document.id,
    )
    fingerprint = _document_split_fingerprint(train, validation, Int(seed))
    return (;
        train,
        validation,
        method=:sha256_rank,
        seed=Int(seed),
        fingerprint,
    )
end

"""
A next-token loader whose windows are indexed within individual documents. Unlike a
flat token stream, it never creates a sample that crosses a document boundary.
"""
struct DocumentDatasetLoader{T<:Integer}
    token_documents::Vector{Vector{T}}
    byte_lengths::Union{Nothing,Vector{Vector{Int}}}
    document_ids::Vector{String}
    seq_len::Int
    batch_size::Int
    stride::Int
    drop_last::Bool
    starts::Vector{Tuple{Int,Int}}
end

function DocumentDatasetLoader(
    token_documents::AbstractVector{<:AbstractVector{T}};
    byte_lengths=nothing,
    document_ids=nothing,
    seq_len::Int,
    batch_size::Int=1,
    stride::Int=seq_len,
    drop_last::Bool=true,
) where {T<:Integer}
    seq_len > 0 || throw(ArgumentError("`seq_len` must be positive"))
    batch_size > 0 || throw(ArgumentError("`batch_size` must be positive"))
    stride > 0 || throw(ArgumentError("`stride` must be positive"))
    isempty(token_documents) && throw(ArgumentError("token documents must not be empty"))

    documents = [Vector{T}(document) for document in token_documents]
    all(document -> all(>(0), document), documents) || throw(ArgumentError(
        "token documents must contain positive 1-based ids",
    ))
    ids = if document_ids === nothing
        ["document-$index" for index in eachindex(documents)]
    else
        String.(collect(document_ids))
    end
    length(ids) == length(documents) || throw(DimensionMismatch(
        "document_ids must match token_documents",
    ))
    length(unique(ids)) == length(ids) || throw(ArgumentError(
        "document_ids must be unique",
    ))

    resolved_byte_lengths = if byte_lengths === nothing
        nothing
    else
        lengths = [Int.(collect(values)) for values in byte_lengths]
        length(lengths) == length(documents) || throw(DimensionMismatch(
            "byte_lengths must match token_documents",
        ))
        for index in eachindex(documents)
            length(lengths[index]) == length(documents[index]) || throw(DimensionMismatch(
                "byte_lengths[$index] must match token_documents[$index]",
            ))
            all(>=(0), lengths[index]) || throw(ArgumentError(
                "byte lengths must be non-negative",
            ))
        end
        lengths
    end

    starts = Tuple{Int,Int}[]
    for (document_index, document) in enumerate(documents)
        last_start = length(document) - seq_len
        last_start < 1 && continue
        append!(
            starts,
            ((document_index, start) for start in 1:stride:last_start),
        )
    end
    isempty(starts) && throw(ArgumentError(
        "no document contains at least seq_len + 1 tokens",
    ))

    return DocumentDatasetLoader(
        documents,
        resolved_byte_lengths,
        ids,
        seq_len,
        batch_size,
        stride,
        drop_last,
        starts,
    )
end

num_samples(loader::DocumentDatasetLoader) = length(loader.starts)
function num_batches(loader::DocumentDatasetLoader)
    samples = num_samples(loader)
    return loader.drop_last ? samples ÷ loader.batch_size : cld(samples, loader.batch_size)
end
Base.length(loader::DocumentDatasetLoader) = num_batches(loader)
Base.eltype(::Type{<:DocumentDatasetLoader{T}}) where {T} = Tuple{Matrix{T},Matrix{T}}

function Base.getindex(loader::DocumentDatasetLoader{T}, batch_index::Integer) where {T}
    1 <= batch_index <= length(loader) || throw(BoundsError(loader, batch_index))
    first_sample = (batch_index - 1) * loader.batch_size + 1
    last_sample = min(first_sample + loader.batch_size - 1, num_samples(loader))
    current_batch_size = last_sample - first_sample + 1
    x = Matrix{T}(undef, loader.seq_len, current_batch_size)
    y = Matrix{T}(undef, loader.seq_len, current_batch_size)

    for (column, sample_index) in enumerate(first_sample:last_sample)
        document_index, start = loader.starts[sample_index]
        document = loader.token_documents[document_index]
        @inbounds for position in 1:loader.seq_len
            x[position, column] = document[start + position - 1]
            y[position, column] = document[start + position]
        end
    end
    return x, y
end

function Base.iterate(loader::DocumentDatasetLoader, batch_index::Int=1)
    batch_index > length(loader) && return nothing
    return loader[batch_index], batch_index + 1
end

"""Construct the legacy flat loader from any supported tokenizer and one text stream."""
function DatasetLoader(
    tokenizer::AbstractTokenizer,
    text::AbstractString;
    kwargs...,
)
    return DatasetLoader(encode(tokenizer, text); kwargs...)
end

"""Exact raw-byte denominator represented by all emitted target-token events."""
function target_byte_count(loader::DocumentDatasetLoader)
    loader.byte_lengths === nothing && return nothing
    emitted_samples = loader.drop_last ? length(loader) * loader.batch_size : num_samples(loader)
    total = 0
    for sample_index in 1:emitted_samples
        document_index, start = loader.starts[sample_index]
        lengths = loader.byte_lengths[document_index]
        total += sum(@view(lengths[(start + 1):(start + loader.seq_len)]))
    end
    return total
end

function _character_byte_lengths(text::String)
    return [ncodeunits(string(character)) for character in text]
end

function _encode_document(tokenizer::AbstractTokenizer, document::TextDocument)
    normalized_text = normalize_text(document.text, _normalization_mode(tokenizer))
    token_ids = encode(tokenizer, normalized_text)
    byte_lengths = if tokenizer isa Tokenizer
        lengths = _character_byte_lengths(normalized_text)
        length(lengths) == length(token_ids) || throw(DimensionMismatch(
            "character byte lengths do not match encoded ids",
        ))
        lengths
    else
        [token_byte_length(tokenizer, id) for id in token_ids]
    end
    sum(byte_lengths) == ncodeunits(normalized_text) || throw(ArgumentError(
        "encoded token byte lengths do not reconstruct document $(repr(document.id))",
    ))
    return (;
        id=document.id,
        token_ids,
        byte_lengths,
        raw_sha256=document.raw_sha256,
        normalized_sha256=_sha256_hex(normalized_text),
        raw_bytes=ncodeunits(document.text),
        normalized_bytes=ncodeunits(normalized_text),
        unicode_scalars=length(normalized_text),
    )
end

function _fit_pipeline_tokenizer(
    tokenizer_type::Symbol,
    train_documents::Vector{TextDocument};
    normalization::Symbol,
    add_unk::Bool,
    unk_token::Char,
    vocab_size::Int,
    min_frequency::Int,
    special_tokens,
)
    normalized_texts = [normalize_text(document.text, normalization) for document in train_documents]
    if tokenizer_type === :character
        normalization === :none || throw(ArgumentError(
            "legacy character tokenizer requires normalization=:none",
        ))
        return fit_tokenizer(join(normalized_texts); add_unk, unk_token)
    elseif tokenizer_type === :byte
        return ByteTokenizer(; normalization, special_tokens)
    elseif tokenizer_type === :byte_bpe
        return fit_byte_bpe(
            normalized_texts;
            normalization,
            vocab_size,
            min_frequency,
            special_tokens,
        )
    end
    throw(ArgumentError("`tokenizer_type` must be :character, :byte, or :byte_bpe"))
end

function _records_loader(
    records;
    seq_len::Int,
    batch_size::Int,
    stride::Int,
    drop_last::Bool,
)
    return DocumentDatasetLoader(
        [record.token_ids for record in records];
        byte_lengths=[record.byte_lengths for record in records],
        document_ids=[record.id for record in records],
        seq_len,
        batch_size,
        stride,
        drop_last,
    )
end

"""
    build_document_dataset(documents; kwargs...)

Split documents first, fit the tokenizer on train documents only, encode each split
independently, and construct boundary-safe loaders.
"""
function build_document_dataset(
    documents;
    tokenizer_type::Symbol=:byte_bpe,
    tokenizer=nothing,
    normalization::Symbol=:none,
    special_tokens=DEFAULT_BYTE_SPECIAL_TOKENS,
    add_unk::Bool=true,
    unk_token::Char='�',
    vocab_size::Int=512,
    min_frequency::Int=2,
    validation_fraction::Real=0.1,
    validation_size=nothing,
    split_seed::Integer=0,
    seq_len::Int,
    batch_size::Int=1,
    stride::Int=seq_len,
    drop_last::Bool=true,
)
    collected = _validate_documents(TextDocument[documents...])
    split = split_documents(
        collected;
        validation_fraction,
        validation_size,
        seed=split_seed,
    )
    resolved_normalization = _validate_normalization(normalization)
    resolved_tokenizer = if tokenizer === nothing
        _fit_pipeline_tokenizer(
            tokenizer_type,
            split.train;
            normalization=resolved_normalization,
            add_unk,
            unk_token,
            vocab_size,
            min_frequency,
            special_tokens,
        )
    else
        tokenizer isa AbstractTokenizer || throw(ArgumentError(
            "`tokenizer` must implement AbstractTokenizer",
        ))
        _normalization_mode(tokenizer) == resolved_normalization || throw(ArgumentError(
            "provided tokenizer normalization does not match pipeline normalization",
        ))
        tokenizer
    end

    train_records = [_encode_document(resolved_tokenizer, document) for document in split.train]
    validation_records = [
        _encode_document(resolved_tokenizer, document) for document in split.validation
    ]
    train_loader = _records_loader(
        train_records;
        seq_len,
        batch_size,
        stride,
        drop_last,
    )
    validation_loader = _records_loader(
        validation_records;
        seq_len,
        batch_size,
        stride,
        drop_last,
    )

    return (;
        tokenizer=resolved_tokenizer,
        train=train_loader,
        validation=validation_loader,
        split,
        encoded=(; train=train_records, validation=validation_records),
        normalization=resolved_normalization,
        boundary_policy=:separate_documents,
        loader_config=(; seq_len, batch_size, stride, drop_last),
    )
end

function _encoded_split_checksum(path::AbstractString)
    return _sha256_hex(read(path))
end

function _write_encoded_split(
    path::AbstractString,
    split_name::Symbol,
    records,
    tokenizer_fingerprint_value::String,
)
    temporary_path = tempname(dirname(abspath(path)))
    try
        open(temporary_path, "w") do io
            println(io, ENCODED_SPLIT_FORMAT, '\t', split_name, '\t', tokenizer_fingerprint_value)
            for record in sort(collect(records); by=record -> record.id)
                token_text = join(record.token_ids, ',')
                byte_length_text = join(record.byte_lengths, ',')
                println(
                    io,
                    bytes2hex(codeunits(record.id)), '\t',
                    record.raw_sha256, '\t',
                    record.normalized_sha256, '\t',
                    record.raw_bytes, '\t',
                    record.normalized_bytes, '\t',
                    record.unicode_scalars, '\t',
                    length(record.token_ids), '\t',
                    token_text, '\t',
                    byte_length_text,
                )
            end
        end
        mv(temporary_path, path; force=true)
    finally
        isfile(temporary_path) && rm(temporary_path; force=true)
    end
    return _encoded_split_checksum(path)
end

function _parse_int_list(value::AbstractString)
    isempty(value) && return Int[]
    return parse.(Int, split(value, ','))
end

function _read_encoded_split(
    path::AbstractString,
    expected_split::Symbol,
    expected_tokenizer_fingerprint::String,
)
    isfile(path) || throw(ArgumentError("encoded split does not exist: $path"))
    lines = readlines(path)
    isempty(lines) && throw(ArgumentError("encoded split is empty: $path"))
    header = split(first(lines), '\t')
    length(header) == 3 || throw(ArgumentError("invalid encoded split header"))
    header[1] == ENCODED_SPLIT_FORMAT || throw(ArgumentError(
        "unsupported encoded split format $(repr(header[1]))",
    ))
    Symbol(header[2]) == expected_split || throw(ArgumentError(
        "encoded split name mismatch",
    ))
    header[3] == expected_tokenizer_fingerprint || throw(ArgumentError(
        "encoded split tokenizer fingerprint mismatch",
    ))

    records = NamedTuple[]
    for line in lines[2:end]
        fields = split(line, '\t'; keepempty=true)
        length(fields) == 9 || throw(ArgumentError("invalid encoded split record"))
        token_ids = _parse_int_list(fields[8])
        byte_lengths = _parse_int_list(fields[9])
        length(token_ids) == parse(Int, fields[7]) || throw(ArgumentError(
            "encoded token count does not match payload",
        ))
        length(byte_lengths) == length(token_ids) || throw(ArgumentError(
            "encoded byte lengths do not match token ids",
        ))
        push!(
            records,
            (;
                id=String(hex2bytes(fields[1])),
                raw_sha256=fields[2],
                normalized_sha256=fields[3],
                raw_bytes=parse(Int, fields[4]),
                normalized_bytes=parse(Int, fields[5]),
                unicode_scalars=parse(Int, fields[6]),
                token_ids,
                byte_lengths,
            ),
        )
    end
    isempty(records) && throw(ArgumentError("encoded split contains no documents"))
    return records
end

function _manifest_documents(dataset)
    split_by_id = Dict(
        document.id => "train" for document in dataset.split.train
    )
    for document in dataset.split.validation
        split_by_id[document.id] = "validation"
    end
    records_by_id = Dict(
        record.id => record for record in vcat(
            collect(dataset.encoded.train),
            collect(dataset.encoded.validation),
        )
    )
    documents = vcat(dataset.split.train, dataset.split.validation)
    return [
        Dict(
            "id" => document.id,
            "split" => split_by_id[document.id],
            "source_id" => document.source_id,
            "source_location" => document.source_location,
            "license" => document.license,
            "raw_sha256" => document.raw_sha256,
            "normalized_sha256" => records_by_id[document.id].normalized_sha256,
            "raw_bytes" => records_by_id[document.id].raw_bytes,
            "normalized_bytes" => records_by_id[document.id].normalized_bytes,
            "unicode_scalars" => records_by_id[document.id].unicode_scalars,
            "tokens" => length(records_by_id[document.id].token_ids),
        ) for document in sort(documents; by=document -> document.id)
    ]
end

function _manifest_fingerprint(manifest)
    output = IOBuffer()
    for key in (
        "schema_version",
        "name",
        "version",
        "normalization",
        "boundary_policy",
        "split_method",
        "split_seed",
        "split_fingerprint",
        "tokenizer_fingerprint",
        "tokenizer_checksum",
        "train_checksum",
        "validation_checksum",
    )
        println(output, key, '=', manifest[key])
    end
    for document in sort(manifest["documents"]; by=document -> document["id"])
        for key in (
            "id",
            "split",
            "source_id",
            "source_location",
            "license",
            "raw_sha256",
            "normalized_sha256",
            "raw_bytes",
            "normalized_bytes",
            "unicode_scalars",
            "tokens",
        )
            println(output, "document.", key, '=', document[key])
        end
    end
    return bytes2hex(sha256(take!(output)))
end

function _write_toml_atomic(path::AbstractString, payload)
    temporary_path = tempname(dirname(abspath(path)))
    try
        open(temporary_path, "w") do io
            TOML.print(io, payload; sorted=true)
        end
        mv(temporary_path, path; force=true)
    finally
        isfile(temporary_path) && rm(temporary_path; force=true)
    end
    return path
end

"""Write tokenizer, encoded splits, and a deterministic dataset manifest."""
function save_dataset_artifact(
    directory::AbstractString,
    dataset;
    name::AbstractString,
    version::AbstractString="1",
)
    isempty(name) && throw(ArgumentError("dataset name must not be empty"))
    isempty(version) && throw(ArgumentError("dataset version must not be empty"))
    absolute_directory = abspath(directory)
    mkpath(absolute_directory)
    tokenizer_path = joinpath(absolute_directory, "tokenizer.toml")
    train_path = joinpath(absolute_directory, "train.tokens.tsv")
    validation_path = joinpath(absolute_directory, "validation.tokens.tsv")
    manifest_path = joinpath(absolute_directory, "manifest.toml")

    save_tokenizer(tokenizer_path, dataset.tokenizer)
    tokenizer_fingerprint_value = tokenizer_fingerprint(dataset.tokenizer)
    tokenizer_checksum = _encoded_split_checksum(tokenizer_path)
    train_checksum = _write_encoded_split(
        train_path,
        :train,
        dataset.encoded.train,
        tokenizer_fingerprint_value,
    )
    validation_checksum = _write_encoded_split(
        validation_path,
        :validation,
        dataset.encoded.validation,
        tokenizer_fingerprint_value,
    )

    manifest = Dict{String,Any}(
        "schema_version" => DATASET_ARTIFACT_VERSION,
        "name" => String(name),
        "version" => String(version),
        "normalization" => String(dataset.normalization),
        "boundary_policy" => String(dataset.boundary_policy),
        "split_method" => String(dataset.split.method),
        "split_seed" => dataset.split.seed,
        "split_fingerprint" => dataset.split.fingerprint,
        "tokenizer_file" => basename(tokenizer_path),
        "tokenizer_fingerprint" => tokenizer_fingerprint_value,
        "tokenizer_checksum" => tokenizer_checksum,
        "train_file" => basename(train_path),
        "train_checksum" => train_checksum,
        "validation_file" => basename(validation_path),
        "validation_checksum" => validation_checksum,
        "documents" => _manifest_documents(dataset),
    )
    manifest["fingerprint"] = _manifest_fingerprint(manifest)
    _write_toml_atomic(manifest_path, manifest)
    return (;
        directory=absolute_directory,
        manifest_path,
        tokenizer_path,
        train_path,
        validation_path,
        fingerprint=manifest["fingerprint"],
    )
end

"""Parse and verify a saved dataset artifact and rebuild boundary-safe loaders."""
function load_dataset_artifact(
    directory::AbstractString;
    seq_len::Int,
    batch_size::Int=1,
    stride::Int=seq_len,
    drop_last::Bool=true,
)
    absolute_directory = abspath(directory)
    manifest_path = joinpath(absolute_directory, "manifest.toml")
    isfile(manifest_path) || throw(ArgumentError(
        "dataset manifest does not exist: $manifest_path",
    ))
    manifest = TOML.parsefile(manifest_path)
    get(manifest, "schema_version", 0) == DATASET_ARTIFACT_VERSION || throw(ArgumentError(
        "unsupported dataset artifact schema version",
    ))
    get(manifest, "fingerprint", "") == _manifest_fingerprint(manifest) || throw(ArgumentError(
        "dataset manifest fingerprint mismatch",
    ))

    tokenizer_path = joinpath(absolute_directory, manifest["tokenizer_file"])
    train_path = joinpath(absolute_directory, manifest["train_file"])
    validation_path = joinpath(absolute_directory, manifest["validation_file"])
    _encoded_split_checksum(tokenizer_path) == manifest["tokenizer_checksum"] || throw(ArgumentError(
        "tokenizer artifact checksum mismatch",
    ))
    _encoded_split_checksum(train_path) == manifest["train_checksum"] || throw(ArgumentError(
        "train artifact checksum mismatch",
    ))
    _encoded_split_checksum(validation_path) == manifest["validation_checksum"] || throw(ArgumentError(
        "validation artifact checksum mismatch",
    ))

    tokenizer = load_tokenizer(tokenizer_path)
    tokenizer_fingerprint(tokenizer) == manifest["tokenizer_fingerprint"] || throw(ArgumentError(
        "tokenizer fingerprint does not match dataset manifest",
    ))
    train_records = _read_encoded_split(
        train_path,
        :train,
        manifest["tokenizer_fingerprint"],
    )
    validation_records = _read_encoded_split(
        validation_path,
        :validation,
        manifest["tokenizer_fingerprint"],
    )
    train_ids = Set(record.id for record in train_records)
    validation_ids = Set(record.id for record in validation_records)
    isempty(intersect(train_ids, validation_ids)) || throw(ArgumentError(
        "train and validation artifacts share document ids",
    ))

    train_loader = _records_loader(
        train_records;
        seq_len,
        batch_size,
        stride,
        drop_last,
    )
    validation_loader = _records_loader(
        validation_records;
        seq_len,
        batch_size,
        stride,
        drop_last,
    )
    return (;
        tokenizer,
        train=train_loader,
        validation=validation_loader,
        encoded=(; train=train_records, validation=validation_records),
        manifest,
        fingerprint=manifest["fingerprint"],
    )
end
