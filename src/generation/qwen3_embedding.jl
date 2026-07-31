using BFloat16s: BFloat16
using LinearAlgebra: norm

const QWEN3_EMBEDDING_RETRIEVAL_INSTRUCTION =
    "Given a web search query, retrieve relevant passages that answer the query"

"""
    qwen3_embedding_query(query; instruction=...)

Apply the instruction-aware query format frozen by the official
SentenceTransformers configuration. Documents are embedded without this
prefix.
"""
function qwen3_embedding_query(
    query::AbstractString;
    instruction::AbstractString=QWEN3_EMBEDDING_RETRIEVAL_INSTRUCTION,
)
    isempty(instruction) && throw(ArgumentError(
        "embedding query instruction must not be empty",
    ))
    isempty(query) && throw(ArgumentError("embedding query must not be empty"))
    return "Instruct: $(String(instruction))\nQuery:$(String(query))"
end

function _qwen3_embedding_text_list(texts)
    values = texts isa AbstractString ? [String(texts)] : String.(collect(texts))
    isempty(values) && throw(ArgumentError(
        "embedding input must contain at least one text",
    ))
    any(isempty, values) && throw(ArgumentError(
        "embedding texts must not be empty",
    ))
    return values
end

"""
    prepare_qwen3_embedding_inputs(
        tokenizer,
        texts;
        max_length=8192,
        padding_side=:left,
    )

Tokenize, right-truncate the text payload while preserving the official final
`<|endoftext|>`, and pad a batch with the exact imported Qwen3-Embedding
tokenizer. Returned token ids are LifeAI 1-based and shaped `(sequence,
batch)`, matching the native model paths.
"""
function prepare_qwen3_embedding_inputs(
    tokenizer::HFQwen3Tokenizer,
    texts;
    max_length::Integer=8_192,
    padding_side::Symbol=:left,
)
    tokenizer.profile === :embedding || throw(ArgumentError(
        "prepare_qwen3_embedding_inputs requires an embedding-profile tokenizer",
    ))
    padding_side in (:left, :right) || throw(ArgumentError(
        "padding_side must be :left or :right",
    ))
    resolved_max_length = Int(max_length)
    1 <= resolved_max_length <= tokenizer.model_max_length || throw(ArgumentError(
        "max_length must be in 1:$(tokenizer.model_max_length)",
    ))
    text_values = _qwen3_embedding_text_list(texts)
    encoded = [
        encode(tokenizer, text; add_special_tokens=true)
        for text in text_values
    ]
    any(isempty, encoded) && throw(ArgumentError(
        "every embedding text must encode to at least one token",
    ))
    original_lengths = length.(encoded)
    truncated = BitVector(length(ids) > resolved_max_length for ids in encoded)
    ids = [
        length(values) > resolved_max_length ?
            [values[1:(resolved_max_length - 1)]; values[end]] : values
        for values in encoded
    ]
    lengths = length.(ids)
    sequence_length = maximum(lengths)
    pad_id = special_token_id(tokenizer, :pad)
    pad_id === nothing && throw(ArgumentError(
        "embedding tokenizer does not define a padding token",
    ))
    tokens = fill(Int(pad_id), sequence_length, length(ids))
    attention_mask = falses(sequence_length, length(ids))
    for batch in eachindex(ids)
        length_i = lengths[batch]
        target = padding_side === :left ?
            ((sequence_length - length_i + 1):sequence_length) :
            (1:length_i)
        tokens[target, batch] = ids[batch]
        attention_mask[target, batch] .= true
    end
    return (;
        tokens,
        attention_mask,
        lengths,
        original_lengths,
        truncated,
        texts=Tuple(text_values),
        padding_side,
    )
end

function _qwen3_embedding_bool_mask(attention_mask, sequence_length, batch_size)
    size(attention_mask) == (sequence_length, batch_size) || throw(
        DimensionMismatch(
            "attention_mask must have shape ($sequence_length, $batch_size)",
        ),
    )
    all(value -> value isa Bool || value == 0 || value == 1, attention_mask) ||
        throw(ArgumentError("attention_mask values must be Bool or zero/one"))
    mask = Bool.(attention_mask)
    for batch in 1:batch_size
        indices = findall(view(mask, :, batch))
        isempty(indices) && throw(ArgumentError(
            "every attention_mask column must contain at least one valid token",
        ))
        length(indices) == last(indices) - first(indices) + 1 || throw(ArgumentError(
            "valid tokens in each attention_mask column must be contiguous",
        ))
    end
    return mask
end

"""
    qwen3_last_token_pool(
        final_hidden,
        attention_mask;
        dimension=size(final_hidden, 1),
        minimum_dimension=1,
    )

Select the final valid token from `(hidden, sequence, batch)`, truncate the
leading MRL dimensions, then normalize each vector in Float32. Truncation
before normalization is part of the contract.
"""
function qwen3_last_token_pool(
    final_hidden::AbstractArray,
    attention_mask;
    dimension::Integer=size(final_hidden, 1),
    minimum_dimension::Integer=1,
)
    ndims(final_hidden) == 3 || throw(DimensionMismatch(
        "final_hidden must have shape (hidden, sequence, batch)",
    ))
    hidden_size, sequence_length, batch_size = size(final_hidden)
    resolved_minimum = Int(minimum_dimension)
    resolved_dimension = Int(dimension)
    1 <= resolved_minimum <= hidden_size || throw(ArgumentError(
        "minimum_dimension must be in 1:$hidden_size",
    ))
    resolved_minimum <= resolved_dimension <= hidden_size || throw(ArgumentError(
        "dimension must be in $resolved_minimum:$hidden_size",
    ))
    mask = _qwen3_embedding_bool_mask(
        attention_mask,
        sequence_length,
        batch_size,
    )
    embeddings = Matrix{Float32}(undef, resolved_dimension, batch_size)
    for batch in 1:batch_size
        last_token = findlast(view(mask, :, batch))
        pooled = Float32.(Array(final_hidden[1:resolved_dimension, last_token, batch]))
        all(isfinite, pooled) || throw(ArgumentError(
            "pooled embedding contains non-finite values",
        ))
        magnitude = norm(pooled)
        isfinite(magnitude) && magnitude > 0 || throw(ArgumentError(
            "pooled embedding must have a finite non-zero norm",
        ))
        embeddings[:, batch] = pooled ./ magnitude
    end
    return embeddings
end

"""
    hf_qwen3_embedding_forward(
        model,
        parameters,
        tokens,
        attention_mask;
        dimension=model.d_model,
    )

Run native BF16 Qwen3 hidden-state inference without allocating vocabulary
logits or retained KV caches, then apply mask-aware last-token/MRL pooling.
"""
function hf_qwen3_embedding_forward(
    model::GPTModel,
    parameters,
    tokens::AbstractMatrix{<:Integer},
    attention_mask;
    dimension::Integer=model.d_model,
)
    _qwen3_validate_semantics(model)
    eltype(parameters.token_embedding.weight) === BFloat16 || throw(ArgumentError(
        "hf_qwen3_embedding_forward requires a BFloat16 parameter tree",
    ))
    sequence_length, batch_size = size(tokens)
    sequence_length > 0 || throw(ArgumentError(
        "embedding tokens must contain at least one position",
    ))
    sequence_length <= model.max_seq_len || throw(ArgumentError(
        "embedding input exceeds model.max_seq_len",
    ))
    token_matrix = Int.(collect(tokens))
    _validate_generation_ids(token_matrix, model.vocab_size)
    mask = _qwen3_embedding_bool_mask(
        attention_mask,
        sequence_length,
        batch_size,
    )
    rope = first(values(model.blocks.layers)).attn.rope
    to_device = _bf16a_device_mover(parameters.token_embedding.weight)
    cos_table = to_device(BFloat16.(rope.cos_cache))
    sin_table = to_device(BFloat16.(rope.sin_cache))
    attention_bias = all(mask) ?
        _bf16a_causal_mask(sequence_length, sequence_length) :
        _bf16a_causal_padding_mask(mask)
    device_mask = to_device(attention_bias)
    caches = Vector{Any}(undef, model.num_layers)
    fill!(caches, nothing)
    result = _bf16a_forward_pass(
        model,
        parameters,
        token_matrix,
        caches,
        cos_table,
        sin_table,
        device_mask;
        start_pos=1,
        project_logits=false,
        capture_trace=false,
        capture_final_hidden=true,
    )
    minimum_dimension = min(qwen3_embedding_spec().minimum_dimension, model.d_model)
    embeddings = qwen3_last_token_pool(
        result.final_hidden,
        mask;
        dimension,
        minimum_dimension,
    )
    return (;
        embeddings,
        attention_mask=mask,
    )
end

"""
    embed_texts(bundle, texts; dimension=bundle.model.d_model, max_length=...)

End-to-end Qwen3 embedding entry point for a bundle returned by
`load_hf_qwen3_embedding_bundle`.
"""
function embed_texts(
    bundle,
    texts;
    dimension::Integer=bundle.model.d_model,
    max_length::Integer=bundle.model.max_seq_len,
    padding_side::Symbol=:left,
)
    hasproperty(bundle, :tokenizer) && hasproperty(bundle, :model) &&
        hasproperty(bundle, :parameters) || throw(ArgumentError(
            "embedding bundle must contain tokenizer, model, and parameters",
        ))
    resolved_max_length = Int(max_length)
    resolved_max_length <= bundle.model.max_seq_len || throw(ArgumentError(
        "max_length exceeds the loaded embedding model context",
    ))
    inputs = prepare_qwen3_embedding_inputs(
        bundle.tokenizer,
        texts;
        max_length=resolved_max_length,
        padding_side,
    )
    forward = hf_qwen3_embedding_forward(
        bundle.model,
        bundle.parameters,
        inputs.tokens,
        inputs.attention_mask;
        dimension,
    )
    return merge(inputs, (; embeddings=forward.embeddings))
end

function _qwen3_normalized_columns(values::AbstractMatrix)
    matrix = Float32.(Array(values))
    all(isfinite, matrix) || throw(ArgumentError(
        "embedding matrix contains non-finite values",
    ))
    for column in axes(matrix, 2)
        magnitude = norm(view(matrix, :, column))
        magnitude > 0 || throw(ArgumentError(
            "embedding matrix contains a zero-norm column",
        ))
        matrix[:, column] ./= magnitude
    end
    return matrix
end

"""Return the cosine-similarity matrix for column-oriented embeddings."""
function qwen3_embedding_similarity(
    queries::AbstractMatrix,
    documents::AbstractMatrix,
)
    size(queries, 1) == size(documents, 1) || throw(DimensionMismatch(
        "query and document embeddings must have the same dimension",
    ))
    query_values = _qwen3_normalized_columns(queries)
    document_values = _qwen3_normalized_columns(documents)
    return transpose(query_values) * document_values
end

struct Qwen3SemanticMemory
    texts::Vector{String}
    embeddings::Matrix{Float32}
    metadata::Vector{Any}

    function Qwen3SemanticMemory(
        texts::Vector{String},
        embeddings::Matrix{Float32},
        metadata::Vector{Any},
    )
        isempty(texts) && throw(ArgumentError(
            "semantic memory must contain at least one document",
        ))
        any(isempty, texts) && throw(ArgumentError(
            "semantic memory documents must not be empty",
        ))
        size(embeddings, 2) == length(texts) || throw(DimensionMismatch(
            "embedding column count must match document count",
        ))
        length(metadata) == length(texts) || throw(DimensionMismatch(
            "metadata length must match document count",
        ))
        normalized = _qwen3_normalized_columns(embeddings)
        return new(texts, normalized, metadata)
    end
end

function Qwen3SemanticMemory(
    texts,
    embeddings::AbstractMatrix,
    metadata=nothing,
)
    text_values = _qwen3_embedding_text_list(texts)
    metadata_values = metadata === nothing ?
        Any[nothing for _ in text_values] : Any[collect(metadata)...]
    return Qwen3SemanticMemory(
        text_values,
        Float32.(Array(embeddings)),
        metadata_values,
    )
end

"""
    build_qwen3_semantic_memory(bundle, documents; metadata=nothing, kwargs...)

Create the Week 22 exact-search baseline. This intentionally keeps all
embeddings in memory and performs dense cosine search; persistence and ANN
indexes are outside the Week 22 scope.
"""
function build_qwen3_semantic_memory(
    bundle,
    documents;
    metadata=nothing,
    dimension::Integer=bundle.model.d_model,
    max_length::Integer=bundle.model.max_seq_len,
)
    texts = _qwen3_embedding_text_list(documents)
    result = embed_texts(
        bundle,
        texts;
        dimension,
        max_length,
        padding_side=:left,
    )
    return Qwen3SemanticMemory(
        texts,
        result.embeddings,
        metadata,
    )
end

function retrieve_qwen3_semantic_memory(
    memory::Qwen3SemanticMemory,
    query_embedding::AbstractVector;
    top_k::Integer=5,
)
    query = reshape(Float32.(collect(query_embedding)), :, 1)
    size(query, 1) == size(memory.embeddings, 1) || throw(DimensionMismatch(
        "query embedding dimension does not match semantic memory",
    ))
    resolved_top_k = Int(top_k)
    1 <= resolved_top_k <= length(memory.texts) || throw(ArgumentError(
        "top_k must be in 1:$(length(memory.texts))",
    ))
    scores = vec(qwen3_embedding_similarity(query, memory.embeddings))
    order = sortperm(
        eachindex(scores);
        by=index -> (-scores[index], index),
    )
    return [
        (;
            rank,
            index,
            text=memory.texts[index],
            score=scores[index],
            metadata=memory.metadata[index],
        )
        for (rank, index) in enumerate(order[1:resolved_top_k])
    ]
end

function search_qwen3_semantic_memory(
    bundle,
    memory::Qwen3SemanticMemory,
    query::AbstractString;
    instruction::AbstractString=QWEN3_EMBEDDING_RETRIEVAL_INSTRUCTION,
    top_k::Integer=5,
    max_length::Integer=bundle.model.max_seq_len,
)
    formatted = qwen3_embedding_query(query; instruction)
    embedded = embed_texts(
        bundle,
        [formatted];
        dimension=size(memory.embeddings, 1),
        max_length,
        padding_side=:left,
    )
    return retrieve_qwen3_semantic_memory(
        memory,
        vec(embedded.embeddings);
        top_k,
    )
end
