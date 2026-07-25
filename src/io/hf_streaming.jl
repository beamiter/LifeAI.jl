using Random: Xoshiro

# Week 13: streamed safetensors loading. The reader indexes tensor locations
# from file headers only; tensor data is read from disk on demand so models
# whose Float32 parameters exceed host RAM can still be verified layer by
# layer with the exact numeric semantics of the in-memory path.

struct _SafetensorsTensorLocation
    path::String
    data_base::Int
    dtype::String
    shape::Vector{Int}
    data_start::Int
    data_stop::Int
end

"""
    HFSafetensorsReader

Header-only index over one `.safetensors` file, a
`model.safetensors.index.json`, or a model directory containing either form.
Tensor bytes stay on disk until `read_safetensors_tensor` is called.
"""
struct HFSafetensorsReader
    source::String
    locations::Dict{String,_SafetensorsTensorLocation}
end

Base.haskey(reader::HFSafetensorsReader, name::AbstractString) =
    haskey(reader.locations, String(name))
Base.keys(reader::HFSafetensorsReader) = keys(reader.locations)

function _index_safetensors_file!(
    locations::Dict{String,_SafetensorsTensorLocation},
    path::AbstractString,
)
    entries, data_base = _safetensors_entries(path)
    for entry in entries
        haskey(locations, entry.name) && throw(ArgumentError(
            "duplicate tensor `$(entry.name)` across safetensors shards",
        ))
        locations[entry.name] = _SafetensorsTensorLocation(
            abspath(path),
            data_base,
            entry.dtype,
            entry.shape,
            entry.data_start,
            entry.data_stop,
        )
    end
    return locations
end

"""
    open_safetensors_reader(path)

Build a `HFSafetensorsReader` from safetensors headers without reading tensor
data. Applies the same strict shard/index validation as `load_safetensors`.
"""
function open_safetensors_reader(path::AbstractString)
    resolved = if isdir(path)
        single = joinpath(path, "model.safetensors")
        index = joinpath(path, "model.safetensors.index.json")
        if isfile(single)
            single
        elseif isfile(index)
            index
        else
            throw(ArgumentError(
                "model directory contains neither model.safetensors nor model.safetensors.index.json: $path",
            ))
        end
    else
        path
    end

    locations = Dict{String,_SafetensorsTensorLocation}()
    if endswith(resolved, ".index.json")
        index = _json_object(resolved)
        weight_map_raw = _json_required(index, "weight_map", resolved)
        weight_map_raw isa JSON3.Object || throw(ArgumentError(
            "safetensors index `weight_map` must be an object: $resolved",
        ))
        weight_map = Dict{String,String}()
        for raw_name in keys(weight_map_raw)
            name = String(raw_name)
            haskey(weight_map, name) && throw(ArgumentError(
                "duplicate tensor `$name` in safetensors index",
            ))
            shard_raw = weight_map_raw[raw_name]
            shard_raw isa AbstractString || throw(ArgumentError(
                "safetensors shard name for `$name` must be a string",
            ))
            weight_map[name] = String(shard_raw)
        end
        isempty(weight_map) && throw(ArgumentError(
            "safetensors index weight_map is empty",
        ))

        root = dirname(abspath(resolved))
        shard_of = Dict{String,String}()
        for shard in sort!(unique!(collect(values(weight_map))))
            shard_path = _safe_shard_path(root, shard)
            isfile(shard_path) || throw(ArgumentError(
                "safetensors shard does not exist: $shard_path",
            ))
            before = Set(keys(locations))
            _index_safetensors_file!(locations, shard_path)
            for name in setdiff(Set(keys(locations)), before)
                shard_of[name] = shard
            end
        end
        for (name, shard) in shard_of
            get(weight_map, name, nothing) == shard || throw(ArgumentError(
                "tensor `$name` is stored in `$shard` but the index assigns a different shard",
            ))
        end
        Set(keys(locations)) == Set(keys(weight_map)) || throw(ArgumentError(
            "safetensors index contains missing or unindexed tensors",
        ))
    else
        isfile(resolved) || throw(ArgumentError(
            "safetensors file does not exist: $resolved",
        ))
        _index_safetensors_file!(locations, resolved)
    end
    return HFSafetensorsReader(abspath(resolved), locations)
end

function _reader_location(reader::HFSafetensorsReader, name::AbstractString)
    location = get(reader.locations, String(name), nothing)
    location === nothing && throw(ArgumentError(
        "missing HuggingFace tensor `$name`",
    ))
    return location
end

"""
    read_safetensors_tensor(reader, name)

Read one tensor from disk and decode it to Float32 with its semantic shape.
"""
function read_safetensors_tensor(reader::HFSafetensorsReader, name::AbstractString)
    location = _reader_location(reader, name)
    raw = open(location.path, "r") do io
        seek(io, location.data_base + location.data_start)
        bytes = read(io, location.data_stop - location.data_start)
        length(bytes) == location.data_stop - location.data_start ||
            throw(ArgumentError("truncated data for safetensors tensor `$name`"))
        bytes
    end
    return _decode_safetensors_values(raw, location.dtype, location.shape)
end

# Minimal AbstractDict facade so `_expect_tensor` and the shared block mapping
# work identically on the streamed reader and the in-memory state dict.
struct _StreamedTensors <: AbstractDict{String,Any}
    reader::HFSafetensorsReader
end

Base.haskey(tensors::_StreamedTensors, name::AbstractString) =
    haskey(tensors.reader, name)
Base.getindex(tensors::_StreamedTensors, name::AbstractString) =
    read_safetensors_tensor(tensors.reader, name)
Base.length(tensors::_StreamedTensors) = length(tensors.reader.locations)
Base.keys(tensors::_StreamedTensors) = keys(tensors.reader)
function Base.iterate(tensors::_StreamedTensors, state...)
    step = iterate(keys(tensors.reader.locations), state...)
    step === nothing && return nothing
    name, next_state = step
    return name => tensors[name], next_state
end

"""
    _read_embedding_rows(reader, name, tokens, d_model, vocab_size)

Gather embedding rows for 1-based `tokens` directly from disk. The HF
embedding is stored row-major as `(vocab_size, d_model)`, so each token id
maps to one contiguous row; whole-matrix materialization is never needed.
Matches `ps.weight[:, token]` of the in-memory embedding lookup bit for bit.
"""
function _read_embedding_rows(
    reader::HFSafetensorsReader,
    name::AbstractString,
    tokens::AbstractMatrix{<:Integer},
    d_model::Int,
    vocab_size::Int,
)
    location = _reader_location(reader, name)
    location.shape == [vocab_size, d_model] || throw(DimensionMismatch(
        "HuggingFace tensor `$name` has shape $(Tuple(location.shape)); " *
        "expected $((vocab_size, d_model))",
    ))
    element_bytes = _SAFETENSORS_DTYPES[location.dtype]
    row_bytes = d_model * element_bytes
    seq_len, batch_size = size(tokens)
    x = Array{Float32}(undef, d_model, seq_len, batch_size)
    open(location.path, "r") do io
        for batch in 1:batch_size, position in 1:seq_len
            token = Int(tokens[position, batch])
            1 <= token <= vocab_size || throw(ArgumentError(
                "token id is outside 1:$vocab_size",
            ))
            seek(io, location.data_base + location.data_start + (token - 1) * row_bytes)
            raw = read(io, row_bytes)
            length(raw) == row_bytes || throw(ArgumentError(
                "truncated data for safetensors tensor `$name`",
            ))
            x[:, position, batch] = _decode_safetensors_values(
                raw,
                location.dtype,
                [d_model],
            )
        end
    end
    return x
end

function _streamed_logits(
    model::GPTModel,
    hidden,
    tensors::_StreamedTensors,
    st_lm_head::NamedTuple,
)
    d_model = model.d_model
    ps = if model.tie_embeddings
        embedding_hf = _expect_tensor(
            tensors,
            "model.embed_tokens.weight",
            (model.vocab_size, d_model),
        )
        (;
            token_embedding=(; weight=permutedims(embedding_hf, (2, 1))),
            lm_head=(;),
        )
    else
        (; lm_head=(; weight=_expect_tensor(
            tensors,
            "lm_head.weight",
            (model.vocab_size, d_model),
        )))
    end
    logits, st_new = _project_logits(model, hidden, ps, st_lm_head)
    return logits, st_new
end

"""
    stream_hf_qwen3_forward(
        model_dir,
        tokens;
        decode_token=nothing,
        max_seq_len=64,
        variant=nothing,
    )

Run the Week 07 forward trace — embedding, every block output, final hidden
state and logits — plus an optional dynamic-KV-cache single-token decode,
while streaming each layer's weights from safetensors on demand. Numeric
semantics are identical to loading the full model: the same block kernels run
in the same order on the same Float32 values; only weight residency differs.
Peak memory stays near the largest single tensor instead of the full
parameter tree. `tokens` is a `(seq_len, batch)` matrix of 1-based ids and
`decode_token` a vector of 1-based ids with one entry per batch item.
"""
function stream_hf_qwen3_forward(
    model_dir::AbstractString,
    tokens::AbstractMatrix{<:Integer};
    decode_token=nothing,
    max_seq_len=64,
    variant=nothing,
)
    isdir(model_dir) || throw(ArgumentError("model directory does not exist: $model_dir"))
    config = load_hf_qwen3_config(
        joinpath(model_dir, "config.json");
        max_seq_len,
        variant,
    )
    model = GPTModel(config)
    _qwen3_validate_semantics(model)

    seq_len, batch_size = size(tokens)
    seq_len > 0 || throw(ArgumentError("`tokens` must contain at least one token"))
    decode_matrix = if decode_token === nothing
        nothing
    else
        matrix = _decode_token_matrix(decode_token, batch_size)
        _validate_generation_ids(matrix, model.vocab_size)
        matrix
    end
    seq_len + (decode_matrix === nothing ? 0 : 1) <= model.max_seq_len ||
        throw(ArgumentError("prompt plus decode token exceeds model.max_seq_len"))

    reader = open_safetensors_reader(model_dir)
    _qwen3_validate_tensor_names(model, Set(String.(collect(keys(reader)))))
    tensors = _StreamedTensors(reader)

    if model.tie_embeddings && haskey(reader, "lm_head.weight")
        tied_head = _expect_tensor(
            tensors,
            "lm_head.weight",
            (model.vocab_size, model.d_model),
        )
        embedding_hf = _expect_tensor(
            tensors,
            "model.embed_tokens.weight",
            (model.vocab_size, model.d_model),
        )
        tied_head == embedding_hf || throw(ArgumentError(
            "tied Qwen3 lm_head.weight does not equal model.embed_tokens.weight",
        ))
        tied_head = nothing
        embedding_hf = nothing
        GC.gc(false)
    end

    st = Lux.initialstates(Xoshiro(0), model)
    token_matrix = Int.(collect(tokens))
    _validate_generation_ids(token_matrix, model.vocab_size)

    x = _read_embedding_rows(
        reader,
        "model.embed_tokens.weight",
        token_matrix,
        model.d_model,
        model.vocab_size,
    )
    embedding = x

    blocks = Tuple(values(model.blocks.layers))
    block_states = Tuple(values(st.blocks))
    block_outputs = Vector{Any}(undef, model.num_layers)
    layer_caches = Vector{Any}(undef, model.num_layers)
    for index in 1:model.num_layers
        block_ps = _qwen3_block_parameters(model, tensors, index - 1)
        x, _, layer_caches[index] = _block_with_kv_cache(
            blocks[index],
            x,
            block_ps,
            block_states[index],
            LayerKVCache();
            start_pos=1,
        )
        block_outputs[index] = x
        block_ps = nothing
        GC.gc(false)
    end

    final_norm_ps = (; scale=reshape(_expect_tensor(
        tensors,
        "model.norm.weight",
        (model.d_model,),
    ), model.d_model, 1, 1))
    final_hidden, _ = model.final_norm(x, final_norm_ps, st.final_norm)
    logits, _ = _streamed_logits(model, final_hidden, tensors, st.lm_head)
    GC.gc(false)

    decode_logits = nothing
    if decode_matrix !== nothing
        y = _read_embedding_rows(
            reader,
            "model.embed_tokens.weight",
            decode_matrix,
            model.d_model,
            model.vocab_size,
        )
        start_pos = seq_len + 1
        for index in 1:model.num_layers
            block_ps = _qwen3_block_parameters(model, tensors, index - 1)
            y, _, layer_caches[index] = _block_with_kv_cache(
                blocks[index],
                y,
                block_ps,
                block_states[index],
                layer_caches[index];
                start_pos,
            )
            block_ps = nothing
            GC.gc(false)
        end
        decode_hidden, _ = model.final_norm(y, final_norm_ps, st.final_norm)
        decode_logits, _ = _streamed_logits(
            model,
            decode_hidden,
            tensors,
            st.lm_head,
        )
        GC.gc(false)
    end

    dense_spec = config.qwen3_variant === nothing ?
        nothing : qwen3_dense_spec(config.qwen3_variant)
    return (;
        embedding,
        blocks=Tuple(block_outputs),
        final_hidden,
        logits,
        decode_logits,
        model,
        config,
        variant=dense_spec,
        source=abspath(model_dir),
    )
end
