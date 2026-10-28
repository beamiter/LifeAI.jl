# Incremental CUDA KV storage. Full-attention layers grow geometrically;
# sliding layers duplicate a circular window so attention always sees an
# ordered, contiguous logical view without concatenating the whole prefix.
struct _FlintCUDAKVStorage{K,V}
    key::K
    value::V
    capacity::Int
    window::Int
end

function _flint_kv_write_kernel!(destination_key, destination_value, key, value,
    first_slot, capacity, duplicate)
    index = (CUDA.blockIdx().x-1) * CUDA.blockDim().x + CUDA.threadIdx().x
    stride = CUDA.blockDim().x * CUDA.gridDim().x
    head_elements = size(key,1) * size(key,2)
    while index <= length(key)
        token = (index-1) ÷ head_elements
        element = (index-1) % head_elements
        slot = (first_slot + token - 1) % capacity + 1
        destination = element + head_elements*(slot-1) + 1
        @inbounds begin
            destination_key[destination] = key[index]
            destination_value[destination] = value[index]
            if duplicate
                destination_key[destination + head_elements*capacity] = key[index]
                destination_value[destination + head_elements*capacity] = value[index]
            end
        end
        index += stride
    end
    return
end

function _flint_kv_write!(storage::_FlintCUDAKVStorage, key, value, first_slot)
    isempty(key) && return storage
    CUDA.@cuda threads=256 blocks=min(cld(length(key),256),65535) _flint_kv_write_kernel!(
        storage.key, storage.value, key, value, first_slot, storage.capacity, storage.window > 0)
    return storage
end

function _flint_kv_storage(key, value, past, window, needed)
    existing = past !== nothing && hasproperty(past, :storage) ? past.storage : nothing
    if existing isa _FlintCUDAKVStorage && existing.window == window &&
            existing.capacity >= needed && size(existing.key,1) == size(key,1) &&
            size(existing.key,2) == size(key,2)
        return existing, past.end_slot
    end
    capacity = window > 0 ? window : max(16,nextpow(2,needed))
    slots = window > 0 ? 2capacity : capacity
    storage = _FlintCUDAKVStorage(
        CUDA.CuArray{Float32}(undef,size(key,1),size(key,2),slots),
        CUDA.CuArray{Float32}(undef,size(value,1),size(value,2),slots),capacity,window)
    old_tokens = past === nothing ? 0 : size(past.key,3)
    if old_tokens > 0
        _flint_kv_write!(storage,past.key,past.value,1)
    end
    return storage, old_tokens
end

function LifeAI._flint_kv_append(key::CUDA.CuArray{Float32,3}, value::CUDA.CuArray{Float32,3},
    past, position, window::Int)
    # Prefill can contain multiple queries and require more than one sliding
    # window temporarily. Keep the established general path for those calls.
    size(key,3) == 1 || return LifeAI._flint_kv_append_reference(key,value,past,position,window)
    size(key) == size(value) || throw(DimensionMismatch("Flint key/value shapes differ"))
    window >= 0 || throw(ArgumentError("Flint KV window must be nonnegative"))
    old_tokens = past === nothing ? 0 : size(past.key,3)
    # A caller with a longer external history still needs its full causal mask.
    # The normal sliding cache already retains exactly window-1 old tokens.
    window > 0 && old_tokens >= window &&
        return LifeAI._flint_kv_append_reference(key,value,past,position,window)
    needed = old_tokens+1
    storage, end_slot = _flint_kv_storage(key,value,past,window,needed)
    slot = window > 0 ? mod(end_slot,window)+1 : needed
    _flint_kv_write!(storage,key,value,slot)
    first_slot = window > 0 ? mod(slot-needed,window)+1 : 1
    attention_slots = first_slot:first_slot+needed-1
    attention_key = view(storage.key,:,:,attention_slots)
    attention_value = view(storage.value,:,:,attention_slots)
    first_position = past === nothing ? position : past.first_position
    keep = window > 0 ? min(needed,window-1) : needed
    cache_slots = first_slot+needed-keep:first_slot+needed-1
    updated = (key=view(storage.key,:,:,cache_slots),value=view(storage.value,:,:,cache_slots),
        first_position=position+1-keep,storage=storage,end_slot=slot)
    return (key=attention_key,value=attention_value,first_position=first_position,updated=updated)
end
