# Bounded device residency for dense Ripple checkpoints. Transient projections
# are retained only for the current decoder layer and freed before the next.
mutable struct RippleOffloadParameters{F,R} <: AbstractDict{String,Any}
    host::Dict{String,Any}
    resident::Dict{String,Any}
    transient::Dict{String,Any}
    fetch::F
    device::R
    resident_bytes::Int
    budget_bytes::Int
    uploaded_bytes::Int
    uploads::Int
    transient_bytes::Int
    peak_transient_bytes::Int
end

Base.length(p::RippleOffloadParameters) = length(p.host)
Base.keys(p::RippleOffloadParameters) = keys(p.host)
Base.haskey(p::RippleOffloadParameters, key) = haskey(p.host, key)
quantized_parameter_bytes(p::RippleOffloadParameters) = p.resident_bytes + sum(quantized_parameter_bytes, values(p.host))
function Base.iterate(p::RippleOffloadParameters, state...)
    entry = iterate(keys(p.host), state...)
    entry === nothing && return nothing
    key, next = entry
    return key => p[key], next
end

_ripple_release_weight!(weight) = nothing
_ripple_synchronize_weight(weight) = nothing
_ripple_release_weight!(w::Int4GroupWeight) = (_ripple_release_weight!(w.packed); _ripple_release_weight!(w.scale); nothing)
_ripple_release_weight!(w::Int8ChannelWeight) = (_ripple_release_weight!(w.q); _ripple_release_weight!(w.scale); nothing)
_ripple_release_weight!(w::NVFP4Weight) = (_ripple_release_weight!(w.packed); _ripple_release_weight!(w.block_scales); nothing)
_ripple_synchronize_weight(w::Int4GroupWeight) = _ripple_synchronize_weight(w.packed)
_ripple_synchronize_weight(w::Int8ChannelWeight) = _ripple_synchronize_weight(w.q)
_ripple_synchronize_weight(w::NVFP4Weight) = _ripple_synchronize_weight(w.packed)

_ripple_activate!(parameters) = nothing
function _ripple_activate!(p::RippleOffloadParameters)
    if !isempty(p.transient)
        _ripple_synchronize_weight(first(values(p.transient)))
        foreach(_ripple_release_weight!, values(p.transient))
        empty!(p.transient)
        p.transient_bytes = 0
    end
    return p
end

function Base.getindex(p::RippleOffloadParameters, name::AbstractString)
    haskey(p.host, name) || throw(KeyError(name))
    haskey(p.resident, name) && return p.resident[name]
    haskey(p.transient, name) && return p.transient[name]
    value = p.host[name]
    mapped = value === nothing ? p.fetch(String(name)) : _flint_weight_to_device(p.device, value)
    p.transient[String(name)] = mapped
    bytes = quantized_parameter_bytes(mapped)
    p.uploaded_bytes += bytes
    p.uploads += 1
    p.transient_bytes += bytes
    p.peak_transient_bytes = max(p.peak_transient_bytes, p.transient_bytes)
    return mapped
end

"""Residency/transfer counters; excludes activations, CUDA allocator caches and KV storage."""
function ripple_offload_stats(bundle)
    p = bundle.parameters
    p isa RippleOffloadParameters || return (offloaded=false,)
    return (offloaded=true, resident_bytes=p.resident_bytes, budget_bytes=p.budget_bytes,
        host_bytes=sum(quantized_parameter_bytes, values(p.host)),
        resident_tensors=length(p.resident), host_tensors=count(!isnothing, values(p.host)),
        uploaded_bytes=p.uploaded_bytes, uploads=p.uploads,
        transient_bytes=p.transient_bytes, peak_transient_bytes=p.peak_transient_bytes)
end

function _ripple_offload_parameters(host, device, budget; fetch=name -> throw(KeyError(name)))
    budget = _hf_strict_host_int(budget, "Ripple resident_bytes")
    budget >= 0 || throw(ArgumentError("Ripple resident_bytes must be nonnegative"))
    p = RippleOffloadParameters(host, Dict{String,Any}(), Dict{String,Any}(), fetch,
        device, 0, budget, 0, 0, 0, 0)
    # Prefer the repeatedly used embedding/head, then complete numeric layer order.
    order = sort!(collect(keys(host)); by=name -> begin
        layer = match(r"^model\.layers\.(\d+)\.", name)
        layer === nothing ? (-1, name) : (parse(Int, layer[1]), name)
    end)
    for name in order
        weight = host[name]
        weight === nothing && continue
        bytes = quantized_parameter_bytes(weight)
        bytes <= p.budget_bytes-p.resident_bytes || continue
        p.resident[name] = _flint_weight_to_device(device, weight)
        p.resident_bytes += bytes
        host[name] = nothing
    end
    return p
end

"""
    offload_ripple(bundle; device, resident_bytes=0)

Move at most `resident_bytes` of host parameters onto the device and upload
remaining parameters one decoder layer at a time. The host parameter dictionary
is copied; underlying arrays are shared. One bundle must not run concurrently.
"""
function offload_ripple(bundle; device, resident_bytes::Integer=0)
    host = Dict{String,Any}(String(k) => v for (k,v) in pairs(bundle.parameters))
    parameters = _ripple_offload_parameters(host, device, resident_bytes)
    return merge(bundle, (; parameters))
end

"""
    load_hf_ripple_streamed(path; device=identity, dtype=BFloat16)

Run the complete unquantized checkpoint with only the current layer's weights
on the device. Each forward rereads the checkpoint; intended for full-weight
numerical validation when BF16 weights exceed host/device memory capacity.
"""
function load_hf_ripple_streamed(path::AbstractString; device=identity, dtype::Type=BFloat16)
    dtype in (Float32, BFloat16) || throw(ArgumentError("invalid Ripple streamed dtype"))
    model = load_hf_ripple_config(path)
    reader = open_safetensors_reader(path)
    shapes = verify_ripple_checkpoint(reader, model)
    host = Dict{String,Any}(name => nothing for name in keys(shapes))
    fetch = name -> _ripple_read_device_tensor(reader,name,
        length(shapes[name]) == 1 ? Float32 : dtype,device)
    parameters = _ripple_offload_parameters(host, device, 0; fetch)
    return (; model, parameters, source=abspath(path), quantization=:float)
end

_ripple_read_device_tensor(reader,name,dtype,device) = device(read_safetensors_tensor(reader,name;target_dtype=dtype))
