# Ripple's published forward rounds theta^exponent in Float32 BEFORE taking
# the reciprocal. theta^(-exponent) changes inverse-frequency bits and those
# small differences become visible at long absolute token positions.
function _ripple_rope_frequencies(dim::Integer,theta::Real)
    !(dim isa Bool) && dim>0 && iseven(dim) || throw(ArgumentError("Ripple RoPE head dimension must be positive and even"))
    !(theta isa Bool) || throw(ArgumentError("Ripple RoPE theta cannot be Bool"))
    base=Float32(theta)
    isfinite(base) && base>0f0 || throw(ArgumentError("Ripple RoPE theta must be finite and positive"))
    exponents=Float32.(0:2:Int(dim)-2)./Float32(dim)
    return 1f0 ./ (base .^ exponents)
end

function _ripple_rope_positions(input,positions)
    ndims(input)==3 || throw(DimensionMismatch("Ripple RoPE input must be (head_dim,heads,tokens)"))
    size(input,1)>0 && iseven(size(input,1)) || throw(ArgumentError("Ripple RoPE head dimension must be positive and even"))
    length(positions)==size(input,3) || throw(DimensionMismatch("Ripple RoPE position count differs from token count"))
    eltype(positions)<:Bool && throw(ArgumentError("Ripple RoPE positions cannot be Bool"))
    if !isempty(positions)
        0<=first(positions)<=last(positions)<=typemax(Int32) || throw(ArgumentError("Ripple RoPE positions exceed supported nonnegative Int32 indexing"))
    end
    return nothing
end

"""Ripple-specific Float32 RoPE with the published reciprocal/power order.

Input has shape `(head_dim, heads, tokens)` and zero-based contiguous positions.
CUDA specializes this method with a separate bounded per-device table cache.
"""
function _ripple_rope(input,positions::AbstractUnitRange{<:Integer},theta)
    _ripple_rope_positions(input,positions)
    isempty(positions) && return copy(input)
    dim=size(input,1);half=dim÷2
    frequencies=_ripple_rope_frequencies(dim,theta)
    angles=reshape(frequencies,half,1).*reshape(Float32.(positions),1,:)
    cosine=reshape(_flint_like(input,cos.(angles)),half,1,:)
    sine=reshape(_flint_like(input,sin.(angles)),half,1,:)
    a,b=view(input,1:half,:,:),view(input,half+1:dim,:,:)
    return cat(a.*cosine.-b.*sine,b.*cosine.+a.*sine;dims=1)
end
