using LifeAI: Int4GroupWeight, Int8ChannelWeight

function quantized_parameters_equal(left, right)
    typeof(left) === typeof(right) || return false
    if left isa Int8ChannelWeight
        return left.q == right.q && left.scale == right.scale
    elseif left isa Int4GroupWeight
        return left.packed == right.packed &&
            left.scale == right.scale &&
            left.group == right.group &&
            left.in_dim == right.in_dim
    elseif left isa AbstractArray
        return left == right
    elseif left isa NamedTuple
        keys(left) == keys(right) || return false
        return all(
            quantized_parameters_equal(getfield(left, key), getfield(right, key))
            for key in keys(left)
        )
    elseif left isa Tuple
        return length(left) == length(right) &&
            all(quantized_parameters_equal.(left, right))
    end
    return left == right
end
