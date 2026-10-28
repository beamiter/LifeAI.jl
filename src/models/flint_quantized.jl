# Weight-only quantization for Flint. Routing, embeddings, normalization, and
# attention sinks retain their floating-point representation. Inference
# reconstructs only the linear weight currently being used.

const FLINT_QUANTIZED_FORMAT_VERSION = 2

_flint_quantizable(name::AbstractString) = name == "lm_head.weight" ||
    occursin(r"\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)\.weight$", name)

function _flint_quantization_spec(scheme, group, calibration)
    resolved_group = group === nothing ? (scheme === :nvfp4 ? 16 : 128) : group
    if scheme === :nvfp4
        _strict_host_int(resolved_group, "NVFP4 group") == 16 || throw(ArgumentError(
            "NVFP4 uses a fixed block size of 16",
        ))
        calibration === :maxabs || throw(ArgumentError(
            "NVFP4 currently supports only :maxabs calibration",
        ))
        return (scheme=:nvfp4, group=16, calibration=:maxabs)
    end
    spec = LinearQuantizationSpec(scheme; group=resolved_group, calibration)
    spec.scheme in (:int4, :int8, :bf16) || throw(ArgumentError("unsupported Flint quantization"))
    spec.calibration === :activation_mse && throw(ArgumentError(
        "Flint supports :maxabs and weight-only :mse calibration; activation calibration is not implemented",
    ))
    return spec
end

_flint_weight_shape(weight::AbstractArray) = size(weight)
_flint_weight_shape(weight::Int8ChannelWeight) = size(weight.q)
_flint_weight_shape(weight::Int4GroupWeight) = (size(weight.packed, 1), weight.in_dim)
_flint_weight_shape(weight::NVFP4Weight) = (size(weight.packed, 1), weight.in_dim)
_flint_weight_shape(weight::AbstractNativeNVFP4Weight) = size(nvfp4_weight(weight))

_flint_weight_to_device(device, weight::AbstractArray) = device(weight)
_flint_weight_to_device(device, weight::Int8ChannelWeight) = Int8ChannelWeight(
    device(weight.q), device(weight.scale), _QuantizedWeightRowSlice(),
)
_flint_weight_to_device(device, weight::Int4GroupWeight) = Int4GroupWeight(
    device(weight.packed), device(weight.scale), weight.group, weight.in_dim,
    _QuantizedWeightRowSlice(),
)
_flint_weight_to_device(device, weight::NVFP4Weight) = NVFP4Weight(
    device(weight.packed), device(weight.block_scales), weight.global_scale, weight.in_dim,
)

function _flint_quantize_tensor(name, weight, spec)
    weight isa AbstractArray{<:AbstractFloat} || throw(ArgumentError(
        "Flint source tensor `$name` must be a floating-point array",
    ))
    all(isfinite, weight) || throw(ArgumentError("Flint tensor `$name` contains nonfinite values"))
    if _flint_quantizable(name)
        weight isa AbstractMatrix || throw(DimensionMismatch("Flint projection `$name` must be a matrix"))
        return spec.scheme === :nvfp4 ? quantize_nvfp4(weight) : _quantize_linear(weight, spec)
    end
    return copy(weight)
end

"""
    quantize_flint_parameters(parameters; scheme=:int4, group=nothing, calibration=:maxabs)

Quantize Flint linear projections, including routed/shared experts and the
output head. Router scores, embeddings, normalization scales, and attention
sinks remain floating point. `:int8` uses symmetric row scales; `:int4` packs
two signed values per byte with input-group scales. This is weight-only
quantization, not integer-only arithmetic. `:mse` optionally calibrates INT4
scales against weight reconstruction error.
`:nvfp4` stores E2M1 values, E4M3FN block scales and one Float32 tensor scale.
Omitted `group` resolves to 16 for NVFP4 and 128 for the other schemes.
NVFP4 permits only fixed block size 16 and `calibration=:maxabs`.
"""
function quantize_flint_parameters(
    parameters::AbstractDict;
    scheme::Symbol=:int4,
    group::Union{Nothing,Integer}=nothing,
    calibration::Symbol=:maxabs,
)
    spec = _flint_quantization_spec(scheme, group, calibration)
    result = Dict{String,Any}(
        String(name) => _flint_quantize_tensor(String(name), weight, spec)
        for (name, weight) in parameters
        if !(String(name) == "lm_head.weight" &&
            get(parameters, "model.embed_tokens.weight", nothing) === weight)
    )
    if haskey(parameters, "lm_head.weight") &&
        get(parameters, "model.embed_tokens.weight", nothing) === parameters["lm_head.weight"]
        result["lm_head.weight"] = result["model.embed_tokens.weight"]
    end
    return result
end

"""
    load_hf_flint_quantized(path; scheme=:int4, group=nothing, calibration=:maxabs,
                            dtype=Float32, device=identity, quantize_on_device=false)

Load and quantize an HF Flint checkpoint one tensor at a time, without keeping
an additional complete unquantized model. `device` is a function such as
`CUDA.cu`; packed values and scales are transferred separately. The returned
bundle has `model`, `parameters`, `source`, and `quantization` fields.
With `quantize_on_device=true`, each floating-point projection is transferred
by `device` before quantization, and its packed result stays on that device.
This avoids CPU quantization and a second transfer of packed data. It requires
`:maxabs` calibration; floating-point reserved tensors retain the usual path.
BF16 source projections keep their exact bits through host loading and are
converted to Float32 by the device quantizer, reducing host conversion and I/O.
"""
function load_hf_flint_quantized(
    path::AbstractString;
    scheme::Symbol=:int4,
    group::Union{Nothing,Integer}=nothing,
    calibration::Symbol=:maxabs,
    dtype::Type=Float32,
    device=identity,
    quantize_on_device::Bool=false,
    on_progress=nothing,
)
    dtype in (Float32, BFloat16) || throw(ArgumentError("Flint dtype must be Float32 or BFloat16"))
    spec = _flint_quantization_spec(scheme, group, calibration)
    quantize_on_device && spec.calibration !== :maxabs && throw(ArgumentError(
        "on-device Flint quantization supports only :maxabs calibration",
    ))
    model = load_hf_flint_config(path)
    reader = open_safetensors_reader(path)
    shapes = flint_expected_tensor_shapes(model)
    for (name, shape) in shapes
        haskey(reader, name) || throw(ArgumentError("missing Flint tensor `$name`"))
        Tuple(reader.locations[name].shape) == shape || throw(DimensionMismatch(
            "Flint tensor `$name` has shape $(reader.locations[name].shape), expected $shape",
        ))
        if _flint_quantizable(name) && spec.scheme === :int4
            shape[2] % spec.group == 0 || throw(ArgumentError(
                "Flint tensor `$name` input dimension is not divisible by group $(spec.group)",
            ))
        end
    end
    parameters = Dict{String,Any}()
    for (index, name) in enumerate(sort!(collect(keys(shapes))))
        target_dtype = endswith(name, ".mlp.gate.weight") || endswith(name, ".softmax_offset") ? Float32 : dtype
        if quantize_on_device && _flint_quantizable(name) && reader.locations[name].dtype == "BF16"
            target_dtype = BFloat16
        end
        weight = read_safetensors_tensor(reader, name; target_dtype)
        parameters[name] = if quantize_on_device && _flint_quantizable(name)
            # Quantization preserves the placement of its input. Do not apply
            # device again to the packed result (which would duplicate copies).
            _flint_quantize_tensor(name, device(weight), spec)
        else
            _flint_weight_to_device(device, _flint_quantize_tensor(name, weight, spec))
        end
        on_progress === nothing || on_progress(name, index, length(shapes))
    end
    return (
        model=model, parameters=parameters, source=abspath(path),
        quantization=(scheme=spec.scheme, group=spec.group, calibration=spec.calibration),
    )
end

function _flint_write_array(directory, filename, array)
    host = Array(array)
    eltype(host) in (Float32, BFloat16, Int8, UInt8) || throw(ArgumentError(
        "unsupported Flint artifact dtype $(eltype(host))",
    ))
    path = joinpath(directory, filename)
    open(path, "w") do io
        write(io, host)
    end
    checksum = open(sha256, path)
    return Dict{String,Any}(
        "file" => filename,
        "dtype" => string(eltype(host)),
        "shape" => collect(size(host)),
        "bytes" => sizeof(host),
        "sha256" => bytes2hex(checksum),
    )
end

function _flint_write_weight(directory, index, name, weight)
    weight isa AbstractNativeNVFP4Weight && (weight = nvfp4_weight(weight))
    stem = joinpath("tensors", lpad(string(index), 6, '0'))
    entry = Dict{String,Any}("name" => name, "shape" => collect(_flint_weight_shape(weight)))
    if weight isa Int8ChannelWeight
        entry["kind"] = "int8"
        entry["values"] = _flint_write_array(directory, stem * ".q.bin", weight.q)
        entry["scale"] = _flint_write_array(directory, stem * ".scale.bin", weight.scale)
    elseif weight isa Int4GroupWeight
        entry["kind"] = "int4"
        entry["group"] = weight.group
        entry["values"] = _flint_write_array(directory, stem * ".q.bin", weight.packed)
        entry["scale"] = _flint_write_array(directory, stem * ".scale.bin", weight.scale)
    elseif weight isa NVFP4Weight
        entry["kind"] = "nvfp4"
        entry["encoding"] = "e2m1-e4m3fn-block16"
        entry["group"] = 16
        entry["values"] = _flint_write_array(directory, stem * ".q.bin", weight.packed)
        entry["scale"] = _flint_write_array(directory, stem * ".scale.bin", weight.block_scales)
        entry["global_scale"] = _flint_write_array(directory, stem * ".global_scale.bin", Float32[weight.global_scale])
    elseif weight isa AbstractArray
        entry["kind"] = "float"
        entry["values"] = _flint_write_array(directory, stem * ".bin", weight)
    else
        throw(ArgumentError("unsupported Flint artifact tensor `$name`"))
    end
    return entry
end

"""
    save_flint_quantized(path, bundle)

Save a Flint bundle as a portable, versioned directory of raw tensors plus a
JSON manifest. Every payload and config has a SHA-256 checksum. The destination
must not exist; an incomplete save never replaces an existing artifact. Payloads
are little-endian, matching supported Julia/HF hosts. Tokenizer assets are
copied when `bundle.source` points to a local HF checkpoint.
"""
function save_flint_quantized(path::AbstractString, bundle)
    Base.ENDIAN_BOM == 0x04030201 || throw(ArgumentError("Flint artifacts currently require a little-endian host"))
    target = abspath(path)
    ispath(target) && throw(ArgumentError("Flint artifact destination already exists: $target"))
    shapes = flint_expected_tensor_shapes(bundle.model)
    actual_names = Set(String.(keys(bundle.parameters)))
    if bundle.model.tie_word_embeddings && "lm_head.weight" in actual_names
        bundle.parameters["lm_head.weight"] === bundle.parameters["model.embed_tokens.weight"] || throw(ArgumentError(
            "a tied Flint output head must alias the input embedding",
        ))
        delete!(actual_names, "lm_head.weight")
    end
    actual_names == Set(keys(shapes)) || throw(ArgumentError(
        "Flint artifact parameters must exactly match the model tensor set",
    ))
    for (name, shape) in shapes
        _flint_weight_shape(bundle.parameters[name]) == shape || throw(DimensionMismatch(
            "Flint tensor `$name` does not match model shape $shape",
        ))
    end
    mkpath(dirname(target))
    temporary = mktempdir(dirname(target); prefix=".flint-save-")
    try
        mkpath(joinpath(temporary, "tensors"))
        config_path = joinpath(temporary, "config.json")
        open(config_path, "w") do io
            JSON3.write(io, flint_config_dict(bundle.model))
        end
        entries = Any[]
        for (index, name) in enumerate(sort!(collect(keys(shapes))))
            push!(entries, _flint_write_weight(temporary, index, name, bundle.parameters[name]))
        end
        tokenizer_assets = Any[]
        if hasproperty(bundle, :source) && isdir(bundle.source)
            for filename in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "generation_config.json", "chat_template.jinja")
                source = joinpath(bundle.source, filename)
                isfile(source) || continue
                cp(source, joinpath(temporary, filename))
                push!(tokenizer_assets, Dict("file" => filename, "sha256" => bytes2hex(open(sha256, source))))
            end
        end
        manifest = Dict{String,Any}(
            "format" => "lifeai-flint-quantized", "version" => FLINT_QUANTIZED_FORMAT_VERSION,
            "endianness" => "little", "config_sha256" => bytes2hex(open(sha256, config_path)),
            "tensors" => entries, "tokenizer_assets" => tokenizer_assets,
        )
        open(joinpath(temporary, "manifest.json"), "w") do io
            JSON3.write(io, manifest)
        end
        mv(temporary, target)
        return target
    finally
        isdir(temporary) && rm(temporary; recursive=true)
    end
end

function _flint_artifact_path(directory, filename)
    filename isa AbstractString || throw(ArgumentError("Flint artifact filename must be a string"))
    isabspath(filename) && throw(ArgumentError("Flint artifact filename must be relative"))
    candidate = normpath(joinpath(directory, filename))
    relative = relpath(candidate, directory)
    (relative == ".." || startswith(relative, ".." * string(Base.Filesystem.path_separator))) &&
        throw(ArgumentError("Flint artifact filename escapes its directory"))
    isfile(candidate) || throw(ArgumentError("missing Flint artifact file `$filename`"))
    resolved = realpath(candidate)
    realrelative = relpath(resolved, realpath(directory))
    (realrelative == ".." || startswith(realrelative, ".." * string(Base.Filesystem.path_separator))) &&
        throw(ArgumentError("Flint artifact symlink escapes its directory"))
    return candidate
end

function _flint_read_array(directory, record, expected_shape; dtype=nothing)
    path = _flint_artifact_path(directory, String(record["file"]))
    types = Dict("Float32" => Float32, "BFloat16" => BFloat16, "Int8" => Int8, "UInt8" => UInt8)
    type_name = String(record["dtype"])
    haskey(types, type_name) || throw(ArgumentError("unsupported Flint artifact dtype `$type_name`"))
    T = types[type_name]
    dtype === nothing || T === dtype || throw(ArgumentError("invalid Flint artifact storage dtype"))
    Tuple(Int.(record["shape"])) == expected_shape || throw(DimensionMismatch("Flint artifact payload shape mismatch"))
    expected_bytes = prod(BigInt.(expected_shape)) * sizeof(T)
    filesize(path) == expected_bytes == record["bytes"] || throw(ArgumentError("Flint artifact payload size mismatch"))
    bytes2hex(open(sha256, path)) == String(record["sha256"]) || throw(ArgumentError("Flint artifact checksum mismatch: $path"))
    values = Array{T}(undef, expected_shape)
    open(path, "r") do io
        read!(io, values)
    end
    all(isfinite, values) || throw(ArgumentError("Flint artifact contains nonfinite values"))
    return values
end

"""
    load_flint_quantized(path; device=identity)

Reload a saved Flint artifact, verifying model tensor names, all shapes, data
types, payload sizes, and SHA-256 checksums before using each tensor. Returns
the same `model`/`parameters` interface as the HF loaders.
"""
function load_flint_quantized(path::AbstractString; device=identity)
    Base.ENDIAN_BOM == 0x04030201 || throw(ArgumentError("Flint artifacts currently require a little-endian host"))
    directory = abspath(path)
    manifest = JSON3.read(read(joinpath(directory, "manifest.json"), String))
    manifest["format"] == "lifeai-flint-quantized" || throw(ArgumentError("not a LifeAI Flint artifact"))
    version = manifest["version"]
    version isa Integer && !(version isa Bool) && version in (1, FLINT_QUANTIZED_FORMAT_VERSION) ||
        throw(ArgumentError("unsupported Flint artifact version"))
    manifest["endianness"] == "little" || throw(ArgumentError("unsupported Flint artifact endianness"))
    config_path = _flint_artifact_path(directory, "config.json")
    bytes2hex(open(sha256, config_path)) == String(manifest["config_sha256"]) || throw(ArgumentError("Flint artifact config checksum mismatch"))
    model = load_hf_flint_config(directory)
    shapes = flint_expected_tensor_shapes(model)
    entries = manifest["tensors"]
    names = String[String(entry["name"]) for entry in entries]
    length(names) == length(unique(names)) || throw(ArgumentError("duplicate Flint artifact tensor"))
    Set(names) == Set(keys(shapes)) || throw(ArgumentError("Flint artifact tensor set does not match model"))
    parameters = Dict{String,Any}()
    for entry in entries
        name = String(entry["name"])
        shape = shapes[name]
        Tuple(Int.(entry["shape"])) == shape || throw(DimensionMismatch("Flint artifact tensor `$name` shape mismatch"))
        kind = String(entry["kind"])
        weight = if kind == "float"
            value = _flint_read_array(directory, entry["values"], shape)
            eltype(value) <: AbstractFloat || throw(ArgumentError("Flint float tensor has an integer dtype"))
            value
        elseif kind == "int8"
            _flint_quantizable(name) || throw(ArgumentError("Flint tensor `$name` must remain floating point"))
            length(shape) == 2 || throw(DimensionMismatch("Flint quantized weight must be a matrix"))
            Int8ChannelWeight(
                _flint_read_array(directory, entry["values"], shape; dtype=Int8),
                _flint_read_array(directory, entry["scale"], (shape[1],); dtype=Float32),
            )
        elseif kind == "int4"
            _flint_quantizable(name) || throw(ArgumentError("Flint tensor `$name` must remain floating point"))
            length(shape) == 2 || throw(DimensionMismatch("Flint quantized weight must be a matrix"))
            group = Int(entry["group"])
            group > 0 && iseven(group) && shape[2] % group == 0 || throw(ArgumentError("invalid Flint INT4 group"))
            Int4GroupWeight(
                _flint_read_array(directory, entry["values"], (shape[1], shape[2] ÷ 2); dtype=UInt8),
                _flint_read_array(directory, entry["scale"], (shape[1], shape[2] ÷ group); dtype=Float32),
                group, shape[2],
            )
        elseif kind == "nvfp4"
            version >= 2 || throw(ArgumentError("NVFP4 requires Flint artifact version 2"))
            _flint_quantizable(name) || throw(ArgumentError("Flint tensor `$name` must remain floating point"))
            length(shape) == 2 || throw(DimensionMismatch("Flint quantized weight must be a matrix"))
            get(entry, "encoding", nothing) == "e2m1-e4m3fn-block16" || throw(ArgumentError("unsupported NVFP4 artifact encoding"))
            get(entry, "group", nothing) == 16 || throw(ArgumentError("NVFP4 artifacts require block size 16"))
            blocks = cld(shape[2], 16)
            NVFP4Weight(
                _flint_read_array(directory, entry["values"], (shape[1], 8 * blocks); dtype=UInt8),
                _flint_read_array(directory, entry["scale"], (shape[1], blocks); dtype=UInt8),
                only(_flint_read_array(directory, entry["global_scale"], (1,); dtype=Float32)),
                shape[2],
            )
        else
            throw(ArgumentError("unsupported Flint artifact weight kind `$kind`"))
        end
        parameters[name] = _flint_weight_to_device(device, weight)
    end
    for entry in get(manifest, "tokenizer_assets", [])
        asset = _flint_artifact_path(directory, String(entry["file"]))
        bytes2hex(open(sha256, asset)) == String(entry["sha256"]) || throw(ArgumentError("Flint tokenizer asset checksum mismatch"))
    end
    return (model=model, parameters=parameters, source=directory)
end
