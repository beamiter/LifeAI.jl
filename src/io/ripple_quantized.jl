const RIPPLE_QUANTIZED_FORMAT_VERSION = 1

function _ripple_verify_payload_checksums(directory,entries,backend)
    backend in (:auto,:system,:julia) || throw(ArgumentError("unknown Ripple checksum backend"))
    backend===:julia && return false
    checker=Sys.which("sha256sum")
    if checker===nothing
        backend===:system && throw(ArgumentError("sha256sum is unavailable"))
        return false
    end
    io=IOBuffer()
    for entry in entries,key in ("values","scale","global_scale")
        haskey(entry,key) || continue
        record=entry[key]
        path=_flint_artifact_path(directory,record["file"])
        name=relpath(path,directory)
        # Unusual escaped names retain the portable per-file Julia verifier.
        any(ch->ch in ('\\','\n','\r'),name) && return false
        digest=String(record["sha256"])
        occursin(r"^[0-9a-fA-F]{64}$",digest) || throw(ArgumentError("invalid Ripple SHA-256 record"))
        println(io,digest,"  ",name)
    end
    command=Cmd(Cmd([checker,"--check","--status"]);dir=directory)
    success(pipeline(command;stdin=IOBuffer(take!(io)))) || throw(ArgumentError("Ripple artifact checksum mismatch"))
    return true
end

function _ripple_artifact_weight(directory, entry, shape;verified_sha256=false)
    name = String(entry["name"])
    Tuple(Int.(entry["shape"])) == shape || throw(DimensionMismatch("Ripple artifact shape mismatch: $name"))
    kind = String(entry["kind"])
    if kind == "float"
        weight = _flint_read_array(directory, entry["values"], shape;verified_sha256)
        eltype(weight) <: AbstractFloat || throw(ArgumentError("Ripple floating tensor has integer storage"))
        return weight
    end
    _ripple_quantizable(name) && length(shape) == 2 || throw(ArgumentError("Ripple tensor must remain floating point: $name"))
    if kind == "int4"
        group = _hf_strict_host_int(entry["group"], "Ripple INT4 group")
        group > 0 && iseven(group) && shape[2] % group == 0 || throw(ArgumentError("invalid Ripple INT4 group"))
        return Int4GroupWeight(
            _flint_read_array(directory, entry["values"], (shape[1],shape[2]÷2); dtype=UInt8,verified_sha256),
            _flint_read_array(directory, entry["scale"], (shape[1],shape[2]÷group); dtype=Float32,verified_sha256), group, shape[2])
    elseif kind == "int8"
        return Int8ChannelWeight(_flint_read_array(directory, entry["values"], shape; dtype=Int8,verified_sha256),
            _flint_read_array(directory, entry["scale"], (shape[1],); dtype=Float32,verified_sha256))
    elseif kind == "nvfp4"
        get(entry,"encoding",nothing) == "e2m1-e4m3fn-block16" && get(entry,"group",nothing) == 16 ||
            throw(ArgumentError("unsupported Ripple NVFP4 encoding"))
        blocks = cld(shape[2],16)
        return NVFP4Weight(
            _flint_read_array(directory,entry["values"],(shape[1],8blocks);dtype=UInt8,verified_sha256),
            _flint_read_array(directory,entry["scale"],(shape[1],blocks);dtype=UInt8,verified_sha256),
            only(_flint_read_array(directory,entry["global_scale"],(1,);dtype=Float32,verified_sha256)),shape[2])
    end
    throw(ArgumentError("unsupported Ripple artifact weight kind: $kind"))
end

"""
    quantize_ripple_checkpoint(source, destination; scheme=:int4, group=nothing,
                               device=identity, dtype=BFloat16, on_progress=nothing)

Convert a complete HF checkpoint one tensor at a time to a checksummed portable
Ripple artifact. `device=CUDA.cu` performs quantization on GPU. The destination
must not exist; incomplete builds are removed without replacing other files.
"""
function quantize_ripple_checkpoint(source::AbstractString,destination::AbstractString;
        scheme::Symbol=:int4,group::Union{Nothing,Integer}=nothing,device=identity,
        dtype::Type=BFloat16,on_progress=nothing)
    Base.ENDIAN_BOM == 0x04030201 || throw(ArgumentError("Ripple artifacts require little endian"))
    dtype in (Float32,BFloat16) || throw(ArgumentError("invalid Ripple artifact dtype"))
    spec = _flint_quantization_spec(scheme,group,:maxabs)
    model = load_hf_ripple_config(source)
    reader = open_safetensors_reader(source)
    shapes = verify_ripple_checkpoint(reader,model)
    target=abspath(destination)
    ispath(target) && throw(ArgumentError("Ripple artifact destination already exists"))
    if scheme === :int4
        all(name -> !_ripple_quantizable(name) || shapes[name][2] % spec.group == 0,keys(shapes)) ||
            throw(ArgumentError("Ripple INT4 input widths must be divisible by group"))
    end
    mkpath(dirname(target))
    temporary=mktempdir(dirname(target);prefix=".ripple-quantize-")
    try
        mkpath(joinpath(temporary,"tensors"))
        write(joinpath(temporary,"config.json"),JSON3.write(ripple_config_dict(model)))
        entries=Any[]
        for (index,name) in enumerate(sort!(collect(keys(shapes))))
            weight=read_safetensors_tensor(reader,name;target_dtype=length(shapes[name])==1 ? Float32 : dtype)
            packed=if _ripple_quantizable(name)
                mapped=device(weight)
                result=spec.scheme===:nvfp4 ? quantize_nvfp4(mapped) : _quantize_linear(mapped,spec)
                host=_flint_weight_to_device(Array,result)
                _ripple_synchronize_weight(result)
                _ripple_release_weight!(result)
                mapped === weight || _ripple_release_weight!(mapped)
                host
            else
                weight
            end
            push!(entries,_flint_write_weight(temporary,index,name,packed))
            on_progress === nothing || on_progress(name,index,length(shapes))
            # Bound host copies and device dequantization temporaries during the
            # 130GB conversion instead of retaining an entire allocator history.
            index % 12 == 0 && GC.gc(false)
        end
        assets=Any[]
        for name in ("tokenizer.json","tokenizer_config.json","special_tokens_map.json","chat_template.jinja","LICENSE")
            file=joinpath(source,name)
            isfile(file) || continue
            cp(file,joinpath(temporary,name))
            push!(assets,Dict("file"=>name,"sha256"=>bytes2hex(open(sha256,file))))
        end
        manifest=Dict("format"=>"lifeai-ripple-quantized","version"=>RIPPLE_QUANTIZED_FORMAT_VERSION,
            "endianness"=>"little","scheme"=>String(scheme),"source"=>abspath(source),
            "source_index_sha256"=>bytes2hex(open(sha256,reader.source)),
            "config_sha256"=>bytes2hex(open(sha256,joinpath(temporary,"config.json"))),
            "tensors"=>entries,"tokenizer_assets"=>assets)
        write(joinpath(temporary,"manifest.json"),JSON3.write(manifest))
        mv(temporary,target)
        return target
    finally
        isdir(temporary) && rm(temporary;recursive=true)
    end
end

"""
    load_ripple_quantized(path; device=identity, resident_bytes=nothing, on_progress=nothing)

Verify and load a portable Ripple artifact. With a byte budget, the remaining
packed weights stay in host RAM and are uploaded only for the active layer.
The budget covers stored weights; leave additional memory for KV/activations.
"""
function load_ripple_quantized(path::AbstractString;device=identity,
        resident_bytes::Union{Nothing,Integer}=nothing,on_progress=nothing,checksum_backend::Symbol=:auto)
    Base.ENDIAN_BOM == 0x04030201 || throw(ArgumentError("Ripple artifacts require little endian"))
    directory=abspath(path)
    manifest=JSON3.read(read(joinpath(directory,"manifest.json"),String))
    get(manifest,"format",nothing)=="lifeai-ripple-quantized" || throw(ArgumentError("not a Ripple artifact"))
    get(manifest,"version",nothing)===RIPPLE_QUANTIZED_FORMAT_VERSION || throw(ArgumentError("unsupported Ripple artifact version"))
    manifest["endianness"]=="little" || throw(ArgumentError("unsupported Ripple artifact endianness"))
    file=_flint_artifact_path(directory,"config.json")
    bytes2hex(open(sha256,file))==manifest["config_sha256"] || throw(ArgumentError("Ripple artifact config checksum mismatch"))
    model=load_hf_ripple_config(directory)
    shapes=ripple_expected_tensor_shapes(model)
    entries=manifest["tensors"]
    names=String[String(entry["name"]) for entry in entries]
    length(names)==length(unique(names)) && Set(names)==Set(keys(shapes)) || throw(ArgumentError("Ripple artifact tensor set mismatch"))
    for asset in manifest["tokenizer_assets"]
        file=_flint_artifact_path(directory,asset["file"])
        bytes2hex(open(sha256,file))==asset["sha256"] || throw(ArgumentError("Ripple tokenizer checksum mismatch"))
    end
    verified_sha256=_ripple_verify_payload_checksums(directory,entries,checksum_backend)
    parameters=Dict{String,Any}()
    for (index,entry) in enumerate(entries)
        name=String(entry["name"])
        weight=_ripple_artifact_weight(directory,entry,shapes[name];verified_sha256)
        parameters[name]=resident_bytes===nothing ? _flint_weight_to_device(device,weight) : weight
        on_progress===nothing || on_progress(name,index,length(entries))
    end
    if resident_bytes!==nothing
        parameters=_ripple_offload_parameters(parameters,device,resident_bytes)
    end
    return (;model,parameters,source=directory,quantization=Symbol(manifest["scheme"]),
        checksum_backend=verified_sha256 ? :system : :julia)
end
