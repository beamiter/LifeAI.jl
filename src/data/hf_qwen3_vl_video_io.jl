# Chapter 152: Qwen3-VL video container demuxing, official frame sampling, and
# fail-closed video file ingestion (MP4/MKV/WebM via ffmpeg/ffprobe).

const QWEN3_VL_DEFAULT_VIDEO_MAX_FILE_BYTES = 512 * 1024 * 1024
const QWEN3_VL_DEFAULT_VIDEO_MAX_DECODED_BYTES = 256 * 1024 * 1024

"""
    Qwen3VLDecodedVideo

Decoded and sampled RGB video frames from a local container file, carrying exact
0-based source frame indices and frame-rate metadata so Qwen3-VL midpoint
timestamps reflect true wall-clock seconds.
"""
struct Qwen3VLDecodedVideo
    frames::Vector{Array{UInt8,3}}
    source_fps::Float64
    sample_fps::Float64
    total_frames::Int
    frame_indices::Vector{Int}
    duration_seconds::Float64
    height::Int
    width::Int
    source_path::String
end

function _qwen3_vl_video_positive_real(value, label::AbstractString)
    value isa Real && !(value isa Bool) || throw(ArgumentError(
        "$label must be a finite positive real number",
    ))
    parsed = Float64(value)
    isfinite(parsed) && parsed > 0.0 || throw(ArgumentError(
        "$label must be a finite positive real number",
    ))
    return parsed
end

"""
    qwen3_vl_sample_frame_indices(total_frames; source_fps, target_fps=2.0,
                                  min_frames=4, max_frames=32, frame_factor=2)

Compute deterministic 0-based frame indices matching Qwen3-VL's official
`smart_nframes` and uniform linspace sampling policy.
"""
function qwen3_vl_sample_frame_indices(
    total_frames;
    source_fps::Real,
    target_fps::Real=2.0,
    min_frames::Integer=4,
    max_frames::Integer=32,
    frame_factor::Integer=2,
)
    total = _qwen3_vl_positive_host_int(total_frames, "total_frames")
    src_fps = _qwen3_vl_video_positive_real(source_fps, "source_fps")
    tgt_fps = _qwen3_vl_video_positive_real(target_fps, "target_fps")
    min_f = _qwen3_vl_positive_host_int(min_frames, "min_frames")
    max_f = _qwen3_vl_positive_host_int(max_frames, "max_frames")
    factor = _qwen3_vl_positive_host_int(frame_factor, "frame_factor")

    min_f <= max_f || throw(ArgumentError(
        "min_frames ($min_f) must not exceed max_frames ($max_f)",
    ))
    min_f % factor == 0 || throw(ArgumentError(
        "min_frames ($min_f) must be divisible by frame_factor ($factor)",
    ))
    max_f % factor == 0 || throw(ArgumentError(
        "max_frames ($max_f) must be divisible by frame_factor ($factor)",
    ))
    total >= factor || throw(ArgumentError(
        "video total_frames ($total) must be at least frame_factor ($factor)",
    ))

    raw_nframes = Float64(total) / src_fps * tgt_fps
    effective_min = min(min_f, total - (total % factor))
    effective_min = max(factor, effective_min)
    effective_max = min(max_f, total - (total % factor))
    effective_max = max(effective_min, effective_max)

    clamped = clamp(raw_nframes, Float64(effective_min), Float64(effective_max))
    rounded = round(Int, clamped / Float64(factor)) * factor
    nframes = clamp(rounded, effective_min, effective_max)

    if nframes == 1
        return Int[0]
    end
    indices = Vector{Int}(undef, nframes)
    step = Float64(total - 1) / Float64(nframes - 1)
    for idx in 1:nframes
        pos = clamp(floor(Int, (idx - 1) * step + 0.5), 0, total - 1)
        indices[idx] = pos
    end
    return indices
end

function _qwen3_vl_find_jll_ffmpeg(tool::Symbol)
    for (pkgid, mod) in Base.loaded_modules
        if pkgid.name == "FFMPEG_jll" && isdefined(mod, tool)
            return Base.invokelatest(getfield(mod, tool))
        end
    end
    # Lazily load Plots (a direct Project.toml dependency) to access FFMPEG_jll's
    # library-path-configured Cmd when system ffmpeg/ffprobe is absent.
    try
        plots_id = Base.identify_package(LifeAI, "Plots")
        if plots_id !== nothing
            Base.require(plots_id)
            for (pkgid, mod) in Base.loaded_modules
                if pkgid.name == "FFMPEG_jll" && isdefined(mod, tool)
                    return Base.invokelatest(getfield(mod, tool))
                end
            end
        end
    catch
    end
    return nothing
end

function _qwen3_vl_resolve_ffmpeg_cmd(tool::Symbol; explicit_path=nothing)
    tool in (:ffmpeg, :ffprobe) || throw(ArgumentError("unsupported ffmpeg tool $tool"))
    if explicit_path !== nothing
        explicit_path isa AbstractString || throw(ArgumentError("$tool path must be a string"))
        path = strip(String(explicit_path))
        isfile(path) || throw(ArgumentError("$tool binary does not exist: $path"))
        return Cmd([path])
    end
    env_key = tool === :ffmpeg ? "LIFEAI_FFMPEG" : "LIFEAI_FFPROBE"
    env_path = strip(get(ENV, env_key, ""))
    if !isempty(env_path)
        isfile(env_path) || throw(ArgumentError("$env_key binary does not exist: $env_path"))
        return Cmd([env_path])
    end
    sys_bin = Sys.which(String(tool))
    sys_bin !== nothing && return Cmd([sys_bin])
    jll_cmd = _qwen3_vl_find_jll_ffmpeg(tool)
    jll_cmd !== nothing && return jll_cmd
    throw(ArgumentError(
        "could not locate $(tool); install $(tool) on PATH, set $env_key, or load FFMPEG_jll",
    ))
end

function _qwen3_vl_parse_rational_fps(raw::AbstractString)
    text = strip(String(raw))
    isempty(text) && return nothing
    if occursin('/', text)
        parts = split(text, '/'; limit=2)
        num = tryparse(Float64, strip(parts[1]))
        den = tryparse(Float64, strip(parts[2]))
        (num === nothing || den === nothing || den <= 0.0 || num <= 0.0) && return nothing
        val = num / den
        return isfinite(val) && val > 0.0 ? val : nothing
    end
    val = tryparse(Float64, text)
    return val !== nothing && isfinite(val) && val > 0.0 ? val : nothing
end

"""
    qwen3_vl_probe_video_file(path; max_file_bytes=512MiB, ffprobe_path=nothing)

Inspect a local video container file via `ffprobe` and return its validated
`(width, height, source_fps, total_frames, duration_seconds, file_bytes)`
metadata. Fails closed on missing, empty, oversized, or non-video files.
"""
function qwen3_vl_probe_video_file(
    path::AbstractString;
    max_file_bytes::Integer=QWEN3_VL_DEFAULT_VIDEO_MAX_FILE_BYTES,
    ffprobe_path=nothing,
)
    max_bytes = _qwen3_vl_positive_host_int(max_file_bytes, "max_file_bytes")
    filepath = String(path)
    isfile(filepath) || throw(ArgumentError("video file does not exist: $filepath"))
    fsize = filesize(filepath)
    1 <= fsize <= max_bytes || throw(ArgumentError(
        "video file size ($fsize bytes) is outside allowed range 1:$max_bytes",
    ))

    probe_cmd = _qwen3_vl_resolve_ffmpeg_cmd(:ffprobe; explicit_path=ffprobe_path)
    full_cmd = `$probe_cmd -v error -select_streams v:0 -show_entries stream=width,height,r_frame_rate,avg_frame_rate,nb_frames,duration -show_entries format=duration -of json $filepath`
    out_buf = IOBuffer()
    err_buf = IOBuffer()
    proc = run(pipeline(ignorestatus(full_cmd); stdout=out_buf, stderr=err_buf))
    proc.exitcode == 0 || throw(ArgumentError(
        "ffprobe failed for $filepath: $(strip(String(take!(err_buf))))",
    ))

    parsed = try
        JSON3.read(String(take!(out_buf)))
    catch error
        throw(ArgumentError("ffprobe produced invalid JSON for $filepath: $(sprint(showerror, error))"))
    end
    streams = get(parsed, :streams, nothing)
    (streams isa AbstractVector && !isempty(streams)) || throw(ArgumentError(
        "no video stream found in $filepath",
    ))
    stream = streams[1]
    width = _qwen3_vl_positive_host_int(get(stream, :width, 0), "video width")
    height = _qwen3_vl_positive_host_int(get(stream, :height, 0), "video height")

    fps = _qwen3_vl_parse_rational_fps(string(get(stream, :avg_frame_rate, "")))
    if fps === nothing
        fps = _qwen3_vl_parse_rational_fps(string(get(stream, :r_frame_rate, "")))
    end
    fps === nothing && throw(ArgumentError("video stream in $filepath has no valid frame rate"))

    duration = nothing
    for candidate in (
        get(stream, :duration, nothing),
        hasproperty(parsed, :format) ? get(parsed.format, :duration, nothing) : nothing,
    )
        candidate === nothing && continue
        val = tryparse(Float64, string(candidate))
        if val !== nothing && isfinite(val) && val > 0.0
            duration = val
            break
        end
    end

    nb_frames = nothing
    raw_nb = get(stream, :nb_frames, nothing)
    if raw_nb !== nothing
        parsed_nb = tryparse(Int, string(raw_nb))
        if parsed_nb !== nothing && parsed_nb > 0
            nb_frames = parsed_nb
        end
    end
    if nb_frames === nothing && duration !== nothing
        nb_frames = max(1, round(Int, duration * fps))
    end
    nb_frames === nothing && throw(ArgumentError(
        "could not determine frame count for $filepath",
    ))
    if duration === nothing
        duration = Float64(nb_frames) / fps
    end
    return (;
        width,
        height,
        source_fps=fps,
        total_frames=nb_frames,
        duration_seconds=duration,
        file_bytes=Int(fsize),
        source_path=abspath(filepath),
    )
end

"""
    qwen3_vl_decode_video_file(path; target_fps=2.0, min_frames=4, max_frames=32,
                               frame_factor=2, max_file_bytes=512MiB,
                               max_decoded_bytes=256MiB,
                               ffmpeg_path=nothing, ffprobe_path=nothing)

Decode a local video container file (MP4/MKV/WebM/AVI) via `ffprobe` + `ffmpeg`
directly into bounded RGB `UInt8` `(H, W, 3)` frames, sampled deterministically
with [`qwen3_vl_sample_frame_indices`](@ref).
"""
function qwen3_vl_decode_video_file(
    path::AbstractString;
    target_fps::Real=2.0,
    min_frames::Integer=4,
    max_frames::Integer=32,
    frame_factor::Integer=2,
    max_file_bytes::Integer=QWEN3_VL_DEFAULT_VIDEO_MAX_FILE_BYTES,
    max_decoded_bytes::Integer=QWEN3_VL_DEFAULT_VIDEO_MAX_DECODED_BYTES,
    ffmpeg_path=nothing,
    ffprobe_path=nothing,
)
    max_dec = _qwen3_vl_positive_host_int(max_decoded_bytes, "max_decoded_bytes")
    meta = qwen3_vl_probe_video_file(path; max_file_bytes, ffprobe_path)
    frame_bytes = _qwen3_vl_checked_mul(
        _qwen3_vl_checked_mul(meta.height, meta.width, "video frame pixels"),
        3,
        "video frame bytes",
    )
    frame_bytes <= max_dec || throw(ArgumentError(
        "single decoded video frame ($frame_bytes bytes) exceeds max_decoded_bytes ($max_dec)",
    ))

    ff_cmd = _qwen3_vl_resolve_ffmpeg_cmd(:ffmpeg; explicit_path=ffmpeg_path)
    decode_cmd = `$ff_cmd -v error -i $(meta.source_path) -map 0:v:0 -vsync 0 -f rawvideo -pix_fmt rgb24 pipe:1`

    raw_frames = Array{UInt8,3}[]
    total_decoded_bytes = 0
    err_buf = IOBuffer()
    frame_buf = Vector{UInt8}(undef, frame_bytes)

    proc = open(pipeline(ignorestatus(decode_cmd); stderr=err_buf), "r")
    try
        while !eof(proc)
            got = readbytes!(proc, frame_buf, frame_bytes)
            got == 0 && break
            got == frame_bytes || throw(ArgumentError(
                "truncated RGB frame from ffmpeg (got $got of $frame_bytes bytes)",
            ))
            total_decoded_bytes += frame_bytes
            total_decoded_bytes <= max_dec || throw(ArgumentError(
                "decoded video stream exceeds max_decoded_bytes ($max_dec)",
            ))
            # rawvideo rgb24 is row-major (H, W, 3) => Julia reshape (3, W, H) then permute (3, 2, 1).
            chw = reshape(copy(frame_buf), 3, meta.width, meta.height)
            push!(raw_frames, permutedims(chw, (3, 2, 1)))
        end
    finally
        close(proc)
        wait(proc)
    end
    success(proc) || throw(ArgumentError(
        "ffmpeg decode failed for $(meta.source_path): $(strip(String(take!(err_buf))))",
    ))

    actual_total = length(raw_frames)
    actual_total >= _qwen3_vl_positive_host_int(frame_factor, "frame_factor") ||
        throw(ArgumentError(
            "decoded video contains only $actual_total frame(s); need at least $frame_factor",
        ))

    indices = qwen3_vl_sample_frame_indices(
        actual_total;
        source_fps=meta.source_fps,
        target_fps,
        min_frames,
        max_frames,
        frame_factor,
    )
    sampled_frames = Array{UInt8,3}[raw_frames[idx + 1] for idx in indices]
    duration = Float64(actual_total) / meta.source_fps
    eff_fps = length(sampled_frames) / max(duration, 1e-6)
    return Qwen3VLDecodedVideo(
        sampled_frames,
        meta.source_fps,
        eff_fps,
        actual_total,
        indices,
        duration,
        meta.height,
        meta.width,
        meta.source_path,
    )
end

"""
    qwen3_vl_encode_video_file(path, frames; fps=4.0, codec="mpeg4", ffmpeg_path=nothing)

Encode a sequence of RGB `UInt8` `(H, W, 3)` frames into a local video container
file using `ffmpeg` (primarily for deterministic container tests and benchmarks).
"""
function qwen3_vl_encode_video_file(
    path::AbstractString,
    frames::AbstractVector{<:AbstractArray{UInt8,3}};
    fps::Real=4.0,
    codec::AbstractString="mpeg4",
    ffmpeg_path=nothing,
)
    !isempty(frames) || throw(ArgumentError("frames must not be empty"))
    fps_val = _qwen3_vl_video_positive_real(fps, "fps")
    h, w, c = size(frames[1])
    c == 3 || throw(DimensionMismatch("frames must be (H, W, 3) RGB UInt8 arrays"))
    for (idx, frame) in enumerate(frames)
        size(frame) == (h, w, 3) || throw(DimensionMismatch(
            "frame $idx size $(size(frame)) does not match frame 1 size $((h, w, 3))",
        ))
    end
    ff_cmd = _qwen3_vl_resolve_ffmpeg_cmd(:ffmpeg; explicit_path=ffmpeg_path)
    out_path = abspath(String(path))
    mkpath(dirname(out_path))
    raw_io = IOBuffer()
    for frame in frames
        # Convert HWC to packed row-major RGB bytes (3, W, H).
        write(raw_io, vec(permutedims(frame, (3, 2, 1))))
    end
    seekstart(raw_io)
    err_buf = IOBuffer()
    encode_cmd = `$ff_cmd -y -v error -f rawvideo -pix_fmt rgb24 -s $(w)x$(h) -r $(fps_val) -i pipe:0 -c:v $(String(codec)) -q:v 1 -pix_fmt yuv420p $out_path`
    proc = run(pipeline(ignorestatus(encode_cmd); stdin=raw_io, stderr=err_buf))
    success(proc) || throw(ArgumentError(
        "ffmpeg encode failed for $out_path: $(strip(String(take!(err_buf))))",
    ))
    return out_path
end
