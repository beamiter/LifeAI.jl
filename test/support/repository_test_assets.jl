using JSON3
using SHA: sha256

if !isdefined(@__MODULE__, :LIFEAI_TEST_ROOT)
    @eval const LIFEAI_TEST_ROOT = normpath(joinpath(@__DIR__, ".."))
end
if !isdefined(@__MODULE__, :LIFEAI_REPO_ROOT)
    @eval const LIFEAI_REPO_ROOT = normpath(joinpath(LIFEAI_TEST_ROOT, ".."))
end

"""Return the unique frozen benchmark artifact with the requested filename."""
function repository_test_asset(filename::AbstractString)
    search_root = joinpath(LIFEAI_REPO_ROOT, "benchmark_results")
    matches = String[]
    for (directory, _, files) in walkdir(search_root)
        filename in files && push!(matches, joinpath(directory, filename))
    end
    length(matches) == 1 || error(
        "expected one repository artifact named $(repr(filename)), found $(length(matches))",
    )
    return only(matches)
end

"""Locate a model reference by metadata instead of its historical directory name."""
function qwen3_reference_directory(
    model_dir::AbstractString;
    revision::Union{Nothing,AbstractString}=nothing,
    compute_dtype::Union{Nothing,AbstractString}=nothing,
)
    reference_root = joinpath(model_dir, "lifeai-references")
    isdir(reference_root) || error("missing model reference directory: $reference_root")
    matches = String[]
    for (directory, _, files) in walkdir(reference_root)
        all(file -> file in files, ("reference.json", "reference.safetensors")) ||
            continue
        metadata = JSON3.read(read(joinpath(directory, "reference.json"), String))
        revision === nothing ||
            String(get(metadata, "revision", "")) == revision || continue
        compute_dtype === nothing ||
            String(get(metadata, "compute_dtype", "")) == compute_dtype || continue
        push!(matches, directory)
    end
    length(matches) == 1 || error(
        "expected one matching model reference under $reference_root, found $(length(matches))",
    )
    return only(matches)
end

const QWEN3_MOE_CHAPTER25_35_PROVENANCE = joinpath(
    LIFEAI_REPO_ROOT,
    "benchmark_results",
    "qwen3_moe_historical_provenance",
    "chapter25_35_source_snapshot.json",
)

_repository_relative_path(path::AbstractString) =
    replace(normpath(String(path)), '\\' => '/')

"""
Return audit fields for one source in a Chapter 25-35 historical report.

The report digest identifies the source bytes used when the timing was recorded.
It is checked against that report's immutable historical snapshot, not against
the mutable current implementation. When the report's original Git commit is
available, the snapshot is also recomputed from that commit. A source archive or
shallow clone can still validate the registered snapshot without Git history.
"""
function qwen3_moe_historical_benchmark_source_status(
    summary_path::AbstractString,
    summary,
    name,
    relative_path::AbstractString,
)
    source_name = String(name)
    canonical_path = _repository_relative_path(relative_path)
    canonical_summary_path = _repository_relative_path(relpath(
        abspath(summary_path),
        LIFEAI_REPO_ROOT,
    ))
    recorded_digest = String(summary["source_sha256"][source_name])
    provenance = JSON3.read(read(
        QWEN3_MOE_CHAPTER25_35_PROVENANCE,
        String,
    ))
    reports = provenance["reports"]
    report_registered = haskey(reports, canonical_summary_path)
    report = report_registered ? reports[canonical_summary_path] : nothing
    sources = report_registered ? report["sources"] : nothing
    snapshot_registered = report_registered && haskey(sources, source_name)
    snapshot = snapshot_registered ? sources[source_name] : nothing
    snapshot_path = snapshot_registered ? String(snapshot["path"]) : ""
    snapshot_digest = snapshot_registered ? String(snapshot["sha256"]) : ""
    source_path_matches = snapshot_registered &&
        canonical_path == snapshot_path &&
        (!haskey(summary, "source_paths") ||
         _repository_relative_path(summary["source_paths"][source_name]) ==
         snapshot_path)
    digest_is_sha256 = occursin(r"^[0-9a-f]{64}$", recorded_digest)
    current_source_exists = snapshot_registered && isfile(joinpath(
        LIFEAI_REPO_ROOT,
        split(snapshot_path, '/')...,
    ))

    git_snapshot_matches = nothing
    git_program = Sys.which("git")
    if snapshot_registered && git_program !== nothing &&
            ispath(joinpath(LIFEAI_REPO_ROOT, ".git"))
        commit = String(report["original_report_commit"])
        object = string(commit, ':', snapshot_path)
        try
            historical_bytes = read(Cmd([
                git_program,
                "-C",
                LIFEAI_REPO_ROOT,
                "show",
                object,
            ]))
            git_snapshot_matches =
                bytes2hex(sha256(historical_bytes)) == snapshot_digest
        catch
            # A source archive or shallow clone can contain the registered
            # snapshot without carrying the historical commit object.
            git_snapshot_matches = nothing
        end
    end

    return (;
        report_registered,
        source_path_matches,
        digest_is_sha256,
        current_source_exists,
        snapshot_registered,
        digest_matches_snapshot=recorded_digest == snapshot_digest,
        git_snapshot_matches,
    )
end
