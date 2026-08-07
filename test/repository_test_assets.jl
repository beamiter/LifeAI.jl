using JSON3

"""Return the unique frozen benchmark artifact with the requested filename."""
function repository_test_asset(filename::AbstractString)
    search_root = joinpath(dirname(@__DIR__), "benchmark_results")
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
