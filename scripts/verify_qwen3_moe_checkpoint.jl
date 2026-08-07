#!/usr/bin/env julia

using JSON3
using LifeAI: verify_qwen3_moe_checkpoint

function usage()
    error(
        "usage: julia --project=. scripts/verify_qwen3_moe_checkpoint.jl " *
        "MODEL_DIR [OUTPUT_JSON] [--fast]",
    )
end

fast = "--fast" in ARGS
positional = filter(!=("--fast"), ARGS)
1 <= length(positional) <= 2 || usage()
model_dir = positional[1]
output_path = length(positional) == 2 ? positional[2] : nothing

timed = @timed verify_qwen3_moe_checkpoint(
    model_dir;
    verify_shard_checksums=!fast,
)
report = timed.value
payload = (;
    schema_version=1,
    model_id=report.spec.model_id,
    revision=report.spec.revision,
    source=report.source,
    config_sha256=report.config_sha256,
    index_sha256=report.index_sha256,
    tensor_count=report.tensor_count,
    tensor_bytes=report.tensor_bytes,
    shard_payload_bytes=report.shard_payload_bytes,
    shard_checksums_verified=report.shard_checksums_verified,
    elapsed_seconds=timed.time,
    allocated_bytes=timed.bytes,
    shards=[(;
        path=shard.path,
        bytes=shard.bytes,
        sha256=shard.sha256,
    ) for shard in report.shards],
)

serialized = sprint(io -> JSON3.pretty(io, payload))
println(serialized)
if output_path !== nothing
    open(output_path, "w") do io
        write(io, serialized)
        write(io, '\n')
    end
end
