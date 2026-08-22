using Test
using SHA

if isdefined(@__MODULE__, :LifeAI)
    using .LifeAI:
        AgentMemoryRecord,
        AgentMemoryStore,
        MAX_AGENT_MEMORY_JOURNAL_BYTES,
        MAX_AGENT_MEMORY_METADATA_ENTRIES,
        MAX_AGENT_MEMORY_METADATA_KEY_BYTES,
        MAX_AGENT_MEMORY_METADATA_VALUE_BYTES,
        MAX_AGENT_MEMORY_RECORD_BYTES,
        MAX_AGENT_MEMORY_TEXT_BYTES,
        _agent_memory_record,
        _agent_memory_record_json,
        append_agent_memory!,
        load_agent_memory_store
    _memory_test_hash(value) = LifeAI._sha256_hex(value)
else
    using JSON3
    struct Qwen3SemanticMemory end
    _sha256_hex(value::AbstractString) = bytes2hex(SHA.sha256(value))
    include(joinpath(@__DIR__, "..", "src", "agent", "memory.jl"))
    _memory_test_hash(value) = _sha256_hex(value)
end

function _memory_test_row(; sequence=1, text="record")
    return Dict{String,Any}(
        "schema_version" => 1,
        "id" => "record-1",
        "sequence" => sequence,
        "text" => text,
        "text_sha256" => _memory_test_hash(text),
        "metadata" => Any[],
    )
end

@testset "memory resource and persistence contracts" begin
    mktempdir() do directory
        path = joinpath(directory, "memory.jsonl")
        store = load_agent_memory_store(path; create=true)
        @test_throws ArgumentError append_agent_memory!(
            store, "x"^(MAX_AGENT_MEMORY_TEXT_BYTES + 1),
        )
        @test_throws ArgumentError append_agent_memory!(
            store,
            "metadata entries";
            metadata=Dict("key-$index" => "value" for index in 1:(MAX_AGENT_MEMORY_METADATA_ENTRIES + 1)),
        )
        @test_throws ArgumentError append_agent_memory!(
            store,
            "metadata key";
            metadata=Dict("k"^(MAX_AGENT_MEMORY_METADATA_KEY_BYTES + 1) => "value"),
        )
        @test_throws ArgumentError append_agent_memory!(
            store,
            "metadata value";
            metadata=Dict("key" => "v"^(MAX_AGENT_MEMORY_METADATA_VALUE_BYTES + 1)),
        )
        @test !ispath(path)
        @test !store.poisoned
    end

    @test_throws ArgumentError _agent_memory_record(
        _memory_test_row(sequence=big(typemax(Int)) + 1), 1,
    )
    @test_throws ArgumentError _agent_memory_record(_memory_test_row(sequence=true), 1)
    @test_throws ArgumentError _agent_memory_record(_memory_test_row(sequence=0), 1)
    primitive_metadata = _memory_test_row()
    primitive_metadata["metadata"] = Any[1]
    @test_throws ArgumentError _agent_memory_record(primitive_metadata, 1)

    mktempdir() do directory
        record = AgentMemoryRecord(
            "duplicate-json", 1, "duplicate keys", Dict("source" => "test"),
            _memory_test_hash("duplicate keys"),
        )
        encoded = _agent_memory_record_json(record)
        top_duplicate = replace(
            encoded,
            "\"id\":\"duplicate-json\"" =>
                "\"id\":\"duplicate-json\",\"id\":\"duplicate-json\"",
        )
        nested_duplicate = replace(
            encoded,
            "\"key\":\"source\"" => "\"key\":\"source\",\"key\":\"source\"",
        )
        for (name, contents) in (
            ("duplicate-top.jsonl", top_duplicate),
            ("duplicate-nested.jsonl", nested_duplicate),
        )
            path = joinpath(directory, name)
            write(path, contents * "\n")
            chmod(path, 0o600)
            @test_throws ArgumentError load_agent_memory_store(path)
        end

        secret = "must-not-appear-in-a-parser-diagnostic"
        malformed_cases = (
            "{\"$secret\":",
            "{\"$secret\":1,",
            "[\"$secret\",",
        )
        for (index, contents) in enumerate(malformed_cases)
            path = joinpath(directory, "malformed-$index.jsonl")
            write(path, contents * "\n")
            chmod(path, 0o600)
            error = try
                load_agent_memory_store(path)
                nothing
            catch caught
                caught
            end
            @test error isa ArgumentError
            @test !occursin(secret, sprint(showerror, error))
        end
    end

    mktempdir() do directory
        path = joinpath(directory, "oversized.jsonl")
        open(path, "w") do io
            truncate(io, MAX_AGENT_MEMORY_JOURNAL_BYTES + 1)
        end
        chmod(path, 0o600)
        @test_throws ArgumentError load_agent_memory_store(path)
    end

    mktempdir() do directory
        path = joinpath(directory, "oversized-line.jsonl")
        write(path, "x"^(MAX_AGENT_MEMORY_RECORD_BYTES + 1) * "\n")
        chmod(path, 0o600)
        @test_throws ArgumentError load_agent_memory_store(path)
    end

    mktempdir() do directory
        path = joinpath(directory, "missing-newline.jsonl")
        record = AgentMemoryRecord(
            "valid-record", 1, "valid JSON", Dict{String,String}(), _memory_test_hash("valid JSON"),
        )
        encoded = _agent_memory_record_json(record)
        write(path, encoded)
        chmod(path, 0o600)
        @test_throws ArgumentError load_agent_memory_store(path)
    end

    mktempdir() do directory
        path = joinpath(directory, "projected-limit.jsonl")
        open(path, "w") do io
            truncate(io, MAX_AGENT_MEMORY_JOURNAL_BYTES - 8)
        end
        chmod(path, 0o600)
        opened = stat(path)
        store = AgentMemoryStore(
            path,
            AgentMemoryRecord[],
            Dict{String,Int}(),
            opened.device,
            opened.inode,
            opened.size,
            false,
        )
        @test_throws ArgumentError append_agent_memory!(store, "would exceed limit"; id="limit")
        @test stat(path).size == MAX_AGENT_MEMORY_JOURNAL_BYTES - 8
        @test isempty(store.records)
        @test !store.poisoned
    end

    if Sys.isunix()
        mktempdir() do directory
            path = joinpath(directory, "permissions.jsonl")
            write(path, "")
            chmod(path, 0o644)
            @test_throws ArgumentError load_agent_memory_store(path)
            chmod(path, 0o600)
            @test isempty(load_agent_memory_store(path).records)
        end

        mktempdir() do directory
            path = joinpath(directory, "linked.jsonl")
            alias = joinpath(directory, "alias.jsonl")
            write(path, "")
            chmod(path, 0o600)
            Base.Filesystem.hardlink(path, alias)
            @test_throws ArgumentError load_agent_memory_store(path)
        end
    end
end
