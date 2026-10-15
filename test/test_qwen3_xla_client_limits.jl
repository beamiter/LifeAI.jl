module Qwen3XLAClientLimitsRegression
using Test
include(joinpath(@__DIR__, "..", "scripts", "qwen3_xla_client.jl"))

@testset "CLI discovers compiled service limits" begin
    received = String[]
    health = function(url; kwargs...)
        push!(received, url)
        HTTP.Response(200, JSON3.write((;
            ready=true, model="Qwen/Qwen3-8B", context_tokens=40960,
            max_new_tokens=40896)))
    end
    _, options = parse_args(["/tmp/model", "--prompt", "hello", "--max-new-tokens", "1000",
        "--endpoint", "http://127.0.0.1:11435/api/generate?x=1"])
    resolve_server_limits!(options; get_health=health)
    @test options[:context] == 40960
    @test received == ["http://127.0.0.1:11435/healthz"]
    options[:max_new_tokens] = 40897
    @test_throws r"exceeds server maximum 40896" resolve_server_limits!(options; get_health=health)
    options[:max_new_tokens] = 1000
    options[:context] = 4096
    @test_throws r"does not match" resolve_server_limits!(options; get_health=health)
    bad_health(url; kwargs...) = HTTP.Response(200, "{\"ready\":true,\"model\":\"Qwen/Qwen3-8B\"}")
    @test_throws r"context_tokens" resolve_server_limits!(options; get_health=bad_health)
    small_health(url; kwargs...) = HTTP.Response(200, JSON3.write((;
        ready=true, model="Qwen/Qwen3-8B", context_tokens=1000, max_new_tokens=1000)))
    options[:context] = nothing
    @test_throws r"no room for prompt" resolve_server_limits!(options; get_health=small_health)
end

end # module
