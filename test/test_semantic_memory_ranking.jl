using Test
using Random
using LinearAlgebra
using LifeAI: Qwen3SemanticMemory, retrieve_qwen3_semantic_memory

@testset "Semantic memory partial ranking" begin
    rng = Xoshiro(78)
    values = randn(rng, Float32, 16, 256)
    memory = Qwen3SemanticMemory(["document-$i" for i in 1:256], values)
    query = randn(rng, Float32, 16)
    normalized = query / norm(query)
    scores = vec(transpose(normalized) * memory.embeddings)
    expected = sortperm(eachindex(scores); by=i -> (-scores[i], i))
    for k in (1, 5, 256)
        actual = retrieve_qwen3_semantic_memory(memory, query; top_k=k)
        @test [result.index for result in actual] == expected[1:k]
        @test [result.rank for result in actual] == collect(1:k)
    end
    tied = Qwen3SemanticMemory(["first", "second", "third"], ones(Float32, 2, 3))
    @test [result.index for result in retrieve_qwen3_semantic_memory(
        tied, Float32[1, 1]; top_k=2,
    )] == [1, 2]
end
