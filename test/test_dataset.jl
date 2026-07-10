using Test
using LifeAI: DatasetLoader, fit_tokenizer, encode, num_samples, num_batches

@testset "DatasetLoader shifted next-token batches" begin
    loader = DatasetLoader(collect(1:10); seq_len=3, batch_size=2, stride=3)

    @test num_samples(loader) == 3
    @test num_batches(loader) == 1
    @test length(loader) == 1

    x, y = loader[1]

    @test size(x) == (3, 2)
    @test size(y) == (3, 2)
    @test x == [1 4; 2 5; 3 6]
    @test y == [2 5; 3 6; 4 7]
    @test y[1:end-1, :] == x[2:end, :]
end

@testset "DatasetLoader overlapping windows" begin
    loader = DatasetLoader(collect(1:7); seq_len=3, batch_size=2, stride=1, drop_last=false)

    @test num_samples(loader) == 4
    @test num_batches(loader) == 2

    batches = collect(loader)
    @test length(batches) == 2
    @test batches[1][1] == [1 2; 2 3; 3 4]
    @test batches[1][2] == [2 3; 3 4; 4 5]
    @test batches[2][1] == [3 4; 4 5; 5 6]
    @test batches[2][2] == [4 5; 5 6; 6 7]
end

@testset "DatasetLoader partial and dropped batches" begin
    partial = DatasetLoader(collect(1:9); seq_len=3, batch_size=2, stride=2, drop_last=false)
    dropped = DatasetLoader(collect(1:9); seq_len=3, batch_size=2, stride=2, drop_last=true)

    @test num_samples(partial) == 3
    @test length(partial) == 2
    @test size(partial[2][1]) == (3, 1)
    @test size(partial[2][2]) == (3, 1)

    @test length(dropped) == 1
end

@testset "DatasetLoader from tokenizer and text" begin
    text = "春眠不觉晓处处闻啼鸟"
    tokenizer = fit_tokenizer(text)
    loader = DatasetLoader(tokenizer, text; seq_len=4, batch_size=1, stride=1, drop_last=false)

    ids = encode(tokenizer, text)
    x, y = loader[1]

    @test vec(x) == ids[1:4]
    @test vec(y) == ids[2:5]
end

@testset "DatasetLoader input validation" begin
    @test_throws AssertionError DatasetLoader([1, 2, 3]; seq_len=0)
    @test_throws AssertionError DatasetLoader([1, 2, 3]; seq_len=2, batch_size=0)
    @test_throws AssertionError DatasetLoader([1, 2, 3]; seq_len=2, stride=0)
    @test_throws AssertionError DatasetLoader([1, 2]; seq_len=2)
    @test_throws AssertionError DatasetLoader([1, 0, 2]; seq_len=2)

    loader = DatasetLoader([1, 2, 3, 4]; seq_len=2)
    @test_throws BoundsError loader[0]
    @test_throws BoundsError loader[length(loader) + 1]
end
