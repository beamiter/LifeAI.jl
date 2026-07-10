using Test
using LifeAI:
    DatasetLoader,
    fit_tokenizer,
    encode,
    num_samples,
    num_batches

@testset "DatasetLoader shifted next-token batches" begin
    token_ids = collect(1:10)

    loader = DatasetLoader(
        token_ids;
        seq_len=3,
        batch_size=2,
        stride=3,
    )

    @test num_samples(loader) == 3
    @test num_batches(loader) == 1
    @test length(loader) == 1

    x, y = loader[1]

    @test size(x) == (3, 2)
    @test size(y) == (3, 2)

    @test x == [
        1 4
        2 5
        3 6
    ]

    @test y == [
        2 5
        3 6
        4 7
    ]

    # 对每条样本来说，target 相对 input 向后移动一个 token。
    @test y[1:(end-1), :] == x[2:end, :]
end

@testset "DatasetLoader overlapping windows" begin
    token_ids = collect(1:7)

    loader = DatasetLoader(
        token_ids;
        seq_len=3,
        batch_size=2,
        stride=1,
        drop_last=false,
    )

    @test num_samples(loader) == 4
    @test num_batches(loader) == 2
    @test length(loader) == 2

    batches = collect(loader)

    @test length(batches) == 2

    x1, y1 = batches[1]

    @test x1 == [
        1 2
        2 3
        3 4
    ]

    @test y1 == [
        2 3
        3 4
        4 5
    ]

    x2, y2 = batches[2]

    @test x2 == [
        3 4
        4 5
        5 6
    ]

    @test y2 == [
        4 5
        5 6
        6 7
    ]

    @test y1[1:(end-1), :] == x1[2:end, :]
    @test y2[1:(end-1), :] == x2[2:end, :]
end

@testset "DatasetLoader keeps partial final batch" begin
    loader = DatasetLoader(
        collect(1:9);
        seq_len=3,
        batch_size=2,
        stride=2,
        drop_last=false,
    )

    # 合法窗口起点为 1、3、5，共 3 条样本。
    @test num_samples(loader) == 3

    # batch_size=2，因此应产生：
    # batch 1：2 条样本
    # batch 2：1 条样本
    @test num_batches(loader) == 2
    @test length(loader) == 2

    x1, y1 = loader[1]

    @test size(x1) == (3, 2)
    @test size(y1) == (3, 2)

    @test x1 == [
        1 3
        2 4
        3 5
    ]

    @test y1 == [
        2 4
        3 5
        4 6
    ]

    x2, y2 = loader[2]

    # 最后只剩 1 条样本，因此 batch 维度为 1。
    @test size(x2) == (3, 1)
    @test size(y2) == (3, 1)

    @test vec(x2) == [5, 6, 7]
    @test vec(y2) == [6, 7, 8]

    @test y2[1:(end-1), :] == x2[2:end, :]
end

@testset "DatasetLoader drops incomplete final batch" begin
    loader = DatasetLoader(
        collect(1:8);
        seq_len=3,
        batch_size=2,
        stride=2,
        drop_last=true,
    )

    # 合法窗口起点为 1、3、5，共 3 条样本。
    @test num_samples(loader) == 3

    # 第三个样本无法组成完整的 batch，因此被丢弃。
    @test num_batches(loader) == 1
    @test length(loader) == 1

    x, y = loader[1]

    @test size(x) == (3, 2)
    @test size(y) == (3, 2)

    @test x == [
        1 3
        2 4
        3 5
    ]

    @test y == [
        2 4
        3 5
        4 6
    ]

    @test_throws BoundsError loader[2]
end

@testset "DatasetLoader from tokenizer and text" begin
    text = "春眠不觉晓处处闻啼鸟"
    tokenizer = fit_tokenizer(text)

    loader = DatasetLoader(
        tokenizer,
        text;
        seq_len=4,
        batch_size=1,
        stride=1,
        drop_last=false,
    )

    ids = encode(tokenizer, text)
    x, y = loader[1]

    @test size(x) == (4, 1)
    @test size(y) == (4, 1)

    @test vec(x) == ids[1:4]
    @test vec(y) == ids[2:5]

    @test y[1:(end-1), :] == x[2:end, :]
end

@testset "DatasetLoader input validation" begin
    @test_throws AssertionError DatasetLoader(
        [1, 2, 3];
        seq_len=0,
    )

    @test_throws AssertionError DatasetLoader(
        [1, 2, 3];
        seq_len=2,
        batch_size=0,
    )

    @test_throws AssertionError DatasetLoader(
        [1, 2, 3];
        seq_len=2,
        stride=0,
    )

    @test_throws AssertionError DatasetLoader(
        [1, 2];
        seq_len=2,
    )

    @test_throws AssertionError DatasetLoader(
        [1, 0, 2];
        seq_len=2,
    )

    loader = DatasetLoader(
        [1, 2, 3, 4];
        seq_len=2,
    )

    @test_throws BoundsError loader[0]
    @test_throws BoundsError loader[length(loader)+1]
end