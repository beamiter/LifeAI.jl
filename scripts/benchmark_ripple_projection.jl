using LifeAI,CUDA,NCCL,JSON3,LinearAlgebra,Random,Test,Statistics
using BFloat16s:BFloat16
CUDA.allowscalar(false);BLAS.set_num_threads(1);Random.seed!(71)
result_dir=isempty(ARGS) ? "benchmark_results/ripple/perf" : ARGS[1];mkpath(result_dir)
records=Any[]
@testset "BF16 direct projection numerical contract" begin
    for (rows,columns,tokens) in ((7,5,1),(33,129,1),(128,5120,1),(1280,5120,1),(5120,1280,1),(5120,5120,1),(12544,5120,1),(17,41,7))
        host=BFloat16.(randn(Float32,rows,columns)./sqrt(Float32(columns)));x=randn(Float32,columns,tokens)
        weight=CUDA.CuArray(host);input=CUDA.CuArray(x);fp32=Float32.(weight)
        prepared=LifeAI._ripple_bf16_projection(weight)
        reference=Array(fp32*input);actual=Array(LifeAI._ripple_linear(prepared,input))
        err=maximum(abs.(actual.-reference));rel=norm(actual-reference)/max(norm(reference),eps(Float32))
        @test actual≈reference atol=3f-5 rtol=3f-5
        @test LifeAI.quantized_parameter_bytes(prepared)==sizeof(host)
        timings=Dict{String,Float64}()
        for (name,projection) in (("reference",weight),("bf16_gemv",prepared),("fp32",fp32))
            for _ in 1:3;value=LifeAI._ripple_linear(projection,input);CUDA.synchronize();CUDA.unsafe_free!(value);end
            samples=Float64[]
            for _ in 1:9
                GC.gc(false)
                seconds=CUDA.@elapsed begin
                    value=LifeAI._ripple_linear(projection,input)
                end
                push!(samples,seconds);CUDA.unsafe_free!(value)
            end
            timings[name]=median(samples)
        end
        record=(rows=rows,columns=columns,tokens=tokens,max_absolute_error=err,relative_l2=rel,seconds=timings)
        push!(records,record);println(JSON3.write(record));flush(stdout)
        CUDA.unsafe_free!(weight);CUDA.unsafe_free!(input);CUDA.unsafe_free!(fp32);GC.gc(false)
    end
end
write(joinpath(result_dir,"projection.json"),JSON3.write((gpu=CUDA.name(CUDA.device()),cases=records)))
