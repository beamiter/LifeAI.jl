using Test,LifeAI,Random

function ripple_int4_gemv_checks(device)
    rng=Xoshiro(53)
    for (rows,width,group) in ((279,8,4),(65,128,128),(33,5120,128),(35,40960,128))
        weight=randn(rng,Float32,rows,width)./sqrt(Float32(width))
        input=randn(rng,Float32,width,1)
        packed=LifeAI._quantize_linear(weight,LinearQuantizationSpec(:int4;group))
        mapped=LifeAI._flint_weight_to_device(device,packed)
        expected=LifeAI._flint_linear(packed,input)
        actual=Array(LifeAI._ripple_linear(mapped,device(input)))
        @test actual ≈ expected atol=3f-5 rtol=3f-5
        # Prefill keeps the existing weight-only Float32 GEMM equations.
        batch=repeat(input,1,2)
        @test Array(LifeAI._ripple_linear(mapped,device(batch))) ≈
            LifeAI._flint_linear(packed,batch) atol=3f-5 rtol=3f-5
    end
end
