# Optional physical-GPU test; CUDA/NCCL are loaded by the caller.
function ripple_nccl_transport_checks(devices)
    backend=ripple_nccl_backend(devices);n=length(devices)
    try
        # Match a full 1,057-token, 5,120-wide hidden-state allreduce, so this
        # exercises the large-message protocol as well as the tiny fixture.
        values=LifeAI._ripple_tp_compute_ranks(backend) do rank
            CUDA.fill(Float32(rank+1),5120,1057)
        end
        expected=Float32(n*(n+1)÷2)
        for pass in 1:3
            LifeAI._ripple_tp_allreduce_batch!(backend,values)
            errors=LifeAI._ripple_tp_compute_ranks(backend) do rank
                maximum(abs.(values[rank+1].-expected))
            end
            @test all(iszero,errors)
            expected*=n
            # Force host finalizers between complete collective groups.
            GC.gc(false)
        end
        @test backend.reductions[]==3*n
        backend.gc_interval=4;backend.decode_since_gc=0;before=backend.gc_runs
        for _ in 1:3;LifeAI._ripple_tp_before_forward!(backend,1);end
        @test backend.gc_runs==before
        LifeAI._ripple_tp_before_forward!(backend,1)
        @test backend.gc_runs==before+1
        LifeAI._ripple_tp_before_forward!(backend,32)
        @test backend.gc_runs==before+2
        @test LifeAI._ripple_tp_compute_ranks(rank->objectid(current_task()),backend)==objectid.(backend.workers)
    finally
        close_ripple_tp_backend!(backend)
    end
end
