module LifeAINCCLExt

using CUDA,NCCL
using BFloat16s:BFloat16
import LifeAI

mutable struct RippleNCCLBackend <: LifeAI.AbstractRippleTPBackend
    devices::Vector{CUDA.CuDevice}
    streams::Vector{CUDA.CuStream}
    communicators::Vector{NCCL.Communicator}
    reductions::Threads.Atomic{Int}
    gathers::Threads.Atomic{Int}
    failure_lock::ReentrantLock
    closed::Bool
end

function LifeAI.ripple_nccl_backend(devices::AbstractVector)
    devs=CUDA.CuDevice[d isa CUDA.CuDevice ? d : CUDA.CuDevice(d) for d in devices]
    !isempty(devs) && length(unique(devs))==length(devs) || throw(ArgumentError("Ripple TP devices must be nonempty and unique"))
    Threads.nthreads()>=length(devs) || throw(ArgumentError("Ripple TP needs at least one Julia thread per GPU; set JULIA_NUM_THREADS"))
    communicators=NCCL.Communicators(devs)
    streams=CUDA.CuStream[CUDA.device!(()->CUDA.CuStream(),dev) for dev in devs]
    return RippleNCCLBackend(devs,streams,communicators,Threads.Atomic{Int}(0),Threads.Atomic{Int}(0),ReentrantLock(),false)
end

LifeAI._ripple_tp_world(b::RippleNCCLBackend)=length(b.devices)
function LifeAI._ripple_tp_with_rank(f,b::RippleNCCLBackend,rank)
    b.closed && throw(InvalidStateException("Ripple NCCL backend is closed",:closed))
    CUDA.device!(b.devices[rank+1]) do
        CUDA.stream!(b.streams[rank+1]) do
            f()
        end
    end
end
LifeAI._ripple_tp_root!(b::RippleNCCLBackend)=CUDA.device!(b.devices[1])
LifeAI._ripple_tp_finish_rank(b::RippleNCCLBackend,rank)=CUDA.synchronize(b.streams[rank+1])

function LifeAI._ripple_tp_abort!(b::RippleNCCLBackend,error)
    lock(b.failure_lock) do
        b.closed && return
        b.closed=true
        for (index,comm) in enumerate(b.communicators)
            try
                CUDA.device!(b.devices[index]) do
                    NCCL.abort(comm)
                    # NCCL 0.1.3 does not clear the handle in abort, but its
                    # finalizer subsequently calls destroy on non-null handles.
                    setfield!(comm,:handle,typeof(getfield(comm,:handle))(C_NULL))
                end
            catch
            end
        end
    end
    return
end

function LifeAI._ripple_tp_run_ranks(f,b::RippleNCCLBackend)
    b.closed && throw(InvalidStateException("Ripple NCCL backend is closed",:closed))
    results=Vector{Any}(undef,length(b.devices))
    @sync for rank in 0:length(b.devices)-1
        Threads.@spawn try
            results[rank+1]=f(rank)
        catch error
            LifeAI._ripple_tp_abort!(b,error)
            rethrow()
        end
    end
    return results
end

function LifeAI._ripple_tp_upload(b::RippleNCCLBackend,rank,raw,dtype,shape,target)
    source_type=dtype=="BF16" ? BFloat16 : dtype=="F32" ? Float32 : throw(ArgumentError("unsupported Ripple TP source dtype"))
    uploaded=GC.@preserve raw begin
        host=unsafe_wrap(Array,Ptr{source_type}(pointer(raw)),Tuple(reverse(shape));own=false)
        value=CUDA.CuArray(host)
        CUDA.synchronize(b.streams[rank+1])
        value
    end
    converted=target===source_type ? uploaded : target.(uploaded)
    output=length(shape)>1 ? permutedims(converted,Tuple(reverse(1:length(shape)))) : converted
    CUDA.synchronize(b.streams[rank+1])
    output===converted || CUDA.unsafe_free!(converted)
    converted===uploaded || CUDA.unsafe_free!(uploaded)
    return output
end

function LifeAI._ripple_tp_allreduce!(b::RippleNCCLBackend,rank,value)
    NCCL.Allreduce!(value,+,b.communicators[rank+1];stream=b.streams[rank+1])
    Threads.atomic_add!(b.reductions,1)
    return value
end
function LifeAI._ripple_tp_allgather(b::RippleNCCLBackend,rank,value)
    world=length(b.devices);rows,tokens=size(value)
    gathered=similar(value,Float32,(length(value)*world,))
    NCCL.Allgather!(value,gathered,b.communicators[rank+1];stream=b.streams[rank+1])
    Threads.atomic_add!(b.gathers,1)
    # NCCL concatenates rank-major buffers. Reorder into (global vocab,tokens).
    return reshape(permutedims(reshape(gathered,rows,tokens,world),(1,3,2)),rows*world,tokens)
end

# One host thread submits each complete group. All allocations and layout
# conversion happen outside it; every rank is enqueued before synchronization.
function _collective_group(f,b)
    try
        NCCL.group() do
            for rank in 0:length(b.devices)-1
                LifeAI._ripple_tp_with_rank(b,rank) do
                    f(rank)
                end
            end
        end
        for rank in 0:length(b.devices)-1
            LifeAI._ripple_tp_with_rank(b,rank) do
                LifeAI._ripple_tp_finish_rank(b,rank)
            end
        end
    catch error
        LifeAI._ripple_tp_abort!(b,error)
        rethrow()
    end
    return
end
function LifeAI._ripple_tp_allreduce_batch!(b::RippleNCCLBackend,values)
    GC.@preserve values begin
        _collective_group(b) do rank
            LifeAI._ripple_tp_allreduce!(b,rank,values[rank+1])
        end
    end
    return values
end
function LifeAI._ripple_tp_allgather_batch(b::RippleNCCLBackend,values)
    gathered=LifeAI._ripple_tp_compute_ranks(b) do rank
        similar(values[rank+1],Float32,(length(values[rank+1])*length(b.devices),))
    end
    GC.@preserve values gathered begin
        _collective_group(b) do rank
            NCCL.Allgather!(values[rank+1],gathered[rank+1],b.communicators[rank+1];stream=b.streams[rank+1])
            Threads.atomic_add!(b.gathers,1)
        end
    end
    return LifeAI._ripple_tp_compute_ranks(b) do rank
        rows,tokens=size(values[rank+1]);world=length(b.devices)
        reshape(permutedims(reshape(gathered[rank+1],rows,tokens,world),(1,3,2)),rows*world,tokens)
    end
end

function LifeAI.close_ripple_tp_backend!(b::RippleNCCLBackend)
    b.closed && return
    for rank in 0:length(b.devices)-1
        LifeAI._ripple_tp_with_rank(b,rank) do
            CUDA.synchronize(b.streams[rank+1])
        end
    end
    if applicable(NCCL.destroy,b.communicators)
        NCCL.destroy(b.communicators)
    else
        for (index,comm) in enumerate(b.communicators)
            CUDA.device!(()->NCCL.destroy(comm),b.devices[index])
        end
    end
    b.closed=true
    return
end

end
