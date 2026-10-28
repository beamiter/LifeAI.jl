using Test, LifeAI, CUDA, Random

CUDA.functional() || error("Long Flint KV tests require a functional CUDA GPU")
CUDA.allowscalar(false)

function long_kv_reference(key,value,past,position,window)
    first = past === nothing ? position : past.first_position
    all_key = past === nothing ? key : cat(past.key,key;dims=3)
    all_value = past === nothing ? value : cat(past.value,value;dims=3)
    keep = window > 0 ? min(size(all_key,3),window-1) : size(all_key,3)
    from = size(all_key,3)-keep+1
    return (key=all_key,value=all_value,first_position=first,
        updated=(key=all_key[:,:,from:end],value=all_value[:,:,from:end],
            first_position=position+size(key,3)-keep))
end

@testset "Reserved full KV accepts chunked prefill without reallocating its prefix" begin
    rng=Xoshiro(20261028041)
    for window in (0,1,2,17,513)
        past=nothing; reference=nothing; position=0; reserved=nothing
        for count in (3,7,64,1,511,4,1,1,512,1,3)
            key=randn(rng,Float32,8,2,count)
            value=randn(rng,Float32,8,2,count)
            expected=long_kv_reference(key,value,reference,position,window)
            previous=past === nothing ? nothing : (key=Array(past.key),value=Array(past.value))
            actual=LifeAI._flint_kv_append(CUDA.cu(key),CUDA.cu(value),past,position,window;
                capacity_hint=2048)
            @test Array(actual.key)==expected.key
            @test Array(actual.value)==expected.value
            @test actual.first_position==expected.first_position
            @test Array(actual.updated.key)==expected.updated.key
            @test Array(actual.updated.value)==expected.updated.value
            @test actual.updated.first_position==expected.updated.first_position
            if past !== nothing
                @test Array(past.key)==previous.key
                @test Array(past.value)==previous.value
            end
            if window==0
                @test actual.updated.storage.capacity==2048
                reserved === nothing && (reserved=actual.updated.storage)
                @test actual.updated.storage === reserved
            end
            past=actual.updated;reference=expected.updated;position+=count
        end
    end
    key=CUDA.zeros(Float32,8,2,3)
    @test_throws ArgumentError LifeAI._flint_kv_append(key,key,nothing,0,0;capacity_hint=-1)
    @test_throws DimensionMismatch LifeAI._flint_kv_append(key,CUDA.zeros(Float32,8,2,2),nothing,0,0)
    @test_throws ArgumentError LifeAI._flint_kv_append(key,key,nothing,0,-1)
    # A reservation increase copies the visible prefix once, then stays stable.
    first=LifeAI._flint_kv_append(key,key,nothing,0,0;capacity_hint=32)
    old=Array(first.updated.key)
    second=LifeAI._flint_kv_append(key,key,first.updated,3,0;capacity_hint=128)
    @test second.updated.storage.capacity==128
    @test Array(first.updated.key)==old
    third=LifeAI._flint_kv_append(key,key,second.updated,6,0;capacity_hint=128)
    @test third.updated.storage === second.updated.storage
end

@testset "One actual-sized full KV layer reserves 256K Float32 tokens" begin
    # About 1 GiB total, matching one of Flint's fourteen full-attention layers.
    # Read only diagnostic slices; no reference allocation scales with N².
    capacity=262144;chunk=4096
    key=CUDA.ones(Float32,128,4,chunk)
    value=CUDA.fill(-1f0,128,4,chunk)
    past=nothing;storage=nothing
    for position in 0:chunk:capacity-chunk
        actual=LifeAI._flint_kv_append(key,value,past,position,0;capacity_hint=capacity)
        past=actual.updated
        storage === nothing && (storage=past.storage)
        @test past.storage === storage
        @test size(past.key,3)==position+chunk
    end
    CUDA.synchronize()
    @test sizeof(storage.key)+sizeof(storage.value)==1024^3
    @test size(past.key)==(128,4,262144)
    @test all(==(1f0),Array(past.key[1:2,1:2,[1,131072,262144]]))
    @test all(==(-1f0),Array(past.value[1:2,1:2,[1,131072,262144]]))
    @test past.first_position==0
end
