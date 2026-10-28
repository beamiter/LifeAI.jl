using Test, LifeAI, Random

function flint_kv_decode_reference(key,value,past,position,window)
    first_position=position
    if past !== nothing
        key=cat(past.key,key;dims=3)
        value=cat(past.value,value;dims=3)
        first_position=past.first_position
    end
    keep=window > 0 ? min(size(key,3),window-1) : size(key,3)
    first=size(key,3)-keep+1
    updated=(key=key[:,:,first:end],value=value[:,:,first:end],
        first_position=position+size(key,3)-(past === nothing ? 0 : size(past.key,3))-keep)
    return (key=key,value=value,first_position=first_position,updated=updated)
end

function flint_kv_decode_checks(device)
    rng=Xoshiro(2026102803)
    for window in (0,1,2,5,17,512)
        actual_past=nothing
        expected_past=nothing
        position=0
        # Enter via multi-token prefill, wrap the sliding ring repeatedly, grow
        # full attention past several allocation boundaries, and then re-enter
        # the general multi-token path before decoding again.
        singles=window == 0 ? 530 : window == 512 ? 1030 : 70
        chunks=[3;fill(1,singles);4;fill(1,23)]
        for count in chunks
            key=randn(rng,Float32,8,2,count)
            value=randn(rng,Float32,8,2,count)
            expected=flint_kv_decode_reference(key,value,expected_past,position,window)
            previous_key=actual_past === nothing ? nothing : Array(actual_past.key)
            previous_value=actual_past === nothing ? nothing : Array(actual_past.value)
            actual=LifeAI._flint_kv_append(device(key),device(value),actual_past,position,window)
            @test Array(actual.key) == expected.key
            @test Array(actual.value) == expected.value
            @test actual.first_position == expected.first_position
            @test Array(actual.updated.key) == expected.updated.key
            @test Array(actual.updated.value) == expected.updated.value
            @test actual.updated.first_position == expected.updated.first_position
            # A failed forward must be able to retry from its prior cache: the
            # append slot cannot alter its currently visible history.
            if actual_past !== nothing
                @test Array(actual_past.key) == previous_key
                @test Array(actual_past.value) == previous_value
            end
            actual_past=actual.updated
            expected_past=expected.updated
            position+=count
        end
    end

    # A discarded forward must be retryable with a different current token.
    # The first append may have written its free slot, but the committed prefix
    # must remain unchanged and retry metadata must not advance twice.
    for window in (0,5)
        key=randn(rng,Float32,4,2,7)
        value=randn(rng,Float32,4,2,7)
        initial=LifeAI._flint_kv_append(device(key),device(value),nothing,0,window)
        past=initial.updated
        original_key=Array(past.key)
        original_value=Array(past.value)
        discarded_key=randn(rng,Float32,4,2,1)
        retry_key=randn(rng,Float32,4,2,1)
        current_value=randn(rng,Float32,4,2,1)
        LifeAI._flint_kv_append(device(discarded_key),device(current_value),past,7,window)
        retry=LifeAI._flint_kv_append(device(retry_key),device(current_value),past,7,window)
        @test Array(past.key) == original_key
        @test Array(past.value) == original_value
        @test Array(retry.key) == cat(original_key,retry_key;dims=3)
        @test retry.updated.first_position == 8-size(retry.updated.key,3)
    end

    # No-past single-token calls also build a valid cache, including window=1
    # where the next call must retain no prior tokens.
    for window in (0,1,8)
        key=randn(rng,Float32,4,2,1)
        value=randn(rng,Float32,4,2,1)
        actual=LifeAI._flint_kv_append(device(key),device(value),nothing,31,window)
        @test Array(actual.key) == key
        @test Array(actual.value) == value
        @test actual.first_position == 31
        @test size(actual.updated.key,3) == (window == 1 ? 0 : 1)
    end
end

@testset "Flint KV append independent reference" begin
    flint_kv_decode_checks(identity)
end

if get(ENV,"LIFEAI_FLINT_KV_CUDA_TESTS","false") == "true"
    @eval using CUDA
    @testset "Flint persistent KV decode on CUDA" begin
        @test CUDA.functional()
        CUDA.allowscalar(false)
        flint_kv_decode_checks(CUDA.cu)
    end
end
