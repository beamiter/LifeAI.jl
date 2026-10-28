# Flint on the DSW workspace

Implementation: `src/io/hf_flint.jl`, `src/data/hf_flint_tokenizer.jl`,
`src/models/flint.jl`, and `src/models/flint_quantized.jl`.
The public API is exported from `src/api.jl`.

The DSW environment already provides Julia and its dependency depot:

```bash
cd /jfs/application/user/bgl/LifeAI.jl
export JULIA_DEPOT_PATH=/mnt/public_data/bgl/envs/lifeai/depot
LIFEAI_JULIA=/mnt/public_data/bgl/envs/lifeai/julia-1.12.6/bin/julia
```

Use `-O1` on this host. The default optimization level stalled in Julia's
`jl_compile_codeinst_now` during validation; the `-O1` native compilation smoke
passed. `--compile=min` works for CPU reference checks but is unsuitable for
CUDA because CUDA requires `llvmcall` compilation. This workaround changes
only the process command, not the shared Julia installation or preferences.

After obtaining the original checkpoint on a filesystem with enough space:

```bash
# The destination must not already exist.
"$LIFEAI_JULIA" -O1 --project=. scripts/quantize_flint.jl \
  /path/to/flint /path/to/flint-int4 int4 128

"$LIFEAI_JULIA" -O1 --project=. scripts/run_flint.jl \
  /path/to/flint-int4 --device cuda --prompt '写一个 Julia 二分查找函数。' \
  --temperature 0 --max-new-tokens 128
```

The Float32 reference and packed INT8/INT4/NVFP4 paths share the same Flint equations.
Packed matrices are reconstructed one projection at a time. This reduces
stored-weight memory; there are no fused low-bit GEMM kernels or measured
full-model throughput claims. Router, embeddings, norms, and attention sinks
remain floating point. The estimated stored parameter payload for the official
config is 20.76 GiB for INT4/group128 and 38.12 GiB for INT8, excluding runtime
buffers, allocations and KV cache.

Validation scripts and artifacts:

- `test/flint/test_flint.jl`: independent NumPy logits, cached/full forward,
  sliding-window boundaries, generation, and quantized logits.
- `test/test_flint_io.jl`: checkpoint validation, tokenizer, text decoding.
- `test/test_flint_quantization.jl`: quantization and artifact checksums.
- `scripts/verify_flint_cuda.jl`: float/INT8/INT4 CPU-CUDA parity and generation.
- `benchmark_results/flint/`: DSW run logs and tiny synthetic CLI artifacts.

Fixture weights are synthetic. On 2026-10-28 the complete checkpoint was
downloaded to `/jfs/application/user/bgl/models/RecursiveSI/flint` at revision
`2a247be0c0aec895ccb0ac7ca92229ec6df37288`: all 37 upstream file hashes and
36,734 tensor shapes passed verification. New code, models and logs use the
JuiceFS workspace, including the migrated Julia runtime/depot. The old
`/mnt/public_data/bgl` path is now a compatibility symlink to that workspace.

## NVFP4

The NVFP4 mode uses rowwise 16-element blocks, packed E2M1 data, E4M3FN block
scales and one Float32 dequantization scale per weight tensor. The data uses
nearest-even rounding. Partial final blocks are zero padded and cropped back
to the original input width during inference. Norms, routers, embeddings and
attention sinks retain floating-point weights.

```bash
"$LIFEAI_JULIA" -O1 --project=. scripts/quantize_flint.jl \
  /path/to/flint /path/to/flint-nvfp4 nvfp4
"$LIFEAI_JULIA" -O1 --project=. scripts/run_flint.jl \
  /path/to/flint-nvfp4 --device cuda --temperature 0 --max-new-tokens 128
```

Omitting `group` selects 16 for NVFP4 and retains 128 for INT4/INT8. NVFP4
rejects any group other than 16 and currently supports `calibration=:maxabs`
only. Artifacts written with the new code use format version 2; the reader
still accepts version 1 artifacts. NVFP4 tensors cannot be labeled version 1.

NVFP4 reference inference reconstructs the active projection as Float32.
The native CUDA path prepares packed weights and swizzled scales for cuBLASLt,
quantizes activations, and executes FP4 GEMM with Float32 accumulation/output.
Use `--nvfp4-backend native` for the native path or `reference` for W4A16.
`auto` (the CLI default) chooses native for CUDA NVFP4 weights, reference for CPU.
Native weights retain the portable source plus a GPU preparation cache; call
`flint_native_fp4_stats(bundle)` to inspect added storage and native call count.
The bounded GEMM workspace is shared across weights in one CUDA context.

```bash
"$LIFEAI_JULIA" -O1 --project=. scripts/run_flint.jl \
  /path/to/flint-nvfp4 --device cuda --nvfp4-backend native --prompt '你好'
"$LIFEAI_JULIA" -O1 --project=. scripts/verify_flint_nvfp4_native.jl
"$LIFEAI_JULIA" -O1 --project=. scripts/benchmark_flint_nvfp4_native.jl --quick
"$LIFEAI_JULIA" -O1 --project=. scripts/profile_flint_nvfp4_native.jl
```

Native activation quantization makes this W4A4 arithmetic, unlike reference
W4A16. Batch/chunk boundaries affect the activation global scale. Benchmark
results include quantization/padding overhead and must not be interpreted as
full-model performance. The portable format remains unchanged.

## Resident service and terminal UI

`scripts/run_flint_server.jl` holds one native NVFP4 bundle across requests.
CUDA loading uses `quantize_on_device=true`: BF16 source projections retain
their bits through host reading/upload, then are quantized on the GPU.
`on_progress(name,index,total)` reports loading/preparation progress, and
`generate_flint_tokens(...; on_token=callback)` supports incremental output and
cooperative cancellation when the callback returns `false`.

The loopback worker provides `/healthz` and `/api/chat`. It validates token
budgets, serializes generation, preserves UTF-8 bytes across token boundaries,
and reports native GEMM calls in the final NDJSON packet. Its default limits
are 2048 total context tokens and 512 generated tokens.
The CUDA/native worker warms prefill and decode before publishing readiness;
`--no-warmup` is available for startup diagnostics. Final packets include the
per-request native call delta separately from the cumulative count.
The optimized single-token path keeps activation scaling on-device and batches
GQA attention heads. Final packets separately publish `prompt_eval_duration`,
`decode_duration`, and `lifeai.decode_tokens_per_second`; `eval_duration` retains
its previous total-generation meaning. See [performance analysis](flint_performance.md)
for the measured 1.24 to 18.02 tok/s improvement, validation and remaining gaps.

The [Lumen Rust terminal](../tools/tui/README.md) uses `lifeai chat` or `lumen`
and keeps a separate history and draft per model. The old `qwen` command and
Python TUI have been removed. The [router](../tools/console/README.md)
serves Qwen3-8B and Flint through one endpoint. Model switching
waits for the previous worker to exit before starting another GPU process.
The deployed controller is
`/jfs/application/user/bgl/deployments/lifeai_console/control.py`.
