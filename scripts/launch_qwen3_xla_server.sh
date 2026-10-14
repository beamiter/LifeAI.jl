#!/usr/bin/env bash
# Set CPU code-generation options before Julia initializes or imports packages.
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
server_args=()
cpu_args=()
for argument in "$@"; do
    case "$argument" in
        -h|--help)
            cat <<'USAGE'
usage: launch_qwen3_xla_server.sh MODEL_DIR [options]

  --cpu-bf16-compat  disable CPU AVX-512-BF16 code generation before Julia starts
  --quiet-startup   omit detailed session loading stages
  --host HOST       bind address (default: 127.0.0.1)
  --port PORT       listen port (default: 11435)
  --profile PATH    deployment profile

Other options are forwarded to run_qwen3_xla_server.jl unchanged.
LIFEAI_JULIA_BIN may select a Julia executable. The CPU compatibility option
does not change GPU BF16 weights or computation.
USAGE
            exit 0
            ;;
        --cpu-bf16-compat)
            # Julia 1.12 / LLVM 18 can fail to lower vector BF16 conversion on
            # AVX-512-BF16 CPUs. GPU weights and XLA computation remain BF16.
            cpu_args=(--cpu-target=native,-avx512bf16)
            ;;
        *) server_args+=("$argument") ;;
    esac
done

exec "${LIFEAI_JULIA_BIN:-julia}" "${cpu_args[@]}" --startup-file=no \
    "--project=$repo_root" "$repo_root/scripts/run_qwen3_xla_server.jl" \
    "${server_args[@]}"
