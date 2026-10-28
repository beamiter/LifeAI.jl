Reproduce the exact 262144-token context boundary with the released Flint
tokenizer. Generated text and binary token IDs are excluded from Git.

```sh
python3 create_corpus.py --tokenizer-dir /path/to/flint --output-dir /path/to/corpus
julia --project=../.. audit_lifeai.jl /path/to/flint /path/to/corpus
python3 benchmark_http.py --prompt /path/to/corpus/flint-261632-prompt-512-output.txt --dry-run
```

The creator requires Python `tokenizers`. `corpus-manifest.json` records the
official counts, file hashes, marker offsets, and expected answer.
`lifeai-tokenization-audit.json` confirms every token ID using LifeAI's actual
`encode` and `apply_flint_chat_template`. `reference-asset-hashes.json` records
the released assets verified against the deployed model copy.

The two budgets are 261632 prompt tokens plus **512 reserved output tokens**,
and 262143 plus one. Generation can stop early at EOS. Identifiers near the
beginning, middle, and end support a retrieval diagnostic; repeated neutral
text is a capacity workload and does not establish general answer quality.

Once the service publishes a ready Flint context of 262144, check the overflow
boundary without inference:

```sh
python3 check_http_boundary.py --prompt /path/to/corpus/flint-262143-prompt-1-output.txt
```

The script refuses to POST while either service publishes another context
limit. It sends 262143 prompt tokens with two reserved output tokens, checks
the worker's structured `context_length_exceeded` HTTP400 and the router's
HTTP400, and confirms the router releases its reservation.

Run actual streaming acceptance separately:

```sh
python3 benchmark_http.py --prompt /path/to/corpus/flint-261632-prompt-512-output.txt --output-dir /path/to/run
```

The driver defaults to loopback11435, temperature0, streaming, and a 7200-second
deadline. It records prefill counts, time to first visible text, total time,
final packet, answer, identifier matches, and GPU memory sampled every two
seconds. GPU memory covers all resident processes. UTF-8 buffering can make
the first visible text arrive slightly after the worker's first token.

`test_benchmark_http.py` exercises transport and measurements against a local
mock HTTP server. Set `FLINT_CORPUS_DIR=/path/to/corpus` before running it, or
run its staged copy alongside the external corpus. It performs no GPU work.

`quality-reproduction.md` describes the varied-text corpus, the separate
260096+2048 budget, strict final-JSON evaluation and portable commands.
`quality-controlled-comparison.json` compares exactly the same 8K/32K prompts
under global and per-token activation scaling; each case is one measurement.
The repeated8K case exhausts its 512-token budget in an unfinished think block
in both modes; raw identifier mentions are not counted as a completed answer.

`attention.jl`, `attention-results.json` and `splitk-attention-results.json`
record bounded FP32 attention. `attention-random256k-results.json` adds the
independent random256K comparison; `attention-tests.log` records108 checks.
`nvfp4-token-results.json` and `nvfp4-token-tests.log` cover token scaling,
the projection microbenchmark and2313 numerical/lifecycle regression checks.
Projection timing is a synthetic diagnostic and is not full-model throughput.

`summary.json` records the final deployment and validation scope.
`full-token-http.json` and `full-token-quality.json` record the successful
260096-input +2048-output-budget request and strict finalJSON pass for all3IDs.
The actual output is214tokens/EOS, clientTTFT885.35s, decode9.73tok/s and peak
sampledGPUmemory66123MiB. `full-http.json` preserves the earlier repeated-text
global-scaling capacity run that failed retrieval; it is not a quality pass.
