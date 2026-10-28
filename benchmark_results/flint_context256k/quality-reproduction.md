Run these commands from the repository root. The generators require Python
`tokenizers` and the released model's tokenizer assets. Use an external corpus
directory: generated text and token-ID binaries are intentionally excluded from Git.

```sh
python3 benchmark_results/flint_context256k/create_quality_corpus.py --tokenizer-dir /path/to/flint --output-dir /path/to/quality
python3 benchmark_results/flint_context256k/create_quality_2048.py --tokenizer-dir /path/to/flint --corpus-dir /path/to/quality
julia --project=. benchmark_results/flint_context256k/audit_quality_lifeai.jl /path/to/flint /path/to/quality
julia --project=. benchmark_results/flint_context256k/audit_quality_lifeai.jl /path/to/flint /path/to/quality quality-long-2048-manifest.json quality-long-2048-lifeai-audit.json
```

The first manifest preserves four 512-output-budget cases: repeated8K,
varied8K, varied32K, and varied256K. Varied records have unique indices and
deterministic combinations of ordinary subjects, locations, objects, weather,
and counts. The additional manifest links the original manifest's hash and
contains a separate 260096-prompt-token case with 2048 reserved output tokens.
Both manifests record every input hash and the three needle positions. The
audits compare all LifeAI token IDs with official tokenizer IDs shifted by one.

After the service is ready, run a controlled 512-budget case:

```sh
python3 benchmark_results/flint_context256k/benchmark_quality.py --corpus-dir /path/to/quality --budget 8192 --style varied --label activation-profile --output-dir /path/to/result
```

The expanded output case requires the service to publish an output limit of
at least2048; the driver checks published limits before submitting:

```sh
python3 benchmark_results/flint_context256k/benchmark_quality.py --corpus-dir /path/to/quality --manifest /path/to/quality/quality-long-2048-manifest.json --budget 262144 --style varied --label activation-profile --output-dir /path/to/full-result
```

`quality-summary.json` validates the entire final JSON object after removing
a completed leading think block and an optional single JSON code fence. It
requires exactly the three expected keys and values. An unfinished think
block, non-JSON text, or extra keys fails. Raw substring presence in reasoning
is reported separately in baseline evidence and does not establish a pass.
The output count is a budget; EOS may stop before it is exhausted.
