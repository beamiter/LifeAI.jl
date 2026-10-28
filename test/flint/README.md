# Flint numerical fixtures

Run the focused suite from the repository root:

```sh
julia --project=. test/flint/test_flint.jl
```

`fixtures/tiny` contains synthetic random weights, not a trained model. Its
independent NumPy reference implements the equations in the Apache-2.0
[Transformers Flint model](https://github.com/huggingface/transformers/blob/main/src/transformers/models/flint/modeling_flint.py).
Regenerate it with NumPy and safetensors installed:

```sh
python3 scripts/export_flint_reference.py
```

The fixture uses Q projection width 16 with hidden width 8, four query heads,
two KV heads, nonzero attention sinks, two dense layers, two MoE layers with
shared experts, and a three-token sliding window. The nine-token reference
therefore checks window eviction as well as full attention. Tests compare
both complete and incremental/chunked inference with the external reference,
then verify greedy decoding, independently quantized INT8/INT4 logits, streamed quantization,
serialization, and rejection of corrupted artifacts.
