# Additional Upgrade Rounds

This ledger records the forty-seven follow-up rounds implemented on top of the
existing hardening work. Each round changes runtime behavior or a public
contract and has a focused regression in the core contract files or the
Chapter 03 reproducible-training test.

1. Reject Boolean and platform-overflowing `top_k` inputs.
2. Keep exactly `top_k` candidates with deterministic original-index tie breaking.
3. Apply stable descending top-p filtering using the smallest sufficient nucleus.
4. Reject logits and temperatures that become invalid at Float32 sampling precision.
5. Cap CPU sampling inputs at one million vocabulary entries.
6. Provide an explicit O(steps) validator for every diffusion recurrence.
7. Add O(1) locally validated beta, alpha, and cumulative-alpha accessors.
8. Add the timestep-zero-aware previous cumulative-alpha accessor.
9. Add the exact DDPM posterior variance API.
10. Add DDPM posterior mean coefficients and array reconstruction.
11. Add deterministic DDIM stepping, including skipped timesteps and the clean endpoint.
12. Add min-SNR weights for epsilon, clean-sample, and velocity prediction.
13. Add validated P2 loss weighting.
14. Bound persistent-memory journals to 64 MiB.
15. Bound each encoded journal record/line to 1 MiB.
16. Bound text and metadata entry/key/value/aggregate sizes.
17. Bound a journal to 100,000 records.
18. Enforce strict JSON/schema types, reject duplicate object keys at every
    nesting level, and convert integers with explicit Bool/overflow rejection.
19. Reject appends whose exact encoded size would exceed the journal budget before writing.
20. Require Unix journals to be root/current-user owned, single-linked, and private.
21. Reject Boolean, non-finite, and Float32-overflowing or underflowing learning rates.
22. Apply the same representability contract to global-gradient clipping limits and epsilon.
23. Reject non-finite host gradient norms before CPU clipping can poison parameters.
24. Require literal Boolean gradient-return flags, reject Boolean train-loop and
    checkpoint-progress counts, and prevent resume epoch/batch arithmetic overflow.
25. Validate every host evaluation NLL and detect aggregate token/NLL overflow.
26. Accumulate evaluation NLL with deterministic compensated summation and reject
    Boolean or platform-overflowing byte denominators.
27. Reject loss, perplexity, and byte-normalized metrics that would silently
    overflow their public Float32 representation.
28. Reject exact target-byte totals that overflow the platform integer before
    they can become a plausible but incorrect evaluation denominator.
29. Reject non-finite, negative (including values that underflow while
    narrowing), and Float32-overflowing training loss and callback gradient
    metrics before an optimiser update can mutate caller-owned state on either
    the host or XLA backend.
30. Refuse a training update before Lux can wrap a `typemax(Int)` step counter,
    and validate checkpoint step counters before saving or restoring them.
31. Validate and snapshot loader length, including the `typemax(Int)` boundary,
    without overflowing the permitted one-past-the-end resume position.
32. Reject nonzero loss and public evaluation metrics that underflow while
    narrowing to their documented Float64 or Float32 representation.
33. Require current checkpoint progress to contain non-negative epoch, batch,
    and step fields, and reject any step inconsistent with the restored state;
    legacy v1 payloads may fill only a missing progress step from that state.
    Progress with epoch zero may only use batch zero.
34. Reject Boolean or platform-overflowing checkpoint format versions and
    resume batches beyond the current loader, while treating exact end-of-loader
    progress as a completed epoch.
35. Detect Float64 underflow in mean and byte-normalized evaluation ratios before
    a nonzero aggregate can silently become an apparently exact zero.
36. Compute global gradient norms with scale normalization so squaring finite
    Float32 or integer leaves cannot first underflow or overflow, and calculate
    clipping ratios in Float64 before enforcing the public Float32 metric contract.
37. Align host and device Qwen3 MoE top-k routing at exact ties by selecting the
    highest remaining expert index, including all-tie and cutoff-tie regressions
    with and without selected-weight normalization.
38. Reject dense and MoE Qwen3 models with an LM-head bias before weight mapping,
    rather than returning an incomplete parameter tree that fails at first forward.
39. Prevent non-finite Qwen3 MoE routing columns from emitting expert index zero:
    host routing rejects them, while compact device routing keeps indices distinct
    and in bounds and preserves non-finite weights as an explicit poison signal.
40. Validate cache-free Qwen3-VL prompt layouts before decoder computation,
    enforcing the model context independently of caller caps and deriving mRoPE
    deltas from attention-valid coordinates; use a finite Float32 causal-mask
    sentinel so fully masked padding queries cannot poison valid-token logits.
41. Require host-representable, non-Boolean integers at shared Qwen prefill,
    decode, Qwen3-VL stop-id, and VL context-override boundaries, converting
    overflow failures into stable `ArgumentError`s before model computation.
42. Use stepwise checked arithmetic for Qwen3 sparse-MoE parameter counts and
    CUDA indexed-workspace bytes, rejecting multiplication or addition overflow
    before a wrapped negative size can reach allocation or capacity planning.
43. Stop Qwen3-VL generation from retaining prompt-sized input embeddings and
    final hidden states by default, while preserving full diagnostics behind the
    explicit `capture_prefill_states=true` compatibility option.
44. Push the lightweight Qwen3-VL policy into dynamic and static cached prefill,
    releasing the initial embedding reference during layer execution and applying
    final RMSNorm only to logits-bearing tail tokens when full hidden capture is off.
45. Harden dense, MoE, and VL Hugging Face scalar config contracts: reject
    Boolean integers/numerics, checked-convert host integers, preserve valid narrow
    integer overrides, and require positive norm/RoPE values at Float32 precision.
46. Preflight raw Qwen3-VL generation options before image/vision compute, then
    validate prompt-dependent context and static-capacity bounds before device
    transfer; normalize explicit capacities without leaking integer overflow.
47. Check every intermediate multiplication and addition in the Qwen3 MoE
    offload memory planner, rejecting wrapped model dimensions or byte totals
    instead of returning deceptively small deployment budgets.

The Julia 1.12.6 Manifest has been regenerated and is now tracked, and the CI
Julia 1.11/Project Julia 1.12 mismatch has been repaired. Full `Pkg.test()` now
imports the real `LifeAI` module before these files, preventing their standalone
fallbacks from polluting later episode tests. The three core focused files
remain independently loadable for fast contract checks; Chapter 03 exercises
the training/evaluation boundary through the real module, and the complete
default suite is the integration gate. CI instantiates the checked-in
`Project.toml`/`Manifest.toml` pair and rejects dependency-file drift after the
suite, so a registry update cannot silently change the validated environment.
