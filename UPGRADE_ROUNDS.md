# Additional Upgrade Rounds

This ledger records the twenty follow-up rounds implemented on top of the
existing uncommitted hardening work. Each round changes runtime behavior or a
public contract and has a focused regression in `test/test_sampling_core.jl`,
`test/test_diffusion_sampling.jl`, or `test/test_memory_core.jl`.

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

The repository's full `Pkg.test()` remains blocked before test execution by
the current Manifest/dependency resolution combination. The three focused test
files are intentionally independently loadable so these contracts remain
verifiable in that environment.
