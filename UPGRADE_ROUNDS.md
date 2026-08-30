# Additional Upgrade Rounds

This ledger records the ninety-five follow-up rounds implemented on top of the
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
48. Require Qwen3 MoE offload expert allowances to cover at least the model's
    per-token routing width, preventing impossible top-k working-set estimates.
49. Align the Qwen3 MoE offload planner with its executable session contract by
    rejecting unsupported multi-batch estimates instead of advertising them.
50. Compute Qwen3 dense parameter counts with arbitrary-precision intermediates
    and reject totals that cannot be represented by the public host `Int` API.
51. Apply the same checked public-count boundary to Qwen3 MoE checkpoint
    formulas while preserving the exact official 30B-A3B parameter total.
52. Make Qwen3 embedding parameter accounting overflow-safe with exact
    intermediates and a checked host-integer return boundary.
53. Count the independent Qwen3-VL language-model head for untied text specs,
    keeping parameter totals consistent with the expected tensor-shape contract.
54. Check Qwen3-VL text, vision, and combined parameter-count ranges with exact
    arithmetic, including patch and spatial-merge dimension squares.
55. Strictly normalize Qwen3 KV-cache estimator integers and model dimensions,
    rejecting Boolean or unrepresentable inputs before checked byte arithmetic.
56. Apply strict host-integer normalization to every Qwen3 MoE offload planner
    input while retaining representable narrow and arbitrary-width integers.
57. Normalize every Qwen3 XLA window integer and check rounded-bucket arithmetic,
    preventing Boolean coercion, conversion leaks, and host-width overflow.
58. Preflight dense Qwen3 BF16 session windows and variants before tokenizer,
    model-weight, or device work, and reuse the same validation during init.
59. Preflight Qwen3 MoE streamed-session numeric, strategy, and cross-option
    contracts before config I/O, RoPE construction, or resident tensor loading.
60. Harden live Qwen3 MoE expert-cache reconfiguration integers so invalid
    Boolean or oversized values fail before cache or buffer-pool mutation.
61. Skip Qwen3-VL pixel transfer and vision-tower execution when raw generation
    explicitly requests zero output tokens, while preserving prompt metadata.
62. Preflight dense Qwen3 config/model/bundle sequence limits, variants, and
    weight dtypes before JSON, tokenizer, model-construction, or weight I/O.
63. Preflight Qwen3 MoE config/model sequence limits and weight dtypes before
    JSON parsing, RoPE construction, or checkpoint I/O.
64. Derive Qwen3-VL tensor-oracle dimensions with exact arithmetic, check each
    shape product, and accumulate checkpoint parameters without host overflow.
65. Derive and accumulate Qwen3-VL BF16 payload bytes with exact arithmetic,
    rejecting byte counts that overflow even when their parameter totals fit.
66. Preflight frozen Qwen3-VL context requests before config reads or the
    checkpoint verifier hashes its multi-gigabyte asset set.
67. Strictly normalize HuggingFace token and vocabulary integers before the
    zero-to-one-based conversion, rejecting Boolean, overflow, and add-one wrap.
68. Reject unsupported safetensors target dtypes before directory selection,
    index parsing, or shard access at every loading entry point.
69. Strictly normalize Qwen3 XLA prompt and padding ids in both the padding
    helper and public generation entry before compiled execution.
70. Strictly normalize reusable Qwen3 BF16 session prompt tokens before reset
    or cache mutation while preserving valid integer widths.
71. Strictly normalize reusable Qwen3 BF16 single-step decode tokens before
    model execution or session-position mutation.
72. Check MultiHeadAttention query/KV dimensions and complete projection/QK
    parameter totals before Qwen model layer construction.
73. Compute estimated and realized Qwen3 quantized tensor bytes with exact
    intermediates, including nested-tree aggregation and public-range checks.
74. Strictly normalize dense and XLA Qwen3 generation stop-id sets before
    zero-output resets, prefill, compiled execution, or session mutation.
75. Strictly normalize dense Qwen3 generation output lengths before context
    arithmetic, zero-output reset, or prefill execution.
76. Validate every Qwen3 XLA device-to-host generated token as one exact,
    host-representable, in-vocabulary integer before results or callbacks.
77. Commit Qwen3 XLA session positions after each successful compiled cache
    mutation and align token callbacks with the dense one-argument contract.
78. Preflight Qwen3 XLA session context, chunk, strategy, and sampling-width
    options before tokenizer, compact-weight, or Reactant work.
79. Give the compact Qwen3 bundle explicit model options and preflight them
    before tokenizer I/O while preserving valid tokenizer-first loading.
80. Preflight frozen Qwen3 embedding context limits and weight dtypes before
    checksum, tokenizer, or checkpoint I/O while preserving revision priority.
81. Fail closed on non-integer or host-overflowing tokens injected by Qwen3 XLA
    service encoders, streaming callbacks, and generator results.
82. Strictly normalize Qwen3 XLA service capacity integers before invoking its
    expensive session loader, including its Int32 device-position limit, while
    preserving valid cross-width inputs.
83. Translate Qwen3 XLA request-option integer overflow into stable 400 errors
    before prompt encoding, request metrics, or generation work.
84. Normalize dense Qwen3 sampled-generation options before prefill can mutate
    the reusable K/V cache, including Float32 representability checks.
85. Reuse strict sampled-option preflight in Qwen3 XLA host sampling before any
    compiled cache mutation or session-position reset.
86. Preflight complete GPTModel parameter counts with exact arithmetic and
    count every Qwen topology without initializing parameter arrays.
87. Strictly normalize Qwen3 MoE offload prefill and decode tokens before
    resetting positions, expert-cache metrics, or checkpoint traffic.
88. Normalize Qwen3-VL generation lengths at both public entry points before
    token or image processing while accepting valid cross-width integers.
89. Preflight RoPE dimensions, Float32 theta/frequencies, and complete cache
    byte counts before allocating the positional tables used by Qwen models.
90. Normalize every RoPE application start position and prove cache bounds
    without overflow before indexing its positional tables.
91. Require exact four-dimensional one-based RoPE tensors, two-dimensional
    one-based caches, and internally consistent RoPE storage before reshape,
    device transfer, slicing, or inbounds mutation.
92. Normalize Qwen3-VL dynamic/static cache batch and capacity integers before
    parameter inspection or storage allocation.
93. Preflight Qwen3-VL static cache tensor, aggregate K/V, and byte counts with
    exact arithmetic before device allocation, including malformed specs.
94. Normalize cache-free, combined vision/text, dynamic, and static Qwen3-VL
    prefill limits before token, vision, or K/V work.
95. Resolve Transformer MLP ratios without machine-integer wraparound and
    preflight complete block parameter counts before layer construction.
96. Normalize GPT head dimensions and reconstructed KV/expert counts without
    accepting booleans or leaking host-integer conversion failures.
97. Share strict Qwen3-VL prompt coordinate/delta validation across cache-free,
    dynamic, and static prefill without unsigned wraparound or cache mutation.
98. Check dynamic and static Qwen3-VL decode-coordinate addition before token,
    profiling, embedding, or cache-write work.
99. Strictly normalize Qwen3-VL vision/text capture layers and reject malformed
    requests before input validation, vision compute, or decoder work.
100. Preflight Qwen3 text generation output/prompt budgets before chat or raw
     input work, with strict host integers and checked context subtraction.
101. Compute Qwen3 MoE BF16 checkpoint bytes exactly before configuration I/O
     or model/RoPE construction, rejecting host-range overflow.
102. Route matrix-form Qwen3-VL image grids through the strict positive host
     integer contract used by tuple/vector grids.
103. Preflight Qwen3-VL visual spans with exact products and prompt bounds
     before generating coordinate tuples or mutating visual masks.
104. Track Qwen3-VL mRoPE bases incrementally so multi-image layouts avoid
     repeatedly rescanning all prior coordinate tuples.
105. Strictly construct Qwen3 MoE shard metadata, rejecting lossy byte coercion,
     negative sizes, and host-range overflow at the public boundary.
106. Strictly normalize Qwen3 MoE checkpoint provenance, counts, dimensions,
     and shard collections before immutable specifications are created.
107. Preflight Qwen3 MoE shard payload totals with exact arithmetic, rejecting
     host overflow or frozen-contract mismatches before checkpoint file I/O.
108. Derive Qwen3 MoE index tensor counts exactly from layers and experts,
     rejecting overflow or inconsistent frozen metadata before checkpoint I/O.
109. Strictly normalize Qwen3 MoE index `metadata.total_size`, rejecting invalid
     JSON values and reusing the checked host integer in verification reports.
110. Reject duplicate Qwen3 MoE shard filenames at specification construction,
     keeping shard-set membership and payload accounting one-to-one.
111. Strictly construct Qwen3 dense-family specifications, normalizing positive
     dimensions and finite Float32 hyperparameters without lossy coercions.
112. Bind automatic Qwen3-VL prefill mRoPE layouts to the text parameters'
     checkpoint, preserving custom special-token and merge contracts.
113. Strictly normalize Qwen3 tokenizer decode ids and byte-length queries,
     rejecting Bool and host-range overflow without leaking conversions.
114. Strictly parse Qwen3 generation ids, sampling integers, and Float32
     probabilities, rejecting Bool and precision-level overflow or underflow.
115. Strictly parse Qwen3 tokenizer context limits and embedding generation
     lengths, rejecting Bool, non-integers, non-positive values, and overflow.
116. Strictly normalize Qwen3 embedding lengths, MRL dimensions, and retrieval
     limits before tokenization, inference, pooling, or similarity work.
117. Strictly construct Qwen3-VL text specifications, normalizing dimensions,
     finite Float64 hyperparameters, Bool flags, and three-lane mRoPE sections.
118. Strictly construct Qwen3-VL vision specifications, normalizing tower
     dimensions and non-negative three-lane DeepStack indexes.
119. Strictly construct Qwen3-VL checkpoint specifications, validating
     provenance, assets, counts, raw token ids, and nested tower contracts.
120. Strictly construct Qwen3-VL asset metadata and reject duplicate asset names,
     keeping the immutable file manifest one-to-one with verification work.
121. Preflight Qwen3-VL mRoPE checkpoint token bounds and prompt context before
     token conversion, grid parsing, mask creation, or coordinate allocation.
122. Derive default Qwen3-VL generation stop ids through checked checkpoint
     token mapping, rejecting invalid BOS/EOS contracts before prompt work.
123. Strictly construct Qwen3 embedding checkpoint specifications, normalizing
     provenance strings and every positive architecture dimension.
124. Strictly construct Qwen3-VL processor specifications, normalizing image
     geometry and finite channel statistics with positive standard deviations.
125. Safely convert zero-based Qwen3 tokenizer vocabulary, added-token, and
     embedding sentinel ids without Bool coercion or one-based overflow.
126. Strictly validate every Qwen3 embedding template type id as an integer,
     preventing Boolean equality from masquerading as numeric zero.
127. Fail closed on non-string Qwen3 embedding template token metadata instead
     of leaking element-conversion exceptions.
128. Enforce the full Qwen3 deployment profile contract at direct construction,
     including strict scalar types and Float32 sampling boundaries.
129. Harden Qwen3 quantization group and layer boundaries, including positional
     plan construction and layer-specific policy lookup.
130. Accumulate verified Qwen3 deployment asset sizes with checked arithmetic
     so multi-file manifests cannot wrap their total byte report.
131. Strictly normalize Qwen3 INT4 clipping ratios at source and Float32
     precision, rejecting Bool, complex, invalid containers, and rounding leaks.
132. Route every Qwen3 activation calibration construction and lookup through
     strict count, layer, source, and second-moment validation.
133. Strictly normalize activation moments at the Qwen3 INT4 quantization
     consumer boundary instead of coercing Bool or leaking conversion errors.
134. Strictly validate optional Qwen3 asset-manifest model and revision strings
     while retaining normalized AbstractString compatibility.
135. Validate Qwen3 INT4 group positivity before modulo or reshape arithmetic,
     replacing zero-group DivideError leaks with stable argument failures.
136. Reject duplicate JSON fields in Qwen3 deployment profiles, asset manifests,
     and individual asset entries before last-value lookup can hide conflicts.
137. Estimate Qwen3 quantized layer bytes from projection baselines plus sparse
     overrides, avoiding attacker-sized loops over declared model depth.
138. Reject variable-width Qwen3 quantized tensor element types before storage
     accounting can undercount object payloads or leak `sizeof` exceptions.
139. Normalize Qwen3 activation-calibration tokens and sequence limits before
     checkpoint I/O, rejecting Bool, non-integers, and host-range overflow.
140. Enforce positive, even, divisible Qwen3 INT4 weight metadata through its
     exported constructor before dequantization or row slicing.
141. Require non-empty model and revision provenance in Qwen3 deployment asset
     manifests even when no external identity expectation is supplied.
142. Canonically validate Qwen3 asset SHA256 strings across the full field,
     retaining uppercase hexadecimal compatibility without suffix ambiguity.
143. Preserve cancellation through Qwen3 deployment JSON reads instead of
     disguising `InterruptException` as malformed input.
144. Contextualize Qwen3 asset size and hashing I/O failures as argument errors
     while preserving cancellation through both operations.
145. Seal Qwen3-VL vision-input construction behind strict grid metadata and
     host-safe patch-dimension arithmetic that cannot wrap or leak exceptions.
146. Restrict Qwen3-VL attention masks to Bool or integer zero/one values,
     rejecting numerically equal floats, complex numbers, and missing data.
147. Validate Qwen3-VL raw-generation message roles before image extraction,
     replacing numeric and symbolic role conversion failures with clear errors.
148. Make Qwen3 INT8 and packed-INT4 dequantization traceable on Reactant while
     preserving exhaustive host nibble semantics and compiled row slicing.
149. Reject contradictory or unrepresentable Qwen3-VL processor pixel budgets
     before resize arithmetic can overflow or produce an impossible geometry.
150. Widen Qwen3-VL resize rounding for host-limit dimensions and verify every
     returned geometry exactly satisfies its factor and closed pixel budget.
151. Seal Qwen3 INT8 weights behind rank, dtype, axes, shape, and device checks
     while retaining a trace-safe internal row-slice construction path.
152. Enforce exact Qwen3 INT4 packed and scale tensor layouts, dtypes, axes, and
     device placement without revalidating trusted slices inside XLA traces.
153. Restrict Qwen3 quantized row slicing to non-empty in-bounds host ranges so
     its trace-safe constructor cannot create malformed INT8 or INT4 weights.
154. Accept one Qwen3-VL image grid as a flat tuple or vector without confusing
     its three dimensions for a collection of three separate image grids.
155. Normalize absent Qwen3-VL image grids as an empty collection and reject
     non-container grid payloads before iteration can leak method errors.
156. Restrict Qwen3 embedding masks to Bool or integer zero/one entries rather
     than accepting numerically equal floats, complex values, or missing data.
157. Seal Qwen3-VL dynamic cache construction behind strict host integers and
     local position, batch, and empty-cache RoPE invariants.
158. Enforce host-safe Qwen3-VL static cache capacity, position, batch, and
     empty-state invariants through its sole inferred inner constructor.
159. Normalize Qwen3 embedding-forward token matrices through strict host
     integers, rejecting Bool, non-integers, and range overflow consistently.
160. Validate every Qwen3 embedding text before String conversion, preserving
     iterable inputs while replacing mixed-payload method errors.
161. Seal Qwen3 sparse-MoE construction behind strict host dimensions and a
     literal-Bool routing-normalization policy across every constructor entry.
162. Normalize Qwen3 host and device top-k controls through strict host
     integers and literal Bool flags, guarding device expert indices at Int32.
163. Normalize Qwen3 CUDA indexed-workspace dimensions and element width as
     positive host integers before checked byte arithmetic.
164. Mask exact zero-weight Qwen3 MoE routes before device reduction so
     Float32 probability underflow cannot turn inactive NaN experts into NaN.
165. Seal Qwen3 MoE dispatch statistics behind host-safe, internally
     consistent counts and detach their per-expert vector from caller aliases.
166. Rank Qwen3 MoE candidates by Float32 logits and flush subnormal routing
     probabilities so CPU, CUDA, and XLA agree on extreme finite inputs.
167. Reject offset-indexed Qwen3 router logits at both host and device API
     boundaries before backend-specific bounds or broadcast failures can leak.
168. Derive every Qwen3 XLA window size inside its sole inner constructor and
     validate legacy full-field construction against the canonical formulas.
169. Seal HuggingFace Qwen3 generation configs behind positive host token ids,
     unique stops, literal sampling flags, and finite Float32 sampling controls.
170. Validate Qwen3 sampling controls in both their original Real domain and
     stored Float32 domain so precision rounding cannot admit out-of-range data.
171. Canonicalize duplicate Qwen3 JSON keys with CPython last-value semantics
     while preserving first insertion position across parsing and rendering.
172. Parse stringified Qwen3 text and VL tool arguments as JSON objects before
     canonical Python-style rendering, closing malformed and duplicate-key bypasses.
173. Normalize Qwen3 tool integer arguments through the strict host boundary so
     oversized JSON integers and unsigned values report stable argument failures.
174. Check Qwen3 integer-tool addition for host overflow so boundary operands
     cannot silently wrap into a plausible but incorrect tool response.
175. Validate Qwen3 file-tool default byte limits eagerly as strict host integers
     within the same bounded range enforced for model-supplied requests.
176. Seal SwiGLU construction behind strict positive host dimensions and a literal
     bias flag while preserving Lux container reconstruction for Qwen models.
177. Validate Qwen3-VL mRoPE section partitions with overflow-safe sums and lane
     bounds so wrapped host arithmetic cannot admit corrupted layouts.
178. Seal Qwen3-VL RoPE layouts behind rank, element-type, and cross-field shape
     invariants while preserving caller-owned host and device array storage.
179. Bind SwiGLU metadata to identity-activated Dense projection dimensions and
     bias state so malformed raw construction cannot corrupt Qwen MLP semantics.
180. Bind GPTModel layer-count metadata to its actual block chain so malformed
     raw Qwen construction cannot diverge cache and weight-loading layer sets.
181. Preflight Qwen3-VL spatial-merge derived widths with exact arithmetic so
     enormous merge factors cannot wrap into plausible vision dimensions.
182. Preflight Qwen3-VL fused QKV widths with exact arithmetic before tensor-shape
     construction so individually valid hidden sizes cannot overflow tripling.
183. Validate Qwen3 deployment probabilities in their original Real domain before
     Float32 storage so just-over-one top-p values cannot round into acceptance.
184. Preflight Qwen3 CUDA route and padded-layout capacities against the Int32
     metadata sentinel before any device allocation or kernel launch.
185. Bind processed Qwen3-VL image tensors to their source resize, patch grid,
     and processor geometry while preserving zero-copy array ownership.
186. Preflight Qwen3 MoE host staging-pool dimensions and aggregate bytes before
     allocating channels or per-expert upload matrices.
187. Preflight reusable Qwen3 MoE safetensors read-buffer dimensions and total
     payload bytes before allocating pool channels or byte vectors.
188. Normalize host Qwen3 generation stop ids through the strict shared token
     boundary so Boolean, floating-point, and oversized ids cannot be coerced.
189. Bind Qwen3-VL vision feature shapes, dtypes, DeepStack arity, and captured
     layer metadata while preserving device arrays and intentional aliases.
190. Seal imported Qwen3 tokenizer construction behind the validated JSON-loader
     path so converting field constructors cannot forge inconsistent vocabularies.

The Julia 1.12.6 Manifest has been regenerated and is now tracked, and the CI
Julia 1.11/Project Julia 1.12 mismatch has been repaired. Full `Pkg.test()` now
imports the real `LifeAI` module before these files, preventing their standalone
fallbacks from polluting later episode tests. The three core focused files
remain independently loadable for fast contract checks; Chapter 03 exercises
the training/evaluation boundary through the real module, and the complete
default suite is the integration gate. CI instantiates the checked-in
`Project.toml`/`Manifest.toml` pair and rejects dependency-file drift after the
suite, so a registry update cannot silently change the validated environment.
