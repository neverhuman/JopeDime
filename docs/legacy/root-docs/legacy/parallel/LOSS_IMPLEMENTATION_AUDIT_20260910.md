# Regression loss implementation audit — 2026-09-10

This is static source evidence only. No optimizer steps, GPU tasks, proposal evaluations, or gradient-contribution experiments ran. Preserve the current objective for the first support/replay experiment; the findings below do not authorize a new objective.

The production training closure in `training.rs` computes weighted sequence NLL + 0.25 pairwise Bradley–Terry + 0.10 listwise + 0.10 runtime Huber + 0.05 context-free pairwise ranking. Stability is multiplied by zero. Every learned loss-log-variance parameter is connected through multiplication by zero so its gradient exists for packed distributed reduction; these are inactive objectives. The separate neural test fixture includes a stability penalty and is not the production objective.

- Sequence NLL masks illegal vocabulary entries before log-softmax, checks each nonpadding supervision token is legal, and applies 5% smoothing over the legal set. It averages valid tokens within each program, then forms a quality-weighted mean of program losses using `0.05 + percentile^4`. Empty legal sets fail; padding is excluded. The denominator clamps avoid division by zero but do not by themselves establish valid empty-group teaching semantics.
- Main reward is the mean of four head outputs. This trains their mean; it does not establish calibration or uncertainty. Ranking groups combine context identity and each contiguous eight-program comparison bank. Pairs across groups are absent. Exact value ties and nonfinite value pairs are skipped. Weight is absolute value difference, doubled when either percentile is at least0.9, then normalized by the mean included-pair weight. Dense pair construction preserves a fixed reduction layout to avoid nondeterministic repeated-index CUDA backward accumulation.
- Listwise targets are softmax(percentile/0.25) for groups with at least three programs and finite percentiles. Loss is cross-entropy against the group reward log-softmax, averaged over valid groups. Exact precision ties are repaired in strict admission v4 before f32 training conversion. No moving-median/frozen-z target change was made in this audit.
- Runtime labels are log1p(original numerical elapsed seconds) in strict teaching shards; the prediction is softplus and then log1p before mean smooth-L1 loss with beta1. Timeout, infrastructure and unknown outcomes are retained separately with measured/censoring fields. They are not automatically equivalent to successful runtime training examples; a censor-aware runtime-only sampling/objective remains unqualified.
- Microbatch losses are normalized locally, divided by accumulation count, and their gradients are packed for distributed reduction. Equal family sampling provides expected family balance, but averaging separately normalized weighted losses is not generally equal to a single globally weighted mean when microbatch/rank weight sums differ. The exact existing normalization must be frozen and a synthetic distributed gradient audit completed before interpreting any hardware-allocation comparison. Do not silently change normalization in a hardware-only benchmark.
- Undefined or nonfinite packed gradients abort the step. Inactive zero-connected heads remain checkpoint-compatible. Per-term gradient magnitudes, empty/failed-group paths, and exact8/4/2-rank equivalence are still unmeasured and required before the relevant experimental qualification. No capacity diagnosis follows from this code inspection.

Source bindings at this inspection (later source changes require rebinding):

```json
{
  "parallel/src/training.rs": "726a610213cc4b20bf576c1a8429bdf329bc1867bf26708aede033f79f10bafd",
  "parallel/src/neural.rs": "34cde3c9c8eda1c8c3566b21faa358dcf1dc08cefe341f493390de46605c75e4",
  "parallel/src/teaching_store.rs": "7269bb0bb19815a6a30e48161a51e8a8cb2d67d56118abd00161c8e192123966"
}
```

Runtime receipt scope: the compatibility field `numerical_runtime_seconds` currently contains elapsed time from authority authorization through observed original executor exit, including dispatch and input verification. It is not an isolated numerical-body or CPU measurement. Runtime-target interpretation and any scope correction require a separately versioned label policy; this release does not claim pure fitting cost.
