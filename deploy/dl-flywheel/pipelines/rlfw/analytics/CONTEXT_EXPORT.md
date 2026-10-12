# Per-context L6 export for CX-MET / G-JUDGE

Write `context-costs.json` in an assigned scratch run directory. The analytics
lane copies it read-only. It neither runs evaluation nor changes the trainer.
CL and CX-OP remote metadata discovery includes this exact filename; local
owners may supply `context_patterns` in CX-MET's manifest.

The schema is `jope.rlfw-scratch-context-costs/v1`. Required top-level fields:

- `non_qualifying: true`, `benchmark_valid: false`, `promotion_input: false`.
- `lane`, `run`, and optional `source_commit` (unknown remains null).
- `run_receipt_sha256`: optional final receipt hash. When present it must name
  a physical copied native R22 receipt with the same anchor. A running export
  may leave it null; the output then labels final run reconciliation unmeasured.
- `comparison_identity`: `anchor_parameter_sha256`, `trie_sha256`,
  `normalizer_sha256`, `validation_split_sha256`, `evaluation_protocol_sha256`.
  Use actual provider identities; do not fabricate absent hashes. The evaluation
  protocol may be the hash of the frozen evaluation implementation.
- `context_ids`: the full fixed validation roster (263 contexts for R22).
- `context_set_sha256`: SHA256 of UTF-8 `json.dumps(sorted(context_ids),
  separators=(",", ":"))`, with no trailing newline.
- `observations`: one block per arm/epoch, with `arm`, `epoch`, `updates`,
  and `contexts`. Include the control block at every evaluated epoch.
- `control_arm`: optional explicit comparator. Otherwise the unique named
  `control` or `r23_control` arm is used. An ambiguous pair of those names is
  refused; no treatment arm is selected as a missing comparator.

Each `contexts` entry contains `context_id` and finite `mean_cost`: the actual
per-context L6 cost whose definition stays fixed across control and treatment.
Optional `items` and `item_set_sha256` preserve per-context evaluation identity;
if present they must match between arms. Missing identities stay unmeasured.
These are cost observations, not predictions of future q and not fold-q scores.

The collector joins by context, epoch, declared evaluation identity and reported
update budget. It emits control-minus-arm pairs, context-macro mean, sample
variance (ddof=1) of paired deltas and each arm's costs, SE of the mean, and an
exploratory normal CI. Epoch repeats are separate cohorts and never expand n.
Incomplete rosters leave full-panel variance and sizing unmeasured. Item-weighted
aggregate training/validation losses can differ from a context-macro mean.

Power sizing uses the measured paired-delta SD, target difference **0.005 L6
cost**, two-sided alpha **0.05**, and power **0.80**. It finds the smallest n
using the two-sided normal power curve. This assumes independent contexts and
does not provide a family gate, multiple-comparison correction, or promotion.
The retained fold-q median SE and the 881-context stand-in do not enter measured
sizing. Constant sample deltas do not certify population noise is zero; n remains
unmeasured when the measured SD is zero.

Source: `context_costs.py` (variance/power/identity rules),
`tests/test_context_costs.py` (actual pairs, missing coverage, epoch cohorts,
physical parent join, exact 881 stand-in boundary, scope clock boundaries).

Native CL `paired_eval.py` also works without changing its running job: the
collector reads `index.json`, `rows.jsonl`, and hash/count-verified
`contexts-*.json`. Only index-confirmed complete arm/epoch cohorts enter the
statistics. Set `a` (validation), `b_prefix_1000` and final `b` (scratch-unused
train, already seen by the v1 base) remain separate; overlapping prefixes are
never added together. Initial `all` checkpoint rows are not broadcast to later
epochs as control.

Native raw `l6_loss` statistics use `l6_raw_*` names. For comparable pooled L6
cost, each context contributes `N*sum(coeff_t*num_i,t/total_den_t)`, using the
coefficients recorded in the index evaluation definition. The mean must match
the saved `l6_macro_loss` within 1e-6; support-denominator changes refuse a pair.
The corresponding `l6_*` variance and size estimate describe that contribution
definition. Full raw numerator/denominator rows remain in pinned input copies.
Absent native trie/normalizer/protocol hashes remain NULL; one copied native
index/JSONL identifies the observational cohort, with input closure explicitly
unqualified. Future strict owner exports can supply those missing identities.

CL in-process v5/v6 exports use `paired-index.json`, `paired-rows.jsonl`, and
`checkpoints-index.json`. Their roster SHA preserves the saved order, including
the sampled `b` roster. Each complete cohort must match the parent receipt's
arm, epoch, updates, parameter SHA and saved set L6 macro. The `a` validation,
`b` sample and `heldout` lineage panels remain separate, with separate comparisons
against explicitly named `control` and `r23` baselines. Unequal update budgets
leave paired summaries unmeasured. No checkpoint tensors are opened.

Host GPU utilization is a timestamped physical device sample, shared by every
workload on that device. A sample is not a full-day time average or attributable
lane utilization; those metrics remain unmeasured without a complete trace.
## Expert-review operational observations (2026-10-11)

`python -m rlfw.analytics.operational --manifest <owned-json> --output <fresh-owned-directory>`
requires a systemd CPU scope capped at 200% and 4 GiB. It copies every referenced
JSON/Parquet input before reading, verifies optional SHA/byte pins, writes new daily
Parquet and renders its readback with the pinned `base_daily` partitions. It has no
learner, evaluator, S3 or writable database interface.

The manifest contains `date`, `base_daily` (file references), optional `throughput`
(`export` and physical finished `receipt` references), `parents`, `feedback`, and
`actor` references. Each reference has `path`, optional `sha256` and `bytes`.
G-S3GP's native `jope.g-s3gp-throughput-denominator/v1` must reconcile its count,
active RUSAGE seconds, constant allocated cores and measured wall duration with
the copied terminal receipt. Programs, compiles and cached disclosures stay separate.
Scratch rates never populate qualified outcomes/hour or non-catalog programs/hour.

The consumer's normalized `jope.cx-met-parent-map/v1` has `rows` containing explicit
`context_id` and `parent_id` from a physical lineage source, whose hashes must be
retained in custody. Parent IDs are never derived from names. Each complete paired
cohort averages cuts within parent, then gives parents equal weight. Missing joins
leave the complete-panel mean, variance and SE NULL. Epochs and export identities
are separate; these inputs do not establish independent new-family generalization.
Identical local/remote copies of the same export/context/epoch are counted once;
conflicting values or identities under one physical export hash are refused.

`jope.rlfw-feedback-age/v1` contains `observed_until` and `events` with exact
`join_key`, `context_id`, `parent_id`, `stage` and `at_unix`. Stages are `proposed`,
`compute_started`, `locally_verified`, `remotely_qualified`, `adopted`,
`first_trained`, `first_evaluated`. First-trained means an actual optimizer update
using that feedback. Adoption, snapshot availability and evaluation cannot fill it.
Missing stages stay NULL; missing first-training clocks remain right-censored.

`jope.rlfw-actor-diagnostics/v1` requires `non_qualifying: true`,
`program_execution: false`, `promotion_input: false`, `lane`, `run`, and
`observations` keyed by actual `objective` and positive `update`. Optional fields
are `actor_grad_norm`, explicit `actor_target_context_ids`, `draws` preserving every
draw position, `B`, `probabilities` and `complete_policy`. D_B and entropy require
the complete normalized actual finite policy, never renormalized top-k mass.
`advantage_support` requires `total`, `nonzero`, `complete: true` and
`hwm_identity_verified: true` to measure nonzero HWM-advantage support.
`archive_exposure_by_parent` counters require `exposure_counter_kind` (`step` or
`cumulative`); successive cumulative snapshots are never summed. Unavailable
qualified outcomes, actor movement/exposure or heldout gain/budget remain unmeasured.

`native_actor_audits` accepts CX-OP's `PASS_INITIAL_NON_QUALIFYING` initial audit.
It retains each probe's own epoch/update, fixed context count and parameter hash.
Raw and coefficient-scaled term gradients are labelled probes; a later
`common_completed_updates` field never relabels an update-zero probe. Reported
D_B for full-trie and measured-leaf conditional pi_M stay separate. The trainer's
clipped-utility HWM-support fraction is scratch evidence, not a qualified reward
closure. These records do not establish actual target exposure or feedback clocks.
