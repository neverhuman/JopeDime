# H157: best-of-K trust-region proposals with inner-fold admission

Status: preregistration only; no H157 scored operation or optimizer update has
run. Execution is authorized only from this source after its preregistration PR
is merged and cross-review passes. Hypothesis: generating eight diverse, low-drift candidates from the
same frozen r882 adapter and admitting a candidate only after strict inner
TRAIN-family selection can improve native regression utility on the fixed
91-context DL-Val panel.

## Inputs and scope

Use the H145 real-S3 TRAIN roster (1,219 contexts, 205 resolved families),
H50 selected/source manifests, H58 feature cache, H142 trust-region recipe and
H148 family-selection conventions. Identity hashes and byte hashes are pinned
in `manifest.json`. The checkpoint is an offline copy of `round-882.pt`,
SHA-256 `0084479aa0703667a0f3ccfe2beda60741c204795ec57b7c08542401ec6cdb64`.
The native validation reference is the pinned H148/H151
`best_dev_checkpoint.json`; the evaluation identity is
`e79a7715c1531ccb8c4055ab96971101cd4034ec3a71d83bca6d25b0e604ca83`.

Only real-S3-derived TRAIN rows from the pinned selected manifest may enter
fit/inner-select. The identity-only known sealed-family deny-list from the
H150 provenance record is applied before partitioning; the deny-list's
limitations remain: no claim of globally complete alias exclusion. No sealed,
holdout, meta-test, or live-loop inputs or outputs are read. Before training,
DL-Val may be read only for the unchanged-r882 two-copy regression gate. No
treatment candidate is evaluated on DL-Val until all candidate choices and
their hashes are frozen. Historical r882
exposure to TRAIN and prior DL-Val selection remain explicit limitations; this
is conditional continuation evidence, not a fresh unseen-family test.

## Family partition and arms

Sort the 205 eligible family IDs, then apply `numpy.random.default_rng(157140)`
permutation and `numpy.array_split(..., 5)`. Group 0 is the inner-select set;
groups 1–4 are fit. Every context from a family stays in its assigned group.
The frozen assignment is `fold_manifest.json`; assert exactly 41 select and
164 fit families, disjointness, and complete coverage before any training.

Fresh candidate seeds are 157101, 157102, and 157103. Per seed, all arms start
from an owned byte-exact copy of r882:

* `R882`: unchanged zero-update baseline.
* `TR1`: one H142 `TR_LR` candidate, exactly 40 updates.
* `BOK8`: eight independently initialized-from-r882 `TR_LR` candidates,
  exactly 40 updates each. Candidate diversity is limited to deterministic
  minibatch order and family-bootstrap resampling seeds; architecture, loss,
  optimizer and hyperparameters are identical.

`TR_LR` is H142's all-candidate argmin-aligned listwise KL coefficient 1.0,
temperature 0.2, AdamW learning rate `1e-5`, weight decay `1e-4`, gradient norm
clip 1.0, 40 updates, batch size 16 and slate size 192. Follow the actual
H142 runner implementation pinned by SHA-256
`d5215234c2659d2a470ac95c0900827bfad79704b7fb5ac7fe952cdb633d4a8e`; its
TR_LR uses temperature 0.2. H142 prose elsewhere says 0.1; the runner and its
manifest are the executable source of truth. The candidate sampler retains the
first 48 fitness-ranked entries and samples the remaining 144 from the rest.
Candidate sampler seed is `seed * 100 + candidate_index` (`0` for TR1, `1–8`
for BOK8); family-bootstrap resampling uses that seed plus `157140`. Follow
the H142 20-round plan, consuming eight paired streams per round and retaining
steps 0 and 1, for 40 effective updates. The loss is per-context SmoothL1
(beta 1.0) plus `1.0 * KL`, with temperature `0.2`, clipping norm `1.0`, and
AdamW `(lr=1e-5, weight_decay=1e-4)`. Only the
29,211 `adapter.*` parameters are trainable. DOPE stays frozen at
`c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814`; verify it
and all non-adapter tensors byte-for-byte before and after every update and
evaluation. Record adapter L2/L-infinity drift; any integrity violation stops
the attempt.

## Regression gate and candidate selection

Before any optimizer update, evaluate two independent r882 copies on all 91
DL-Val contexts. They must match exactly; the first reference comparison must
reproduce native mean utility `1.1377941898164525` and all 91 context utilities
within `1e-7`. Any mismatch ends the attempt before training; no tuning or
retry.

For each seed, score TR1 and each BOK8 candidate on the 41 inner-select
families. Native utility is negative original fitness for the first
minimum-cost selected candidate, or `-2` on failed gate; do not gate-filter or
clip utility. Average paired utility differences within each family; the candidate score is
the median of these 41 family gains. Compare each candidate to r882 on the same
families. Use 2,000 paired family
bootstrap draws (resample 41 families with replacement; deterministic seeds
157141 + 100 * seed_index + candidate_index, where `candidate_index=0` for TR1
and 1–8 for BOK8). Assign stable candidate IDs `bok-01` through `bok-08` in candidate-index order.
Select only the BOK8 candidate with highest observed median family gain; exact
ties for the best score keep r882. Do not use a candidate-ID tie-break for a
tied maximum. TR1 bootstrap seed is
`157141 + 100 * seed_index`; BOK8 candidate bootstrap seed is
`157141 + 100 * seed_index + candidate_index` for candidate indices 1–8. Do not fall back to a
lower-ranked candidate if the unique winner fails admission. Admit TR1 or the BOK8
winner only when its observed median family gain is strictly greater than
`+0.001` and the percentile 2.5% bootstrap bound is strictly greater than
zero. Values within `1e-12` of `0.001` are ties and reject. If the BOK8
winner fails, select r882. Persist selection scores,
bootstrap summaries and chosen checkpoint hashes before any DL-Val read.

## Endpoints and decision

Primary endpoint: selected BOK8 minus R882 native utility on 91 DL-Val contexts,
averaged within each of the 19 families, equal-family median across the three
seeds. Report the 10,000-draw paired family bootstrap percentile 95% CI. Also
report TR1 minus R882 under the same calculation and apply Holm step-down to
the two one-sided family-bootstrap tests `{BOK8, TR1}`. For each treatment,
report raw 95% percentile CI, one-sided bootstrap p-value
`(1 + count(bootstrap_contrast <= 0)) / (draws + 1)`, Holm-adjusted p-value
and a step-down lower bound: order by raw p-value (ties by arm name), then use
lower quantile `0.025` for rank 1 and `0.05` for rank 2. Use the Holm lower
bound for admission. For each paired family-bootstrap draw, resample the 19 DL-Val families once,
compute each seed’s equal-family mean contrast on that draw, then take the
median of the three seed contrasts. The reported point estimate is the median
of the three seed-level equal-family mean contrasts. The BOK8 result is eligible
for `ADVANCE` only if its median gain is strictly greater than `+0.001` and its
Holm-adjusted lower 95% bound is strictly greater than zero. Otherwise the
result is `RULE_NEUTRAL`/`NO_ADMISSION`; ties reject. No checkpoint is promoted
or wired into the live loop.

Report Elo relative to R882 = 1500 using H145's canonical 400-iteration
Bradley–Terry MM method and family-averaged pairwise comparisons, with 1,000
shared-family bootstrap draws (seed 157143). The H145
1,219-context/205-family TRAIN panel is descriptive only and must explicitly
state TRAIN exposure. Report regression utility, lift and Elo; do not report
AUC or speed metrics. This is regression only; binary work is deferred.

## Resources, custody, and execution order

Run CPU-only on xbabe2, `nice 19`, at most four total threads, aggregate RSS
strictly below 6,000,000,000 bytes. Stop before or during work if load exceeds
85% of the 102-core host, disk use reaches 95%, free space falls below 40 GB,
or memory pressure threatens the RSS limit. Check load, memory, swap and disk
before launch and continuously during updates. xbabe3 and xbabe0 are forbidden.
No GPU and no `qualification.lock` use for H157.

Estimated actual updates: 3 × 40 TR1 + 3 × 8 × 40 BOK8 = 1,080. After PR
merge and the exact-head cross-review passes, place a launch receipt at
`/home/ubuntu/jope_plan/evidence/H157/launch.json` with `source_commit`, `manifest_sha256`, `merged_source_verified=true`,
`cross_review_pass=true`, `reviewer="luna_v1_rnk11"`, `review_pr`,
`reviewed_source_commit`, and `merged_pr`. From the repository root, run:

```sh
nice -n 19 python3 research/flywheel_selfplay/h157_best_of_k_trust_region/runner.py \
  --output /home/ubuntu/jope-data/h157/attempt-001 \
  --launch /home/ubuntu/jope_plan/evidence/H157/launch.json
```

No scored work or optimizer updates are permitted before that prerequisite. Preserve all
large inputs, scores, traces, checkpoints and resource samples outside Git in
H157 custody; Git may contain only this preregistration, small manifests,
data-free tests and small final metrics/receipts. The closeout is a separate
PR with README hypothesis/setup/outcome/metrics/verdict/next and small
`metrics.json`, after execution. No raw objects or checkpoint files in Git.
