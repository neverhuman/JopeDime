# H324 — Huber-style context influence cap for LambdaRank members

Owner rd_astra [ASTRA]; authority `~/nudges/hw_t1535_astra_h324.md`.
TRAIN-only, one CAP arm paired against unchanged H303 NESTED BASE. No dev91 shot.
This protocol must be published non-force to `origin/v1/next` before any fit.
Template H322 close `21010fdc5553a8d2791f8c8e260e7790c92715b8`.

## Hypothesis and precise intervention

Test whether reducing the relative fitting weight of contexts with large raw
utility spreads improves gain outside the top ten families. This is a heuristic
query-weight intervention, not a Huber loss or a proven bound on actual gradients.
BASE LambdaRank receives ordinal relevance labels, not raw utilities; large raw
spreads therefore do not directly imply large lambda gradients. Mean-one
normalization can make some final sample weights exceed one.

For EACH actual ranker fitting multiset C, including repeated bootstrap contexts:

1. `spread_c = max_k(u_ck) - min_k(u_ck)` over the same K=8 BASE candidate slots.
2. `delta = median([spread_c for c in C])`, counting every occurrence.
3. `raw_w_c = min(1, delta / max(spread_c, 1e-9))`.
4. `w_c = raw_w_c / mean([raw_w_c for c in C])`; repeat on all eight slots.

Compute in float64. Require finite utilities, spreads and positive finite weights.
If delta is zero, the requested mean-one normalization is undefined: stop with
ZERO_MEDIAN_SPREAD / INCOMPLETE, without substituting a formula or scoring a subset.
Zero-spread contexts otherwise receive raw weight one. Capped fraction means
`raw_w_c < 1` before normalization. Record count, fraction, delta, context order,
spread/weight hashes, and normalized weight quantiles at
0/.01/.05/.25/.50/.75/.95/.99/1 for every actual fitting multiset.

Verified sentinel: pinned H287 `topk.py` at commit
`213b30ad52e1ce48927bba960ed609bd98ab4dfe`, line162 constructs
`np.where(gate, -fitness, -2.0)`; selected `utility_k` is this raw utility.
Do not clip, normalize, relabel or alter the failure sentinel.

Only ranker sample weights change from BASE. Preserve LambdaRank, BASE ordinal
labels and unchanged label_gain behavior, all features, seeds, original family
identities, LightGBM parameter recipe and selection rules. Two fit boundaries:
`topk.fit_ranker`, `bagged.fit_ranker_n`. Three source call sites:
the inner and final ranker calls in `veto.run_cell_veto`, and the bag call in
`r302.fit_and_freeze_member`. No weight argument reaches `veto.fit_gate`;
invalid/veto classifiers remain byte-identical source and unweighted.
Each of 36 cells has 819 weighted ranker fits and 195 unweighted gate fits.
Selected lr/tree count/margins/config may change as a consequence of weighting.
The 18-config TAU/k/agreement grid and inner mean-family-gain objective stay BASE.

## Inputs, regression and evaluation boundary

H308 P0/P1/P2, roots30001/30002/30003, four original-family-grouped outer folds;
1,219 TRAIN contexts and205families. Byte-pinned H310/H303/H302/H287 modules.
Use only offline copies recovered from the exact-version H318 archive in
[base_archive_pin.json](base_archive_pin.json). Before CAP fitting, replay all
36 frozen BASE fold-root hashes, nine pick/utility panels, and three medians:
P0 .0643163334285782, P1 .04821480030677507, P2 .04360142642936269.
Any mismatch is REGRESSION_FAIL and stops the wave.

Each fitting worker physically reads only its outer-TRAIN pools, cost members
and allowed cache byte ranges. No held-out packet prebuild; historical BASE
outcomes cannot be opened in workers. All36 fsynced fit locks and model/input/
weight/source hashes precede the separate once-only CAP outer scoring process.
Audit original config objective, weight routes and exact unweighted final gate
hashes against BASE. No partial scoring if fits/audits are incomplete.

## Metrics and frozen decision

Raw utility gain against r882 slot0. Compute within-family context means, then
equal-family mean per partition/root cell. Pooled gain is the median of nine
cell mean gains, preserving H308 convention; report literal family medians too.
Report all nine cells: gain/CI, drop-top1/5/10, LOFO minimum, W/T/L tolerance1e-4,
harm (loss greater than.01, equivalently gain<-.01), invalid picks, switches, Elo,
switched-set seed-Jaccard and config agreement. Report per-fit capped fractions
and weight quantiles, summarized across fitting occurrences; no outer-label
weight calculation is required for scoring. LOFO minimum equals drop-top1.

Use 10,000 paired whole-family bootstrap draws, RNG324191, with identical205family
draws shared by both arms and all nine dependent cells. For each arm separately,
sort and drop its ten largest sampled family gains, then take the median of nine
CAP-minus-BASE differences. Two-sided CI uses .025/.975; one-sided95LCB uses .05.
Intervals condition on fitted policies and do not repeat model selection.

CAP_ADVANCE iff ALL hold:

- Paired drop-top10 median over9cells >= +.002 AND one-sided95LCB >0.
- CAP pooled full gain >=.90*BASE pooled full gain.
- CAP median harmed-family rate <=BASE.
- For at least2of3roots, median paired drop-top10 over its3partitions >0.

The +.002 threshold is inclusive as specified; zero differences, zero LCB, and
margins below+.002 do not advance. Otherwise CAP_NOT_ADVANCED. Separately label
CONFIG_INSENSITIVE if exact (TAU,k,a) matches BASE in>=30of36fold-root locks;
this diagnostic does not replace or relax the advance gate. No endpoint tuning.
If advanced, next is a separately preregistered one-shot dev91 vsr3175, not H324.

Elo label: **reference = r882 slot0 (DL-Train outer); r3175 not measured on TRAIN**.
Use the unchanged400-iteration Bradley–Terry recipe; no AUC or speed headline.

## Resources, custody and publication

xbabe2 CPU only, nice19, numeric threads1, at most6cores4/5/6/8/33/41 with fresh
mpstat and peer/SMT-disjoint checks before launch. MemoryMax6GiB, MemorySwapMax0,
aggregateRSS<=6GiB, hostCPU<=85%, load1<=85, MemAvailable>=20GiB.
Own scratch `/home/ubuntu/jope-data/h324-astra` <=2,000,000,000bytes. Reclaim only
after create-only S3 exact-version GET/SHA/member recovery verification.
Prefix `s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/H324/`.
Compute stop17:30Z, close17:45Z November4; never reset these deadlines.
If throughput projects past compute stop, declare INCOMPLETE without silently
reducing the protocol. No GPU/qualification.lock, dev91/DL-Val/DL-Test/sealed/
fresh cohorts, foreign jobs or live writers. Original local input S3 VersionIds
may remain unestablished; retain permitted range/packet hashes honestly.

`dope = {"composite_sha256_prefix":"c51a56d0","loaded":false,"scope":"CPU selection layer only"}`.
Existing cached feature vectors may be read within permitted TRAIN ranges;
DOPE weights are not loaded, modified or qualified.

Private index/ref rd/h324-cap; no new worktree. Non-force v1/next only, no PR,
main, fastlane, hosted CI or release changes. Final source/README/metrics<50KB,
one row each wave/overnight index, locked FINDINGS/claim/comms closeout.
Focused formula, boundary, actual LightGBM route and decision tests plus full
weight/config/input/numerical review are required; same-agent review is labelled.
