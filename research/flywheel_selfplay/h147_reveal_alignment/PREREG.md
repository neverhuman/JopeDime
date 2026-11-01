# H147 — isolate training and serving reveal alignment

**Status: proposal / preregistration only. No execution is authorized by this document.**
Owner: rd_astra [ASTRA], under `idle_t2020.md`, section `rd_astra`. No model forward,
optimizer update, candidate scoring, new dataset payload read, or live-loop change has occurred.
The intended result is a bounded continuation diagnostic, not RNK-10/Q1 qualification.

## Question and prior evidence

Can changing which four candidates are revealed during continuation improve ranking under the
fixed serving history? H144 changed weight decay but not validation choices; its effective shrink
was small. H145 found a TRAIN ranking preference among historically exposed checkpoints, not a
qualified new candidate. Another optimizer dose is not the only remaining lever.

The code-level mismatch is concrete: H144's `science.batch_plan` puts fitness-ranked candidates
first, and `runner.predict` reveals the first four positions. Its evaluator passes storage order,
so the revealed candidates differ between training and assessment. H58 changed this history,
masked the loss, changed priors/context/scale, froze a subnetwork and used different initialization
together. H108 studied prior replacement under the repaired history. Neither is the matched
r882 continuation contrast below. H147 isolates history on a common candidate set while retaining
the live zero-prior, context, scale, frozen state and optimizer. It does **not** assert that reveal
alignment is an entirely untried idea or that those earlier studies proved its benefit.

Evidence: [H58 preregistration](../h58/PREREG.md), [H108 readout](../h108_w1_prior_residual/README.md),
[H144 code](../h144_reg_sweep/runner.py), [H144 result](../h144_reg_sweep/README.md),
and [H145 readout](../../h145/README.md). Reference execution source is H144 merged commit
`aad54937c43ecd413ac2990d23ac5bff3e0dc27f`; this proposal does not modify it.

## Data boundary and fixed splits

Use the approved real-S3-derived H50 **DL-Train** roster: 1,219 contexts / 205 families,
source manifest SHA `99a53569fcaf59f9309e69329c830ae1684adff382da3ae22e0b2645c4ebc4d4`,
selected metadata SHA `981960ad628f5029f8c54552c7b1cb1d440072e7e9dcffecdea52c6a3493a0f1`.
The mixed selected file contains validation identity metadata; reading that metadata for exclusion
is explicitly disclosed. No DL-Val pool, outcome, feature array or model scoring enters H147.
No sealed, holdout, outer-test or meta-test payload may be opened.

[fold_manifest.json](fold_manifest.json) fixes all five assessment folds. Sort families, permute
with NumPy `default_rng(147140)`, and split into five groups of 41. Every family and all its contexts
stay together. Train each cell on the other four groups. No endpoint, normalization, scale, mask,
history, seed or arm may be selected using assessment outcomes. Every context is assessed once per
arm/seed by the model whose continuation excluded that family.

This is **continuation-held-out**, not pretraining-held-out: r882 has prior exposure to DL-Train.
No unseen-family/generalization or production-acceptance claim follows from these folds.
The authoritative catalog's known sealed-ID exclusions and TRAIN membership must be checked by
identity only; its missing global sealed alias roster must remain disclosed, never reported as a
complete universal exclusion certificate.

Before a future execution, an implementation must bind a TRAIN-only feature/input packet and all
payload SHA-256s to this roster. Derive frozen tokens from the permitted training tables or consume
an already verified TRAIN-only token packet. Do not load H58's full 1,310-context feature cache and
call it a zero-validation-read closure. The current proposal has no executable, feature packet,
materialized model copy or consumed outcome ledger. Those are explicit pre-execution requirements.

## Frozen candidate and common recipe

Every cell starts from the same copied r882 checkpoint, SHA
`0084479aa0703667a0f3ccfe2beda60741c204795ec57b7c08542401ec6cdb64`, with a fresh optimizer.
A recovery source is the H144 `exact-inputs.npz` archive identified in [manifest.json](manifest.json);
use the exact S3 version and member SHA before accepting a copy. No live-folder read is required.

Only the 47 named non-anchor adapter tensors / 29,211 parameters may train. DOPE stays
`c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814`; all other parameters and
buffers must remain byte-identical after every update and inference. Preserve eval mode, zero
`ctx_feat`, zero `base_prior_cost` and `sos_prior`, scale 0.4208, active fused code and normalizers.
No injected dropout, architecture change, prior fitting, native GP fit or additional training head.

Use AdamW lr 0.0001 / weight decay 0.0001, batch 16, slate 192, clip 1.0, 40 updates per cell.
Targets are the H144 target mapping (passing fitness clipped to [-4, 2], gate failures cost 2).
Loss is SmoothL1 beta 1 plus 0.1 times target-to-prediction KL, temperature 0.2. When masked,
both terms and their reductions use only the 188 unrevealed slots; target softmax is renormalized
on exactly that support. No change to failure labels or cost/utility direction.

Seeds are **147101, 147102, 147103**. For fold k, use effective stream seed `seed + 1000000*k`.
Use the H144 20-round / two retained steps batch stream (consume eight candidate draws per round,
retain the first two); apply the deterministic common-support rule below to each retained draw.
All arms receive identical contexts, candidate sets and Torch seeds within each seed/fold/step.
Three arms × three seeds × five folds = **45 cells / 1,800 planned optimizer updates**.

## Common support and three fixed arms

For each 192-candidate training slate, define anchors A as the original storage indices 0..3,
not the four best-fitness entries. Start support S with the set union of A and the stable top-48
fitness indices (ties by original storage index). Fill to exactly 192 unique indices by the existing
paired stream's uniform sampling without replacement from the complement. If a pool has fewer than
192 entries, fail the inventory gate before any training; do not silently pad or change the budget.

Define B as the four best-fitness members of S, with ties by original storage index. All arms see
exactly S, including all failures. Revealed labels come only from the arm's four declared anchors;
no unrevealed fitness/gate may affect model input features or prediction-time support. Historical
fitness may construct the common *training* support and targets, as in the live recipe.

| Arm | First four positions / revealed candidates | Training loss |
|---|---|---|
| SUPPORT_CTRL | B; remaining S in stable fitness order | all 192 |
| STORAGE_REVEAL | A in storage order; remaining S in stable fitness order | all 192 |
| STORAGE_MASK | exactly STORAGE_REVEAL | unrevealed 188 only |

SUPPORT_CTRL is a matched-support control, **not a byte-identical replay of H144's sampler**:
mandatory inclusion of A changes some training candidate sets. Report that difference. The primary
causal history contrast is STORAGE_REVEAL minus SUPPORT_CTRL. STORAGE_MASK minus STORAGE_REVEAL
isolates the additional loss-mask change. Keep these interpretations separate from total benefit.
If A equals B for a slate, record the null intervention; never replace the slate after seeing that fact.

At assessment, every arm and r882 use the identical complete pool in storage order, anchors A,
full-pool centering, and stable storage-index tie breaking. Candidates are ranked **before** inspecting
unrevealed gate/fitness. A failing selected candidate receives utility -2; it cannot be filtered away.

## Endpoints and fixed decision

Primary is **native top-1 utility**, `-fitness[argmin cost]` for a passing selected candidate and -2
otherwise, including revealed positions as in the unchanged live readout. Higher is better.
For each seed, assemble all 205 continuation-held-out families across the five models, average
contexts within family and give families equal weight. The point statistic is the median of the
three seed mean gains. Use the same estimator for every comparator and bootstrap replicate.

Compute 10,000 paired family-cluster draws with RNG **147141**; draw the same 205 family indices
for all three seeds, recompute the median, and use two-sided 95% percentile intervals. Report gains
versus r882 and the paired SUPPORT_CTRL. One-sided p = `(1 + count(bootstrap gain <= 0))/10001`.
For each non-control arm, take the maximum of its p-values versus r882 and SUPPORT_CTRL (the
intersection claim requires both). Holm-adjust those two arm-level p-values. No other arm can advance.

An arm advances only if all are true:

- Median gain exceeds +0.001 versus **both** r882 and SUPPORT_CTRL.
- Both unadjusted 95% CI lower endpoints exceed zero and Holm-adjusted intersection p < 0.05.
- At least two of three seed gains exceed zero against each comparator; ties reject.
- No more than 10% of the 205 families have negative three-seed mean gain versus r882.
- The three largest positive family contributions account for at most 80% of total positive gain,
  using the seed defining the median r882 gain (ties in seed ordering broken by ascending seed).

All cells must complete; missing cells are INCOMPLETE, never zero. An all-tie comparison fails.
Report every gate, per-seed gains, family win/tie/loss, invalid counts and adapter L2 drift. Report
family Bradley–Terry Elo with r882 fixed at 1500 using the pinned repository Elo implementation,
all contenders in one panel, tie margin 1e-4, prior 0.01, 400 iterations, and family-bootstrap CI; never AUC. Elo is descriptive, not an extra
outcome-selected gate. Report seed adapter/prediction equality and effective replication explicitly.

Descriptive only: verified U4 on the union of anchors and the four lowest-cost **unrevealed** choices,
without gate-filtering before selection; unrevealed top-1; percentage of A=B training slates;
loss/gradient mass attributable to revealed positions; and fraction of choices still on anchors.
These diagnostics cannot rescue a failed primary or choose a new endpoint/arm. Report the masking
contrast even if negative. No DL-Val terminal panel or serving promotion is part of H147.

## Proposed execution envelope and launch requirements

This section is a **proposed cap**, not a compute grant. The current assignment ends at preregistration.
A later implementation/launch assignment must bind actual executor source, exact input closure,
command, host, absolute deadline and single-use ledger before any scoring. No tuning is authorized
between publication of this proposal and that binding. Changing the science requires a visible,
pre-outcome amendment; source/import repairs must not silently change it.

Proposed placement: xbabe4's Linux `jope-ci` environment, using the maintained remote test/launch
path; no xbabe0, no new xbabe2 target directory and no work in another lane. Sequential cells,
nice 19, one math worker, total process-tree OS threads including supervisor <=4, summed RSS
<6,000,000,000 bytes, free disk >=40 GB, owned scratch <3 GB. Pause only our worker group at host
CPU >85% or available memory <16 GiB, and charge pauses to the deadline. Proposed total CPU-time
cap 7,200 seconds and wall cap 10,800 seconds including preprocessing, baseline, cells and readout;
if the admitted host cannot fit the cap, stop INCOMPLETE rather than enlarge it. No retries in this
proposal. No GPU, qualification lock, CI launch, signing, live-loop writes or external promotion.

Required pre-execution checks: manifest SHA disagreement; validation/sealed/holdout path refusal;
family leakage; exact common S across arms; candidate permutation and revealed-label mapping;
zero masked-anchor gradients; target softmax normalization; cost/utility sign; outcome-blind
choice; full frozen-state byte checks; exact cell identity/atomic started-ledger refusal; estimator,
Holm, ties and missing-cell controls; and resource-limit refusal including supervisor threads.
Baseline on TRAIN must reproduce two independent r882 loads exactly before continuation; record
full native utility vectors and hashes. Do not compare a TRAIN family mean with the historical
91-context validation mean 1.1377942. The historical value is not H147's baseline target.

Any later run needs create-only numerical custody under
`s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/H147/<run-id>/`, exact-version
SHA and archive-member recovery checks before owned bulk cleanup, and a separate reviewed closeout.
Keep original shared inputs. Even ADVANCE would mean a follow-up proposal, not automatic deployment.
