# H145 — larger family-grouped TRAIN assessment

**Complete. Registered proposal: `LARGER_PANEL_SHADOW_TRIAL`. No live gate change.**

The 1,219-context / 205-family panel ranks the saved snapshots by equal-family native
utility as **R782 > R785 > R882 = LATEST(R2555) > R859**. The published 91-context
flywheel order is **R859 = R882 = LATEST > R785 > R782**. Spearman over the four
unique functions is -0.949; this is descriptive and compares different family/context weighting.
LATEST was frozen before outcomes and is byte-identical in model state to R882.

This is **selection cross-fitting of models already trained on these families**.
It is not unseen-family model validation. The proposal is a prospective shadow
trial of the larger development panel, with future fit/assessment isolation; no
live-loop, serving, Q1 or guidance-default change was made or qualified.

## Registration and construction

[PREREG.md](PREREG.md) merged in [#1269](https://github.com/neverhuman/JopeDime/pull/1269)
as `6fc5cd8c74736cc91d32a5fb9f5d36a72fc3f6e9` at 01:49:02Z, before the first
H145 construction/outcome read at 2026-11-01T01:57:56.177673+00:00. Execution source:
`7c6e4eecb4cf873b277a0272e6a72f109652f058`. The latest snapshot was R2555, SHA-256
`f566a2f08ef215f5afc328c0d94374498a06d4f9a8c35b07ae5f4d4ebd6249a6`.
The initially listed R2553 rotated away before snapshotting; no tensor or outcome
was decoded, and the final R2555 pin was fixed before registration.

All 1,219 TRAIN regression identities match the approved real-S3 catalog and the
pinned selected roster. Five family folds each hold out 41 of 205 families; every
context is assessed exactly once. Fit/evaluation context counts are
930/289, 994/225, 1013/206, 976/243, 963/256. Every split has zero family or
context intersection. The 91-TRAIN-context noise control, selected by identity
hash before outcomes, contains **75 families**; it is not the original 19-family
DL-Val panel. Full roster, fold lists and hashes are retained outside Git.

The endpoint is the frozen first-argmin native cost selection, minus fitness for
a valid selection and -2 for an invalid selection. No unrevealed gate filtering
occurs before selection. H139/H143 frozen TRAIN outputs were recovered from
versioned S3 custody, verified against source/checkpoint/protocol/roster hashes,
and re-ranked; shared R882 values agree exactly. **Zero new model forwards or
optimizer updates** were performed. Only TRAIN subtrees were decoded from mixed
custody JSON; DL-Val outcome subtrees were skipped lexically. The old ranking uses
published aggregate values, including R785's rounded 1.137595.

## Paired results and rank stability

Gain is adapter minus R882, higher is better. CIs and SEs resample whole families
jointly across adapters (10,000 draws). Bootstrap first-place probabilities count
four unique functions; LATEST is an alias, not another independent contender.
Elo uses family-grouped Bradley–Terry, R882=1500, with 1,000 family-bootstrap draws.

| Adapter | Mean gain [95% CI] | Paired SE | P(first) | Elo [95% CI] | Invalid picks |
| --- | --- | --- | --- | --- | --- |
| R782 | +0.013260 [+0.002485, +0.027479] | 0.006435 | 0.9918 | 1501.91 [1497.13, 1506.66] | 13 |
| R785 | +0.001018 [-0.000657, +0.003518] | 0.001101 | 0.0061 | 1499.62 [1497.10, 1502.13] | 16 |
| R882 | +0.000000 [+0.000000, +0.000000] | 0.000000 | 0.0018 | 1500.00 [1500.00, 1500.00] | 17 |
| LATEST (= R882) | +0.000000 [+0.000000, +0.000000] | 0.000000 | 0.0018 | 1500.00 [1500.00, 1500.00] | 17 |
| R859 | -0.000131 [-0.000659, +0.000254] | 0.000234 | 0.0003 | 1500.08 [1498.73, 1501.44] | 17 |

Absolute native utilities, with the same equal-family weighting:

| Adapter | Mean utility | Bootstrap SE | 95% CI |
| --- | --- | --- | --- |
| R782 | 1.398615 | 0.136611 | [1.153719, 1.690033] |
| R785 | 1.386373 | 0.137106 | [1.140610, 1.677585] |
| R882 | 1.385355 | 0.137170 | [1.139390, 1.676938] |
| LATEST | 1.385355 | 0.137170 | [1.139390, 1.676938] |
| R859 | 1.385224 | 0.137150 | [1.139344, 1.676586] |

R782 is chosen on all five fit partitions. Its pooled held-out selected-minus-R882
gain is **+0.013260**, SE **0.006435**, CI **[+0.002485, +0.027479]**. Two of its
five individual held-out folds have negative gains. Its native-utility advantage
is conditional on this development cohort; its Elo interval **includes 1500**.
Native utility magnitude and pairwise wins are different estimands.

## Noise-size control and registered decision

Full/small paired SE ratios: **R782 1.906971**, **R785 0.399000**, **R859 0.167422**.
The preregistered median is **0.399000 <= 0.75**, with three nonzero-denominator
comparisons. This is **not uniform noise reduction**: R782's paired SE is higher
on the full panel. The small panel happens to have less R782 gain variability;
a larger panel can expose rare invalid-selection events that a small sample misses.
Nor is the 75-family TRAIN control a precision estimate for the historical
19-family validation panel.

All registered proposal gates pass: complete panel, median SE ratio, R782 first
probability 0.9918 >=0.80, five-of-five selection, cross-fit gain >0.001 and positive
family-bootstrap lower bound. The decision remains as registered; no threshold or
verdict was changed after observing results. Repeated folds and aliases are not
independent model replications.

**Proposal (documentation only):** trial the larger TRAIN panel as an additional
shadow development-gate diagnostic, preserving family-cluster uncertainty and
invalid-pick reporting. Do not switch the live promotion gate from this study.
Before a live change, use future model fitting that excludes assessment families,
prospectively test cliff robustness and winner-specific precision, and establish
qualification under the existing release/Q1 rules. An apparent gain on familiar
TRAIN families does not establish generalization or causally identify selection bias.

## Verification, resources and custody

- Eight data-free controls pass on xbabe3, including family isolation/weighting,
  duplicate/nonfinite refusal, lexical DL-Val exclusion and exact canonical Elo
  agreement on repeated-family resamples. Full-panel Elo also matches the canonical
  tracker at two-decimal publication precision. These are bounded controls, not a
  product or model qualification suite.
- Checkpoint SHA-256 pins pass for all five arms. All **222 non-adapter tensors**
  are byte-identical to R882; DOPE identity is `c51a56d0…`. The full R2555 state
  matches R882. Cached forward DOPE receipts match before and after. No DOPE write,
  GPU, signing, dataset duplication, live-directory write or protected-data use.
- Analysis ran on **xbabe2**, nice 19, one math thread, wall
  **20.386s**, parent process CPU **4.586s**,
  peak RSS **559,752 KiB**. Child AWS-client CPU is not separately
  reported; the cumulative CPU guard includes completed children. No guard abort
  or study retry occurred. CUDA remained uninitialized. Only off-host data-free
  tests ran; no Cargo build/large target was created locally.
- Runtime inputs, per-context/family arrays, 10,000 bootstrap means, 1,000 Elo draws,
  fold rosters, checkpoint identities and versioned input receipts:
  `/home/ubuntu/jope-data/h145/`. Detailed command/exit/source/hash receipt:
  `/home/ubuntu/jope_plan/evidence/H145/RESULT-RECEIPT.json`. Source caches remain
  recoverable from their pinned H139/H143 S3 versions; the latest snapshot is
  retained separately on xbabe1 and in the bounded H145 input directory.
- [metrics.json](metrics.json) is copied byte-for-byte from the owning run. Source
  [run.py](run.py), [analysis.py](analysis.py), [test_h145.py](test_h145.py) and the
  unchanged preregistration preserve the recipe. The exclusive `RUN_STARTED.json`
  refuses a second study start; do not reset its budget or overwrite evidence.
