# H73 — family-balanced sampling

**Status:** CPU diagnostic complete; future GPU comparison preregistered but not launched.

## Hypothesis

Large families may contribute a larger share of the adapter's context-uniform training objective than their share of contexts, and may contain an outsized share of contexts whose predicted best program fails the gate (`utility = -2.0`). H73 measures this concentration before testing family-balanced weighting or sampling.

## Setup

The diagnostic uses the admitted H50 real-S3-derived packet and the H67 committed training-only roster. It parses only the first `train` array in `selected.json` and stops before the next top-level field; it opens pool and JTT1 table files only for those 1,219 DL-Train context hashes. Each payload hash is checked against its train row. Five whole-family folds use H67's seed-4900 assignment. No DL-Val, fold-B, outer-test, holdout, or sealed payload was read.

The model is the immutable offline H51 copy of round 882 (SHA256 `0084479aa0703667a0f3ccfe2beda60741c204795ec57b7c08542401ec6cdb64`), with DOPE frozen at `c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814` and 29,211 trainable adapter parameters. The live loop was not opened or changed. The model ran only on CPU; no optimizer step was performed.

For each context, the reported H48 CONTROL objective is mean smooth-L1 on `fitness`/`+2.0 gate-failure` costs plus `0.1 × KL(softmax(-target/0.2) || softmax(-prediction/0.2))`. Results include unweighted context loss, the same objective under H48's full-cohort family-balanced context weights, and per-family raw/balanced loss profiles. The `-2.0` selected-context analysis counts contexts where the model's predicted argmin is gate-failing; a separate candidate-level component measures loss on failed program targets.

## CPU diagnostic result

- 1,219 contexts across 205 resolved family groups; 1–36 contexts per family, median 5, context-count Gini 0.217.
- The ten largest families contain 15.42% of contexts. Raw family size and total loss share have Spearman rho 0.388. The ten highest raw-loss families account for 23.55% of unweighted loss mass but contain 10.91% of contexts, showing that concentration reflects both family size and high per-context loss.
- Raw loss shares have Gini 0.569, top-family share 3.34%, and top-ten share 23.55%. Under the full-cohort family-balanced weighting, top-ten family loss share is 19.41%; differing within-family loss levels remain.
- Mean context objective is 14.450 (median 1.318; 90th percentile 62.109; maximum 170.433), so the context-loss mass is strongly right-tailed.
- The selected candidate is a gate failure with utility `-2.0` in 17/1,219 contexts (1.39%). Those contexts contribute 0.98% of unweighted context objective mass. Failed candidate `+2.0` targets contribute 6.42% of the signed candidate-level objective mass.
- Fold means range from 12.846 to 16.675 across the five whole-family partitions. Since r882 was trained on all 1,219 contexts, these are descriptive fold summaries, not out-of-sample CV results.
- The metrics file is 30,810 bytes. Runtime guards observed peak host CPU 61.45%, maximum RSS 4,021,817,344 bytes, and minimum `MemAvailable` 71,461,974,016 bytes.

The distribution is consistent with family-level loss concentration: the highest-loss families carry more mass than a context-count view alone suggests. This scalar-loss diagnostic does not measure per-family gradient norms, establish that family size caused the concentration, or qualify a new model. The future experiment remains conditional on the exposed r882 initialization.

## Reproduction and next step

```sh
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 nice -n 19 \
  python3 research/flywheel_selfplay/h73_family_balanced_sampling/diagnostic.py \
  --input ~/jope-data/h50/input \
  --checkpoint ~/jope-data/h51/ckpt/round-882.pt \
  --roster research/flywheel_selfplay/h67_w4_active_reveal_2plus2/family_fold_manifest.json \
  --output research/flywheel_selfplay/h73_family_balanced_sampling/metrics.json
```

`PREREG.md` freezes the later CONTROL / inverse-square-root weighting / uniform-per-family comparison, seeds 12–14, the family-bootstrap admission rule, and a nine-cell budget projection below 60 minutes. It may be launched only after H57, H59, and H61 are complete and a new owner launch authorization exists. No lock waiter is active.

## Artifacts

- `diagnostic.py`: training-only streaming loader, frozen CPU inference, objective decomposition, and resource guards.
- `test_diagnostic.py`: synthetic tests including proof that the reader stops before parsing the following `dev` field.
- `metrics.json`: compact aggregate and anonymous per-family loss profiles. The profile column order is declared in the JSON; rows are sorted by unweighted family loss share and contain context count, raw loss share, family-balanced loss share, mean context loss, and selected `-2.0` count. No family/context IDs, raw tables, program pools, or checkpoint bytes are included.
