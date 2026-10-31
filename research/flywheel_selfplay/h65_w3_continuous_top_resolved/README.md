# H65 — W3 continuous outcomes and top-resolved loss

**Status: CPU preparation only.** This directory does not contain training outcomes or a study
result. The W3 outcome export and W1 winner configuration are not yet available under the
admission contract below. The family-fold projection was smoke-checked against W1's existing
train-only manifest; no live-S3 outcome smoke, model fit, GPU work, or scoring was run.

## Hypothesis (verbatim from `rd_breakthrough/PROPOSALS.md`, W3)

> Separate prediction of measured quality and feasibility preserves useful supervision that a
> flat failed-candidate target discards, while top-resolved comparisons spend gradient on
> consequential choices.

## Arms and protocol (verbatim from W3)

> r882 continuity plus matched models with (a) current SmoothL1/KL, (b) H58-D loss, (c) a
> top-resolved listwise target, and (d) decomposed quality/feasibility heads. The two primaries are
> c−b and d−a; this is not a post-hoc best-of-four contest. H59's completed clipping/normalization
> result must be incorporated before launching overlapping work.
>
> Same history/prior contract, seeds 701–703, five folds, 40 terminal updates. For c, normalize
> passing regret by `max(F_(16)−F_(1), .05)`, clip to [0, 6], and keep H58's failure handling; if
> fewer than 16 pass use the last passing order statistic, with an explicit no-passer path. For d,
> use measured mean log-ratio on completed rows, passing residual `F−.9*logratio`, and a gate label;
> mask missing/unreconstructable components. Fit-only calibration maps these to expected original
> utility, not an asserted confidence bound. Keep width/compute-matched scalar and shuffled-auxiliary
> controls in the registered implementation protocol.

## Advance and kill gates (verbatim W3 plus common contract)

W3's measured metrics, semantics, and success/kill conditions:

> Original gate, Elo and raw val_reg_inner remain unchanged. Advance on the common gate and no
> excess gate-failure rate over control. Kill if improvements exist only in auxiliary loss, if
> shuffled auxiliary targets match the gain, or if near-pass failures are promoted by changing
> evaluation semantics. A loss-only unit-consistency check must preserve zero-update choices.

The common advance gate is a **positive adjusted family-mean lower bound above +.001**, median
family delta ≥ 0, and at most 10% of families losing more than .01, against the strongest matched
control. W3 uses legacy-selection raw family utility as its primary track; verified U4 is reported
separately. Family Elo and raw `val_reg_inner` remain separately named. These are future study
criteria, not results from this preparation.

## Data and fold provenance

The committed [`fold_manifest.json`](fold_manifest.json) is derived only from the W1
`jope.w1.fold-manifest/v1` metadata at preparation time. Its source SHA256 is recorded in that
file. The source declares 1,219 DL-Train contexts, 205 families, fold seed 4900, zero DL-Val
contexts loaded, zero excluded payloads opened, and zero outer rows delivered. H65 retains the
same five family folds and original training context IDs; it adds no labels or outcomes. The
projected metadata also binds the selected/champion identities, frozen DOPE and r882 checkpoint
identities, and the source's resolved dataset/table alias counters.
The preparation command reads the offline fold metadata at
`/home/ubuntu/jope-data/w1/fold_manifest.json`; it does not fetch new S3 objects.

The common contract admits `program/fitness/logratio/gate` arrays only with completed-status
provenance and exact fold components where already admissible. It prohibits raw mixed S3 scores.
The inventoried retained-score JSONL contains outer-test fields and is not an eligible raw input;
a separately prepared field-allowlisted, role-verified export is a prerequisite. The W3 export is
not currently available, so this PR does not read it or claim a W3 outcome smoke. The separate
catalog plumbing smoke below is not a W3 row or outcome.
`S3_DATASET_CATALOG.md` governs dataset selection but does not substitute for W3 outcome-row
admission. No DL-Val, outer-test, sealed, holdout, or meta-test data is accessed.
The H59 dependency is unresolved for launch: current H59 preregistration says no outcome work has
run, while W3 requires incorporating H59's completed clipping/normalization result before
overlapping study work.

## Runner contract and limits

`runner.py` builds the family-fold manifest and emits a CPU-only, `preparation_only` plan. Its
`plan` and `admission` commands require `--comparator-binding` pointing to
`comparator_binding.json`. The runner verifies that W1 remains `NO_ADVANCE` with no accepted
winner or prior, and binds the strongest comparator's pinned metrics identity for comparisons
only. This comparator is not a W1 winner or prior. `admission` also validates the pinned H59
disposition and H113 preregistration. Actual training is deliberately not implemented here. Do
not launch overlapping W3 work until the W3 row export passes admission and the study's separate
GPU gate is approved.

The separate `smoke-s3` command verifies one catalog-v246 real-regression **training** blob
(dataset `b9be38a2d975279c`, 85 rows × 10 columns) with read-only S3 GET, SHA256, and CSV-shape checks.
It does not request the test blob or feed those general-corpus rows into W3. Its committed receipt
is input-plumbing evidence only and does not remove the W3 outcome-export blocker. The smoke uses
the host AWS provider for GET-only requests; it does not create or overwrite S3 objects.

Local checks:

```sh
nice -n 10 python3 -m unittest discover -s research/flywheel_selfplay/h65_w3_continuous_top_resolved -p 'test_*.py' -v
nice -n 10 python3 research/flywheel_selfplay/h65_w3_continuous_top_resolved/runner.py prepare-folds --w1-fold-manifest /home/ubuntu/jope-data/w1/fold_manifest.json --output research/flywheel_selfplay/h65_w3_continuous_top_resolved/fold_manifest.json
nice -n 10 python3 research/flywheel_selfplay/h65_w3_continuous_top_resolved/runner.py plan --help
nice -n 10 env AWS_PROFILE=veox-jepson-prod python3 research/flywheel_selfplay/h65_w3_continuous_top_resolved/runner.py smoke-s3 --receipt research/flywheel_selfplay/h65_w3_continuous_top_resolved/DRY_RUN_RECEIPT.json
git diff --check
```
