# H458 — test-informed flywheel, Track D: evaluation-matched ICL specialists

**In-corpus wave.** Every dev91 number produced here carries `in_corpus_dev91_disclosure = true` and
`canonical_promotion_forbidden = true`. The family pools contain each context's SEARCH and AUDIT rows, so a
full-pool specialist's utility is mostly recall of labels it was trained on (closure class `recall`). It is reported
beside the fair anchors and beside a leave-context-out `skill` arm. No sealed cohort is opened; inference reads only
a context's SEARCH rows and selects by CV on them. See `PREREG.md`.

## Why

The ICL tower adds about +5 Elo to the legacy stack on `dev91-audit-v1`. Its training does not resemble evaluation:
contexts of about 150 rows against 485–1,183, bootstrap-duplicated rows, about 65% synthetic columns, the table's own
target in 44% of real episodes, a dev91 share near 2%, and neither rank-Gauss targets nor the CW/struct feature
blocks that the column builder feeds. H458 trains on episodes built by the column builder's own functions.

## Files

| File | Role |
|---|---|
| `paths_h458.py` | `sys.path` order, `JOPE_REF_CODE` neutralised, dependency sha pins, write-path guard |
| `family_pool_h458.py` | de-duplicated union of each dev91 family's tables; leave-context-out view |
| `episodes_h458.py` | evaluation fold pipeline as a training episode; unique-row sampling; slot arms; batch sizing |
| `train_h458.py` | trainer: exposure-indexed schedule and checkpoints, measured duplicate ledger, loud OOM policy |
| `probe_h458.py` | SEARCH-only out-of-fold utility under the evaluation's fold pipeline (reads no audit label) |
| `selftest_h458.py` | 14 CPU gates; receipt in `/home/ubuntu/jope_plan/evidence/H458/trackD/` |
| `launch_specialist.sh` | resource-capped launch of one specialist run from the staged runtime copy |
| `PREREG.md` | hypothesis, fixed treatment, pilot arms, gates, looks, resources, retention |

The evaluation code is reused by import from the pinned stage folders (`cols_104m_e8_gpresid_h452.py` and its
dependencies); nothing is copied. The staged runtime copy `/home/ubuntu/h458-stage/code` carries `CODE.sha256`,
`deps_h458.json` and frozen copies of the unpinned `h452-stage` modules under `vendor/`.

## Run

```bash
python3 -B family_pool_h458.py build                      # once; create-once, then read-only
python3 -B selftest_h458.py                               # CPU; must be 14/14 before any training on GPU
./launch_specialist.sh m_60b6_rand 60b608db rand <init.pt> ema
python3 -B probe_h458.py --ckpt <run>/spec_e01000.pt --families 60b608db --out <probe.json>
```

## Status

Results are recorded in `metrics.json` once the pilot gates in `PREREG.md` have been read.
