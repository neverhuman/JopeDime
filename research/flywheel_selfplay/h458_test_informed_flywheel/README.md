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
| `train_h458.py` | trainer: exposure-indexed schedule and checkpoints, measured duplicate ledger, trim-and-retry OOM policy, optional data workers on other hosts |
| `probe_h458.py` | SEARCH-only out-of-fold utility under the evaluation's fold pipeline (reads no audit label) |
| `cols_h458.py` | scored column builder for specialists (GP anchor off; canonical-slot path; cross-fit routing; closures `recall`, `skill-xfit5`, `fair`, `fresh-probe` with their refusals); `_load_col_block` layout + sidecar written per context |
| `selftest_h458.py` | 17 CPU gates; receipt in `/home/ubuntu/jope_plan/evidence/H458/trackD/` |
| `launch_specialist.sh` | resource-capped launch of one specialist run from the staged runtime copy |
| `pilot_driver.sh` | trains the slot arms of one family side by side, then probes every milestone checkpoint |
| `queue_h458.sh` | one specialist per family, a few at a time, with resume and SEARCH-only probes |
| `stage_h458.sh` | create-once, read-only runtime copy with `CODE.sha256`, pinned dependencies and vendored modules |
| `sync_remote_h458.sh` | prepares another host for remote data workers and verifies every byte there |
| `PREREG.md` | hypothesis, fixed treatment, pilot arms, gates, looks, resources, retention |

The evaluation code is reused by import from the pinned stage folders (`cols_104m_e8_gpresid_h452.py` and its
dependencies); nothing is copied. The staged runtime copy `/home/ubuntu/h458-stage/code` carries `CODE.sha256`,
`deps_h458.json` and frozen copies of the unpinned `h452-stage` modules under `vendor/`.

## Run

```bash
python3 -B family_pool_h458.py build                      # once; create-once, then read-only
python3 -B selftest_h458.py                               # CPU; must be 16/16 before any training on GPU
./launch_specialist.sh m_60b6_rand 60b608db rand <init.pt> ema
python3 -B probe_h458.py --ckpt <run>/spec_e01000.pt --families 60b608db --out <probe.json>
python3 -B cols_h458.py --closure recall --arm canon --ckpt <run>/spec_e02000.pt --families 60b608db --out-dir <block>
REMOTE_WORKERS=xbabe3:16 WORKERS=0 ./launch_specialist.sh <run> <family8> canon <init.pt> ema   # episodes built on xbabe3
```

## Status (2026-11-14, pilot; SEARCH-only out-of-fold probe, three-mode mean u, no audit label read)

| Family | Arm | Init | 1,000 exposures | 2,000 exposures | Gate line (D-2) |
|---|---|---|---|---|---|
| 60b608db | recall, random slots | 0.424 | 0.463 | 0.513 | 0.68 |
| 60b608db | recall, canonical slots | 0.413 | 0.601 | 0.851 | 0.68 |
| 90a58676 | recall, random slots | 0.672 | 0.904 | 1.202 | 1.10 |
| 90a58676 | recall, canonical slots | 0.679 | 2.455 | 3.197 | 1.10 |

Cross-fitted skill arm on 60b608db (rows the weights never trained on, gain over the fair init on the same rows):
random slots +0.01 to +0.02 at 1,000 exposures and fading; canonical slots -0.02 to -0.04 at 1,000 and -0.09 to
-0.15 at 2,000. The recall gain is lookup, and a canonical-slot specialist is worse than its init outside its pool.
Canonical slots are therefore the scored path of the recall arm only, and `cols_h458.py` refuses closure `recall`
on any context that is not fully inside the pool.

First official look (2026-11-14 23:49Z, arm `h458d.RECALL_CANON_E2000_PILOT20`, pre-registered, one of four looks): the
legacy stack with the canonical-slot recall columns is above the best of all 33 field entrants on 19 of the 20 pilot
contexts (gate: 16) and beats Kumo Large on 19; the same stack without them beats it on 2. It is recall of labels
the specialists trained on, it covers 20 of 91 contexts and it has no Elo. Receipt:
`/home/ubuntu/jope-data/h458-quarantine/looks/pilot20_20261114T2352Z/`.

Skill on rows no model saw is measured separately on Track S's fresh-row probe board (`dev91-freshprobe-v1`: rows in
no bank and no pool, 10 families): `cols_h458.py --closure fresh-probe` scores each specialist and its
pre-fine-tune checkpoint on the same contexts. A probe block can never be stacked on the real board.

Where the wave stands against the field is in the Elo report, `/home/ubuntu/jope_plan/evidence/H458/trackD/elo_report_20261114/`
(revision 7: fifth of 34 on the board, seventh of 23 against external models only; every number in-corpus or fair
closure as labelled).

Full results are recorded in `metrics.json` once the gates in `PREREG.md` have been read for all families.
