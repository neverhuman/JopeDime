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
| `gp_chain_h458.py` | S9 staged residual GP: a loop over the H453 evolution engine's own entry point (single run, residual chains, evolution out of fold); reads no audit label |
| `selftest_h458.py` | 18 CPU gates; receipt in `/home/ubuntu/jope_plan/evidence/H458/trackD/` |
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
-0.15 at 2,000. From this control I predicted that a canonical-slot specialist is worse than its init outside its
pool. The fresh-row board refuted that (below). Canonical slots stay the scored path of the recall arm only, and
`cols_h458.py` refuses closure `recall` on any context that is not fully inside the pool.

First official look (2026-11-14 23:49Z, arm `h458d.RECALL_CANON_E2000_PILOT20`, pre-registered, one of four looks): the
legacy stack with the canonical-slot recall columns is above the best of all 33 field entrants on 19 of the 20 pilot
contexts (gate: 16) and beats Kumo Large on 19; the same stack without them beats it on 2. It is recall of labels
the specialists trained on, it covers 20 of 91 contexts and it has no Elo. Receipt:
`/home/ubuntu/jope-data/h458-quarantine/looks/pilot20_20261114T2352Z/`.

Second and last official look of Track D (2026-11-15 03:33Z, arm `h458d.RECALL_CANON_E2000_ALL91`, pre-registered, look 2
of 4). IN-CORPUS RECALL, the specialists were trained on these audit rows with their labels: the legacy stack with
one specialist per family is 2343.8 Elo (95% interval 2181 to 2619), rank 1 of 34, +272.6 over Kumo Large in the
same fit, and first of 23 against the 22 external models only (405.2 of 418 family-games). The same stack without
the recall block is 1859.0, rank 5. It is above every entrant on 66 of 91 contexts and beats Kumo Large on 74. The
registered gate D-4 is not met: 11 of the 16 target families are swept, 13 were asked. Receipt:
`/home/ubuntu/jope-data/h458-quarantine/looks/all91_20261115T0327Z/`, evidence
`/home/ubuntu/jope_plan/evidence/H458/trackD/look2/`.

Skill on rows no model saw is measured separately on Track S's fresh-row probe board (`dev91-freshprobe-v1`: rows in
no bank and no pool, 10 families): `cols_h458.py --closure fresh-probe` scores each specialist and its
pre-fine-tune checkpoint on the same contexts. A probe block can never be stacked on the real board. Result
(2026-11-15 03:12Z, three pre-registered arms, scored once): the specialists are +0.474 u above their init (family
mean; 34 of 46 contexts, 7 of 10 families, sign-flip p 0.055 two-sided). The pool still holds dev91 audit labels, so
the number is in-corpus; it says the gain is not only lookup, and nothing about a family without a specialist.

Where the wave stands against the field is in the Elo report, `/home/ubuntu/jope_plan/evidence/H458/trackD/elo_report_20261114/`
(revision 7: fifth of 34 on the board, seventh of 23 against external models only, before the recall look; every
number in-corpus or fair closure as labelled).

S9, staged residual GP (owner's ask of 2026-11-15): `gp_chain_h458.py` runs GP, fits the next GP run to the residual of
the running sum, and repeats. The engine's own curve says depth overfits (Ultra-GP alone: 1835 Elo at 50 generations,
1815 at 1,000), so the arms spend a tier's generations on honesty and diversity instead: residual chains with
cross-fitted stage columns, and evolution out of fold, where K outer folds each run the chain on the other folds and
the held-out fold is predicted by a pipeline that never saw it. Every arm is read first on SEARCH rows and on the
fresh-row probe board; the PREREG, the budget table and all scoring are Track S's (S9). An out-of-fold arm takes a `--base-block` column only when the block also holds it nested on that arm's own outer folds (`nest_lab_`, `nest_q_`, `nest_a_`, compared as sets of row indices); a base that is merely out of fold on all SEARCH rows is refused, because it would carry the held-out fold's labels into the fit (probe ledger line 45 measured what that does to a stack: 92% of the weight and 7 losses of 10).

Full results are recorded in `metrics.json` once the gates in `PREREG.md` have been read for all families.
