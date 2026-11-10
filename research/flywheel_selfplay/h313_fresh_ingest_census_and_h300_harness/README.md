# H313: fresh-ingest census (cohort v7) and the H300 confirmation harness

**Verdict.**
- **Census: `NO_NEW_CATALOG`.** No fresh, disjoint regression population exists.
- **Harness: READY.** Six frozen ENS_VOTE2 NESTED choosers are locked on DL-Train only, behind a fresh-path gate.

**Not scored.** No fresh population was scored, and none can be: the one-shot confirmation (H314) is blocked on new data,
not on code.

**Records.**
- **Prereg:** [PREREG.md](PREREG.md), commit `6772accf302aa4bf0d3b13059e05199b633de769`, pushed non-force to
  `origin/v1/next` at 10:02:45Z. That was before any census GET (the first, the regression's catalog GET, came at
  10:05:46Z).
- **Owner:** rd_opus [OPUS].
- **Assignment:** `~/nudges/hw_t1000_opus_h313.md`.

## Hypothesis

H310 (DEV91_TRANSFERRED: +0.0153 vs r3175 slot0 over 19 families) needs a once-scored, fresh, disjoint S3 population.
H313 asks two questions:
1. Is there one yet, under the unchanged H286/H299 rules?
2. Can the frozen chooser be locked and gated so that H314 can score it in one shot?

## Part A: census (metadata only; [census_h313.py](census_h313.py))

### A0. Regression: EXACT

H299's `regress_v5.py` and `scout.py` were re-run unchanged (scripts from `origin/v1/next`, reference tree `acb07d1139`,
pins verified) on catalog v246 VersionId `eJuErIwj…`. They reproduced:
- `NO_COHORT`;
- coverage 2,814 / 674 / 14 / 1,248;
- the 2,729-family partition 60 / 480 / 788 / 60 / 692 / 649;
- **all 26 exclusion-set sha256s**, the coverage sha, the flywheel roster counts and the regression receipt;
- the exclusion-sets file sha `0d93ce94…`, equal to H299's manifest.

### A1. Census: no new catalog, no registry additions

| S3 location (`veox-jopedime-datasets-use2`) | objects | modified after 2026-10-30T04:15Z | latest |
|---|---|---|---|
| `catalog/` | v239…**v246** (`LATEST` pointer = `v246`, last modified 2026-10-05; v246 has one version, `eJuErIwj…`) | 0 | 2026-10-05 |
| `full-source-registry/` | 378 | **0** | ≤ 2026-10-30 |
| `split-registry/` | 3,850 | **0** | ≤ 2026-10-30 |
| `datasets/` | 8,819 | 0 | 2026-10-05 |
| `ingest/`, `raw/` | 12, 873 | 0 | 2026-09-27 |
| `regression-hunt/` | 6,088 | 0 | 2026-10-05 |
| `research/corpus-benchmark/` | 131,895 | 0 | 2026-09-30 |
| `jarambit/evo_continuous/` | 172,777 | 49,022 | 2026-11-04 09:57 |

**Result.** There is no catalog newer than v246 and no registry object after the cutoff. Under the unchanged rules the
v246 universe is exhausted (H299), so **0 qualifying families: `NO_NEW_CATALOG`**.
[cohort_v7_manifest.json](cohort_v7_manifest.json) records an empty v7A/v7B, the split rule (salt `H313-v7-split-v1`)
and the catalog identities.

### A2. What would reach the 120 floor

**Neither candidate prefix holds new datasets:**
- **`jarambit/evo_continuous/`** is active, but its post-cutoff objects are genome-evolution "islands" outputs
  (retention receipts, plates, rescreens, bindings to genome/GCI files). No catalog dataset id appears in any
  post-cutoff key (0 / 4,750). One `bindings.json` was inspected to confirm this.
- **`research/corpus-benchmark/`** is a 2026-09-28 algorithm benchmark over existing corpus datasets, including
  BeyondArena, which the sealed rule excludes.

**Observed qualification yield.** H286–H294 qualified 1,153 of 2,609 scanned families (incl. failures): 44.2%.
- H286 200/480;
- H288 300/692;
- H292 295/649;
- H294 358/788.

**Ingest needed.** About **272 new regression families**:
- each a new `dedupe_fingerprint` disjoint from the 2,729 scanned, the 674 base-excluded and all H299 closures;
- not sealed and not BeyondArena;
- ingested to `raw/` → `datasets/` → a **catalog v247**;
- each given an S1-prod GP TRAIN-side run (summary plus teacher, ≥ 2 accept lines).

That should yield 120 qualifying families (about 60 per v7 half). By task, all must be **regression**. By source, no
current source has unused families, so this needs a genuinely new raw source. No ingest was done, and nothing was
written to the datasets bucket.

## Part B: harness ([harness_h313.py](harness_h313.py))

### Six frozen choosers ([lock_manifest.json](lock_manifest.json), sha `5b993129…`, fsynced)

All six are H303 NESTED via H310's code at `819a45aa5e`: lock key 4, the 18-config grid, inner family CV on all 205
DL-Train families, 1e-4 ties, and the TRAIN-only loader. 0 DL-Val rows were read.

| seed | source | chosen (TAU, k, a) | inner gain | fit_lock sha256 |
|---|---|---|---|---|
| 30001 | H310 lock (S3 VersionId `1JWgsGRc…`, sha = H310 receipt) | 0.10, 5, 2 | 0.05640 | `a06cb7f0…` |
| 30002 | H310 lock | 0.05, 5, 2 | 0.05496 | `d78922f4…` |
| 30003 | H310 lock | 0.05, 5, 2 | 0.04473 | `40f02d99…` |
| 30004 | fitted here | 0.10, 5, 2 | 0.04535 | `3edf0ec2…` |
| 30005 | fitted here | 0.10, 5, 2 | 0.05818 | `e684c199…` |
| 30006 | fitted here | 0.05, 5, 2 | 0.06584 | `396dd73a…` |

The member seeds for 30004–30006 come from the H300 rule `core.stream_seed(root, "H300_ENS_VOTE2_MEMBER", m)`, checked
first against H302's table for 30001–30003:
- 30004: 1952251935, 1554721030, 514753433;
- 30005: 1316784594, 642759331, 744149750;
- 30006: 1442930024, 1192635410, 1213673495.

These seeds were fitted independently of H312.

### Fresh gate

An audit hook refuses any open of a fresh data path (`fresh_pools`, `fresh_cohort`, `fresh_top8`, `fresh_v7`,
`cohort_v7`, `/v7/`) until `lock_manifest.json` exists **and** all 6 fit_lock shas verify. A tampered lock closes the
gate again. [test_h313.py](test_h313.py) tests this, along with the decision rule, the exact and Monte Carlo sign-flip
test, the 6-seed summary, the v7 split, and the census regression comparator (4 tests).

### Decision rule draft for H314 (ties never win)

**FRESH_CONFIRMED** iff all of:
- the median over 6 seeds of the family-equal gain vs **r3175 slot0** is > +0.001;
- the pooled whole-family bootstrap LCB95 is > 0 (RNG 302150, 10k);
- the family sign-flip p is < 0.05 on ≥ 4/6 seeds (exact for ≤ 20 families, else Monte Carlo 200k, RNG 313150);
- invalid picks are ≤ r3175's on each seed;
- harmed is ≤ 15% on each seed.

**Elo label:** "reference = r3175 slot0 (fresh v7)".

### Dry run (DL-Train P0 fold 0 pseudo-fresh: 57 families / 335 contexts; H303's out-of-fold fold-0 locks, 3 seeds)

**Harness regression.** Seed 30001's picks equal H303's `18fdb2538cea…`, so the harness reproduces H303 exactly. The
TRAIN-only loader read 2,811,012 rows, 0 of them DL-Val.

**Outcome.** Reference = r882 slot0 (DL-Train pseudo-fresh dry run):
- gains +0.0269 / +0.0304 / +0.0288 (median +0.0288, pooled CI [+0.0075, +0.0512], pooled LCB95 +0.0109);
- sign-flip p 0.0027 / 0.029 / 0.0054 (Monte Carlo);
- invalid 4/5/4 vs 5;
- harmed 8.8% / **15.8%** / 12.3%;
- Elo 1741 / 1687 / 1686.

**Rule result.** With a 2/3 sign-flip threshold for 3 seeds, the dry run comes out `FRESH_NOT_CONFIRMED`, solely from
**c5 (harmed > 15% on seed 30002)**.
- The **15% harm gate is the binding criterion** at about 57 families. H303/H308 harmed shares run 6–12%, so a
  ~120-family v7 cohort may still hit it on one of six seeds.
- That is a property of the draft rule for H314's owners to weigh. H313 does not change it.
- This is a pipeline exercise on already-analysed DL-Train OOF, not new evidence.

## Deviations (declared)

1. **Lock launch attempts.** The first three lock launches stopped within seconds, before any fit lock and without
   reading any fresh or dev data. All logs are in the S3 tar.
   - **Attempt 0:** the gate needle `fresh_` matched this study's own folder name. The needles were narrowed to fresh
     data paths.
   - **Attempt 1:** H310's frozen dev guard also refuses `fresh_`. The lock and dry-run phases then ran from
     **byte-identical copies** of the code in `~/jope-data/h313-opus/code` (sha recorded in
     [manifest.json](manifest.json) and equal to the committed file).
   - **Attempt 2:** H302's member-seed table stops at 30003, so seeds 30004–30006 got the H300 stream rule, verified on
     30001–30003.
2. **Dry-run launch 0** failed because `aws` was not on the systemd unit's PATH. The reference tars were pre-placed by
   exact-version GET with sha check, and the code verifies them.
3. **PREREG wording.** The PREREG said the dry-run fold has 46 families. P0 fold 0 actually has 57 test families / 335
   contexts, as H305 recounted. The fold rule is unchanged.
4. **Code published after the prereg.** The code and tests were published with the closeout. The PREREG specified their
   behaviour before any census GET.

## Next step

1. **H314 is blocked on data, not code.** A new raw source must be ingested to catalog v247: about 272 new regression
   families with S1-prod GP runs, giving about 120 qualifying.
2. **Then** run the unchanged H299/H313 census to freeze v7, compute r3175 GPU-exact costs for the v7A contexts, and
   score once with the 6 locks in this lock manifest.
3. **Before H314.** Owners should confirm the 15% harm gate (the binding criterion in the dry run) once H312 closes.
4. **Until then,** DL-Train/dev91 evidence stands at: H308 +0.045 vs r882 OOF; H310 +0.0153 vs r3175 on dev91
   (concentrated).

## Files and custody

| File | Contents |
|---|---|
| [PREREG.md](PREREG.md) | frozen protocol (commit `6772accf30`) |
| [census_h313.py](census_h313.py) | regression comparator, census listing, verdict, ingest estimate |
| [harness_h313.py](harness_h313.py) | locks, fresh gate, 6-seed statistics, decision, dry run |
| [test_h313.py](test_h313.py) | 4 data-free tests |
| [metrics.json](metrics.json), [cohort_v7_manifest.json](cohort_v7_manifest.json), [lock_manifest.json](lock_manifest.json), [s3_manifest.json](s3_manifest.json), [manifest.json](manifest.json) | metrics, empty v7 manifest, 6 lock shas, S3 VersionIds, pins |

**S3:** `s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/H313/` holds:
- `h313_harness_census.tar.gz` (VersionId `1XFeNFRwVIsL0Zd8WAoIjbzjyiaixkcy`, sha `c42b656f…`, 136 MB), with the 6
  locks and members, census evidence and all attempt logs;
- `lock_manifest.json`, `dryrun.json`, `census.json` and `cohort_v7_manifest.json`.

Every object passed an exact-version GET and sha match before scratch was deleted.
