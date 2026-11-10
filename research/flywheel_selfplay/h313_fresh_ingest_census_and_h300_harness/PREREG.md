# H313: fresh-ingest census (cohort v7) and the H300 confirmation harness for frozen ENS_VOTE2 NESTED

**Owner:** rd_opus [OPUS]. **Assignment:** `~/nudges/hw_t1000_opus_h313.md` (Grok Bot H-wave pass, 2026-11-04 10:00Z,
decided). **Claim:** `~/jope_plan/ns/receipts/H313_claim.txt`.

**Scope.**
- Metadata plus DL-Train only. **No fresh, sealed, holdout or meta_test feature or label rows, no DL-Val feature rows
  and no DL-Test** are read.
- No fresh population is scored: the one-shot scoring is H314, decided after H312 closes.
- xbabe2 CPU only, with no GPU and no neural forward or update; DOPE `c51a56d0` is untouched.
- No S3 writes outside `experiments/rd_breakthrough_20261030/H313/`, and no writes to the datasets bucket.

**Publication.** This prereg is pushed non-force to `origin/v1/next` **before any census GET** (including the regression's
catalog GET).

## Why

H310 is DEV91_TRANSFERRED: median +0.0153 vs r3175 slot0, over 19 families. A breakthrough claim needs a once-scored,
fresh, disjoint S3 population (the H299/H300 path). H299 found NO_COHORT in catalog v246. H313 builds the census and the
harness so that confirmation can run as soon as H312 closes.

## Part A: census ([census_h313.py](census_h313.py))

**A0. Regression first.** H299's pinned `regress_v5.py` and `scout.py` are re-run **unchanged**. They come from
`origin/v1/next`, with the reference tree at H299's `reference_commit` `acb07d1139`, on catalog v246 VersionId
`eJuErIwj0Pgu.lAgflPcWvvRKyuwptLC`. They must reproduce:
- `NO_COHORT`;
- catalog coverage 2,814 / 674 / 14 / 1,248;
- the 2,729-family eligible-universe partition: H283 60, H286 480, H294 788, H285 60, H288 692, H292 649;
- every exclusion-set `set_sha256` in H299's `metrics.json`.

Any mismatch means **REGRESSION_FAIL and a stop**.

**A1. Census (metadata only).**
1. A version listing of `s3://veox-jopedime-datasets-use2/catalog/`. If a catalog **newer than v246** exists, take an
   exact-version GET of the latest catalog JSONL. Catalog records are metadata: ids, task, hashes and provenance.
2. A version listing of the `full-source-registry/v1/` and `split-registry/v1/` prefixes, wherever they exist in the
   datasets bucket. Objects last-modified after 2026-10-30T04:15Z get exact-version GETs, if they are registry metadata.
   Dataset bodies are never read.

**Rules.** The **unchanged H286/H299 rules and all H299 exclusions** apply:
- the 1,219 / 91 / 93 flywheel rosters and the 1,780-context closure;
- H287/H291/H295 lineage;
- all prior scanned families including failures, and cohorts v1–v6;
- P3 heldout/dev/sealed_excluded coarse closures;
- catalog sealed/BeyondArena.

**H286 qualification is unchanged:** an admitted S1 TRAIN-side summary plus teacher, ≥ 2 accept lines. A catalog
regression family without an S1-prod GP run cannot qualify. Such families are counted as "ingested but not run", by
source.

**Split.** Qualifying families are split A/B by SHA256(UTF8("H313-v7-split-v1") NUL UTF8(family_key)), last byte parity:
0 = v7A, 1 = v7B.

**Verdict:**
- **COHORT_V7_READY** if ≥ 120 qualifying families;
- **COHORT_V7_SHORT** (n, by source) if 1–119;
- **NO_NEW_CATALOG** if no catalog newer than v246 and no qualifying registry additions.

**Output.** `cohort_v7_manifest.json` holds ids and hashes only.

**A2.** If the verdict is SHORT or NO_NEW_CATALOG, report the ingest volume needed to reach 120 (families by task and
source), plus candidate S3 prefixes (`research/corpus-benchmark`, `jarambit/evo_continuous`) under H286 rules. This is a
metadata listing only, with no ingest.

## Part B: H300 harness ([harness_h313.py](harness_h313.py))

**Frozen chooser.** H303 NESTED ENS_VOTE2 via the H310 code at `819a45aa5e`:
- lock key 4;
- the 18-config grid TAU {0.05, 0.10, 0.15} × k {5, 6, 7} × a {2, 3};
- inner 4-fold family CV over all 205 DL-Train families;
- 1e-4 conservative ties;
- the TRAIN-only loader.

**Root seeds 30001–30006.** These include H312's 30004–30006, fitted independently in this lane.
- **Seeds 30001–30003:** H310's archived locks (S3 H310 tar VersionId `1JWgsGRc…`), sha-verified against H310's lock
  receipt. No refit.
- **Seeds 30004–30006:** fitted here with H310's lock code, on DL-Train only.

**Locks.** A lock manifest holds the 6 fit_lock sha256s and is fsynced.

**Fresh gate.** An audit hook refuses any fresh-path open (`fresh_`, `cohort_v7`, `/v7/`) until the lock manifest exists
**and** all 6 fit_lock shas verify. A test shows that fresh-path reads raise until all 6 locks exist.

### Decision rule draft (for H314; ties never win)

**FRESH_CONFIRMED** iff all of:
1. The median over the 6 seeds of the family-equal gain vs **r3175 slot0** is > +0.001.
2. The pooled whole-family bootstrap LCB95 is > 0. The bootstrap is RNG 302150 with 10,000 draws, and is shared across
   seeds; the pooled statistic is the median across seeds.
3. The family-level sign-flip p (one-sided, mean statistic) is < 0.05 on ≥ 4/6 seeds. The test is exact for ≤ 20
   families; otherwise Monte Carlo with 200,000 flips, RNG 313150.
4. Invalid picks are ≤ r3175's invalid count on each seed.
5. The harmed share (loss > 0.01) is ≤ 15% on each seed.

**Elo label:** "reference = r3175 slot0 (fresh v7)".

**Not scored in H313.** H314 needs r3175's costs for the fresh contexts (the H281/H283 GPU-exact path).

### Dry run (DL-Train pseudo-fresh only)

The full score-and-decide pipeline runs on **DL-Train P0 outer fold 0**: 46 families / 335 contexts, out-of-fold for
H303's archived fold-0 locks of seeds 30001–30003 (S3 H303 tar `hshnLPUZ…`).
- **Reference:** the pool's slot 0 (**r882**), because no r3175 DL-Train costs exist. The Elo label is "reference = r882
  slot0 (DL-Train pseudo-fresh dry run)".
- **Harness regression:** seed 30001's dry-run picks must equal sha `18fdb2538cea…`.

This exercises the pipeline, not new science. It uses no dev91 and no fresh rows.

## Resources

- **Hosts:** xbabe2 CPU, nice 19, `taskset` 53, 54, 55, 116 (≤ 4 cores, 93–95% idle; host 85% idle).
- **Memory:** systemd `MemoryMax=4G`; stop if MemAvailable < 20 GiB or host CPU > 85%.
- **Scratch:** ≤ 2 GB in `~/jope-data/h313-opus`, archived to S3 `…/H313/` (exact-version GET and sha verify), then
  deleted.
- **Deadlines:** compute stop 11:45Z; close 12:00Z.
