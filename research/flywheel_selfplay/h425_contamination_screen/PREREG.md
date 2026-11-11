# H425 PREREG — contamination screen: copies of the in-context model's training tables among DEV and CONFIRM

**Status.** Published on `origin/v1/next` before the screen opens any `DEV` or `CONFIRM` table. The thresholds below
are fixed here. What was looked at is in §8. Claim: `~/jope_plan/ns/receipts/H425_claim.txt` (2026-11-11T10:07Z).

**Authority.** Owner order of 2026-11-10 (APEX campaign). Relayed by ENG:watcher on 2026-11-11 (stamped 3:06 AM
MST): a pre-registered, feature-level contamination screen of `DEV` and `CONFIRM` against `FIT-ICL`, with the
target-only screen as baseline; a frozen exclusion list; H424's `I` vs `B` recomputed on `DEV` without it as a
pre-declared sensitivity read with no new verdict; on `CONFIRM` only feature / metadata overlap, no scoring, no
model predictions, no stack fits; CPU only, at most 8 threads on xbabe2, at most 2 hours; any `CONFIRM` copy is
recorded and escalated before a champion read.

**Closed.** dev91-audit-v1, DL-Val, DL-Holdout and sealed data are never read. No model runs, no GP worker runs,
nothing is fitted or scored.

## 1. Question
H424's column comes from a model trained on the `FIT-ICL` families. Its target-only look found `DEV` tables that
share target values with `FIT-ICL` tables, some of them the same public dataset under another name. How many `DEV`
and `CONFIRM` tables are copies of `FIT-ICL` tables at the feature level, and does H424's gain survive without them?

## 2. Cohorts and access
- **Reference:** every `FIT-ICL` table of the manifest (440, all tiers).
- **Queries:** the 115 core `DEV` contexts and all 241 `CONFIRM` contexts (142 core).
- **`CONFIRM` — a recorded deviation.** The cohort manifest says `CONFIRM` is "read once by the champion wave". This
  screen opens the `CONFIRM` table files inside the sandbox to hash their **feature** columns. It is ordered in
  the relayed note with these limits, which the code enforces:
  - the target column of a `CONFIRM` table is never turned into keys, and the baseline target screen does not run
    on `CONFIRM`;
  - no value, key, hash or sketch of any table leaves the sandbox: the output holds scores, flags and the
    identifiers of matched `FIT-ICL` tables only;
  - nothing is scored, predicted or fitted on `CONFIRM`.
- `DEV` tables are opened in the same way (features and, for the baseline, targets).

## 3. Screens and thresholds (`screen_h425.py`)
Every cell is reduced to a key: its value at 6 significant digits. Each screen compares a query table with one
`FIT-ICL` table at a time.

| Screen | Evidence | Rule |
| --- | --- | --- |
| **S1 specific columns** | A column is *specific* if it has ≥ 100 distinct keys and at least half of them carry ≥ 4 significant digits. Two specific columns *match* if the Jaccard estimate of their key sets (256-hash sketches) is ≥ 0.05, *strongly* if ≥ 0.5 | With N = max(3, 30% of the query's specific columns): **copy** if ≥ N columns match strongly within one `FIT-ICL` table; **possible** if ≥ N columns match; with only 1 or 2 specific columns, **possible** if all match strongly |
| **S2 rows** | Row hash = hash of the row's sorted feature keys (column order ignored). Applies to tables with ≥ 4 feature columns and ≥ 50% distinct rows | **copy** if ≥ 5% and ≥ 20 of the query's distinct rows occur in one `FIT-ICL` table |
| **S3 column ranges** | (min, max) of a column at 4 significant digits; a pair is *rare* if it occurs in ≤ 10 of the 796 screened tables | **possible** if ≥ 3 and ≥ 50% of the query's rare pairs occur in one `FIT-ICL` table |
| **S4 source names** | Readable family keys reduced to token sets (generic tokens such as "regression", "runtime", "dataset" removed) | **possible** if the token sets are equal, or share ≥ 2 tokens with Jaccard ≥ 0.75 |
| **B target only (baseline; never on `CONFIRM`)** | distinct target keys of the query, if ≥ 20 | **target-only** if ≥ 50% of them occur among one `FIT-ICL` table's target keys |

**Tier of a context:** `copy` (S1 copy or S2) → else `possible` (any other S1, S3 or S4 flag) → else
`target_only` (B alone) → else `clear`.

- Why two feature tiers: tables are row samples of at most about 1,600 rows, so two samples of one source share
  only part of their values. Many strongly matching columns cannot arise by chance; a few weakly matching columns
  can, for columns on a coarse grid.
- S1 does not see datasets whose values all lie on a coarse grid (for example two decimals in [0, 1]); S2 does if
  rows are shared; neither does if only the row sample differs. This limit is known in advance.

## 4. Outputs, frozen before the sensitivity read
- `EXCLUSION_DEV.json`: the tier, the flags, the scores and the matched `FIT-ICL` tables of each of the 115 `DEV`
  contexts. `CONFIRM_FLAGS.json`: the same for the 241 `CONFIRM` contexts (no target screen).
- Both are cut from the screen's output by `lists_h425.sh` and published on `origin/v1/next` with the screen's
  sha256 before `sens_h425.py` runs.

## 5. Sensitivity read of H424 (no new verdict)
H424's stored per-context values (`I`, `B`) are restated by `h335_stats.primary_test` on these subsets of `DEV`:

| Row | Contexts |
| --- | --- |
| all | 115 (H424's primary, restated) |
| without copy | tier `copy` removed |
| **without feature flags (the pre-declared read)** | tiers `copy` and `possible` removed |
| without any flag | also `target_only` removed |
| flagged only | the removed contexts of the pre-declared read, on their own |

- **Statement fixed in advance:** H424's gain "survives" if the pre-declared read has a clipped family-mean gain
  above 0 with one-sided sign-flip p < 0.025. It "does not survive" otherwise. Either way H424's verdict is not
  changed here; a non-surviving read is recorded in both READMEs and on the board.
- If H424's second read (hour-24 checkpoint) exists by then, the same table is produced for it.
- What the table means: it removes copies the screen can see. It cannot prove the remaining contexts are clean.

## 6. `CONFIRM`: what counts as failure
- **Failure:** at least one `CONFIRM` context of tier `copy`. It is recorded here and on the board, and escalated to
  the owner before any champion read; nothing else is done with `CONFIRM`.
- Tier `possible` on `CONFIRM` is listed and escalated as suspects, named as such.
- The development rehearsal (§8) makes failure the expected outcome.

## 7. Integrity gates (`gates_h425.sh`, after the lists are published, before the sensitivity read)
1. This PREREG's commit is older than the screen's output; the staged PREREG equals the published one.
2. The screen ran in the sandbox with the guard, on exactly `DEV` (115) and `CONFIRM` (241, 142 core) against 440
   `FIT-ICL` tables.
3. No `CONFIRM` target was keyed and no baseline result exists for a `CONFIRM` context.
4. The two lists are cut from that screen output (sha256) and cover 115 and 241 contexts.
5. The lists on `origin/v1/next` equal the local ones, and no sensitivity output exists yet.
6. H424's value file is the one H424 published.
7. Budget: at most 8 processes, at most 2 hours.
8. Published stage, code hash verified.

## 8. What was looked at before this PREREG
- H424's target-only look (17 of 105 `DEV` contexts flagged; the gain without them +0.0087 u) and H424's
  per-context gains. **No feature-level score of any `DEV` or `CONFIRM` table has been computed.**
- **A rehearsal of this code on `FIT-SEL` (a design cohort) against `FIT-ICL`:** of 106 contexts, 18 `copy`,
  12 `possible`, 8 `target_only`, 68 `clear`. Several `copy` cases match 45 to 285 columns strongly; several
  `FIT-SEL` families match the same `FIT-ICL` table. Within `FIT-ICL` itself, 62 of 440 tables are a `copy` of a
  table of another family. So one source dataset often appears as several families, and the cohort split by
  family key separates them.
- **The tier rule was changed once after the first rehearsal run:** the first version had one feature flag; the
  rehearsal showed two clearly different kinds of S1 match (dozens of strong columns against three weak ones), so
  `copy` and `possible` were separated. All thresholds are otherwise the ones first written.

## 9. Budget
xbabe2 cpus 32–39, 8 processes, CPU only, no GPU; the rehearsal took 30 seconds; cap 2 hours. The H424 second-read
driver (cpus 16–31) and the training runs are not touched.

## 10. Limits
- Value-based screens miss a copy whose features were rescaled or re-encoded, and miss coarse-grid datasets unless
  rows are shared. Names help only for the families that have readable ones (26 of 106 in the rehearsal).
- A flagged table need not have leaked anything: the model was trained to predict each training table's own target,
  and a copy may carry a different target. The screen measures exposure of the features, not harm.
- `DEV` against `FIT-SEL` or DL-Train, and `CONFIRM` against cohorts other than `FIT-ICL`, are not screened here.

## 11. Files
`screen_h425.py` (sandbox), `lists_h425.sh`, `sens_h425.py`, `gates_h425.sh`, `stage_h425.sh`.
