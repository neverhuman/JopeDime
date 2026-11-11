# H426 PREREG — the DEV tests of H420–H425 restated with clusters as the unit (a restatement; no new verdicts)

**Status.** Published on `origin/v1/next` before any cluster-level number is computed. Claim:
`~/jope_plan/ns/receipts/H426_claim.txt` (2026-11-11T10:37Z). **No verdict of any wave is changed by this wave**,
whatever it shows; a conclusion that would flip is named in the README, on the board and in the queue file.

**Authority.** Relayed by ENG:watcher on 2026-11-11 (stamped 3:38 AM MST): restate the `DEV` tests H420 through H425
from stored values only, with `DEV` contexts that match the same training table or each other under the frozen H425
lists grouped into clusters; CPU only, at most 8 threads, at most 1 hour; no model forward pass, no GPU, no
`CONFIRM`, never dev91; the hour-24 driver is left alone.

## 1. Why
H425 found that one source dataset often appears as several families. Every `DEV` test so far treated its 115
families as independent units. If some are copies of each other, the tests had fewer independent units than they
assumed.

## 2. Inputs (stored files only)
The stage records of H417, H420 and H421 on `DEV`; the per-context value files of H422 and H424; the frozen H425
list and the screen output it was cut from. No table is opened, no model or GP worker runs. `CONFIRM` is not
touched (it has no stored test).

## 3. Clusters
Two `DEV` contexts are linked when the H425 screen matched them to the same `FIT-ICL` table. Clusters are the
connected components.

| Clustering | Links through |
| --- | --- |
| **`feature` (pre-declared)** | the matches of the feature screens S1–S4 where they flagged (tiers `copy` and `possible`) |
| `copy` | S1-copy and S2 matches only |
| `any` | the `feature` links plus the target-only baseline's matches |
| `shape` | the `any` links plus: same row count and same feature count for tables with ≥ 20 features (manifest only) |

- **Known limit:** the stored lists compare `DEV` with `FIT-ICL` only and keep one best match per screen. Two `DEV`
  contexts that are copies of each other but match no `FIT-ICL` table, or different ones, stay separate. `shape` is
  a crude, pessimistic stand-in for that: it also joins unrelated tables of equal shape.
- "Each other": direct `DEV`–`DEV` matching needs the tables and is outside "stored values only"; it is not done.

## 4. Restatement
For every comparison below, the wave's own statistic (`h335_stats.primary_test`: clipped unit-mean gain, exact
sign-flip over units, leave-one-unit-out minimum, median) is computed twice: with the original units (families;
it must reproduce the stored numbers to 1e-12 or the run stops) and with clusters as units. Thresholds are the
waves' own: gain above 0, one-sided p < 0.025, leave-one-unit-out minimum above 0, median not negative.

| Wave | Comparison | Role in its wave |
| --- | --- | --- |
| H420 | `TBOTH` vs `B`; `TLIB` vs `B`; `TGX` vs `B` | primary; two secondaries |
| H421 | `TPORT` vs `B*`; `TPORT` vs `TRAND` | primary; control |
| H422 | `S` vs `B`; `A` vs `B`; `SA` vs `B` | primary; two reported arms |
| H424 | `I` vs `B` (and the second read, if its value file exists when this runs) | primary |
| H425 | `I` vs `B` without the feature-flagged contexts | its sensitivity read |

- H423 is not restated: it ran on P201b, which has no frozen list.
- **A flip** = the pass / fail of a comparison differs between the original and the `feature` clustering. H422's
  three-part rule (gain ≥ +0.010 u, test pass, Elo interval above 0) is evaluated both ways as well. Flips under
  the other clusterings are listed separately.
- Reported: units per clustering, cluster sizes, ΔElo with a cluster bootstrap for the `feature` clustering.

## 5. Gates
The script stops unless every stored original is reproduced and the screen file is the one the frozen list was cut
from. `gates_h426.sh` adds: this PREREG is older than the output; the H425 list equals the published one and the
pulled record copies match their manifests; the sandbox view still holds no `CONFIRM` table; at most 8 threads and
1 hour; published stage.

## 6. What was looked at before this PREREG
The frozen H425 list (which `FIT-ICL` tables several `DEV` contexts share: at most three per table), and a check of
the clustering code on the `FIT-SEL` rehearsal file (97 units from 106 contexts; no outcome value involved).
**No cluster-level test statistic has been computed.**

## 7. Expected
Few clusters larger than one, so the number of units should fall only from 115 to about 108, and p-values should
move little. A large change would mean a few big clusters carry a result.

## 8. Files
`restate_h426.py`, `gates_h426.sh`, `stage_h426.sh`.
