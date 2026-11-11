# H421 PREREG addendum 1 — launch: host change (recorded deviation), baseline, and the selection objective

**Status.** Published on `origin/v1/next` before the harvest, the fill and the selection run from published code,
and before any DEV deploy. No DEV or CONFIRM context has been read or evaluated by this wave. What already ran from
unpublished development code is listed in §5.

**Authority.** The go-ahead for this pre-registered wave was relayed by ENG:watcher on 2026-11-11 00:22Z (5:22 PM
MST): run on a host with disk headroom (xbabe2, not xbabe3), with a bounded CPU budget, leaving the 104M training
group and all GPUs untouched and staying off the xbabe0 cores; record the host change before launching.

## 1. Deviation: hosts
| | PREREG §6 | Now |
| --- | --- | --- |
| Fill | xbabe3 cpus 32–63, 96–127 and xbabe1 cpus 19–50, 83–114 | **xbabe2 cpus 32–63, 96–127** (docker sandbox, nice 19) |
| Selection and gate | xbabe2, ≤ 16 threads | xbabe2, unchanged |
| DEV deploys | xbabe3 | **xbabe2 cpus 32–63, 96–127**, in H420's DEV run directory |

Reason: xbabe3's root disk fell below the CI runner's 50 GB reserve on 2026-11-10 (H420's README, deviation 2).
xbabe2 has about 500 GB free and these CPUs became idle when H420's DEV deploys ended. No job runs on xbabe0,
xbabe1 or xbabe3, and no GPU is used.

## 2. Baseline (PREREG §2)
H420 closed `LANG_CONFIRMED` with `TBOTH` passing, so `B*` = H420 `TBOTH` (sources GS, GE, GX; stock library +
`LIB_EXT`). Its DEV records are stored in the H420 DEV run directory and are not recomputed.

## 3. How the steps of §3 are carried out (points the PREREG left open, and one correction)
1. **Harvest.** Program text is read from the stored evaluation memos. Near-duplicates: same text after every
   numeric literal is rounded to 2 significant digits. On P201E the stack weights are the locked design's rule
   applied to the stored matrices (`meta_h416.meta_weights("BCAR", Q, y, cid)`); the same call reproduces the stored
   FIT-SEL weights exactly, which the harvest checks.
2. **Fill.** One SEARCH evaluation per candidate and context (60 s limit). The engine's SEARCH evaluation is a
   2-fold inner cross-validation and returns each fold's out-of-fold predictions, so one evaluation gives the
   out-of-fold SEARCH prediction the selection needs. Stored memo results are reused; only `search`-kind
   evaluations are requested (asserted).
3. **Correction to the selection objective.** §3.3 builds the Caruana stack over "`B*`'s library plus the
   candidates selected so far". That objective would mislead in two ways:
   - Programs the pipeline already puts into every context (library target variants, search seeds) rank high in the
     harvest and would be selected again, although adding them as presets changes nothing.
   - A preset gets stack weight only if it enters the context's stack candidate set. In the smoke contexts the 48
     best search fitnesses lie within 0.005–0.06 of the best, so a preset almost never enters the top block; it
     enters through the head-diversity slots or not at all.

   **The objective used:** per design context, the Caruana stack (`meta_h416.kcar`, 100 steps) over the candidate
   set the deploy would build: the context's stored stack candidates (the columns of its stored H matrix) plus the
   presets under test (`LIB_EXT` and the selected candidates), admitted by the deploy's own rule
   (`deploy_h415.hf_candidates`: top 48 by search fitness; of the rest, heads not in the top block, at most 2 per
   head, the 16 best). All columns are the 2-fold out-of-fold SEARCH predictions; the stored candidates' columns are
   read from the memo, not re-evaluated. A preset that is already a stored candidate of a context changes nothing
   there. Utility = −ln(RMSE / RMSE of the fold-mean predictor); objective = family mean.
   - **Limits of this model:** the stored candidate sets of the design contexts come from H416 / H417 (GS, GE, GL
     and the stock library); `LIB_EXT` is added by the rule, and GX is not represented (it was null in H420).
     Predictions come from fits on half the SEARCH rows, where the deploy's stack uses 5-fold fits.
4. **Coverage ablation (`TCOV`).** Same candidate-set rule; objective = family mean of the best single column's
   utility.
5. **Gate (§3.4), as registered.** Families are split in two halves by `sha256("H421-gate" ‖ NUL ‖ family_id) mod 2`.
   Random set: 20 harvested candidates, numpy `default_rng(421101).choice`. Sign-flip: 100,000 flips, seed 421102,
   one-sided. Pass: family-mean difference above 0 with p < 0.05.
6. **Selection pool.** The 20 presets are selected from the harvested candidates; library presets of `B*` are in the
   pool only so that their evaluations exist for the objective.

## 4. Budget (bounded; corrections of §6 in bold)
| Step | Cap | Expected from the timing sample |
| --- | --- | --- |
| Fill | 60 worker-hours, as registered; 8 processes × 8 CPUs, each stops before a new context at 7.5 worker-hours | about 17 worker-hours (209 worker-seconds per context on 4 contexts), about 30 minutes |
| Selection and gate | 4 core-hours, as registered | minutes |
| DEV deploys, 3 variants | **15 worker-hours of new evaluations and 6 hours of wall-clock on the 64 CPUs** | H420's library variant: 76 minutes and about 4 worker-hours |

- **Why the DEV row changed:** §6 said "60 core-hours". A deploy process is bound by its Python thread, so the 64
  CPUs are mostly idle while it replays searches from the memo; measured in H420, one library variant held the 64
  CPUs for 76 minutes (81 core-hours of allocation) for about 4 worker-hours of evaluation. The old figure mixed the
  two units.
- If the fill would exceed its cap, the pool is cut from the bottom of the harvest rank, as registered.
- Compute deadline: 72 hours after the fill starts.

## 5. What ran before this addendum (development code, design contexts only, no AUDIT value, no DEV context)
- A harvest on the stored records: 291 design contexts in 291 families; 4,028 groups; a group's family count is 66
  at rank 1, 23 at rank 10, 9 at rank 30, 3 at rank 60 and 1 at rank 120; 18 of the top 120 groups are library
  presets, so the pool is 40 library presets + 102 harvested candidates. The published harvest is run again and its
  pool is the one used.
- A smoke fill on 4 design contexts (296 worker runs, 837 worker-seconds, 1 timeout) and the selection code on
  those 4 contexts, as code checks. Their numbers are not used.
- The stored memos of the 291 design contexts were copied into the fill's run directory (P201E from xbabe0's disk,
  FIT-SEL from the H420 run on xbabe2).

## 6. DEV test (§4), unchanged
`TPORT`, `TRAND`, `TCOV` = `B*` plus 20 presets each (`train_h421.py`), in H420's DEV run directory so that they
share its memo; the replay check (0 worker runs) runs before the first variant; primary `TPORT` versus `B*`,
control `TPORT` versus `TRAND`. The selection result and the three preset files, with their sha256, are published
before the first DEV deploy starts. If the gate fails, the wave stops as `PORT_NULL_TRAIN` and DEV is not deployed.

## 7. Files
`harvest_h421.py`, `fill_h421.py`, `select_h421.py`, `train_h421.py`, `h421_shards.sh`, `stage_h421.sh`.
