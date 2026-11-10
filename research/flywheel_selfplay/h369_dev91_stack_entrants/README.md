# H369 — H364 XSTACK-2L once-scored on the fixed holdout board `dev91-audit-v1`

**Result: `h364.XSTACK_2L` = 1726.8 AUDIT-Elo** (95% CI 1633–1852). It ranks 6 of 24.

- It is the **best model this project has trained**: +19 Elo over `h335.H335_AGG` (1707.8).
- It is **not a new leader.** Kumo L (2047.9), TabFM ensemble (1900.9), TabFM (1875.7), TabPFN 3.5 (1822.9) and
  TabPFN v3 (1787.6) stay ahead.
- Elo depends on the field, so every entrant moved slightly when 2L joined. Kumo was 2036.0 before.
- No existing entrant was refit, re-run or rescored.

- **PREREG:** [`PREREG.md`](PREREG.md) at `e75d2b935f`, with amendment 1 at `b54b875fc6`. It was published before the
  one-shot unlock.
- **Owner:** `cl_fwriter`, Claude Opus 5.5.
- **Request:** owner, 2026-11-05: "score all new waves on the same holdout and update the ELO scores".
- **Numbers:** [`metrics.json`](metrics.json). The board files are `scores/h364.XSTACK_2L.jsonl`, `LADDER.md`,
  `ladder.json` and `ARCHIVE.json` under `research/elo/reference-board/boards/dev91-audit-v1/`.

## 2L vs H335 AGG on dev91 (paired, descriptive)
| Quantity | Value |
| --- | --- |
| Family-mean AUDIT u, H335_AGG → 2L | 1.5592 → 1.5607 |
| Gain | **+0.0015** (0.15% RMSE cut) |
| Families won / tied / lost | 14 / 3 / 2 |
| Clipped sign-flip p (H335 primary procedure) | 0.045. Would not pass the 0.025 bar; not a pre-registered test here |
| LOFO min | +0.0011 |
| Family LCB (mean − 1.96 sd/√19) | −0.0001 |
| Second level active | 55 / 182 (context, seed) units; 38 contexts tie exactly |
| Context gain range | −0.046 … +0.028; none below −0.05 |

- **On TRAIN P201d, H364 measured +0.0036** (sign-flip p 0.011, +31 Elo). On dev91 the direction holds (14 of 19
  families won), but the effect is less than half that size.
- **Head to head against the foundation models**, the 2L gain is:

  | vs | Gain | Families won / lost |
  | --- | --- | --- |
  | TabPFN v3 | −0.082 | 2 / 17 |
  | TabPFN 3.5 | −0.080 | 5 / 14 |
  | TabFM | −0.160 | 4 / 15 |
  | Kumo L | −0.375 | 1 / 18 |

  See [`LADDER_focus_2L.md`](LADDER_focus_2L.md), the ladder with 2L as the focus.

## Updated ladder (`tools/ladder.py`, anchor `h335.H257_REPLICA` = 1500)
| Rank | Entrant | AUDIT-Elo | 95% CI |
| --- | --- | --- | --- |
| 1 | kumo_tabular_large_e8 | 2047.9 | 1860–2340 |
| 2 | tabfm_ensemble | 1900.9 | 1732–2136 |
| 3 | tabfm | 1875.7 | 1706–2108 |
| 4 | tabpfn_v3p5 | 1822.9 | 1650–2077 |
| 5 | tabpfn_v3 | 1787.6 | 1651–1963 |
| **6** | **h364.XSTACK_2L (new)** | **1726.8** | 1633–1852 |
| 7 | h335.H335_AGG | 1707.8 | 1620–1826 |
| 8 | h335.OLD_MATCHED_CV3 | 1627.1 | 1564–1707 |
| 9–24 | Opus5p5_Extra 1578.2 … BANK_NN1 1196.2, same order as before except one swap | | |

- **The one swap:** `h335.H317_BASE` (1527.4) now sits just above `autogluon_300s.v1` (1526.4). They were tied at
  1526.4 before. Moves like this are field effects of adding an entrant; full table in `LADDER.md`.

## How it was scored
- **X0.** H335 Stage V's stored `AGG.g50.CV3@12` output for each search seed.
  - Its stored weights reproduce the board's `h335.H335_AGG` per-seed values within **2.7e-15**, on all 182 units.
  - They also reproduce the stored SEARCH log-ratios within 3.6e-15. The tolerance was 1e-6.
- **Alignment.** GP records, the board envelope (rows, folds, X, y), the AUDIT baseline, member RMSEs and the stored
  library prediction RMSEs agree within **3.8e-15** (tolerance 1e-9).
- **Library AUDIT predictions** are the stored board records of the four classical learners (prediction sha checked).
- **Library SEARCH OOF** comes from new fits on dev91 SEARCH rows only. 364 / 364 completed; 0 failed.
- **Board value** is the mean of the two search seeds, as for every `h335.*` GP entrant.

## Deviations and disclosures
1. **Lock attempt 1 was refused** (amendment 1, before any dev91 read).
   - The LOCK listed the dev91 `BOARD.json` among the files `unlock_stage_v` hashes before it enables Stage V, and
     the guard refused that path.
   - No marker was created.
   - Attempt 2 (LOCK `1ce9a152`) moved it to `inputs_sha256`, which is checked after the unlock.
2. **The library step ran on three hosts** (owner, ~13:50Z: "use the FULL power of our compute across xbabe1,
   xbabe2, xbabe3, and xbabe4").
   - **Setup common to all hosts:**
     - The work was split by context, disjointly.
     - Every host used the same locked code (CODE.sha256 `1eb31fdc…`), the same inputs and the same harness
       (`score_reference.py` `169cb04e…`, `models.json` `20d100c4…`).
     - All hosts have the same CPU model (Threadripper PRO 3995WX), Python 3.12.3 and identical learner versions.
   - **The PREREG's bwrap sandbox ran only on xbabe2**, for 151 outputs.
     - xbabe3 restricts unprivileged user namespaces (AppArmor), so bwrap cannot create its network namespace. That
       host security setting was left unchanged.
     - xbabe1 has no bwrap.
     - On those two hosts, 213 outputs (xbabe1 85, xbabe3 128) ran in a clean `env -i` environment. The code opens
       only `--inputs`, and those hold only the assigned contexts' SEARCH envelopes and stored predictions.
   - **xbabe4 was not used.** It is macOS, without the learner packages.
   - **xbabe2's runner was stopped twice and its unfinished contexts moved.** Only in-flight jobs were lost; outputs
     are written atomically.
     - **Why:** lightgbm and xgboost ran 20–60× slower there than on the other hosts. The likely cause is OpenMP
       contention on CPUs other lanes were also using.
   - **Provenance:** [`lib_host_map.tsv`](lib_host_map.tsv) records the host of each output.
   - **Cross-host check:**
     - All 364 jobs were re-run on xbabe3.
     - catboost, lightgbm and xgboost are byte-identical across hosts and with or without the sandbox.
     - random_forest differs in 69 contexts by at most 2.0e-13 relative (5.8e-11 absolute). This is parallel tree
       averaging order (8 threads), and it also occurs between two xbabe3 runs.
3. **xbabe3 load promise.** xbabe3 ran at up to 104 CPUs, above the earlier ≤ 800% batch promise for that CI host, on
   the owner's explicit instruction. The host was idle (load 1) and CI triggers are paused.
4. **dev91 is a reuse cohort** (H257 tuned on it). The H364 constants were chosen on TRAIN only.

## Not scored
- **H354/H357 `X_CL`** re-solves the GP stack's first-level weights. That needs AUDIT predictions for GP candidates X0
  gave zero weight, which means new GP worker AUDIT runs on dev91. It also failed its TRAIN confirmation (p 0.074).
- **H354 `X_LLM` / `X_ALL`** additionally need LLM-regressor SEARCH OOF predictions. H354 did not advance them.
- **Other lanes' H352–H368 waves** are owned by other agents.

## Reproduce
```sh
R=research/flywheel_selfplay/h369_dev91_stack_entrants/run_h369.sh
bash $R stage TAG; export H369_CODE=~/h369-stage/code-TAG
bash $R lock; bash $R unlock; bash $R prepare; bash $R align
bash $R lib            # or the distributed form: evidence/H369/{dist_h369.sh, run_lib_nosb.sh}
bash $R score; bash $R rows
evidence/H369/board_h369.sh ladder; evidence/H369/board_h369.sh archive --upload
```
