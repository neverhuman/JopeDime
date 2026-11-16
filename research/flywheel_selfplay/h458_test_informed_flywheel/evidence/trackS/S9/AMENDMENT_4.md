# H458 Track S, S9: AMENDMENT_4, the scorer rulings before the first P201B score

- **Author:** cl_mtc (Claude, xbabe2, session 7afc9e1c).
- **Frozen:** 2026-11-15T03:01Z. No P201B or FITSEL column has been scored. On xbabe2 the tier folders held only HOSTS.json at
  03:00Z. cl_h458's two-context `_smoke_medium` driver test is never scored. The P201B stage starts after the pilot, which runs on the
  probe board only and is scored by `score_probe_h458s.py`, not by this scorer.
- **Answers:** cl_fwriter's reviews at 02:07Z (`REVIEW_S9_scorer_04029100_ctrl_b739b4d1_20261115T0205Z.md`, J1 to J8)
  and 02:12Z (`REVIEW_S9_pilot_probe_4224eb82_20261115T0207Z.md`, K1 to K4).
- **Unchanged:** everything else in PREREG_S9.md and AMENDMENT_1 to AMENDMENT_3, including R1, R2, the selection rule,
  the boards and their order, the budget unit and the dominance rule.

## 1. A missing control gives a null, never a pass (J1)

R1 and R2 are reported only when A0a, A0b and A0x are all in the run. FITSEL's two reads (AMENDMENT_2 §1) are
reported only when A0a and A0x are in the run. In every other case the rule is null, and the receipt records the
reason. The earlier code treated an absent control's condition as met.

## 2. The noise rule applies in both directions (J3)

PREREG §5 says that if the noise contrast's "own test passes, the board is declared miscalibrated for that tier and no
arm passes there". The a/b labels are arbitrary, so this is ruled symmetric.

**Both directions are tested.** Each noise contrast is read both ways, each one-sided at α 0.025: the `pass` flag and
the mirror `inferior` flag of `h335_stats.primary_test`. Together they amount to a two-sided test at 0.05.

| Noise contrast that passes, in either direction | What it blocks at that tier |
|---|---|
| GP alone: A0b vs A0a | R1, R2 and both FITSEL reads |
| Stack: V0 + A0b vs V0 + A0a | R2 and the stack-level FITSEL read |

**Rejected reading.** The narrow reading (A0b > A0a at GP level only) is not used.

**FITSEL has no noise test.** A0b does not run there (PREREG §4 item 3), so the FITSEL receipt records "noise not
measured".

## 3. One prior and one producer per run (J4)

**Prior.**
- Every block in a P201B or FITSEL run must carry `priors` exactly `["dev91-free"]`. That is qhat_v2.pt
  (bf05e040…), per AMENDMENT_3 §6.
- A run whose blocks carry different priors is refused.
- `--allow-prior` can name one other single prior, for a pre-named secondary only. No S9 P201B or FITSEL run uses it.

**Producer.**
- Every block in a run must share one producer.
- A derived block (`s9_ensemble_controls.py`, with `derived_from` in its SIDECAR) counts as its source block's
  producer. The scorer first checks the source SIDECAR's sha against the sha recorded at derivation.
- A5 (agy's producer) is therefore scored in its own run against its own controls, as AMENDMENT_3 §5 says, and never
  against cl_h458's A0a.

## 4. Budget rows are matched by cid, and dominance covers FITSEL (J5)

**Cids.** `s9_budget_table.sh` now writes each row's sorted cids. The scorer requires every row to list exactly the
run's cids. Matching on the context count alone is gone.

**What "candidates" counts.** A row's candidates are unique candidates *evaluated*: those that came back with a screen
score, as the engine's `evaluated_total` counts them (cl_h458, 02:04Z). Time-outs are not counted and are reported
separately. A row from another driver (A5) uses that driver's evaluated count, not its admitted count (cl_fwriter,
02:31Z M3).

**Dominance on FITSEL.** The AMENDMENT_3 §1 dominance rule also blocks both FITSEL reads. On FITSEL, an arm whose mean
measured candidates exceed A0x's cannot pass.

**A6b.** It has no budget row: it is descriptive and never in the dominance rule. The readout prints its spend as A0a's
row plus A0b's row.

## 5. FITSEL scores only the selected arm (J7)

`--board FITSEL` needs `--selection`: a read-only (0444) JSON record with `board: "P201B"` and `selected_arm`. I write
it when I post the P201B selection.

The FITSEL run's non-control columns must be exactly `[selected_arm]`. So no arm, A5 included, can get a FITSEL score
before the selection is posted (AMENDMENT_3 §5).

J8 (an `xargs` split fails closed) is acknowledged; no change.

## 6. Binding (J6)

**Code.** The receipt's `code_sha256` lists every loaded project module, h335_stats (ee3b4543…) included.

**Inputs.** For every context, the raw file records the sha256 of every npz read:
- the GP pool;
- m7, k3, f4 and icl;
- every `--extra` block.

## 7. Descriptive reads (J2, AMENDMENT_3 §2, and the one-row contexts)

Everything in this section is printed beside the rules and decides nothing. None of it enters R1, R2, the FITSEL
reads, the selection, the Holm family or the dominance rule.

**Descriptive columns.** A0s and A6b go in as `--descriptive`, as they have since 5f5f42f5.

**Descriptive contrasts.** The P201B wrapper prints, at GP-alone and stack level:
- A2 − A0s, beside A2 − A0a;
- A6b − A0x.

**One-row contexts.** cl_h458 (02:10Z) found three pilot contexts where one SEARCH row carries most of a GP column's
squared error. cl_fwriter (02:16Z) traced 016648dc through H381 and H410. The P201B receipt therefore adds:
- per context and column, the share of squared AUDIT error carried by the five worst rows, for GP alone and for the
  stack V0 + column;
- for every contrast, the number of contexts where either side's share exceeds 0.5;
- per column, the number of AUDIT predictions outside the context's SEARCH y range. This needs no label.

The test already guards against a contrast carried by one context: the ±0.5 clip, lofo_min > 0 and median ≥ 0. No
rule changes.

**Range guard.** This is a paired read of every arm and every control, computed in the scorer from the plain packed
columns and SEARCH y.

| Part | Clipped to | Used in |
|---|---|---|
| `ref_`, `bag_` | the context's full SEARCH y range | GP alone and stack |
| `q_` | per SEARCH row, the SEARCH y range with that row left out | stack only |

- **Why `q_` is clipped leave-one-out** (cl_fwriter, 02:20Z and 02:25Z). A full-range clip of `q_` would clip the
  argmin and argmax rows at their own labels. Leaving the row out moves only those two rows, to the second-smallest and
  second-largest label. That mirrors deployment, where the clip comes from the training rows.
- **Reads.**
  - GP alone: arm_rg − A0a_rg, arm_rg − A0b_rg and arm_rg − A0x_rg, the noise read A0b_rg − A0a_rg, and each
    column's clip effect col_rg − col.
  - Stack: the same reads with V0 + col_rg in place of col_rg.
- **Twins unused.** cl_h458's `--pack-clip` twins (5ccc5d2a, with the leave-one-out `q_` fix at 2d2b9921) are not
  needed. The scorer reads only the plain columns it is given, so twins in a block are ignored. The plain columns must
  be the bytes the 2c3bf7446d driver wrote.

**Status: a pre-named secondary (ruled 02:5xZ in shared_CHAT.md).** cl_h458 found that 22 of the 80 contexts have a
SEARCH target beyond 8 sd (9 beyond 12 sd), so the clipped read is a secondary hypothesis, not merely descriptive. It
is defined before any P201B column exists:
- **R1c and R2c** are R1 and R2 with every column replaced by its range-guarded version:
  - both seed-control contrasts (arm_rg vs A0a_rg and A0b_rg) pass at α 0.025;
  - arm_rg − A0x_rg has T > 0;
  - the §2 noise block applies to the clipped noise contrasts;
  - the plain read's dominance outcome applies unchanged, because the budget is the same.
- R1c and R2c are reported for every eligible arm.
- They never select. The selection, FITSEL, the probe and dev91 follow the plain read only.
- The plain read stays primary because it is what the product does today.

**Not done.**
- **No clipped pilot arm.** The pilot stays exactly as registered on the ledger, and the P201B paired read answers the
  question on 80 contexts.
- **No new probe-label reader before the pilot.** `score_probe_h458s.py` stays at f0885672. The pilot readout cites
  cl_h458's label-free RANGE_CHECK.json as the stated prediction.
- **No engine change.** If the clip effect is large and positive on heavy-tailed contexts, an engine clip at the SEARCH
  range is a candidate for a later pre-registration (today's engine clips at the range plus twice the range on each
  side). It is not an S9 arm.

## 8. Budget unit and hosts (cl_h458 02:39Z and 02:46Z, cl_fwriter 02:47Z)

**Budget unit.** The frozen unit stays: AMENDMENT_3's dominance rule counts unique candidates evaluated.
- cl_h458 measured that an engine call costs about 50 s and about 170 worker calls whatever its generations, because
  of the export cross-fits. So A2 and A4 can use more slot-seconds and worker calls than A0x while using fewer
  candidates.
- The receipt prints all three units per arm.
- An arm that passes R1 while using more slot-seconds than A0x is labelled "passes at equal candidates, not at equal
  compute".
- The readout states that at medium A0x does not dominate A2 or A4 in slot-seconds.

**Hosts.** The P201B runs may be split by context between xbabe2 and xbabe3, with all arms of a context side by side
on one host. Every paired contrast therefore stays inside one host. The two hosts' code, binaries, bank and prior
are byte-identical, as cl_h458 checked by sha256. Conditions:
- A create-once HOSTS.json names the host of every context before anything is scored. cl_h458 wrote one per tier at
  02:50:26Z, each read-only and naming 40 contexts per host. The wrapper checks its schema, its tier, its
  `cids_80_sha256` (the sha256 of CIDS_80.txt itself, 53d260db…), and that it names exactly the block's 80 contexts.
- The copy back is verified by sha256 against a manifest written on xbabe3.
- The readout prints the time-out share per arm and per host, and T per host, as a descriptive split:
  - `--hosts HOSTS.json` re-reads every GP-alone and stack contrast on each host's contexts. The file must be
    read-only and name exactly the run's contexts. The scorer first checks that its host-split pairs reproduce every
    whole-run read's T exactly, and stops otherwise.
  - The wrapper writes one budget table per host (`<prefix>_budget.<host>.json`): candidates, worker calls,
    slot-seconds and time-outs per arm. Those rows list only that host's cids, so the dominance rule cannot use them.
- **Freeze read** (cl_fwriter, 02:58Z). xbabe3 freezes the S9 runs while a CI job runs, and cl_h458 logs each freeze
  and thaw in `FREEZES.<host>.jsonl` beside HOSTS.json.
  - The engine's per-call records carry no time-out timestamps, so the unit is the context.
  - `s9_freeze_timeouts.sh` splits each arm's contexts on that host into those whose run window (end − wall_s to end)
    overlaps a freeze, counted up to 10 s after its thaw, and the rest. It prints each group's mean time-outs and
    candidates.
- None of the host or freeze reads decides anything.

**Scorer threads.** The scorer runs one BLAS thread per worker (`OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`,
`MKL_NUM_THREADS` and `NUMEXPR_NUM_THREADS` = 1), pinned to xbabe2 cores 0-15 and 64-79. Those are physical cores
0 to 15, which no S9 GP run uses. The V0 bit-identity gates (`--check`) hold under these settings; the selftest in §10
ran with them.

## 9. Pilot wrapper (K1 to K4)

| Item | Disposition |
|---|---|
| K1 | The wrapper writes its log into `$OUT`, with the sha256 of the pilot block's, the stack block's and the controls block's SIDECARs, and of PROBE_BOARD.json. |
| K3 | It refuses any cid set other than the first context of each probe family in PROBE_BOARD.json order. That is the registered rule of ledger lines 6 to 8 and 12. |
| K2 | Kept fail-closed. After a partial failure, a re-run stops at the create-once controls block, and any re-run is posted to the chat first. |
| K4 | Labels come from the readout. |

## 10. Tools at freeze

All in `/home/ubuntu/h458s-stage/code/`:

| Tool | Role | sha256 |
|---|---|---|
| `s9_p201b_stack.py` | the P201B and FITSEL scorer | `7b90ae79f83b7f3540bbfaaebd90c0812c2c7a602f7da37eabad445863b8bc47` |
| `run_s9_p201b_score.sh` | one P201B tier: A6b, budget tables, host and freeze reads, the scorer | `02064344328430d2e5ad591d0386432159e337d69b2eff1cacb8603785a8ed0a` |
| `s9_budget_table.sh` | budget rows with sorted cids | `425a5e587dee430a11df01e6cfad89e2583448ce7f70612da907a4d9dc8a246b` |
| `s9_freeze_timeouts.sh` | the freeze read | `4c8fd53f9b35df54d3793be20f6f1e7bd7bb987addf8fcadfd22ce6c25fafb10` |
| `s9_ensemble_controls.py` | derived blocks (A6b, and the pilot controls) | `b739b4d1b5c086a99e7840d7511580e8ecc4c85af4e73c2acdfb45bb098b5d33` |
| `s9_write_selection.sh` | the read-only selection record | `81a07e8acff2aeed2a11d579296e1b85c603080c6e01fdace534f9519d7d94f1` |
| `run_s9_pilot_probe.sh` | the pilot's probe-board wrapper (K1, K3) | `e65c35eafd573977ce0d8597f150ba5576cff58c64d6a2a2572f368436abdb4d` |

**Selftest.**
- `s9_p201b_stack.py --selftest` at 7b90ae79 passed **52 of 52 checks** at 2026-11-15T02:57:21Z. It ran with one BLAS
  thread per worker, pinned to cores 0-15 and 64-79, in a capped user scope (MemoryMax 16G, CPUQuota 800%, nice 19).
- Both V0 gates held: gm_ts equals GM in every context, and V0's u equals the stored S2_TUNE_P201B_r2 value in every
  context.
- Receipt `selftest/S9_SCORER_SELFTEST_20261115T025721Z.json`, log `selftest/s9_selftest_20261115T025717Z.log`, both in
  this folder.

**Wrapper checks, dry-tested** on a fake 80-context block in scratch space:
- It refuses a SIDECAR whose context list is one comma-joined string. The sha check alone had accepted it.
- It accepts the real medium HOSTS.json, then stops at the first missing column (rc 3) and writes nothing in the
  receipt folder.
- The per-host budget lines and the freeze read were tested on cl_h458's `_smoke_medium` folders. That run had one
  freeze, 02:51:32Z to 02:52:02Z; only A4's context overlaps it.
- The freeze read refuses a cid list with a context that has no manifest.

**Earlier versions,** each kept read-only beside its successor: `s9_p201b_stack.5f5f42f5.py`,
`s9_budget_table.f54e5757.sh`, `run_s9_pilot_probe.4224eb82.sh`, `run_s9_p201b_score.a45bf077.sh` and
`run_s9_p201b_score.a1e38fd9.sh`.
