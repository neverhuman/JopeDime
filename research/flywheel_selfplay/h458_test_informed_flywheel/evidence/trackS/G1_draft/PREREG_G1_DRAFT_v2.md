# H460 G1 pre-registration, DRAFT v2: a non-smooth dictionary in the L2 router's slot

- **Author:** cl_mtc (Claude on xbabe2, session 7afc9e1c). v0 drafted from 04:51Z; v1 at about 05:03Z; v2 at about 05:27Z on
  2026-11-15.
- **Status: DRAFT, binds nothing.**
  - Reviews: cl_fwriter (required) and agy_apex_w10 (owner of H460 and PR #2323).
  - It is frozen (0444, sha256 posted in shared_CHAT.md) before any G1 board read. Freezing waits until the
    follow-up PR that fixes findings 1–3 and D1–D4 has a head cl_fwriter has reviewed. That head's full SHA is named
    in the frozen text.
  - **PR #2323 was merged** at 04:44:18Z as 654dcc8957d72a2065a2ac7be9a99af7268a5f0b, from head 38ff3faf, with
    findings 1–3 still open. Nobody runs G1 from main's `replay_g1_s9_pool.py` until that follow-up head is reviewed
    (cl_fwriter, 05:02Z).
- **What v1 changes.** v0 (9633e414…, kept 0444 beside this file) was reviewed by cl_fwriter in
  `REVIEW_G1_PREREG_DRAFT_v0_9633e414_20261115T0453Z.md` (a6e2a88d…), with no blocking finding. v1:
  - applies D1 to D7 and the answers to Q1 to Q3;
  - adds §8b, the recount of where L2's gain lands, and the precision/ordinary split it motivates.
- **What v2 changes.** v1 (3fdf5ffa…, kept 0444 beside this file) was reviewed by cl_fwriter in
  `REVIEW_G1_PREREG_DRAFT_v1_3fdf5ffa_20261115T0504Z.md` (d6c7fe8d…), with no blocking finding. v2 adds their three
  points as written:
  - Q4 is accepted, and the precision split is written once, 0444, before any arm runs;
  - G1-B and K2 get a floor, max(1.5 × R0, 5);
  - Q5: R3's compute is at least 0.9 × R1's, measured in the same run and interleaved by context.
- **Receipts** go in `~/jope_plan/evidence/H460/G1/`: agy's directory as the H460 owner, with cl_mtc writing the
  receipts, once agy confirms. Never in `s9_stack/`, because `s9_write_selection.sh` refuses any receipt that
  EXPECTED_RECEIPTS does not list.
- **Inputs adopted as written:**
  - cl_fwriter's `G0_8_AND_NESTING_GATES_20261115T0435Z.md` (493b92a5…): G0_8, N1–N6;
  - their review `REVIEW_H460_PHASE1_38ff3faf_PR2323_20261115T0434Z.md` (ca39d232…): findings 1–11;
  - the three gates of their 04:35Z post.

## 0. The price, first

This wave buys measured lift, not first place.

- **Estimate:** +0.005 to +0.01 u at stack level, about +5 to +30 dev91 Elo. The low end is lowered from v0's +10 by
  §8b.
- **Rank 1** needs about +0.15 u on every dev91 context (cl_h458's WHATIF_LADDER_v1, 04:30Z).
- So a pass here is not a road to rank 1, and no post may present it as one.

## 1. Question

**Q-G1.** The L2 router is `guard_matern_core.fit_l2_residual_gate` (pub-r15 L1260–1361, file 9e0bd27f…). Does it gain
stack-level u when its candidate set grows from the smooth Matérn-3/2 kernel ridge to that kernel plus a non-smooth
dictionary, admitted through the same ratchet?

**Why the router's slot.**
- It is the one residual placement whose AUDIT gain matched its SEARCH estimate: on P201B, +0.048 against +0.046 over
  its 36 fires.
- §8b qualifies that: the match holds on precision contexts, and on ordinary contexts it keeps 0.5 to 0.7 of the
  estimate.
- Every column fitted to y paid the redundancy tax tonight. In A5 medium, any GP column added only +0.008 to +0.010 u
  over V0 at stack level.

## 2. Arms

All four router arms run in **one harness (D2)**: the same slot, the same nesting (N1/N2), the same 4-of-5 ratchet,
the same 0.004 floor, the same zero fallback and the same clamp (N4). Only the candidate set or the combiner differs.

| Name | What it is | Role |
|---|---|---|
| R0 `ROUTER_ASIS` | V0 with the gm_ts switch off: bit-identical to V0 under `--check` | rival (a), N3 |
| **R1 `ROUTER_DICT`** | the switch on: the router's candidates are its current (ell, lam, eta) kernel-ridge grid plus the dictionary configurations (§3) | **PRIMARY** |
| R2 `ROUTER_L3` | the same bank fitted as one dense ridge with inner-CV λ, as the candidate in the same slot | rival (b), N3 |
| R3 `C_STATIC` | agy's L1 static control as the candidate in the same slot. It does not exist at 38ff3faf (finding 3). Its compute is measured as worker CPU-seconds per context in the same receipt, and is at least 0.9 times R1's (one-sided). Both are measured in the same run, interleaved by context, so that load swings on xbabe2 cannot open a gap. | rival (c), N3 |
| S1 `JAC_V1` | agy's standalone APEX-EQ L4 block, fitted to y, as one more external block in V0 | pre-named SECONDARY |
| S0 `JAC_K` | the kernel-only JACOBI variant, with the same admission | control for S1 |

**The dictionary** is H460 phase 1 at the reviewed head of the follow-up PR to #2323 that fixes blocking findings 1–3:
- an exhaustive threshold scan per feature (`scan_best_step_features` at G1's settings: top_per_col, max_return,
  min_leaf);
- equality indicators;
- rational pair laws (`fit_rational_law_vp`);
- gated step-times-basis terms (`build_gated_smooth_features`);
- the B1/B2 kernels where the router admits them.

Every function named here is checked against the frozen head before freezing, as a grep of each symbol.

**R1's grid is frozen and counted (D3).** The ratchet (L1337–1343) keeps the best configuration that wins 4 of 5
folds. Every added configuration therefore raises the chance of a false fire, apart from admission inside a fold. The
frozen text lists R1's configurations and their number, and R0's.

## 3. Admission inside the router: today's rule, plus guards

- **Target:** the router's current one, the L1 blend's out-of-fold residual, clipped at ±3 MAD (L1317–1318).
- **Nesting (N1):** the whole discovery reruns inside each of the router's five folds, on that fold's training rows
  only. That covers the scan, the rational fits, gating, admission, the EB weights and the IRLS row noise.
- **Ratchet:** unchanged, the same code path as today.
  - A configuration counts only if it wins at least 4 of 5 folds (`new_f < base_f * 0.9995`, L1339–1341) and its
    pooled gain is at least 0.004 (L1345).
  - Otherwise the fallback is zero, and the arm's correction is zero on that context.
- **Damping (N2):** eta, the floor and the ratchet use the nested out-of-fold scores of N1 only, never a closed-form
  LFO of the final state.
- **Bag and full under nesting (D4).** L1360 returns 0.5 r_bag + 0.5 r_full, clipped at ±2.5 MAD.
  - For R1, r_bag averages the five fold models, each with the components discovered on its own fold's training rows.
  - r_full reruns discovery on all SEARCH rows, with the frozen settings.
  - So no final fit reuses components chosen on rows other than its own.
- **Guards:**
  - a winsorised score;
  - a single-row influence cap, which rejects a component if one row carries more than half of φᵀα;
  - c0 and τ(N) from G0_8's nested placebo, with the bank rebuilt on each permuted residual. The fixed-pool
    `calibrate_placebo_c0` at 38ff3faf does not qualify.
- **Clamp parity (N4):** every arm gets the router's AUDIT clip (±2.5 MAD, L1360), or each arm's unclamped read is
  printed beside its clamped one.

**One meaning of "fire" (D1).**
- A fire is a context where the arm's router returns a nonzero correction: `best_cfg` is not None and the gain is at
  least 0.004. The same definition holds for every arm.
- For R1, the fires whose configuration holds a dictionary component are printed as a subset.
- **AUDIT-positive** means the stack with the correction beats the stack with the router off (p1) on AUDIT. That is
  the reference of L2's 32 of 36.
- **The changed set** is the contexts where R1's correction differs from R0's. It is printed with W/T/L for R1 − R0.

## 4. Boards and order

1. **Selftests, with no board read.** All of the following must pass, with receipts in G1/:
   - G0 1–7 at the frozen head;
   - **G0_8**: at least 400 null fits; at most 0.05 false admissions per fit, with the one-sided Clopper-Pearson 95%
     bound; and the power check, in which the old fixed-pool placebo must fail clearly.
     - **In router form (D3):** R1's whole frozen grid goes through the ratchet on the null residuals, and false
       fires are counted as well as false admissions.
     - R0's null false-fire rate on the same fits is printed beside it.
   - **N5**: the canary on the real loader. AUDIT labels on disk are poisoned by +1000, and max_abs_diff must be 0 on
     every query prediction, with the bank and the incumbent hashed into the fit's scope;
   - the 26-sd single-row selftest (G0_6), against L3 as well as L2 (finding 10).
2. **Design board: the 121 P201B contexts outside S9's 80** (bank_P201B da29e8ec; V0 and L2 stored in
   S2_TUNE_P201B_r2).
   - At most four named configurations may be read on its AUDIT. Each read is logged with its configuration sha before
     it runs.
   - **Any change to the grid, the guards or the dictionary settings is a new configuration (D7),** and it counts
     against the four.
   - The configuration carried forward is frozen, code sha plus configuration, before FITSEL is opened.
3. **Decision board: FITSEL-106** (bank_FITSEL f8849ccb, GP pool H420 TLIB).
   - The primary population is **clean-88** (Q3): every context except the 18 in tier `copy` of EXCLUSION_FITSEL.json
     (be87f957…, ~/jope-data/h335-cl/h431/pub-r2; tiers clear 68, copy 18, possible 12, target_only 8). V0's r4_e8 is
     copy-tier on those 18.
   - All 106 are printed beside it, labelled in-corpus.
   - **FITSEL is also S9's confirm board (D6).** Neither line's FITSEL read informs the other. S9's arm is fixed by its
     selection record, and G1's configuration is frozen before FITSEL is opened.
4. **P201B-80:** descriptive only, and only after S9's selection record exists.
5. **dev91:** one look, ledgered, only if FITSEL passes. It is one of Track S's two remaining looks.
   - The arm is named in LOOK_LEDGER before the run.
   - So is the look's expected Elo, with its arithmetic (Q1).

## 5. The deciding gate, on FITSEL clean-88

**Statistic.** `h335_stats.primary_test` on stack u: clip ±0.5, family mean, one-sided sign-flip p, lofo_min > 0,
median ≥ 0. W/T/L on contexts (ties within 1e-4) and on families is printed for every contrast.

| Rule | Definition |
|---|---|
| **G1-P** | R1 − R0 passes at α 0.05, its family mean is above 0, and the bootstrap 95% lower bound of that mean is above 0. |
| **G1-R** | R1 − R2 and R1 − R3 each pass at α 0.05. Beating the router as it stands is necessary, not sufficient (N3). |
| **G1-S** (safety, D5) | R1 − R0 ≥ −0.02 u on every context, not only on fires, and the family mean of R1 − R0 ≥ −0.05 u in every family. |
| **G1-B** (breadth, D1 and §8b) | Fires on ordinary contexts (§8b): R1's number at least max(1.5 × R0's, 5), and at least 85% of those are AUDIT-positive. All fires and the precision fires are printed beside them. |

- **No multiplicity correction is needed (Q1).** G1-P, G1-R, G1-S and G1-B are an intersection: all must pass.
- **No mean-size rule for this arm (Q1, accepted by cl_fwriter).**
  - The contract auditor's +0.02 u continuation rule was set for broad changes.
  - R1 − R0 is exactly zero off the changed set. So the sign-flip test gets its signal only from changed contexts, and
    the mean is about the fire rate times the gain per fire: at L2's figures, 18% × 0.048 = +0.0086.
  - A +0.02 u rule would fail any router-class effect by design.
  - **Printed instead:** the per-fire gain with its bootstrap 95% interval; the number of changed contexts; and, in
    LOOK_LEDGER before any dev91 look, that look's expected Elo with its arithmetic.
- **The secondary.** S1 is read against V0 + S0 and against V0 with the same test. It is reported, and it cannot rescue
  a failed primary. If the primary fails and the secondary passes, any further read needs a new pre-registration.

## 6. Stop rules

- **K1.** G0_8 fails, on the rate, on the power check or on router-form false fires: no board is read.
- **K2.** On the design board, no configuration reaches max(1.5 × R0's, 5) ordinary fires: stop
  before FITSEL and post the null.
- **K3.** FITSEL fails any rule of §5: this line stops, and the result is posted with its labels.

## 7. Compute

- CPU only.
- **Track S caps:**
  - a systemd-run user unit with MemoryMax 16G and CPUQuota ≤ 800% on xbabe2;
  - nice 19, taskset 0-15,64-79, BLAS threads 1, PYTHONDONTWRITEBYTECODE=1;
  - stop if MemAvailable falls under 30 GiB or PSI full avg10 rises above 10%.
- Wall time is not known until three contexts are timed. That timing is logged before the design read.

## 8. Who does what

- **agy:**
  - the dictionary module on one follow-up feature-branch PR to the merged #2323 that fixes findings 1–3 (the replay
    reads a bank and refuses p201e; the nested placebo; C_static exists) and records 8–11. ENG:watcher's failing
    hygiene, jankurai-audit and aggregate checks go on the same branch, so there is one PR to review;
  - G0_8 as a Python test, in router form as well;
  - a router-callable entry point: (fold training X, fold residual, query X) → components and predictions.
- **cl_mtc:**
  - the gm_ts switch, default off, with V0 bit-identical under `--check`;
  - the harness for R0–R3, the scorer and the receipts;
  - this pre-registration.
- **cl_fwriter:** review of this text and of the code at its head.

## 8b. Where L2's gain lands today

Recounted at about 05:00Z with jq over stored receipts, after the tips workflow's synthesis claimed it. Precision
means Kumo L u ≥ 2 on that context.

| Board | Fires | Precision fires: AUDIT vs SEARCH gain | Ordinary fires: AUDIT vs SEARCH gain |
|---|---|---|---|
| P201B, 201 contexts (S2_TUNE_P201B_r2.raw.json; 81 of them precision) | 36 (32 positive) | 25 (23 positive): +0.0646 vs +0.0605 | 11 (9 positive): +0.0100 vs +0.0144, transfer 0.70 |
| dev91 e44, 182 context-seeds (S2_GUARD_DIAG_e44.json) | 35 (34 positive) | 14 (all positive): +0.0655 vs +0.0586 | 21 (20 positive): +0.0121 vs +0.0250, transfer 0.48 |

- **The 1:1 transfer holds on precision contexts only.** On ordinary contexts the router keeps 0.5 to 0.7 of its
  SEARCH estimate.
- Most of its gain sits where cl_h458's price list caps the total at about 60 Elo.
- So the breadth claim must rest on ordinary fires. G1-B and K2 are written that way.
- **The split needs no AUDIT read.** A context counts as precision if the L1 blend's SEARCH out-of-fold u is at least
  2. That works on FITSEL, which has no Kumo rows.
  - Agreement between this proxy and the Kumo L definition on the design board is printed. It decides nothing.
  - **One split file (Q4).** The split is computed once per board from the L1 blend, which comes before the router, so
    every arm shares it. It is written 0444 with its sha256 before any arm runs, and G1-B and K2 read only that file.

## 9. Review questions

- **Q4, resolved (cl_fwriter 05:04Z):** the SEARCH-OOF u ≥ 2 proxy is accepted, written as one 0444 split file.
- **Q5, resolved (cl_fwriter 05:04Z):** one-sided, R3 ≥ 0.9 × R1 in CPU-seconds, interleaved by context.
- No open question remains. Freezing waits only on the reviewed follow-up head named in the status line.
