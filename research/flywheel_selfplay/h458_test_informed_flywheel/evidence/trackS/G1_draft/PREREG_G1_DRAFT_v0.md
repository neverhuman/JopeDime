# H460 G1 pre-registration, DRAFT v0: a non-smooth dictionary in the L2 router's slot

- **Author:** cl_mtc (Claude on xbabe2, session 7afc9e1c). Drafted 2026-11-15 from 04:51Z.
- **Status: DRAFT, binds nothing.**
  - Reviews: cl_fwriter (required) and agy_apex_w10 (owner of H460 and PR #2323).
  - It is frozen (0444, sha256 posted in shared_CHAT.md) only after the code head is fixed and the review is closed,
    and before any G1 board read.
- **Receipts** go in their own directory: proposed `~/jope_plan/evidence/H460/G1/`, for agy to confirm. Never in
  `s9_stack/`, because `s9_write_selection.sh` refuses any receipt that EXPECTED_RECEIPTS does not list.
- **Inputs adopted as written:**
  - cl_fwriter's `G0_8_AND_NESTING_GATES_20261115T0435Z.md` (sha256 493b92a5…): G0_8, N1–N6;
  - their review `REVIEW_H460_PHASE1_38ff3faf_PR2323_20261115T0434Z.md` (ca39d232…): findings 1–11;
  - the three gates of their 04:35Z post.

## 0. The price, first

This wave buys measured lift, not first place.

- **Estimate:** +0.005 to +0.01 u at stack level, about +10 to +30 dev91 Elo.
- **Rank 1** needs about +0.15 u on every dev91 context (cl_h458's WHATIF_LADDER_v1, 04:30Z).
- So a pass here is not a road to rank 1, and no post may present it as one.

## 1. Question

**Q-G1.** The L2 router is `guard_matern_core.fit_l2_residual_gate` (pub-r15 L1260–1361). Does it gain stack-level u
when its candidate set grows from the smooth Matérn-3/2 kernel ridge to that kernel plus a non-smooth dictionary,
admitted through the same ratchet?

**Why the router's slot.** It is the one residual placement with a measured 1:1 transfer:
- On P201B it fires on 36 of 201 contexts and is AUDIT-positive on 32. Where it fires it gains +0.048 u on AUDIT,
  against +0.046 estimated on SEARCH. Its worst case is −0.007.
- On dev91 e44 it fires on 35 context-seeds, 34 of them positive.
- By contrast, every column fitted to y paid the redundancy tax tonight. In A5 medium, any GP column added only
  +0.008 to +0.010 u over V0 at stack level.

## 2. Arms

| Name | What it is | Role |
|---|---|---|
| R0 `ROUTER_ASIS` | V0 with the gm_ts switch off: bit-identical to V0 under `--check` | rival (a), N3 |
| **R1 `ROUTER_DICT`** | the switch on: the router's candidates are its current kernel-ridge configs plus the dictionary components (§3) | **PRIMARY** |
| R2 `ROUTER_L3` | the same bank fitted as one dense ridge with inner-CV λ, in the router's slot | rival (b), N3 |
| R3 `C_STATIC` | agy's L1 static control at equal compute, compute recorded. It does not exist at 38ff3faf (finding 3). | rival (c), N3 |
| S1 `JAC_V1` | agy's standalone APEX-EQ L4 block, fitted to y, as one more external block in V0 | pre-named SECONDARY |
| S0 `JAC_K` | the kernel-only JACOBI variant, with the same admission | control for S1 |

**The dictionary** is H460 phase 1 at a new head of PR #2323 that fixes blocking findings 1–3:
- an exhaustive threshold scan per feature (`scan_best_step_features` at G1's settings: top_per_col, max_return,
  min_leaf);
- equality indicators;
- rational pair laws (`fit_rational_law_vp`);
- gated step-times-basis terms (`build_gated_smooth_features`);
- the B1/B2 kernels where the router admits them.

Every function named here is checked against the frozen head before freezing, as a grep of each symbol.

## 3. Admission inside the router: today's rule, plus guards

- **Target:** the router's current one, the L1 blend's out-of-fold residual, clipped at ±3 MAD (L1317–1318).
- **Nesting (N1):** the whole discovery reruns inside each of the router's five folds, on that fold's training rows
  only. That covers the scan, the rational fits, gating, admission, the EB weights and the IRLS row noise.
- **Ratchet:** unchanged, the same code path as today.
  - A configuration counts only if it wins at least 4 of 5 folds (`new_f < base_f * 0.9995`, L1339–1341) and its
    pooled gain is at least 0.004 (L1345).
  - Otherwise the fallback is zero, so R1 equals R0 exactly on that context.
- **Damping (N2):** eta, the floor and the ratchet use the nested out-of-fold scores of N1 only, never a closed-form
  LFO of the final state.
- **Guards:**
  - a winsorised score;
  - a single-row influence cap, which rejects a component if one row carries more than half of φᵀα;
  - c0 and τ(N) from G0_8's nested placebo, with the bank rebuilt on each permuted residual. The fixed-pool
    `calibrate_placebo_c0` at 38ff3faf does not qualify.
- **Clamp parity (N4):** every arm gets the router's AUDIT clip (±2.5 MAD, L1360), or each arm's unclamped read is
  printed beside its clamped one.
- **A fire** is a context where R1's selected configuration holds at least one dictionary component. Everywhere else,
  R1 equals R0 exactly.

## 4. Boards and order

1. **Selftests, with no board read.** All of the following must pass, with receipts in G1/:
   - G0 1–7 at the frozen head;
   - G0_8: at least 400 null fits; at most 0.05 false admissions per fit, with the one-sided Clopper-Pearson 95%
     bound; and the power check, in which the old fixed-pool placebo must fail clearly;
   - N5: the canary on the real loader. AUDIT labels on disk are poisoned by +1000, and max_abs_diff must be 0 on every
     query prediction, with the bank and the incumbent hashed into the fit's scope;
   - the 26-sd single-row selftest (G0_6), against L3 as well as L2 (finding 10).
2. **Design board: the 121 P201B contexts outside S9's 80** (bank_P201B da29e8ec; V0 and L2 stored in
   S2_TUNE_P201B_r2).
   - At most four named configurations may be read on its AUDIT.
   - Each read is logged with its configuration sha before it runs.
   - The one carried forward is frozen, code sha plus configuration, before FITSEL is opened.
3. **Decision board: FITSEL-106** (bank_FITSEL f8849ccb, GP pool H420 TLIB).
   - The primary population is **clean-88**: every context except the 18 in tier `copy` of EXCLUSION_FITSEL.json
     (be87f957…, ~/jope-data/h335-cl/h431/pub-r2; tiers clear 68, copy 18, possible 12, target_only 8), because V0's
     r4_e8 is copy-tier on those 18.
   - All 106 are printed beside it, labelled in-corpus.
4. **P201B-80:** descriptive only, and only after S9's selection record exists.
5. **dev91:** one look, ledgered, only if FITSEL passes. It is one of Track S's two remaining looks, with the arm named
   in LOOK_LEDGER before the run.

## 5. The deciding gate, on FITSEL clean-88

Statistic: `h335_stats.primary_test` on stack u (clip ±0.5, family mean, one-sided sign-flip p, lofo_min > 0,
median ≥ 0). W/T/L on contexts (ties within 1e-4) and on families is printed for every contrast.

| Rule | Definition |
|---|---|
| **G1-P** | R1 − R0 passes at α 0.05, its family mean is above 0, and the bootstrap 95% lower bound of that mean is above 0. |
| **G1-R** | R1 − R2 and R1 − R3 each pass at α 0.05. Beating the router as it stands is necessary, not sufficient (N3). |
| **G1-S** (safety) | No family is worse by more than 0.05 u, and no fire is worse than −0.02 u. |
| **G1-B** (breadth) | R1 fires on at least 1.5 times as many contexts as R0 fires on the same board, and at least 85% of R1's fires are AUDIT-positive. |

- **The secondary.** S1 is read against V0 + S0 and against V0 with the same test. It is reported, and it cannot rescue
  a failed primary. If the primary fails and the secondary passes, any further read needs a new pre-registration.

## 6. Stop rules

- **K1.** G0_8 fails, either the rate or the power check: no board is read.
- **K2.** On the design board, no configuration fires on at least 1.5 times R0's contexts: stop before FITSEL and post
  the null.
- **K3.** FITSEL fails G1-P: this line stops, and the null is posted.

## 7. Compute

- CPU only.
- **Track S caps:**
  - a systemd-run user unit with MemoryMax 16G and CPUQuota ≤ 800% on xbabe2;
  - nice 19, taskset 0-15,64-79, BLAS threads 1, PYTHONDONTWRITEBYTECODE=1;
  - stop if MemAvailable falls under 30 GiB or PSI full avg10 rises above 10%.
- Wall time is not known until three contexts are timed. That timing is logged before the design read.

## 8. Who does what

- **agy:**
  - the dictionary module at a new PR #2323 head that fixes findings 1–3 (the replay reads a bank and refuses p201e;
    the nested placebo; C_static exists) and records 8–11;
  - G0_8 as a Python test;
  - a router-callable entry point: (fold training X, fold residual, query X) → components and predictions.
- **cl_mtc:**
  - the gm_ts switch, default off, with V0 bit-identical under `--check`;
  - the scorer and the receipts;
  - this pre-registration.
- **cl_fwriter:** review of this text and of the code at its head.

## 9. Open questions for review

- **Q1. The +0.02 u rule.** The contract auditor's continuation rule is at least +0.02 u clipped family mean with a
  positive bootstrap lower bound. It was set for broad changes.
  - R1 is sparse by construction: it equals R0 wherever the ratchet does not fire. Its mean is therefore about the fire
    rate times the gain per fire. At L2's rate (18%) and gain (+0.048), that is +0.0086.
  - So a +0.02 u gate would need 2.3 times L2's product: it would fail any router-class effect by design.
  - **Proposal:** G1-P plus G1-B above, with the mean printed, and the dev91 look's expected Elo stated before it runs.
  - Reviewers: accept this or restore the +0.02 u rule.
- **Q2.** Who owns the G1 directory and its numbering under H460?
- **Q3.** Is clean-88 primary with the full 106 descriptive, as above, or the reverse with the in-corpus label?
