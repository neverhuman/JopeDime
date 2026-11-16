# H458 Track S, S4: test-time compute for the ICL column (PREREG, DRAFT for review)

- **Author:** cl_mtc (Claude, xbabe2, session 7afc9e1c). Track S owner.
- **Status:** DRAFT. Nothing here has run. It is frozen (0444, SHA256SUMS) only after the team reviews it in
  shared_CHAT.md, and before any S4 column exists.
- **Owner rules that apply:**
  - Inference uses train-CV only. Every S4 computation for a context reads that context's SEARCH rows, its AUDIT or
    probe X, and frozen weights. It never reads AUDIT y, probe y, or a sibling context's labelled rows.
  - Training-time test insight (global knobs chosen on AUDIT labels of tuning contexts) is allowed.
  - Training on dev91 with different folds is fine, and in-corpus labels stay (Jepson, 2026-11-15 01:42Z, relayed by
    ENG:watcher).
- **Chat history this follows:**
  - cl_mtc 00:45Z (proposal) and 00:49Z (points taken).
  - cl_h458 00:47Z: a light fine-tune on 60b6 gave +0.01 to +0.02 u, peaking early.
  - agy 00:47Z: S4b-GPResid.
  - cl_fwriter 00:47Z and 00:48Z: the base facts and challenges.

## 1. Questions

| Question | Test |
|---|---|
| **Q-A (S4a, views)** | Does a Kumo-style view ensemble beat today's 8-view robust-ridge recipe, on the same checkpoint, folds and slots, as a stack column? |
| **Q-B (S4b, fine-tune)** | Does a per-context fine-tune on SEARCH rows improve the ICL column as a stack column? Asked twice: always on, and kept only where an honest inner-CV read says it helps. |
| **Q-C (S4b-GPResid, agy)** | Does fine-tuning on the residual of a GP anchor help, as one more stage of the owner's chain? Conditional, §4.4. |

**What decides.** The stack-level read decides, because Elo comes from the ladder, not from the column alone. The
column-alone read is reported beside it.

## 2. Bases and their provenance

| Role | Checkpoint | sha256 | What it trained on | dev91 exposure | Probe coverage |
|---|---|---|---|---|---|
| Probe base (in-corpus) | r4_L_A final, `~/jope-data/h450-quarantine/r4/finals/r4_L_A/ckpt_h450_final.pt` | 6b244ac784a16a53dd57a62f1048949387c7c87edb13a8583f02794a85378151 | init = the fair base below; then the r4 bank 9dfc60ed… (3,525 tables) | **In-corpus.** dev91 SEARCH ∪ AUDIT with y, about 255 exposures per AUDIT row | r4 bank: 0 hits (S7 build) |
| Fair base | H449 all_non_d91_L, `~/jope-data/h449-non-d91/ckpts_v2/all_non_d91_L/ckpt_unfair_final.pt` | b854b518b07b2c243bc62069b36e9cbd05b0a5cd77ca5faba60b5347623a3a29 | init cl_apex ICL_L_g1_w2 ckpt_h030 (637bc123…); then bank be7de9bc… (3,392 tables: 0 dev91, 0 D91-S3-COUSIN), 16,192 steps | **None in its own bank.** The init's APEX RealBank has not been checked (cl_fwriter 00:53Z). | H449 banks: 0 hits. Init bank: not checked. |

**Caveats on both bases.**
- **CONFIRM partition.** Both trained on 281 CONFIRM-partition tables through the non-canonical bucket hash (S8).
  The sealed APEX-CONFIRM contexts are not among them.
- **P201B exposure unknown.** Neither base's exposure to P201B's tables is established, so every P201B read carries
  that label, as the S9 receipts do.
- **Lineage.** r4_L_A descends from the fair base, so the pair isolates what r4's dev91-inclusive training added.

**Labels.**
- Every dev91 number on the probe base carries `in-corpus` and `canonical_promotion_forbidden`.
- A fair-base number is `fair closure (lineage root unchecked)` until the init's bank passes the S7 coverage check.
- Pre-launch task T0: I run `s7_bank_coverage.py` on that bank once its path and sha are named.

## 3. Boards, order and look budget

**1. Tuning (P201B, the 121 contexts outside S9's 80).**
- Fixes the fine-tune's two global knobs (§4.2) and checks timing.
- Training-time use of AUDIT labels. These contexts are never gate contexts.

**2. Gate (P201B-80, the S9 set).**
- The 80 P201B contexts with the smallest sha256(cid). Joined sha f3bf27eb…, list in `gp_chain/p201b_v1/CIDS_80.txt`.
- §5 decides here.

**3. Probe (dev91-freshprobe-v1).**
- Only the arm selected at the gate, against its control.
- Named on PROBE_LEDGER.jsonl with columns, baseline and prediction before it is scored, and scored once by
  `score_probe_h458s.py` f0885672.
- It confirms skill on rows no bank holds.

**4. dev91 (one official look).**
- Only if the probe read's T > 0.
- The primary arm is named in the look ledger beforehand.
- The look counts against Track S's two (Track S has used none). A second S4 look needs a new pre-registration.

## 4. Arms

### 4.1 Shared construction

**Pipeline.** Every S4 column is built by one new builder, `cols_s4_h458s.py`. It runs the same pipeline as
`cols_h450_gpu.py` (cc858eb9) for `icl_robridge_a`:
- REL structural columns (≤ 18);
- the robust-ridge residual target;
- the guard clip at the range ± 1.5 × span;
- random feature-ID slots.

The arms differ from that recipe only as §4.2 and §4.3 state.

**Folds.**
- One repeat of 5 target-stratified folds per context: the context's own `lab` folds, as `cols_h450_gpu` repeat 0
  uses. This is half of gpu450's two repeats, to halve the GPU cost.
- `q_` is out-of-fold on SEARCH, `bag_` is the mean of the 5 fold models on AUDIT, and `ref_` is a refit on all SEARCH
  rows.
- Every arm and its control share folds, struct columns, ridge base and seeds.

**Model.** The L checkpoint alone (EMA weights). No M pairing, so the comparison is clean.

**Controls.**
- C0a and C0b are today's recipe with view seeds a and b: 8 views of column permutations, mean.
- C0b − C0a is the S4 noise unit (§5).

### 4.2 S4b: fine-tune arms (Q-B)

**Fine-tune, per fold model.**
- It starts from the base weights.
- It trains on that fold's training rows only. The refit uses all SEARCH rows.
- Episodes are drawn from those rows:
  - a random 80/20 context/query split;
  - a fresh random feature-ID slot draw;
  - 20% feature dropout (agy);
  - the same struct columns and robust-ridge residual target as evaluation.
- The loss is the model's training loss, plus an L2-SP penalty α‖θ − θ0‖² with α = 0.003.
- Optimizer: AdamW, betas (0.9, 0.999), no weight decay (L2-SP replaces it), gradient clip 1.0, 4 episodes per step.

**Global knobs.** These are fixed on the tuning contexts before any gate column exists:
- lr ∈ {3e-6, 1e-5} × steps ∈ {10, 20, 40}, from one run per lr that saves weights at steps 10, 20 and 40.
- The rule: the (lr, steps) cell with the largest family-mean column-alone Δu over zero-shot, read on the tuning
  contexts' AUDIT.
- The step curve is posted. cl_h458 saw the gain peak early, so a 10-step winner is plausible.

**Arms.**

| Arm | Rule |
|---|---|
| **B1 (always on)** | Every fold model and the refit are fine-tuned. |
| **B2 (CV-kept)** | Per context, fine-tune or zero-shot. For the `q_` rows of fold k, the decision uses only the other four folds: keep the fine-tune if its OOF u on those folds' rows beats zero-shot by ≥ 0.01. `bag_` and `ref_` use the all-folds OOF gain at the same threshold. No row's own label decides its own `q_`. |

### 4.3 S4a: view-ensemble arms (Q-A)

**A16: 16 views.**
- View v uses target transform T[v mod 4] and feature transform F[⌊v/4⌋ mod 2], with a latin column order (a cyclic
  shift of one random base permutation) and FlipSign 0.5 per feature.
- The 16 view predictions are combined by a 20% trimmed mean per row.
- All transforms are fitted on the fold's training rows only.
- Target transforms T, each applied to the robust-ridge residual and inverted inside the bin integral (the
  `mean_warped` path):
  - identity;
  - Yeo-Johnson;
  - robust scale (median and IQR / 1.349) with a clip at ±3;
  - rank-Gauss (a normal quantile of the empirical CDF, inverted by piecewise-linear interpolation of the training
    residuals).
- Feature transforms F: none, or a per-feature Yeo-Johnson applied first. "None" is today's input: the model
  normalises every feature internally, with a robust z and a clip at ±8 robust units (`icl_model_h450.normalise_f64`,
  `CLIP_X` 8), so a power transform first changes what a heavy-tailed feature keeps.
- The model standardises y internally and its 1,024 bins cover ±40 sd (`BIN_RANGE` 40). The target transforms
  therefore act before that standardisation, and `mean_warped` integrates their exact inverse over the bins.

**A32: the same recipe with 32 views.** It answers whether more views keep paying.

A16 and A32 run on the zero-shot weights. The view recipe is not combined with a fine-tune in this pre-registration.

### 4.4 S4b-GPResid (Q-C, agy, conditional)

**Target.** The fine-tune and prediction target is the residual of a GP anchor column instead of the ridge residual.
The anchor is the S9 A0a column at the medium tier, from cl_h458's P201B block:
- its `q_` for the fold's training rows;
- `ref_` for the refit.
The ICL column is then anchor + ICL residual.

**Condition.** It runs only if S9's A0a medium block exists for all 80 gate contexts and its `q_` is out-of-fold.

**Labels.**
- P201B reads are labelled `fair-gp-search-only` (anchor) on the fair base, and `in-corpus` on the probe base.
- cl_fwriter's block on a "fair" label for r4 descendants stands.

**Ownership.** agy owns the arm's design. I build and score it under this PREREG only if agy agrees to these
settings. Otherwise it is dropped here.

### 4.5 Arm × base matrix and cost

| | Probe base (r4_L_A) | Fair base (H449 L) |
|---|---|---|
| C0a, C0b | yes | yes |
| A16, A32 | yes | yes |
| B1, B2 | yes | yes |
| B3 GPResid (Q-C) | conditional | conditional |

**Cost.** Not estimated from first principles.
- Task T1 times one context of each arm on 3 tuning contexts on the assigned GPU.
- The budget table (GPU-seconds per context per arm) is posted before the gate run.
- If the gate set would exceed 6 GPU-hours, the gate runs on the probe base first. The fair base follows only if the
  probe base passes §5.
- I estimate, untimed, ≤ 2 GPU-hours for C0, A16 and A32, and 2 to 4 for B1 and B2 on 80 contexts, on a 3090.

## 5. Gate rules (P201B-80)

**Utility and test.** Exactly as S9: u = −ln(RMSE / RMSE of the SEARCH mean) on AUDIT, and `h335_stats.primary_test`
(clip ±0.5, family mean, one-sided sign-flip p, lofo_min > 0, median ≥ 0). One family per context on P201B.

**Stack.** V0 is the S9 legacy stack (m7, k3, f4, icl r4_e8 family, plus the H416_CONFIRM GP pool), with the S4
column as one more external block. Computed by `s9_p201b_stack.py`'s machinery (the S4 scorer reuses it with the V0
bit-identity gates).

**Rules.**

| Rule | Definition |
|---|---|
| Noise | C0b − C0a, read at column-alone and stack level, each in both directions at α 0.025 (AMENDMENT_4 §2 of S9, applied here). Column-alone noise blocks S1 and S2. Stack noise blocks S2. |
| **S2 (stack, deciding)** | V0 + arm against V0 + C0a and V0 + C0b. Both pass at α 0.025. |
| S1 (column alone, reported) | arm − C0a and arm − C0b, both passing at α 0.025. It never selects by itself. |

**Selection.** Per base, the S2-passing arm with the largest stack T over C0a. Holm over the S2 tests of both bases is
reported.

**W/T/L.** W/T/L on contexts (ties within 1e-4) and on families is printed for every contrast (cl_h458's 00:47Z ask).

**Descriptive reads.** These are printed beside the rules and decide nothing:
- the tail-share and range-guard reads of S9's AMENDMENT_4 §7;
- the B2 keep rate per context;
- A32 − A16.

**Stop rule.** If no arm passes S2 on a base, S4 reports that base's null and goes no further for it.

## 6. Probe and dev91

**Probe.** The selected arm and its C0a run on dev91-freshprobe-v1 with the same builder. The ledger entry is written
first. It is reported whatever its sign, and T ≤ 0 stops S4 before dev91.

**dev91 look.**
- Primary: base + selected arm against base + C0a, on the selected arm's base.
- If both bases have a selected arm, the fair base's arm is primary and the probe base's is a pre-named secondary in
  the same receipt.
- Reported in full whatever its sign, with the in-corpus and fair labels of §2 and the closure guard.

## 7. Integrity

**Inference closure.** Per context, the builder reads only:
- the context's SEARCH rows and y;
- its AUDIT or probe X;
- the base weights.

The fine-tune runs on those SEARCH rows only. The builder refuses a bank that holds `ya`, as `cols_h450_gpu` does
(L470-471).

**Seeds.** Seeds are a function of (cid, fold, arm, view) only, so every rerun is bit-reproducible on one GPU type. A
cross-GPU rerun is not compared bit for bit (cross-arch tolerances, 10-07).

**Outputs.** Each block is written create-once (0444) with a SIDECAR:
- the base sha and closure;
- the knobs and code sha;
- `q_` honesty (out-of-fold);
- per-context GPU seconds.

**Selftests.** The builder's selftest covers:
- zero-shot C0 equals `cols_h450_gpu`'s `icl_robridge_a` fold predictions on one context, within a stated tolerance
  (bit-identity is not expected, because the repeat structure differs);
- a fine-tune of 0 steps equals zero-shot exactly;
- the identity-only, one-view A recipe equals C0 with one view.

## 8. Resources

**Compute.**
- A GPU slice is requested in shared_CHAT.md, with none claimed unilaterally: about 4 to 8 GPU-hours on one 24 GB card
  in total, including tuning.
- CPU parts (struct columns and the ridge base) run in a user unit with MemoryMax 16G, CPUQuota ≤ 800% and nice 19.
- On xbabe2 they run on cores outside cl_h458's 32-63 and 80-127 while S9 runs.
- Stop rules: MemAvailable < 30 GiB, or PSI full > 10%.

**Paths.** `~/h458s-stage/code`, `~/jope-data/h458s-quarantine/s4/`, and `~/jope_plan/evidence/H458/trackS/S4/`.
