# H458 Track S, S4: test-time compute for the ICL column (PREREG)

- **Author:** cl_mtc (Claude, xbabe2, session 7afc9e1c). Track S owner.
- **Status:** FROZEN at 2026-11-15T04:00Z (0444, listed in SHA256SUMS). No S4 column, S4 block, S4 builder run or T0 or T1 task
  has run; the S4 builder `cols_s4_h458s.py` is still being written and reviewed. This file is r3: draft r2
  (`drafts/PREREG_S4.DRAFT_r2.md`, 32724db9…) plus the T0 float32 and column-copy text (§2) and the answers to
  cl_fwriter's review of r2.
- **r3 changes** answer cl_fwriter's review of r2 (`~/jope_plan/evidence/H458_review/
  REVIEW_S4_PREREG_draft_r2_32724db9_20261115T0318Z.md`, sha256 7a265c8d…), findings Q1 to Q9, as posted at 03:27Z:
  - Q1 (§4.4, §5, §6): B3 is read at the gate only and cannot be selected for the probe or dev91 here;
  - Q2 (§4.4): B3 uses B1's tuned (lr, steps) and is always on;
  - Q3 (§4.2): the knobs are tuned separately per base, from the refit's AUDIT predictions;
  - Q4 (§4.1, §4.3, §7): the seed rule;
  - Q5 (§3, §6): one probe read per base, and the rule for the dev91 primary;
  - Q6 (§8): S4's GPU memory cap;
  - Q7 (§5): what the noise unit measures;
  - Q8 (§2): how T0b loads dev91 files;
  - Q9: cited times corrected (cl_fwriter's labels ran ahead of the clock; corrected by cl_fwriter at 03:14Z).
  - Also new in r3: §2's float32 tolerance, the scale-free column-copy test and the T0 tool selftest.
- **r2 changes** answered cl_fwriter's review of r1 (`REVIEW_S4_PREREG_draft_52320611_20261115T0252Z.md` and its
  CORRECTION, posted about 02:52Z and 02:55Z), cl_h458 (02:56Z) and agy (02:54Z):
  - §2: the lineage root's bank, with the provenance corrected (P1), and T0 widened;
  - §4.1: the C0 selftest tolerance as a number, and C0 named as not the 2 × 5 production recipe (P3a);
  - §4.2: B2 relabelled "approximately honest (nested-free keep rule)" (P3c); view seeds (P4); why the knobs are
    tuned on the column-alone read (P5); `b2_fold_agree_rate` (agy);
  - §4.4: B3's anchor fixed now as S9 A4 at medium, with agy's bag and ref construction, and B3 must also beat its
    anchor;
  - the "about 255 exposures per AUDIT row" figure is dropped (P6): no record I can cite holds it.
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
| Probe base (in-corpus) | r4_L_A final, `~/jope-data/h450-quarantine/r4/finals/r4_L_A/ckpt_h450_final.pt` | 6b244ac784a16a53dd57a62f1048949387c7c87edb13a8583f02794a85378151 (equals cl_fwriter's R4_FINAL_RECEIPT final_sha256) | init = the fair base below (manifest init b854b518); then the r4 bank 9dfc60ed… (3,525 tables) | **In-corpus.** The r4 bank holds dev91 SEARCH ∪ AUDIT with y (`d91_h450.py` L245-274). | r4 bank: 0 hits (S7 build) |
| Fair base | H449 all_non_d91_L, `~/jope-data/h449-non-d91/ckpts_v2/all_non_d91_L/ckpt_unfair_final.pt` | b854b518b07b2c243bc62069b36e9cbd05b0a5cd77ca5faba60b5347623a3a29 | init cl_apex ICL_L_g1_w2 ckpt_h030 (637bc123…, verified); then bank be7de9bc… (3,392 tables: 0 dev91, 0 D91-S3-COUSIN), 16,192 steps | **None in its own bank.** The lineage root's bank is checked by T0. | H449 banks: 0 hits. Root bank: T0. |

**The lineage root's bank** (cl_fwriter, about 02:52Z and 02:55Z, read from configs and resume records on three
hosts):
- ICL_L_g1_w2 ckpt_h030 trained on exactly one real bank, `/home/ubuntu/jope-data/apex-icl/bank/fit_icl_bank.npz`,
  sha256 f6e283cebf9c7e40cd62b7cedf8f1283d010f7dc5feff0ab5045d0b2fd8e8ce7. Those are the 419 FIT-ICL tables of H419
  addendum 1, with the same sha on all three hosts.
- So did the rest of its group (w0 on xbabe3, w1 on xbabe0) and its init, ICL_L_r4's checkpoint `L_r4_clone_src.pt`
  (b767b725). ICL_L_r4 trained from scratch: init null, clone null.
- The 104M root therefore saw fit_icl_bank and synthetic data only. agy's receipt c465426e finds 0 of 201 P201B
  contexts in fit_icl_bank by cid, family and H425 copy tier.

**T0, before any S4 column, CPU only, no label of AUDIT or the probe read.**
- **T0a.** fit_icl_bank against the probe rows, with a ledger entry first. It uses `s7_bank_coverage.py`'s matching
  (feature vectors and full rows, in the scaled and the raw space), with one change, below.
- **T0b.** fit_icl_bank against the dev91 contexts: SEARCH rows (full row and feature vector) and AUDIT X (feature
  vector only, so no AUDIT label is read).
  - It loads each dev91 file by a key allow-list (SEARCH X, SEARCH y, AUDIT X), never touches `ya` or `y_audit`, logs
    the keys it read, and never loads a whole file (cl_fwriter Q8; the H305 lesson that a whole-array load of a mixed
    cache counts as a prohibited read).
- **T0c.** The two bases' own banks (be7de9bc… and 9dfc60ed…) against P201B's SEARCH and AUDIT X. This settles
  "P201B exposure unknown" for them. For fit_icl_bank, agy's c465426e answers it.

**fit_icl_bank stores float32** (its `data.npy` is `<f4`). `s7_bank_coverage.py`'s tolerances, 1e-9 in the scaled
space and 1e-8 × range + 1e-9 in the raw space, are tighter than float32 rounding, about 6e-8 relative. Run as it is,
the tool would miss true copies and report a false zero.
- Every T0 match against a float32 bank therefore adds 2^-22 × max |value| per column to both tolerances.
- The earlier coverage receipt, on the float64 bank_v2_everything_f64.r3, is unaffected.

**Spaces.** A bank may hold a table in a different space from the target rows (raw against min-max scaled). So T0b and
T0c add a scale-free column-copy test in H425's sense, as agy's `audit_h459_fit_icl_overlap.py` (47f69186…) applies
it, beside the row tests.

**Tools.** T0's extensions keep the old tool read-only beside the new one. The new one's selftest:
- plants a float32-rounded copy of one target row in a toy bank and finds it;
- plants a min-max-scaled copy of one target column and finds it by the column test;
- finds nothing in a bank with no planted copy.

**Caveats on both bases.**
- **CONFIRM partition.** Both trained on 281 CONFIRM-partition tables through the non-canonical bucket hash (S8).
  The sealed APEX-CONFIRM contexts are not among them.
- **P201B exposure.** Until T0c reports, every P201B read carries "P201B exposure unknown", as the S9 receipts do.
- **Lineage.** r4_L_A descends from the fair base, so the pair isolates what r4's dev91-inclusive training added.

**Labels.**
- Every dev91 number on the probe base carries `in-corpus` and `canonical_promotion_forbidden`.
- A fair-base number is `fair closure (lineage root bank: coverage pending)` until T0a and T0b report no hit. A hit
  turns the label into `in-corpus (lineage root)`, with the hit counts.

## 3. Boards, order and look budget

**1. Tuning (P201B, the 121 contexts outside S9's 80).**
- Fixes the fine-tune's two global knobs (§4.2) and checks timing.
- Training-time use of AUDIT labels. These contexts are never gate contexts.

**2. Gate (P201B-80, the S9 set).**
- The 80 P201B contexts with the smallest sha256(cid). Joined sha f3bf27eb…, list in `gp_chain/p201b_v1/CIDS_80.txt`.
- §5 decides here.

**3. Probe (dev91-freshprobe-v1).**
- Only the arms selected at the gate, at most one per base (§5), each against its own base's C0a. B3 is never sent
  (§4.4).
- Named on PROBE_LEDGER.jsonl with columns, baseline and prediction before it is scored, and scored once by
  `score_probe_h458s.py` f0885672.
- It confirms skill on rows no bank holds.

**4. dev91 (one official look).**
- Only if a probe read's T > 0 (§6 names which arm is primary).
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
- So C0 is **not** the 2 × 5 production recipe. It is that recipe's repeat 0. Arms and controls share it, so the
  contrasts are fair, but C0's level is not gpu450's.
- `q_` is out-of-fold on SEARCH, `bag_` is the mean of the 5 fold models on AUDIT, and `ref_` is a refit on all SEARCH
  rows.
- Every arm and its control share folds, struct columns, ridge base and seeds.

**Model.** The L checkpoint alone (EMA weights). No M pairing, so the comparison is clean.

**Controls.**
- C0a and C0b are today's recipe with view-seed sets a and b: 8 views of column permutations, mean.
- C0b − C0a is the S4 noise unit (§5).

**Seeds** (cl_fwriter Q4).
- View and slot seeds are a function of (cid, fold, view-seed set a or b, view). Every arm except C0b uses set a.
- Fine-tune episode seeds are a function of (cid, fold, step), shared by B1, B2 and B3.
- B2 does not train again: it reuses B1's fine-tuned fold models and refit, and C0a's zero-shot predictions.
- A16 and A32 draw their base column permutation and FlipSign signs from set a, so A32's first 16 views are exactly
  A16's 16 views.

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
- B1 and B2 predict with C0a's view seeds and slots, so B − C0a isolates the fine-tune.

**Global knobs.** These are fixed on the tuning contexts before any gate column exists:
- lr ∈ {3e-6, 1e-5} × steps ∈ {10, 20, 40}, from one run per lr that saves weights at steps 10, 20 and 40.
- **Per base** (cl_fwriter Q3): the two bases are tuned separately, 2 bases × 2 lr runs, and each base uses its own
  cell.
- **Refit only.** The tuning read uses the refit's AUDIT predictions (`ref_`), so tuning fine-tunes one model per
  context and lr, not six. The fold models then use the chosen cell.
- **Size.** All 121 tuning contexts, unless T1 (§4.5) puts tuning above 1.5 GPU-hours per base. Then it uses the first
  60 of the 121 by sha256(cid), and the post says so before tuning starts.
- The rule: the (lr, steps) cell with the largest family-mean column-alone Δu over zero-shot, read on the tuning
  contexts' AUDIT.
- The step curve is posted. cl_h458 saw the gain peak early, so a 10-step winner is plausible.
- **Why the column-alone read tunes the knobs while the stack decides** (P5). A knob changes the column. The
  column-alone read measures that change directly. The stack read passes it through the guard's discrete admit and drop
  decisions, which make it a step function of the column near the guard's thresholds: a poor signal for choosing
  between six nearby cells. S2 then tests, on other contexts, whether the chosen column helps the stack.

**Arms.**

| Arm | Rule |
|---|---|
| **B1 (always on)** | Every fold model and the refit are fine-tuned. |
| **B2 (CV-kept)** | Per context, fine-tune or zero-shot. For the `q_` rows of fold k, the decision uses only the other four folds: keep the fine-tune if its OOF u on those folds' rows beats zero-shot by ≥ 0.01. `bag_` and `ref_` use the all-folds OOF gain at the same threshold. |

**B2's keep rule is approximately honest (a nested-free keep rule)**, not strictly honest (cl_fwriter, about 02:55Z):
- fold k's keep bit is decided from the OOF u of the fold-j models (j ≠ k) on fold-j rows;
- each fold-j model was fine-tuned on every row outside fold j, which includes fold k's rows and their labels;
- so a fold-k row's label shapes the weights of the models that decide its keep bit.

The effect is small and indirect, and it is conservative at S2: it can only make B2's `q_` look better than it is, so
the stack over-weights B2, and S2 is read on AUDIT rows that no model saw. A strictly honest version would need an
inner CV inside each fold's training rows, at about twice the fine-tune cost. It is not run here.

The SIDECAR logs `b2_fold_agree_rate` (agy): the share of contexts where all five fold keep bits equal the all-folds
bit used for `bag_` and `ref_`. It shows how often the threshold switches.

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

**Anchor: S9 A4 at the medium tier**, from cl_h458's P201B block. It was fixed at 03:03Z, before S9 scored anything,
whatever S9 selects.
- **Why A4, not A0a** (cl_h458 02:56Z, cl_fwriter about 02:57Z). A4's `q_` is honest: it is evolved out of fold, and its
  manifest has `honest_search_column` true. A0a's program was selected on the same SEARCH rows, so its `q_` carries
  selection optimism and its SEARCH residuals are too small.
- **Why fixed.** "The S9-selected arm" would make B3's S2 read depend on S9's selection on the same 80 contexts. A
  fixed anchor keeps it independent.

**Construction** (agy's detail, with A4 in place of A0a). Here `r_k` is fold k's ICL residual model and `r_refit` is
the refit's.
- Fold k's model trains on `y − q_A4` over fold k's training rows.
- `q_B3 = q_A4 + r_k(x)` on fold k's rows.
- `bag_B3 = bag_A4 + mean_k r_k(Xa)`.
- `ref_B3 = ref_A4 + r_refit(Xa)`.
- B3's `q_` is approximately honest in B2's sense. The anchor's out-of-fold values on fold k's training rows come from
  A4 programs that saw fold k's rows. It is conservative at S2 for the same reason as B2.

**Condition.** B3 runs only if S9's A4 medium block exists for all 80 gate contexts with `honest_search_column` true.

**Knobs** (cl_fwriter Q2). No A4 block exists on the tuning contexts, so B3 is not tuned. It uses B1's tuned (lr,
steps) for its base, unchanged, and is always on, as B1 is.

**Gate only** (cl_fwriter Q1, option a). A4 exists only on P201B-80, so B3 has no anchor and no anchor contrast on the
probe or dev91 contexts. B3 is therefore read at the gate only. It is outside §5's selection and is never sent to the
probe or dev91 in this pre-registration. A B3 gate pass is reported as such. Carrying B3 further needs a new
pre-registration that first builds A4 (frozen driver 2c3bf7446d, medium, SEARCH rows only) on those contexts.

**B3 must also beat its anchor.** S2 for B3 adds a third contrast: V0 + B3 against V0 + A4, at α 0.025.
- Otherwise a pass could come from A4's GP information alone, since V0 holds no S9 column.
- V0 + A4 comes from the S9 medium receipt's raw per-context stack u, bound by sha.
- The S4 scorer first checks that the two runs' V0 are bit-identical in every context.

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
- Task T1 times one context of each arm on 3 tuning contexts on the assigned GPU, under §8's memory cap, and times one
  refit fine-tune of 40 steps per base for the tuning cost.
- The budget table (GPU-seconds per context per arm, and the tuning estimate) is posted before tuning starts.
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
| **S2 (stack, deciding)** | V0 + arm against V0 + C0a and V0 + C0b. Both pass at α 0.025. For B3, V0 + B3 against V0 + A4 must pass too (§4.4). |
| S1 (column alone, reported) | arm − C0a and arm − C0b, both passing at α 0.025. It never selects by itself. |

**Selection.** Per base, the S2-passing arm with the largest stack T over C0a, among A16, A32, B1 and B2. B3 is never
selected (§4.4). Holm over the S2 tests of both bases, B3's included, is reported.

**What the noise unit measures** (cl_fwriter Q7). C0b − C0a is view-seed noise only. For B1 and B2 the fine-tune's
episode seed is a second noise source, and it is not measured here. The readout says so beside every B contrast. The
family sign-flip test remains the inference.

**W/T/L.** W/T/L on contexts (ties within 1e-4) and on families is printed for every contrast (cl_h458's 00:47Z ask).

**Descriptive reads.** These are printed beside the rules and decide nothing:
- the tail-share and range-guard reads of S9's AMENDMENT_4 §7;
- the B2 keep rate per context;
- A32 − A16.

**Stop rule.** If no arm passes S2 on a base, S4 reports that base's null and goes no further for it.

## 6. Probe and dev91

**Probe** (cl_fwriter Q5).
- Each base's selected arm (at most one per base, §5) and that base's C0a run on dev91-freshprobe-v1 with the same
  builder. Each arm is probed against its own base's C0a.
- The ledger entry is written first. Each read is reported whatever its sign.
- If no probe read has T > 0, S4 stops before dev91.

**dev91 look.**
- Primary: the fair base's selected arm if its probe T > 0; otherwise the probe base's selected arm if its probe T > 0.
  The primary is base + arm against base + C0a, on that arm's base.
- If both bases' arms have probe T > 0, the probe base's arm is a pre-named secondary in the same receipt.
- The primary and any secondary are named in the look ledger before the look.
- Reported in full whatever its sign, with the in-corpus and fair labels of §2 and the closure guard.

## 7. Integrity

**Inference closure.** Per context, the builder reads only:
- the context's SEARCH rows and y;
- its AUDIT or probe X;
- the base weights.

The fine-tune runs on those SEARCH rows only. The builder refuses a bank that holds `ya`, as `cols_h450_gpu` does
(L470-471).

**Seeds.** Seeds follow §4.1's rule and nothing else, so every rerun is reproducible on one GPU type up to amp
kernels. A cross-GPU rerun is not compared bit for bit (cross-arch tolerances, 10-07).

**Outputs.** Each block is written create-once (0444) with a SIDECAR:
- the base sha and closure;
- the knobs and code sha;
- `q_` honesty (out-of-fold);
- per-context GPU seconds.

**Selftests.** The builder's selftest covers:
- zero-shot C0 equals `cols_h450_gpu`'s `icl_robridge_a` repeat-0 fold predictions on one context, recomputed on the
  same GPU: max over rows of |C0 − ref| / sd(SEARCH y) ≤ 1e-4. Amp kernels need not be bit-reproducible, so bit
  identity is not required;
- a fine-tune of 0 steps equals zero-shot exactly;
- the identity-only, one-view A recipe equals C0 with one view.

## 8. Resources

**Compute.**
- About 4 to 8 GPU-hours on one 24 GB card in total, including tuning.
- **Card: the xbabe2 3090** (cl_h458 02:56Z; agy yields its option, 02:54Z), 24,576 MiB.
  - The 3.2 GiB of fleet_M_x2 (pid 1293542) stays; only Jepson may stop it.
  - agy's X3-B M partner (pid 1143890, from 03:23Z) is capped at `--cuda-vram-frac 0.30`, 7.4 GiB.
  - **S4's memory cap** (cl_fwriter Q6). The builder caps its own allocator with
    `torch.cuda.set_per_process_memory_fraction`:
    - 8.5 GiB while any non-owner process shares the card, which keeps the card at or under about 19.5 GiB, Jepson's
      80% target (3.1 + 7.4 + 8.5 plus about 0.5 GiB of CUDA context);
    - 12 GiB only when S4 is alone with fleet_M_x2.
  - nvidia-smi's per-process memory is read and posted before each GPU start. T1 measures S4's peak under the cap.
  - If S4 needs more than its cap, it chunks queries; it never raises the cap without a post.
  - Data loaders use a few threads, on cores 0-15 and 64-79, never on S9's 32-63 and 80-127.
- CPU parts (struct columns and the ridge base) run in a user unit with MemoryMax 16G, CPUQuota ≤ 800% and nice 19.
- On xbabe2 they run on cores outside cl_h458's 32-63 and 80-127 while S9 runs.
- Stop rules: MemAvailable < 30 GiB, or PSI full > 10%.

**Paths.** `~/h458s-stage/code`, `~/jope-data/h458s-quarantine/s4/`, and `~/jope_plan/evidence/H458/trackS/S4/`.
