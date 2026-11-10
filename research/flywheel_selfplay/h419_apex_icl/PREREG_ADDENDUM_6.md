# H419 PREREG addendum 6 — prior version 2: simple function families; one recipe for both sizes

**Status.** Published on `origin/v1/next` before any training process is restarted with the new arguments. No
prediction on a test cohort (stored P201b matrices, APEX-DEV, CONFIRM) exists. Everything below uses the TRAIN-side
monitor (106 FIT-SEL contexts, SEARCH rows only), stored H417 SEARCH-row matrices of the same contexts, and
synthetic tasks.

**Reason.** Owner instructions of 2026-11-10 (get the win rates up; make the larger model work), and what the
checks of 16:40–17:05Z found.

## 1. Findings since addendum 5
**(a) Against our own GP programs on the monitor's rows the model is far behind** (`icl_refs.py`; the H417 deploy stored
out-of-fold predictions of each context's stack candidates for the same folds):

| Predictor on fold 0 | Median u | Mean u |
| --- | --- | --- |
| Ridge | 0.636 | 1.061 |
| 24M model (5.0 h), plain / residual of ridge | 0.713 / 0.919 | 1.036 / 1.356 |
| 104M model (4.6 h), plain / residual of ridge | 0.587 / 0.880 | 0.904 / 1.299 |
| Best GP program by cross-validation on folds 1–4 | 1.200 | 1.738 |
| Caruana blend of the GP programs, fitted on folds 1–4 | 1.228 | 1.774 |

- The 24M residual readout beats the best GP program on 0.10 of the 106 contexts and the blend on 0.04.
- **As a stack column it adds nothing:** Caruana with the model's out-of-fold column (24M, residual readout) uses it
  in 38% of contexts at a mean weight of 0.05 and changes fold-0 u by +0.001 (19% better, 19% worse, 62% equal).
- **As a corrector of the blend, untrained for it, it hurts:** blend + model beats the blend on 0.18–0.22 of contexts.

**(b) Why: the synthetic prior has almost no easy tasks, and real tables are mostly easy.**

| Tasks | Ridge u, mean / median | Share with ridge u above 1 |
| --- | --- | --- |
| Real monitor contexts | 1.06 / 0.64 | 0.33 |
| Prior version 1: `scm` / `gp` / `tree` / `mix` | 0.27 / 0.12 / 0.15 / 0.04 (means) | 0.02 / 0.03 / 0.00 / 0.00 |
| New families: `lin` / `add` / `addint` | 1.39 / 0.63 / 0.65 (means) | 0.51 / 0.22 / 0.20 |

The 24M model on fresh draws of the new families (`icl_simple.py`): on purely linear targets its median u is 0.63
against ridge's 1.11 (it wins 35%); on additive targets 0.72 against 0.60. It never learned to fit simple structure
precisely, because training never asked for it. The residual-of-ridge readout patches the linear part only.

## 2. Change: prior version 2 (`icl_prior.PROBS_V2`)
| Family | Version 1 | Version 2 |
| --- | --- | --- |
| `scm` random causal graphs | 0.35 | 0.20 |
| `gp` random-Fourier GP draws | 0.25 | 0.15 |
| `tree` random forests | 0.20 | 0.15 |
| `mix` GP with forest | 0.20 | 0.10 |
| `lin` linear, dense with decaying coefficients or sparse | — | 0.10 |
| `add` additive: a sum of random univariate functions (linear, saturating, hinge, V, quadratic, slow sine, step, log, sigmoid) | — | 0.20 |
| `addint` additive plus 1–5 pairwise products | — | 0.10 |

Noise, target warps, feature warps, missing values, column injection and the real-table tasks are unchanged.
Version-1 draws are bit-identical to before (tested against stage `r8`), so the existing synthetic validation set
does not change.

## 3. Where it applies
- **104M group** (`ICL_L_g1_w0/w1/w2`): `--prior v2` added to the addendum-5 arguments (`--p-resid 0.5 --row-floor 1024`).
  The rolling restart of addendum 5 had reached worker 2 only; it was paused, and all three workers are now moved to
  the same arguments in one pass, by the same procedure and gates (code stage `r9`).
- **24M run `ICL_M_r3`** (xbabe1): restarted from its own checkpoint with `--prior v2 --p-resid 0.5 --row-floor 1024`.
  This replaces "the 24M run is not changed" of addendum 5. Reason: finding (b) says the old prior is the main
  defect, and leaving our best model and fastest GPU on it for 40 more hours would only produce a control that the
  xbabe4 arms give more cleanly (§4). Both sizes now train on one recipe, which also makes the size rule a like-for-
  like comparison again.
- **Rules.** The size rule stays as amended in addendum 5 (residual readout; hour-12 reference 0.868 / 0.919 from the
  24M checkpoint of 6.3 h, taken before this change). The 24M hour-12 on-track rule stays on the plain readout.

## 4. Control: one more arm on xbabe4
`ICL_M_x4d_v2`: the recipe of `ICL_M_x4c_res` plus `--prior v2`. With the arms already running this gives a ladder at
24M from one initial checkpoint on one machine, compared at equal training hours:

| Arm | Prior | Residual-target tasks |
| --- | --- | --- |
| `ICL_M_x4b_inj` | version 1 | no |
| `ICL_M_x4c_res` | version 1 | yes |
| `ICL_M_x4d_v2` | version 2 | yes |

- **Reading at hour 24** (mean of the last five evaluations; plain-readout median u and residual-readout median u on
  the monitor): `PRIOR_V2_HELPS` if `x4d` is above `x4c` by at least 0.03 on both; `HURTS` if below by at least 0.03 on
  either; `NO_EVIDENCE` otherwise. Reported besides: the share of contexts where blend + model beats the blend.
- Four arms share one Apple GPU, so each sees fewer tasks than planned; the comparisons stay matched in hours. The
  new arm runs with a memory cap (`PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.2`) and its own memory guard.

## 5. What is tracked
Every evaluation of a restarted run adds a block on the simple families alone (`u_simple`, `u_lin`, `u_add`,
`u_addint`, each with ridge and with the residual readout). The numbers that must move:
1. plain-readout median u on the monitor (24M now 0.71; ridge 0.64; best GP program 1.20);
2. `u_lin` against ridge on linear tasks (24M now about 1.06 against 1.44 in the validation block);
3. the share of contexts where blend + model beats the blend (now 0.18–0.22; it has to pass 0.5);
4. the stack test of §1(a) (now +0.001).

## 6. Expectations and risks
- **Expected:** `u_lin` and the plain real-table readout rise within hours. **Not promised:** that the model reaches
  the GP programs in this cycle; the gap is 0.3–0.5 u.
- **Third change in two hours on the 104M group** (residual tasks, longer rows, prior). On that group the effects
  cannot be separated; §4 separates them at 24M.
- **Old validation numbers will look worse.** The version-1 synthetic validation set stays as it is, but a model
  trained on version 2 spends 40% of its tasks elsewhere. A fall there is not, alone, a regression.
- **Training loss is not comparable across the change.**
- **Share of simple tasks is a guess** (0.40). It was not tuned; the real-table distribution of ridge utility sits
  between `lin` and `add`.

## 7. Code
`icl_prior.py` (families `lin`, `add`, `addint`; `PROBS_V2`, `PROBS_SIMPLE`), `icl_train.py` (`--prior`, the simple
validation block), `icl_inject.py` (family probabilities passed through), `icl_simple.py` (new), `icl_refs.py`
(`stack`), `icl_diag.py` (out-of-fold columns, corrector readouts). Checks before publication: version-1 draws
identical to stage `r8`; all families finite with non-constant targets; a 27-step run with all three arguments on
the Apple GPU logs every block.
