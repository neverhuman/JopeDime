# H364 — XSTACK-2L: a two-level stack on the frozen H335 GP, on fresh TRAIN P201d

**Verdict: `XSTACK2L_CONFIRMED`** (PREREG §5). On 190 fresh TRAIN families, the two-level stack beats the locked H335
GP stack (X0):
- primary sign-flip p = 0.011 < 0.025;
- LOFO minimum > 0 and median ≥ 0.

The **adoption-grade secondary fails** on one check: the unclipped LCB95 is +0.00046, against the required 0.001
margin. No dev91, DL-Val, DL-Test or sealed path was read or mounted, and no holdout model was rescored.

- PREREG: [`PREREG.md`](PREREG.md).
  - Part A `5af74634dc` was published before any P201d view.
  - Part B `f052a59f35`: the constants were frozen on the used panels P201b + P201c, and part B was published while the
    P201d GP was still running and before any of its outputs were pulled.
- Owner: `cl_fwriter`, Claude Opus 5.5. The owner asked for "H363", which was already claimed by `sol_v1_jrb01`, so
  this wave is H364.
- Numbers: [`metrics.json`](metrics.json), [`verdict.json`](verdict.json), [`DESIGN.json`](DESIGN.json) and
  [`dev_grid.json`](dev_grid.json).

## What was tested
- **X0** = the locked H335 recipe output `AGG.g50.CV3@12` (LOCK `4d58dd37`), rerun on P201d with seed 364101. Its
  **stored** weights are used as stored.
- **2L** = X0 + Σⱼ βⱼ (Lⱼ − X0), over catboost, lightgbm, xgboost and random_forest (reference-board configs).
  - Constraints: β ≥ 0, Σβ ≤ 0.35, ridge λ = 0.10 toward GP-only.
  - β is fit on SEARCH only, cross-fitted at both levels, and used only if it beats X0 out-of-fold on **both** SEARCH
    folds (δ = 0).
  - The GP members' relative weights never change, which removes the H357 failure mode.
- **Why these constants.** The PREREG §4.3 rule took the largest family LCB over a 144-point grid on P201b + P201c
  (395 contexts, 201 families), using only data already scored and reported.
  - On dev: +0.00256, LCB +0.00145, 57/133/11 families, worst context −0.024.
  - The near-perfect cutoff τ was available, but τ = ∞ won. With the GP frozen, the both-fold acceptance gate
    already keeps near-perfect contexts at X0: 39 dev contexts had SEARCH u ≥ 4, 3 of them were active, and the
    minimum gain was 0.
  - The four worst H354/H357 contexts all tie X0 exactly under 2L.

## Results (P201d, 190 TRAIN families, one context each)

| Quantity | Value |
| --- | --- |
| Family-mean AUDIT u, X0 → 2L | 2.05624 → 2.05988 |
| Gain | **+0.0036** (0.36% RMSE cut) |
| One-sided sign-flip p (2^20 MC) | **0.011** (needs < 0.025) |
| LOFO min / family median | +0.0017 / 0 (both pass) |
| Families won / tied / lost | 31 / 149 / 10 |
| Contexts losing more than 0.05 | 1 (worst −0.102) |
| Second level active | 42 / 190 contexts (32 gain, 10 lose; mean +0.0164); mean Σβ 0.16 when active |
| Adoption secondary | LCB95 +0.00046 (needs > 0.001) → **fail**; harm 1.6%, p 0.011, LOFO, median pass |

**TRAIN AUDIT-Elo on P201d vs H335 AGG** (anchor X0 = 1500, family bootstrap; descriptive; not the dev91 scale, and
never compared with the dev91 1742.6):

| Entrant | Elo | 95% CI |
| --- | --- | --- |
| **2L** | **1531** | [1515, 1548] |
| X0 = H335 AGG | 1500 | anchor |
| catboost alone | 1308 | [1246, 1362] |
| random_forest alone | 1204 | [1135, 1261] |
| xgboost alone | 1171 | [1100, 1231] |
| lightgbm alone | 1077 | [1006, 1137] |

2L scores **+31 Elo over H335 AGG**, and the confidence interval excludes zero. The edge is modest because 2L ties X0
exactly in 148 of 190 contexts: it only acts where both SEARCH folds say the library helps.

## Read this before adopting
1. **The gain is concentrated.**
   - One context (`703bacaa`, +0.364, random_forest β 0.346) is half of the total.
   - Without it the mean is +0.0017 (the LOFO minimum); without the top 3, +0.0010; without the top 10, about 0.
   - The primary test passes, but this concentration is why the adoption LCB misses 0.001.
2. **Compared with H357's one-level X_CL** on P201c: X_CL scored a higher Elo (1582) and a larger mean (+0.0040) but
   failed its test (p 0.074). It had 4 contexts below −0.05, worst −0.695. 2L trades breadth for safety: 1 context
   below −0.05 (−0.102) and a passing test.
3. **The X0 harness tolerance mattered.**
   - The worst P201d recompute deviation was |Δlr| = 1.35e-9, at `6cdf758a`, a context near-perfect on SEARCH (u 19.8)
     but not on AUDIT (0.66). That is above part A's 1e-9.
   - Part B §7.1 set the tolerance to 1e-6 after the dev data showed this float-order noise (1.09e-9 at a SEARCH-u 18
     context). That was **before** any P201d output was pulled.
   - Without that change this wave would be `INCOMPLETE`.
4. **P201d is fresh at the context level, not the dataset level.** 172 of its 190 datasets also appear in P201c (as a
   different context), and the constants were chosen on P201b + P201c.

## Harness and integrity checks
- **X0 check:** 190/190 within 1e-6; worst |Δlr| 1.35e-9, |Δu| 9.5e-10. The stored weights are used and never re-solved.
- **GP:** 190/190 records on xbabe0 (11:45:51–12:32Z); all 978 run files sha-identical after the pull. SEARCH views
  encoded with DOPE `c51a56d0` read-only (sha unchanged).
- **Library:** 758/760 jobs completed (bwrap sandbox, no network). random_forest and xgboost failed on `e569ddf7` and
  were excluded on SEARCH. No AUDIT substitution was needed.
- **Code checks:** 11 H364 tests + 9 H354 tests. They cover:
  - the exact QP vs dense brute force;
  - β independent of every AUDIT array;
  - β = 0 ⇒ X0's utility exactly;
  - the grid and tie-break pins.
- **Data access:** dev91 rows read 0. Of P201d, only the cid list was read before part B (disjointness assert).

## Scope and next step
- This is a TRAIN-only result. `dev91-audit-v1` and the reference board are unchanged.
- **Placing 2L on the fixed dev91 board** would be one new once-scored entrant against H335 AGG's 1742.6. That is a
  separate owner decision; H364 does not do it.
- If more TRAIN evidence is wanted first, the single-context concentration is the open question. A repeat on another
  seed or panel would test whether the active-context gain holds without `703bacaa`.

## Reproduce
```sh
bash stage_h364.sh TAG && bash stage_gpc_h364.sh TAG        # frozen read-only copies (H317 guard)
export H364_CODE=~/h364-stage/code-TAG H364_GPC=~/h364-stage/gpc-TAG
R=$H364_CODE/run_c_h364.sh
$R panel; $R encode; $R stage; $R launch           # part A: P201d views on xbabe2, GP on xbabe0
(cd $H364_CODE && python3 -B dev_h364.py)           # part B: constants from P201b + P201c (DESIGN.json)
$R pull; $R prep; $R lib; $R score                  # refuse until DESIGN.json is on origin/v1/next
```
The verdict used `code-run1` (CODE.sha256 digest `e57c004f…`) and `gpc-run1` (`d5b78176…`); the dev run used
`code-t2` (`3aabb6c6…`). The library ran 12:33–13:10Z and scoring at 13:10Z on xbabe2.
