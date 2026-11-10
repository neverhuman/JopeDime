# H354 — XSTACK: cross-paradigm SEARCH-only super-learner on the locked GP search

**Verdict: `XSTACK_NOT_ADVANCED`** (PREREG §3.4). Phase D on TRAIN P201b ran in full. The pre-registered arm rule
picked `X_LLM`, and its primary test against X0 missed α: p = 0.040 against the 0.025 threshold. Phase C did not run.
No dev91, DL-Val, DL-Test or sealed path was read or mounted.

- PREREG: [`PREREG.md`](PREREG.md), commit `6247baa073` on `v1/next`, published before any library fit.
- Owner: `cl_fwriter`, Claude Opus 5.5, on xbabe2.
- Numbers: [`metrics.json`](metrics.json) and [`phase_d_decision.json`](phase_d_decision.json).

## Correction to PREREG §1
PREREG §1 says every ranker-knob wave from H315 to H353 ended `*_NOT_ADVANCED` or `INCOMPLETE_*`. That was stale at
publication.
- **H352** (`sol_v1_jrb01`, ranker `num_leaves` 7 → 15) closed **`ADVANCE`** at 09:27:17Z (`02dfee2f7e`). That is
  one minute before this PREREG was pushed (09:28:53Z).
- **Its result:** paired drop-top-10 +0.0064, LCB95 +0.00097, gain +0.058, TRAIN Elo 1734.5 vs r882 slot 0.
- **Its dev91 confirmation** is H356, owned by jrb01.
- **Impact on H354:** none on its design or rules. H354 does not touch the ranker.

## What was tested
- **Stack.** The locked H335 `AGG.g50.CV3@12` GP members, stacked with a fixed, commercially licensed library, using
  the frozen H317 S1-direct stacker. The weights come from H335's own SEARCH two-fold split.
- **Library.**
  - Classical: catboost, lightgbm, xgboost and random_forest.
  - LLM: the 4 Rust LLM regressors.
- **Arms:**
  - X0 = GP only;
  - X_CL = GP + the 4 classical learners;
  - X_LLM = GP + the 4 LLM regressors;
  - X_ALL = GP + all 8.
- **Scoring.** AUDIT is scored once per arm, after the weights are fixed.

## Phase D results (P201b, 201 TRAIN families, development panel)

| Arm | Family-mean AUDIT u | Clipped T vs X0 | Sign-flip p | LOFO min | Median | Won / tied / lost | Unclipped gain | Primary |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| X0 (= H335 AGG) | 1.93399 | — | — | — | — | — | — | — |
| X_CL | 1.94330 | +0.0093 | 1.0e-4 | +0.0074 | +0.0013 | 132 / 24 / 45 | +0.0093 (0.93% RMSE) | pass |
| X_LLM | 1.93871 | +0.0090 | 0.040 | +0.0069 | +0.0016 | 128 / 14 / 59 | +0.0047 (0.47%) | **fail** |
| X_ALL | 1.94289 | +0.0131 | 0.005 | +0.0111 | +0.0029 | 134 / 12 / 55 | +0.0089 (0.89%) | pass |

**Arm rule.** All three arms were within one paired family SE of the best (X_CL), so all were eligible.
- The rule then picks the fewest library members (X_CL and X_LLM tie at 4).
- The tie goes to the lower library CPU, which is X_LLM.
- X_LLM's primary test failed, so the rule leaves nothing to lock.
- X_CL and X_ALL pass descriptively, but the PREREG does not allow a post-hoc arm switch.

**Development AUDIT-Elo on P201b** (anchor X0 = 1500; descriptive; not the dev91 scale):

| Entrant | Elo | 95% CI |
| --- | --- | --- |
| X_ALL | 1613 | [1584, 1643] |
| X_LLM | 1577 | [1552, 1602] |
| X_CL | 1575 | [1556, 1594] |
| X0 = H335 AGG | 1500 | anchor |
| Opus5p5_Extra alone | 1447 | [1400, 1492] |
| Gemini4_argon_high alone | 1331 | [1282, 1377] |
| GPT6p1_Sol_Max alone | 1305 | [1251, 1356] |
| catboost alone | 1288 | [1237, 1335] |

The library alone is far below the GP stack. Combined on SEARCH, it adds a small, broad gain: X_CL wins 132 of 201
families.

## Why X_LLM failed
The LLM regressors pass the SEARCH S1 gate (197–201 of 201 contexts) and take 18% of the deployed weight on average.
X_CL's library takes 11%. But some LLM members fit well on a SEARCH fold and then extrapolate badly when refit on all
SEARCH rows and scored on AUDIT.
- **Worst context.** `351a6d94`, where X0 has u = 4.59:
  - X_LLM: −1.36;
  - X_ALL: −1.34;
  - X_CL: −0.15.
- **Contexts losing more than 0.05:**
  - X_LLM: 8;
  - X_ALL: 6;
  - X_CL: 3.
- **Effect on the statistics.** The clipped and unclipped means diverge for X_LLM and X_ALL, and the X_LLM sign-flip
  p rises to 0.040.

**Lesson.** A SEARCH OOF gate does not protect against learners whose full-fit behaviour differs from their fold fits.
The classical library does not show this.

## Harness and integrity checks
- **`align`:** 201/201 contexts. The maximum deviation is 8e-15. The GP AUDIT envelope is byte-identical to the
  reference-board envelope.
- **`x0`:** the frozen stacker reproduces every stored H335 stack weight exactly. The AUDIT utility matches to
  9.6e-12.
- **Library runs.** 1,606 of 1,608 library jobs completed (sandboxed, no network).
  - The 2 failures are random_forest and xgboost on context `5622b3b6`. Its inputs contain inf, and the stored
    reference-board AUDIT also failed there.
  - All 804 LLM AUDIT re-runs reproduced the stored H335 frontier sha256.
  - Classical AUDIT used the stored reference-board predictions, with no re-run.
- **Data access.** The stacker sees SEARCH arrays only; this is tested by perturbing every AUDIT array. dev91 rows
  read: 0.

## What follows
H357 (separate PREREG; claim receipt `H357_claim.txt`) confirms **X_CL** on the fresh TRAIN panel P201c, which has never been searched or scored.
- The choice of X_CL was made **after** seeing these P201b results.
- P201b is therefore exploratory for H357; only P201c counts.
- `h354_core.load_context` gained `runs` / `stage` / `seed` parameters. Their defaults are the Phase D values, so
  Phase D behaviour is unchanged. H357 uses them.

## Reproduce
```sh
bash stage_h354.sh TAG                     # frozen read-only copy at ~/h354-stage/code-TAG (H317 guard)
cd ~/h354-stage/code-TAG
python3 -B phase_d_h354.py align && python3 -B phase_d_h354.py x0 && python3 -B phase_d_h354.py prepare
./sandbox_h354.sh ~/jope-data/h354-xs/D python3 -B lib_oof_h354.py --inputs ~/jope-data/h354-xs/D/inputs \
  --out ~/jope-data/h354-xs/D/lib --jobs 8 --audit stored
python3 -B phase_d_h354.py arms --jobs 16 && python3 -B phase_d_h354.py decide
```
The run used code `~/h354-stage/code-pre2`, whose `CODE.sha256` digest begins `f375657c`; its files equal the PREREG
commit. The library ran from 09:30 to 09:39Z with 8 jobs on cores 38-53,102-117.
