# H151 — Lookahead vs drift-matched AdamW step-size controls on the r882 continuation

**Owner:** `rd_argon` `[ARGON]`. Authority: `~/nudges/h151_rd_argon_t2055.md` & `~/nudges/h151_rd_argon_t2130.md`. Claim: `~/jope_plan/ns/receipts/H151_claim.txt`.
`rd_astra` cross-reviews read-only.

**Status:** preregistration committed before any scored inference or optimizer update in H151. Exploratory research on repeatedly exposed `DL-Val`; not sealed-test qualification or serving authorization.

## Question and provenance

In H146 (`research/flywheel_selfplay/h146_lookahead_exec`, merged `8b025049a9e43e4228cd61ecd122b49ba1c36eb9`, closed in #1284), Lookahead (`LA`, $k=5, \alpha=0.5$ over inner AdamW `lr=1e-4, wd=1e-4`) beat the live-recipe `CTRL` on `3/3` seeds by `+0.0140594` (`95% CI [+0.0014398, +0.0331268]`, Elo `1555.26`) while halving adapter L2 drift (`~0.217–0.219` vs `CTRL` `~0.438–0.443`), though neither arm beat unchanged `r882` (`1.1377942` context mean / `1.1536153` 19-family mean).

**Question:** Is Lookahead's `+0.0141` advantage over `CTRL` a Lookahead-specific trajectory effect (slow-weight interpolation after 5-step fast exploration) or simply the consequence of a smaller effective step size ($\alpha=0.5$ halving adapter displacement)?

**Code provenance:**
- Reuses the merged H146 harness (`research/flywheel_selfplay/h146_lookahead_exec/{runner.py,runtime.py,science.py,monitor.py,test_h146.py}`) without editing H146 files.
- Changes only the wave/path/seed/fold constants, the four fixed arms (`CTRL`, `CTRL_HALF`, `CTRL_QTR`, `LA`), the prespecified drift-matched control selector and inclusive 15% multiplicative manipulation check (`match_drift_control`), the primary `LA` minus drift-matched `CTRL` contrast (`mechanism_verdict`), unified 4-arm Holm step-down vs `r882` (`decide`), and cross-arm identical-argmin pseudoreplication reporting.

## Fixed arms, seeds, and folds

Every cell starts from a SHA-pinned copy of `round-882.pt` (`0084479aa0703667a0f3ccfe2beda60741c204795ec57b7c08542401ec6cdb64`) with a fresh AdamW optimizer (`weight_decay = 1e-4`), `40` updates (`20` rounds $\times 2$ retained steps from the H140/H146 8-draw paired sampler), batch `16`, slate `192` (`48` fitness-ranked + `144` random), gradient norm clip `1.0`, SmoothL1 ($\beta=1.0$) + `0.1` listwise KL ($\tau=0.2$), eval mode throughout, and only the `47` non-anchor `adapter_10tok` tensors (`29,211` parameters) trainable.

| Arm | Inner AdamW `lr` | `weight_decay` | Lookahead $(k, \alpha)$ | Evaluated weights at step 40 |
| --- | --- | --- | --- | --- |
| `CTRL` | `1e-4` (`0.0001`) | `1e-4` | `None` | step-40 fast weights |
| `CTRL_HALF` | `5e-5` (`0.00005`) | `1e-4` | `None` | step-40 fast weights |
| `CTRL_QTR` | `2.5e-5` (`0.000025`) | `1e-4` | `None` | step-40 fast weights |
| `LA` | `1e-4` (`0.0001`) | `1e-4` | $k=5, \alpha=0.5$ | synced slow = fast weights at step 40 (`sync = [5, 10, 15, 20, 25, 30, 35, 40]`) |

- **Seeds:** `151101, 151102, 151103`.
- **Family fold seed:** `151140` (`205` `DL-Train` families split into 5 disjoint 41-family folds; Fold 0 has `251` contexts and is secondary/descriptive only; Folds 1–4 have `968` fit contexts; see [fold_manifest.json](fold_manifest.json)).
- **Bootstrap seed:** `151141` (`10,000` shared family-cluster draws).
- **Update budget:** `4 arms x 3 seeds x 40 updates = 480 updates` on `xbabe1` CPU (`nice 19`, total OS threads $\le 4$, aggregate RSS $< 6\times 10^9$ B, free disk $\ge 40$ GB, load guard `85`).

## Drift-matched control and manipulation check

Let $D_a = \text{median}_{s \in \{151101, 151102, 151103\}} \text{drift}(a, s)$ be the median final adapter L2 drift from `r882` for arm $a$.
- **Drift-matched control ($c^*$):** Among the plain AdamW controls $c \in (\text{CTRL\_HALF}, \text{CTRL\_QTR}, \text{CTRL})$, select:
  $$c^* = \arg\min_{c} |D_c - D_{\text{LA}}|$$
  (ties broken by fixed order `CTRL_HALF`, `CTRL_QTR`, `CTRL`).
- **Manipulation check (inclusive multiplicative band `DRIFT_MATCH_BOUNDS = (0.85, 1.15)`):**
  $$0.85 \, D_{\text{LA}} \le D_{c^*} \le 1.15 \, D_{\text{LA}}$$
  Evaluated directly via `0.85 * la_med <= matched_med <= 1.15 * la_med` so exact endpoints (`0.85 * LA` and `1.15 * LA`) pass without floating-point subtraction rounding. If $D_{c^*}$ lies outside $[0.85 \, D_{\text{LA}}, 1.15 \, D_{\text{LA}}]$, the manipulation check status is `DRIFT_MATCH_INVALID`, `decision["LA"]["gates"]["drift_match"]` is `False`, `decision["LA"]["advance"]` is `False`, and the wave mechanism verdict is `DRIFT_MATCH_INVALID`.

## Primary estimand, secondary contrasts, and decision rules

1. **Regression gate (before any cell):** Unchanged `r882` must reproduce all `91/91` `DL-Val` reference utilities (`best_dev_checkpoint.json`) with `max_context_error == 0.0` and `mean_error == 0.0` (`1.1377941898164525` context mean, `1.1536152769912449` 19-family mean).
2. **Primary contrast (`primary_la_minus_matched_ctrl`):** Equal-family native argmin utility (`-fitness` if valid else `-2.0`) on the `91` `DL-Val` contexts (`19` families) for `LA` minus the drift-matched control $c^*$, median across the 3 seeds, with a 10,000-draw shared 19-family bootstrap 95% percentile CI (`BOOT_SEED = 151141`), `ties=reject`.
3. **Secondary contrasts and 4-arm Holm multiplicity (`HOLM_M = 4`):**
   - Every executed arm (`CTRL`, `CTRL_HALF`, `CTRL_QTR`, `LA`) is contrasted vs `r882`, vs `CTRL`, and vs $c^*$ on 91-ctx `DL-Val` (`19` families), with paired family Elo vs `r882 = 1500.0` (ties `0.5`, regularization `0.01`).
   - Because the dispatch specifies that any arm can trigger `ADVANCE` if it beats `r882` on `3/3` seeds with `CI LB > 0`, all four executed arms (`CTRL`, `CTRL_HALF`, `CTRL_QTR`, `LA`) are included in `decide(contrasts, drift_match)` under a single 4-hypothesis Holm step-down family (`HOLM_M = 4`), with no undocumented `CTRL` bypass.
   - For `LA`, per-arm `decision["LA"]["gates"]` includes `drift_match = drift_match["drift_match_valid"]` (with `efficacy_only_advance` recorded separately for diagnostic transparency), so `decision["LA"]["advance"]` and the aggregate verdict are always consistent.
   - Descriptive Fold 0 (`41` `DL-Train` families, `251` contexts) contrasts and invalid-pick counts.
   - Across-seed (`seed_pseudoreplication`) and cross-arm within-seed (`cross_arm_argmin_pseudoreplication`) identical-argmin utility/choice flags.
4. **Verdicts:**
   - **`DRIFT_MATCH_INVALID`**: $D_{c^*} \notin [0.85 \, D_{\text{LA}}, 1.15 \, D_{\text{LA}}]$.
   - **`LOOKAHEAD_SPECIFIC`**: `drift_match_valid` holds AND `LA` minus $c^*$ has `median_gain > 0.0` and 95% bootstrap `ci95[0] > 0.0` (`ties=reject`).
   - **`STEP_SIZE_ONLY`**: `drift_match_valid` holds AND (`ci95[0] <= 0.0` or `median_gain <= 0.0`, i.e., CI spans 0 or matched control $\ge$ `LA`).
   - **`ADVANCE` (vs `r882`)**: Only if an arm in `decision` (`CTRL`, `CTRL_HALF`, `CTRL_QTR`, `LA`) satisfies all per-arm gates: `seeds_beating == 3` (`3/3` seeds `> 0`), `ci95[0] > 0.0`, `median_gain > 0.001`, `top3_positive_share <= 0.8`, Holm-adjusted $p < 0.05$ (`HOLM_M = 4`), and (for `LA`) `drift_match == True`. Otherwise `r882_verdict = NO_ADVANCE` and no serving promotion occurs.
