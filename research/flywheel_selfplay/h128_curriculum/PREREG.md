# H128 Preregistration — Loss-Ordered Self-Paced Curriculum (`CURRIC` & `ANTI` vs `CONTROL`)

- **Wave**: `H128`
- **Owner**: `rd_argon` (`[ARGON]`)
- **Authorization**: `/home/ubuntu/nudges/rd_argon_t1130.md` (`2026-10-31T17:30:00Z`)
- **Execution Mode**: CPU-only (`nice 15`, `<= 6` threads (`4` used), `RSS <= 6 GB`, `MemAvailable >= 30 GB`, `load1 <= 86.7`, `disk <= 94%`, `0` GPU seconds, no `qualification.lock`)
- **Source Base (`origin/main`)**: `8c73150986d77436b6a206a4e39474e9067e257b`

---

## 1. Scientific Hypothesis & Mechanism

Every recent 40-step AdamW fine-tune from `r882` has declined on fresh family-held-out `DL-Train` assessment (`H125` trained `CONTROL` `-0.063180`, `H126` trained `CONTROL` `-0.045027` vs unchanged `r882`). A plausible mechanism is that severe high-loss contexts (where `r882` incurs large smooth-L1 + KL listwise errors on $-2.0$ gate-failure candidates) dominate early AdamW updates and pull the `29,211`-parameter `adapter_10tok` away from `r882`'s high-utility ranking basin.

`H128` tests whether a **loss-ordered self-paced curriculum (`CURRIC`)** — sampling at update $u \in \{0, \dots, 39\}$ only fit-fold contexts whose frozen `r882` baseline loss lies in the easiest $q_u$ fraction, with $q_u$ rising linearly from $0.50$ to $1.00$ across the `40`-update budget — prevents early high-loss destabilization and improves held-out family utility over uniform sampling (`CONTROL`) and unchanged `r882`, while including a reversed hard-to-easy schedule (`ANTI`) as a directional/placebo control.

### Three Preregistered Arms

All three arms use the exact `H126`/`H59`/`H97` `r882` optimizer and step budget:
- Initialized from an independent, SHA-256-verified fresh copy of `r882` (`/home/ubuntu/jope-data/h51/ckpt/round-882.pt`, `sha256 = 0084479aa0703667a0f3ccfe2beda60741c204795ec57b7c08542401ec6cdb64`).
- `DOPE` backbone frozen (`sha256 = c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814`, `eval()` mode locked, verified before and after every cell); only the `29,211` non-anchor `adapter_10tok` parameters (`initial_adapter_sha256 = 740174e9f0d29b69ea1b3dad2e4545e40eff541501e6bfdec1e79dbe96912183`) are trainable.
- `20` rounds $\times$ `2` AdamW steps/round = `40` optimizer updates per cell (`batch_size = 16` contexts/step, `slate_size = 192` candidates/context, `lr = 1e-4`, `weight_decay = 1e-4`, `grad_clip = 1.0`).
- Per-context objective: `h59.preregistered_targets` with `F.smooth_l1_loss(pred, target, beta=1.0).mean(dim=1) + 0.1 * F.kl_div(F.log_softmax(-pred / 0.2, dim=-1), F.softmax(-target / 0.2, dim=-1)).sum(dim=1)`.

1. **`CONTROL` (Plain Uniform Fit-Fold Sampling)**:
   - At each update $u = 2(\text{rnd}-1) + \text{step} \in \{0, \dots, 39\}$, samples `16` contexts uniformly with replacement from the rotated full fit-fold context list ($N_{\text{fit}} = 975$ contexts in `folds 1..4`) using `paired_stream_seed(seed, rnd, step)` and `cached_training_indices(..., slate_size=192)`.
2. **`CURRIC` (Easy-to-Hard Self-Paced Loss-Ordered Curriculum)**:
   - **Frozen Pre-Update Fit-Fold Loss Ranking (`0` optimizer updates)**: Before any optimizer update, evaluate unchanged `r882` on each of the $N_{\text{fit}} = 975$ fit-fold (`folds 1..4`) contexts using the deterministic top-`192` candidate slate (`orders[c][:192]`) to obtain the frozen baseline per-context loss $\ell_0(c)$. Assessment families (`fold 0`, `41` families / `244` contexts) are strictly excluded from this ranking.
   - Sort the $975$ fit-fold contexts in ascending order of $(\ell_0(c), c)$: $c_{(1)}, c_{(2)}, \dots, c_{(975)}$ (easiest to hardest).
   - At update index $u \in \{0, 1, \dots, 39\}$ ($U = 40$), define the curriculum pace:
     $$q_u = 0.5 + 0.5 \cdot \frac{u}{U - 1} = 0.5 + \frac{0.5 u}{39} \in [0.5, 1.0], \qquad K_u = \max\bigl(16, \lceil q_u \cdot N_{\text{fit}} \rceil\bigr)$$
   - The eligible context set at update $u$ is the easiest $K_u$ fit contexts $E_u^{\text{CURRIC}} = \text{sorted}(\{c_{(1)}, \dots, c_{(K_u)}\})$. Contexts are sampled via the same round rotation (`rot = rnd % len(E_u)`) and deterministic `random.Random(paired_stream_seed(seed, rnd, step)).choices(rotated, k=16)` and candidate slates (`cached_training_indices`).
3. **`ANTI` (Hard-to-Easy Direction / Placebo Control)**:
   - Uses the exact same frozen fit-fold loss ranking and linear pace $q_u \in [0.5, 1.0]$ ($K_u = \max(16, \lceil q_u \cdot N_{\text{fit}} \rceil)$), but restricts the eligible context set at update $u$ to the **hardest** $K_u$ fit contexts $E_u^{\text{ANTI}} = \text{sorted}(\{c_{(N_{\text{fit}} - K_u + 1)}, \dots, c_{(N_{\text{fit}})}\})$.

---

## 2. Fresh Family-Grouped `DL-Train` Partition & Paired Seeds

- **Source Corpus**: `1,219` `DL-Train` contexts across `205` families (`/home/ubuntu/jope-data/h51/input/selected_contexts.json`, `sha256 = 981960ad628f5029f8c54552c7b1cb1d440072e7e9dcffecdea52c6a3493a0f1`).
- **Fresh Family Split**: `fold_seed = 128076` (`np.random.default_rng(128076).permutation(205)` split into 5 folds via `np.array_split`, `fold_manifest_sha256 = 010e5c3857acf7b7e62ed2da10bc5f332bf7e65fcd62f82a7d80ea995e5b7e8c`):
  - **Assessment (`fold 0`)**: `41` families, `244` contexts (held out from all curriculum ranking, calibration, and training updates).
  - **Fit (`folds 1..4`)**: `164` families, `975` contexts.
- **Paired Seeds**: `76, 77, 78` (`3` arms $\times$ `3` seeds = `9` cells, `360` total AdamW updates).

---

## 3. Pre-Update Gates

1. **Unchanged-`r882` Regression-First Gate (`0` updates)**:
   - Load two independent fresh copies of `r882` (`net_a`, `net_b`).
   - Verify exact-zero (`0.0`) max absolute prediction and utility difference across all `244` `fold 0` assessment pools, and verify `0.0` loss and gradient difference on a reference fit-fold minibatch before any optimization.
   - Emit create-only S3 receipt `regression_receipt.json` before calibration or training.
2. **Frozen Fit-Fold Loss Ranking Calibration (`0` updates)**:
   - Compute $\ell_0(c)$ across the `975` fit-fold (`folds 1..4`) contexts on unchanged `r882` under `torch.no_grad()`, verify `adapter_sha256` and `DOPE` SHA-256 are unchanged (`adapter_mutated = false`), and emit create-only S3 receipt `curriculum_ranking_manifest.json` before the first optimizer update.

---

## 4. Endpoints, Statistical Readout & Admission Rules

- **Evaluation Metric**: H61 unrevealed-candidate utility (`top-4` argmax utility on each of the `244` `fold 0` assessment pools, aggregated equally within each of the `41` assessment families).
- **Readout Comparisons**:
  1. **Paired vs `CONTROL`** (`CURRIC - CONTROL`, `ANTI - CONTROL`) across the `41` assessment families and seeds `76, 77, 78`.
  2. **Paired vs Unchanged `r882`** (`CURRIC - r882`, `ANTI - r882`, `CONTROL - r882`) across the `41` assessment families and seeds `76, 77, 78`.
- **Bootstrap & Multiple Testing**:
  - `10,000`-draw paired family-cluster bootstrap (`bootstrap_seed = 128076`) over the `41` assessment families (averaging across the 3 paired seeds per family) reporting two-sided `95% CI` and one-sided lower confidence bound `LCB95`.
  - 2-arm Holm step-down across `{CURRIC, ANTI}` at family-wise $\alpha = 0.05$ (ordered thresholds `0.025`, `0.050`).
- **Admission Gate (`ADMITTED` vs `NO_ADMISSION`)**:
  - An arm is admitted only if **all** hold on `fold 0` (`41` families):
    1. `median_seed_gain > 0` (strictly positive across seeds `76–78`; ties = reject),
    2. `LCB95 > +0.001` (passing 2-arm Holm step-down at $\alpha = 0.05$),
    3. Per-seed harmed-family fraction ($\Delta_f < -0.01$) $\le 10\%$ ($\le 4/41$ families) in **every** seed (`76, 77, 78`),
    4. Directionality check: `CURRIC` strictly exceeds `ANTI` (`median_seed_gain(CURRIC - ANTI) > 0`).
  - **Breakthrough Escalation Rule**: If `CURRIC`'s `95%` family-bootstrap lower bound vs unchanged `r882` is `> 0` with family-harm caps ($\le 10\%$) passing, flag **`CANDIDATE BREAKTHROUGH`** and stop for overseer red-team.
- **Elo Reporting**: Report both pooled paired family Elo (`95%` bootstrap CI) and joint Bradley-Terry Elo (`fit_bradley_terry_elo`) anchored at `CONTROL = 1500` and `r882 = 1500` (never AUC).
