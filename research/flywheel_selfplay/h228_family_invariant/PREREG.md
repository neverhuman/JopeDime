# H228 — Family-Invariant Adapter Continuation: AND-Mask ($\tau=0.8$) and GroupDRO ($\eta=0.01$)

Prospective preregistration under `~/nudges/hw_t1640_argon_h228.md` (`rd_argon` `[ARGON]`); no H228 GPU outcomes exist at preregistration. Merge this source before any native regression, smoke, or training. This wave reuses the verified H219/H220 GPU0 execution harness (`research/flywheel_selfplay/h219_fbal_accum_run/`, `research/flywheel_selfplay/h220_family_mixup/`), changing only the per-step gradient-combination rule across family-stratified microbatches.

## Hypothesis and Fixed Design

Plain AdamW continuation from `round-882.pt` (`r882`) overfits across dataset families (`H176` `DATA_SATURATED`; `H219` `FBAL` and `H220` `MIX` closed `NO_ADVANCE`). H228 tests whether restricting adapter continuation updates to directions whose gradient signs agree across families (`ANDMASK`, Parascandolo et al. 2020) or up-weighting the highest-loss families (`GDRO`, Sagawa et al. 2020) suppresses family-idiosyncratic overfitting and yields positive held-out-family lift versus `r882`.

### Arms (Adapter-Only from Read-Only `r882` Copy; DOPE Frozen `c51a56d0`)

- **`R882` (Reference, `0` updates)**: Immutable `round-882.pt` (`SHA256: 0084479aa0703667a0f3ccfe2beda60741c204795ec57b7c08542401ec6cdb64`), DOPE composite `c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814` verified before and after every fit.
- **`CTRL` (Control, `40` updates/fit)**: Live AdamW continuation recipe identical to H219 `CONTROL` (context-uniform sampling via `legacy.sample_microbatches(groups, "CONTROL", training_seed)`, `4` microbatches of `16` contexts per step, `lr=1e-4`, `weight_decay=1e-4`, `betas=(0.9, 0.999)`, gradient norm clip `1.0`).
- **`ANDMASK` (Primary Candidate, `40` updates/fit)**: Uses the H219 `FBAL` family-stratified sampler (`legacy.sample_microbatches(groups, "FBAL", training_seed)`, verified $\ge 4$ distinct families per microbatch/step; observed $12 \le M_{t,m} \le 16$ per microbatch and $43 \le M_t \le 58$ per step). Within each family-stratified microbatch, computes per-family mean loss $L_{t,m,f}$ and per-family active-parameter gradient $g_{t,m,f} = \nabla_{\theta_{\text{active}}} L_{t,m,f}$ across the $41$ active adapter tensors ($6$ inactive adapter tensors remain `None`). Keeps only coordinates whose gradient sign agrees in $\ge \tau = 0.8$ of the $M_{t,m}$ families (`5 * max(pos_count, neg_count) >= 4 * M_{t,m}`), takes the mean across families of the kept components, and zeroes all other coordinates:
  $$g^{\text{ANDMASK}}_{t,m} = m_{t,m} \odot \left(\frac{1}{M_{t,m}} \sum_{f=1}^{M_{t,m}} g_{t,m,f}\right), \qquad m_{t,m,j} = \mathbb{I}\!\left(\max\!\left(\frac{1}{M_{t,m}}\sum_f \mathbb{I}(g_{t,m,f,j} > 0),\; \frac{1}{M_{t,m}}\sum_f \mathbb{I}(g_{t,m,f,j} < 0)\right) \ge 0.8\right)$$
  Accumulates $\frac{1}{4}\sum_{m=0}^3 g^{\text{ANDMASK}}_{t,m}$, clips global L2 norm to `1.0`, and steps AdamW (`lr=1e-4`, `weight_decay=1e-4`, `betas=(0.9, 0.999)`).
- **`GDRO` (Co-Candidate in Holm Family, `40` updates/fit)**: Uses the same paired H219 `FBAL` family-stratified microbatches and candidate slates as `ANDMASK`. Maintains simplex weights $q \in \Delta^{|\mathcal{G}_{\text{FIT}}|-1}$ initialized uniformly ($q_f^{(0)} = 1 / |\mathcal{G}_{\text{FIT}}|$) at the start of each fit. On each family-stratified microbatch with observed families $\mathcal{F}_{t,m}$ and detached per-family losses $\hat{L}_{t,m,f}$, updates $q_f \leftarrow q_f \exp(\eta \hat{L}_{t,m,f})$ with fixed $\eta = 0.01$ (Sagawa et al. 2020), renormalizes $q$ over $\mathcal{G}_{\text{FIT}}$, normalizes active step weights $w_{t,m,f} = q_f / \sum_{f' \in \mathcal{F}_{t,m}} q_{f'}$, and combines $\sum_{f \in \mathcal{F}_{t,m}} w_{t,m,f} g_{t,m,f}$ under the same `4`-microbatch accumulation, clip `1.0`, and AdamW step.
- **`ANDMASK_SHUF` (Placebo Control, `40` updates/fit)**: Uses the same paired H219 `FBAL` family-stratified microbatches and candidate slates as `ANDMASK`, but replaces $m_{t,m}$ with a deterministic random coordinate permutation mask (`sci.stream(seed, outer, step, micro, "andmask_shuf")`) that keeps the **exact same per-step/per-microbatch coordinate count and kept fraction** as `ANDMASK` on that `(seed, outer, step, micro)`.

## Cohort, Folds, and Ordering

- **Real S3 Data Only**: Pinned `selected.json` (`1,219` `DL-Train` regression contexts across `205` families, plus `91` `DL-Val` regression contexts across `19` disjoint families) verified against [S3_DATASET_CATALOG.md](file:///home/ubuntu/jope_plan/master/source/S3_DATASET_CATALOG.md) and [provenance.json](provenance.json). Zero synthetic, sealed, holdout, or meta-test reads.
- **Seeds and Partitions**: Seeds `228101, 228102, 228103` with fresh family-grouped 5-fold partitions `228141, 228142, 228143` ([fold_manifest.json](fold_manifest.json)). [fold_jaccard.json](fold_jaccard.json) records all pairwise fold overlaps against `H109/H210/H215/H217/H218/H219/H220/H221/H222/H223/H224/H225` (maximum mean best-match Jaccard `0.185110 < 0.20`).
- **Pre-Fit Native91 Gate**: Before any fit or optimizer update, unchanged `r882` is evaluated on all `91` `DL-Val` contexts and must match all `91` pinned utilities in `best_dev_checkpoint.json` bit-exact with mean `1.1377941898164525`. Cached static `r882` forward must also match native `r882` bit-exact on all `1,219` `DL-Train` contexts.
- **Smoke + Full Budget + Global Lock**:
  - Timing smoke: `1` FIT-only `CTRL` run (`seed=228001, outer=0, steps=4`), `0` held-out reads, included in the single attempt budget.
  - Full OOF training: `4` arms $\times$ `3` seeds $\times$ `5` folds = `60` independent fits $\times$ `40` updates/fit = `2,400` full optimizer updates (`2,404` total updates including smoke).
  - All `60` trained adapter states and proofs are fsynced to `selection-lock.json` and S3-recovered (`IfNoneMatch='*'`, exact `VersionId` GET + SHA256 verify) **before** any held-out `A` (`DL-Train` OOF) or trained `DL-Val91` utility is read.

## Readout and Admission Gates

- **Primary Contrast**: `ANDMASK` vs `R882` on 5-fold family-grouped OOF (`205` families / `1,219` `DL-Train` contexts across `3` seeds).
- **Holm-Bonferroni (`m = 2`)**: Applied across `CANDIDATES = ("ANDMASK", "GDRO")` (`CTRL` and `ANDMASK_SHUF` are controls; `ANDMASK_vs_CTRL`, `GDRO_vs_CTRL`, `ANDMASK_SHUF_vs_CTRL`, and `ANDMASK_vs_ANDMASK_SHUF` are also reported).
- **Admission (`CANDIDATE` / `BREAKTHROUGH`)**:
  - `CANDIDATE` on TRAIN OOF requires:
    1. Median equal-family gain across `3` seeds strictly $> +0.002$ (`math.fsum` per-seed family mean; exact ties reject),
    2. 10,000-draw paired family bootstrap (`BOOT_SEED = 228150`) 95% CI lower bound $> 0$ and Holm-adjusted $p < 0.05$,
    3. All `3/3` seed gains $> 0$, and
    4. Harm fraction (families with 3-seed mean gain $< -0.01$) $\le 10\%$.
  - `BREAKTHROUGH` requires passing the TRAIN OOF `CANDIDATE` gate **and** passing the `DL-Val91` (`91` contexts / `19` families, `VAL_BOOT_SEED = 228152`) gate (`median_gain > +0.002`, `CI95_lower > 0`, `3/3` positive seeds, `harm <= 10%`).
  - Also reported per arm: family-grouped Elo vs `R882` (`r882 = 1500`, `ELO_SEED = 228151`), per-step kept-fraction traces, and GroupDRO weight range. Regression only; no AUC.

## Resource Guards and Deadlines

- **Host / GPU**: `xbabe2` `GPU0` (`GPU-44c63d19-492a-ab46-cc6a-56e12a2ef51b`) under `~/qualification.lock` FIFO (`flock`; released on exit).
- **Limits**: `nice 19`, $\le 4$ CPU threads (`os.sched_setaffinity(0, {0, 1, 2, 3})`), `1` numeric thread, process-tree `RSS < 6 GiB` (`6,000,000,000` B), `MemAvailable >= 20 GiB`, `free disk >= 300 GB`.
- **Self-Pause on High Host Load**: If `load5 > 60.0` (strictly below `85.0` and `0.85 * 76 = 64.6`), the supervisor pauses **only** its own worker process group with `SIGSTOP` and resumes with `SIGCONT` when `load5 <= 55.0`. Never signals any other process (`H223`, `H224`, `H225`, `H227`, or live flywheel PID `2297929`).
- **Deadlines**: Prereg PR merged by `18:30Z`; single attempt compute stop `23:30Z`; closeout PR by `00:00Z`.
