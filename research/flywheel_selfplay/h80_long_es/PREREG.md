# H80 — Long Direct-Utility Evolution Strategies Successor (`ES` vs `ESH`, 16 Generations, Pop 32)

Status: preregistered design written and committed before any H80 runner, coordinator, or test code (`2026-10-31`, owner directives `agy_fw_audit_h77_capstop_h79_t0005.md` and `agy_fw_audit_h80_renumber_fifo_t0030.md`; pre-outcome wave label renumbered from `H79` to `H80` because `H79` was assigned to `h79_w4_prior_2plus2_cached` in PR #885). CPU preparation only in the initial PR; GPU execution is gated on PR merge and runs exclusively under `~/jope_plan/qualification.lock` via the shared FIFO ticket queue (`research/flywheel_selfplay/qualification_lock_fifo.py`, PR #832), releasing `~/jope_plan/qualification.lock` after every cell and re-queuing at the tail of the FIFO (`after_release=True`).

---

## 1. Motivation, H77 Pre-Outcome Cap Stop & A Priori Horizon/Population Trade-Off

1. **Why H77 Stopped Pre-Outcome**:
   - In [`research/flywheel_selfplay/h77_long_es/`](../h77_long_es/), the preregistered `TIMING` cell (`timing/ES/seed18`, `1` generation, `32` antithetic pairs / `population = 64` on `164` Fold-A families / `1,001` contexts) measured `120.445342 s` generation wall (`1.8820 s` per perturbation) and `170.716799 s` total coordinator cell wall (`~50.27 s` model/data/staging setup overhead).
   - At `population = 64` and `24` generations across `7` cells (`3 ES + 3 ESH + 1 NULL`), projected GPU cell wall was `20,475.71 s > 9,000.0 s` (`2.5 h` cap). The H77 preregistration guard stopped cleanly pre-outcome (`0` outcome cells started, `fold_b_scored = false`, `0` DL-Val reads), and H77 was closed out without post-hoc amendment.
2. **A Priori Calibration of `pop` and `generations` from the H77 Timing Cell Only**:
   - Every ES perturbation evaluates the `29,211`-parameter `adapter_10tok` across the exact `1,001` Fold-A contexts (`1.881958 s` per perturbation measured in `H77` `timing/ES/seed18`).
   - At `P_pairs = 16` antithetic pairs (`population = 32`, matching H57's population size), one generation evaluates `32` perturbations and takes:
     $$t_{\text{gen}}(\text{pop}=32) = 0.5 \times 120.445342\text{ s} = 60.222671\text{ s}$$
     (corroborated by H57's measured `~61.5 s/gen` at `population = 32`).
   - Adding the measured H77 per-cell coordinator setup/staging/save overhead (`50.27 s` per cell), a cell of $G$ generations at `population = 32` requires `60.222671 * G + 50.27 s`.
   - **Trade-Off Decision (More Generations over Larger Population)**:
     - Within the `9,000.0 s` cumulative GPU cell-wall cap across `7` outcome cells (`3 ES + 3 ESH + 1 NULL`) plus Fold-B scoring (`~120 s`) and admitted terminal DL-Val reserve (`~60 s`), we explicitly prioritize **more optimization generations (`G = 16`, `2x` H57's `8`-generation horizon)** over doubling the per-generation population (`pop = 32` / `16` antithetic pairs rather than `pop = 64`):
       - **Why `G = 16, pop = 32` over `G = 8, pop = 64`**: Both require `512` Fold-A perturbation evaluations per cell (`~963.6 s` generation wall + `50.27 s` setup = `1,013.8 s/cell`), but `Adam(lr = 0.01)` with centered-rank shaping bounds step magnitudes per generation; H57 (`pop = 32`, `8` generations) already achieved positive mean utility gain (`+0.0156`) and `+37.7` Elo on `seed 16` but was step-budget limited in suppressing tail family harm (`ESH`). Doubling the sequential `Adam` update horizon from `8` to `16` generations allows the `ESH` harm-penalty gradient to accumulate twice as many corrective updates on harmed Fold-A families while keeping per-generation rank estimation at H57's proven `32`-candidate resolution.
     - **A Priori Wall Budget Projection (from H77 Timing Cell)**:
       - Per-cell GPU wall (`16` generations, `pop = 32`): `16 * 60.222671 s + 50.27 s = 1,013.83 s`
       - `7` outcome cells (`ES` seeds `21, 22, 23`; `ESH` seeds `21, 22, 23`; `NULL` seed `21`): `7 * 1,013.83 s = 7,096.8 s`
       - Single `TIMING` verification cell (`1` generation, `pop = 32`, `seed 21`): `60.22 s + 50.27 s = 110.5 s`
       - Fold-B scoring (`1` pass over `41` families for `r882` + `6` candidates + `1` null) + admitted terminal DL-Val reserve (`2` generation-equivalents = `120.45 s` in the `project_wave_wall_from_timing` formula `measured_gen_seconds * (7 * 16 + 2)`): `~120.45 s`–`180.0 s`
       - **Total Projected Wave GPU Wall**: `114 * 60.222671 s = 6,865.4 s` generation-equivalent (`~7,387.3 s` including per-cell setup) `<= 9,000.0 s` cap (leaving `> 1,610 s` safety margin for host jitter and disk-guard checks).

---

## 2. Fixed Inputs, Integrity Pins & In-Sample vs Out-of-Sample Scope Note

Reuse the exact H57 (`research/flywheel_selfplay/h57/runner.py`) / H77 real-S3 H50 packet, metadata, and `seed = 4900` five-family-fold assignment:
- **Base Checkpoint (`r882` copy only)**: Staged private copy of `/home/ubuntu/jope-data/h51/ckpt/round-882.pt` (`SHA256: 0084479aa0703667a0f3ccfe2beda60741c204795ec57b7c08542401ec6cdb64`). Never touch the live `xbabe1` (`flywheel-full-20261030`) loop.
- **Frozen DOPE Backbone**: `SHA256: c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814` (`c51a56d0`), verified before and after **every** ES generation and before/after every Fold-B or terminal DL-Val scoring pass.
- **Real S3 Source Manifest**: `/home/ubuntu/jope-data/h50/input/source_manifest.json` (`SHA256: 99a53569fcaf59f9309e69329c830ae1684adff382da3ae22e0b2645c4ebc4d4`).
- **Selected Roster & Fused Model Source**:
  - `selected.json`: `981960ad628f5029f8c54552c7b1cb1d440072e7e9dcffecdea52c6a3493a0f1`
  - `cpu_exact_fused.py`: `b9f1468a61aa4dc888c8f2931f9c759f7d4f057793a77ccb05eed6300fb4f9ab`
- **Perturbed Parameter Inventory**: Only the `29,211`-parameter `adapter_10tok` tensors are perturbed or updated.
- **Split & Leakage Firewall**:
  - **Fold A (fitness split)**: Four of five family-grouped folds of `DL-Train` from assignment seed `4900` (`164` families, `1,001` contexts).
  - **Fold B (admission split)**: Untouched fifth family-grouped fold of `DL-Train` from assignment seed `4900` (`41` families, `218` contexts). Scored **once** per final candidate (`2 arms x 3 seeds = 6` candidate reads plus descriptive `NULL`), paired with `r882`. Never used in ES fitness, hyperparameter tuning, or generation selection.
  - **DL-Val (terminal diagnostic split)**: `91` regression contexts across `19` disjoint families. Opened **only** for admitted arms, at most **once** per seed (`3` reads per admitted arm) after Fold-B admission decisions are frozen.
  - **Forbidden Data**: Zero sealed, holdout, meta-test, or outer-fold rows may be opened (`holdout_payloads_opened == 0`, `outer_rows_delivered == 0`). Paths resolving to protected or live-loop directories are rejected before read.
- **Red-Team Note on `r882` Training Scope**:
  > r882 was trained on all 1,219 train contexts incl. fold B, so fold-B paired deltas are in-sample vs r882; only DL-Val (once, admitted arms only) is out-of-sample.

---

## 3. Arms & Fixed Search Protocol

- **Perturbation Scale**: Single fixed `sigma = 0.001` for all arms.
- **Fresh Independent Seeds**: `21, 22, 23` (strictly disjoint from H77 timing seed `18` and preregistered seeds `18, 19, 20`, H57 seeds `15, 16, 17`, and H59/H61/H76 seeds `12, 13, 14`).
- **Horizon & Population**:
  - `G = 16` completed generations per cell (`2x` H57 default horizon).
  - `P_pairs = 16` antithetic pairs (`population = 32`): for each pair $i \in \{1, \dots, 16\}$, evaluate $\theta + \sigma \epsilon_i$ and $\theta - \sigma \epsilon_i$.
  - **Optimizer**: Centered-rank fitness shaping over the `32` population fitness values in `[-0.5 + 1/64, +0.5 - 1/64]`, antithetic ES gradient
    $$g = \frac{1}{2 \cdot 32 \cdot \sigma} \sum_{i=1}^{16} \big(r_{2i-1} - r_{2i}\big) \epsilon_i,$$
    and mean-parameter update via `Adam` with learning rate `lr = 0.01` on $-\nabla_{\text{ES}}$.
  - **Final Mean Theta Only**: The candidate produced by each cell is the final mean parameter vector $\theta_{16}$ after generation `16`. No best-generation pick, early stopping, or best-perturbation replacement is permitted.

### Candidate & Control Arms (`2 arms x 3 seeds + 1 NULL + 1 TIMING`)

1. **`ES` (`sigma = 0.001`, seeds `21, 22, 23`) — Family-Balanced Fold-A Utility**:
   - For candidate perturbation $\tilde{\theta}$, let $u_f(\tilde{\theta})$ be the mean direct utility on Fold-A family $f \in F_A$ ($|F_A| = 164$, each family weighted equally `1 / 164`).
   - Fitness is:
     $$F_{\text{ES}}(\tilde{\theta}) = \frac{1}{|F_A|} \sum_{f \in F_A} u_f(\tilde{\theta}).$$
2. **`ESH` (`sigma = 0.001`, seeds `21, 22, 23`) — Family-Balanced Fold-A Utility with Harm Penalty**:
   - Let $u_f(\theta_{\text{r882}})$ be the baseline `r882` utility on Fold-A family $f \in F_A$, and let $\text{gain}_f(\tilde{\theta}) = u_f(\tilde{\theta}) - u_f(\theta_{\text{r882}})$ be the paired Fold-A family gain.
   - The per-family harm shortfall below the `-0.01` threshold is $\min(0, \text{gain}_f(\tilde{\theta}) + 0.01) \le 0$ (with nonnegative harm magnitude $h_f(\tilde{\theta}) = -\min(0, \text{gain}_f(\tilde{\theta}) + 0.01) = \max(0, -0.01 - \text{gain}_f(\tilde{\theta})) \ge 0$).
   - With $\lambda = 1.0$ fixed a priori, `ESH` penalizes families with $\text{gain}_f < -0.01$ by subtracting $\lambda \cdot \text{mean}_f(h_f)$ (equivalently adding the negative shortfall $\lambda \cdot \text{mean}_f(\min(0, \text{gain}_f + 0.01))$ so that harm strictly reduces fitness):
     $$F_{\text{ESH}}(\tilde{\theta}) = \frac{1}{|F_A|} \sum_{f \in F_A} u_f(\tilde{\theta}) \;-\; \lambda \cdot \frac{1}{|F_A|} \sum_{f \in F_A} \big|\min\!\big(0,\; \text{gain}_f(\tilde{\theta}) + 0.01\big)\big| = \frac{1}{|F_A|} \sum_{f \in F_A} u_f(\tilde{\theta}) \;+\; \lambda \cdot \frac{1}{|F_A|} \sum_{f \in F_A} \min\!\big(0,\; \text{gain}_f(\tilde{\theta}) + 0.01\big), \qquad \lambda = 1.0.$$
3. **`NULL` (`null_s001`, seed `21`, `sigma = 0.001`, `16` generations, pop `32`) — Permuted-Fitness Drift Control**:
   - Shares the exact initial `r882` weights, perturbation stream (`seed 21`), population (`32`), `16` generations, Fold-A evaluations, and `Adam(lr=0.01)` optimizer with `ES` seed `21`, but independently permutes the `32` centered-rank fitness assignments each generation before forming the ES gradient.
   - Descriptive only; cannot be selected or admitted.
4. **`TIMING` (`timing`, `ES`, seed `21`, `sigma = 0.001`, `1` generation, pop `32`) — Pre-Outcome Timing Verification Cell**:
   - Runs `1` full `32`-perturbation generation on `Fold A` (`seed 21`), records wall time, and discards weights/scores (excluded from outcomes, never touches Fold B or DL-Val).
   - Verifies that `measured_gen_seconds * (7 * 16 + 2) <= 9,000.0 s` before launching outcome cells.

### Keyed Perturbation Stream

- To guarantee identical perturbation draws between `ES`, `ESH`, and `NULL` for matched `(seed, generation)`:
  $$\text{stream\_seed}(\text{seed}, \text{generation}) = \text{seed} \cdot 1\,000\,003 + 1 \cdot 97\,409 + \text{generation} \cdot 1\,009, \qquad \text{seed} \in \{21, 22, 23\},\; \text{generation} \in \{1, \dots, 16\}.$$
- For `NULL` (`seed 21`), the rank permutation RNG at generation $g$ is seeded with $\text{stream\_seed}(21, g) + 17$.

---

## 4. Fold-B Scoring, Admission Gate & Elo Reporting

1. **Single Fold-B Pass per Candidate**:
   - After all `6` candidate cells (`ES` seeds `21, 22, 23` and `ESH` seeds `21, 22, 23`) and the `NULL` cell (`seed 21`) complete, score `r882` once and each candidate final mean-theta adapter once on the `41` untouched Fold-B families (`seed 4900`).
2. **H57 Admission Rule (Applied Independently to `ES` and `ESH`)**:
   - For each arm $A \in \{\text{ES}, \text{ESH}\}$, form the `41 x 3` matrix $G^{(A)}_{f, s} = u_f(\theta^{(A, s)}_{16}) - u_f(\theta_{\text{r882}})$ of paired Fold-B family utility gains across families $f \in F_B$ ($|F_B| = 41$) and seeds $s \in \{21, 22, 23\}$.
   - Compute the one-sided `95%` family-cluster bootstrap lower confidence bound (`LCB95`, `5th` percentile over `10,000` cluster resamples of the `41` families, retaining all `3` seeds per resampled family, deterministic RNG seed **`80_4900` for `ES` and `80_4901` for `ESH`**, updated pre-outcome with the `H79 -> H80` label renumbering).
   - Compute each family's median gain across the `3` seeds, $m_f = \text{median}_{s \in \{21,22,23\}} G^{(A)}_{f, s}$, the pooled family median $\text{median}_{f \in F_B}(m_f)$ (with two-sided `95%` family-cluster bootstrap CI using deterministic RNG seed `80_5900` for `ES` and `80_5901` for `ESH`), and the family harm fraction:
     $$\text{harm}(A) = \frac{1}{|F_B|} \sum_{f \in F_B} \mathbb{I}\big(m_f < -0.01\big).$$
   - **Admission Gate**: Arm $A$ is admitted if and only if all three conditions hold strictly/non-strictly as in H57:
     1. $\text{LCB95}_{\text{mean}}(A) > 0.001$ **strictly** (exact tie `0.001` fails),
     2. $\text{median}_{f \in F_B}(m_f) \ge 0.0$,
     3. $\text{harm}(A) \le 0.10$ ($\le 10\%$ of families have across-seed median gain below `-0.01`).
   - Exact zero-gain ties (`G == 0`) fail admission. Missing or incomplete cells prevent admission. Do not select between `ES` and `ESH` by comparing Fold-B scores; evaluate each arm's admission gate independently.
3. **Family-Grouped Elo vs `r882` (Per Seed and Median, Not AUC)**:
   - For each candidate arm and seed $s \in \{21, 22, 23\}$, compute family-grouped Elo vs `r882` (reference rating `1500.0`) on Fold B using `h55.elo_diagnostic`, and report:
     - Per-seed Elo rating and $\Delta\text{Elo}$ vs `r882` (plus family W/T/L counts),
     - The **median** $\Delta\text{Elo}$ (and median rating) across the `3` seeds `21, 22, 23` for each arm.
4. **Admission-Gated Terminal `DL-Val` Read**:
   - `DL-Val` (`91` contexts, `19` families) is read **only** for admitted arms, at most **once** per seed (`21, 22, 23`), after `outcomes/admission.json` is frozen. If neither `ES` nor `ESH` is admitted, `0` `DL-Val` reads occur (`validation_payloads_opened == False`).

---

## 5. Resource, Lock & Disk Controls

- **Exclusive GPU Lease via FIFO Ticket Queue (`wave_name="H80"`, Release After Every Cell)**:
  - Run root: `/home/ubuntu/jope-data/h80/h80-wave`.
  - Every GPU cell (`TIMING`, `6` candidate cells, `1` `NULL` cell, `fold-b`, and any admitted `terminal-val` reads) acquires `~/jope_plan/qualification.lock` via `research/flywheel_selfplay/qualification_lock_fifo.py` (`PR #832`), using `wave_name="H80"`, `30.0 s` polling, dead-PID ticket pruning in `~/jope_plan/qualification.queue/`, and `>= 45.0 s` legacy-waiter backoff.
  - **Strict Lock Fairness**: Release `~/jope_plan/qualification.lock` after **EVERY** cell and re-queue at the **TAIL** of the FIFO (`after_release=True`). Never hold the lock across cells and never re-take immediately. Signed `W1` (`rd_argon`) goes ahead of `H80` if ticketed first; an `H82` SAM wave (`sol_v1_hwv`) may interleave with short `~70 s` cells as expected.
- **Total GPU Cell-Wall Cap ($\le 2.5\text{ h} = 9,000\text{ s}$)**:
  - Hard cap: `MAX_WAVE_GPU_WALL_SECONDS = 9000.0` (`2.5 hours`) cumulative GPU cell wall across timing, candidate, null, Fold-B, and any admitted terminal DL-Val cells, **including any failed/retry/void GPU cell wall**.
- **Host & Disk Safety Guards**:
  - `CPU <= 85%` (`<= 85 / 102` cores) before launch and during child execution with backoff; `nice >= 10`, `OMP_NUM_THREADS=4`, `MKL_NUM_THREADS=4`, `OPENBLAS_NUM_THREADS=1`, `NUMEXPR_NUM_THREADS=1`.
  - Child `RSS < 6 GiB`, host `MemAvailable >= 16 GiB`, `cuda:0` free memory `>= 6 GiB` (`6144 MiB`).
  - Disk guard on `/` (`xbabe2`): launch requires `< 94.5%` used and `>= 1 GiB` free; runtime aborts if disk used reaches `>= 95.0%`.
- **Execution Gate**:
  - Log milestones in `/home/ubuntu/jope_plan/ns/receipts/H80_t0030.md` and append a one-line receipt in `~/jope_plan/ns/comms.md` when the H80 waiter starts.
