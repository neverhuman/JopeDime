# H77 — Long Direct-Utility Evolution Strategies with Family-Harm Penalty (`ES` vs `ESH`)

Status: preregistered design written and committed before any H77 runner, coordinator, or test code (`2026-10-31`, owner directive `agy_fw_audit_h77_long_es_t2335.md`). CPU preparation only in the initial PR; GPU execution is gated on PR merge and runs exclusively under `~/jope_plan/qualification.lock` via the shared FIFO ticket queue (`research/flywheel_selfplay/qualification_lock_fifo.py`, PR #832), queued behind existing `H76`/`D2`/`W1` tickets with no immediate re-take.

---

## 1. Motivation & Context (Post-H57 / H59 / H61 Closeout, PR #854)

1. **H61 (`S0`–`S3` selection ablation)**: Made `0 / 240` round-end checkpoint accepts across all four selection rules (`S0` last-step, `S1` Fold-B mean gain, `S2` Fold-B LCB95, `S3` Fold-B Elo). Selection is not the bottleneck: every gradient live-step candidate from `r882` degrades utility.
2. **H59 (target / loss stabilization)**: Produced `0 / 4` admitted arms (`CLIP` was least harmful, still negative on DL-Val).
3. **H57 (8-generation direct-utility antithetic ES, pop 32)**: The only H46–H61 method family with a positive pooled Fold-B family mean gain (`+0.0156` for `sigma=0.001`, `+0.0140` for `sigma=0.003`) and a single seed (`seed 16`) reaching `+37.7` Elo vs `r882`. However, H57 failed admission because its pooled family median gain was `0.0000`, its one-sided family-cluster `LCB95` was `-0.027`, and its family harm fraction (`gain < -0.01`) was `22%–32%` (exceeding the `<= 10%` no-harm ceiling).
4. **H77 Hypothesis**: Extending direct-utility antithetic ES at `sigma = 0.001` from `8` to `24` generations with `2x` antithetic pairs (`32` pairs, population `64`) on fresh seeds (`18, 19, 20`) reduces gradient-estimate variance, and adding an explicit a priori family-harm penalty (`ESH`) directly suppresses the lower tail (`gain_f < -0.01`) that blocked H57 admission while preserving positive mean utility gain.

---

## 2. Fixed Inputs, Integrity Pins & Scope

Reuse the exact H57 (`research/flywheel_selfplay/h57/runner.py`) / H55 real-S3 H50 packet, metadata, and `seed = 4900` five-family-fold assignment:
- **Base Checkpoint (`r882` copy only)**: Staged private copy of `/home/ubuntu/jope-data/h51/ckpt/round-882.pt` (`SHA256: 0084479aa0703667a0f3ccfe2beda60741c204795ec57b7c08542401ec6cdb64`). Never touch the live `xbabe1` (`flywheel-full-20261030`) loop.
- **Frozen DOPE Backbone**: `SHA256: c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814` (`c51a56d0`), verified before and after **every** ES generation and before/after every Fold-B or terminal DL-Val scoring pass.
- **Real S3 Source Manifest**: `/home/ubuntu/jope-data/h50/input/source_manifest.json` (`SHA256: 99a53569fcaf59f9309e69329c830ae1684adff382da3ae22e0b2645c4ebc4d4`).
- **Selected Roster & Fused Model Source**:
  - `selected.json`: `981960ad628f5029f8c54552c7b1cb1d440072e7e9dcffecdea52c6a3493a0f1`
  - `cpu_exact_fused.py`: `b9f1468a61aa4dc888c8f2931f9c759f7d4f057793a77ccb05eed6300fb4f9ab`
- **Perturbed Parameter Inventory**: Only the `29,211`-parameter `adapter_10tok` tensors are perturbed or updated.
- **Split & Leakage Firewall**:
  - **Fold A (fitness split)**: Four of five family-grouped folds of `DL-Train` from assignment seed `4900` (`164` families).
  - **Fold B (admission split)**: Untouched fifth family-grouped fold of `DL-Train` from assignment seed `4900` (`41` families). Scored **once** per final candidate (`2 arms x 3 seeds = 6` candidate reads), paired with `r882`. Never used in ES fitness, hyperparameter tuning, or generation selection.
  - **DL-Val (terminal diagnostic split)**: `91` regression contexts across `19` disjoint families. Opened **only** for admitted arms, at most **once** per seed (`3` reads per admitted arm) after Fold-B admission decisions are frozen.
  - **Forbidden Data**: Zero sealed, holdout, meta-test, or outer-fold rows may be opened (`holdout_payloads_opened == 0`, `outer_rows_delivered == 0`). Paths resolving to protected or live-loop directories are rejected before read.

---

## 3. Arms & Fixed Search Protocol

- **Perturbation Scale**: Single fixed `sigma = 0.001` for all arms.
- **Fresh Independent Seeds**: `18, 19, 20` (disjoint from H57 seeds `15, 16, 17` and H59/H61/H76 seeds `12, 13, 14`).
- **Horizon & Population**:
  - `G = 24` completed generations per cell (`3x` H57 default horizon).
  - `P_pairs = 32` antithetic pairs (`population = 64`, `2x` H57 population): for each pair $i \in \{1, \dots, 32\}$, evaluate $\theta + \sigma \epsilon_i$ and $\theta - \sigma \epsilon_i$.
  - **Optimizer**: Centered-rank fitness shaping over the `64` population fitness values in `[-0.5 + 1/128, +0.5 - 1/128]`, antithetic ES gradient
    $$g = \frac{1}{2 \cdot 64 \cdot \sigma} \sum_{i=1}^{32} \big(r_{2i-1} - r_{2i}\big) \epsilon_i,$$
    and mean-parameter update via `Adam` with learning rate `lr = 0.01` on $-\nabla_{\text{ES}}$.
  - **Final Mean Theta Only**: The candidate produced by each cell is the final mean parameter vector $\theta_{24}$ after generation `24`. No best-generation pick, early stopping, or best-perturbation replacement is permitted.

### Candidate & Control Arms (`2 arms x 3 seeds + 1 NULL + 1 TIMING`)

1. **`ES` (`sigma = 0.001`, seeds `18, 19, 20`) — Family-Balanced Fold-A Utility**:
   - For candidate perturbation $\tilde{\theta}$, let $u_f(\tilde{\theta})$ be the mean direct utility on Fold-A family $f \in F_A$ ($|F_A| = 164$, each family weighted equally `1 / 164`).
   - Fitness is:
     $$F_{\text{ES}}(\tilde{\theta}) = \frac{1}{|F_A|} \sum_{f \in F_A} u_f(\tilde{\theta}).$$
2. **`ESH` (`sigma = 0.001`, seeds `18, 19, 20`) — Family-Balanced Fold-A Utility with Harm Penalty**:
   - Let $u_f(\theta_{\text{r882}})$ be the baseline `r882` utility on Fold-A family $f \in F_A$, and let $\text{gain}_f(\tilde{\theta}) = u_f(\tilde{\theta}) - u_f(\theta_{\text{r882}})$ be the paired Fold-A family gain.
   - The per-family harm shortfall below the `-0.01` threshold is $\min(0, \text{gain}_f(\tilde{\theta}) + 0.01) \le 0$ (with nonnegative harm magnitude $h_f(\tilde{\theta}) = -\min(0, \text{gain}_f(\tilde{\theta}) + 0.01) = \max(0, -0.01 - \text{gain}_f(\tilde{\theta})) \ge 0$).
   - With $\lambda = 1.0$ fixed a priori, `ESH` penalizes families with $\text{gain}_f < -0.01$ by subtracting $\lambda \cdot \text{mean}_f(h_f)$ (equivalently adding the negative shortfall $\lambda \cdot \text{mean}_f(\min(0, \text{gain}_f + 0.01))$ so that harm strictly reduces fitness rather than canceling $u_f$):
     $$F_{\text{ESH}}(\tilde{\theta}) = \frac{1}{|F_A|} \sum_{f \in F_A} u_f(\tilde{\theta}) \;-\; \lambda \cdot \frac{1}{|F_A|} \sum_{f \in F_A} \big|\min\!\big(0,\; \text{gain}_f(\tilde{\theta}) + 0.01\big)\big| = \frac{1}{|F_A|} \sum_{f \in F_A} u_f(\tilde{\theta}) \;+\; \lambda \cdot \frac{1}{|F_A|} \sum_{f \in F_A} \min\!\big(0,\; \text{gain}_f(\tilde{\theta}) + 0.01\big), \qquad \lambda = 1.0.$$
   - Note: Since $\min(0, \text{gain}_f + 0.01) \le 0$ and is strictly negative if and only if $\text{gain}_f < -0.01$, this applies a linear penalty equal to the excess harm below the `-0.01` family-harm threshold (`F_ESH <= F_ES`, with equality iff no Fold-A family has `gain_f < -0.01`, and slope `2x` on harmed families when $\lambda = 1$). (Literal subtraction of the already-negative $\min(0, \text{gain}_f + 0.01)$ would give $u_f - (u_f - u_{f,\text{r882}} + 0.01) = u_{f,\text{r882}} - 0.01$, canceling out harmed family utility.)
3. **`NULL` (`null_s001`, seed `18`, `sigma = 0.001`, `24` generations, pop `64`) — Permuted-Fitness Drift Control**:
   - Shares the exact initial `r882` weights, perturbation stream (`seed 18`), population (`64`), `24` generations, Fold-A evaluations, and `Adam(lr=0.01)` optimizer with `ES` seed `18`, but independently permutes the `64` centered-rank fitness assignments each generation before forming the ES gradient.
   - Descriptive only; cannot be selected or admitted.
4. **`TIMING` (`timing`, `ES`, seed `18`, `sigma = 0.001`, `1` generation, pop `64`) — Pre-Outcome Timing Cell**:
   - Runs `1` full `64`-perturbation generation on `Fold A` (`seed 18`), records wall time, and discards weights/scores (excluded from outcomes, never touches Fold B or DL-Val).
   - Used by the wave coordinator to verify and lock the cumulative GPU cell-wall cap ($\le 2.5\text{ h} = 9,000\text{ s}$ target; if projected total wall for `7` cells $\times 24$ generations + Fold-B + terminal DL-Val reserve exceeds `9,000 s`, the wave stops cleanly before launching outcome cells).

### Keyed Perturbation Stream

- To guarantee identical perturbation draws between `ES`, `ESH`, and `NULL` for matched `(seed, generation)`:
  $$\text{stream\_seed}(\text{seed}, \text{generation}) = \text{seed} \cdot 1\,000\,003 + 1 \cdot 97\,409 + \text{generation} \cdot 1\,009, \qquad \text{seed} \in \{18, 19, 20\},\; \text{generation} \in \{1, \dots, 24\}.$$
- Thus for any seed $s \in \{18, 19, 20\}$ at generation $g = 1$, `ES` and `ESH` evaluate the exact same `64` perturbations $\theta_0 \pm \sigma \epsilon_i$; only their fitness aggregation (`ES` vs `ESH`) differs. For `NULL` (`seed 18`), the rank permutation RNG at generation $g$ is seeded with $\text{stream\_seed}(18, g) + 17$.

---

## 4. Fold-B Scoring, Admission Gate & Elo Reporting

1. **Single Fold-B Pass per Candidate**:
   - After all `6` candidate cells (`ES` seeds `18, 19, 20` and `ESH` seeds `18, 19, 20`) and the `NULL` cell (`seed 18`) complete, score `r882` once and each of the `6` candidate final mean-theta adapters once on the `41` untouched Fold-B families (`seed 4900`).
2. **H57 Admission Rule (Applied Independently to `ES` and `ESH`)**:
   - For each arm $A \in \{\text{ES}, \text{ESH}\}$, form the `41 x 3` matrix $G^{(A)}_{f, s} = u_f(\theta^{(A, s)}_{24}) - u_f(\theta_{\text{r882}})$ of paired Fold-B family utility gains across families $f \in F_B$ ($|F_B| = 41$) and seeds $s \in \{18, 19, 20\}$.
   - Compute the one-sided `95%` family-cluster bootstrap lower confidence bound (`LCB95`, `5th` percentile over `10,000` cluster resamples of the `41` families, retaining all `3` seeds per resampled family, deterministic RNG seed `77_4900` for `ES` and `77_4901` for `ESH`).
   - Compute each family's median gain across the `3` seeds, $m_f = \text{median}_{s \in \{18,19,20\}} G^{(A)}_{f, s}$, the pooled family median $\text{median}_{f \in F_B}(m_f)$ (with two-sided `95%` family-cluster bootstrap CI), and the family harm fraction:
     $$\text{harm}(A) = \frac{1}{|F_B|} \sum_{f \in F_B} \mathbb{I}\big(m_f < -0.01\big).$$
   - **Admission Gate**: Arm $A$ is admitted if and only if all three conditions hold strictly/non-strictly as in H57:
     1. $\text{LCB95}_{\text{mean}}(A) > 0.001$ **strictly** (exact tie `0.001` fails),
     2. $\text{median}_{f \in F_B}(m_f) \ge 0.0$,
     3. $\text{harm}(A) \le 0.10$ ($\le 10\%$ of families have across-seed median gain below `-0.01`).
   - Exact zero-gain ties (`G == 0`) fail admission. Missing or incomplete cells prevent admission. Do not select between `ES` and `ESH` by comparing Fold-B scores; evaluate each arm's admission gate independently.
3. **Family-Grouped Elo vs `r882` (Per Seed and Median)**:
   - For each candidate arm and seed $s \in \{18, 19, 20\}$, compute family-grouped Elo vs `r882` (reference rating `1500.0`) on Fold B using the H55/H57 family-grouped win/tie/loss diagnostic (`h55.elo_diagnostic`), and report:
     - Per-seed Elo rating and $\Delta\text{Elo}$ vs `r882` (plus family W/T/L counts),
     - The **median** $\Delta\text{Elo}$ (and median rating) across the `3` seeds `18, 19, 20` for each arm.
4. **Admission-Gated Terminal `DL-Val` Read**:
   - `DL-Val` (`91` contexts, `19` families) is read **only** for admitted arms, at most **once** per seed (`18, 19, 20`), after `outcomes/admission.json` is frozen. If neither `ES` nor `ESH` is admitted, `0` `DL-Val` reads occur (`validation_payloads_opened == False`).

---

## 5. Resource, Lock & Disk Controls

- **Exclusive GPU Lease via FIFO Ticket Queue**:
  - Every GPU cell (`TIMING`, `6` candidate cells, `1` `NULL` cell, `fold-b`, and any admitted `terminal-val` reads) acquires `~/jope_plan/qualification.lock` via `research/flywheel_selfplay/qualification_lock_fifo.py` (`PR #832`), using `wave_name="H77"`, `30.0 s` polling (`LOCK_POLL_SECONDS = 30.0`), dead-PID ticket pruning in `~/jope_plan/qualification.queue/`, and `>= 45.0 s` legacy-waiter backoff (`LEGACY_BACKOFF_SECONDS = 45.0`).
  - No immediate re-take: after releasing any cell lock, the coordinator yields (`after_release=True`) and enqueues a new ticket at the tail behind any waiting `H76` / `D2` / `W1` / other wave tickets.
- **Total GPU Cell-Wall Cap ($\le 2.5\text{ h} = 9,000\text{ s}$)**:
  - Hard cap: `MAX_WAVE_GPU_WALL_SECONDS = 9000.0` (`2.5 hours`) cumulative GPU cell wall across timing, candidate, null, Fold-B, and any admitted terminal DL-Val cells (lock wait time is tracked separately under a `3 h` maximum lock-wait bound).
  - The `1`-generation `TIMING` cell (`ES`, `seed 18`, `pop 64`) runs first. Projected wave wall is `measured_gen_seconds * (7 * 24 + 2)` (covering `168` total outcome generations across `7` cells plus a `2`-generation equivalent buffer for model setup, Fold-B, and terminal scoring). If the projection exceeds `9,000 s` (or the configured cap), the coordinator stops cleanly before running any outcome cell.
- **Host & Disk Safety Guards**:
  - `CPU <= 85%` before launch and during child execution; `nice >= 10`, `OMP_NUM_THREADS=4`, `MKL_NUM_THREADS=4`, `OPENBLAS_NUM_THREADS=1`, `NUMEXPR_NUM_THREADS=1`.
  - Child `RSS < 6 GiB` (`RSS_LIMIT_KIB = 6 * 1024 * 1024`), host `MemAvailable >= 16 GiB`, `cuda:0` free memory `>= 6 GiB` (`6144 MiB`).
  - Disk guard on `/` (`xbabe2`): launch requires `< 94.5%` used and `>= 1 GiB` free; writer/runtime aborts if disk used reaches `>= 95.0%` (`abort if xbabe2 disk > 95%`).
- **Execution Gate**:
  - Open PR with `PREREG.md`, `runner.py`, `run_wave.py`, `build_metrics.py`, `audit.py`, and `test_h77.py`. Hand off to `super_merger_codex`. Do **not** start the GPU waiter until the PR merges to `main`. Record milestones in `~/jope_plan/ns/receipts/H77_t2335.md`.
