# H226 — frozen GATED_025 on the retained-best r3175 base (fresh-fold confirmation), xbabe2 CPU, 0 updates

**Owner:** rd_argon [ARGON]. **Assignment:** `~/nudges/hw_t1435_argon_h226.md` (ENG:watcher, 2026-11-02 14:35Z).
**Claim:** `~/jope_plan/ns/receipts/H226_claim.txt`. **Lane:** `~/JopeDime/.agent/worktrees/h226-argon`, under the
H-wave exemption at DECISIONS line 73.

This is a diagnostic only. It cannot advance a model, a serving release or a live flywheel checkpoint. DOPE stays
`c51a56d0`, and nothing in the live loop changes. **This prereg merges before anything runs** (`prereg PR -> local
targeted tests -> merge prereg PR -> smoke -> full run`).

## Why

- [H210](../h210_blend025_gated_retry/README.md) (+0.0140 [+0.0022, +0.0355], harm 3.4%, 3/3 seeds, Elo 1505.6) and
  [H217](../h217_gated025_confirm/README.md) (+0.0216 [+0.0038, +0.0418], harm 4.4%, 3/3 seeds, Elo 1504.8) confirmed
  that `GATED_025` (the H129 `BLEND_025` recipe gated by the outcome-free `zgap` cost signal with a FIT-only threshold
  chosen under a 5% FIT harm cap) reliably improves over `r882` on 205-family pooled cross-fit.
- On DL-Val flywheel91 (91 contexts, 19 families), `GATED_025` on `r882` lifted the context mean from `1.1377942` to
  `1.139246` (`+0.00145`, family-bootstrap CI95 `[+0.00038, +0.00270]`), falling `0.00055` short of the `+0.002`
  plateau margin (`1.1397942`).
- Meanwhile, the live flywheel (`xbabe1` `flywheel-full-20261030`) has advanced to retained-best `round-3175.pt`
  (`SHA256: 1d6ea900ebed8d2a05c3b3f37576adc15aadd2b53c585bfd5cb71a103e9ff97b`, DL-Val91 context mean
  `1.1389572403748425`, `+0.001163` over `r882`).

**Hypothesis (hw_t1435).** Applying the frozen H217 `GATED_025` recipe to the retained-best `r3175` base (`G3175`, zero
adapter updates) compounds with `r3175` on 205-family pooled cross-fit and lifts the DL-Val91 context mean past both
the `+0.002` plateau margin vs `R882` (`> 1.1397942`) and `R3175` itself (`G3175 - R3175` family-bootstrap
`LCB95 > 0`), eliminating winner's-curse selection bias of `r3175` on DL-Val.

## Frozen recipe (identical to H210/H217; no retuning)

A unit test compares the function bodies of `zgap`, `disagreement`, `tau_grid`, `gated`, `choose_tau` and `blend`
byte-for-byte with H210 and H217.
- **Blend.** For base $B \in \{\text{R3175}, \text{R882}\}$, $\text{blend}_B = B + 0.25 \cdot (\text{GBM}_{\text{cal}} - B)$
  in cost space (H129). $\lambda = 0.25$ is fixed.
- **Gate.** $\text{zgap}(B, \text{blend}_B) = (B[b] - B[a]) / \operatorname{std}(B)$, where $a$ is $B$'s first argmin
  and $b$ is $\text{blend}_B$'s first argmin ($0$ when $a = b$; $+\infty$ if $\operatorname{std}(B) = 0$). The signal is
  outcome-free (costs only). The gate opens iff $\text{zgap} \le \tau$; otherwise the arm falls back to $B$.
- **$\tau$ procedure.**
  - **Grid:** $-\infty$, the FIT quantiles q10/q25/q50/q75/q90 of the statistic among switching FIT contexts, and $+\infty$.
  - **Selection:** chosen per cell on FIT folds only, maximizing the FIT equal-family native gain over the arm's own
    base ($B = \text{R3175}$ for `G3175` and `G3175_DIS`; $B = \text{R882}$ for `G882`) subject to FIT harm $\le 5\%$.
    Ties go to the smaller $\tau$.
  - **No outcome leakage:** no A, B or DL-Val outcome is used to choose calibration or $\tau$.
- **Model.** The H129 HistGBM recipe (`192` sampled programs per context, `1,089` features: `program_412`, `lm48_z`,
  `dope629`) and positive-slope linear calibration from inner out-of-fold FIT predictions. Because HistGBM features and
  targets depend only on frozen DOPE features and pool utilities (not the adapter), the 45 HistGBM fits (`3` seeds $\times$
  `15` exclusion sets) are shared identically between the `R3175`-based and `R882`-based arms.
- **Endpoint.** Flywheel protocol `fixed-dev-cost-argmin-negative-fitness/v1` (merged H139 `native`): full-pool first
  argmin, utility $-\text{fitness}$ when valid and $-2.0$ when invalid. **Exact ties are zero gain.**

## Arms (zero optimizer updates; read-only checkpoint copies)

| Arm | Base | Cost | Role |
|---|---|---|---|
| `R882` | `round-882.pt` (`0084479a…`) | frozen `r882` forward | reference (Elo 1500); DL-Val91 `1.1377941898164525` bit-exact gate |
| **`G3175`** | `round-3175.pt` (`1d6ea900…`) | `B3175` if `zgap(r3175, B3175) <= tau_cell`, else `r3175` | **primary and co-primary** |
| `R3175` | `round-3175.pt` (`1d6ea900…`) | frozen `r3175` forward | retained-best base alone; DL-Val91 `1.1389572403748425` exact gate |
| `G882` | `round-882.pt` (`0084479a…`) | `B882` if `zgap(r882, B882) <= tau_cell`, else `r882` | H217 `GATED_025` replication anchor on fresh folds |
| `G3175_DIS` | `round-3175.pt` (`1d6ea900…`) | `B3175` if `disagreement(gbm, r3175) <= tau_cell`, else `r3175` | H129 disagreement placebo on `r3175`, report-only |
| `B3175` | `round-3175.pt` (`1d6ea900…`) | ungated `r3175 + 0.25*(gbm_cal - r3175)` | component, report-only |
| `B882` | `round-882.pt` (`0084479a…`) | ungated `r882 + 0.25*(gbm_cal - r882)` | component, report-only |

## Exact regression gates first

Before any HistGBM fit, stage-1 lock or A scoring:
1. **Structural check:** `round-882.pt` (`0084479aa0703667a0f3ccfe2beda60741c204795ec57b7c08542401ec6cdb64`) and
   `round-3175.pt` (`1d6ea900ebed8d2a05c3b3f37576adc15aadd2b53c585bfd5cb71a103e9ff97b`) must match their pinned
   SHA-256 digests, record `dope_sha256 == c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814`, share
   identical non-`adapter_10tok.` tensors (`torch.equal`), and differ only in `adapter_10tok.` parameters.
2. **`R882` DL-Val91 gate:** 91/91 per-context utilities must match `best_dev_checkpoint.json` bit-exact, with context
   mean `1.1377941898164525`.
3. **`R3175` DL-Val91 gate:** 91-context native utility mean must equal `1.1389572403748425` (from
   `adapter_admissions.jsonl` round 3175, within `1e-12`).
Refuse to proceed if either check fails.

## Fresh partitions

- **Seeds and folds.** Seeds **226101–226103**, with fold seeds **226141–226143**, one distinct 5-fold family partition
  per seed. Sample seed `226140`, bootstrap seed `226150`, Elo seed `226151`, plateau seed `226152`.
- **Partitions.** Built with H124 `folds` on the 205 DL-Train families (1,219 contexts), cyclic cells $A = k$,
  $B = (k+1) \bmod 5$, $\text{FIT} = \text{other three}$. Pinned in [fold_manifest.json](fold_manifest.json).
- **Overlap (all $\le 0.20$).**
  - Within H226, pairwise mean best-match Jaccard is `0.136–0.179`.
  - Cross-wave max mean best-match Jaccard: vs `H203` `0.176`, vs `H210` `0.178`, vs `H217` `0.182`, vs `H218` `0.176`,
    vs `H222` `0.189`, vs `H223` `0.185`, vs `H224` `0.172`, vs `H225` `0.168`. The code refuses any partition with
    best-match Jaccard $> 0.20$.

## Fits, lock and scoring

- **Fits.** Per seed, 5 outer fits (exclude $\{A, B\}$) and 10 inner fits (exclude $\{A, B, j\}$): **45 HistGBM fits**.
- **Stage-1 lock.** `stage1_lock.json` is written exclusively and fsynced **before any A or DL-Val scoring**. Per cell
  it records the calibration, the chosen $\tau$ and full FIT grid for `G3175`, `G3175_DIS` and `G882`, the B-panel
  native hashes for `B3175` and `B882`, and the fit prediction hashes.
- **Stage 2.** Pooled A: each context is scored once per seed with its cell's outer model, calibration and $\tau$.

## Statistics, decision rules and diagnostics

**Primary (`G3175` vs `R882`, pooled-A native, 205 families).**
- Statistic: median over the 3 seeds of the per-seed equal-family native gain.
- **CI95:** 10,000-draw paired family-cluster bootstrap (seed `226150`), reduced over seeds on axis 0.
- Harm: fraction of families whose 3-seed mean delta is $< -0.01$.
- **CONFIRMED: G3175** iff all three hold:
  1. **LCB95 > 0**;
  2. **3/3 seeds positive**;
  3. **harm $\le 10\%$**.
  Otherwise **NOT_CONFIRMED**.

**Secondary contrasts (pooled-A native and U4 report-only).**
- `G3175` vs `R3175` (does the gate add on top of the `r3175` base?);
- `G882` vs `R882` (replication anchor on fresh folds `226141–226143`);
- `R3175` vs `R882` (does `r3175` alone improve over `r882` on 205-family DL-Train?);
- `G3175` vs `G882` (paired comparison of gated `r3175` vs gated `r882` on the same cells);
- `G3175_DIS` vs `R882` and vs `R3175` (H129 disagreement placebo on `r3175`);
- `B3175` and `B882` ungated components.

**Co-primary (`PLATEAU_WIN` on DL-Val flywheel91, 91 contexts, 19 families).**
- **PLATEAU_WIN** iff all three hold:
  1. DL-Val91 context-mean gain of `G3175` vs `R882` is **$> +0.002$**;
  2. 10,000-draw family-cluster bootstrap (seed `226152`) **LCB95 of `G3175` vs `R882` is $> 0$**;
  3. 10,000-draw family-cluster bootstrap (seed `226152`) **LCB95 of `G3175` vs `R3175` alone is $> 0$** (and gain
     $> 0$), eliminating winner's-curse selection bias of `r3175` on DL-Val.
  Exact ties are not wins. Otherwise **NO_PLATEAU_WIN**.

**Combined verdict.** `<primary> + <co-primary>`, e.g. `CONFIRMED: G3175 + PLATEAU_WIN` or
`CONFIRMED: G3175 + NO_PLATEAU_WIN`.

**Diagnostics.**
- Median + 95% CI across 3 seeds, per-seed sign, harm, and Bradley–Terry Elo (`R882 = 1500` and vs `R3175`, 400
  iterations, prior 0.01, tie margin 1e-4, 1,000 draws, seed `226151`).
- Number of DL-Val91 contexts whose top-1 pick flips (`G3175` vs `R3175`, `G3175` vs `R882`, `R3175` vs `R882`,
  `G882` vs `R882`, `G3175` vs `G882`), reported across cells (`contexts_any_cell`, `contexts_majority_cells`,
  `contexts_all_cells`, `mean_across_cells`, `per_seed_mean`, `per_cell_counts`).
- Only Elo and utility are reported: **no AUC and no speed numbers**.

## Read boundary and frozen state

- **Reads.** Real-S3-derived DL-Train packet (`1,219` contexts, `205` families; H124 pinned `selected.json`
  `981960ad…`) and the `91` DL-Val contexts (regression and co-primary), plus read-only copies of `round-882.pt` and
  `round-3175.pt` under `~/jope-data/h226/ckpt/`.
- **Never read or written.** Sealed, holdout or meta-test payloads (`0` reads, asserted in code), and
  `~/jope-data/flywheel-full-20261030` on `xbabe1` (never touched).
- **Frozen state.** DOPE stays `c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814`, verified before and
  after both `R882` and `R3175` forward passes. `0` optimizer updates.

## Host, guards, smoke and deadlines (hw_t1435)

- **Host:** `xbabe2` CPU only, `nice -n 19`, whole-tree affinity $\le 4$ cores (`os.sched_setaffinity` to 4 allowed
  CPUs, `OMP_NUM_THREADS=4`, all other thread pools `1`, worker tree cap `4.5` cores over 30 s). Never `xbabe0` or
  `xbabe3`.
- **Hard aborts:** process-tree RSS $\ge 4\text{ GiB}$, `MemAvailable < 20 GiB` (swap is full), free disk $< 300\text{ GB}$,
  worker affinity $> 4$ cores, worker tree $> 4.5$ cores over 30 s, or reaching the absolute compute deadline.
- **Load guard PAUSES (never aborts):** when `load5 > 60` (or `load1` / host CPU $> 85\%$ for 300 s), the monitor
  pauses only this worker's process group (`SIGSTOP`), rechecks every `60 s`, and resumes (`SIGCONT`) when
  `load5 <= 55`; between fits the runner also sleeps in `60 s` increments while `load5 > 60` until `load5 <= 55`.
- **Execution order:** merge prereg PR $\to$ `--smoke` (verifying host admission, source binding, exact `R882` and
  `R3175` DL-Val91 regression gates, and 1 HistGBM fit on a separate `smoke-*` root) $\to$ full 45-fit run.
- **Deadlines:** absolute compute stop **2026-11-02T18:30:00Z** (`1793644200.0`), hard closeout **19:00Z**.

## Custody

- **Prereg:** `s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261102/H226/prereg-<head>/` (create-only).
- **Run & closeout:** `s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261102/H226/<run_id>/`, with small
  `metrics.json` ($< 100\text{ KB}$), `README.md`, `custody.json`, and updated index rows in git.
