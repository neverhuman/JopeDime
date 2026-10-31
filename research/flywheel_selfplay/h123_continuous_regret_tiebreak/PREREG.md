# H123 — continuous-regret tie-break of the H114 lattice

**Status: no-artifact proposal only; no outcomes or execution.** H114's
committed summary reports 21 candidates at the post-r882 maximum, but does not
identify their rounds. The owner-provided H123 unstick directive reports that
0 of the 21 `.pt` checkpoints remain. This lane did not access xbabe1 or the
raw H114 log; the raw read-only log copy was not found in this lane's custody.
No model, dataset payload, S3 object, GPU, or qualification lock was accessed.

## Hypothesis and frozen candidate rule

H114 reports 21 distinct candidate rounds tied at utility
`1.1377131278566008`, each below r882. If at least three of those candidates'
adapter artifacts are verified from the authorized read-only H114 log copy,
evaluate only that top-utility candidate set. On a new deterministic five-fold
whole-family split of real H50 DL-Train data (sort family IDs, permute with
NumPy `default_rng(12300)`, distribute round-robin to five folds), compute each
candidate's continuous mean regret on the four training folds:

`regret(context) = mean_i(max_j(true_utility[context,j]) - predicted_utility[context,i])`.

Average context regret within each family, then average family means equally.
Select the candidate with the lowest training-family mean regret; ties resolve
by lower H114 round number, then ascending artifact SHA-256. The selection is
fixed before examining the held-out fold. This metric follows H121's H114(a)
co-readout; it never replaces H114's official argmin utility or changes the
live flywheel admission rule.

Because the owner-reported inventory is 0/21 retained artifacts, retain this
proposal only and do not score. The proposal may be reconsidered only if a
future authorized inventory verifies at least three top-tied candidate
artifacts.

## Conditional evaluation plan

Only if the read-only copy establishes at least three usable top-tied
artifacts, freeze the candidate inventory and this rule before any scoring.
Use the same selected candidate and a fresh r882 copy as the paired reference;
use only real H50 DL-Train families, with the first four folds for the fixed
selection rule and the fifth fold for evaluation. Pair seeds `61`, `62`, `63` across the
selected candidate and r882. Primary is the median across seeds of paired,
equal-family mean utility delta (candidate minus r882). Report the family
cluster-bootstrap one-sided 95% LCB and two-sided 95% CI, and the fraction of
families with delta `< -0.01`. Advance only if median delta is strictly
positive, LCB is strictly greater than `+0.001`, and harm is at most 10% of
families; ties fail. Report family-grouped Elo and 95% CI versus r882=1500.
Report Elo, never AUC. Do not multiply choices by testing multiple candidates
on the held-out fold: the training-fold rule selects exactly one candidate.

DOPE remains frozen at SHA-256
`c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814`; trainable
parameters are limited to the adapter. Inputs are real S3 H50 DL-Train only.
Never read DL-Val, sealed, holdout, outer, or meta-test payloads. Keep scratch
below 2 GB and remove only H123 scratch after its evidence has a verified S3
receipt, if execution is later approved. No GPU, ticket, waiter, lock, training,
or launch is authorized by this proposal.

---

# H123 Preregistration & Prospective Top-Cluster Retention Proposal: Continuous-Regret Tie-Breaker inside the Top-Utility Cluster (Post-R882)

- **Hypothesis ID**: `H123`
- **Owner**: `rd_argon` (`[ARGON]`)
- **Date**: `2026-10-31`
- **Branch**: `rd/h123-continuous-regret-tiebreak`
- **Execution Mode**: CPU-only preregistration, feasibility census, and admission-rule + prospective checkpoint-retention proposal (`no-artifact` branch per `/home/ubuntu/nudges/rd_argon_h123_t0800.md`, `/home/ubuntu/nudges/rd_argon_h123_t0905.md`, and `/home/ubuntu/nudges/h123_unstick_t0955_rd_argon.md`)
- **Custody & Safety Invariants**:
  - Frozen DOPE SHA-256 verified before and after on CPU (`torch.device("cpu")`): `c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814` (`params_sha256`: `75084b2c4e31e9a87a6a12eb14942ca4e8c92dd7a820bd9e77fc2dc97a6e004f`, `buffers_sha256`: `ca8cc329fa495679564112e54f2e1717b98aa050a2854be2f04fbdf2c3504be6`)
  - Pinned `r882` baseline checkpoint copy: `/home/ubuntu/jope-data/h51/ckpt/round-882.pt` (`0084479aa0703667a0f3ccfe2beda60741c204795ec57b7c08542401ec6cdb64`)
  - Pinned `selected.json` manifest: `/home/ubuntu/jope-data/h50/input/selected.json` (`981960ad628f5029f8c54552c7b1cb1d440072e7e9dcffecdea52c6a3493a0f1`)
  - Zero `DL-Val` terminal reads (`0`), zero outer-test reads (`0`), zero sealed meta-test reads (`0`), zero GPU seconds (`0.0 s`), zero writes into `xbabe1:~/jope-data/flywheel-full-20261030/`

---

## 1. Falsifiable Hypothesis

`H114` demonstrated that across `1,198` post-`r882` rounds (`r883..r2080`), top-1 discrete argmin validation utility takes only `20` distinct values on `91` validation contexts (`19` families), and the post-`r882` ceiling `1.1377131278566008` (`8.106196e-5` below `r882`'s `1.1377941898164525`) is reached repeatedly (`17` times in `r883..r2080`, and `21` times through `r2140`: rounds `899, 907, 931, 935, 1003, 1039, 1043, 1091, 1119, 1171, 1183, 1215, 1223, 1247, 1255, 1283, 1291, 1639, 1690, 1745, 1964`). Because `restore_best_model_and_optimizer` rejects every round with $U_{\text{val}}(r) < U_{\text{best}}$, all `21` tied-at-max candidates are discarded as indistinguishable ties even when their full slate rankings differ beneath the argmin.

**Hypothesis (`H123`)**: Breaking ties among top-cluster candidates ($U_{\text{val}}(m) \ge U_{\text{best}} - \epsilon_{\text{cluster}}$, with $\epsilon_{\text{cluster}} = 10^{-4}$) by minimizing **continuous all-candidate family-equal mean regret** on calibration families (`folds 0..3`, `164` `DL-Train` families) selects a candidate $m^*$ that strictly improves official top-1 utility over a fresh copy of `round-882.pt` on held-out `DL-Train` scoring families (`fold 4`, `41` families) across paired seeds `61, 62, 63`.

---

## 2. Feasibility Census of Historical Tied-at-Max Checkpoints (`21` Rounds)

Before any outcome scoring, we audited both `xbabe1:~/jope-data/flywheel-full-20261030/run-20261030/` and S3 custody (`s3://veox-jopedime-learning-use2/experiments/flywheel-full-20261030/run-20261030/`) for the `21` tied-at-max rounds (`val_utility_mean = 1.1377131278566008`):

1. **`xbabe1` local filesystem (`0 / 21` retained)**:
   - `xbabe1:~/jope-data/flywheel-full-20261030/run-20261030/rejected/` retains only a single rolling rejected checkpoint (`round-2140.pt` at census time). Zero (`0/21`) of the tied-at-max `.pt` adapter checkpoints remain on `xbabe1`.
2. **S3 custody (`s3://veox-jopedime-learning-use2/experiments/flywheel-full-20261030/run-20261030/`)**:
   - `9 / 21` rounds (`899, 907, 931, 935, 1003, 1039, 1043, 1091, 1119`) exist as cold S3 objects under `rejected/round-<N>.pt` (`7,503,218` bytes each; uploader stopped at `round-1139.pt`).
   - `4 / 21` rounds (`1639, 1690, 1745, 1964`) exist as compressed archive bundles under `retention/retention-<N>-*.tar.zst`.
   - `8 / 21` rounds (`1171, 1183, 1215, 1223, 1247, 1255, 1283, 1291`) fall in the custody gap between the `rejected/round-<N>.pt` uploader cutoff (`r1139`) and the `retention/` bundle cutover and are **permanently unretained everywhere**.

Per owner directives (`/home/ubuntu/nudges/rd_argon_h123_t0905.md` and `/home/ubuntu/nudges/h123_unstick_t0955_rd_argon.md`), since `0 / 21` tied-at-max `.pt` checkpoints are retained on `xbabe1` (`< 3` local artifacts), `H123` executes the **no-artifact branch**: preregistering the exact split and scoring protocol, recording the full 21-round custody census in `/home/ubuntu/jope_plan/ns/receipts/H123_feasibility_census.json` and S3 (`s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/H123/h123_prereg_census_20261031T1607Z/H123_feasibility_census.json`, `VersionId = unoe4daW1iSezdnzQj1sbLIzYDmfbbAc`), and specifying the prospective top-cluster retention + admission-rule design.

---

## 3. Preregistered Family-Grouped `DL-Train` Split & Evaluation Protocol

### 3.1 Fresh Family-Grouped 5-Fold Partition (`fold_seed = 12300`)
- **Cohort**: `1,219` `DL-Train` contexts across `205` families from `/home/ubuntu/jope-data/h50/input/selected.json` (`981960ad628f5029f8c54552c7b1cb1d440072e7e9dcffecdea52c6a3493a0f1`).
- **Rule**: Sort the `205` unique `family_id` strings lexicographically, permute with `np.random.default_rng(12300).permutation(205)`, and assign `fold_k = sorted(families[perm[k::5]])` for $k \in \{0, 1, 2, 3, 4\}$ (`41` families per fold, `0` cross-fold family overlap).
- **Fold Manifest SHA-256**: `877dcb227b34dfc2306d597824f04d90368473df765683031f6e6e128731df53`
- **Fold Breakdown**:
  - **Calibration Folds (`folds 0..3`)**: `164` families, `986` contexts (`fold 0`: `263` contexts / `41` families; `fold 1`: `228` contexts / `41` families; `fold 2`: `245` contexts / `41` families; `fold 3`: `250` contexts / `41` families). Used **exclusively** to rank top-cluster candidates by continuous all-candidate mean regret and pick a single candidate $m^*$ before any scoring-fold evaluation.
  - **Held-Out Scoring Fold (`fold 4`)**: `41` families, `233` contexts. Untouched during candidate selection; used **once** to compare the single calibration-picked candidate $m^*$ against `round-882.pt` across paired seeds `61, 62, 63`.

### 3.2 Continuous All-Candidate Mean Regret Pick Rule (Calibration Folds `0..3` Only)
For each candidate adapter $m$ in the top utility cluster ($U_{\text{val}}(m) \ge U_{\text{best}} - \epsilon_{\text{cluster}}$, $\epsilon_{\text{cluster}} = 10^{-4}$) and each context $c$ in calibration folds `0..3` with unrevealed slate $I_c$ (paired across seeds $s \in \{61, 62, 63\}$):
1. Compute the model's predicted cost $\hat{\ell}_{m,c,i}$ on every unrevealed candidate $i \in I_c$ and convert to utility orientation $\hat{s}_{m,c,i} = -\hat{\ell}_{m,c,i}$.
2. Define the softmax-weighted expected utility over $I_c$ at temperature $\tau = 0.10$ (standardizing $\hat{s}_{m,c,\cdot}$ to unit std within $I_c$ so scale shifts cannot game $\tau$):
   $$w_{m,c,i} = \frac{\exp(\tilde{s}_{m,c,i} / \tau)}{\sum_{j \in I_c} \exp(\tilde{s}_{m,c,j} / \tau)}, \qquad r_{m,c} = \max_{j \in I_c} u_{c,j} - \sum_{i \in I_c} w_{m,c,i}\, u_{c,i}$$
   alongside the rank-weighted continuous regret across all candidates in $I_c$.
3. Aggregate with equal weight per family across the `164` calibration families:
   $$R_{\text{cal}}(m) = \frac{1}{|F_{\text{cal}}|} \sum_{f \in F_{\text{cal}}} \frac{1}{|C_f|} \sum_{c \in C_f} \bar{r}_{m,c}$$
4. **Fixed Pre-Scoring Pick**: Select the single candidate
   $$m^* = \arg\min_{m \in \mathcal{C}_{\text{top}}} R_{\text{cal}}(m)$$
   with deterministic tie-break by lowest round number. **Only $m^*$ and `round-882.pt` may be evaluated on `fold 4`.**

### 3.3 Primary Endpoint & Strict Admission Gate (Scoring Fold `4`, `41` Families, Seeds `61, 62, 63`)
On held-out `fold 4` (`41` families, `233` contexts), evaluate $m^*$ and a fresh copy of `round-882.pt` on official unrevealed argmin utility across paired seeds `61, 62, 63`. Let $\Delta_{f,s} = U_{f,s}(m^*) - U_{f,s}(\text{r882})$ and $\Delta_f = \frac{1}{3}\sum_{s \in \{61,62,63\}} \Delta_{f,s}$.

Admission requires **all three** conditions simultaneously:
1. **Positive Seed Median**: $\text{median}_{s \in \{61,62,63\}}\!\left(\frac{1}{41}\sum_{f \in F_4} \Delta_{f,s}\right) > 0$.
2. **Strict Family-Cluster Bootstrap Margin**: One-sided `10,000`-draw family-cluster bootstrap (`bootstrap_seed = 12300`) lower 95% confidence bound on the `41`-family mean delta satisfies:
   $$\text{LCB}_{95}(\bar{\Delta}_{F_4}) > +0.001 \quad \text{(strict; exact ties fail)}.$$
3. **Harm-Tail Bound**: Fraction of scoring families with $\Delta_f < -0.01$ satisfies:
   $$\frac{1}{41}\sum_{f \in F_4} \mathbb{I}(\Delta_f < -0.01) \le 0.10 \quad (\le 4 \text{ of } 41 \text{ families}).$$
4. **Co-Readout**: Family-grouped Bradley-Terry Elo + `95%` family-bootstrap CI (`10,000` draws, `r882` anchored at `1500.0`). Never report AUC.

---

## 4. Proposal: Admission-Rule Change & Prospective Top-Cluster Checkpoint Retention

The H123 census exposes two coupled failure modes in the live flywheel loop (`run_fused_closed_loop.py`):
1. **Discrete Step-Function Blindness**: Official top-1 argmin utility over `91` validation contexts (`19` families) is a piecewise-constant step function where `1,198` post-`r882` rounds take only `20` distinct values and hit the `1.1377131278566008` ceiling `21` times (`8.106e-5` below `r882`) without any tie-break signal.
2. **Ephemeral Top-Cluster Checkpoint Loss**: Because `restore_best_model_and_optimizer` immediately reverts rejected rounds and `xbabe1`'s local `rejected/` directory keeps only 1 rolling file while S3 switched mid-run from `rejected/round-<N>.pt` (`<= r1139`) to `retention/retention-<N>-*.tar.zst`, `21/21` tied-at-max checkpoints were deleted from `xbabe1` and `8/21` (`r1171..r1291`) were lost permanently across all storage tiers.

### 4.1 Prospective Top-Cluster Retention Contract (`top_cluster/` Ring Buffer + Create-Only S3 Custody)
Before `restore_best_model_and_optimizer` reverts a candidate at round $r$:
1. **Top-Cluster Predicate**: Define a candidate round $r$ as **top-cluster** whenever its validation utility satisfies:
   $$U_{\text{val}}(r) \ge U_{\text{best}} - \epsilon_{\text{cluster}}, \qquad \epsilon_{\text{cluster}} = 10^{-4}$$
   (which captures exact ties with $U_{\text{best}}$ as well as the `1.1377131278566008` second lattice level `8.106e-5` below `r882`).
2. **Compact Metrics & Continuous Regret Sidecar (`< 4 KB`)**: For every round $r$, log `val_continuous_regret_mean` and `val_expected_utility_tau010` alongside `val_utility_mean` in `adapter_admissions.jsonl`.
3. **Bounded Local Top-Cluster Ring Buffer (`<= 32` Checkpoints, `<= 240 MB`)**:
   - Whenever $U_{\text{val}}(r) \ge U_{\text{best}} - \epsilon_{\text{cluster}}$, atomically save ONLY the `29,211`-parameter `h40b` adapter state dict (`~120 KB` uncompressed, or `7.5 MB` full checkpoint) to `<run_dir>/top_cluster/round-<r>.pt` plus `<run_dir>/top_cluster/round-<r>.meta.json` (`sha256`, `val_utility_mean`, `val_continuous_regret_mean`, `dope_sha256`).
   - Maintain at most `K = 32` checkpoints locally in `<run_dir>/top_cluster/`, evicting only the highest-regret (`worst R_val`) checkpoint when full, and **never** evicting a checkpoint until its create-only S3 upload (`--if-none-match "*"`) to `s3://veox-jopedime-learning-use2/experiments/<campaign>/<run_id>/top_cluster/round-<r>.pt` has returned a verified `VersionId` and stream-back SHA-256 match.

### 4.2 Two-Stage Admission Rule (Official Utility Non-Degradation + Continuous-Regret Tie-Break + Strict Family-Bootstrap Margin)
Replace the single-scalar `val_utility_mean > best_val_utility` check in the admission controller with a two-stage gate:
1. **Stage 1 (Official Top-1 Utility Floor + Continuous-Regret Tie-Break on Calibration Split)**:
   - Require $U_{\text{cal}}(r) \ge U_{\text{cal}}(\text{incumbent}) - \epsilon_{\text{tie}}$ ($\epsilon_{\text{tie}} = 10^{-4}$, i.e., within at most one single-context tie flip on the lattice) **AND** strict improvement in continuous family-equal mean regret:
     $$R_{\text{cal}}(r) < R_{\text{cal}}(\text{incumbent}) - \delta_{\text{regret}}, \qquad \delta_{\text{regret}} = 5 \times 10^{-4}.$$
2. **Stage 2 (Held-Out Family-Cluster Bootstrap Confirmation before Incumbent Replacement)**:
   - Any candidate passing Stage 1 is evaluated against the incumbent on the held-out `DL-Train` family scoring fold (`41` families, paired seeds) and promoted to incumbent **if and only if** `median_seed_delta > 0`, `LCB95 > +0.001` (`10,000` family-cluster bootstrap draws), and `harmed_family_frac <= 0.10`.
