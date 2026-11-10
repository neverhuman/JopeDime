# H304 Preregistration — `h304_lcb_margin_promotion_replay` (`rd_argon`)

- **Wave ID**: `H304` (`h304_lcb_margin_promotion_replay`)
- **Owner**: `rd_argon`
- **Mode**: **Analysis-Only (`xbabe2` CPU only, `nice 19`, `taskset -c 0-3`, `RSS <= 4 GiB`, `0` neural forwards/updates, `0` GPU use, `0` fresh-cohort reads)**
- **Pinned Inputs**:
  - `H289` `prepared-closure.tar.gz`: `s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/H289/astra-t0005/inputs/prepared-closure.tar.gz` (`VersionId = dj7fcZrNhjkNwY2s5DkfOah1nfAUObnL`, `sha256 = 66bc3154f4f2ec49bc9237163712826ec0bda82f32e3ba34c4881ba73b4bd071`)
  - `prepared/history.npz`: `sha256 = e6214cd5f3f3322a6ef22bfa7b3f7a797e106869c9195cdfe18d217b6ac7f446` (`1,500` rounds `3175..4675` excluding unevaluated round `4331`, `91` `dev` contexts across `19` families, per-context per-round utilities and selected-program hashes present)
  - `H289` `recorded-snapshot.tar.gz` (`adapter_admissions.jsonl`): `s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/H289/astra-t0005/inputs/recorded-snapshot.tar.gz` (`VersionId = PyUdeyhv8LqMyLfHB1ondkNF7dITELec`, `adapter_admissions.jsonl` `sha256 = 7c01de3b03eeb993d67c84266649a732365afdeaa86c593295b73a9d05b0b868`)
  - Fixed `evaluation_identity`: `e79a7715c1531ccb8c4055ab96971101cd4034ec3a71d83bca6d25b0e604ca83`
  - Frozen `DOPE` (`r3175`) `sha256`: `c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814` (untouched)

---

## 1. Hypothesis & Motivation

`H301` (`ed8956e2c1`) established **`SELECTION_NOISE_PLATEAU`**: `r3175` (`dev91` mean `1.1389572403748425`) is a winner's-curse maximum over `~1,500–4,000` noisy 2-update proposals (`post-r3175` `N=1,499` mean `1.1320995299`, SD `0.002607`, $q_{0.95} = 1.1354387884$, expected Gaussian order-statistic max `1.140464 > 1.138957`), while the flywheel promotion rule (`accept iff dev91 mean > best`) admits tie/noise wins on a 91-context, 19-family validation set.

`H304` tests whether a stricter **selection rule** (`R1_MARGIN` `m in {0.001, 0.0035}`, `R2_LCB`, or `R3_MEDIAN_OF_K`) replayed on a **SELECT** half of the `19` `dev91` families produces promotions that generalize to the unseen **EVAL** half of families, using only the read-only recorded `H289` `history.npz` (`VersionId = dj7fcZrNhjkNwY2s5DkfOah1nfAUObnL`) and `adapter_admissions.jsonl`.

---

## 2. Step 1 Regression Verification (Completed Before Preregistration)

Verified in [`regression_receipt.json`](regression_receipt.json) prior to computing any `EVAL`-half metric:
- `r3175` `dev91` mean utility: `1.1389572403748425` (bit-exact).
- Post-`r3175` candidate count (`rounds 3176..4675`, excluding unevaluated round `4331`): `N = 1,499` (exact).
- Post-`r3175` candidate mean utility: `1.1320995298853955` (matches `H301` `1.1320995299` bit-for-bit).
- Post-`r3175` candidate `95th` percentile ($q_{0.95}$): `1.1354387884248622` (matches `H301` `1.1354387884` bit-for-bit).
- Journal cross-verification: all `1,500` rounds in `history.npz` match `adapter_admissions.jsonl` `candidate_utility` to `<= 1e-15`.

---

## 3. Family-Split Cross-Selection & Replay Protocol

### 3.1 Pre-Declared Split Seeds (`30401 / 30402 / 30403`) & `200`-Split Sensitivity (`304150`)
For each pre-declared split seed `seed in {30401, 30402, 30403}` and each of the `19` sorted coarse `dev91` families `family`:
```python
digest = hashlib.sha256(
    b"H304-split-v1\x00" + str(seed).encode("utf-8") + b"\x00" + family.encode("utf-8")
).digest()
parity = digest[-1] % 2  # 0 -> SELECT half, 1 -> EVAL half
```
This deterministic hash parity yields:
- **Seed `30401`**: `10` `SELECT` families (`46` contexts) / `9` `EVAL` families (`45` contexts)
- **Seed `30402`**: `12` `SELECT` families (`56` contexts) / `7` `EVAL` families (`35` contexts)
- **Seed `30403`**: `6` `SELECT` families (`26` contexts) / `13` `EVAL` families (`65` contexts)

In addition to the 3 primary split seeds, we report a **`200`-split sensitivity** (`RNG = 304150`) using `200` unique random balanced splits (`9` `SELECT` families / `10` `EVAL` families drawn without replacement via `np.random.default_rng(304150)`).

### 3.2 Replay Horizons & Promotion Rules
We replay the chronological candidate sequence using **only `SELECT`-half utilities** to make acceptance decisions, and then score the resulting incumbent trajectory on **`EVAL`-half families only**:
- **Primary Replay Sequence (`r3175`-Initialized, `t = 0 .. 1499`)**: Initialized at `t = 0` with `r3175` (`round 3175`), replaying the `1,499` subsequent candidate rounds `3176 .. 4675` in chronological order. We evaluate both `SELECT`-half context-mean selection (matching the live flywheel's context-mean rule on the `SELECT` half) and `SELECT`-half family-equal mean selection.
- **Secondary Sensitivity Replay (`r3176`-Initialized Post-Peak Stream, `t = 0 .. 1498`)**: Because `r3175` was already selected as the full-91-context peak in the live run, we also report the replay initialized at the first post-peak candidate `r3176` (`round 3176`) across `3177 .. 4675` (`1,498` steps) to measure how each rule climbs out of a typical `~1.132` candidate state using `SELECT`-half utilities alone.

Under all rules, **ties ($|\Delta_{\text{select}}| \le 10^{-4}$) are never accepted** (`TIE_EPS = 1e-4`):
1. **`R0_BASELINE`**: Accept candidate $u_t$ iff $\Delta_{\text{select}}(u_t, u_{\text{inc}}) > 10^{-4}$ (we also report literal `> 0.0` without tie guard as a diagnostic).
2. **`R1_MARGIN_001` (`m = 0.001`)**: Accept candidate $u_t$ iff paired `SELECT`-half gain $\Delta_{\text{select}}(u_t, u_{\text{inc}}) > 0.001$.
3. **`R1_MARGIN_0035` (`m = 0.0035`, `H301` 2-context step)**: Accept candidate $u_t$ iff paired `SELECT`-half gain $\Delta_{\text{select}}(u_t, u_{\text{inc}}) > 0.0035$.
4. **`R2_LCB`**: Accept candidate $u_t$ iff $\Delta_{\text{select}}(u_t, u_{\text{inc}}) > 10^{-4}$ AND the paired whole-family bootstrap `LCB95` (`0.05` quantile over `1,000` draws of `SELECT`-half families with replacement, deterministic `RNG = 304200 + seed`) is strictly `> 0.0` (we also report the two-sided `0.025` quantile variant `R2_LCB_2SIDED`).
5. **`R3_MEDIAN_OF_K` (`K = 5`)**: Per-context utilities and selected-program hashes exist in `history.npz` (`shape = (1500, 91)`). The active incumbent at step $t$ is the per-context median utility vector over the last $\min(|H_{t-1}|, 5)$ accepted checkpoints; candidate $u_t$ is accepted iff $\Delta_{\text{select}}(u_t, \tilde{u}_{\text{inc}}) > 10^{-4}$, updating the accepted checkpoint buffer $H_t = H_{t-1} \cup [u_t]$ and recomputing the per-context median over the last $\min(|H_t|, 5)$ accepted checkpoints.

---

## 4. Estimands & Decision Criteria

For each split seed $s \in \{30401, 30402, 30403\}$ and each rule $R \in \{\text{R0}, \text{R1\_001}, \text{R1\_0035}, \text{R2\_LCB}, \text{R3\_MEDIAN\_OF\_5}\}$:
- **`EVAL`-Half Family-Equal Utility (Final Incumbent & Time-Averaged Incumbent)**: Equal weight across `EVAL`-half families, reporting level and paired difference **Rule minus `R0_BASELINE`**, median across the `3` split seeds, and **`10,000`-draw whole-family bootstrap `95%` CI** (`RNG = 304150`, including one-sided `LCB95 = q05` and two-sided `[q025, q975]`).
- **Promotion Count & Promotion Precision**: Number of accepted promotions $N_{\text{prom}}$ and fraction of promotions that improve `EVAL`-half family-equal utility by $> 10^{-4}$ (`precision = n_eval_improved / n_promotions`; defined as `0.0` when $N_{\text{prom}} = 0$).
- **Winner's-Curse Shrinkage**: $\Delta_{\text{select}} - \Delta_{\text{eval}}$ (mean over accepted promotions and net final-incumbent shrinkage from `t = 0`).
- **MM Bradley-Terry Elo**: Computed on `EVAL`-half contexts with `r3175` anchored at `1500.0` (`400 * log10`), with `10,000`-draw whole-family bootstrap `95%` CI (`RNG = 304150`). Never AUC, no speed verdicts.

### Decision Rule
1. Choose the single best non-`R0` rule **on split seed `30401` only** (by highest final-incumbent `EVAL`-half family-equal gain over `R0_BASELINE`, tie-broken by time-averaged `EVAL`-half family-equal gain over `R0_BASELINE`), then freeze that choice for evaluation across all `3` split seeds (`30401, 30402, 30403`).
2. Verdict is **`RULE_ADVANCES`** iff the frozen best non-`R0` rule satisfies all three conditions:
   - **Condition 1**: Median `EVAL`-half family-equal gain vs `R0_BASELINE` across the `3` split seeds is **`> +0.001`**, **AND**
   - **Condition 2**: Per-seed whole-family bootstrap **`LCB95 > 0` on `3/3` split seeds**, **AND**
   - **Condition 3**: **Promotion precision `> R0_BASELINE`** (across the 3 split seeds).
3. Otherwise verdict is **`RULE_NOT_ADVANCED`**. Note: split seeds partition the 19 `dev91` families rather than re-running training seeds, so even a pass is a recommendation for a future flywheel-shadow wave, not a breakthrough.
