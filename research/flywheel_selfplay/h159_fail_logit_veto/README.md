# H159 — TRAIN-calibrated `fail_logit` veto at selection time (zero-update) on `r882`

**Status: complete. Verdict: `NO_ADVANCE` (`primary_veto_fix_dl_val: RULE_NEUTRAL / 0.0000`, `secondary_veto_cv_dl_train: -0.0033`)**
(preregistered in [PREREG.md](PREREG.md), PRs [#1334](https://github.com/neverhuman/JopeDime/pull/1334) & [#1336](https://github.com/neverhuman/JopeDime/pull/1336), merged source `a816ca28c326a3bfb7b84ae8363c628dbd8dfc53`; single zero-update CPU run on `xbabe2`).

## Hypothesis and Setup

- **Hypothesis**: `H143` found every live admission since `r774` was a `-2` cliff flip, and `H152`/`H139` deficits are one cliff context each. The frozen `FlywheelGameNet` anchor inside `r882` already emits per-candidate `fail_logit` (`net.adapter_10tok.predict_costs(...)` returns both `total_cost` and `fail_logit`). `H159` tested whether a zero-update selection-time decision rule — *"pick $\arg\min$ `total_cost` among candidates with $\sigma(\text{fail\_logit}) < t$, else fall back to plain $\arg\min$ `total_cost`"* — with $t \in \mathcal{T} = \{0.50, 0.60, 0.70, 0.80, 0.90, 0.95\}$ calibrated only on `DL-Train` families cuts `-2` picks and raises equal-family utility over `r882`'s plain $\arg\min$.
- **Arms & Protocol**:
  - `R882`: Plain $\arg\min$ ($t = 1.0$; exact `91/91` `1.1377941898` baseline gate required first).
  - `VETO_FIX` (Primary on 91-ctx `DL-Val` / 19 families): $t^*$ fit by equal-family utility on all `205` `DL-Train` families (`1,219` contexts) and across 3 family-resampling seeds (`159101, 159102, 159103`), breaking exact ties toward the largest $t \in \mathcal{T}$ (most conservative veto).
  - `VETO_CV` (Secondary out-of-fold on 1,219-ctx / 205-family `DL-Train`): 5-fold family cross-fit (`fold_seed = 159140`, `41` families per fold, context counts `[245, 240, 225, 268, 241]`).
  - `ORACLE_INFO` (Descriptive diagnostic only — never used for selection): $\sigma(\text{fail\_logit})$ quantiles and count of `-2` cliff picks avoidable at any $t \in \mathcal{T}$ or any $t \in (0, 1)$.

## Outcome & Mechanism Finding

1. **`R882` already folds `fail_logit` into `total_cost`, leaving all 17 `DL-Train` `-2` cliffs at $\sigma(\text{fail\_logit}) < 0.035$**:
   - Because `FlywheelGameNet.predict_costs` already adds the soft barrier `softplus(gate_barrier_weight) * sigmoid(fail_logit)` into `anchor_total`, `90%` of `R882` plain-$\arg\min$ winners already have $\sigma(\text{fail\_logit}) \le 0.0306$ on `DL-Train` (`p50 = 0.0127`, `p90 = 0.0306`, `max = 0.8860`; only `23 / 1,219` winners have $\sigma(\text{fail\_logit}) \ge 0.50$) and $\le 0.0335$ on `DL-Val` (`p50 = 0.0117`, `p90 = 0.0335`, `max = 0.8090`; only `3 / 91` winners $\ge 0.50$).
   - Crucially, **all `17` `-2` cliff picks made by `R882` on `DL-Train` are `gate_head` false negatives with $\sigma(\text{fail\_logit}) \in [0.0021, 0.0350]$ (`0 / 17` avoidable at any $t \in \{0.50, 0.60, 0.70, 0.80, 0.90, 0.95\}$)**!
   - Conversely, the `23` `DL-Train` contexts where `R882`'s winner has $\sigma(\text{fail\_logit}) \in [0.50, 0.8860]$ are valid candidates; vetoing them at $t \in \{0.50, 0.60, 0.70, 0.80\}$ adds `+1` `-2` cliff pick (`18` vs `17`) and lowers `DL-Train` equal-family utility (`1.3811–1.3829` vs `1.3854`).
2. **`VETO_FIX` (Primary on 91-ctx `DL-Val`): `RULE_NEUTRAL` (`0.0000` gain vs `R882`)**:
   - On `DL-Train`, $t \in \{0.90, 0.95\}$ vetoes `0` `R882` winners ($\max \sigma = 0.8860 < 0.90$) and achieves the maximum equal-family utility (`1.3853554`). Both the deterministic point fit and all `3/3` family-bootstrap seeds (`159101, 159102, 159103`) select **$t^* = 0.95$**.
   - Applied to 91-context `DL-Val` (where $\max \sigma = 0.8090 < 0.95$), $t^* = 0.95$ changes **`0 / 91` candidate choices across all 3 seeds**, yielding exact zero primary delta: **`0.0000` (`95% CI [0.0000, 0.0000]`, `p_one_sided = 1.0`, `0/3` seeds beating `R882`, `0` invalid picks)**.
3. **`VETO_CV` (Secondary 5-Fold Cross-Fit on 1,219-ctx / 205-family `DL-Train`): `NO_ADVANCE` (`-0.0033` vs `R882`)**:
   - In 5-fold cross-fitting, fold-level noise occasionally selects $t \in \{0.50, 0.60, 0.70, 0.80\}$ on 1–2 folds (`[0.6, 0.95, 0.95, 0.95, 0.95]` deterministically; `[0.5, 0.8, 0.95, 0.95, 0.8]`, `[0.6, 0.95, 0.6, 0.95, 0.8]`, `[0.7, 0.95, 0.95, 0.95, 0.7]` across seeds `159101–159103`), changing `7–12 / 1,219` choices out-of-fold, increasing `-2` picks from `17` to `18`, and yielding median equal-family gain **`-0.0033` (`95% CI [-0.0091, +0.00004]`, `p_one_sided = 0.9652`, `0/3` seeds > `R882`)**.

## Key Metrics Table

**Baseline `R882` gate:** `PASS` (`91/91` exact context matches, `max_context_error = 0.0`, context mean `1.1377941898`, equal-family mean `1.1536153`, `0` invalid picks on `DL-Val`; `17` invalid picks on `DL-Train`).

| Arm / Threshold | Panel | Context Mean Utility | Equal-Family Mean Utility | Gain vs `R882` [95% CI] | Seeds > `R882` | Changed Choices | Invalid (`-2`) Picks | Verdict |
|---|---|---:|---:|---|---:|---:|---:|---|
| `R882` (plain argmin) | `DL-Val` (`91 / 19`) | `1.137794` | `1.153615` | `0.0000` `[0.0000, 0.0000]` | — | `0 / 91` | `0` | baseline |
| **`VETO_FIX` ($t^* = 0.95$, `3/3` seeds)** | **`DL-Val` (`91 / 19`)** | **`1.137794`** | **`1.153615`** | **`0.0000` `[0.0000, 0.0000]` (`p = 1.0`)** | **`0/3`** | **`0 / 91`** | **`0`** | **`RULE_NEUTRAL` / `NO_ADVANCE`** |
| `R882` (plain argmin) | `DL-Train` (`1219 / 205`) | `1.423073` | `1.385355` | `0.0000` `[0.0000, 0.0000]` | — | `0 / 1219` | `17` | baseline |
| **`VETO_CV` (5-fold cross-fit)** | **`DL-Train` (`1219 / 205`)** | **`1.420215–1.420810`** | **`1.381952–1.382489`** | **`-0.0033` `[-0.0091, +0.00004]` (`p = 0.965`)** | **`0/3`** | **`9–12 / 1219`** | **`18`** | **`NO_ADVANCE`** |
| Grid $t = 0.50$ (`ORACLE_INFO`) | `DL-Val` / `DL-Train` | `1.152717` / `1.419799` | `1.167909` / `1.381095` | `+0.0143` (val) / `-0.0043` (train) | — | `3` / `23` | `0` / `18` | descriptive |
| Grid $t = 0.60$ (`ORACLE_INFO`) | `DL-Val` / `DL-Train` | `1.148175` / `1.421049` | `1.163559` / `1.382844` | `+0.0099` (val) / `-0.0025` (train) | — | `3` / `18` | `0` / `18` | descriptive |
| Grid $t = 0.70$ (`ORACLE_INFO`) | `DL-Val` / `DL-Train` | `1.140647` / `1.420982` | `1.156348` / `1.382821` | `+0.0027` (val) / `-0.0025` (train) | — | `2` / `15` | `0` / `18` | descriptive |
| Grid $t = 0.80$ (`ORACLE_INFO`) | `DL-Val` / `DL-Train` | `1.142451` / `1.421048` | `1.158076` / `1.382913` | `+0.0045` (val) / `-0.0024` (train) | — | `1` / `13` | `0` / `18` | descriptive |
| Grid $t \in \{0.90, 0.95\}$ (`ORACLE_INFO`) | `DL-Val` / `DL-Train` | `1.137794` / `1.423073` | `1.153615` / `1.385355` | `0.0000` (val) / `0.0000` (train) | — | `0` / `0` | `0` / `17` | descriptive |

*(Note on `vs_r882` Elo when `changed_choices == 0`: under strict `ties = "reject"`, `0` wins out of `19` zero-difference families maps to the clamped `1e-6` floor `-900.0`, while `ties = "half"` Elo is `1500.0`.)*

## Execution, Resource Bounds & Custody

- **Single Scored Execution (`xbabe2` CPU)**:
  - Merged source commit: `a816ca28c326a3bfb7b84ae8363c628dbd8dfc53` (`manifest_sha256 = aa52d3558a062be165bd4459cfeb549867e793b208215c9e5e9d4083339e4866`; follows pre-baseline `0`-context preflight schema fix PR [#1336](https://github.com/neverhuman/JopeDime/pull/1336)).
  - Optimizer updates: `0`; scored contexts: `1,310` (`91` `DL-Val` + `1,219` `DL-Train`).
  - Wall clock: `365.35 s` (`6.09 min`) at `nice 19`, `CUDA_VISIBLE_DEVICES=""` (`cuda_initialized == False`).
  - Supervisor resource samples (`1,056` samples): max OS threads **`2`** (cap $\le 4$), peak RSS **`3,243,368,448 B`** (`3.24 GB`, cap $< 6.0\text{ GB}$), min free disk `662.3 GB` ($\ge 40\text{ GB}$), max user-visible disk used `91.57%` ($< 95\%$), max scratch `8.00 MB` ($< 3.0\text{ GB}$), max load `81.52` ($\le 85.0$), min `MemAvailable` `67.75 GB` ($\ge 16\text{ GiB}$).
  - Bytewise identity of all 47 adapter tensors (`adapter_sha256 = 9c3ba940d9eb60aa2a39dd57c229c0952927ea1c2f22c372ae9764f43baa8669`), non-adapter state (`frozen_sha256 = 8d1070faf574ee1cd2a4e630887e1662b43189b3f89b04982143ef2704818aae`), and DOPE (`c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814`) verified before and after scoring; `0` protected/holdout/sealed reads (`audit.json`).
- **Custody & Cleanup**:
  - S3 archive (`21/21` files restore-verified byte-for-byte): `s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/H159/argon-h159-20261101a/`
  - Local custody: `/home/ubuntu/jope-rescue-20260921/h159-rd_argon-20261101/` and `/home/ubuntu/jope_plan/evidence/H159/`
  - Scratch `/home/ubuntu/jope-data/h159` deleted after S3 restore verification.

## Verdict & Next Step

- **Verdict**: **`NO_ADVANCE`** (`VETO_FIX` on `DL-Val` is `RULE_NEUTRAL` at $t^* = 0.95$ with `0/91` changed choices; `VETO_CV` on `DL-Train` is `-0.0033`).
- **Next step**: Do not add a post-hoc `fail_logit` threshold veto on top of `r882`'s `total_cost`. Because all `17` `-2` cliff picks on `DL-Train` occur at low anchor `fail_logit` ($\sigma(\text{fail\_logit}) < 0.035$), preventing `-2` cliff flips requires training or calibrating a separate multimodal/operator-aware gate classifier rather than thresholding the existing anchor `fail_logit`.
