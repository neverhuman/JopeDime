# H304 — LCB & Margin Promotion-Rule Replay on `dev91` Family Splits (`h304_lcb_margin_promotion_replay`)

- **Wave ID**: `H304` (`h304_lcb_margin_promotion_replay`)
- **Owner**: `rd_argon`
- **Mode**: **Analysis-Only (`xbabe2` CPU only, `nice 19`, `taskset -c 0-3`, `RSS <= 4 GiB`, `0` neural forwards/updates, `0` GPU use, `0` fresh-cohort reads)**
- **Status / Verdict**: **`COMPLETE` / `RULE_NOT_ADVANCED`**
- **Preregistration commit (pushed to `origin/v1/next` BEFORE any `EVAL`-half number was computed)**: `1c0622736fa77d265777132378674a07d5efcf89`
- **Pinned Inputs**:
  - `H289` `prepared-closure.tar.gz`: `VersionId = dj7fcZrNhjkNwY2s5DkfOah1nfAUObnL` (`sha256 = 66bc3154f4f2ec49bc9237163712826ec0bda82f32e3ba34c4881ba73b4bd071`)
  - `prepared/history.npz`: `sha256 = e6214cd5f3f3322a6ef22bfa7b3f7a797e106869c9195cdfe18d217b6ac7f446` (`1,500` rounds `3175..4675` excluding unevaluated round `4331`, `91` contexts across `19` families)
  - `adapter_admissions.jsonl`: `sha256 = 7c01de3b03eeb993d67c84266649a732365afdeaa86c593295b73a9d05b0b868` (`evaluation_identity = e79a7715c1531ccb8c4055ab96971101cd4034ec3a71d83bca6d25b0e604ca83`)
  - Frozen `DOPE` (`r3175`) `sha256`: `c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814` (untouched)

---

## 1. Hypothesis & Setup

`H301` (`ed8956e2c1`) established **`SELECTION_NOISE_PLATEAU`**: `r3175` (`dev91` mean `1.1389572403748425`) is a winner's-curse maximum over `~1,500–4,000` noisy 2-update proposals, and the live flywheel promotion rule (`accept iff dev91 mean > best`) admits tie/noise wins on a 91-context, 19-family validation set.

`H304` tests whether a stricter **selection rule** (`R1_MARGIN` `m in {0.001, 0.0035}`, `R2_LCB` 1,000-draw whole-family bootstrap `LCB95 > 0`, or `R3_MEDIAN_OF_K` per-context median over the last $K=5$ accepted checkpoints; ties $|d| \le 10^{-4}$ never accepted) replayed on a **`SELECT`** half of the `19` `dev91` families produces promotions that generalize to the unseen **`EVAL`** half of families:
1. **Step 1 (`H301` Regression Verification)**: Passed bit-for-bit (`True`) before preregistration (`r3175` mean = `1.1389572403748425`, `N = 1,499` post-`r3175` mean = `1.1320995298853955`, $q_{0.95} = 1.1354387884248622$, all `1,500` rounds verified against `adapter_admissions.jsonl` to `<= 1e-15`; see [`regression_receipt.json`](regression_receipt.json)).
2. **Step 2 (Family-Split Cross-Selection)**:
   - Split the `19` `dev91` families into `SELECT` (parity `0`) and `EVAL` (parity `1`) halves by `SHA256("H304-split-v1\x00" + str(seed) + "\x00" + family)[-1] % 2` across the `3` pre-declared split seeds `30401 / 30402 / 30403` (`10/9`, `12/7`, and `6/13` `SELECT/EVAL` families; `46/45`, `56/35`, and `26/65` `SELECT/EVAL` contexts), plus a `200`-split sensitivity (`RNG = 304150`, `200` unique random `9`-`SELECT` / `10`-`EVAL` family splits).
   - We evaluate both the **primary `r3175`-initialized replay** (`rounds 3175..4675`, `1,500` rounds starting at `r3175`) and the **post-peak stream sensitivity** (`rounds 3176..4675`, `1,499` rounds initialized at `r3176` so candidates are not starting from the global 91-context winner `r3175`), under both family-equal and context-mean `SELECT`-half scoring.

---

## 2. Primary Outcome: `r3175`-Initialized Replay (`rounds 3175..4675`, Split Seeds `30401 / 30402 / 30403`)

On all 3 pre-declared hash-parity split seeds (`30401, 30402, 30403`), **zero later candidates (`0 / 1,499`) beat `r3175` on the `SELECT` half (`> 1e-4`) under ANY rule** (`R0_BASELINE`, `R1_MARGIN_001`, `R1_MARGIN_0035`, `R2_LCB`, `R3_MEDIAN_OF_K`), whether `SELECT` utility is family-equal (`primary`) or context-mean (`sensitivity A`). Why: `r3175`'s lead over the post-`r3175` plateau (`+0.006858` over the mean, `+0.003518` over $q_{0.95}$) is concentrated in 2 families (`fam-08c91457...` and `fam-866f66b1...`), and in all 3 hash splits `30401`, `30402`, `30403`, both of those families landed in the `SELECT` half.

| Rule | Promotions by Seed (`30401, 30402, 30403`) | Promotion Precision | Final Incumbent Round by Seed | Median Final `EVAL` Gain vs `R0` (`[Pooled 95% CI]`) | Per-Seed 1-Sided `LCB95` vs `R0` | Median Time-Avg `EVAL` Gain vs `R0` (`[95% CI]`) | Median Winner's-Curse Shrinkage ($\Delta_{\text{sel}} - \Delta_{\text{eval}}$) | Median `EVAL` MM BT Elo vs `r3175` (`[95% CI]`) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **`R0_BASELINE`** | `0, 0, 0` (`0`) | `0.00%` | `3175, 3175, 3175` | `+0.000000` (`[0.000000, 0.000000]`) | `0.0, 0.0, 0.0` | `+0.000000` (`[0.000000, 0.000000]`) | `+0.000000` | `1500.00` (`[1500.00, 1500.00]`) |
| **`R1_MARGIN_001` (`m=0.001`)** | `0, 0, 0` (`0`) | `0.00%` | `3175, 3175, 3175` | `+0.000000` (`[0.000000, 0.000000]`) | `0.0, 0.0, 0.0` | `+0.000000` (`[0.000000, 0.000000]`) | `+0.000000` | `1500.00` (`[1500.00, 1500.00]`) |
| **`R1_MARGIN_0035` (`m=0.0035`)** | `0, 0, 0` (`0`) | `0.00%` | `3175, 3175, 3175` | `+0.000000` (`[0.000000, 0.000000]`) | `0.0, 0.0, 0.0` | `+0.000000` (`[0.000000, 0.000000]`) | `+0.000000` | `1500.00` (`[1500.00, 1500.00]`) |
| **`R2_LCB` (`1k` family boot)** | `0, 0, 0` (`0`) | `0.00%` | `3175, 3175, 3175` | `+0.000000` (`[0.000000, 0.000000]`) | `0.0, 0.0, 0.0` | `+0.000000` (`[0.000000, 0.000000]`) | `+0.000000` | `1500.00` (`[1500.00, 1500.00]`) |
| **`R3_MEDIAN_OF_K` ($K=5$)** | `0, 0, 0` (`0`) | `0.00%` | `3175, 3175, 3175` | `+0.000000` (`[0.000000, 0.000000]`) | `0.0, 0.0, 0.0` | `+0.000000` (`[0.000000, 0.000000]`) | `+0.000000` | `1500.00` (`[1500.00, 1500.00]`) |

- **Preregistered Decision Check**:
  - Best non-`R0` rule chosen on seed `30401`: `R1_MARGIN_001` (all rules tied at `0.0`).
  - Condition 1 (`median eval gain vs R0 > +0.001`): **`FAIL`** (`0.000000 <= +0.001`).
  - Condition 2 (`LCB95 > 0 on 3/3 split seeds`): **`FAIL`** (`[0.0, 0.0, 0.0]`).
  - Condition 3 (`promotion precision > R0`): **`FAIL`** (`0.0% == 0.0%`).
  - **Primary Verdict**: **`RULE_NOT_ADVANCED`**.

---

## 3. `200`-Split Sensitivity (`9` `SELECT` / `10` `EVAL` Families, `RNG = 304150`) & Post-Peak Stream Replay (`r3176`-Initialized)

### 3.1 `200`-Split Sensitivity on `r3175`-Initialized Replay (`rounds 3175..4675`)
Across `200` random balanced `9`-`SELECT` / `10`-`EVAL` family splits, **`75 / 200` (`37.5%`)** place `r3175`'s winning families in the `EVAL` half so that later candidates beat `r3175` on the `SELECT` half:

| Rule (`r3175`-Init, `200` Splits) | Splits w/ Promotions (`/ 200`) | Total Promotions (`Eval Improved / Harmed`) | Promotion Precision (`Harm Rate`) | Median Final `EVAL` Gain vs `R0` | Mean Final `EVAL` Gain vs `R0` (`[q025, q975]`) | Split `W / T / L` vs `R0` | Mean Winner's-Curse Shrinkage ($\Delta_{\text{sel}} - \Delta_{\text{eval}}$) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **`R0_BASELINE`** | `75 / 200` | `78` (`3 / 75`) | **`3.85%`** (`96.15%` harm!) | `+0.000000` | `+0.000000` (`[0.000000, 0.000000]`) | `0 / 200 / 0` | **`+0.003608`** |
| **`R1_MARGIN_001`** | `61 / 200` | `61` (`0 / 61`) | **`0.00%`** (`100.0%` harm!) | `+0.000000` | `+0.001125` (`[0.000000, +0.016067]`) | `14 / 186 / 0` | `+0.002418` |
| **`R1_MARGIN_0035`** | `61 / 200` | `61` (`0 / 61`) | **`0.00%`** (`100.0%` harm!) | `+0.000000` | `+0.001125` (`[0.000000, +0.016067]`) | `14 / 186 / 0` | `+0.002418` |
| **`R2_LCB`** | **`0 / 200`** | **`0`** (`0 / 0`) | `0.00%` (`0.00%` harm) | `+0.000000` | **`+0.002387`** (`[0.000000, +0.016067]`) | **`75 / 125 / 0`** | **`+0.000000`** |
| **`R3_MEDIAN_OF_K`** | `75 / 200` | `164` (`0 / 150`) | `0.00%` (`91.46%` harm) | `+0.000000` | `+0.000000` (`[0.000000, 0.000000]`) | `0 / 200 / 0` | `+0.003602` |

**Key Diagnostic Finding**: When `R0_BASELINE` promotes a post-`r3175` candidate on a `9`-family `SELECT` half, **`96.15%` (`75 / 78`) of those promotions harm the unseen `10`-family `EVAL` half** (mean winner's-curse shrinkage `+0.003608`). `R1_MARGIN` (`0.001` and `0.0035`) only blocks `17 / 78` false-positive promotions (`61 / 61` remaining promotions still harm `EVAL`), while `R2_LCB` (`1k` family bootstrap `LCB95 > 0`) **blocks `78 / 78` (`100%`) of false-positive promotions**, beating `R0` on all `75 / 200` splits (`mean +0.002387`) **solely by refusing to promote**.

### 3.2 Post-Peak Stream Sensitivity (`r3176`-Initialized, `rounds 3176..4675`, Climbing Out of `~1.132`)
When initialized at the first post-peak candidate `r3176` (`round 3176`) across `rounds 3176..4675`:
- **On the `3` Split Seeds (`30401, 30402, 30403`)**:
  - `R0_BASELINE` makes `[3, 1, 1]` promotions (`3 / 5 = 60.0%` precision, median shrinkage `+0.002543`), `R1_MARGIN_001` makes `[1, 1, 1]` (`2 / 3 = 66.7%` precision, filtering out two noise steps on seed `30401`), and `R1_MARGIN_0035` makes `[0, 1, 1]` (**`2 / 2 = 100.0%` precision** — Condition 3 `PASS`).
  - However, `R0_BASELINE`, `R1_MARGIN_001`, `R1_MARGIN_0035`, and `R3_MEDIAN_OF_K` all converge to the **exact same final incumbent on `3/3` split seeds** (`median final EVAL gain vs R0 = 0.000000 [0.000000, 0.000000]`, `Elo vs r3175 = 1220.41`), while `R2_LCB` (`[0, 0, 0]` promotions) is over-conservative on `6–12` families and loses `-0.004123 [-0.008725, 0.000000]` vs `R0`.
- **Across all `200` Balanced Splits (`r3176`-Initialized)**:
  - `R1_MARGIN_0035` increases promotion precision from `R0`'s `34.00%` (`154 / 453` improving `EVAL`, `54` harming `EVAL`) to **`44.86%`** (`109 / 243` improving `EVAL`, `11` harming `EVAL`), yet its final `EVAL` gain vs `R0` has **`median = 0.000000`** and **`mean = -0.000162`** (`16` wins, `177` ties, `7` losses vs `R0`), and every active rule suffers **`+0.0044` to `+0.0063` winner's-curse shrinkage** ($\Delta_{\text{select}} - \Delta_{\text{eval}}$).

---

## 4. Conclusion & Next Step

- **Verdict**: **`RULE_NOT_ADVANCED`**. Tightening scalar checkpoint-promotion rules on the `19`-family `dev91` validation set (`R1_MARGIN`, `R2_LCB`, `R3_MEDIAN_OF_K`) cannot create cross-family generalization out of the post-`r3175` 2-update adapter proposals: either the rule abstains (`0` promotions, `0.0` gain vs `R0` when anchored at `r3175`) or, when climbing from a post-peak state (`r3176`), converges to the same terminal checkpoint with `+0.0057..+0.0063` winner's-curse shrinkage.
- **Next Step**: Do not modify the live flywheel promotion rule; focus selection-layer compute on the confirmed `ENS_VOTE2` per-context slot-selection architecture (`H294` `CONFIRMED_FRESH`, `H302` `INNERCV_CONFIRMED_TRAINING_SEEDS`).
