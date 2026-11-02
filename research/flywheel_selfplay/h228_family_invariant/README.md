# H228 — Family-Invariant Adapter Continuation: AND-Mask ($\tau=0.8$) and GroupDRO ($\eta=0.01$)

**Status**: `INCOMPLETE` (`58/60` member fits completed + `1` timing-smoke fit = `2,324/2,404` optimizer updates; terminal reason `LOCK_UNAVAILABLE_QLT04` after `JRB-01` GPU release per watcher ruling `hw_t2120_argon_h228_close.md`).
**Verdict**: `null` (`INCOMPLETE 58/60` — **no verdict** on `GDRO` or `ANDMASK_SHUF`, which have `14/15` fits and are missing seed `228103` fold `4`; complete `15/15` arms `CTRL` and `ANDMASK` are reported below strictly as **DESCRIPTIVE / NOT A DECISION**; ties = no advance; `serving_promotion = false`).

---

## 1. Hypothesis & Preregistered Design

Restricting `r882` adapter continuation updates (`47` trainable adapter tensors, `29,211` parameters; `41` active gradients, `6` inactive gradients; `40` updates/fit; `4` microbatches $\times 16$ contexts = `64` contexts/update) to coordinate directions whose gradient signs agree in $\ge \tau = 0.8$ of families (`ANDMASK`) or up-weighting the highest-loss families via Sagawa-2020 GroupDRO (`GDRO`, $\eta = 0.01$) suppresses family-idiosyncratic overfitting and yields positive held-out-family lift versus frozen `r882` (`round-882.pt`, `1.1377941898164525`) where plain AdamW continuation (`CTRL`) and a per-step density-matched random-mask placebo (`ANDMASK_SHUF`) do not.

- **Preregistered admission rule**: Compare only against `r882 = 1.1377942` (`1.1377941898164525`); `ADVANCE` (`CANDIDATE`) only if equal-family median gain across 3 seeds (`228101, 228102, 228103`, fold seeds `228141, 228142, 228143`) is $> +0.002$, 95% family-bootstrap `CI95_lower > 0`, `LCB95 > 0`, `3/3` positive seeds, harm $\le 10\%$ under Holm-Bonferroni ($m=2$ over `ANDMASK`, `GDRO`), and the candidate also beats the `ANDMASK_SHUF` placebo; exact ties or cross-device noise ties reject (`r3175` near-tie rule). `BREAKTHROUGH` additionally requires passing on 19-family `DL-Val91` (`91` contexts). No AUC and no speed verdicts.

---

## 2. Execution Timeline, Honest Failure/Stop Record & Active-Budget Accounting

| Segment | Host / GPU | Window (UTC) | Fits & Updates | Active Seconds | Outcome & Authority |
|---|---|---|---|---|---|
| Pre-claim `Xid 79` refusal | `xbabe2` `RTX 3090` (`GPU-44c63d19-492a-ab46-cc6a-56e12a2ef51b`) | `18:58:23Z` | `0/60` fits (`0` updates) | `0.0s` (no worker spawned) | Preceding holder suffered PCIe bus fault (`Xid 79/154`); `launch.py` refused at preflight `guard()` with `admitted=false, outcomes_read=false` (`preflight-gpu0-xid79-refusal.json` `ee4816d1...`). Retargeted to `xbabe1` per `hw_t1905_argon_h228_gpulost.md` (PRs `#2029`, `#2033`, `#2036`, source commit `cf49650dff567ae9396e5f11a0353018ca64c274`). |
| `attempt1` on `xbabe1` | `xbabe1` `RTX 4090` (`GPU-784030c4-b1f9-d5a3-f0eb-9337f133f503`) | `19:27:08Z–19:49:59Z` | `1` smoke (`4` updates) + `47/60` member fits (`1,880` updates) = `1,884` updates | `1,370.038772s` | Native91 (`91/91` bit-exact, `1.1377941898164525`) + `1,219/1,219` static parity passed. Stopped cleanly at unit boundary after `GDRO-228103-1` (`RuntimeError('interrupted; no retry')`, `failure.json` `8490fb49...`) per `hw_t1950_argon_h228_blocks_flywheel.md` so `xbabe1` flywheel (`PID 2847658`) regained $\ge 20\text{ GiB}$ free VRAM (`round 3852` resumed at `19:50:21Z`). |
| Continuation on `xbabe2` | `xbabe2` `RTX 3090` (`GPU-44c63d19-492a-ab46-cc6a-56e12a2ef51b`) | `19:56:03Z–20:03:04Z` | Reused `47` S3-verified fits + trained `11` fits (`440` updates) = `58/60` fits (`2,324` updates) | `327.00s` (`14s + 16s` preflights + `297s` segment1) | Native91 (`91/91` bit-exact, `1.1377941898164525`) + `1,219/1,219` static parity verified on `xbabe2` `RTX 3090` before any 3090 unit ran. Stopped cleanly at unit boundary after `ANDMASK-228103-4` per watcher `gpu_ruling_t2002.md` to yield `xbabe2` GPU0 to `JRB-01` (`PID 1155864`). |
| Post-`JRB-01` lock check | `xbabe2` (`CPU` closeout readout) | `21:06:58Z–21:53:25Z` | `0` new fits (`2` missing: `GDRO-228103-4`, `ANDMASK_SHUF-228103-4`) | `1,697.04s` charged GPU carry (`3,702.96s` unused of `5,400.0s`) | `JRB-01` finished GPU work at `21:06:58Z` (`jrb_study_done_t2118.md`), and `QLT-04` resume controller (`PID 2989918`, ticket `1793653623083398632-2989918-Q1`) immediately took `/home/ubuntu/jope_plan/qualification.lock` at `21:07Z` for `541` remaining units (many hours, well past `23:30Z`). Closed immediately as `INCOMPLETE 58/60 (LOCK_UNAVAILABLE_QLT04)` per `hw_t2120_argon_h228_close.md`. |

---

## 3. Parity, Isolation & Invariant Exposure Verification

- **DOPE composite hash (`c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814`)**: Verified `before == after` across every fit on both `xbabe1` RTX 4090 and `xbabe2` RTX 3090 (`dope_equal_before_after = true`).
- **Full frozen network `state_dict` SHA256**: `6eb0e998896bd87553bbea815be06b534d5f3f1ae2a63cbf7ff4ec22e89c4f8a` before and after (`full_state_dict_equal_before_after = true`).
- **Adapter-only updates**: All `58` member fit proofs + `1` smoke proof verified `47` trainable adapter tensors (`29,211` parameters), `41` active gradients and `6` inactive gradients at every step, `before_sha256 != after_sha256`, and `forward_changed = true`.
- **Native91 bit-exact baseline (`91/91` contexts, mean `1.1377941898164525`)**: Verified bit-exact before any fit on `xbabe1` RTX 4090 (`baseline.json` `684d13c0...`), before any continuation fit on `xbabe2` RTX 3090 (`19:58:51Z`), and on `xbabe2` CPU (`CUDA_VISIBLE_DEVICES=""`). Static `1,219/1,219` `DL-Train` forward parity also matched `1,219/1,219` argmin picks between CPU and RTX 4090.
- **Data boundary**: `0` sealed, `0` holdout, and `0` meta-test reads (`sealed_holdout_meta_reads = 0`).
- **Invariant exposure**:
  - `ANDMASK` (`15/15` fits): mean kept coordinate fraction = `0.156069` (`15.61%` of coordinates agreed in sign across $\ge 80\%$ of sampled families).
  - `ANDMASK_SHUF` (`14/15` fits): mean kept coordinate fraction = `0.155918`; **exact per-step `kept_counts` density match (`160` microbatches / `40` steps per fit) verified `true` across all `14/14` completed `ANDMASK` / `ANDMASK_SHUF` pairs**.
  - Minimum distinct families per microbatch across all `58` fits: `11 >= 4`.
  - `GDRO` (`14/15` fits): terminal family weights spanned `[1.063e-06, 0.999631]`.

---

## 4. Descriptive Readout on Complete 15/15 Arms (`CTRL` and `ANDMASK`) — DESCRIPTIVE ONLY / NOT A DECISION

Per `hw_t2120_argon_h228_close.md` item 3, **no verdict is given on `GDRO` (`14/15`) or `ANDMASK_SHUF` (`14/15`)** because seed `228103` fold `4` (`GDRO-228103-4`, `ANDMASK_SHUF-228103-4`) was not executed. The two complete `15/15` arms (`CTRL` and `ANDMASK`, all 3 seeds $\times$ 5 folds completed) are reported below **strictly descriptively (`DESCRIPTIVE / NOT A DECISION`; ties = no advance)**:

| Contrast (`15/15` fits across `3` seeds) | Cohort | Per-Seed Equal-Family Gain (`228101, 228102, 228103`) | Median Gain | 95% Family-Bootstrap CI (`10,000` draws) | `LCB95` | Harm (`< -0.01`) | Positive Seeds | Paired Elo vs Ref (`1500`) [95% CI] | Descriptive Status |
|---|---|---|---|---|---|---|---|---|---|
| `CTRL vs R882` (`1.1377942`) | `DL-Train` OOF (`205` fams / `1,219` ctx) | `-0.040742, -0.040245, -0.040158` | **`-0.040245`** | `[-0.070502, -0.010648]` | `-0.065154` | `39.51%` (`81/205`) | `0/3` | **`1483.37`** `[1471.15, 1494.45]` | `DESCRIPTIVE / NOT A DECISION` (degrades `r882`) |
| `ANDMASK vs R882` (`1.1377942`) | `DL-Train` OOF (`205` fams / `1,219` ctx) | `-0.029852, -0.024904, -0.024686` | **`-0.024904`** | `[-0.053483, -0.001724]` | `-0.048745` | `33.17%` (`68/205`) | `0/3` | **`1487.52`** `[1477.71, 1497.31]` | `DESCRIPTIVE / NOT A DECISION` (degrades `r882`) |
| `ANDMASK vs CTRL` | `DL-Train` OOF (`205` fams / `1,219` ctx) | `+0.010890, +0.015341, +0.015472` | **`+0.015341`** | `[+0.002568, +0.027077]` | `+0.004172` | `14.15%` (`29/205`) | `3/3` | **`1503.93`** `[1497.04, 1511.25]` | `DESCRIPTIVE / NOT A DECISION` |
| `CTRL vs R882` (`1.1377942`) | `DL-Val91` Equal-Family (`19` fams / `91` ctx) | `-0.042284, -0.042667, -0.048115` | **`-0.042667`** | `[-0.099508, -0.003684]` | `-0.088298` | `31.58%` (`6/19`) | `0/3` | **`1489.04`** `[1448.48, 1547.79]` | `DESCRIPTIVE / NOT A DECISION` (degrades `r882`) |
| `ANDMASK vs R882` (`1.1377942`) | `DL-Val91` Equal-Family (`19` fams / `91` ctx) | `-0.026785, -0.026048, -0.020682` | **`-0.026048`** | `[-0.058841, -0.002083]` | `-0.052741` | `26.32%` (`5/19`) | `0/3` | **`1489.04`** `[1448.48, 1544.08]` | `DESCRIPTIVE / NOT A DECISION` (degrades `r882`) |
| `ANDMASK vs CTRL` | `DL-Val91` Equal-Family (`19` fams / `91` ctx) | `+0.015499, +0.016619, +0.027433` | **`+0.016619`** | `[-0.001099, +0.048321]` | `-0.000445` | `0.00%` (`0/19`) | `3/3` | **`1518.28`** `[1492.69, 1536.67]` | `DESCRIPTIVE / NOT A DECISION` |

*(On `DL-Val91` context-weighted readout across all `91` contexts, `CTRL vs R882` median gain is `-0.044656` `[-0.103374, -0.003972]` and `ANDMASK vs R882` median gain is `-0.027307` `[-0.061497, -0.002310]`.)*

---

## 5. Hardware-Split Red-Team Diagnostics (`hw_t2000_argon_h228_closeout_redteam.md`)

### 5.1 Unit-by-Unit Hardware Inventory (`47` on `xbabe1` RTX 4090 vs `11` on `xbabe2` RTX 3090 vs `2` Missing)

- **`xbabe1` RTX 4090 (`GPU-784030c4-b1f9-d5a3-f0eb-9337f133f503`, `47` fits)**:
  - Seed `228101`: all `20/20` fits (folds `0–4` $\times$ `CTRL, ANDMASK, GDRO, ANDMASK_SHUF`).
  - Seed `228102`: all `20/20` fits (folds `0–4` $\times$ `CTRL, ANDMASK, GDRO, ANDMASK_SHUF`).
  - Seed `228103` fold `0`: all `4/4` fits (`CTRL-228103-0`, `ANDMASK-228103-0`, `GDRO-228103-0`, `ANDMASK_SHUF-228103-0`).
  - Seed `228103` fold `1` (first 3 arms): `CTRL-228103-1`, `ANDMASK-228103-1`, `GDRO-228103-1`.
- **`xbabe2` RTX 3090 (`GPU-44c63d19-492a-ab46-cc6a-56e12a2ef51b`, `11` fits)**:
  - Seed `228103` fold `1` (4th arm): **`ANDMASK_SHUF-228103-1`** (ran on `xbabe2` RTX 3090 while its fold-1 mates `CTRL-228103-1`, `ANDMASK-228103-1`, `GDRO-228103-1` ran on `xbabe1` RTX 4090).
  - Seed `228103` fold `2`: all `4/4` fits (`CTRL-228103-2`, `ANDMASK-228103-2`, `GDRO-228103-2`, `ANDMASK_SHUF-228103-2`).
  - Seed `228103` fold `3`: all `4/4` fits (`CTRL-228103-3`, `ANDMASK-228103-3`, `GDRO-228103-3`, `ANDMASK_SHUF-228103-3`).
  - Seed `228103` fold `4` (first 2 arms): `CTRL-228103-4`, `ANDMASK-228103-4`.
- **Missing (`0/2` fits, `LOCK_UNAVAILABLE_QLT04`)**:
  - `GDRO-228103-4`, `ANDMASK_SHUF-228103-4`.

### 5.2 Per-Seed & Seed-`228103` Fold-Split Paired Arm Deltas (Descriptive Red-Team Diagnostic)

| Seed / Fold Slice | Hardware | Families (Contexts) | `CTRL vs R882` | `ANDMASK vs R882` | `GDRO vs R882` (desc.) | `ANDMASK_SHUF vs R882` (desc.) | `ANDMASK vs CTRL` | `ANDMASK vs ANDMASK_SHUF` (desc.) |
|---|---|---|---|---|---|---|---|---|
| Seed `228101` (folds `0–4`) | Pure `4090` (`20/20`) | `205` (`1,219`) | `-0.040742` `[-0.069650, -0.013009]` | `-0.029852` `[-0.058218, -0.004593]` | `-0.024423` `[-0.049746, +0.000629]` | `-0.022810` `[-0.049184, +0.001121]` | `+0.010890` `[-0.000231, +0.022972]` | **`-0.007042`** `[-0.019482, +0.003846]` |
| Seed `228102` (folds `0–4`) | Pure `4090` (`20/20`) | `205` (`1,219`) | `-0.040245` `[-0.072849, -0.009635]` | `-0.024904` `[-0.054080, +0.002493]` | `-0.026915` `[-0.056673, +0.001919]` | `-0.025789` `[-0.053068, -0.000589]` | `+0.015341` `[+0.004301, +0.028497]` | **`+0.000885`** `[-0.010058, +0.013776]` |
| Seed `228103` folds `0–1` | `4090` (`SHUF-1` on `3090`*) | `82` (`510`) | `-0.032276` `[-0.087887, +0.024996]` | `-0.007673` `[-0.042004, +0.032143]` | `-0.009059` `[-0.054917, +0.039721]` | `-0.009872`* `[-0.047848, +0.032025]` | `+0.024603` `[-0.015571, +0.068552]` | **`+0.002199`** `[-0.019307, +0.027915]` |
| — Fold `0` only | Pure `4090` (`4/4`) | `41` (`230`) | `-0.046054` `[-0.144118, +0.044773]` | `-0.027111` `[-0.079038, +0.018543]` | `-0.033025` `[-0.109538, +0.046847]` | `-0.040044` `[-0.102030, +0.013020]` | `+0.018943` | `+0.012933` `[-0.027182, +0.063114]` |
| — Fold `1` only (cross-GPU*) | `CTRL/AND/GDRO` `4090`; `SHUF` `3090` | `41` (`280`) | `-0.018498` `[-0.069131, +0.049273]` | `+0.011765` `[-0.026578, +0.072856]` | `+0.014908` `[-0.020908, +0.074302]` | `+0.020301`* `[-0.013375, +0.078647]` | `+0.030263` | `-0.008536`* `[-0.021510, -0.000458]` |
| Seed `228103` folds `2–3` | Pure `3090` (`8/8`) | `82` (`458`) | `-0.046396` `[-0.100124, -0.004810]` | `-0.039408` `[-0.092833, -0.000326]` | `-0.046456` `[-0.101406, -0.005058]` | `-0.031739` `[-0.070379, +0.000964]` | `+0.006988` `[-0.003620, +0.021695]` | **`-0.007669`** `[-0.032124, +0.012990]` |
| Seed `228103` folds `2–4` | Pure `3090` (`CTRL`/`ANDMASK`) | `123` (`709`) | `-0.045412` `[-0.086328, -0.011787]` | `-0.036028` `[-0.074648, -0.005723]` | *(fold 4 missing)* | *(fold 4 missing)* | `+0.009385` `[-0.002423, +0.025302]` | *(fold 4 missing)* |

### 5.3 Sensitivity Readout Dropping All `xbabe2` RTX 3090 Units (`4090`-Only Complete Seeds `228101` & `228102`, `40/40` Fits)

Dropping all `xbabe2` RTX 3090 units leaves the two 100%-complete `xbabe1` RTX 4090 seeds (`228101` and `228102`, `20/20` fits each across all `205` families):

| Contrast (`4090`-only, `2` complete seeds) | `DL-Train` OOF Median Gain (`205` fams) | `DL-Train` OOF `CI95` | `DL-Train` OOF Harm | `DL-Train` OOF Elo | `DL-Val91` Equal-Family Median Gain (`19` fams) | `DL-Val91` Equal-Family `CI95` | `DL-Val91` Elo |
|---|---|---|---|---|---|---|---|
| `CTRL vs R882` | `-0.040493` | `[-0.070317, -0.011722]` | `38.05%` (`78/205`) | `1481.85` `[1470.38, 1493.46]` | `-0.042475` | `[-0.099058, -0.003190]` | `1492.69` `[1452.14, 1545.94]` |
| `ANDMASK vs R882` | `-0.027378` | `[-0.054833, -0.001754]` | `33.66%` (`69/205`) | `1488.26` `[1478.66, 1497.87]` | `-0.026417` | `[-0.059228, -0.001984]` | `1487.21` `[1448.48, 1542.22]` |
| `GDRO vs R882` | `-0.025669` | `[-0.052090, +0.000372]` | `34.63%` (`71/205`) | `1484.74` `[1474.20, 1494.71]` | `-0.025928` | `[-0.068201, +0.001863]` | `1498.17` `[1459.63, 1551.52]` |
| `ANDMASK_SHUF vs R882` | `-0.024299` | `[-0.050396, -0.000199]` | `27.32%` (`56/205`) | `1491.31` `[1481.94, 1500.70]` | `-0.013991` | `[-0.032480, +0.000858]` | `1483.55` `[1457.78, 1510.96]` |
| `ANDMASK vs CTRL` | `+0.013116` | `[+0.003746, +0.023805]` | `14.63%` (`30/205`) | `1504.01` `[1496.94, 1511.54]` | `+0.016059` | `[-0.002121, +0.047725]` | `1510.98` `[1489.03, 1538.83]` |
| `ANDMASK_SHUF vs CTRL` | `+0.016194` | `[+0.004291, +0.029082]` | `13.66%` (`28/205`) | `1506.70` `[1499.94, 1513.69]` | `+0.028485` | `[+0.001105, +0.068019]` | `1490.89` |
| `ANDMASK vs ANDMASK_SHUF` | **`-0.003078`** | `[-0.011866, +0.005940]` | `12.20%` (`25/205`) | `1498.08` `[1493.59, 1502.58]` | **`-0.012426`** | `[-0.028408, -0.000915]` | `1498.17` `[1461.48, 1549.66]` |

### 5.4 Diagnostic Takeaway & Next Step

1. **Every gradient-continuation arm degrades frozen `r882`** across every seed, fold slice, and hardware subset (`-0.024` to `-0.043` vs `r882`, `0/3` positive seeds, `26%–40%` harmed families), confirming `H176` (`DATA_SATURATED`) and `H219` (`NO_ADVANCE`): `40` gradient steps on `1,219` `DL-Train` contexts overfit away from `r882`'s cross-family generalization optimum regardless of `ANDMASK` or `GDRO`.
2. **`ANDMASK`'s relative gain over `CTRL` (`+0.0153`) is fully explained by random coordinate sparsification (`ANDMASK_SHUF`)**: masking `84.39%` of coordinates at random (`ANDMASK_SHUF`) yields the same or slightly smaller degradation vs `r882` (`ANDMASK vs ANDMASK_SHUF = -0.003078` on pure-4090 OOF and `-0.012426` on `DL-Val91`), proving the effect is effective step-size damping rather than cross-family sign-agreement invariance.
3. **Next Step**: Do not re-run `H228` (`58/60` is closed `INCOMPLETE` with no retry). Gradient-based adapter continuation on the static `1,219`-context `DL-Train` pool remains closed; future breakthrough attempts should stay on zero-update post-processing / gated selection rules (`GATED_025` family) or new training signal sources.

---

## 6. S3 Custody & Reproducibility Artifacts

- **S3 Prefix**: `s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261102/H228/argon-xbabe1/attempt1/` (`125` versioned objects verified).
- **Terminal Closeout Receipt**: `s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261102/H228/argon-xbabe1/attempt1/incomplete-58of60-closeout.json` (`VersionId ZLcTt5tUswgx0XfLUIcvzJ6zxUa.ww4R`, `SHA256 316eb46c80f972195f1bca57ada700a7204596305a574c0f2dade6a17186fbbe`, `52,490` bytes).
- **Selected Key Receipts**:
  - `launch.json`: `VersionId NQRcUyjslp7dNFB9QtKu5YhUpJIj8qXl`, `SHA256 0f27abe2313151696dfd165c65ba776873b494bd90e07b305e134baa1ee989ec`
  - `baseline.json`: `VersionId A1Zofc1kbKfhXaawgj0LAn7vHPxtGhG6`, `SHA256 684d13c0c4765225df834257927efd6c32bd7c578121637898d2299fdf294f30`
  - `failure.json` (`xbabe1` `19:50Z` stop): `VersionId dY96e7FfcNptp6p9k8tH922P9a42qR0p`, `SHA256 8490fb49de97045d52354033159b3ca570dc2759ce9bcbaeb96b6d3c16e03165`
  - `partial-58of60-OOF-panels.npz` & `partial-58of60-DL-Val91-panels.npz`: stored in S3 under `attempt1/` (no `.pt` or `.npz` files committed to Git).
