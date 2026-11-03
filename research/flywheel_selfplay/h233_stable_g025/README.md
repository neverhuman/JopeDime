# H233 — `STABLE_G025` Pick-Stability Unanimity Vote Gate Inside Frozen H217 `GATED_025` Recipe

- **Status:** **`CLOSED INCOMPLETE`** (`44/75` HistGBM member subfits completed — Seed `233101` `25/25`, Seed `233102` `19/25`, Seed `233103` `0/25`; stopped at `2026-11-03T01:20:50.84Z` by `monitor.py` hard host-memory guard `MemAvailable < 20 GiB`)
- **Verdict:** **`CLOSED INCOMPLETE`** (hypothesis untested at preregistered 3-seed power; **no efficacy verdict, no candidate admission, no serving promotion**)
- **Owner:** `rd_argon` `[ARGON]` (`~/nudges/hw_t2205_argon_h233.md`, closeout `~/nudges/hw_t0125_argon_h233_closeout.md`, claim `~/jope_plan/ns/receipts/H233_claim.txt`)
- **Preregistration:** [PREREG.md](PREREG.md) (`SHA256 6f2bec113c1fc3161db0ff822c8472881ffbd83262772eb597f0d4e9114fddad`, PR `#2065`, reviewed head `136475b7d746f4c64c1bcc31c95dd6f87a36a345`, merged to `main` at `822ac8d42626d90dc21f46b35676b8a5a79d3b2a`)
- **Fold Manifest:** [fold_manifest.json](fold_manifest.json) (`SHA256 0967a8d0a65d4213305c54dd1d98e2e0240f469b799ba6e84e5e3e578199d1cf`; fresh seeds `233101–233103`, fold seeds `233141–233143`; max best-match Jaccard vs `H203–H231` = `0.1993 <= 0.20`)
- **Closeout Artifacts:** [metrics.json](metrics.json) · [manifest.json](manifest.json)

---

## 1. Hypothesis & Experimental Setup

### 1.1 Hypothesis
`H217` confirmed `GATED_025` (`+0.0216` equal-family native gain over `r882`, `4.4%` family harm), while `H223` showed that averaging the continuous cost predictions of `3` family-bootstrap `HistGBM` members (`GATED_BAG3`) diluted the `zgap` signal (`NO_ADVANCE`), and `H228` closed gradient-based adapter continuation on the static `1,219`-context `DL-Train` pool.

**`H233` (`STABLE_G025`)** tested whether using `3` `HistGBM` members (`m0` = unweighted `H217` fit, `m1–m2` = `FIT`-family cluster bootstrap refits with the identical recipe) as a **discrete pick-unanimity vote gate** — rather than averaging their continuous costs — could filter out marginal `GATED_025` switches while keeping `m0`'s exact `zgap` calibration and cost blend:

$$\text{cost}_{\text{STABLE\_G025}}(c) = \begin{cases} b^{(0)}_c & \text{if } \text{zgap}(r_c, b^{(0)}_c) \le \tau_{\text{cell}} \;\land\; \arg\min b^{(0)}_c = \arg\min b^{(1)}_c = \arg\min b^{(2)}_c, \\ r_c & \text{otherwise}, \end{cases}$$

where $b^{(m)}_c = r_c + 0.25\,(\hat{u}^{(m)}_c - r_c)$ is member $m$'s blended cost vector and $\tau_{\text{cell}}$ is chosen on `FIT` only using member `0` under the frozen `H210`/`H217` rule (maximizing `FIT` equal-family native gain subject to `FIT` family harm $\le 5\%$).

### 1.2 Registered Arms & Budget
- **`R882`**: Frozen reference checkpoint (`0084479aa0703667a0f3ccfe2beda60741c204795ec57b7c08542401ec6cdb64`, `DL-Val91` context-mean utility `1.1377941898164525`, anchored at Elo `1500.0`).
- **`G025`**: Frozen `H217` `GATED_025` recipe (`zgap <= tau_cell` on member `0` only; replication anchor).
- **`STABLE_G025`** *(sole primary, Holm $m=1$)*: `zgap <= tau_cell` **and** unanimous first-argmin pick across all `3` outer `HistGBM` members (`m0, m1, m2`), else `r882`.
- **`STABLE_SHUF`** *(matched-rate placebo)*: `zgap <= tau_cell` **and** a boolean mask obtained by deterministically permuting the 3-member unanimity indicator across contexts within each `(seed, outer_k)` cell (`SHUF_SEED = 233160`), preserving the exact cell unanimity rate.
- **`BAG_VOTE_UNGATED`** *(report-only)*: Apply $b^{(0)}_c$ iff the 3 blended members' first-argmin picks agree, with no `zgap` gate.
- **Fit Budget**: Per seed (`233101, 233102, 233103`), `5` outer exclusion pairs (`exclude {A, B}`) $\times$ `3` members (`0, 1, 2`) = `15` fits + `10` inner exclusion triples (`exclude {A, B, j}`) $\times$ `1` member (`0`) = `10` fits $\implies$ **`25` `HistGBM` fits/seed (`75` total)** on `xbabe2` CPU at `nice 19`, `0` optimizer updates.

---

## 2. Execution Outcome & Abort Cause Diagnosis

### 2.1 Execution Timeline & Unit Accounting
1. **`TRAIN`-Only Smoke Check (`smoke-20261102T2225Z`)**: Completed `PASS` in `57.61s` wall (`4` `HistGBM` fits — `1` outer unit $\times 3$ members + `1` inner unit $\times 1$ member on a 25-context `TRAIN` subset; peak RSS `969,433,088 B` = `0.90 GiB`, peak `tree_cores = 7.04 <= 8.0`, `0` `DL-Val` reads, `0` protected reads, `DOPE c51a56d0...` `before == after`).
2. **Primary Attempt (`run-20261102T2226Z`, `task-631`)**:
   - Launched at `2026-11-02T22:26:30Z` on `xbabe2` CPU (`nice 19`, `OMP_NUM_THREADS=7` when `load5 < 50` else `3`, `MKL/OPENBLAS/torch=1`, `CUDA_VISIBLE_DEVICES=""`).
   - Passed the mandatory `91/91` `DL-Val91` native `r882` regression gate bit-exactly (`1.1377941898164525`, `BASELINE PASS: 91/91 native r882 exact`).
   - Completed **`44/75` `HistGBM` member subfits** (`24/45` fit units; `48` total `HistGBM` fits including `4` smoke fits), with every `.npz` prediction archive SHA256-verified against its `.json` receipt:
     - **Seed `233101`**: **`25/25` subfits complete** (`5/5` outer units $\times 3$ members = `15` + `10/10` inner units $\times 1$ member = `10`).
     - **Seed `233102`**: **`19/25` subfits complete** (`5/5` outer units $\times 3$ members = `15` + `4/10` inner units `exclude012-m0`, `exclude013-m0`, `exclude014-m0`, `exclude023-m0`; missing `6` inner units `024, 034, 123, 124, 134, 234`).
     - **Seed `233103`**: **`0/25` subfits complete** (`25` subfits missing).
   - **Monitor Guard Stop (`2026-11-03T01:20:50.84Z`)**: `monitor.py` terminated the worker process group with `SIGTERM` (`exit_code = -15`, `status = "INCOMPLETE"`, `reasons = ["MemAvailable<20GiB"]`) after `10,454.27s` total elapsed wall (`2,606.67s` paused across `5` `load5 > 60` intervals, `7,847.60s` active wall, `40,379.20s` cumulative worker-tree CPU).
   - **No Retry / No Resume**: Per [PREREG.md](PREREG.md#L65-L66) lines 65–66 (*"One primary attempt (`75` HistGBM fits across 3 seeds). No retry if the run aborts or misses a gate."*), the unused `--resume` path in `runner.py` was **not** invoked (`resume_invoked = false`, `retry_attempts = 0`).

### 2.2 Red-Team Diagnosis of `MemAvailable < 20 GiB` Abort
- **Host-Wide Memory Pressure, Not Own RSS**: Across the entire `10,454.27s` run, H233's own process-tree RSS peaked at **`3,587,100,672 B` (`3.3407 GiB`)**, strictly below its `4.0 GiB` (`4,294,967,296 B`) cap, and its 30-second rolling CPU usage peaked at **`6.9324` cores** (`<= 8.0` limit).
- At `2026-11-03T01:20:08.59Z` (`utc = 1793668808.59`), `monitor.py` had **already `SIGSTOP`-paused** H233's worker tree because host `load5` crossed `60.0` (`load5 = 61.06`, `load1 = 105.27`). While H233 sat paused at **`457,138,176 B` (`0.4257 GiB`) RSS** and `0.38` rolling cores, concurrent `xbabe2` workloads drove `load1` to `129.78` (`load5 = 73.38`) and pushed host-wide `MemAvailable` down from `27.18 GB` (`01:20:43Z`) to `< 20 GiB` (`21,474,836,480 B`) at `01:20:50.84Z` (also observed by concurrent wave `H237`, which recorded a `24s` memory pause at `~19.7 GiB` during `00:38–01:23Z`), while `xbabe2` swap (`8.0 GiB` total) was `99.99%` full (`8,589,099,008 B` used, `831,488 B` free).

---

## 3. Descriptive-Only Seed-`233101` Metrics (`1/3` Seeds Complete)

> **WARNING — DESCRIPTIVE ONLY (NO EFFICACY VERDICT, NO ADOPTION):**
> Preregistered confirmatory inference requires all `3` seeds (`233101–233103`, `75/75` fits). Because only Seed `233101` (`25/25` fits, `5/5` outer cells, `205` families, `1,219` contexts) completed all 5 outer folds, the tables below are reported **strictly as descriptive post-run diagnostics** per `hw_t0125_argon_h233_closeout.md`. They carry **no statistical efficacy verdict, no Holm rejection, and no candidate adoption**.

### 3.1 Seed `233101` Pooled-`A` Out-of-Fold (`205` Families, `1,219` Contexts — Descriptive Only)

*(Metric directions: higher Equal-Family Native Gain vs `R882` is better; higher **Elo Lift vs Incumbent** ($\text{Elo} - 1500.0$) is better; lower Family Harm (`< -0.01`) is better.)*

| Arm / Paired Contrast (Seed `233101` only, `205` fams) | Equal-Family Native Gain | 10k Family-Bootstrap `CI95` | Family Harm (`< -0.01`) | **Elo Lift vs Incumbent (`R882` / Ref)** | **Elo Lift `CI95`** | Gate Open Rate (Overall / Switching) |
|---|---|---|---|---|---|---|
| `G025 vs R882` *(replication anchor)* | `+0.016429` | `[+0.002585, +0.038454]` | `4.88%` (`10/205`) | **`+2.90`** (`1502.90`) | `[-3.81, +9.72]` | `74.32%` / `32.49%` |
| `STABLE_G025 vs R882` *(primary arm, 1 seed)* | `+0.012502` | `[+0.000456, +0.033610]` | **`0.98%`** (`2/205`) | **`+1.76`** (`1501.76`) | `[-1.30, +4.69]` | `52.26%` / `9.55%` |
| `STABLE_SHUF vs R882` *(permuted-vote placebo)* | `+0.014821` | `[+0.001296, +0.036861]` | `1.46%` (`3/205`) | **`+2.47`** (`1502.47`) | `[-2.61, +7.44]` | `41.67%` / `17.99%` |
| `BAG_VOTE_UNGATED vs R882` *(vote without `zgap`)* | `+0.010239` | `[-0.008300, +0.035192]` | `4.88%` (`10/205`) | **`+4.51`** (`1504.51`) | `[-0.22, +9.82]` | `55.87%` / `19.01%` |
| **`STABLE_G025 vs G025`** *(paired)* | **`-0.003926`** | `[-0.010132, +0.000301]` | `5.37%` (`11/205`) | **`-1.14`** (`1498.86`) | `[-6.45, +4.83]` | — |
| **`STABLE_G025 vs STABLE_SHUF`** *(paired)* | **`-0.002319`** | `[-0.008419, +0.001724]` | `2.93%` (`6/205`) | **`-0.71`** (`1499.29`) | `[-5.41, +4.15]` | — |

- **Seed `233101` Switching & Vote Diagnostics**: Across the `1,219` `A`-fold contexts of seed `233101`, `zgap != 0` (the member-`0` blend switched `r882`'s argmin pick) on `38.06%` (`464/1,219`) of contexts, and the `3` outer `HistGBM` members were unanimous on `55.87%` (`681/1,219`) of contexts (`55.87%` for `STABLE_SHUF` by exact per-cell construction). Among switching contexts (`zgap > 0`), `G025` opened on `32.49%`, whereas `STABLE_G025` opened on only `9.55%` (requiring both `zgap <= tau_cell` and 3-member pick unanimity). That extra selectivity cut family harm 5-fold (`0.98%` vs `4.88%` for `G025`), but also reduced equal-family mean gain (`+0.012502` vs `+0.016429`, paired `-0.003926`, Elo lift `-1.14`) and did not beat the matched-rate random permutation `STABLE_SHUF` (`-0.002319`, Elo lift `-0.71`).

### 3.2 Seed `233101` `DL-Val91` Secondary (`19` Families, `91` Contexts, `5` Outer Cells — Descriptive Only)

| Arm / Paired Contrast (Seed `233101` only, `19` fams) | Equal-Family Native Gain | 10k Family-Bootstrap `CI95` | Family Harm (`< -0.01`) | **Elo Lift vs Incumbent (`R882` / Ref)** | **Elo Lift `CI95`** | Context-Mean Gain (`91` ctx) |
|---|---|---|---|---|---|---|
| `G025 vs R882` | `-0.000513` | `[-0.002103, +0.000882]` | `0.00%` (`0/19`) | **`+7.31`** (`1507.31`) | `[-21.95, +44.08]` | `-0.000536` `[-0.002221, +0.000930]` |
| `STABLE_G025 vs R882` | **`+0.000709`** | `[+0.000000, +0.002126]` | `0.00%` (`0/19`) | **`+7.31`** (`1507.31`) | `[+0.00, +21.95]` | `+0.000740` `[+0.000000, +0.002223]` |
| `STABLE_SHUF vs R882` | `-0.001158` | `[-0.002977, +0.000149]` | `5.26%` (`1/19`) | **`+0.00`** (`1500.00`) | `[-18.28, +21.95]` | `-0.001209` `[-0.003109, +0.000157]` |
| `BAG_VOTE_UNGATED vs R882` | `+0.001127` | `[-0.000810, +0.003394]` | `0.00%` (`0/19`) | **`-7.31`** (`1492.69`) | `[-36.67, +21.95]` | `+0.001177` `[-0.000863, +0.003538]` |
| **`STABLE_G025 vs G025`** *(paired)* | **`+0.001222`** | `[-0.000844, +0.004213]` | `0.00%` (`0/19`) | **`-7.31`** (`1492.69`) | `[+0.00, +0.00]` | — |
| **`STABLE_G025 vs STABLE_SHUF`** *(paired)* | **`+0.001867`** | `[-0.000149, +0.005144]` | `0.00%` (`0/19`) | **`+3.65`** (`1503.65`) | `[+0.00, +18.28]` | — |

### 3.3 Completed `FIT`-Calibrated Cell Summary (`5/5` Cells of Seed `233101` + Cell `0` of Seed `233102`)

| Cell (`seed/outer`) | `A` Fold (`ctx`) | `B` Fold | `FIT` Folds (`ctx`) | Calibration `[a, b]` | Selected `tau_cell` | `FIT` Gain | `FIT` Harm | `FIT` Open Rate | `A` Unanimous Vote Count (`Rate`) |
|---|---|---|---|---|---|---|---|---|---|
| `233101/0` | `0` (`295`) | `1` | `[2, 3, 4]` (`692`) | `[-1.125742, +0.349671]` | `0.001489` | `+0.001577` | `2.44%` | `70.95%` | `167/295` (`56.61%`) |
| `233101/1` | `1` (`232`) | `2` | `[0, 3, 4]` (`742`) | `[-1.059914, +0.495662]` | `0.007478` | `+0.032215` | `4.88%` | `72.64%` | `127/232` (`54.74%`) |
| `233101/2` | `2` (`245`) | `3` | `[0, 1, 4]` (`748`) | `[-0.941379, +0.369651]` | `0.005285` | `+0.029587` | `4.07%` | `76.74%` | `156/245` (`63.67%`) |
| `233101/3` | `3` (`226`) | `4` | `[0, 1, 2]` (`772`) | `[-0.934356, +0.734662]` | `0.008252` | `+0.004453` | `4.88%` | `76.55%` | `135/226` (`59.73%`) |
| `233101/4` | `4` (`221`) | `0` | `[1, 2, 3]` (`703`) | `[-0.990510, +0.761984]` | `0.008840` | `+0.012333` | `2.44%` | `73.12%` | `96/221` (`43.44%`) |
| `233102/0` | `0` (`233`) | `1` | `[2, 3, 4]` (`750`) | `[-0.975306, +0.605453]` | `0.001534` | `+0.003850` | `1.63%` | `67.33%` | `100/233` (`42.92%`) |

---

## 4. Red-Team Audit Summary

1. **`DOPE` Checkpoint Immutability**: `DOPE` (`c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814`) was verified before and after `TRAIN` smoke (`smoke-metrics.json`), `DL-Val91` baseline regression (`regression/dope-secondary.json`), and `TRAIN` baseline (`train/dope-train.json`).
2. **Zero Optimizer Updates & CPU-Only Execution**: `optimizer_updates = 0`, `GPU_used = false`, `CUDA_initialized = false`. Neither `DOPE` nor `r882` (`0084479aa0703667a0f3ccfe2beda60741c204795ec57b7c08542401ec6cdb64`) was trained or modified.
3. **Read Boundary & Split Isolation**:
   - `smoke-20261102T2225Z`: `0` `DL-Val` reads (`dl_val_contexts_read = 0`), `0` protected reads.
   - `run-20261102T2226Z`: `1,219` `DL-Train` contexts (`205` families); `91` `DL-Val` contexts (`19` families) read strictly for the mandatory `91/91` `r882` baseline regression check and outer-model feature predictions (`0` `DL-Val` outcomes used in any `HistGBM` fit, calibration, or `tau_cell` choice; `train_cache_dl_val_rows_decoded = 0`).
   - `sealed = 0`, `holdout = 0`, `meta_test = 0`, `live_loop = 0`, `protected_reads = 0`.
4. **Metric Direction**:
   - **Cost Space**: Lower predicted cost is better (`argmin` selects the candidate program with minimum predicted cost).
   - **Utility Space**: Higher flywheel native utility is better (`-fitness` for valid `gate == 1` programs, `-2.0` for invalid `gate == 0` programs; `R882` `DL-Val91` mean = `1.1377941898164525`).
   - **Elo Lift Space**: Higher Elo lift vs incumbent ($\text{Elo} - 1500.0$, anchored at `R882 = 1500.0`) is better.
   - **Harm Space**: Lower family harm fraction (share of families with mean native utility delta `< -0.01`) is better.

---

## 5. Verdict & Next Step

- **Verdict**: **`CLOSED INCOMPLETE`** (`44/75` subfits completed; hypothesis untested at preregistered 3-seed power; no efficacy verdict, no candidate adoption, no serving promotion).
- **Next Step**: Do not resume or retry under `H233`. Rerun the 3-member unanimity-vote hypothesis only if `xbabe2` swap and memory headroom allow (or with a pause-and-resume memory hysteresis guard rather than a hard abort at `MemAvailable < 20 GiB`), under a **separate new H number and new preregistration**. Even on the single completed seed (`233101`), `STABLE_G025` reduced `DL-Train` OOF family harm from `4.88%` to `0.98%` and edged `G025` on `DL-Val91` (`+0.000709` vs `-0.000513`), but trailed `G025` (`-0.003926`, Elo lift `-1.14`) and `STABLE_SHUF` (`-0.002319`, Elo lift `-0.71`) on `DL-Train` OOF mean gain because requiring 3-member argmin unanimity vetoed too many positive switches (`open_rate` among switching contexts dropped from `32.49%` to `9.55%`).

---

## 6. S3 Exact-Version Custody

- **S3 Prefix**: `s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261102/H233/` (`122` versioned objects, `150,906,204` bytes total).
- **Terminal Closeout Receipt**: `s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261102/H233/run-20261102T2226Z/incomplete-44of75-closeout.json`
  - `VersionId`: `0UdGa5gMryvEdgeWuoKlROS0zOnfu0pu`
  - `SHA256`: `8aecaf438c07901389932e7d2083beac083f1553d93549cddc20105f30326d26` (`20,257` bytes; restore-verified by `VersionId`).
- **Full S3 Custody Manifest**: `s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261102/H233/s3_manifest.json`
  - `VersionId`: `KgvWT1lj4JlUbs7KtzdX8kh65IwHDUq2`
  - `SHA256`: `c5363cc5f82aae0f69b405fa1202654d8d1d14ba740d35b60c27d1b9531bb3d4` (`58,828` bytes).
