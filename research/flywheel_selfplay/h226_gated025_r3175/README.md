# H226 — frozen `GATED_025` on the retained-best `r3175` base (`xbabe2` CPU, 0 updates)

**Status:** `INCOMPLETE` — **Verdict:** `INCONCLUSIVE` (`REFUSED_AT_R3175_REGRESSION_GATE`; primary gate `FAIL_R3175_NATIVE_ARCHIVE_PARITY`; plateau verdict `NOT_EVALUATED_GATE_REFUSED`).
**Owner:** `rd_argon` `[ARGON]`. **Assignment:** `~/nudges/hw_t1435_argon_h226.md` and `~/nudges/hw_t1530_argon_h226_deadline.md`.
**Preregistration:** [PREREG.md](PREREG.md) (`SHA256: fc89fa3264c66baf41e8ac07bbcd9e3d349c6eded238a734de341213da`), merged in PR `#1882` (`58e147e9ba2fda6da93ce3f52c9eccffa8e81ef5` $\to$ `9d7704b2cd1e5301378656b75f1a71c072f1f134`) before any execution.

The guarded smoke execution (`smoke-20261102` at merged source `9d7704b2cd1e5301378656b75f1a71c072f1f134`) passed the structural checkpoint audit and reproduced the `R882` `DL-Val91` baseline bit-for-bit (`91/91` exact, context mean `1.1377941898164525`), then **refused to proceed** at the mandatory `R3175` `DL-Val91` parity gate (`runner.py:619`):

```text
BASELINE PASS: 91/91 native r882 exact 1.1377941898164525
RuntimeError: R3175 91/91 native regression failed: got 1.1366829009431036, expected 1.1389572403748425
```

Per `hw_t1435` (*"Exact gates first: R882 DL-Val91 bit-exact 1.1377941898164525 and R3175 == 1.1389572403748425 (from adapter_admissions.jsonl). Refuse to proceed if either mismatches"*) and `hw_t1530` (*"If both exact regression gates (R882 1.1377941898164525, R3175 1.1389572403748425) pass ... No other change (seeds, folds, arms, thresholds frozen)"*), no HistGBM fit (`0/45`), stage-1 lock or `A`-fold scoring was executed. Because `eval_base_checkpoint` evaluated both frozen base checkpoints across all `91` `DL-Val` contexts (`19` families) and all `1,219` `DL-Train` contexts (`205` families) before line `619` checked the `R3175` `DL-Val91` mean, the saved base outputs (`smoke-20261102`) also establish two new diagnostic facts without any additional model forward:
1. **Batch shape (`b=1` vs `b<=8`) is eliminated as the cause of the [H201](../h201_strict_admission/README.md) `r3175` parity failure.** `h124.baseline` evaluates one context per forward (`b=1`) on CPU (`torch.set_num_threads(1)`) and produces the exact same `1.1366829009431036` `DL-Val91` mean and exact same `3.719329833984375e-05` cost inversion between pool indices `68` (`-2.077629327774048`) and `521` (`-2.077592134475708`) on context `25aff0bcd9fb3ea4dd28c6a71463b1e02d8293401f40a2e48e56c2b5a3ab54d8` as H201's `b<=8` CPU Engine.
2. **First `205`-family `DL-Train` (`1,219` contexts) evaluation of `R3175` vs `R882`.** On the `1,219` training contexts (`205` families), `R3175` flips the top-1 choice on `45/1,219` contexts (`3.69%`) and achieves **lower** native utility than `R882`: equal-family mean `1.381525` vs `1.385355` (gain **`-0.003831`**, `95%` family-bootstrap CI `[-0.013024, +0.002296]`, family harm `3.90%` (`8/205` families `< -0.01`), context mean `1.419535` vs `1.423073` (`-0.003538`)).

---

## Hypothesis and setup

- **Hypothesis (`hw_t1435`).** Applying the frozen [H217](../h217_gated025_confirm/README.md) `GATED_025` recipe (`BLEND_025` with $\lambda = 0.25$, gated by outcome-free `zgap` with a FIT-only threshold under a `5%` FIT harm cap) to the retained-best live flywheel checkpoint `round-3175.pt` (`G3175`, zero adapter updates) compounds with `r3175` on `205`-family pooled cross-fit (`G3175` vs `R882` `LCB95 > 0`, `3/3` seeds positive, harm $\le 10\%$) and clears the `+0.002` `DL-Val91` plateau margin (`> 1.1397942`) while also beating `R3175` alone on `DL-Val91` (`G3175 - R3175` `LCB95 > 0`).
- **Arms & Partition Plan.** Seven zero-update arms (`R882`, `G3175`, `R3175`, `G882`, `G3175_DIS`, `B3175`, `B882`) on the pinned H124/H139 real-S3 cohort (`205` `DL-Train` families / `1,219` contexts; `19` `DL-Val` families / `91` contexts; `0` sealed, holdout or meta-test reads). Seeds `226101–226103` with fresh family-grouped 5-fold partitions `226141–226143` pinned in [fold_manifest.json](fold_manifest.json) (max best-match Jaccard `0.1885 <= 0.20` across `H203`, `H210`, `H217`, `H218`, `H222–H225`).
- **Mandatory Pre-Fit Gates ([PREREG.md](PREREG.md)).**
  1. Structural equivalence of read-only copies `round-882.pt` (`0084479aa0703667a0f3ccfe2beda60741c204795ec57b7c08542401ec6cdb64`) and `round-3175.pt` (`1d6ea900ebed8d2a05c3b3f37576adc15aadd2b53c585bfd5cb71a103e9ff97b`), frozen `DOPE` `c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814` before and after every forward, `0` optimizer steps, and `torch.equal` on all non-`adapter_10tok.` tensors.
  2. `R882` `DL-Val91` bit-exact `91/91` match with `best_dev_checkpoint.json` (`1.1377941898164525`).
  3. `R3175` `DL-Val91` context mean `== 1.1389572403748425` (from `adapter_admissions.jsonl` round `3175`, within `1e-12`). Refuse to proceed if either check fails.

---

## Outcome and regression-gate refusal

| Gate | Requirement | Observed (`smoke-20261102`) | Status |
|---|---|---|---|
| **1. Structural & DOPE freeze** | `r882` (`0084479a…`), `r3175` (`1d6ea900…`), `DOPE` `c51a56d0…` before/after, non-adapter `torch.equal` | Exact SHA match; `165` non-adapter tensors `torch.equal`; `47` `adapter_10tok.` tensors (`29,211` params) differ; `DOPE` unchanged across all 4 forwards (`0` updates) | **PASS** |
| **2. `R882` `DL-Val91` parity** | `91/91` bit-exact match with `best_dev_checkpoint.json`, mean `1.1377941898164525` | `91/91` bit-exact (`1.1377941898164525`) | **PASS** |
| **3. `R3175` `DL-Val91` parity** | Context mean `== 1.1389572403748425` (`adapter_admissions.jsonl` round `3175`) | **`1.1366829009431036`** ($\Delta = -0.0022743394317389$, `90/91` match, `1/91` context argmin flip) | **FAIL (REFUSED)** |

### Single-context `R3175` parity discrepancy (`25aff0bc…`) and `DL-Val91` pick flips vs `R882`

Across the `91` `DL-Val` contexts (`19` families), `R3175` on `xbabe2` CPU (`b=1` single-context forward in `h124.baseline`, `torch.set_num_threads(1)`) changes the top-1 selected program relative to `R882` on **5 of 91 contexts** (`86/91` identical top-1 picks):

| Context SHA-256 (prefix) | Family ID (prefix) | `R882` pick | `R3175` CPU pick | `R882` utility | `R3175` CPU utility | $\Delta u$ (`R3175 - R882`) | `R3175` CPU cost at `R882` pick | `R3175` CPU cost at `R3175` pick | Cost gap (`pick882 - pick3175`) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `15e9e1e67bb6f9d4…` | `fam-469b72b7…` | `744` | `376` | `0.39453004` | `0.37248266` | `-0.02204738` | `-1.54761660` | `-1.54996407` | `+0.00234747` |
| **`25aff0bcd9fb3ea4…`** | **`fam-9f67f144…`** | **`521`** | **`68`** | **`0.49282787`** | **`0.28586298`** | **`-0.20696489`** | **`-2.07759213`** | **`-2.07762933`** | **`+0.00003719`** |
| `2d7f56bce09d1364…` | `fam-538e6406…` | `503` | `142` | `0.60510263` | `0.43459473` | `-0.17050789` | `-1.95986211` | `-1.96023309` | `+0.00037098` |
| `2e8f7388b3d3ac44…` | `fam-90a58676…` | `245` | `471` | `0.13571560` | `0.20987152` | `+0.07415591` | `-2.15971494` | `-2.16048503` | `+0.00077009` |
| `983bb1e5e396d436…` | `fam-769243ee…` | `831` | `328` | `0.26810405` | `0.49234101` | `+0.22423696` | `-2.07290959` | `-2.07314944` | `+0.00023985` |

- On the four non-`25aff0bc…` contexts (`15e9e1e6…`, `2d7f56bc…`, `2e8f7388…`, `983bb1e5…`), CPU `R3175` matches the live `xbabe1` CUDA journal (`best_dev_checkpoint.json`) bit-for-bit; their net utility sum is $+0.10583760 / 91 = +0.00116305$, which produced the archived live scalar $1.13779419 + 0.00116305 = 1.13895724$.
- On context `25aff0bcd9fb3ea4dd28c6a71463b1e02d8293401f40a2e48e56c2b5a3ab54d8`, `R882` separated pool index `521` (`-2.08025217`) from pool index `68` (`-2.07342577`) by `-0.00682640`. The `r3175` adapter shifted those two candidates to within **`3.719329833984375e-05`** (`-2.077592134475708` vs `-2.077629327774048`). On live `cuda:0`, index `521` remained the argmin; on CPU (`b=1` here and `b<=8` in H201), index `68` is the argmin, reducing `DL-Val91` utility by $0.20696489 / 91 = 0.00227434$ to `1.1366829009431036`.

---

## Descriptive base metrics and Elo (`smoke-20261102` saved outputs, 0 additional forwards)

The table below reports the post-refusal descriptive comparison of `R3175` vs `R882` from the saved `smoke-20261102` base cost files (`train-r882/r882-train-costs.npz`, `train-r3175/r882-train-costs.npz`, `regression-r882/r882-secondary-costs.npz`, `regression-r3175/r882-secondary-costs.npz`) using the preregistered [analysis.py](analysis.py) functions (`10,000`-draw paired family-cluster bootstrap, seeds `226150` / `226151` / `226152`). No `G3175`, `G882`, `G3175_DIS`, `B3175` or `B882` cross-fit scores exist because execution stopped at the preregistered gate before any HistGBM fit.

| Split & Endpoint | `R882` (reference) | `R3175` (CPU `b=1`) | Gain (`R3175 - R882`) | Paired family bootstrap `CI95` | Harm (`< -0.01`) | Top-1 pick flips | `R3175` Elo (`R882 = 1500`) `[CI95]` |
|---|---:|---:|---:|---|---:|---:|---|
| **`DL-Train` (`1,219` ctx, `205` fam) — Native equal-family** | `1.385355` | `1.381525` | **`-0.003831`** | `[-0.013024, +0.002296]` | `3.90%` (`8/205`) | `45 / 1,219` (`3.69%`) | `1501.87` `[1497.85, 1505.63]` |
| **`DL-Train` (`1,219` ctx, `205` fam) — Native context mean** | `1.423073` | `1.419535` | **`-0.003538`** | — | — | `45 / 1,219` (`44` $\Delta u \neq 0$) | — |
| **`DL-Train` (`1,219` ctx, `205` fam) — U4 equal-family** | `1.609212` | `1.609844` | `+0.000632` | `[-0.000710, +0.002100]` | `1.95%` (`4/205`) | — | `1501.86` `[1497.69, 1505.86]` |
| **`DL-Val91` (`91` ctx, `19` fam) — Native context mean** | `1.137794` | `1.136683` | **`-0.001111`** | `[-0.006667, +0.002907]` | — | `5 / 91` (`5.49%`) | — |
| **`DL-Val91` (`91` ctx, `19` fam) — Native equal-family** | `1.153615` | `1.152551` | **`-0.001064`** | `[-0.006667, +0.002907]` | `10.53%` (`2/19`) | `5 / 91` (`5.49%`) | `1496.35` `[1481.72, 1507.31]` |
| **`DL-Val91` (`91` ctx, `19` fam) — U4 equal-family** | `1.277656` | `1.277738` | `+0.000082` | `[-0.000378, +0.000637]` | `0.00%` (`0/19`) | — | — |

Both `R882` and `R3175` select `0` invalid (`-2.0`) programs on `DL-Val91` (`0/91`) and `3` invalid (`-2.0`) programs on `DL-Train` (`3/1,219`).

---

## Resource guard, verification and S3 custody

- **Resource guard (`smoke-20261102.resource_summary.json`).** Monitored on `xbabe2` CPU only (`nice -n 19`, worker CPU affinity `[0, 1, 3, 5]`, `4` cores max, single-thread inner math `OMP/MKL/OPENBLAS/torch = 1`, `CUDA_VISIBLE_DEVICES = ""`). Elapsed wall time `1,077.81 s`, including `300.78 s` paused cleanly by the `load5 > 60 -> resume <= 55` pause/resume guard (`reasons: []`). Peak process-tree RSS `1,024,901,120 B` (`977.42 MiB < 4 GiB` limit), peak tree cores `1.142 <= 4.0`, minimum `MemAvailable` `96.15 GiB >= 20 GiB`, minimum free disk `682.11 GB >= 300 GB`.
- **Local checks.** `31 passed` in [test_h226.py](test_h226.py) (`pytest`), `ruff check` and `ruff format --check` clean, `git diff --check` clean, and all five `jope-records` subcommands (`verify-source-inventory`, `verify-test-inventory`, `repo-layout`, `parity-check`, `paper-trace`) pass.
- **S3 custody ([custody.json](custody.json)).** All `18` artifacts (`H226_smoke.claim`, `launch.json`, `logs/smoke-20261102.*`, `smoke-20261102/execution-identity.json`, `smoke-20261102/metrics.json`, `smoke-20261102/data/split.json`, and all four `regression-*`/`train-*` cost `.npz` and `dope-*.json` receipts) are uploaded create-only (`IfNoneMatch='*'`) to `s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261102/H226/smoke-20261102/` and verified by versioned `HEAD` readback (`custody_self_version_id: SFcNUVISj0R_A3ES3zPu5LGuK0umRvtL` in [custody.json](custody.json)).

---

## Verdict and next step

- **Verdict:** `INCOMPLETE` / `INCONCLUSIVE` (`REFUSED_AT_R3175_REGRESSION_GATE`; `FAIL_R3175_NATIVE_ARCHIVE_PARITY`; `0/45` HistGBM fits executed; `0` optimizer updates).
- **Next step:** Do not promote or stack `GATED_025` on `round-3175.pt` under CPU replay: `round-3175.pt` fails CPU-vs-archive parity on `DL-Val91` (`1.1366829` vs `1.1389572`, `-0.001111` below `R882`) even at `b=1`, and degrades `205`-family `DL-Train` native utility by `-0.003831` (`[-0.013024, +0.002296]`) relative to `R882`. Keep `round-882.pt` (`R882`) + `GATED_025` ([H210](../h210_blend025_gated_retry/README.md) / [H217](../h217_gated025_confirm/README.md)) as the qualified frozen reference while candidate-generation headroom (`H227`) and strict-margin admission ([H201 SMA proposal](../h201_strict_admission/SMA_PROPOSAL.md)) are evaluated.
