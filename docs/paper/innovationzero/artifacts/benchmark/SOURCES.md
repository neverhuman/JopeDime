# assets/automl-elo/v1 — provenance

The four-entrant AutoML Elo of the native general regressors: `Opus5p5_Extra`,
`Gemini4_argon_high`, `GPT6p1_Sol_Max` and `GPT6_Astra_Max`. Every entrant implements the same
`RegressorApi` (`new(seed)`, `fit(train, n_rows, n_cols)` with the target last, and
`predict(features, n_rows, n_features)`). Each is fitted on the training rows of a cell and scored
on that cell's held-out rows only.

| Entrant | Elo | 95% CI |
| --- | --- | --- |
| Opus5p5_Extra | 1641 | 1599–1689 |
| Gemini4_argon_high | 1561 | 1519–1607 |
| GPT6p1_Sol_Max | 1456 | 1414–1497 |
| GPT6_Astra_Max | 1342 | 1289–1388 |

- Figures: [`automl-elo.svg`](../../../../../assets/automl-elo/v1/figures/automl-elo.svg),
  [`automl-pairwise.svg`](../../../../../assets/automl-elo/v1/figures/automl-pairwise.svg) and
  [`algorithm-source-size.svg`](../../../../../assets/automl-elo/v1/figures/algorithm-source-size.svg), each with a PNG beside it.
- Numbers: [`RESULTS.json`](RESULTS.json) holds the ratings, intervals and six pairwise tests.

## Panel and protocol

- **Panel.** 64 dataset families, each with 3 folds and 3 seeds (41, 73, 109), for 576 cells per
  entrant. Inputs are byte-identical across entrants and pinned by SHA-256.
- **Score per family and seed.** RMSE pooled over the 3 folds.
- **Match outcome per family.** The mean over the three seeds of a win (lower RMSE), loss or tie.
  A tie is equality within a relative 1e-12, and counts as 0.5.
- **Rating.** A Bradley-Terry Elo fitted by the unchanged `elo_core.rs` (centre 1500, 250
  iterations, regularization 1e-4). The interval is a 10,000-resample family bootstrap that reuses
  the same resample indices as the historical three-entrant analysis.
- **Pairwise tests.** Each pair's geometric-mean RMSE ratio, with a family-bootstrap 95% interval
  and p-value. The three Gemini comparisons are also Holm-adjusted.
- **Coverage.** Complete: every one of the 64 families × 6 pairs has a result
  (`complete_pairwise_coverage: true`), and no cell failed.

**Reading the results.** The Elo ranks how often an entrant wins a family, not the size of its
errors. None of Gemini4_argon_high's three RMSE-ratio comparisons is significant after Holm
adjustment (adjusted p = 0.19, 0.37, 0.37). Its geometric-mean RMSE is about 15% *higher* than
GPT6p1_Sol_Max's (ratio 0.85, CI 0.62–1.06), although it beats Sol on 66% of families. These are
64 families from one panel; nothing here generalises beyond it.

## How it was produced

### The three historical entrants: reused, not refitted

Their 1,728 fits come from the three-entrant run
`regression-flywheel-20261021`, `newonly-run` and `newonly-seeds73109`, on xbabe3:

- archive `s3://veox-jopedime-learning-use2/staging/regression-flywheel-20261021/w000003/sha256/48065da046d63dcf8202299a6953731c994f35c11b94e96ca14c779dc5bc162e.tar.gz`;
- summary: [`HISTORICAL_THREE_ENTRANT.json`](HISTORICAL_THREE_ENTRANT.json); pointer: [`HISTORICAL_ARCHIVE.json`](../../../../../assets/automl-elo/v1/HISTORICAL_ARCHIVE.json).

Before any Gemini cell was used, the fitter recomputed all 10,001 historical three-entrant fits
(the point estimate plus 10,000 bootstrap resamples). The largest absolute difference was 0.0
([`ELO_FITTER_QUALIFICATION.json`](../../../../../assets/automl-elo/v1/ELO_FITTER_QUALIFICATION.json)).

### Gemini4_argon_high: 576 new fits

Run on xbabe2, 2026-10-22, 12 cells at a time.

- **Isolation.** Each cell ran in its own `systemd-run --scope` with `AllowedCPUs` set to one CPU,
  `CPUQuota=100%`, `MemoryMax=8 GiB`, `MemorySwapMax=0` and `TasksMax=64`. This is the per-cell
  envelope of the historical run, and the worker refuses to start outside it.
- **Admission.** A new cell started only while at least 24 GiB of memory was available and the
  1-minute load was below 70% of the host's CPUs.
- **No time cutoff.** No fit or predict time limit applied, by owner ruling.
- **Source.** `crates/jope-automl/src/Gemini4_argon_high.rs`, byte-identical,
  SHA-256 `d5cd85fcf5e2bea8513d7c8665ba9f17b5fc4b70cd1570fdcedd569c376053d3`.
- **Worker.** The prepared four-model successor worker with only its module path and method name
  changed. Built with `rustc 1.97.1 --edition=2021 -C opt-level=3 -C codegen-units=1`; binary
  SHA-256 `534e6d8f163af9d70a077ff2bfc75b42cdb914186fb036a53371124974dc3b30`.
- **Scorer.** The historical run's exact `score` binary, SHA-256
  `4a42ab6599e5145eaf46e4b4481a0dff9386f7bb339c7787bf9085fc020a2b6d`.
- **Qualification.**
  - 3 cells completed before launch.
  - A repeated cell gave byte-identical predictions.
  - Every cell's cgroup readback showed `memory.max` 8589934592, `memory.swap.max` 0, `cpu.max`
    `100000 100000` and a single effective CPU.
  - Every scope was gone after its cell.
- **Outcome.** 576/576 completed: 0 algorithm failures, 0 OOM kills. Per-cell RMSE, R², timings,
  peak memory and prediction SHA-256 are in
  [`gemini4-argon-high-cells.csv`](gemini4-argon-high-cells.csv); totals in
  [`GEMINI_COSTS.json`](GEMINI_COSTS.json).
- **Timings.** Fit wall time is xbabe2's, and the historical entrants ran on xbabe3, so the times
  are indicative only. Gemini4_argon_high's 576 fits took 1,986 s in total (the largest 56 s),
  against 62 s for Astra, 72 s for Sol and 459 s for Opus.

### Analysis and figures

- The analysis is the prepared four-model `analyze_four.py` with three edits: the Gemini name, the
  cell manifest and the cell directory. It wrote `RESULTS.json` and `match-matrix.txt` (64 rows by
  the 6 pairs, in sorted-name order).
- The figures come from `render_gemini4.py` with matplotlib 3.10.8, using deterministic SVG
  metadata. Two renders were byte-identical. The pins are in
  [`FIGURE_VALIDATION.json`](../../../../../assets/automl-elo/v1/FIGURE_VALIDATION.json).
- Source sizes are counted with the same lexical counter as the earlier count, which reproduces the
  previous entrants' totals exactly ([`SOURCE_COUNTS.json`](SOURCE_COUNTS.json)).

Custody of the run, the per-cell receipts and these scripts is outside the repository:

| File under `/home/ubuntu/jope-evidence/gemini4-argon-high-20261022/` | SHA-256 |
| --- | --- |
| `run/source/worker.rs` | `7ba5e8c9781172be6000e5b89878009791dcd933169656db94e2693792624ae9` |
| `run/cell.sh` | `049aa2e4255312d5162a2930463e5dec6db881d3135065d88a6c9841cee26465` |
| `run/cell_inner.sh` | `2acbfa76dbe22217f2e4c2360f4caaa82b00ada7accddfb38f72d4eba3508ce8` |
| `run/cells.jsonl` (576 cells, inputs pinned) | `c6a3004458f5d25b34ff6be14553bf8f1833991a586ee11d21bf0c09e2ca7e23` |
| `analysis/analyze_gemini4.py` | `c34c746ef5ac28435e1d16bd0c7e5c9fbbb1e6fb7b0cec431a77fb509130abf6` |
| `analysis/elo_driver.rs` | `475401528fa92473c3e5a2dd3ccecd74951fea1d7b1b03e63c871cc82ab159c7` |
| `analysis/elo_core.rs` (unchanged historical core) | `cc9378902551695881e3af8727f3adbb4dbd9f4972e3e494ef60403863f16d9c` |
| `analysis/render_gemini4.py` | `457286e8b87553828a2a742ab73639e2021435c909fc308e27684c1c4e926fa8` |
| `analysis/lexical_counter.rs` | `fe0c077114f1138a5337a04cd860b00d46931bec9c460731dd3817775ec739ff` |

The cell protocol, worker template, missing-cell manifest, Elo driver and analysis were prepared by
the Codex four-model task (`four-model-20261022`). By owner decision on 2026-10-22,
`Gemini4_argon_high` replaced that task's earlier `gemini-argon-4` source as the Gemini entrant.

