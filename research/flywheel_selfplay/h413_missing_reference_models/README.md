# H413: the 5 missing reference models, scored once on dev91-audit-v1

**Status:** complete, 2026-11-07. Owner: `cl_fwriter` (Claude Opus 5.5).
**Owner order (Jepson, 2026-11-07):** "Please score the missing ones Mitra, TabDPT, etc.. (also, make sure theese get
saved in the right locaiton so we don't have to redo them... we really want these scores to live in S3."
Receipt: `~/jope_plan/ns/receipts/H413_owner_order.{txt,json}` (0444; model ids, spec sha256s and harness code sha256s
pinned before the dev91 run).

## What was missing
Five reference models had a dev91 score only under H257's two-fold split (`h257.*`), so they could not be placed in
the dev91 AUDIT ladder next to our GP stack, the other foundation models and the frontier LLM regressors. H413 scored
each of them **once** on the 91 `dev91-audit-v1` contexts with this board's harness. Nothing else was run or rescored.

| New id | Model | Weights (same file as H257) | Settings |
| --- | --- | --- | --- |
| `kumo_tabular_small_e8.v1` | Kumo-Tabular small, 8 estimators (NVIDIA structured-data-models) | `small/regressor.pt` `fc672f53…`, HF `nvidia/Kumo-Tabular` revision v1.0.1 | the `kumo_tabular_large_e8.v1` path with size small; `torch.manual_seed(seed)` first, because sdm draws ensemble permutations from the entropy-seeded torch RNG |
| `tabpfn_v3p5_e1.v1` | TabPFN 3.5, 1 estimator | the `tabpfn_v3p5.v1` file | tabpfn 9.2.0, `n_estimators=1` |
| `tabicl_v2.v1` | TabICL v2 regressor | `tabicl-regressor-v2-20260212.ckpt` | tabicl 2.2.0 (H257's source commit), vendor defaults, no download |
| `tabdpt_1p3.v1` | TabDPT 1.3 regressor | `tabdpt1_3.safetensors` `97dc3b60…` | tabdpt 1.3.1 (H257's source commit), vendor defaults (`n_ensembles` 8); flash attention and compile are CUDA-only, so off |
| `mitra_v2_zeroshot.v1` | Mitra v2 regressor, zero-shot | `autogluon/mitra-regressor-2` snapshot `246c7d31…` (`model.safetensors` `8c90b8f2…`) | H257's `_pred_ofm` + `Mitra.run` decode ported line for line (1000-bin head over `linspace(-0.5, 1.5, 1001)`, no fine-tune, no bagging); CPU float32 where H257 used CUDA bf16 |

Every spec is pinned in [`models.json`](../../elo/reference-board/models.json). `tools/score_reference.py check`
refuses version or weights drift.

**Mitra history.** The first Mitra attempt (`mitra_v2.v1`, AutoGluon 1.5.0 `MitraRegressor`) failed 5/5 TRAIN smoke
contexts: AutoGluon 1.5.0 cannot decode the v2 checkpoint's 1000-bin head. It was removed before any dev91 run and has
no board row. The zero-shot port above is the only working v2 decode.

## How it was run
- **Harness:** `tools/score_reference.py` (sha256 `ae9a8f1f…`), `tools/run_board.sh`, `tools/sandbox_refboard.sh`
  (bwrap; the new venvs `venv-icl` and `venv-mitra` are mounted read-only). One fit per (model, context): SEARCH rows →
  AUDIT rows, u = −ln(RMSE_audit / RMSE of the SEARCH mean); failures are final.
- **Host:** xbabe2 CPU (GPU0 unavailable), `taskset 38-53,102-117`, nice 19, 4 jobs × 8 threads.
- **Run window:** 2026-11-07 17:13:52Z – 18:07:46Z (unit `h413-dev91`, exit 0).
- **Smoke first (TRAIN only):**
  - all 5 models complete on the refboard TRAIN smoke boards;
  - Kumo-S and the Mitra port give identical predictions on two copies of the 5-context board in separate processes
    (cross-process determinism). Unseeded Kumo does not.
- **Result:** 455/455 records `completed`, no failure or timeout.
  - One spec and harness identity per model, equal to the receipt.
  - Every `preds_sha256` matches its file (455/455).
  - The 25 committed score files are byte-unchanged.

| Id | Median wall (s) | Max wall (s) | Peak RSS (MiB) | Sum of wall (h) |
| --- | ---: | ---: | ---: | ---: |
| `tabpfn_v3p5_e1.v1` | 10.5 | 18.7 | 2,863 | 0.27 |
| `kumo_tabular_small_e8.v1` | 19.2 | 105.9 | 1,369 | 0.70 |
| `tabicl_v2.v1` | 21.2 | 74.5 | 3,639 | 0.62 |
| `tabdpt_1p3.v1` | 18.0 | 22.0 | 806 | 0.44 |
| `mitra_v2_zeroshot.v1` | 37.7 | 243.0 | 4,453 | 1.37 |

## Result: the dev91-audit-v1 ladder, 30 entrants
`tools/ladder.py` (anchor `h335.H257_REPLICA` = 1500, family bootstrap 95% CI); full table in
[`LADDER.md`](../../elo/reference-board/boards/dev91-audit-v1/LADDER.md).

| Rank | Entrant | AUDIT-Elo | 95% CI | Family-mean u |
| ---: | --- | ---: | --- | ---: |
| 1 | `kumo_tabular_large_e8.v1` | 2059.5 | 1883–2321 | 1.9356 |
| 2 | `tabfm_ensemble.v1` | 1899.7 | 1742–2112 | 1.7675 |
| **3** | **`kumo_tabular_small_e8.v1`** | **1888.6** | 1734–2112 | 1.7855 |
| 4 | `tabfm.v1` | 1874.4 | 1717–2084 | 1.7202 |
| 5 | `tabpfn_v3p5.v1` | 1838.6 | 1670–2081 | 1.6411 |
| **6** | **`tabpfn_v3p5_e1.v1`** | **1825.9** | 1653–2060 | 1.5887 |
| 7 | `tabpfn_v3.v1` | 1781.8 | 1650–1951 | 1.6427 |
| **8** | **`mitra_v2_zeroshot.v1`** | **1715.0** | 1563–1910 | 1.5596 |
| 9 | `h364.XSTACK_2L` (ours) | 1706.3 | 1621–1824 | 1.5607 |
| **10** | **`tabicl_v2.v1`** | **1698.6** | 1555–1878 | 1.4878 |
| 11 | `h335.H335_AGG` (ours) | 1689.5 | 1609–1801 | 1.5592 |
| **12** | **`tabdpt_1p3.v1`** | **1683.7** | 1529–1877 | 1.5179 |
| … | 18 more (frontier LLMs, AutoGluon, classic learners, other H335 rows) | | | |

- Ratings of the 25 earlier entrants moved only because the field changed. Their family-mean u are unchanged, as are
  their score files.
- **Our best model (`h364.XSTACK_2L`) moves from rank 6 to rank 9.** Only Kumo-S beats it by the H335 primary test
  (+20.1% RMSE cut, 16 of 19 families). TabPFN 3.5 (1 estimator) is ahead without passing. Mitra ranks one place
  above it on pairwise family wins (12 of 19) with an equal mean u. TabICL and TabDPT rank below it. All four are
  level with it within the Elo intervals.

**Head to head vs our `h364.XSTACK_2L`** (from `tools/ladder.py --focus <id>`; gain = family-equal mean of u(model) − u(XSTACK_2L); RMSE cut = 1 − exp(−gain); clipped T / sign-flip p / LOFO min = the H335 primary procedure; pass = p < 0.025 and LOFO > 0):

| Model | Gain | RMSE cut | Families W/T/L | Sign-flip p | LOFO min | Passes |
| --- | ---: | ---: | --- | ---: | ---: | --- |
| `kumo_tabular_small_e8.v1` | +0.2248 | +20.1% | 16/0/3 | 0.009138 | +0.0948 | yes |
| `tabpfn_v3p5_e1.v1` | +0.0280 | +2.8% | 15/0/4 | 0.2026 | +0.0261 | no |
| `mitra_v2_zeroshot.v1` | -0.0012 | -0.1% | 12/0/7 | 0.3613 | -0.0068 | no |
| `tabicl_v2.v1` | -0.0729 | -7.6% | 10/0/9 | 0.6243 | -0.0387 | no |
| `tabdpt_1p3.v1` | -0.0428 | -4.4% | 12/0/7 | 0.4587 | -0.0172 | no |

The same table is in [`h2h_xstack.md`](h2h_xstack.md). Each model's full ladder view against every entrant is in
[`focus/LADDER_focus_<id>.md`](focus/) (`tools/ladder.py <board root> --anchor h335.H257_REPLICA --focus <id>`).

## Where the scores are
- **git:** `research/elo/reference-board/boards/dev91-audit-v1/scores/<id>.jsonl` (one line per context), plus
  `LADDER.md`, `ladder.json`, the README section 1 rows and `SCORE_LOCATIONS.json`.
- **S3 (create-only, exact-version GET and member check):** the board archive in
  [`ARCHIVE.json`](../../elo/reference-board/boards/dev91-audit-v1/ARCHIVE.json). It holds all 30 score files and the
  AUDIT prediction files `preds/<model>/<cid>.f64`, including the 455 new ones.
- **Host:** the board root `xbabe2:~/jope-data/h335-stagev/refboard/dev91-audit-v1/`.
- **Run record:** the H413 closeout bundle (TRAIN smoke boards, stderr logs, receipts, scripts and logs):
  [`CLOSEOUT.json`](CLOSEOUT.json).
- Never rerun these ids: read the stored rows (board rule 1). A changed library, weights file, config, seed or device
  is a new id (rule 2).

## Not done
- No model of ours, no earlier reference row and no frontier row was run or rescored.
- Sealed / holdout / DL-Test stay closed; DOPE `c51a56d0` is not touched.
