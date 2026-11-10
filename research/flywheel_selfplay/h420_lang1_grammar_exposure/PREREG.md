# H420 PREREG — language exposure: what the worker runs and the grammar never emits

**Status.** Published on `origin/v1/next` before any H420 search run or deploy on a scored panel and before any
H420 AUDIT value is read. §8 lists what existed before this PREREG. The readout code (§6) is published with it.

- **Owner:** `cl_apex` (Claude Fable 5, xbabe2 session 947340e7). Claim: `ns/receipts/H420_claim.txt`.
- **Authority:** owner order, 2026-11-10 ("improve the GP language"; "run on the full compute of xbabe0–3").
- **Hard constraints:**
  - No dev91, DL-Val, DL-Holdout or sealed byte is read. The APEX `CONFIRM` cohort is neither deployed nor read.
  - No Rust change. The frozen H317–H417 code bytes are unchanged; `lang_h420.install` wraps six functions of the
    loaded grammar module at run time.
  - No stored entrant is rerun or rescored. Commits go to `v1/next` only.

## 1. Question
The superset worker `2a848c8f` accepts far more than the search grammar generates (worker source `feat-math`:
`spec.rs`, `gpr.rs`, `ops.rs`, `math_expr.rs`). Does exposing that surface improve the stored `h416.SXB` recipe on
fresh families?

| Surface | Stock grammar | Exposed in H420 |
| --- | --- | --- |
| `gpr` kernel | `rbf`, `rbf+linear` | `rq`, `rq+linear`, `rbf+rq`, `rbf*linear`, `rbf+periodic`, `rbf*periodic`, `rq*linear`, `rbf+rbf` |
| `gpr` steps / inducing rows `m` | 20 or 40 / default 500 | up to 160 / up to 2,000 |
| `krr` `m` | 200 or 400 | up to 2,000 |
| `histgbm` | depth 2–6, rounds ≤ 280 | depth to 8, rounds to 500, `min_leaf`, `lambda`, `bins` |
| `histgbm2` | leaves ≤ 31, rounds × bags ≤ 1,200 | leaves to 127, `min_leaf`, `col_subsample` 0.5, rounds × bags ≤ 2,400 |
| tree heads | trees ≤ 150; oblivious depth ≤ 6, rounds ≤ 500 | trees to 300; depth to 8, rounds to 1,200, `subsample`, `rsm` |
| `ksmooth`, `ensemble` | narrow | `k` to 256; wider depth and rounds |
| stages | 25 operators | plus `norm_row`, `row_diff_stats`, `row_spectral`, `fft_mag`, `polar`, `step`, and symbolic `map` / `derive` with parenthesis-free expressions (`x-row_mean`, `x/row_l2`, `c3*c7`, `c1/c4`, …) |

Two delivery routes are tested, alone and together:
1. **GX, an extended search run.** H417's GL run (H415-L screener, seed 417101, 50 generations, 6 live of 48
   screened) with the extended generators on: each generated head comes from the extended ranges with probability
   0.5, each generated stage is replaced by a new stage of its group with probability 0.2, and parameter mutation
   uses the wider domains. Seeds, budgets, screener and evaluation limit are H417's.
2. **LIB_EXT, 20 fixed library presets** (`lang_h420.LIB_EXT`: ten `gpr`, two `krr`, two `ksmooth`, two `histgbm`,
   three stage-plus-tree, one spline ridge), added to the 20 presets of the stored library.

One rule added after the probe (§8): a `periodic` kernel primitive becomes `rq` on tables wider than 64 columns. It
applies to the search run and the deploy alike. Stock programs never carry that primitive.

## 2. Design
All variants are the stored H416 `SXB` deploy (`deploy_h416.h416_unit`, mode confirm, design `BCAR.mix`, sha256
`0f0c96ad…f8c4`), three pooled search runs plus library, on H417's `BASE` panel (core `DEV` 115 and core `FIT-SEL`
106 contexts, one per family).

| Variant | Search runs | Library | Compares |
| --- | --- | --- | --- |
| `B` | GS, GE, GL (H417's stored runs) | stock | same-host replica of H417's deploy |
| `TGX` | GS, GE, **GX** | stock | replace one run, equal live evaluations |
| `TLIB` | GS, GE, GL | stock + LIB_EXT | 20 more fixed presets |
| `TBOTH` | GS, GE, **GX** | stock + LIB_EXT | both |

- **Hosts.** The deploys of `DEV` run on xbabe2 (bubblewrap sandbox) and those of `FIT-SEL` on xbabe3 (docker
  sandbox), because xbabe0 is full with H417. All four variants of a panel run on one host in one run directory,
  one after the other, sharing one evaluation memo (GS + GE + GL + GX concatenated). The paired comparisons are
  therefore same-host. `B` against H417's xbabe0 values is reported as a host replicate, not tested.
- **GX search hosts.** GX runs on xbabe2 for `DEV` and on xbabe1 for `FIT-SEL` (docker sandbox; its run directory
  is then copied to xbabe3). GS, GE and GL ran on xbabe0. All four hosts have the same CPU model and worker binary;
  a busier host can only cost GX evaluations through the 60 s limit, which works against the treatment.
- **xbabe3 note.** The owner's CI rule pauses the sandbox while a GitHub runner container is busy. An evaluation
  in flight during a pause can time out. This affects all variants alike; pause counts are reported.
- **Replay.** The deploy replays each search run from the memo. GX replays with the extended generators on in its
  own thread; GS, GE and GL replay with the stock generators. `train_h420.py replaycheck` runs first on 4 contexts
  and must show 0 worker runs for every source.

## 3. Tests (fixed)
Entrant: `P3+LIB3.SXB` of each variant. Value: AUDIT u = −ln(RMSE / RMSE of the SEARCH-mean predictor). Cohort:
core `DEV`, 115 families. Paired by context against variant `B`.

- **Primary:** `TBOTH` versus `B`, `h335_stats.primary_test`: family-mean gain clipped at ±0.5 above 0, sign-flip
  p < 0.025 one-sided, leave-one-family-out minimum above 0, median ≥ 0.
- **Secondary (Holm at 0.025 over two):** `TGX` versus `B`; `TLIB` versus `B`. Same test.
- **Reported, not tested:** the same three comparisons on `FIT-SEL`; `KFB` of each variant; family-bootstrap ΔElo;
  how often an extended program or library preset is a stack member; timeouts by head; worker-seconds per variant.

**Verdicts.** `LANG_CONFIRMED` if the primary passes. `LANG_PARTIAL` if only a secondary passes. `LANG_NULL`
otherwise. A passing route enters the champion recipe of the later champion wave; a null route does not.

**Compute.** `TGX` replaces a run at equal live evaluations. `LIB_EXT` adds 20 programs per context to about 900
search evaluations and the fold fits of those it selects; worker-seconds are reported per variant. No "more compute
only" control is run for the library route, since the stored deploy takes three sources.

## 4. Detectable effect
With 115 families, the test detects a clipped family-mean gain of roughly +0.02 u (2% RMSE) or more. A smaller true
gain will read as null.

## 5. Integrity gates (each must hold or the run is void)
- Guard v2 installed, 0 refusals; `same_family_bank_hits` = 0; APEX self-check passes; manifest sha256 pinned.
- `lang_h420.selfcheck`: with wrappers installed and generation off, 3,000 generated programs equal the stock stream.
- Every arm has exactly one stage record per context before its memo is concatenated; memo sha256 manifests kept.
- `replaycheck` shows 0 worker runs per source.
- Pub stage only: the staged PREREG equals the published one; code sha256 recorded in every run config.

## 6. Readout
`readout_h420.py` implements §3 on the pulled, sha-compared run directories (paired tests against `B`, Holm for the
secondaries, family-bootstrap ΔElo, worker-seconds, host replicate against H417). It is published with this PREREG.

## 7. Resources
- xbabe2: cpus 32–63, 96–127, nice 19. xbabe3: cpus 32–63, 96–127, nice 19, paused under the CI rule.
- xbabe1 (GX search for `FIT-SEL` only): cpus 19–50, 83–114, nice 19, paused while its runner containers use more
  than 8 cores.
- Compute deadline 2026-11-13T06:00Z. Disk on xbabe3 stays above 60 GB free.

## 8. What existed before this PREREG
- H417's PREREG, cohort manifest and BASE search runs (running on xbabe0; no H417 AUDIT value has been read).
- Worker-acceptance probes on two DL-Train contexts (`H420_SMOKE_P2`, `H420_SMOKE_P3`; SEARCH evaluation only):
  55 of 55 programs completed on a 12-column table; on a 365-column table 27 of 28 completed and `gpr_per` timed
  out, which led to the periodic rule in §1. I read status, seconds and the gate flag, no utility.
- A two-context smoke on panel `SMK2` (`H420_SMOKE_A`), status fields only: the GX run finished both contexts
  (334 worker runs each, 1 timeout); the replay check returned 644 results per source with 0 worker runs for GS,
  GE and GX. The first deploy attempt stopped before any evaluation on a mistyped design checksum, since fixed.
  The smoke also showed that the bubblewrap launcher left math-library threading unpinned; it now sets one thread,
  as the docker launcher does.

## 9. Files
| File | Role |
| --- | --- |
| `lang_h420.py` | extended generators, library presets, wrappers, self-check |
| `gp_h420.py` | GX search run; worker-acceptance probe |
| `train_h420.py` | deploy variants, replay check |
| `readout_h420.py` | paired readout of §3 |
| `run_h420.sh`, `run_h420_local.sh`, `h420_chain.sh`, `stage_h420.sh` | orchestration and staging |
