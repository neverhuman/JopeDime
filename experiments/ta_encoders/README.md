# TA encoders: TA_Hyperion_v8 and TA_Dope_V2 (target-aware)

Owner order (Jepson, 2026-10-07 2:15pm MT): build label-aware Hyperion v7 and Dope encoders from row-level data, train on
all admitted S3 row data across the GPUs, and track the GP-HWM probe correlation over training. This lifts the DL-training
hold for these two models only. The paused production DL arm is untouched.

## Why
The GP-HWM probe (experiments/gp_hwm_probe, 2,689 contexts, 463 families) put foundation at R² 0.476 and tabpfn35 at about
0.38–0.42, with hyperion_v7 and dope near 0. Hyperion v7 pools X-only row embeddings. Dope v1 sees the label only marginally
(target moments and per-feature |Pearson| in its 856-d sketch, then a frozen router). GP fitness depends on how y depends on X.

## Data (train-only)
- Source: `inputs/jtt1-encoder-s0/` in the MAIN v2-trainonly tree. These are stripped JTT1 payloads with n_test == 0.
  Contexts come from the manifests' `jtt1-encoder-s0` objects; every GET uses an exact VersionId and is sha256-verified.
- Admitted: 4,300 contexts from 865 datasets, 6,277,962 outer-train rows. G3 is excluded by context id (research-only
  rights, no row-data ruling), but none of the G3 contexts have s0 payloads anyway.
- Not used: unstripped `inputs/jtt1-encoder/` (v1, which includes outer-test blocks) and the local catalog trees, which are
  unstripped and unpublished. See the run report.
- Per context: train_view slices only the outer-train byte range, then the same target-copy drop as the v7 encoder.
  Contexts wider than 64 features get an XGBoost gain-importance order fit on outer-train rows (seed 20261007; 148 contexts),
  capped at 256 stored features. Up to 2,048 rows are stored, as a fixed-seed shuffle. Every reduction is recorded in index.json.

## TA_Hyperion_v8 (`ta_hv7_rows.py`, `ta_hyperion_v8.py`)
1. Frozen, pinned Hyperion v7 trunk (strict load, weights sha 05c17edd) gives per-row 512-d embeddings. It uses the v7
   recipe: per-column train q01/q99 quantisation, all six window sizes, cell mean, train-cell min-max, then the row mean.
   Two cuts were made for throughput: 1 sweep (production uses 3) and inputs capped at 512 rows × 64 features. fp16 autocast.
2. New target-aware head (about 6M params):
   - Each row token is the v7 row embedding plus a label channel Emb([y, 1]). Query rows get a [MASK] token instead.
   - A 6-layer transformer where keys are support rows only, so query labels can never enter.
   - The context embedding (1024) is built from label-conditioned attention pooling (8 seeds) plus per-dimension corr(z, y),
     cov(z, ridge residual) and the in-context ridge LOO R².
   - It also outputs row-level states.

## TA_Dope_V2 (`ta_dope_v2.py`, Dope repo)
- Per-feature tokens:
  - Dope v1's 12 per-feature stats.
  - 9 target-aware stats: signed Pearson, Spearman, eta² over 8 bins, corr(x², y), R², ridge coefficient and |coefficient|,
    drop-one gain, and rank slope.
  - A learned DeepSets sketch of the (x, y) pairs.
- Global token: Dope v1's target block plus learnability stats (ridge LOO R², 5-NN LOO R², residual shape, |corr|/eta² summaries).
- A 4-layer set transformer, then CLS plus 4 pooling seeds, gives a 1024-d context embedding.
- A row head (per-cell MLP pooled over features) provides the row embeddings and the masked-y predictions.
- Training caps each context at 128 features.

## Objective (self-supervised; GP-HWM labels never used in training)
- Each step samples 64–384 outer-train rows per context. 25% become query rows (y hidden). The rest are split into two
  disjoint support halves.
- Loss = masked-y MSE of the queries from each half (y standardised with that half's own support stats), plus
  0.1 × InfoNCE between the two halves' context embeddings.
- 4% of datasets (crc32 % 25) are held out for the SSL validation MSE.

## Leak controls
- `ta_leak_test.py` runs before training and must pass:
  - T1: appending a synthetic outer-test block leaves the decoded train view byte-identical.
  - T2: a must-fail wrong decoder does see the synthetic block.
  - T3: unstripped payloads are refused.
  - T4: changing query-row targets leaves every model output bit-identical, while changing support targets changes them.
- The checkpoint probe adds two controls:
  - TA_yshuf: the same model encodes the contexts with y shuffled inside each context, which removes the X→y signal.
  - SHUF:TA: GP labels are permuted inside each outer-train fold.

## Checkpoint evaluation (`ta_probe.py`, `ta_ridge.py`)
- Runs at step 0 and then every 20 minutes. Each run encodes the 2,689 probe contexts (support = the first ≤384 stored rows)
  and runs the fast-sweep ridge estimator verbatim: same units, target, family-grouped folds (seed 1000 + r, rep 0), inner
  CV, and family bootstrap.
- Arms: TA, TA+F, TA_yshuf, SHUF:TA, plus the references F and B.
- Output goes to `timeseries.jsonl`. The dashboard section is `ta_dash.py` (TA training).

## Compute
- TA_Hyperion_v8 trains on xbabe1 (RTX 4090), which also built the v7 row cache. TA_Dope_V2 trains on xbabe3 (RTX 3090).
  Each host trains its model in-process with bf16 autocast. There's no DDP: the models are small and the hosts are only
  linked over ethernet.
- xbabe2's GPU is left alone: it hosts the determinism pilot, the B-b06 encode queue has priority, and its IO pressure is
  already above 15%.
- `ta_common.Guard` pauses when host RAM drops below 17% or GPU free memory below 2.5 GB, and caps our duty cycle at 35%
  while another process is busy on the GPU. It never signals any other process.
