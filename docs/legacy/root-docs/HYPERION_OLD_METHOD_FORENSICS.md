# Old Python Hyperion forensic receipt

Date: 2026-08-09 UTC  
Authority host: `xbabe1`  
Authority root: `/home/ubuntu/seed_data_cache/Hyperion_Entanglement`

## Finding

The January 100-sweep cell tensors were written by
`process_regression_embeddings_100.py`, not by either benchmark named during
the investigation. The writer's SHA-256 is
`b5f03a771c6b71650c6030294bd45bf4e833fd4102f8b54d4cc9427d91215b24`.
It imports `Hyperion_Entanglement_v6.py`, SHA-256
`44d8653c9984ba297b0a72af99e96bd6c60d1199d49994bbcecb09b1c7d9cc96`.

The surviving evidence does **not** support an honest claim of byte-exact
reproduction. The closest and most likely checkpoint is:

```
runs/hyperion_v6_20260127_083701/ckpt_10.pt
sha256 cce25e793f9d970bc9d6fbc4a1af9e2412a640712114f715afa95546369425c5
```

It loads into the surviving V6 source with `strict=False`, leaving 30 missing
and 33 unexpected keys. Missing keys include live `input_proj` parameters, so
part of the forward model is randomly initialized. The January writer seeds
only row and column permutations; it records neither model-initialization RNG
nor the global CUDA RNG used by padding. It was also edited after generation,
and no pre-edit source snapshot or RNG receipt survives.

The checkpoint's exact 523-tensor training architecture does survive as
`hyperion_v5_modules/model.py` (SHA-256
`31e095d483013689307da7a191ed54b0e3c1ffafdf66d9779c4fd467a3fd180a`),
but its four-argument forward signature is incompatible with the January
writer's explicit three-argument call. A pre-cache PYC proves the root V6
`input_proj` architecture already existed before tensor generation. Moreover,
the January 29 `benchmark_live.log` records this same missing/unexpected-key
failure and the `strict=False` fallback. The partial-load path is therefore
historical behavior, not an artifact of later source drift.

Accordingly, the generated `.vs.orig` files are a deterministic reconstruction
of the old Python semantics, not a claim that the lost January bytes were
recovered.

## Artifact distinction

- The January batch writer saved `*_4scales_100sweeps.npy` with shape
  `(100, 4, N, D)` and `*_encode_100sweeps.npy` with shape `(100, 128)`.
- `analyze_100_sweeps_autotune.py` is a later tensor consumer, not the writer.
- `catboost_sweep_lift_benchmark.py` is a July benchmark, not the writer.
- `reproduce_facebook_sweep_lift.py` is a July forensic runner. Its `.npz`
  files contain a generated stack plus configuration/benchmark receipts; they
  are not the January source artifacts.
- The older root-level `results.npz` and `inference_results.npz` have the V5
  inference schema (`prob`, `emb`, `X`, `errors`) and are a separate lineage.

Pinned SHA-256 values for the three requested scripts are:

| Script | SHA-256 |
| --- | --- |
| `analyze_100_sweeps_autotune.py` | `fcd4b169b5ec4339c275cd5aba9ca7dfa35803adc8cda4c5466388620adc6989` |
| `reproduce_facebook_sweep_lift.py` | `9974f998989a097d5f935e318c4cfbf6486714e9d61c4128ef35b846b90065e5` |
| `catboost_sweep_lift_benchmark.py` | `ca540106693292bdb5462a8181b309e207feb6f483bdfffbb1eb36b2add975e4` |

## Checkpoint screen

The investigation enumerated all 135 non-symlink checkpoint files older than
the cached Facebook artifact and evaluated both historical rank scales for each
checkpoint: 270 CUDA configurations in total. Each candidate used AMP, the old
windows `[3, 6, 9, 12]`, shifts `[(0, 0), (1, 1)]`, sweep seed 42, and the
original global padding RNG. Candidate sweep zero was compared with cached
Facebook sweep zero.

| Candidate | Rank scale | Mean absolute difference | Correlation |
| --- | ---: | ---: | ---: |
| `v6_083701/ckpt_10.pt` | 100 | 0.02448368 | 0.95134392 |
| `v6_083701/ckpt_5.pt` | 100 | 0.04156983 | 0.95217995 |
| `v6_085411/ckpt_10.pt` | 100 | 0.05968378 | 0.92180803 |

No candidate matched the cached array or its SHA-256. Refined checkpoints were
materially farther away. The cached Facebook tensor has SHA-256
`10c3fda1d9d6b50e740af2f0a75cc06c7de416925f4c03043688874a360f8150`.

The winning fingerprint was reproduced with the wrapper's declared global
seed 42: sweep-zero MAD `0.0244836839714`, RMSE `0.0292310584952`, and
correlation `0.951343924995`. A bounded seed screen from 0 through 63 also kept
42 best; seed 17 was the next material candidate. Seed 42 is therefore the
most accurate declared reconstruction seed in the tested range, but it is not
represented as recovered historical RNG state.

## Negative recovery audit

The audit searched filesystem/editor histories, Claude backups, local
application histories, temporary directories, archives, Docker metadata,
home/NAS replicas, shell and job histories, journal/accounting sources, and
available snapshot/deleted-inode mechanisms. It found no pre-edit writer,
pre-edit writer PYC, launch record, or saved RNG state. The checkpoint payload
contains only `cfg`, `model`, and `step`; a recursive audit found no CPU, CUDA,
generator, or RNG state. Together with the 270 failed tensor matches, this
rules out an honest byte-recovery claim from the surviving evidence.

## Frozen reconstruction recipe

The reconstruction implemented in
[`scripts/reconstruct_old_hyperion_vs.py`](../scripts/reconstruct_old_hyperion_vs.py)
pins and verifies all source/checkpoint hashes before GPU work. It then:

1. Reads the headerless `train.csv` and `test.csv`, removes the final target
   column before any encoder call, fills missing feature values with zero, and
   converts features to float32.
2. Concatenates feature rows in train-then-test order. This is intentionally
   transductive at the feature level: the historical benchmark encoded one full
   dataset before making its 80/20 model split. No labels enter Hyperion.
3. Applies per-column `(argsort(argsort(x)) + 1) / N * 100` normalization.
4. Runs 100 sweeps with row and column seed `42 + sweep * 777`, windows
   3/6/9/12, shifts `(0,0)` and `(1,1)`, CUDA AMP, and original global-RNG
   padding.
5. Averages the four scales within each sweep and then all 100 sweeps. This is
   the faithful `all100_mean` collapse. On the cached Facebook tensor it scored
   R2 0.30504 versus 0.28438 for the later auto-EWMA collapse and 0.14509 raw.
6. Splits rows back into train/test matrices and exclusively creates
   `train3.vs.orig` and `test3.vs.orig`; an existing path is never replaced.

The later per-sweep-percentile transform scored higher on the single Facebook
benchmark, but it is not the original collapse and is therefore not placed
under the `.orig` name.

Two complete CUDA generations are compared with
[`scripts/verify_old_hyperion_vs_replays.py`](../scripts/verify_old_hyperion_vs_replays.py).
The verifier requires complete 20-dataset manifests, matching source,
checkpoint, configuration, input, full-sweep and collapsed-cell identities,
then directly compares every byte of all 40 `.vs.orig` outputs. It refuses to
write a receipt if any identity or output differs.

## Installed cohort

The verified matrices are installed beneath
`sotarena/data/tabular-regression-v1/<dataset-id>/` under the distinct names
`train3.vs.orig` and `test3.vs.orig`. The copy used an ignore-existing policy
after a zero-existing-file precheck. The installation verifier re-read every
value, required finite values and exact shapes, and matched every file SHA-256
to the byte-identical replay.

The immutable evidence bundle is
[`manifests/hyperion-old-orig-20260809`](../manifests/hyperion-old-orig-20260809/):
it contains structured provenance, both complete generation manifests, the
40-file replay comparison, and the installed-file receipt.
