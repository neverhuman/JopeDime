# HWV-01: H-wave Golden45 CPU preparation

[CPU-PREPARATION.json](CPU-PREPARATION.json) records the native baseline replay: all 304 ratings reproduce with maximum absolute Elo difference 0.0.

No H38 or r782 Golden45 rating is claimed yet. The existing Jope Dime Regressor
reference is **1687.0 Elo, rank 31, coverage 42/45**; it is a different entrant.

| Entrant | Rank | Elo | Scored datasets |
| --- | ---: | ---: | ---: |
| H38 F2_SMART_LM80_STD_SVD32 | Unrated | Unavailable | 0/45 (not attempted) |
| GOV-03 r782 | Unrated | Unavailable | 0/45 (not attempted) |
| Prior Jope Dime Regressor | 31 | 1687.0 | 42/45 |

Zero recorded candidate scores means execution is pending; it does not assign
zero fitness or count missing outcomes as losses. The preparation receipt lists
all 45 missing dataset IDs and the artifact/execution reason for each candidate.

## Design note

Freeze the 304-method, 45-dataset, 8,718-row benchmark-valid ladder and its 45
protocol-correction exclusions. Recompute through the existing `jope-arena elo`
implementation and compare the committed leaderboard plus every rating to 0.1 Elo.
Leave all existing rows and protocol files unchanged. Hash the three H38 artifacts
against HWV-02 custody and the r782 checkpoint against GOV-03. Preserve the loaded
DOPE composite `c51a56d0` (a tensor/normalizer identity, not a physical-file digest).
Record both candidates as unrated until their actual predictions and RMSE rows exist.

Subsequent qualification needs an executable H38 candidate with its fitted transforms,
and a pinned r782 inference/scoring closure. Selection may use only public-development
training inputs. Score frozen selections on the Golden45 public-development protocol;
never open DL-Holdout, sealed meta-test or any unrelated dataset. Use real S3-backed
inputs, retain missing/error outcomes, and publish rank, Elo and coverage separately
for regression. Binary is NOT_RUN. No candidate inherits another Jope method's rating.

Any GPU work must acquire both the owner-requested `~/qualification.lock` and the
existing shared `~/jope_plan/qualification.lock`, after the fused qualification and
H53b work. Check free VRAM before inference. No lock acquisition authorizes running
with an incomplete candidate/source/data identity. Use nice 10, four build/math
threads and back off above host load 85. This preparation runs CPU code only.

## Reproduce preparation

Run from the task branch, with `CARGO_TARGET_DIR` unset and the lane's reused target:

```sh
env CARGO_BUILD_JOBS=4 CARGO_INCREMENTAL=0 CARGO_PROFILE_DEV_DEBUG=line-tables-only \
  nice -n 10 research/elo/prepare-h-wave --repo "$PWD" \
  --checkpoint PATH_TO_GOV03_ROUND782_PT --output-dir NEW_EXTERNAL_RECEIPT_DIRECTORY
just hygiene
just lint-paths
```

`prepare-h-wave` creates reference CSV/JSON, native command output and `receipt.json`
outside the working tree. It checks the three H38 SHA-256 values from HWV-02, r782's
full-file hash, frozen registry/correction identities, and a native baseline Elo
recompute. It never opens dataset tables, imports a neural runtime or accesses CUDA.

The recovered H38 source keeps trained models in an in-memory dictionary. Its public
result/ledger artifacts are not a runnable model plus fitted normalization/projection
state; replaying the original whole runner would also access prohibited historical
holdout inputs. Do not run it wholesale or substitute an h45/fused model for H38.
GOV-03 fixes r782 (`427b0d4608e94f60f5f97d79b5c1c4ce9a7efd59d17e7853f714e901f3a4af0c`),
superseding the task title's stale generic h40b pointer.

This is preparation for HWV-01, not its completion. A complete result still requires
both candidates' real Golden45 rows, coverage at least 40/45 or listed missing reasons,
artifact-bound runtime receipts, and recomputation of the augmented ladder. No GPU
job has been started by this directory.
