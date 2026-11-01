# Frozen r782 custody inventory — RES-01

The frozen-source GPU dry run, manifest-only import replay and byte-identical S1 evaluator rebuild are verified. This inventory does not certify the R0 exporter, full M1/M2/M3 parity, model quality or serving promotion.

GOV-03 #573 pins r782 SHA256 `427b0d4608e94f60f5f97d79b5c1c4ce9a7efd59d17e7853f714e901f3a4af0c`. The new trace used immutable source from the preceding r776 lineage revision `07e03984de9a1a7be5b0c94bba166423aa21a549`, including its Python package/configuration, and the exact historical native telemetry extension SHA256 `93acd3087e04b5b26d6ff74a47752d463bd7104b3ad1c32b5642e992676bc76f`. Source files were checked against Git blobs, and Micro code/assets against read-only xbabe1 custody hashes. Today's changed deployed runner was not substituted.

## GPU dry run and imported closure

The original frozen main function ran with `--dry-run --resume round-782.pt`, all1219 training and91 development contexts, one step declared and batch size1219. It verified the prepared packet and replayed90 historical winner admissions (five strict promotions from existing scores). It performed **zero optimizer updates and zero new native fits**. Because original dry-run returns before checkpoint loading, the wrapper separately constructed the model on CUDA and strictly loaded r782 under the same lock. DOPE composite remained `c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814`.

The trace records1063 imported files, including transient imports. `rust_evaluator` resolved to `/home/ubuntu/JopeDimeMicro/python/rust_evaluator.py`, SHA256 `53422343699172b61ed69ed707015558f7cf1a344e5449f30e1eb9d708aef900`. The CPU-exact source, FusedLandmarkWorkspace, native HPC library, rank_prepare.py, feature-name order, champions, DOPE, checkpoint and constructor model/projection assets are all pinned. Each imported file lists its path, hash and observed origin. Git snapshots carry verified commits; unversioned custody/runtime files explicitly retain null commits and their exact hashes.

A fresh process restricted `sys.path` to manifest roots and enforced exact file paths/hashes for imports and direct loaders. The complete original dry run and strict CUDA load passed:1062 observed files, no extra file and no changed hash. Removing the required phase12 model source produced the expected outside-manifest ImportError. The smaller replay file count excludes the parent measurement launcher.

The launcher held the real exclusive `qualification.lock` throughout. Original main received a duplicate of the same locked open-file description and executed its unchanged flock call; replay subprocesses inherited that descriptor. This preserves exclusivity without a lock bypass. The successful lease ran19:01:22–19:02:19UTC on2026-10-30, then released in finally after CUDA cleanup and child exit. First-attempt failure against xbabe2's stale global telemetry extension is retained; the retry used private staged files without modifying shared installations or another job.

GPU replay environment: xbabe2 RTX3090, Python3.12.3, torch2.7.0, numpy1.26.4. The earlier xbabe1 CPU trace used torch2.6.0+cu124/numpy2.5.4 and remains separately retained. This is a new reproducibility trace, not a recovered training-time import log or a numerical parity result.

## Evaluator source and reproducible build

The S1 binary's embedded paths identify retained source revision `6bf65b12e9f0e5ec2697720a6119f13ea27d4835`. All170 actual source dependency inputs match that revision. The bounded Rust1.97.1 build reused this lane's target with nice10/jobs8 and completed in114.2s. Its output is **byte-identical** to custody SHA256 `e3be938644722d14a1c1a0e82168518838c4d174cfe031076a339745c48870a8`.

```sh
nice -n 10 cargo +1.97.1 build --offline --locked --release \
  --manifest-path /home/ubuntu/scratch/jopedime-dl-training-v1/build/s1-src-6bf65b12/deploy/lambda/training/teacher-v2/rust/lambda-worker/Cargo.toml \
  --target-dir /home/ubuntu/JopeDime/.agent/worktrees/hwv-04/target \
  --message-format=json-render-diagnostics
```

`CARGO_BUILD_JOBS=8`, `CARGO_INCREMENTAL=0`, `CARGO_PROFILE_DEV_DEBUG=line-tables-only`; no global CARGO_TARGET_DIR or Rustflags/release-profile override. The retained manifest specifies opt3, fat LTO, one codegen unit, stripped symbols and panic abort. The command was reconstructed and proven, rather than labeled a recovered historical build receipt. Native telemetry's build provenance is a separate known limit; its exact historical binary pin is preserved.

## Family roster and downstream use

MANIFEST.json attaches all205 training-family IDs (1219contexts) and the separate19 development-family IDs (91contexts), with no overlap. These supply the exclusion list for this training cohort to QLT-02. All prepared tables are train-only (`nq=0`), and original outer-query/holdout/sealed tables were not delivered. RES-02 can vendor/pin the relevant custody bytes; unknown Git origins are recorded honestly rather than guessed.

Evidence: `~/jope_plan/evidence/RES-01`, with receipt paths and SHA256 values in the manifest. Bulk inputs/runtime and dry-run outputs live outside Git under `~/jope-data/res-01`. No training service, global package installation or foreign checkout was changed.


## Bulk custody relocation (2026-11-01, REL-07 T1045)

RES-02 PR #641 placed four bulk binaries in the tracked custody tree: NumPy's native
extension (7,463,681 B), the frozen telemetry extension (66,754,912 B), the frozen
projection (101,734,579 B), and r782 checkpoint (7,493,476 B). Their total is
183,446,648 B. These existing payloads violate the repository's 5 MiB tracked-file
cap; the newer custody-addition guard correctly checks new additions but cannot
remove historical payloads. Neither cap is raised or exempted.

The four original hash/length pins stay in **unchanged MANIFEST.json**. EXTERNAL.json
identifies their existing immutable Git recovery commit, paths and object IDs.
All other vendored files and scientific records remain. Default verification streams
the four original Git blobs and checks every SHA-256 and length without restoring
bulk bytes into the working tree. Missing recovery history fails explicitly; there
is no network fetch or missing-input PASS. An explicit external cache supports
shallow/offline source copies:

```sh
python3 research/flywheel_selfplay/custody/verify_custody.py
python3 research/flywheel_selfplay/custody/verify_custody.py --external-root "$CUSTODY_CACHE"
```

The cache layout is `sha256/<original digest>` under the supplied absolute root.
Symlinks, missing payloads, hash/length changes, duplicate/changed external pins,
and unlisted vendored files are rejected. External input verification is part of
the original complete closure count; it is never a skipped subset. Existing Git
history is retained, not rewritten. To recover a payload, use the exact
`recovery_commit:git_path` in EXTERNAL.json with `git show`, write it only under
external custody, and verify its SHA-256 and length before use.

Before removal from the current tree, every payload was copied to
`xbabe1:/home/ubuntu/jope-data/custody/res08-t1045/sha256/`, independently restored
and checked under `/home/ubuntu/jope_plan/custody/res08-t1045/sha256/`, and compared
byte-for-byte by digest/length with its original immutable Git object. The recovery
receipt is `/home/ubuntu/jope_plan/evidence/REL-07/res08-t1045/INDEPENDENT-RECOVERY.json`.
These are custody operations, not model loading, training or qualification.
