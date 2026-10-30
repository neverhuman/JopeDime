# Frozen r782 custody inventory — RES-01 progress

**BLOCKED: the original GPU-locked dry run and full historical import closure are not qualified.** This is a measured progress inventory, not an R0/M1 pass or serving promotion.

GOV-03 #573 pins r782 SHA256 `427b0d4608e94f60f5f97d79b5c1c4ce9a7efd59d17e7853f714e901f3a4af0c`. An isolated CPU process on xbabe1 used the actual flywheel interpreter (Python3.12.3, torch2.6.0+cu124), imported the deployed runner, initialized the native landmark workspace, constructed the CPU fused model and loaded the frozen checkpoint strictly. DOPE composite remained `c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814`. It ran no optimizer update or GPU work and did not call the original main function.

The trace records966 imported files, including a transient torch module absent from final `sys.modules`. The proposer/featurizer actually imported is `/home/ubuntu/JopeDimeMicro/python/rust_evaluator.py` SHA256 `53422343699172b61ed69ed707015558f7cf1a344e5449f30e1eb9d708aef900`. The Micro `rank_prepare.py`, feature-name order, FusedLandmarkWorkspace, native HPC library, CPU-exact source, champions, checkpoint, DOPE and constructor weight/projection assets are hash-pinned in MANIFEST.json. Installed runtime files carry their observed distribution/version and file hashes; unknown upstream commits remain null.

A second CPU process restricted import roots and enforced exact manifest paths/hashes for every import. It loaded r782 successfully with no unlisted file or changed hash. Removing the required phase12 model source caused the intended ImportError. This proves the recorded CPU import/load path, not the unavailable original locked dry run.

## Evaluator source and reproducible build

Embedded source paths identify retained source revision `6bf65b12e9f0e5ec2697720a6119f13ea27d4835`; the retained Rust toolchain is1.97.1. A bounded CPU-only build reused this lane's target, nice10/jobs8, and completed in114.2s. All170 actual source inputs listed by the generated dependency file match that revision, including the normalized grammar asset paths. The rebuilt binary is **byte-identical**, SHA256 `e3be938644722d14a1c1a0e82168518838c4d174cfe031076a339745c48870a8`. The complete command, environment controls, source hashes, compiler note and stdout/stderr are retained outside Git and hashed in the manifest. The command was reconstructed and then proven; it is not described as a recovered historical build receipt.

```sh
nice -n 10 cargo +1.97.1 build --offline --locked --release \
  --manifest-path /home/ubuntu/scratch/jopedime-dl-training-v1/build/s1-src-6bf65b12/deploy/lambda/training/teacher-v2/rust/lambda-worker/Cargo.toml \
  --target-dir /home/ubuntu/JopeDime/.agent/worktrees/hwv-04/target \
  --message-format=json-render-diagnostics
```

`CARGO_BUILD_JOBS=8`, `CARGO_INCREMENTAL=0`, `CARGO_PROFILE_DEV_DEBUG=line-tables-only`; no global CARGO_TARGET_DIR or Rustflags/release-profile override. The source manifest supplies release opt3, fat LTO, one codegen unit, stripped symbols and panic abort.

## Historical source boundary and remaining work

The r782 lineage's preceding round776 manifest names JopeDime revision `07e03984de9a1a7be5b0c94bba166423aa21a549`. Its four executed source hashes match recovered immutable Git blobs. The freeze guard also matches that revision. Today's deployed runner, promotion helper and incumbent helper differ; MANIFEST.json carries both sets explicitly and does not replace the historical identity with the current deployment. External Micro files are an unversioned custody deployment, so their current hashes are verified without inventing origin commits.

Both qualification locks were occupied during read-only nonblocking probes. Original main acquires the lock and queries CUDA before its `--dry-run` branch, which also returns before checkpoint loading. The CPU strict-load proof therefore cannot be labeled that original dry run. Complete the frozen-source locked dry run and historical import provenance before marking RES-01/R0 complete; leave the existing flywheel and other qualification jobs untouched.

The manifest attaches the full205 training-family IDs (1219contexts) and the separate19 development-family IDs (91contexts), with no overlap. These are metadata only and supply the exclusion list for this cohort to QLT-02. No training/heldout/outer-query arrays were opened by this inventory.

Evidence: `~/jope_plan/evidence/RES-01`; all receipt paths and SHA256 values are in MANIFEST.json. Local card gates and PR publication remain pending while the original dry-run requirement is blocked.
