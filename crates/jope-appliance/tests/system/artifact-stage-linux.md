# INT-02 Linux artifact publication stages

Recorded source: `6adc02e3cf3a60b90dbcebbde277c516931504bb` (the history-preserving BackendPlan/Continue integration).
[artifact-stage-linux.json](artifact-stage-linux.json), SHA-256 `aa84c79cdcfce977c88a55bdb2e1096854b30accf6fc5c31ad835f36f4631d6d`, contains the generated receipt. This evidence-only addition changes no executable or test source from that commit.

The actual Linux x86-64 background-engine matrix passed R4, S2 and61 SIGKILL cases at11 hooks. Its45 CAS-publication cases cover all9 traced occurrences of each stage:

| Hook | Observed occurrences | Recovery checked |
|---|---:|---|
| `after_write` |9|Staging cleanup, committed prefix and continuation|
| `after_file_sync` |9|Staging cleanup, committed prefix and continuation|
| `after_hash_verify` |9|Staging cleanup, committed prefix and continuation|
| `after_rename` |9|Reference verification, committed prefix and continuation|
| `after_directory_sync` |9|Reference verification, committed prefix and continuation|

Every selected cut was an owning supervisor's actual SIGKILL (`-9`), with recovery followed by a completed continuation, contiguous sequences and the preserved durable prefix. The complete matrix also checks before/after journal commits and four schema/recovery commits. All61 runs completed, and cleanup reported zero errors. Each row's original report, raw-case list, baseline trace, execution receipt and exact worker artifact receipt has a byte count and SHA-256 in the sidecar. No per-blob input/model classification is inferred from ordinal alone.

This is real-engine evidence over the admitted96-row S3 TRAIN fixture21fbfe8b53dbcd60 (full TRAIN SHA-256 `d9a0b97e91ff9d04bc87fefa4671a49c40cee21dfac05be58a7fc7ce08045931`). The normal internal assessment partition comes from those TRAIN bytes. No external holdout/test/sealed input is used. The sidecar contains no data rows, model payloads, authentication, scores or new scientific result. It does not qualify a signed release or learned continuation.

The service native1/8-worker pause/SIGKILL test separately passed on this Linux source; its exact Cargo-reported library hash and compiler identity are included. Mac R4/S2/61-cut and1/8-worker results are separate operational receipts, and are not relabeled as Linux cases. Synthetic later-prefix CAS fixtures are a separate test, not these45 real-engine publication cuts.

Reproduction uses the source above in a claimed checkout, one reused target, `nice -n10`, Cargo jobs4, incremental0, line-table debug and one math/test thread. Build service/all-target artifacts and appliance system tests with `--locked --features jope-service/failpoints --message-format=json`; retain the exact compiler-reported executable/library, never select the newest cached file. Set `JOPE_BUILD_SOURCE_SHA`/`JOPE_INT02_SOURCE_SHA` to the source pin, `JOPE_INT02_BINARY` to the retained executable, `JOPE_DSC_SERVICE_RLIB` to the exact library, and use short owned paths outside `/tmp` for `TMPDIR`/`JOPE_INT02_RUNTIME_ROOT`. Then run:

```sh
INT02_ENGINE_BOUNDARY_CUTS=1 JOPE_INT02_EVIDENCE_DIR="$OWN_EVIDENCE_DIR"   cargo test --locked -j4 -p jope-appliance --features jope-service/failpoints   --test system live_terminal_attach_and_mcp_reconnect_preserve_the_same_run_projection   -- --exact --nocapture --test-threads=1
```

The recorded run used the identical retained Cargo system-test executable directly through `remote_test.sh`/the owned `cargo-int02-gates` helper, so concurrent cache producers cannot replace its binary or dependencies. The actual executable argv, source, library/binary hashes, compiler, exit, original receipt locations and negative controls are in the sidecar. Six metadata-admission controls reject abort instead of SIGKILL, a missing occurrence, an incomplete continuation, dirty staging, a cleanup error and a changed source. The raw successful receipts remain unchanged.

No GPU, signing, CI, speed gate, source/model archive or shared-target cleanup was performed. Ordinary scoped replay and source-independent metadata checks do not establish whole-product qualification.
