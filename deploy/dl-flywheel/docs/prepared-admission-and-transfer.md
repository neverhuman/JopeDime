# Prepared measured expansion and checkpoint-preserving recovery

This source update reduces the GPU stall caused by synchronous serial hydration of measured expansion artifacts. It preserves the logical campaign `fec184217e7c50a3fc3bc1cc7e33edafdbd218d1fafa1bd0d68debf626dcd4ed`, original checkpoint 429 ancestry, AdamW/controller/CPU and CUDA random state, mean-baseline supervised objective and global deadline 1791592146.380285. The 50,000-update limit is measured from original checkpoint 429; transfers do not reset it. This is a bounded research continuation, not production admission or reinforcement learning.

## Exact background preparation

The measured 128 extension closes 6,053 exact-version objects, totaling 4,714,394,119 bytes. It contains 7,248 contexts: 6,982 training and 266 validation, with 128 variants and 1,027 measured variant program rows. Prepared artifact SHA `b8518a6e350018e1fa6e9ba3c23d14c3f635096f5fda7ea7af16f453e1edbb53` is 13,520,220 bytes. Its float32 vectors are bit-preserving. Qualified hydration took 589,749,287 ns; total CPU preparation took 169.674 seconds with zero network GETs, S3 writes, optimizer updates or planner mutations. A separate archive sealed two exact-version paper objects.

`prepared_snapshot.py` verifies context identities, inherited lineage roles, measured records, target identities, preprocessing, qualified adapter and the complete manifest. `prepared_admission.py` binds the successful configuration, status and artifact, retains all existing variants and result references, and rechecks current authority. Only the final model's licensed frozen Kumo Small block of 2,051 values is admitted. Preparation, epoch-boundary admission and actual optimizer consumption are distinct states.

## Transfer, retirement and cold reservation recovery

`reconcile.capture` reconciles every committed batch with its immutable proof, ordered membership, mandatory pre-optimizer proposal ACK, physical checkpoint, optimizer-control state, LR/plateau counters, coverage and completed full-validation panel. Unknown reservations refuse transfer. An active ancestor must be physically frozen; stopped owners require empty exact cgroups and absent original PIDs. No ancestor rollback is permitted after planner or manifest drift.

The resume3 transfer captured checkpoint 3,296 and physically committed 3,297. Its ancestor retirement command returned a nonzero acknowledgment after actually killing the ancestor. Cleanup then stopped the successor. All failed evidence is preserved. The retirement fix records intent, reads back the physical ancestor cgroup and PID, accepts a physically completed retirement even after command acknowledgment failure, and never stops a verified committed successor because a later control-plane acknowledgment failed.

The stopped successor left original batch 3,298 reserved. `cold_reservation.py` is a narrowly pinned exception: its exact, fsynced trace ends at START of CPU input construction, before the mandatory synchronous-FULL outbox ACK and before any optimizer stage. There are no ACK rows, emissions, decoded files or physical checkpoints for that reservation. Both original writer cgroups are empty. The certificate therefore proves the optimizer never started for this exact reservation; it does not authorize retry of an unknown update. All unrelated or ACKed reservations remain refusals.

The recovery keeps the same reservation ID, ordered membership and fingerprint in place. It independently recomputes the intent digest, restores physical checkpoint 3,297 and executes the existing batch once before requesting any new reservation. It never clears, requeues or reserves that batch again. Canonical JSON normalization makes the fresh tuple-bearing capture comparable to the sealed JSON list representation without dropping any custody field. Recovery4's representation refusal is retained separately; recovery5 has the reviewed normalization and a sealed-capture roundtrip regression.

The prepared candidate remains HELD until the existing partial epoch and all fixed/full validation finish. Adoption then appends the qualified manifest before the next reservation. Descendants keep their parent's role; validation never enters optimizer targets. The normalizer, teacher trie, objective and original fixed monitor panels remain unchanged. Physical committed context IDs alone increase unique coverage; repeated shuffled exposures increase presentations.

## Validation and evidence

Five original transfer integrity tests and the private CUDA restoration probe passed. The successful CUDA probe compared all 48 parameter tensors, AdamW state and CPU/CUDA random state after the next dropout update, bit for bit; receipt SHA `16d2ee67c5e3f69e8eecd547cff23b5b1380050566e3b56259396a10fa647b22`. The failed first 1 GiB probe remains retained. Successful qualification performed three private-clone updates and zero live optimizer/planner/outbox/S3 operations.

The successor suite covers open-epoch continuation, stale progress, authority drift, exact executor/adoption receipts, retirement acknowledgments, same-reservation bit-exact continuation, ACK/trace/optimizer-stage refusal, rehashed membership tampering and sealed JSON comparison. These CPU fixtures do not establish live admission or consumption. Live launch and first-commit receipts remain under their immutable proposal namespaces; a running service alone is insufficient acceptance.

The verified native custody helper reads an anchored outbox in one physical read-only SQL transaction. Its 18 tests include a concurrent appender. Native/emitted Rust tolerance remains 1e-9. Parity failures are nonnumeric custody; unknown execution or S3 outcomes are never retried automatically.

## Remaining constraints

Native drain workers claim fresh native observations only. Generated `flywheel_variant` observations remain durably pending until a separately reviewed adapter bridges exact RegisteredInputs variant identities to the variant observe/evaluate/publish dependency queue and returns verified publication to the model outbox. It must retain role/phase identity, emitted Rust/native parity, all seven artifacts and exact-version publication verification. The 1,027 curated measured variant records do not mean every future generated variant has been evaluated.

The reviewed publisher follows only a verified active-executor pointer. It preserves original ancestry and expired historical archive clocks. The dashboard retains independent Linear/Log X and Linear/Log Y, explicit omission of nonpositive coordinates, coverage gaps and separate fixed versus expanded validation. Utilization counters are distinct from actual CUDA event time and executing worker counts.

## Source qualification

```sh
python3 -B deploy/dl-flywheel/check_source.py
python3 -B deploy/dl-flywheel/run_successor_offline.py --output OWN_NEW_EVIDENCE_DIR
```

The runner requires 51 non-skipped CPU tests in isolated transfer (21), prepared (12) and custody (18) shards. The full offline runner invokes it and retains separate receipts. Tests import only checked-in modules; no private workspace path supplies fixtures. The source manifest preserves imported runtime origin hashes and normalized checked-in hashes. Source review, a pushed commit, current-head CI, native GitHub approval and production promotion remain separate milestones.
