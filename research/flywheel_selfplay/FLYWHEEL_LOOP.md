# Closed fused flywheel — 2026-10-29

The executable restart is `research/flywheel_selfplay/run_fused_closed_loop.py`.
The promotion implementation is `automatic_promotion.py`; its journal is the recovery
source for the training candidate arrays. This is a development experiment, not
serving promotion or sealed evaluation.

```
training-only real table -> CPU-exact landmarks -> frozen DOPE -> 10x32 tokens
                                                                |
             +--------------------------------------------------+
             v
     JopeDime adapter trains on current candidate pools
             |
             v
     generate typed AST edits (propose_flywheel_candidates)
             |
             v
     fused adapter ranks proposals -> exact Rust training-only CV
             |
             v
     completed + CV gate + strict incumbent improvement?
             | yes
             v
     fsync promotions.jsonl -> append all parallel pool arrays
             |                 + append champion parent
             +-----------------+----> next round trains/selects on these pools
```

## Files and functions

1. `run_fused_closed_loop.load_packet` validates train-only JTT1 headers (`nq=0`),
   hashes, DL-Train membership and family-disjoint development partition. The
   custody exporter strips outer rows before transfer; it does not deliver sealed
   families. Source corpus: historical S3 experiment
   `s3://veox-jopedime-learning-use2/experiments/flywheel-selfplay-20261023/`.
2. The pinned CPU-exact `FusedJopeNetwork` uses
   `hpc.landmark_workspace.FusedLandmarkWorkspace.fit_transform_and_extract_bundle_high`,
   `_run_dope_mlc_v3_and_lm48` and `_project_stage1b_tokens_batched_gpu`.
   Only `adapter_10tok` parameters outside its frozen anchor train.
   `python/src/jope/north_star/dope_freeze_guard.py` locks eval/gradient state,
   excludes DOPE from AdamW, and hashes parameters plus normalizers each round.
3. `rust_evaluator.propose_flywheel_candidates` generates typed mutations and
   family-excluding transfers. `adapter_10tok.predict_costs` selects four of eight
   proposals per training context. `evaluate_gp_batch_rust(run_outer_test=False)`
   invokes the hash-pinned Rust two-fold CV worker on training-only input.
4. `automatic_promotion.promote` requires completed full training CV, its gate,
   matching program/context identity, finite arrays and strict lower fitness than
   the current passing incumbent. Utility is negative fitness. Missing confidence
   intervals are recorded as unavailable, never fabricated or a statistical pass.
5. `promote` writes/fsyncs a journal event before `apply_event` updates program,
   projection, signature, fitness, logratio, gate, prior and program IDs together.
   `replay` restores admissions idempotently. The same code admits the historical
   90 verified programs; non-improving archive admissions are distinguished from
   incumbent promotions. Five ordered historical ratchet promotions occur.
6. Each next round selects from the mutated pools, guarantees top candidates in
   training slates, and uses newly promoted DSLs as generation parents. Checkpoints,
   optimizer update evidence, freeze hashes, dev utility and throughput are saved
   per round; status is written to `~/jope_plan/ns/FLYWHEEL_GP_FUSED_STATUS.md`.

## Why the October 24 result did not establish the requested durable loop

The premise “injection never ran” is not literally supported by the retained source.
`selfplay_loop.run_live_rust_gp_selfplay_expansion` already contains an in-memory
append block (original lines 955–975), and its queue supplied evolved training
examples. `flywheel_selfplay_report.json` records 1,772 injected evolved children
and 510 full-corpus optimizer steps for ROUND_1. These are distinct mechanisms.

The missing connection is durable candidate-pool promotion: `load_all_pools_and_features`
reads original `pool_path` files; the append block only mutates RAM. It does not
persist a pool revision or journal that the next process reloads. The original
round function saves checkpoints and retained scores through
`persist_round_artifacts_to_s3`; saving scores/weights does not update disk pools.
It also admits every full-CV passing candidate rather than only strict incumbent
improvements, and adds the best passing result to champions without an improvement
requirement. Thus persistence and promotion were different operations.

The later `verify_flywheel_winner_pool_integration_v2.build_versioned_training_winner_pools`
explicitly performed the missing separate experiment, materializing
`pools_v2a_strict_train_q25` and `pools_v2b_dl_train_all90`. The committed
`flywheel_winner_pool_integration_v2_report.json`, step 3 and step 5, documents all
90 Full-JTT1-qualified novel programs across 16 DL-Train contexts and matched
430-/510-step comparisons. It reports six beating the original frozen incumbent
(after verifying seven formerly screening-only candidates), whereas
`flywheel_closure_and_next_phase_report.json`, part 2, reports the earlier five.
The new journal's five promotions compare sequentially against a moving incumbent;
this count has a different denominator from six versus the frozen original best.
Neither historical comparison demonstrated a statistically established dev utility
improvement. No such claim is inferred from admission, training loss or freeze checks.

## Reproduce

Use a dedicated CUDA-12.4 environment on xbabe1 (driver 570): PyTorch 2.6.0+cu124,
NumPy, SciPy, pytest and psutil. The shared cu130 environment cannot use this driver.
Hold `~/jope_plan/qualification.lock`; the executable acquires it nonblocking.
Never invoke Phase-2 CUDA kernels or change another process. External source and
model hashes are captured in the run manifest. Bulk inputs/checkpoints stay outside
Git at `/home/ubuntu/jope-data/flywheel-gp-fused/`.

```
python -m pytest -q research/flywheel_selfplay/test_automatic_promotion.py
python research/flywheel_selfplay/run_fused_closed_loop.py \
  --input /home/ubuntu/jope-data/flywheel-gp-fused/input \
  --output /home/ubuntu/jope-data/flywheel-gp-fused/dry-run --dry-run
python research/flywheel_selfplay/run_fused_closed_loop.py \
  --input /home/ubuntu/jope-data/flywheel-gp-fused/input \
  --output /home/ubuntu/jope-data/flywheel-gp-fused/run-20261029 \
  --rounds 2 --steps 64 --batch-size 32
```

Development contexts/s counts actual stochastic training presentations and is
separate from unique GP contexts/s. The old report's 183.26/133.06/211.24 seconds
measure GP expansion only; comparing those against new training-plus-GP wall time
is not an equivalent speedup measurement. Token throughput requires actual emitted
source and token counts; ranking/optimizer steps are not generated code tokens.

## Restart acceptance corrections and continuation

The first two GPU-head rounds trained and retained identical DOPE hashes, but all
GP calls failed the pinned parser admission because `nq=0`. They are not GP passes.
`automatic_promotion.training_cv_envelope` preserves every training byte and adds
10 structural zero records to satisfy `parse_s1_data`'s envelope check. The parser
only decodes training rows (teacher-v2/rust/lambda-worker/src/main.rs, lines 556–608).
No original query cells are recovered or supplied. A direct native RED/GREEN probe
is retained in custody. The first padded attempt reused the old evaluator cache;
subsequent rounds use a distinct round cache and retain all failed receipts.

`--resume CHECKPOINT` strictly reloads model/optimizer state, verifies the original
DOPE hash and replays the promotion journal into both pools and mutation parents.
`--continuous` continues until stopped or a guard fails; default execution remains
two rounds. Continuation keeps the qualification lock, uses one pinned CPU-exact
path, and stops if the run exceeds 10 GiB, free disk falls below 50 GiB, all GP
calls fail, DOPE changes, or parameters fail to update. It deletes no evidence.

The initial target batch32 used little memory. Batch9216 of stochastic real-context
slates reaches approximately 80–90% device memory with the 90% allocator cap.
This is repeated training presentation of 16 unique contexts, not 9,216 unique
new datasets. Per-round GP DSL token rates use actual cl100k_base counts; they do
not assert Rust/all-language emission throughput.

## First corrected real-data rounds

Round 4: 64 native CV evaluations, 63 passing, 2 strict promotions, 41 changed
head tensors; 125.83s wall, 275.25 training presentations/s, 20.27 GP DSL tokens/s.
Round 5: 64 evaluations, 62 passing, 3 strict promotions, 41 changed tensors;
99.06s wall, 462.95 presentations/s, 25.54 GP DSL tokens/s. Both used batch9216
and 2 optimizer updates; these repeat stochastic views of 16 contexts. DOPE
composite SHA-256 remained c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814.
Development utility fell from 2.24815246 to 2.18063793; there is no quality-gain
claim. Older round4/5 `GP.novelty` denotes unique/evaluated fraction (0.891/0.875),
not novelty against the pool; the final runner records both metrics separately.
Committed compact evidence and full promotion journal live in
`research/flywheel_selfplay/results/fused-20261029/`; tables and checkpoints remain
outside Git. Continuation resumes the same journal and checkpoint lineage.

Continuation completed round13 before allocator fragmentation caused a CUDA OOM
in the next incomplete round. That partial optimizer outcome remains unknown;
resume uses the last fully written checkpoint and the durable journal. The
recovery batch is 8192 with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
The runner now returns unused cached segments between rounds, writes explicit
failure status on exceptions, and reports continuous next-round ETA correctly.
The failed log and manifests remain in custody; no other jobs were interrupted.

## Development adapter admission (16:35Z owner nudge)

Checkpoint retention runs at startup and after every scored round. It retains
round882, the canonical `best_dev_checkpoint.json` target, the actual startup
`--resume` checkpoint, the last20 accepted root checkpoints, the latest logical
restart frontier, and explicit `--retain-checkpoint` pins. Unqualified/orphan and
partial checkpoints remain. Preserve the earlier r785 witness using
`--retain-checkpoint round-785.pt`.

Rejected checkpoint payloads and scored Rust candidate scratch are archived before
removal. Every GP proposal receives a lossless postfix tree and exact constant
lexemes in a compressed JSONL shard, with evaluated scores, cached pool scores or
an explicit unselected status. `rebuild_program(tree)` verifies the original source
SHA. These trees reconstruct GP programs; neural best/resume checkpoints retain
model and optimizer state separately. Root round receipts, score JSONL, promotions
and admissions remain in place. At8MiB, summary JSONL and run-local logs rotate
through verified archives; externally redirected live stdout is outside this scope.

Retention creates a tar.zst and hash manifest outside guarded roots and the checkout,
independently restores every body/size/mode, uploads archive and manifest with the
host AWS CLI, then downloads their exact S3 versions and checks full hashes before
any deletion or rotation. Tree shards undergo the same S3 body verification before
scored scratch cleanup. The approved learning experiments prefix is used; existing
host profile `veox-jepson-prod` supplies credentials. No credentials are provisioned
or passed to containers. Changed pointers/sources, active readers, upload failures
and insufficient restore headroom defer removal. Unknown evidence remains. The
original10GiB combined-output and50GiB free-space guards still apply.

Defaults: `--checkpoint-keep-last 20`,
`--checkpoint-archive-dir ~/jope-data/archive/flywheel-retention` and
`--retention-s3-prefix s3://veox-jopedime-learning-use2/experiments/flywheel-full-20261030/run-20261030/retention/`.
Local archives are retained. Host tools `aws`, `zstd` and `lsof` are required.
`retention-round-N.json` records each post-scoring outcome and bytes after cleanup;
archive manifests/receipts retain the recovery route. Maintenance adds time beyond
the existing measured training/GP window. Preparation and offline proofs do not
update the running process: the watcher owns deployment at a safe restart.

Strict adapter admission is a separate opt-in policy. An invocation that omits
both `--strict-adapter-admission` and `--adapter-min-gain` keeps the existing
behavior, including acceptance of exact development-utility ties. To opt in,
pass both flags; candidates are accepted only when their utility is strictly
greater than the incumbent utility plus the supplied finite, nonnegative margin.
There is deliberately no default margin: H49-SEL calibration is still pending.
This flag does not authorize restarting or reconfiguring the live xbabe1 loop.

The maintenance companion `rotate_flywheel_logs.py` additionally bounds the local
summary and direct `custody/*.log` files to `--keep-rows` (default20). Run it from
merged source with `--run`, `--custody`, `--archive-root`, `--round-lock` and the
actual Python `--writer-pid`. It takes the existing qualification lock between
rounds, stores full original bytes in zstd, verifies independent decompression and
S3 exact-version downloads of archive plus manifest, then rewrites only the tail.
It preserves the inode for append-mode stdout/stderr, including incomplete final
rows. Unknown readers/writers, non-append descriptors, changed sources and failed
custody checks defer that file without truncation. Archives live outside the run,
custody directory and checkout; the50GiB free-space reserve remains in force.

This companion runs without primary-source changes or a restart when stdout/stderr
already use append mode (`>>`). It polls every30seconds and waits for a locked round
boundary, rotating at most4eligible files per boundary (summary first); simultaneous companions cannot run. A primary PID change stops the
companion. Start it again with the new PID after any later primary restart. It
never changes checkpoints, training inputs, scores, promotion/admission journals
or DOPE. Summary/log bytes are bounded; retained round/score/tree receipts still
grow and must be included when reporting whole-run growth. Full rotation receipts
and source hashes accompany the immutable archives. This is custody maintenance,
not Python telemetry computation or collection.

`dev_incumbent.DevIncumbent` preserves the best development utility under a fixed
comparison identity. The runner first resumes the latest logical round, then
restores the highest-utility historical adapter and AdamW state (most recent tie:
round40). Training proposals that reduce utility are saved under `rejected/`;
both model and optimizer are restored before GP ranking. Equal scores may advance
training, preventing long argmin plateaus from suppressing every optimizer update.
`best_dev_checkpoint.json` binds the incumbent checkpoint hash, fixed development
identity and utility; `adapter_admissions.jsonl` records each acceptance/rollback.
The pointer is published after a durable accepted-state checkpoint. Old files and
rejected proposals are retained within the original10GiB campaign ceiling.

The owner0950 candidate-diversity safeguard keeps this admission rule unchanged.
Batch permutations now have a private `round-step/v1` seed derived from the root
seed, logical round and optimizer step, never the retained best round. The existing
Rust sampler preserves whole-family blocks and full1219 coverage. Every completed
round records each batch seed, context count and order SHA; candidate parameter SHAs
are also written to the admission journal. On resume, completed round receipts and
the admission journal rebuild a rejected-hash index under the fixed evaluation
identity. Previously rejected parameter bytes are refused **before** Train/Val
candidate selection is evaluated. This unexpected integrity event restores the
incumbent, writes `candidate_refusal_round-N.json`, and fails closed without another
evaluation or optimizer update; it does not invent a completed round or cached score.

Exact parameter duplication and discrete utility recurrence are separate measures:
before this patch, rounds965..1024 had60 distinct parameter SHAs but59/60 utilities
equal to an earlier rejection under the same evaluator identity (beginning775).
The safeguard does not claim to fix this quality plateau. Report the same utility
recurrence fraction and distinct parameter count for the first10 completed rounds
after an authorized deployment; until then, the after result is unknown. H-wave
copy experiments and H49-SEL acceptance-margin calibration remain separate work.

Paired CPU replay at identical inputs reproduces the round41 regression exactly;
all non-adapter tensors remain identical. See the committed dev-regression receipt.
The static process command's round13 resume argument described one startup load,
not a per-round reset: AdamW steps206/210/212/250 at rounds38/40/41/60 demonstrate
continued optimization. Keep LR1e-4 and two steps while guarding dev; do not infer
that more updates or higher LR will fix a misaligned training objective. New
receipts log update norms, gradient norms, actual optimizer steps and phase timings.

## Rejection fixed-point escape (17:05Z owner nudge)

Each logical round gets its own logged Python/NumPy/Torch seed and training-order
rotation. A rejected adapter triggers a one-step proposal at one-quarter of the
retained optimizer's LR; repeated failed alternatives reduce further, bounded at
LR1e-6 and at most the original two optimizer updates. Accepted gentle LRs persist
in the incumbent optimizer. Model/AdamW rollback and the best-dev score gate remain.
Candidate parameter hashes distinguish actual proposals from identical dev argmins.

After K=3 repeated rejections, GP generates16 proposals instead of8, rotates
mutation parents, excludes programs already in the context pool, and allocates
its unchanged four native evaluations to two model-ranked plus two seeded
exploration candidates. GP always runs on the retained incumbent even when an
adapter proposal is rejected. Replayed/newly promoted champions explicitly carry
DL-Train eligibility for family-disjoint retrieval. All changes and actual
native evaluation/promotion counts are in the round receipt; improvements are
measured, never guaranteed. DOPE, Phase2 exclusions and resource ceilings persist.

## Low-margin projected updates (17:40Z owner nudge)

At best checkpoint95, context8's raw predicted costs differ by only3.50475e-5;
rejected round112 reverses that margin to−5.00679e-6 even at LR1e-6. This is a
small but real ranking boundary, not an exactly equal floating-point tie.

The optimizer now preserves label-free prediction margins for uncertain incumbent
rankings. A fixed global activation threshold1e-3 identifies low-margin cases
(two in this cohort); each retains at least half its anchor-checkpoint margin.
After each AdamW step, at most two small gradient-projection passes repair violated
score margins by moving only trainable adapter parameters. Other parameter
directions remain free to learn. Reference IDs/floors are pinned in
adapter_margin_constraints.json and checkpoint payloads, rather than recalculated
and relaxed at each restart. Further rejection-driven LR shrinking is disabled
while these constraints are active. Failed projection retains the best model and
lets GP continue; no optimizer/native-evaluation ceiling is increased.

This explicitly uses existing development feature slates for prediction-stability
tuning and the prior model's choices as references. Candidate CV utilities are
not gradient targets. The cohort and RAW ARGMIN evaluation rule remain unchanged;
there is no context-specific score override, routing fallback or metric relabeling.
Consequently this development diagnostic is not independent validation. Outer and
sealed families remain outside the input closure; DOPE remains frozen.

## Flat-time training input cache (18:35Z owner nudge)

Live profiling without stopping the worker observed2.64million minor faults and
35.31s kernel CPU over127.55s spanning completed rounds156–158, with no major
faults, process read I/O or swap. Projection/backward stayed about.24s while
forward/input/targets grew22.65→55.06s. The old batch8192×192×412 path materialized
roughly2.6GB of NumPy features per step plus temporary copies and host transfers.

`DevicePoolCache` retains original per-context tensor rows on the GPU and gathers
by context offsets plus exact slate indices. No arithmetic or selection formula
changes. Promotion/array replacement invalidates only that context's cache.
Training labels, gates and revealed observations use the same gathered indices;
single-context GP/dev/projection calls retain their existing path. Each startup
requires bit-identical legacy/cached features, targets and predicted costs on
real repeated-context slates. Large gathered slates are released before the next
step. Receipts expose device packing, model/target timings, index bytes, avoided
host-feature bytes and cache transfers; these are nested within the existing
forward/input total. Raw argmin, margin projection, budgets and DOPE freeze persist.

Live qualification: rounds162–164 took57.54/59.20/63.77s total and
2.110/2.001/1.996s training; device packs.070/.060/.056s. Exact startup
input/cost parity and35 focused tests passed. Each round retained raw dev
2.267816,2 optimizer updates,64 native evaluations and unchanged DOPE.
See `results/fused-20261029/dev-regression/device_cache_live_proof.json`.
Historical Oct23–24 GP-only133–211s has a different scope.

GP promotion-drought escape (1935 nudge): accepted adapter ties previously
disabled exploration indefinitely because its trigger watched only rejections.
The independent GP clock now requires18 consecutive completed, evaluated rounds
with zero promotions. It schedules3 exploration rounds, then3 ranked-only
cooldown rounds; only actual completed rounds with this policy advance its
phase. Promotion or missing/unknown/evaluation-free history resets the drought.
The existing proposal generator produces16/context during exploration, and
`gp_selection_indices` chooses two ranked plus two seeded random alternatives.
Max4 native evaluations/context,64/round; training/LR/dev/DOPE unchanged.
Pre-outcome policy: `results/fused-20261029/dev-regression/gp_exploration_preregistration.json`.
Full per-round/per-context audit: `results/fused-20261029/dev-regression/gp_plateau_audit.md`.
No promotion or dev improvement is promised by exploration.

Rust-only telemetry direction supersedes the Python adapter (Jepson21:47Z).
`fused_telemetry.py` and its Python telemetry tests are removed. The pending
legacy enabler is stopped; the current training process remains telemetry-off.
All new telemetry metric computation,aggregation,clocking and sink writing must
run in `crates/jope-telemetry` through `jope._telemetry`,with only one-line calls
at PyTorch loop boundaries. Preserve `JOPE_TELEMETRY_RUN_DIR`,the events schema
and all stable metric names/ids,including `io.batch_gb_s`0x041b.

PR453 stays draft until luna_tel_train's native StepTimer/StepBatch/IO/flywheel
API lands. Current generic `Recorder.emit_scalar` is not a substitute for moving
rate computation and lifecycle/aggregation into Rust. Update this branch onto
that actual API without rewriting shared history; use targeted local native
tests and a direct extension smoke test,no new GitHub CI dispatch. Then install
the compatible native wheel and perform the original completed-checkpoint
handoff,preserving the pool/promotions,optimizer and frozen DOPE. Until then,
do not restart with the superseded Python adapter or resurrect its enabler.

Priority2205 recovery uses the unchanged logical batch8192/checkpoint360,with
telemetry off while the Rust API is developed. `wait_for_gpu_headroom` is DL
allocation control: release unused own cache,wait for20GiBfree,back off5s at
startup and before each round. The measured training peak is about17GiB.
This check cannot replace a common exclusion lock across all GPU launchers.
Two guard tests cover busy→busy→ready and insufficient hardware. Actual
rounds361/362 completed50.07/52.22s,with unchanged DOPE and64 native evaluations
each. Full checkpoint/best-pointer/journal-prefix verification is in
`results/fused-20261029/gpu_guard_recovery_proof.json`.

Native port integration handoff (reviewed uncommitted P1 API on xbabe2):
`Recorder.begin/mark/put/end/commit` and `open_flywheel_step` are being added.
The latter currently targets the older W trainer and `record_io` requires the
legacy resident-NumPy batch dictionary. The fused worker instead delivers
cached GPU pool gathers plus CPU indices/cold-pool payload. Confirm native
fused/frozen-DOPE metadata and a compatible native IO path before adopting the
factory. Never reconstruct the eliminated multi-GB CPU slates to fit an IO
API. Rust must count bytes,compute rates,aggregate and write sinks; Python
passes existing DL values/buffers with one-line calls. Unsupported IO timing
must remain unavailable rather than fabricated. Use only the landed,locally
qualified API/wheel; the in-progress owned Rust checkout is not a stable ABI.

GPU handoff protocol (2040 owner direction): startup qualification and each
round hold xbabe1's `/home/ubuntu/jope_plan/qualification.lock`. At round end,
finish queued GPU work,synchronize,release unused allocator cache,unlock and
yield1second. Other xbabe1 GPU jobs acquire the same host-local lock and check
remaining VRAM (the model remains resident). Phase3 runs on xbabe2 with its own
independent lock. A separate EX|NB descriptor observed the released window at
round677. Resource guard still waits for20GiBfree at the next round start.

Storage recovery receipt `results/fused-20261029/storage_custody_recovery.json`
binds the S3 version,fullarchive/manifest SHA,independent POSIX restoration of
all1068 checkpoint bodies/modes,and protected best/anchor/journal hashes. Only
verified old nonbest checkpoints were pruned; all score/round receipts and
scratch remain. Run size fell10.753GB→2.749GB under the unchanged10GiB ceiling.
Telemetry-off resumed661;662/663 verified frozen DOPE and journal continuity.

### Restart scale cycle correction (owner1645)

A read-only audit of rounds1341–1360 found20distinct candidate parameter hashes
and40distinct batch seeds/orders, with20distinct update norms clustered by four
phases and only4development selection vectors. The full-cohort path already uses
round/step-hashed seeds and family-shuffled batches; `training_rotation` is not its
batch selector. The cause of the short scale cycle is the explicit `round % 4` LR
and sigma tables in warm restart. Restoring Adam step479 from best882 is deliberate
incumbent rollback, and the full-cohort path overrides generic retry LR/steps.

`seeded-adapter-warm-restart/v2` replaces only that four-phase scale table with a
deterministic round/seed hash. Independent log-uniform LR and sigma values stay
within the existing bounds: LR2.5e-5..3e-4, sigma1e-5..3e-4. The policy is fixed
before its new development outcomes, uses no development metrics, and logs its
version/bounds/actual scales per round. Noise uses the existing private generator;
Adam moments/step, two optimizer updates, model lineage, DOPE freeze, corpus and
admission rule remain unchanged. Replay with the same round/seed/state is exact.

This removes the programmed scale period; it does not guarantee new hard-argmin
selection vectors or better validation utility. Repeatedly observed DL-Val91 is a
selection cohort, so its gains would remain adaptive development observations, not
unbiased generalization evidence. Do not choose bounds or retain proposals by
post-hoc validation tuning. Existing sealed/outer folds remain outside the worker.
H52 runs its separately preregistered copies; this change launches no copy/grid
experiment. Runtime qualification measures proposal scales/update norms and hard
selections separately, and reports a persistent functional plateau if it remains.
