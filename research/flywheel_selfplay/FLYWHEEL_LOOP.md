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
