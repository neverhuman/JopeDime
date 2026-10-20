# Next wave: fresh GP feedback and native GPU learning

## Objective and decision

Test whether fresh, comparable GP outcomes consumed by a native child policy
improve subsequent search beyond equal additional optimization. The historical
value adapter nominates this experiment; its positive native ranking comparisons
do not qualify the flywheel hypothesis, whose primary interval included zero.

Non-goals: replacing production defaults, changing GP semantics/vocabulary,
opening sealed or outer-O data, restarting Qwen R4, allocating new paid capacity,
or filling GPUs with unproductive workloads.

Assumptions: latest direct human authority permits research on xbabe1..4 and
requires aggregate host CPU <=70%. Current data/role/evaluator/runtime and source
ownership must still be resolved. Missing identities are stop conditions, not
fields to infer. Fresh outcome acquisition and numerical evaluation remain
pending an exact owner/resource manifest. Old admissions never renew themselves.

## Read first, in order

1. `/home/ubuntu/JopeDime/AGENTS.md`, `MASTER_PLAN.md`, `AGENT_CHAT.md`, current
   worktrees/open PRs and `/home/ubuntu/JopeDime/COMMS.md`.
2. Current release-closure branch's
   `docs/plans/release-closure-20261019/CONTROLLED_FEEDBACK_RESEARCH_PLAN.md` and
   `F0_METADATA_CENSUS_20261020.json`; existing
   `/home/ubuntu/jope-evidence/release-closure-20261019/monitor-0638-20261020/F1_F2_DESIGN_AND_GAPS_20261020.json`.
3. This branch's `README.md`, `feedback.py`, `prepare.py`, `run.py`, `predict.py`,
   `audit.py`, and immutable `preregistration.json`, `comparison.json`,
   `post-run-audit.json`, corrected teacher receipt and source hashes.
4. Canonical native source:
   `deploy/dl-flywheel/jope_flywheel/training_snapshot_native.py`,
   `pipelines/native/{neural_native_batch.py,neural_native_train.py,neural_storage.py,neural_outbox.py,neural_native_stream.py}`,
   relative to `deploy/dl-flywheel` where applicable. Locate actual current
   functions before editing: `NativeContextLoader.get`, `objective`,
   `run_microbatches`, `run`, `checkpoint`, `Outbox.finish`,
   `BatchStream.reserve/commit`.
5. `/home/ubuntu/xbabe2_kicker/fleetcap/{README.md,run-capped,fleetcapd.py}`,
   actual service/drop-ins/PID environment/cgroups on each Linux node, and
   actual Metal runtime/process/resource guards on xbabe4.
6. `catalogs/{s3.json,training-artifacts.json}` and
   `docs/runbooks/{training-artifact-locations.md,ingest-winners.md}`;
   actual HWM comparison and scored-program contracts.

## Ownership, resources and forbidden writes

Admiral names the existing resource owner and numerical evaluator owner. Existing
release integration Root retains PR387 and its checkout; its source writer must
provide a stopped-head/path lease before an integration edit. This research Root
owns only `research/native-feedback-upgrade-20261020/**` on its new branch and its
own external custody. Reuse an existing suitable lane; no competing worktree,
shared index mutation, new reviewer or overlapping source claim.

xbabe1's Qwen R4 remains with its current runtime owner; inference is not native
optimizer progress. Initially nominate idle RTX3090 nodes xbabe2 and xbabe3 for
paired native studies. Both have 128 logical CPUs and 24GiB VRAM. xbabe4 is a
32-CPU-core, 80-GPU-core Apple M3 Ultra with 512GiB unified memory; it needs an
existing Metal-capable runtime qualification and its own resource route.

Before each launch, qualify <=70% aggregate CPU with operating margin, measured
foreign load, a continuous one-second controller and safe throttling/pausing of
only claimed jobs. 70% equals 89.6 core-equivalents on Linux and 22.4 on this Mac.
Current Linux target0.70 is not itself proof of a hard ceiling. Preserve existing
RAM/swap/disk protections and all unrelated jobs. Do not edit the shared fleetcap
source or restart its service without the resource owner's path/control lease.

Benchmark productive microbatch packing using real complete sequences. Freeze an
explicit VRAM reserve and use measured peak allocated/reserved memory and useful
examples/second. Increase batch/accumulation only if that improves useful work;
avoid memory-filling reservations, truncation, silent batch changes and fabricated
GPU training. No blanket CPU/GPU benchmark reruns.

Forbidden: production model manifests/latest aliases; old campaigns, checkpoint
429 and historical 17773; R3/R4 source, queues and workers; outer-O/sealed rows;
generated evidence/target health files; foreign job signals; shared runner/global
auth changes; destructive cleanup or unverified storage reclamation.

## Ordered work packets

1. **R0: resource and identity closure, no scientific work.** Capture hardware,
   current owners, exact native checkpoints/encoders/vocabulary/preprocessing,
   data roles, evaluator/HWM identities and resource settings in
   `research/native-feedback-upgrade-20261020/next/manifest.json`. Bind source and
   launch bounds before results. Resource owner supplies actual guard/readback;
   absence stops dependent numerical launches. Log admitted and pending fields
   separately. Never edit a historic permit into a fresh one.

2. **R1: fresh target bridge.** On a claimed new branch from current main,
   implement research-local `next/neural_feedback.py` and
   `next/test_neural_feedback.py`, then an opt-in native integration only after
   its owner's lease. Use strict versioned rows with context/parent/role,
   dataset/cut/fold/evaluator/vocabulary/protocol, program/profile, incumbent
   score/direction, attempt/status/censoring, immutable result reference and
   charged/physical cost. Compare high-water marks only within the exact identity.
   Refuse unknown, incompatible, unsupported or nonfinite results. Preserve
   failed/censored attempts. Make publication and consumption idempotent,
   including failed attempts; do not repeatedly reacquire a failed key without a
   declared retry policy. A result hash must flow into a changed target snapshot,
   actual optimizer batch and child checkpoint. Tests cover ties, reversed score
   direction, stale/incorrect identity, duplicate/failure replay and crash points.

3. **R2: native GPU warm-start controls.** Proposed initial engineering ceiling:
   at most128 additional optimizer updates per trained arm/seed, three fixed
   seeds, sequential controls per host and <=60min per host including closure.
   Freeze final checkpoint selection; do not add updates after viewing results.
   xbabe2 starts from17773; xbabe3 from429. For each parent checkpoint match
   architecture, optimizer initialization, LR, tokens/context sampling and
   updates between old-data continuation and uniform fresh-feedback arms.
   Explicitly label a fresh optimizer as warm-start, not exact resume. Use
   existing `PolicyLM` and native masks; retain objective numerator/denominator,
   supported contexts/pairs, unclipped gradients, update norms, LR, actual
   runtime/VRAM and parent/child hashes. Prove interrupted state parity before
   claiming exact recovery. A device unsupported by the pinned runtime is a
   named blocker, not an unrecorded installation/fallback.

4. **R3: prospective GP search comparison.** Before outcomes, evaluator owner
   freezes a sufficiently powered TRAIN acquisition roster and separate
   development assessment families using the retained lineage union, exact S3
   VersionIds/hashes, current rights and a parent-level analysis. Do not reuse
   the current214 contexts as untouched confirmation. Proposed plumbing starts
   with the existing16TRAIN-parent cohort only if owner verifies eligibility;
   it cannot power a generalization claim. Freeze identical logical slots and
   actual physical cost caps for continuation/uniform/targeted arms, original
   native and GP-only controls. Count incumbent checks, failures, cache reads,
   diagnostics and refits. Select on inner-development data only; outer-O and
   sealed data remain unavailable. Primary endpoint: paired parent-level
   end-to-end GP score lift of uniform-feedback child versus continuation at
   fixed total acquisition+training+search cost. Targeted-vs-uniform and
   memory-only contrasts are secondary. Freeze direction, practical margin,
   uncertainty/multiplicity and failed-fit treatment in the manifest.

5. **R4: useful integration and PR.** If an engineering acceptance case passes,
   prepare an opt-in value-guide adapter in
   `deploy/dl-flywheel/pipelines/native/neural_value_guide.py` and focused
   `tests_native/test_neural_value_guide.py` on the claimed source branch.
   Tests prove exact checkpoint/encoder/grammar binding, default incumbent
   selection, immutable research output prefixes, outcome-free inference,
   safe download/restore and no production alias writes. Do not weaken native
   refusals to make the research path appear deployed. Follow the existing
   one-PR queue, exact-head CI and actual independent review; retain a rollback
   identity and verify receiving-main. No self-approval or forced merge.

Parallel execution is limited to disjoint claimed output/model roots on nodes2/3;
R0/R1 source and shared manifests serialize. Do not parallelize writes to the
native trainer, common resource controller, canonical ledger or PR integration.

## Acceptance, receipts and stop rules

Current research validation, from this package directory:

```sh
rtk proxy env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  OPENBLAS_NUM_THREADS=2 python3 -m unittest discover -s . -p test_feedback.py -v
rtk proxy env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  OPENBLAS_NUM_THREADS=2 python3 audit.py /absolute/restored/new-output-root
```

For future native source tests, pin the existing qualified interpreter and use
the current owner-approved `run-capped`/taskset/nice envelope. Run the focused
feedback/value-guide tests plus affected train/batch/outbox/snapshot regressions.
Zero collected tests, skipped required devices, missing interpreters, source
drift and fixture-only passes do not count as numerical qualification. Derive
the exact invocation from the current owned checkout; do not reuse an expired
campaign wrapper or create an uncontrolled Cargo target.

Completion requires nonzero fresh GP results; changed versioned TRAIN targets;
actual consumed native child updates; child-guided next proposals; matched-cost
parent-level comparisons and their failure-aware interpretation; independent
source review; immutable checkpoint/raw-result references and full restoration
proof; explicit publication/PR/receiving-main state. Inconclusive lift remains a
valid result, not a reason to move the threshold or rerun an unchanged trial.

Stop dependent work on stale ownership, missing resource/identity admission,
>=70% host CPU or insufficient headroom, RAM/disk/VRAM guard breach, nonfinite
gradient/state, source/input/evaluator drift, test leakage, unsupported transport,
uncertain interrupted liabilities, failed restoration or exhausted provider.
Preserve original failures and continue independent work that remains admitted.

Every20min checkpoint records source/base/head, owner, admitted versus pending
state, actual updates/fits/HWM changes, primary/secondary lift and uncertainty,
CPU/VRAM/latency/cost, exact next executable deliverable and expected checkpoint.
Write compact `card.json`, `metrics.json`, `wave.manifest.json`, `context.md` and
index additions through the relevant owner. Bulk data/checkpoints stay compressed
outside Git with exact S3 VersionIds and member restoration. Final stopped-head
handoff names dirty paths, commands/exits/test counts, immutable receipts, actual
owner consumption, blockers and released/retained leases.
