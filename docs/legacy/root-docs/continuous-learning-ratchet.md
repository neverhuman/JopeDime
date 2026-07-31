# Continuous-learning Rust authority

The `ratchet` module is the authority boundary for continuous model-seeded GP.
Python remains responsible for pinned training and inference. It cannot select
the retained model population, admit attempts, advance state, or publish a
serving checkpoint.

## Immutable flow

1. A `DatasetLearningState.v2` snapshot reconstructs current labels from
   append-only observations and monotonic revisions. Track-1 and Track-2
   histories remain independent. A confirmed Global-CV champion and an exact
   train-side Track-1 frontier are the only known-best anchors.
2. `SeedInferenceRequest.v2` binds the serving wave, state, dataset encoding,
   policy, corrector, registry, vocabulary, and draw seed. Python returns
   exactly 96 canonical proposals in `SeedInferenceResponse.v2`.
3. Rust validates every identity and retains exactly 64 proposals: reward 32,
   disagreement 16, novelty 8, and seeded-temperature 8.
4. `Generation0Receipt.v3` contains exactly 256 unique semantics with physical
   topology `32 archive + 64 model + 8 GPR + 60 mutation + 60 crossover + 32
   random`. Mandatory roles are separate bindings, so shared anchors occupy
   one physical slot. Offspring refills are deterministic and bounded.
5. `AttemptAdmissionManifest.v3` binds that generation, the current serving
   wave, the latest dataset-state pointer, dataset bytes, folds, scheduler
   draw, policy/corrector, and a precommitted 10+2 proof slate. The admission
   file is written last and is immutable.
6. One micro-batch contains eight distinct scheduler draws and one immutable
   `ReplayManifest.v2` with a frozen 200-group reservoir and 100 globally novel
   eight-dataset steps. Its completed checkpoint advances only
   `training-head.json`.
7. A recurring `LearningWaveManifest.v1` is eligible at the later of 25,000
   new labels and 30 minutes. It must match an accepted training-head object
   and pass the recurring train-only and corrector gates before
   `waves/current.json` advances.

## Failure behavior

All mutations use a single-writer guard, immutable content-addressed objects,
and atomic pointer replacement. A digest mismatch, missing anchor, malformed
generation, incomplete model quota, replay reuse, holdout leakage, or multiple
writer stops the relevant operation. Already committed attempts remain bound
to their original manifests after a serving-wave change. An authority fault
can stop new admissions and restore the pointer to an already accepted
immutable ancestor without rewriting evidence.

Rollout switches default off. Wave authority is rejected unless dataset state
v2, scheduler v4, replay v2, and generation-zero v3 are all enabled.
