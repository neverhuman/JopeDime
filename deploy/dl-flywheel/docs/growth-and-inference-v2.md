# JopeDime growth and inference pipeline v2

Observed 2026-10-09 UTC. This document describes the implementation and promotion criteria; counters in the live dashboard remain the source of current progress.

## Why growth stopped

The first controller limited all unadmitted variants to 128. Publishing data, finishing Kumo encoding, or scoring a GP did not release capacity. Only verified training/validation exposure did. Consequently 138 published variants minus 10 admitted variants exhausted the queue. This was a scheduling bottleneck, not evidence that augmentation or GPU encoding could not be faster.

The replacement uses independent queues. It retains the existing SQLite identities, S3 objects, parents and immutable generation configuration. It does not mark data trained to free space.

| Stage | Work and completion evidence | Capacity |
|---|---|---|
| Generate | Deterministic augmentation; source, target, split, recipe and content hashes | At most128 reserved/unpublished inputs |
| Publish raw | Dataset bytes, source/transform metadata and receipt; exact S3 VersionId readback and SHA256 | Four concurrent publishers; generation waits only on this buffer or encoding backlog |
| Encode | Kumo Small only, 2,051 values, with source/target/encoder identity | At most4,096 unencoded inputs; xbabe1 GPU dedicated to required encoding |
| Infer | Verified input and encoding, frozen model/checkpoint and train-fit normalizer | Independent bounded durable proposal queue; no GP labels required |
| Execute | Exact dataset/split + canonical GP cache lookup; absent results run native/Rust evaluator | Local CPU workers; deduplicate exact evaluation keys |
| Publish score | Actual RMSE, baseline, GP string, emitted code, evaluator identity and receipts | S3 verification required before a result is reusable |
| Train | Next frozen snapshot includes verified measured labels | Snapshot adoption and actual optimizer exposure counted separately |

The raw-data target remains100,000; the backlog limits govern in-flight work, not the final corpus size. The original24-hour controller deadline is preserved. This initial operational window is bounded and must be renewed through the existing monitor with an explicit new run receipt; the service never silently extends its own authority or invents completion.

## Required encoder roster

The current model consumes **Kumo Small only**. It does not consume eight encoders. No PriorLabs3.5 or other unused/commercial encoder is scheduled to fill idle GPU time. A future encoder change requires a new model/input version and measured benefit.

Old immutable generation metadata contains a historical admission description mentioning eight encoders. That prose is not an execution roster. The v2 pipeline configuration and per-variant publication receipt explicitly declare `required_encoders: ["kumo_tabular_s"]`; they supersede that historical description without rewriting already published evidence. The generator's exact deterministic metadata is preserved so existing native/encoder verification continues to work.

## First prediction without an initial GP panel

A new variant needs verified raw data and the required encoding to enter inference. It does **not** need eight precomputed GP scores. The first model proposal is durably recorded with its exact checkpoint, context, canonical program and evaluation identity. If its exact measurement is absent, the CPU evaluator must run it and publish the result. Later proposals reuse exact cache matches and otherwise add new measurements.

A model guess is never treated as a measured target. The first measured program can supply quality, feasibility and runtime supervision. It is not automatically a cross-entropy teacher merely because it is the only candidate tested. Pairwise/listwise loss needs multiple comparable measured candidates within the same data run. Unsupported objective terms are masked, never replaced by a fabricated zero reward. This prevents a self-imitation loop from being mistaken for improvement.

## Data diversity and leakage controls

The qualified starting pool contains1,068 parent lineages. Generation rotates across parents before adding a second variant of any parent. The current transform protocol allows feature removal, independent nuisance columns and0–5% target noise calibrated on designated inner-fit rows. It preserves the parent's scoring split. Unknown categorical semantics prevent indiscriminate Gaussian feature noise. Every variant receives new content and target identities when appropriate and fresh encoding/GP measurements.

Descendants remain in their parent's training or validation partition. The approximately95/5 assignment is by lineage, not a random cut of individual siblings. Validation descendants never enter optimizer training.100,000 variants would not constitute100,000 independent source families; papers must report both counts. Increasing genuinely different original datasets remains valuable.

The current recipe preserves existing inner folds; new random fold generation is a separate future protocol, not a claimed feature of v2. New folds must be recorded in their exact identity, preserve outer held-out rows, and receive fresh encodings and GP scores as required by the model/evaluator protocol.

## Compute allocation and useful throughput

- xbabe1: required Kumo GPU encoding with resident weights across a packet, plus separately bounded CPU evaluation where memory allows.
- xbabe2: current model training, checkpoint-based inference, queue coordination and raw S3 publication.
- xbabe3: parallel native GP evaluation and available training/inference capacity as individually qualified.

Measure contexts encoded/second, publication latency, unique measured GP evaluations/second and trained contexts/second. GPU memory occupancy and CPU utilization are supporting diagnostics, not success metrics. Current native training profiling found decoding/checkpoint/I/O time larger than optimizer compute; allocating memory alone cannot fix those waits. Preserve host headroom and Grok ownership while increasing useful concurrent work.

## Current training and interpretation

The fresh native run loads6,857 original training contexts and263 reserved validation contexts. The fixed16-train/16-validation monitoring panels have both improved; full validation uses all263 at epoch completion.5,461 additional eligible original contexts still need the appropriate input/label joins. Lazy per-context program windows do not imply all millions of stored program records have already been consumed.

The dashboard distinguishes original inventory, augmented S3 inputs, required encodings in S3, measured GP outcomes, pending proposals, frozen-snapshot admission and actual unique optimizer exposure. Previous13-context qualification results remain archived separately. No pending proposal contributes to wins or Elo.

## Ownership, checks and next handoff

Source work belongs in the scoped `deploy/dl-flywheel/` release proposal and PR277. Canonical Grok work, legacy outputs and immutable S3 evidence are preserved. Root owns growth/publication and dashboard; the encoder owner writes only its packet/lease and verified-stage tables; the native resolver owns actual evaluations; snapshot admission owns measured-target acceptance.

The growth controller is `work/flywheel-20261008/proposals/growth-pipeline-v2/growth_pipeline.py` in the local working packet. It reuses the original generator and exact-version publisher, replaces only the backlog policy and explicitly extends publication to sequences after127. Unit tests cover the old128-item stall, raw/encoding backpressure, reserved unknown work, incorrect stage identities, target bounds, parent rotation, rollback and new sequence publication. Deployment requires read-only database identity/state validation before stopping the old owned controller, then a repeated check under the new controller lock.

Completion means observed S3 counts rise, only KumoSmall is encoded, a label-free new input produces a model proposal, that proposal is actually measured and saved, and its supported measured targets enter a later optimizer snapshot. A staged worker, generated file or promising loss curve alone does not establish that full loop.
