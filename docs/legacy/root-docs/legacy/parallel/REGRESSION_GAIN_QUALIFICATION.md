# Regression-gain qualification implementation

## Production interruption on September 9

The rollout and qualification handoff interrupted production. Qualification blocked proposal generation while the GP queue drained. The previous generation's final search committed at 22:03:11 UTC; observer receipts show zero active and queued searches from 22:03:29 through 22:17:23. Stopping qualification released the GPUs at 22:12:24, but the controller then prioritized monitoring and training before generating replacement searches. Generation 16 was enqueued at 22:17:52, and the 22:18:26 observer receipt showed 104 active searches and 316 queued. This was an operational failure, not useful qualification occupancy.

Deployment v4, committed at 22:19:14 UTC, schedules proposal generation before monitoring or training and refills below three worker waves (312 outstanding searches). It preserves an already running GPU stage. The observer now reports an empty GP queue as a warning immediately and an incident after 120 seconds, even when the worker service is alive. GP service PID 2342 was preserved during restoration. Production serving remains cycle 59, provisionally unqualified. Both qualification services are stopped with automatic restart disabled.

The first trial's four arms failed before training because of an invalid absolute initialization path. An append-only correction classifies that result as technically inconclusive; its original `rejected` receipt must not be interpreted as a negative scientific finding. The retry completed only broad replay with carried optimizer state (1,000 local steps), then was stopped during broad replay with reset optimizer state. No frozen validation comparison ran and no candidate was promoted. Partial artifacts and all 3,000 conservatively reserved GPU node-seconds remain in the shared budget ledger; reserved time is not measured GPU occupancy.

The local qualification driver now refuses to request a GPU handoff without a fresh, running GP receipt and at least 312 outstanding searches. It checks during owned training and monitoring work and yields if fewer than 208 remain. Queue depletion produces an inconclusive interruption receipt and releases the handoff. This driver change is tested but has not been launched on the production node. Production remains on the stopped qualification configuration.

Restoration verification and interruption receipts are retained under `runs/regression-gain-20260909/incident-restoration`, `deployment-v4`, `trial`, and `trial-v2`. The Python checks cover queue scheduling, current teacher reporting, promotion isolation, replay policy, label admission, resource budgets and qualification yielding: 60 tests passed locally; the deployed scheduling/reporting subset passed 43 tests on the node.

At 22:26:17 UTC, verification recorded 104 active GP searches, 277 queued, 29,209 completed searches, and 17,307,030 completed evaluations. All eight GPUs were training at 51–68% instantaneous utilization. Cycle 63 had reached step 128 on newly packed snapshot 63 (7,702,147 records); the serving checkpoint remained cycle 59. GP completed 39 searches and 39,000 evaluations after the empty-queue count of 29,170. These receipts establish restored operation; they do not establish transferable model improvement.

## Implemented safeguards and qualification components

The continuous controller separates serving checkpoints from training candidates. The legacy `checkpoint` field remains a serving alias. Migration is provisional, and it binds the serving corpus separately because the proposal runtime verifies the checkpoint/corpus pair. A training completion consumes its corpus snapshot and publishes an immutable candidate; it cannot advance serving. New training initializes from serving, including after rejected candidates.

Promotion requires a durable decision before pointer replacement. Missing, mismatched, incomplete or nonfinite evidence is inconclusive. The gate implements the declared 1% direct-proposal improvement and positive family-bootstrap lower bound against recovery and serving, the 0.5% NLL/1% ranking guards, and matched-search non-regression. Qualification remains adaptive validation. It is not an independent benchmark result.

Continuous teacher reports derive improvements from generation quality receipts and verify saved search hashes. Reports identify their source watermark, new improvements, families/cuts, discovery origins and time since improvement. Legacy reports without origin/identity fields remain marked partial. Operational status, diagnostic regression, teacher collection and proposer qualification are separate. Reporter heartbeat age and completed-but-unreported generation age are checked independently.

The archived-program inventory preserves all source rows and historical information without promoting scores to labels. Exact grammar compatibility uses the native encoder and parser without projection or numeric snapping. The first audit contains 3,904 canonical programs, 455 exactly compatible, including all 110 previously identified compatible bank programs. A further indexed read of the strict historical store found 2,880 source observations for 77 of those 110; four programs have known current-validation dataset overlap. Missing observations, unmapped family aliases, evaluator receipt verification and incomplete selection histories keep these candidates in collection-only diagnostic status. No historical labels were admitted.

Replay qualification is opt-in through the dedicated `replay_training` rank executable. Its frozen policy identifies changed training groups, alternates fresh and broad portions with family balancing, records actual fresh-group sampling, and falls back to broad replay when no fresh groups exist. Sampling depends on corpus and checkpoint lineage; interrupted runs preserve the policy and counter. Carried AdamW retains moments and bias-correction step, with explicit objective, parameter, shape and state verification. The normal production training mode is unchanged. A separate proposal executable uses the same extracted pure-v4 sampling implementation without reading corpus scores or retained teachers.

Packing instrumentation records source hashing, parsing/cache access, merging, and packing/write/verification time outside the corpus identity. Existing immutable cache tests verify byte-identical records, labels, index, provenance and vocabulary, and detect changed input content. No modification-time identity shortcut was introduced.

## Preserved evidence

Local evidence is under `parallel/runs/regression-gain-20260909`. Remote evidence is under `/home/ubuntu/jope-dime/parallel/runs/regression-gain-20260909`. Deployment directories preserve source, old state/control, unit overrides and activation receipts. Qualification attempts, including failed launches, remain separate and share a conservative budget ledger. Source builds are separate from serving binaries.

## External-drive maintenance window

No new recursive reads, remounts or disruptive recovery were attempted against `/mnt/easystore`. Accessible preserved copies remain the recovery source.

An owner-approved maintenance window is required before draining dependent processes and normal unmount/remount. At that window: inspect `/proc/self/mountinfo` and the already known mount/device configuration; identify and gracefully drain only confirmed dependent services; run a normal `umount /mnt/easystore` with a bounded command timeout; stop if busy or unresponsive; if normal unmount succeeds, remount using the recorded original device and options; perform a bounded top-level inventory before any deeper access. Preserve command results and mount configuration. Forced/lazy unmount, USB reset, unrelated process termination and reboot require separate authorization and are excluded.

## Claims still requiring experimental evidence

Training throughput, teacher collection and a completed training block do not establish proposal improvement. Context rotation and an order-sensitive reward model must remain separate experiments until the replay comparison is qualified. Two promoted discovery cycles are required before claiming that the full flywheel has demonstrated transferable gains. A negative or technically incomplete trial leaves serving unchanged.
