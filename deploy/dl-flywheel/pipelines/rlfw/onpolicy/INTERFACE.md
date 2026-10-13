# C4 / PR-12 and S7 / S8 interface

Build-only pure libraries. CL owns runner_v6 integration, L6, L_PI and the KL
anchor. These modules import no trainer, optimizer, model, cloud client or GPU
runtime. They do not launch collection or change any incumbent ledger.

1. At a round boundary, seal a `PolicyIdentity` (checkpoint, trie, vocabulary,
   normalizer, generation protocol, temperature). Use dropout-off, float64,
   summed EOS-inclusive `TrieLogProb` readouts. Predeclare each `GroupSpec` with
   every iid-with-replacement draw, its distinct draw ID and behavior readout,
   generation update and monotonic time. Repeated program SHAs remain repeated
   draws. Freeze a verified incumbent and the identical sample/incumbent utility
   rule. Current `HWMKey` is reused; a future key migration needs an adapter.
   All groups for a context in that round must share the identical parent,
   comparison and incumbent publication identity, including NULL. Distinct
   contexts may have distinct incumbents.
2. `verify_q(key, publication, parent=..., role_check=..., verify_closure=...)`
   admits only measured, qualified-complete integer q_grid. Callbacks MUST freshly
   verify TRAIN role/parent and actual seven-body closure, q recomputation, parity
   and all comparison/program/execution joins. The library cannot verify remote
   bytes; a flag or head-only metadata check is not a production adapter. Callback
   exceptions fail closed. Synthetic verification appears only in test fixtures.
3. Deliver immutable outcomes to `RoundCollector.resolve`. Unqualified failures,
   parity, infrastructure unknowns and incomplete groups are quarantined WHOLE;
   no retry, shrink, floor reward or retroactive zero. This applies the owner's
   N-1 requirement strictly, even where older plan text proposed failure utility
   or replacing parity draws with h. `close()` consumes the round once. The runner
   must persist round IDs, outcomes, quarantine and optimizer/checkpoint CAS in
   its own durable journal; this in-memory library makes no crash-recovery claim.
4. Recompute current draw likelihoods and actor→learner probe KL. Call `objective`
   with current checkpoint, update and monotonic seconds from the SAME clock
   domain and a fresh `role_check(spec)` callback before each actor step.
   Returned `gate` includes lag_updates, lag_seconds, ratios, ESS,
   joint_ratio, correction and refusal reason. Every changed checkpoint uses exact
   IS, including lag 1. Support/T/normalizer/protocol mismatch, any ratio outside
   inclusive [0.5,2], ESS < G/2, or probe KL >=0.05 refuses the entire actor group.
   Outside 64 updates / 180 s, exact IS is still permitted by spec §4.5/§8.
   Historical/unknown-μ, dedup, rerank and mutation samples never enter this API;
   they may be routed to L_PI/L6 through CL's own eligibility checks.
5. `terms.losses(differentiable_current_logprobs)` returns `(L_RLOO,L_HWM)` with
   frozen float coefficients; only the log-probabilities retain gradients. Dense
   RLOO uses per-item IS and independent leave-one-out baselines. G=1 explicitly
   uses fixed-zero REINFORCE because a leave-one-out mean is undefined. PKPO uses
   joint IS across all G raw draws, NOT per-item IS. No std normalization, ratio
   clipping or self-normalization. Raw q stays intact; samples and h are clipped
   identically to +/-log(1000) by default. Solved clipped incumbents and absent
   verified incumbents give connected-zero HWM support.
6. PKPO weights already normalize by C(G,B). Their sum estimates ∇J_B exactly;
   the spec's literal marginal loss divided by G instead estimates ∇J_B/G.
   For G=B, PKPO equals the marginal advantage, WITHOUT that extra divisor.
   Keep this scale explicit when selecting λ_PG and λ_HWM. Gate selection can
   bias the admitted distribution; unbiasedness is the unfiltered estimator
   identity, not a promise about gated/censored runtime groups.
7. Reduce supported group losses by context, then parent with
   `parent_macro_mean`. Keep separate support masks for dense and HWM losses.
   The runner combines these with unchanged L6/L_PI/KL and records gradient
   cosine/norm ratios. KL probe >0.03 is an alarm; >=0.05 is a hard stop here.
   Entropy/effective-support and distinct-SHA alarms remain CL/metrics integration
   work, not a new entropy loss.
8. `ReplaySampler.batch` returns one parent per batch using four exact 25% quota
   blocks, selecting parent before context before row. Consolidate a row's bucket
   memberships. The caller supplies recent-gain/hard-pair/exploration evidence
   and an explicit rolling family batch cap/window, and rechecks role per call.
   `ContextScheduler.next` selects exact ten-slot blocks of 5/3/1/1, balanced by
   parent within pool. No-winner contexts retain NULL h. Development is explicitly
   `optimizer_allowed=False` and dispatches only to eval. Missing/capped pools
   raise `SamplingUnavailable` and retain their quota; no fallback or silent
   redistribution. Persist scheduler/quota RNG state in the runner before using
   this pure in-memory API for a resumable campaign.

The numerical tests exhaust three-action ordered groups via exact multinomial
counts (including G=16), compare an independent CDF derivative and finite
differences at 1e-9, compare sorted PKPO with literal subsets, and check seeded MC
against its measured standard error. Finite MC is NOT claimed exact to 1e-9.
Equal-budget arena lift versus L_PI and live update/throughput qualification are
unmeasured and remain phase-2 gates; fixture success does not authorize launch.

## GB-03 archive and frozen-round actor updates

archive.archive_target constructs a detached reward-directed teacher from
VerifiedQ outcomes for ONE comparable TRAIN context. Duplicate programs are
collapsed; conflicting scores are refused. The default reference is uniform
over deduplicated qualified programs (alpha0). A supplied mapping of frozen
sampler sequence likelihoods plus its PolicyIdentity selects the distinct
frozen-policy-conditioned reference (alpha1). No adaptive archive frequency is
treated as an iid propensity. Raw q is retained, centering cancels, and the
exponent is not clipped. ArchiveTarget.loss checks fresh TRAIN role and exact
summed EOS-inclusive differentiable sampler readouts. New support needs a
matched-support reference; no cross-support forward KL is authorized here.
ArchiveTarget construction (including dataclasses.replace) accepts only a
tuple of built-in int/float coefficients and snapshots them to Python floats.
Tensor coefficients are refused even when detached, so neither an autograd
graph nor a mutable tensor alias can change or train the frozen teacher.

frozen_round.FrozenRoundStep is the exact frozen branch: pass immutable raw
GroupSpec draws and an EXPLICIT scope, either full_support_iid or
measured_leaf_pi_M. The latter describes restricted historical measured-leaf
sampling, not full-support on-policy learning. Resolve every draw with VerifiedQ
or a declared quarantine outcome. step probes actual current actor parameters,
optimizer-update counter, support and dropout through ActorState before and
after likelihood recomputation. Drift is refused rather than repaired by IS
inside this exact branch. The original S7 objective remains available separately.

Any quarantined/incomplete group blocks the ENTIRE wave: no partial-wave
renormalization or action-dependent complete-case update. An eligible wave
reduces groups by context and parent, builds detached RLOO/PKPO coefficients and
invokes apply_update(loss) at most once. Closure or a callback exception consumes
the round. The caller owns actual optimizer/parameter-hash proof, durable CAS,
failure recovery and checkpoint publication; this in-memory guard is not durable
exactly-once execution. Lambda weights must retain the registered loss scale.

CPU tests use synthetic seven-body q closure, the actual PolicyLM class in a
small synthetic configuration, EOS-inclusive three-program masks and raw iid
duplicate draws. They measure actor/shared-trunk gradients, top-q probability and
exact expected toy quality after one small optimizer step. This is source/fixture
evidence, not production q qualification or unseen-family improvement.

CX-OP R10/R11/R23/R24 GPU evidence uses original ea078 weights/v1 moments and
historical native measurements on measured-leaf pi_M. It lacks production N1
Rust closure and cannot be relabeled full support. Per-term raw/applied gradients
and full-trie AND pi_M high-q probability movement remain separate observations.
Parent-cluster L6 intervals from the repeatedly inspected val263/54 parents are
development diagnostics; no qualified winner or sealed adoption follows.
