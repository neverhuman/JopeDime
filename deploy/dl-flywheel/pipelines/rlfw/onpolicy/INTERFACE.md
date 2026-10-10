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
