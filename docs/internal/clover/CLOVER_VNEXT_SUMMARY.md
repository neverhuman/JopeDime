# Jailgun flywheel run e7a8a34d — digest

Status (updated 13:50 MT): run COMPLETED. All stages finished on klara-jonsson-data-gmail-com:
explore 5/5, compare, synthesize, critique, revise. Final concept:
final/export/artifacts/febcc196-1b7b-4033-bb8c-8195f579017e/final.md (`jailgun runs result` in
final/runs-result.json). Revise input left out comparison.json (approved patch); the synthesis already
incorporates the comparison.

## FINAL revised recommendation (revise stage, final.md)
"CLOVER vNext: Qualified Discovery with Learned Evaluation Allocation"
1. **Split the system.** CLOVER-Discover is a qualified program-discovery engine: conventional GP,
   reproducible execution, verification. It must be useful with no neural controller. CLOVER-Learn is a
   learned evaluation-allocation controller, accepted only if controlled experiments show incremental
   value over Discover.
   Decision: "Adopt CLOVER-Discover as the foundational architecture; treat CLOVER-Learn as falsifiable
   research extensions."
2. **Three separate hypotheses:**
   - H1: GP repeatedly finds useful programs outside the 6,640-program trie.
   - H2: a learned selector picks better programs from identical frozen pools.
   - H3: fresh feedback beats frozen, offline-replay and restart-only controls.
   - Conditional H4: learned guidance improves GP generation itself.
   If only H1 holds, ship a verified GP engine and don't claim an RL flywheel.
3. **RL / self-play.** No self-improving-RL claim until fresh feedback causally improves independently
   confirmed selection beyond the strongest non-learning baseline. PolicyLM, outcome distillation, edit-
   gain modeling, RLOO/PKPO and preference optimization are candidates, not commitments. They compete
   only after a simple outcome-predictor ranking proves useful. AWR-style targets on a selectively
   measured archive are not unbiased.
4. **Loss and learning contract:**
   - Frozen candidate pools; log behavior propensities (no fictitious propensities for deterministic
     rules). Randomized exploration sized by an explicit cost/coverage tradeoff, not a fixed percentage.
   - Model three targets separately: qualified reward, execution status (fail/timeout/parity), and
     resource cost.
   - Freeze reward normalization before validation; calibrate per family and incumbent-strength stratum.
   - Illustrative acquisition score: E[(Q-h)+] - lambda(b)·E[cost] (stated as a heuristic).
   - New support requires a positive-support reference and a new versioned stability protocol; the old
     trie KL threshold does not carry over.
5. **Metrics ("game" design).** Keep best-of-8 discovery surplus at 8 charged attempts, but alongside
   independent selected-program performance and resource-adjusted performance. Experiments:
   - A: does learning select better programs?
   - B: does neural guidance improve discovery vs ordinary GP with the same operators and resources?
   - C: does fresh feedback cause improvement?
6. **Phases 0-5 are gates, not commitments:** comparability, then useful support expansion, then ranking
   learnability, then fresh-feedback value, then guided discovery, then independent qualification and
   handoff. There is no requirement to advance through all of them.
7. **S3 / storage:**
   - Immutable content-addressed blobs and complete manifests with verified read-back and manifest-last
     publication. Protocol-aware caching. Persistent fail-closed HOLD.
   - Treat the stricter spec caps (25 GB/day, 1.5M PUTs/day vs 200 GB / 2M in code) as planning limits
     until the owner resolves the conflict.
   - No deletion, expiry or lifecycle change is authorized. A 2x throughput gain is a target, not a fact.
8. **First P0 work:** a single experiment-identity contract; an audit of evaluator quantile semantics and
   encoder label boundaries; fault-tested HOLD and recovery. Falsification rule example: if conventional
   GP meets or beats guided GP on quality and cost, keep ordinary GP.

## Final-stage outputs (compare -> synthesize -> critique)
- **Compare ranking** (model judgments; weights Usefulness 30, Feasibility 25, Novelty 20, Evidence 15,
  Risk 10): C5 81.55, C4 81.15, C3 81.00, C1 75.75, C2 73.30. Synthesis notes the top three are within
  0.55 and not meaningfully ordered.
- **Synthesis, "CLOVER vNext: verified, budgeted, expanding-support expert iteration":** C5's governance
  and verification, C4's budget-first pilots, C3's expanding-support learning, and C1's executable handoff
  plus an incumbent-retain/challenger/unresolved decision. C2's edit-gain model is a conditional research
  extension. Search comes before RL: prove bounded GP finds qualified non-trie programs first. Then use
  sampler-consistent measured-outcome distillation with a positive-support reference. RLOO and
  best-of-eight policy gradients are optional. Six control arms: frozen, expanded GP, support-only,
  restart, feedback, policy-seeded GP. Staged 0-4.
- **Rejected/deferred by synthesis:**
  - Rejected: more DPO sweeps on the old trie; L6 as the success metric; synthetic tasks as final
    evidence; automated S3 expiry.
  - Deferred: immediate on-policy RL; MCTS over token prefixes (no intermediate rewards); self-play and
    checkpoint leagues.
- **Critique (latest model guidance):**
  - The synthesis is a strong governance framework, but its algorithm is underdefined and its causal
    design is confounded.
  - Primary fix: split CLOVER into (a) a qualified program-discovery engine and (b) a learned search
    controller (evaluation allocation). Establish (a) before claiming (b) adds value.
  - Use four separated authorities: discovery, verification, learning, and judging.
  - Separate the generator, support registry, and selector.
  - Training: freeze support per round, log selection propensities, reserve randomized exploration, and
    train a calibrated outcome predictor with outcome-status masking. Compare against a non-policy-
    gradient selector trained on the same data.
  - Split J8 into three metrics: discovery surplus at 8 charged attempts, independent selected-program
    loss difference, and resource-adjusted (time/cost) performance. Specify how replacement and
    duplicates are charged.
  - Objective hierarchy: independently evaluated selected-program quality under a preregistered budget
    comes first. Then feedback beyond support/GP. Then operational delivery. Edit-gain, sequential
    acquisition and richer RL are exploratory.
  - Explicit stop rules, for example: if ordinary GP matches policy-guided GP at equal budget, keep GP.
    If the feedback learner does not beat restart-only, don't claim a flywheel.
  - "Permit the simplest successful system to remain the final architecture."

## Explore-stage digest (candidate-1..5)

## Cross-tab consensus
1. **RL/self-play: don't call it AlphaZero/MuZero.** Tabs 1, 4, 5 agree the system is a one-step
   contextual bandit over a finite program catalog; "multi-step self-play would not add implementation
   value." Start as "amortized verified search" / bounded batch expert iteration. Defer edit-MDP,
   learned dynamics, tree search, and checkpoint leagues until a demonstrated exploration problem exists.
2. **Main bottleneck is fixed action support, not the optimizer.** Tabs 2–5: add bounded typed GP
   mutations / beyond-trie programs (tab 2: learn on verified parent→child *edits*, predicting
   edit-gain distributions with a fixed exploration share). Separate the effects of support expansion,
   LR restart, feedback learning, and search seeding with matched controls (frozen v1, support-only v1,
   CONTROL vs LIVE learners, GP-only).
3. **Loss design.** Keep incumbent-relative best-of-eight as the headline search objective, plus
   independent outer/held-out predictive confirmation of the program actually selected (tabs 2, 3, 5).
   Fix the train-sample vs sampler likelihood mismatch; forward-KL to the old reference is infinite once
   new programs get mass, so use a finite expanded/positive-support reference (tabs 2, 3, 5). Verify the
   best-of-budget gradient by exhaustive finite-action enumeration (tab 5 flags B=8 vs G=16 group mismatch).
   L6, preference loss, KL compliance, and byte novelty must not stand in for search performance;
   reported DPO configs violated the KL guard (tab 5). Grammar-normalized CE stays auxiliary (tab 4).
4. **"Game"/curriculum design.** Treat it as a curriculum, not two-player self-play; measure everything
   against the registered real incumbent. Freeze expanded support per paired round. Pilot caps from tab 4:
   ≤512 TRAIN-only discovery attempts, 6 GPU-hours, 2048 extra scoring attempts before any sealed judge.
5. **Evaluation integrity first.** Reconcile experiment/evaluator identities into one authoritative
   campaign manifest; repair quantile semantics with fixtures and native-Rust parity; keep UNKNOWN
   outcomes and failures; use genuinely unexposed dataset families. The reported 881-context power figure
   is not sufficient on its own (all 5 tabs).
6. **S3 / storage schema.** Tab 4: reported rate × footprint ≈ 115.3 GB/day vs a 25 GB/day spec cap — a
   binding contradiction. Recommendations: immutable, role-separated evidence shards/packs, manifest-last
   publication with verified read-back, exact-version custody, checkpoint archival; instrument storage
   ops; no deletion/retention change without separate authorization. Packing cuts request counts, not bytes.
7. **Product framing / fallback.** Tab 1: ship a budget-capped program-selection workbench for ML
   engineers (candidate-recommended / incumbent-retained / unresolved). Tab 4: if learning adds no value,
   ship verified GP search without a learner.

Caveat repeated by every tab: all system/performance facts are "reported" from the input snapshot and
were not independently verified.
