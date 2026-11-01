# RNK-10 matched CPU acceptance (TRAIN diagnostic only)

**State:** prospective manifest. No terminal outcomes have been read for this protocol.
**Owner:** `luna_v1_rnk11`. **Partner:** `luna_v1_res08` may reuse the existing verified CPU fixture closure without copying it.
**Deadline:** 2026-11-01 02:30 UTC. **Candidate:** the frozen `fad792d982ac` research adapter, bytes SHA-256 `ce39c75f37624e64dfbf89e6493adbf74811a5248f9355997161491e93a132ef`, with DOPE identity `c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814`.

The candidate is research-only and is not a signed bundle, `VerifiedBundle`, production artifact, qualification decision, or enablement grant. The previously approved r882 checkpoint import remains a separate research-only option; it is not mixed into this single-candidate run. This admission uses the already present frozen `fad792` bytes so the partner can reuse its verified CPU fixture closure unchanged.

## Fixed cohort and arms

The immutable machine-readable roster is `evaluation_manifest.json` (SHA-256 recorded in the owner handoff). It binds the RES-07 packet index `27905f29703431bedacac90eed1bd54257fd0c59bdf24b56605d0b9416ae1393`, capture manifest `bbc89c29a43c4137a4759bb97ff10e49bbcbf6113bbbb9061e2658f3c0a55a4c`, selected TRAIN metadata `981960ad628f5029f8f54552c7b1cb1d440072e7e9dcffecdea52c6a3493a0f1`, and S3 dataset catalog `e5f732fd784bf67bfc16f4173c585b0861e6e8a6026808df857097a721bbf0ba`. There are 50 `DL-Train` regression contexts from source split `train` or `inner`, 16 families, all mapped to the real S3 catalog; each context has fixed width-12 and width-20 slates. No DL-Val, outer-test, holdout, sealed, or meta-test payload is in scope.

For every context/slate, slots 0–3 are the already-revealed common history. The **guided** arm ranks all remaining slate entries with the pinned frozen adapter; the **unguided** arm uses frozen `anchor_cost` values returned by the same CPU adapter and selects the four lowest-cost unrevealed entries. Both arms use the same slate, context features, revealed history, candidate support, gate rules, and outcome read. No arm, seed, slate, cohort member, or threshold may be selected after outcomes are seen.

## Outcomes, read ledger, and endpoints

`outcome_ledger.json` reserves one fresh terminal outcome query for the union of the two arms' chosen unrevealed entries on each of 100 slates (maximum 800 unique candidate outcomes; duplicates across arms are fetched once). The initial four revealed outcomes are conditioning inputs already present in the immutable packet and are not new outcome reads. The ledger starts `UNUSED`; the partner must atomically change it to `CONSUMED` before the first terminal outcome request and record exact selected candidate IDs, S3 object/version identities, returned byte hashes, and UTC timestamps. A failed or partial run is `INCOMPLETE`; the budget cannot be reset or repeated.

For each arm/slate, U4 is the maximum of the true utility over the four common revealed entries and that arm's four selected unrevealed entries; utility is `-fitness` for gate-passing outcomes and `-2.0` for gate failures. Average the two slate widths within context, contexts within family, then give each of the 16 families equal weight. Paired delta is guided minus unguided. Report every context/slate and family result, not only aggregates.

Pre-registered gates (all required):

1. Maximum per-slate guided-to-frozen-anchor proposal KL is at most `0.35`, as computed by the pinned CPU adapter using unit-temperature `softmax(-cost)` over all unrevealed proposals.
2. Median of the 16 family-mean U4 deltas is strictly greater than zero.
3. A deterministic 10,000-draw family bootstrap (NumPy `default_rng(181100)`, equal-probability sampling of 16 whole families with replacement) has a two-sided 97.5% percentile interval whose lower bound is strictly greater than zero.
4. Family-grouped Bradley–Terry Elo for guided versus unguided, equal family weight and anchor 1500, is strictly greater than 1500. Record method/source/version and full inputs.

If any gate fails, report `NOT_QUALIFIED`; no threshold changes, reruns, tuning, or adoption follow. Even a pass is only a TRAIN diagnostic and does not establish generalization, S4/M5/Q1, serving, production, or default enablement. No optimizer updates are permitted.

## Evaluator and resource contract

The exact candidate, packet, cohort, slate identities, and source pins are in `evaluation_manifest.json`. The CPU admission implementation is pinned to JopeDime source `8cc068adee08f45290ee1d0d69e1b6bc5f7aa0de` (`crates/jope-service/src/ranker/admission.rs` SHA-256 `b433a1ee533b02d3ad108f5a85932e9a38d4ebd86b53c87519b2441e12aca039`; `crates/jope-service/tests/support/cpu_parity.rs` SHA-256 `20ec633061cef38e2fa4f9b7083fac2c3907a9ea2722e74eb20b2069d806f9f6`). Utility definition is pinned to `research/flywheel_selfplay/h48/run.py` SHA-256 `756960a0e2bfdf23e75012fb347ce772fb6fdeaa4fb4fc2d4178dfaeb9eac9da`; the pinned source computes `proposal_kl_to_anchor` at unit temperature. The partner must record the exact invoked CPU runner path and hash before consuming the ledger; no substitute evaluator is admitted.

Run one time only, CPU-only, `nice -n 10`, one worker, at most one math thread (`OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`), RSS at most 6 GiB, host CPU utilization at most 85%, wall time at most 45 minutes and absolute deadline 2026-11-01 02:30 UTC. Stop before the deadline if the exact evaluator cannot be pinned or any cap is exceeded. No GPU, optimizer, qualification lock, sealed/holdout/meta-test access, live-folder writes, signer/key operation, or retry.

Historical CPU composition evidence is limited to proposal-cost/admission parity and the first four already-revealed inputs; its `Q1_ready=false` and `quality_gate_granted=false` remain unchanged. The new ledger is solely for unrevealed terminal outcomes and must be consumed once as described above.
