# Causal cumulative integration, version 2 / wire 60

This opt-in ordered stage implements left-rectangle and trapezoid integration. Each group starts from an explicit finite initial vector. The first observation returns that vector exactly; subsequent observations require a finite positive time gap. Width is preserved in 1..512 columns, and accumulation saturates at ±1e12 with the existing magnitude-clamp fault. Nonfinite feature inputs contribute zero with an input fault; checkpoints retain their raw words. Timestamp ties, reversals, nonfinite timestamps and invalid row shapes are refused transactionally.

Programs use additive variant 5 / feature 256 and reader version 2. The model binds its program, ordered input identity, training cut, downstream plain Ridge model and accumulator checkpoint. Only the declared training prefix fits the downstream model. Native prediction, resumed prediction, all three train/predict recipes and fitted C++ use the same ordered semantics. No new dependency or historical emitter body is introduced.

The explicit Experimental profile admits one optional stage, two modes, an all-positive-zero initial vector and a prior in [0,0.25]. Grow creates a node, Change switches its mode while preserving its actual child and input identity, and Drop restores that child. Imported non-menu initial words remain readable and removable, but Change refuses them. Disabling growth retains decoding and Drop. IID tasks, opaque recipe children and excessive metadata are refused before generation.

Per-operation acceptance covers:

- A shortest existing-tree witness: row-only lag/roll read identical current rows while the two causal histories require different integrals.
- Six synthetic training cohorts: left area, ramp area, irregular area, group exposure, pulse dose and a level control. Five paired seeds × 64 evaluations × two arms yield 3840 evaluations and 7680 training-prefix fitness fits. Prototype and selected-artifact fits are counted separately.
- Native execution, routed program/model fitting, compiled Rust/Python/C++17 recipes and fitted C++, with exact prediction, fault and checkpoint words. GP-created and changed nodes produce 20 actual export artifacts, including a removed-child path.
- Explicit numerical guards, NaN/Inf/zero rules, width bounds, saturation faults and transactional failures. The corpus includes 70 independent rational references, tiny and overflowing areas, irregular gaps, group resets, raw checkpoints, maximum width and train-to-test continuation.
- Bounded Experimental admission, lossless codecs, removal without removing parsing, train-only fit, deterministic width, and token/byte costs and paired wins/ties/losses for the utility winners.
- Additive identifiers/helpers, unchanged historical values and source identities, version-envelope rejection, direct empty-row refusal, neutral fixture names, and no new feat-math dependency.

These synthetic measurements qualify Experimental behavior. Fixed-cohort W §17 G-A through G-E and product qualification remain required before Baseline promotion.

Acceptance commands: `cargo nextest run --locked -P pr -p feat-math --test cumulative_integration_v2 --test ordered_axis_contract --test-threads 2`, strict feat-math Clippy, the standing PR and full suites, emitter portability/sanitizer checks, lane audit, exact-head automatic CI and an independent native review. Evidence is retained under `/home/ubuntu/jope-evidence/kernel2/bt/B14_CUMULATIVE_INTEGRATION/`; external preparation receipts remain distinct from shipping-head receipts. Actual workspace `just required` is reported separately, including the accepted unchanged 81-location debt when it exits 101.
