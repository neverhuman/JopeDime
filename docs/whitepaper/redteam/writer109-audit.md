# Value-map and native-comparison precision

The map pools every non-padding program token, including the beginning/end tokens and fixed engine/task header. All four two-layer ranking heads share the same interaction vector; the native ranking objective acts on their mean. Separate targets or bootstrap samples are not implied.

Both writer and ranking token tables allocate 657 rows for a 401-token grammar, leaving 256 reserved rows in each table outside valid token indexing and output projection. The saved parameter total includes these rows and the zero-weight stability/loss-scale terms. It is not a count of learned or exercised parameters; no checkpoint qualification follows from a registered parameter count.

The native objective keys ranking by corpus context and eight-row comparison index. Separate comparisons from one context remain separate. Pairwise ranking skips non-finite targets and exactly equal target values. The residual report's dataset-key and epsilon-tie descriptions do not match this implementation and were not imported. The Python ranking helpers group by dataset identifier and also skip exact equality; they do not establish the report's proposed epsilon threshold.

Source: `crates/jope-policy/src/config.rs`, `model.rs`; `crates/jope-engine/src/neural.rs`, `training/objective.rs`; reference comparison `python/src/jope/model.py`, `policy/losses.py`. Full source hashes, exact edit ledger, baseline/final clean builds, reference/placeholder/asset checks and page proofs are retained in the delivery receipt.

All 40 descriptive TBDs and the full Results, abstract and Discussion/Conclusion remain unchanged. Numbered equations, prior scientific figures/data/generators, bibliography, title, table geometry and global layout controls are retained. Only the map paragraph and Training Details prose change; page reflow is recorded separately. This is a documentation clarification based on source inspection, with no experiment or new scientific observation.
