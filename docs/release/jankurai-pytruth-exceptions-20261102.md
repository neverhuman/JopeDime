# Jankurai pytruth exact-file exception (2026-11-02)

## HLT-005: portable Python parity oracle

`crates/feat-math/tests/portable/full_ref.py` is the Python implementation of
the portable math parity oracle. It is test/reference code outside the product
runtime. P-1B campaign receipts `crates/jope-campaigns/manifests/advanced-v3/task-receipts/P-1B/0003.v1.json`
and `0004.v1.json` pin its SHA-256 to
`62fe871b64c03b66d883a1690bb50823c17458649b878585752debb238d3c200`.
`agent/boundaries.toml` classifies this exact file as a non-product Python
reference path for Python boundary scoring. It remains visible to other
detectors: the prior `agent/audit-policy.toml` scanner exclusion was removed
because it did not affect the Python analyzer. The file and pinning receipts
are unchanged.

This exception addresses only the finding's evidence that Python appears
outside the allowed runtime roots. It does not claim that the aggregate
`python/` product-truth finding or the Python runtime-share evidence is fixed.
Keep HLT-005 open until a fresh audit no longer reports the remaining
fingerprints or an owned source boundary is implemented.

## Dead Python contract duplicate removed

On the t2015 source, `python/src/jope/contracts.py` had no tracked imports or
callers outside its own definitions. Its family split, split-view and binary
target contract implementations are now owned by `crates/jope-datasets`; the
live fused-data path already calls that implementation through
`jope._telemetry`. The unused 385-line Python duplicate was removed instead of
maintaining a second implementation. The t2015 change does not alter
`python/src/jope/data/active_fused.py` or its Rust-backed behavior.

This cleanup reduces one dead Python product-tree module. It does not establish
that either aggregate HLT-005 fingerprint has cleared; keep the cap open until
the next applicable audit confirms the result.
