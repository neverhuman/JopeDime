# Jankurai pytruth exact-file exception (2026-11-02)

## HLT-005: portable Python parity oracle

`crates/feat-math/tests/portable/full_ref.py` is the Python implementation of
the portable math parity oracle. It is test/reference code outside the product
runtime. P-1B campaign receipts `crates/jope-campaigns/manifests/advanced-v3/task-receipts/P-1B/0003.v1.json`
and `0004.v1.json` pin its SHA-256 to
`62fe871b64c03b66d883a1690bb50823c17458649b878585752debb238d3c200`.
The exact file is excluded from language-surface scoring in
`agent/audit-policy.toml`; its contents and pinning receipts are unchanged.

This exception addresses only the finding's evidence that Python appears
outside the allowed runtime roots. It does not claim that the aggregate
`python/` product-truth finding or the Python runtime-share evidence is fixed.
Keep HLT-005 open until a fresh audit no longer reports the remaining
fingerprints or an owned source boundary is implemented.
