# Exact finite-support arena v1

`arena_exact` computes development-only best-of-B quantities for an authenticated, normalized,
finite discrete law when draws are IID with replacement and the entire declared law is measured
in the matching I29 universe. Rational integer masses are summed against one denominator. Duplicate
canonical actions with equal outcomes aggregate; conflicting outcomes are rejected. Failed actions
retain explicit probability mass and never become score zero.

For ordered score atoms, the implementation uses cumulative mass differences
`F_i^B - F_(i-1)^B`, with log/`expm1` arithmetic for close probabilities. Missing support is
reported as missing mass and yields `partial`; no renormalized conditional result is called full
policy exact. B=0 returns the declared incumbent or `unavailable`. Adaptive, correlated,
without-replacement, deduplicated, and selected-handoff modes are unsupported pending separately
validated algorithms. A production verifier must authenticate the I06 frozen law receipt and bind
I29 universe, checkpoint and evaluator identities; the local verifier trait is a draft seam.

`DiscoveryAuc` requires a named linear or `log1p` clock, units, common horizon and incumbent. It uses
right-continuous step interpolation, records failures and censoring, and is explicitly diagnostic;
D26 endpoint adoption and I16 clock integration remain unresolved. These outputs have no live
promotion, generalization, serving, or next-actor authority.
