# Q1 outcome-blind pre-run amendment R19

Authority: ENG:watcher, owner-authorized ruling R19 in `lanes_t0915.md`,
2026-11-01 16:15 UTC. Q1 has **NOT RUN**; no Q1 outcome has been observed.
This amendment must be committed before any qualification run. It does not sign
a plan, qualify an evaluator, consume a ledger or change a guidance default.

## Exhaustive census and original minimum

The original preregistration requires at least **five distinct families** in each
small, large, narrow and wide subgroup. Wide remains `features > 20`.
The outcome-blind R18 census exhausted all 28 P3 `sealed_excluded` families and
395 typed identity keys against the pinned regression catalog and training closure.
Its three eligible catalog matches were already among the four selected wide
families. There is no fifth eligible distinct wide family. RECS2020v7 is already
selected; adding it again fails duplicate dataset/family validation. Gina is in
the frozen r782 training closure; mfeat overlaps it. Neither can be reclassified.

The exact externally retained receipts are:

- `evidence/QLT-04/res03-t0840/R18-AUDIT.json`, SHA-256
  `f0fd94ed3848dc7af06ee0e25e08f41ea73828b2cf3387484c53a34c8dc2f242`.
- `evidence/QLT-04/qlt02-t0840/R18-VERIFICATION.json`, SHA-256
  `234a47d271a3728ac3f92904098ef97d9e7d13ac281aac587e61584e38075f0d`.

These paths are relative to the host's `jope_plan` evidence root. Both receipts
match all twelve census pins and conclude that no new eligible family exists.

## Run-only amendment

For this fixed r782 Q1 run, the signed plan may set
`minimum_wide_families: 4`. The existing 153-cut roster stays byte-identical:
[`ab434d145b8cec60399b35c35a1ce9551a881843e8136b1c757e9c93ae62bd9a.json`](rosters/ab434d145b8cec60399b35c35a1ce9551a881843e8136b1c757e9c93ae62bd9a.json),
SHA-256 `ab434d145b8cec60399b35c35a1ce9551a881843e8136b1c757e9c93ae62bd9a`.
Do not add, duplicate, drop or rebind any family or cut. Missing authoritative
identities remain disqualifying; this amendment supplies none.

The wide subgroup row in [REPORT.md](REPORT.md) must be labelled exactly:

> 4/5 families, below preregistered minimum, descriptive only, no subgroup verdict

Wide estimates and their existing intervals may be reported descriptively. They
receive no subgroup PASS/FAIL verdict and do not enter the subgroup verdict
conjunction. The small, large and narrow subgroups retain the minimum of five,
the -0.02 non-inferiority margin and lower quantile `0.05 / 4`; the correction is
not recalculated to three. An overall amended Q1 decision must identify R19 and
must not be presented as a wide subgroup verdict.

## Unchanged requirements

The primary median dataset-level normalized effect estimate, 10,000 paired
resamples, seed 20261030, percentile 95% interval, strictly positive primary lower
endpoint, all other thresholds and the smallest-eligible-dataset-SHA within-family
selection rule remain unchanged. Evaluation seeds stay **0, 1, 2**; each policy
retains **24 charged admissions** and **600 seconds** per dataset/seed. CPU limits,
three-fold cuts, complete-panel requirements, failure accounting and single-use
ledger enforcement remain unchanged. No budget, retry or campaign deadline resets.

Candidate r782 bundle SHA-256:
`94fd9587675c7806bd0a0582b6e97540cd2fdc7ebbc51958411b905185cad85c`.
Frozen DOPE SHA-256:
`c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814`.
All remaining dataset, cut, training-closure, backend, evaluator, vocabulary and
protocol pins stay unchanged. Only the separately pinned evaluator may open sealed
rows. All identity-closure and actual runtime qualification gates still apply.
The final source commit binds this amendment; the signature still binds the entire
exact plan, preregistration and backend. API01 files at most one `kind=q1` request
only after actual readiness is `signable:true`; ENG signs. API01 never signs.
