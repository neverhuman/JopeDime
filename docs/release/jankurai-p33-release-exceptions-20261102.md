# P33 exact-file Jankurai exception

**Owner authorization:** Jepson, delivered by Grok Bot in `gbw_2122_luna_jk_release.md` (2026-11-02). **Auditor:** Jankurai 1.7.6. **Review by:** 2027-02-01.

This exception covers one exact documentation file: `docs/testing.md`. It addresses HLT-025 and HLT-026, which the full auditor continued to report at that path after P33 removed the cost-budget and release-readiness sections. The committed guide is a test-command and review-evidence guide; it does not make a per-run spend claim or attest to a release candidate. The full audit on `fe5d7593cf8ca84f20f680e42167efca6bfb96b5` is retained at `/home/ubuntu/jope_plan/evidence/jankurai/P33/full-audit-fe5d759.json` and records the two path-level findings despite the absence of those topics in the file.

Operational resource limits and per-run evidence remain governed by the agent contract and the run's own receipt. Release procedures and candidate evidence remain in the release runbooks. This exception does not claim that any run is authorized or any release is ready.

The auditor excludes only `docs/testing.md` from Jankurai scoring. It does not suppress other Jankurai paths, Gitleaks, records gates, link checks, or CI tests. Remove this exact-file entry if the guide later makes run-specific cost claims or attests to a release candidate, and rerun the full audit.
