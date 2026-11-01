# QLT-04 wide-family gap audit — T1210

Time: 2026-11-01 19:23 UTC
Candidate: PR #1620 exact source head `856b9c0b7a4302c110d9480925189caa46a62408` (now merged).
Method: metadata-only; no dataset payload, S3 GET, Q1 run, or sealed/holdout contents read.

## Findings for the two owner-ruling candidates

### R1 — mfeat-zernike (`0380f8df655797ba`)

The catalog metadata identifies this as a regression dataset with 47 features and canonical SHA-256 `0380f8df655797ba171b61995645ae435b05da2d6484ec04cffd95064652ae02`; its recorded dedupe fingerprint is `fe5aadc8442944397afad2210522900d6ab17fb8`. It is absent from the pinned 153-record source packet and therefore absent from the 97-family PR #1620 selection. More decisively, the exact-head R9 review derived all five typed identity keys with pinned `family_keys.py` and found overlap with the frozen r782 training closure on canonical SHA, dataset hash, dedupe fingerprint, normalized URL, and OpenML ID. The family key itself is in that closure. Thus it fails R1/R9's zero-overlap eligibility condition; the earlier check that considered only a subset of identity domains was incomplete. Do not add it as an independent wide family.

Evidence: `/home/ubuntu/jope_plan/evidence/QLT-04/qlt02-t0815/R9-FINAL-REVIEW.json` (all pins match; explicitly corrects the earlier incomplete mfeat check); `/home/ubuntu/jope_plan/evidence/QLT-04/gov03-t0545/R1-MFEAT-METADATA-CHECK.json`; catalog and code pins in those receipts match catalog SHA-256 `e5f732fd784bf67bfc16f4173c585b0861e6e8a6026808df857097a721bbf0ba` and `family_keys.py` SHA-256 `9b688e9b8a36a32485524243dc1adea7d30ee7404df4027d2b4d17bd6a900f7e`.

### R14 — RECS 2020 v7 (`8d5f6d3120d2cbc2`)

This is a 400-feature regression dataset with canonical SHA-256 `8d5f6d3120d2cbc2535151c7d455e7a536ca9768f773e15edd45932ec38b3592`, recorded family key `3e82de5441dd180574e99eaab8dbf2f78f91ca0e`, and accepted P3 owner-family identity `fam-66943d2aa93bc6881f66b3f351afa5f67e4cbfcd0eafd99633e852c4fc31e4eb`. R14's record is not itself selected because PR #1620 selects the smallest dataset SHA per frozen owner family. Its same-family RECS 2015 representative (`12fd2dc4af621833`, SHA-256 `12fd2dc4af6218333aa6a3ca16676cfc8b5bdaa1353083776fc4b8046562e05c`) is selected instead. Both have 400 features. The RECS family is therefore already counted once among the three selected wide families; R14 cannot add a fourth distinct family.

Evidence: exact PR #1620 selection file `PR1620-files/10-ed9d27ede248679fe2a651a5f37e37babca4efd07a2db600e0dbf35121c56c21.json` in `/home/ubuntu/jope_plan/evidence/QLT-04/qlt02-t1145/`; R14 admission `/home/ubuntu/jope_plan/evidence/QLT-04/api01-t0830/R14-ADMISSION.json` and verification `/home/ubuntu/jope_plan/evidence/QLT-04/qlt02-t0830/R14-VERIFICATION.json`.

## R9 fallback outcome

Applied the pinned catalog-order/P3 sealed-family census result from R18, which is the R9 fallback's exhaustive check for another wide regression family with an accepted family key and zero overlap against the frozen r782 training closure. The census is pinned to catalog SHA-256 `e5f732fd784bf67bfc16f4173c585b0861e6e8a6026808df857097a721bbf0ba`, training-closure file SHA-256 `8e786b2dcb153a5c1c4f43dbcb773b53df6f0dd7d325c94f8d7ec2f73352d3e8`, and `family_keys.py` SHA-256 `9b688e9b8a36a32485524243dc1adea7d30ee7404df4027d2b4d17bd6a900f7e`. Its catalog-order census found only three accepted-key wide regression matches in the pinned sealed-family metadata: RECS 2020, Wang aqueous-solubility, and RECS 2015. Each belongs to a family already represented by the source roster's wide set. There is no next distinct eligible family after those existing families are excluded.

Evidence: `/home/ubuntu/jope_plan/evidence/QLT-04/res03-t0840/R18-AUDIT.json` (SHA-256 `f0fd94ed3848dc7af06ee0e25e08f41ea73828b2cf3387484c53a34c8dc2f242`) and independent verification `/home/ubuntu/jope_plan/evidence/QLT-04/qlt02-t0840/R18-VERIFICATION.json` (all source pins match; zero protected-content reads). The verified R9 fallback result is **no eligible additional wide-family picks**. Accordingly, there are no two valid IDs/hashes to name; naming any would contradict the exhaustive pinned census or require relaxing a frozen identity/eligibility rule.

## Disposition / handoff

No fix to PR #1620 is warranted: its one-per-owner-family selection correctly retains RECS 2015 and drops RECS 2020 as a same-family variant. mfeat-zernike is not eligible because of training-closure overlap. The requested two additional picks cannot be supplied under R9's current pinned inputs; the 5-wide gate remains unmet (3/5), and Q1 remains `NOT_RUN`. No roster, plan, source, training closure, or owner-family identity was changed. No PR was opened and no signing request was filed.
