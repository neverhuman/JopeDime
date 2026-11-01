# Section 4–6 source recheck and live admission refresh — t2220

Source base: `b55ed85e8db81f8c6ae1415ad20417bb2c9e92ed` (updated from `7200cdc16e4701855e9a33e86941254cfe9c4825`; Opus #1312 bibliography/layout receipt carried unchanged). Read-only aggregate capture on xbabe1 at `2026-11-01T04:22:37.513756+00:00`; SHA-256 of the stable complete-line prefix: `c0ef2c4c2258962e536b3d9677cc7c9c3af9ff975bb7ef94c58793faf42c89ed` (970513 bytes, no incomplete trailing bytes). No dataset, weights, final-evaluation or protected payload files were opened; no experiment was executed.

## Evidence and interpretation

- Every one of the 1,859 consecutive admission records, rounds 775–2633, enters the new CSV and both plots. All 695 earlier committed admission rows agree field-for-field at their logged precision; 1,164 additional already-logged records extend the displayed window.
- Four acceptances remain: strict gains at 785 and 859, accepted ties at 782 and 882. All 1,751 rounds after 882 are rejected. The final recorded best is 1.1377941898164525 and its logged identity is round 882; this is not a strict gain at 882.
- One evaluation identity occurs in the captured prefix: `e79a7715c1531ccb8c4055ab96971101cd4034ec3a71d83bca6d25b0e604ca83`. This field does not by itself qualify the installed source, adapter or checkpoint. Existing source-to-run binding remains TBD.
- The gap curve uses exact decimal subtraction of each candidate from its post-decision recorded best. Every adaptive round contributes once; repeated scores are retained. Zero includes both strict gains and ties. No confidence, independent-trial, first-proposal or deployment-throughput claim is added.
- Source metadata, CSV, derived provenance, generating script and two vector PDFs are committed. The script can extract the captured aggregate prefix with its receipt, or replay directly from the committed CSV/metadata. Historical CSVs, figures, generators and provenance remain byte-identical.

## H56 recheck

Re-read/grep Sections 4–6 on current main and compare the 20 source/note pins from the preceding review. The native writer, separate ranking heads, rank-weighted parent route, grammar/parse checks, checkpoint loader, reference objective, adapter revealed-history ordering, strict program-pool admission and tie/rollback rules are unchanged. No additional writer/value-net overclaim was found after #1304. Retain the existing distinctions between reference specifications, implemented source, logged development observations and qualified deployment. The new Section 6 wording explicitly keeps the older 25-summary and operating-measurement window separate from the extended admission prefix.

## Ownership and validation

Only Section 6's two figure imports, their captions/counts and related coverage wording change in compiled TeX. Labels, float settings and all other figure blocks remain unchanged. Sections 1–3, Results/Meaning/abstract, every appendix, refs.bib, title, main/macros and the layout report remain outside this edit. All 28 TBD bodies and source locations must remain intact. No internal experiment names appear in the rendered manuscript or new plots.

Validation receipts cover deterministic extraction/plot replay, independent decimal and acceptance classification checks, source and data pins, clean nice19 serial make, references/imports/TBDs, paired page proofs and the five static records. These are documentation and source checks, not runtime/learning qualification or a hosted CI matrix.

Opus #1312 removed 34 uncited entries upstream. The final base has 124 unique, cited keys; the bibliography, layout report and all reserved sources are carried verbatim. Its committed PDF text matches the prior baseline.
