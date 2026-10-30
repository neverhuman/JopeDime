# Source drafts (verbatim, as supplied 30 Oct 2026)
Per-section folders `sections/NN_name/sources/` hold every supplied draft. The initial scaffold promoted the first `.tex` listed below. Active sections now integrate editorial revisions; the supplied files remain a verbatim source archive.

| Section | Initial scaffold source | Bib fragment | Prose / notes | Alternate drafts |
|---|---|---|---|---|
| 01 Introduction | 01M674EB8BRSAQ6GADKHM7PCHR.tex | 01M674EB831X3J4BGXR1XHMVV2.bib | EB911 (prose md) | MWJAS (.txt, alt LaTeX "Can Machines Invent?") |
| 02 Anatomy | 01M674EAQRHZMKY43998YKE84N.tex | 01M674EAJBM7YG27XC7BAMWB0K.bib | EB6BR (prose) | MWJ7H (alt tex), MWKBX (alt editor note) |
| 03 Road to Zero | 01M674EABGB3EKY2HXARNEDCR2.tex | 01M674E9ZAFGC8RHCD449H62S7.bib | EAFZE (prose) | MWJCP (.txt, alt "Economics & Failure Modes" — overlaps Sec 5) |
| 04 Game | 01M674E9SDXZMYJAB01T3GVFPX.tex | 01M674E9T2Q3XGX2RZ7WKGC8NA.bib | E9Y6K (prose) | MYAMR (.txt, alt tex) |
| 05 Price of Play | 01M674E9NTSN94FJAJ8DKPSM8C.tex | 01M674E8W4GHBMFP0PTCBPW7M2.bib | EAF8K (prose) | MYBBH (alt tex), MXH0T (alt editor note + claim ledger) |
| 06 Method | 01M674E8XQWQHQEJPY314EYFM4.tex | 01M674E8YGFSDP5KWWBQPJBVRV.bib | E8XA0 (editor note), E8ZD9 (full source + ledger), E8YNJ (prose) | — |
| shared_bib | — | MWHRW, MXEW6 (refs.bib as .md), MXAKC (references.bib .txt) | — | alternate master bibliographies, not yet merged |

## Editorial reconciliation

The active manuscript separates historical implementation specifications,
proposed evaluation protocols and measurements derived from the committed
CSVs. A numerical assertion in an alternate draft still needs its own
provenance before becoming an empirical result.

- Anatomy retains the distinction between preserving a winning output and
  improving the proposer from the [alternate Section 2 draft](sections/02_anatomy/sources/01M674MWJ7HV6DJNX3882N18Q1.tex).
- Method follows the historical reference and limitations in the
  [Section 6 editor note](sections/06_method/sources/01M674E8XA0S4ZS5BT6FVH5QKX.md).
  The [alternate reference description](sections/06_method/sources/01M674E8YNJ677EGYWZ5ZM63BA.md)
  concerns a different archive-constrained policy; its architecture is kept
  separate from the historical reference and the later adapter snapshot.
- The [alternate economics draft](sections/03_road/sources/01M674MWJCPWDSDQ7TY3TFEEVE.txt)
  contributes qualitative failure categories. Its numerical rates, cost
  matrix and continuous-latent architecture are not connected to the
  documented reference or committed CSV inputs, so they are not promoted
  into results.
- Price adopts the [supplied cost-accounting draft](sections/05_price/sources/01M674EAF8KG9MQT1QDR3QSCJQ.md)'s
  candidate-level ledger: a repair emitting a revised algorithm is another
  attempt, and every operation is charged once. The independent-try cost
  projection is distinguished from observed batch cost per verified win;
  the draft's alternate numerical pricing table is not promoted.
- Results also retains the method note's historical search diagnostics.
  The [retained search summary](../internal/asm-manual/sections/targets.md)
  distinguishes paired final-search observations from the broader seed audit;
  shared seeds, repeated contexts and unequal times prevent treating them
  as independent first proposals. Their source/protocol receipts remain
  explicit placeholders. The [earlier ranking claim](../paper/claims.json)
  identifies fresh-program pairwise accuracy on development contexts, while the
  [model record](../performance.md#neural-inference-and-refinement)
  distinguishes its model from the historical reference and later studies.
  Its reported estimate and interval are kept with that earlier model;
  source/checkpoint/protocol receipts remain required before final use.
- Abstract, Results and Meaning retain explicit evidence placeholders.
  The training-status account uses [admissions.csv](figures/data/admissions.csv)
  and [rounds.csv](figures/data/rounds.csv), distinguishes accepted flags from
  strict gains, and limits summary-level claims to retained rows.
