# Source drafts (verbatim, as supplied 30 Oct 2026)
The `supplied-sources/sections/` folders retain the supplied drafts byte-for-byte. This index has updated relative links. Historical method records with internal identifiers are archived outside the paper tree under `docs/internal/whitepaper/`. Active sections integrate editorial revisions; archived text does not supply the pending manuscript conclusions.

| Section | Initial scaffold source | Bib fragment | Prose / notes | Alternate drafts |
|---|---|---|---|---|
| 01 Introduction | 01M674EB8BRSAQ6GADKHM7PCHR.tex | 01M674EB831X3J4BGXR1XHMVV2.bib | EB911 (prose md) | MWJAS (.txt, alt LaTeX "Can Machines Invent?") |
| 02 Anatomy | 01M674EAQRHZMKY43998YKE84N.tex | 01M674EAJBM7YG27XC7BAMWB0K.bib | EB6BR (prose) | MWJ7H (alt tex), MWKBX (alt editor note) |
| 03 Road to Zero | 01M674EABGB3EKY2HXARNEDCR2.tex | 01M674E9ZAFGC8RHCD449H62S7.bib | EAFZE (prose) | MWJCP (.txt, alt "Economics & Failure Modes" — overlaps Sec 5) |
| 04 Game | 01M674E9SDXZMYJAB01T3GVFPX.tex | 01M674E9T2Q3XGX2RZ7WKGC8NA.bib | E9Y6K (prose) | MYAMR (.txt, alt tex) |
| 05 Price of Play | 01M674E9NTSN94FJAJ8DKPSM8C.tex | 01M674E8W4GHBMFP0PTCBPW7M2.bib | EAF8K (prose) | MYBBH (alt tex), MXH0T (alt editor note + claim ledger) |
| 06 Method | [Archived method draft](historical-method-draft.tex) | 01M674E8YGFSDP5KWWBQPJBVRV.bib | [Archived editor note](historical-source-notes.md); other supplied prose remains archival | — |
| shared_bib | — | MWHRW, MXEW6 (refs.bib as .md), MXAKC (references.bib .txt) | — | alternate master bibliographies, not yet merged |

## Editorial reconciliation

The active manuscript separates historical implementation specifications,
proposed evaluation protocols and measurements derived from the committed
CSVs. A numerical assertion in an alternate draft still needs its own
provenance before becoming an empirical result.

- Anatomy retains the distinction between preserving a winning output and
  improving the proposer from the [alternate Section 2 draft](supplied-sources/sections/02_anatomy/sources/01M674MWJ7HV6DJNX3882N18Q1.tex).
- Method follows the historical reference and limitations in the
  [Archived editor note](historical-source-notes.md).
  The [alternate reference description](supplied-sources/sections/06_method/sources/01M674E8YNJ677EGYWZ5ZM63BA.md)
  concerns a different archive-constrained policy; its architecture is kept
  separate from the historical reference and the later adapter snapshot.
- The [alternate economics draft](supplied-sources/sections/03_road/sources/01M674MWJCPWDSDQ7TY3TFEEVE.txt)
  contributes qualitative failure categories. Its numerical rates, cost
  matrix and continuous-latent architecture are not connected to the
  documented reference or committed CSV inputs, so they are not promoted
  into results.
- Price adopts the [supplied cost-accounting draft](supplied-sources/sections/05_price/sources/01M674EAF8KG9MQT1QDR3QSCJQ.md)'s
  candidate-level ledger: a repair emitting a revised algorithm is another
  attempt, and every operation is charged once. The independent-try cost
  projection is distinguished from observed batch cost per verified win;
  the draft's alternate numerical pricing table is not promoted.
- Results reserves historical search and ranking diagnostics as placeholders until Jepson supplies the source. Archived notes and prior records are context; they do not fill the current diagnostic placeholders.

- Abstract, Results and Meaning retain explicit evidence placeholders.
  The training-status account uses [admissions.csv](../../whitepaper/figures/data/admissions.csv)
  and [rounds.csv](../../whitepaper/figures/data/rounds.csv), distinguishes accepted flags from
  strict gains, and limits summary-level claims to retained rows.
