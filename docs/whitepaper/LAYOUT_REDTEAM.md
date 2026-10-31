# White paper layout and format red team — 2026-10-31

Scope: layout, typesetting and formatting of `eurekazero_whitepaper.pdf`, then
clarity and structure. Claims, numbers, results, citations and `\TBD` markers
were out of scope and are unchanged.

- **Baseline:** `main` at `f738b81453be316a3d17f69f9ae5595c4917f5d1`.
  The baseline built cleanly: 22 pages, 0 LaTeX warnings, 0 undefined
  references or citations, 0 overfull or underfull boxes.
- **Method:** a `latexmk` build, then every page rendered with `pdftoppm` at
  100 dpi and read one by one.
- **Result after the fixes:** 21 pages, 0 warnings, 0 undefined, 0 overfull or
  underfull boxes. The reference list shrinks from about 7.3 pages to about 5.6.

**Text check.** Before and after, the extracted PDF text was normalized: rejoin
line-end hyphens, drop hyphens, sort the words. The two word lists differ only in:
- line breaks inside URLs and page ranges;
- page-number tokens;
- glyph extraction from the regenerated schematics;
- the Box 2 title separator.

The number of `\TBD` markers in the compiled sources is unchanged at 43.

Severity reflects the effect on a reader of the printed paper: **High** means
illegible or ambiguous, **Medium** means visibly unprofessional, and **Low** is
polish.

## Fixed in this pass

| # | Sev. | Finding (page numbers refer to the baseline) | Fix |
|---|---|---|---|
| L1 | High | Figures 3 and 5 are 4.5 in schematics shrunk to the 3.4 in column. Their text prints at 4.2–5.5 pt (Fig. 5 boxes at 4.2–4.4 pt), below any venue minimum (pp. 7, 10). | Regenerated at 7.0 in wide by `figures/scripts/make_schematics.py`, with box text at 7.3–8.5 pt and headers at 8.2–9.1 pt. Placed full width (`figure*`). The labels are unchanged. |
| L2 | High | Table 3 sets five text columns in one column at `\footnotesize`, with 2.8 pt column separation. Cells wrap up to five lines, and row boundaries are ambiguous: "FunSearch; AlphaEvolve" runs into "Research swarms" (p. 6). | `table*` at `\small`, with column widths rebalanced. |
| L3 | High | Algorithm 1 at column width: 5 of 9 lines wrap. The continuation lines lose their block indentation, and the ▷ comments land on the wrapped line (p. 11). | `algorithm*`. All 9 lines now fit on one line each. |
| L4 | Med | Table 1 (five text columns) is cramped at `\footnotesize` with 2.8 pt separation. "Campbell [15]" and similar names break inside the citation (p. 3). | `table*` at `\small`. |
| L5 | Med | `twocolumn` implies `\flushbottom`. Vertical glue is stretched to fill columns, leaving visible holes, for example after Table 4 (p. 7) and before "The historical reference" (p. 11). Moving floats also produced an underfull `\vbox` (badness 2564). | `\raggedbottom` in `macros.tex`. |
| L6 | Med | The reference list is set at body size with typewriter URLs, taking 7.3 of 22 pages. Monospace URLs break mid-token across 2–3 lines. | `\bibfont` set to `\small`, `\bibsep` to 3 pt, `\urlstyle{same}`. |
| L7 | Med | Tables 5 and 6 use justified `X` columns. Stretched interword gaps and hyphenation inside `\TBD` cells ("fail-ures", "abla-tion") (pp. 13–14). | Ragged-right `X` columns. |
| L8 | Low | Section headings at `\Large` and subsections at `\large` over a 10 pt body are heavier than NeurIPS/ICML. The Section 3 title wraps to two lines (p. 4). | Sections `\large`, subsections `\normalsize`, all bold. |
| L9 | Low | Captions at body size are hard to tell apart from running text in dense columns. | `\captionsetup{font=small}`. |
| L10 | Low | Box titles are inconsistent: "Box 1: …" and "Box 3: …" use a colon, "Box 2 — …" an em dash. | "Box 2: Self-play in 90 seconds". |
| L11 | Low | Both schematic headers print "search  = 1" with a doubled space, caused by mathtext relation spacing around a leading `=`. | Plain-text "search = 1". |

The before/after page comparisons are in [`redteam/`](redteam/):

| Image | Shows |
|---|---|
| [01](redteam/layout-01-table1-full-width.jpg) | Table 1 |
| [02](redteam/layout-02-table3-full-width.jpg) | Table 3 |
| [03](redteam/layout-03-fig3-schematic.jpg) | Figure 3 |
| [04](redteam/layout-04-fig5-schematic.jpg) | Figure 5 |
| [05](redteam/layout-05-algorithm1-full-width.jpg) | Algorithm 1 |
| [06](redteam/layout-06-table6-ragged-cells.jpg) | Table 6 |
| [07](redteam/layout-07-references-compact.jpg) | References |

## Open — recommended, not changed here

| # | Sev. | Finding | Recommendation | Owner |
|---|---|---|---|---|
| O1 | Med | **Fixed in upgrade PR 2** (redrawn as a TikZ diagram). Figure 1 is a two-row booktabs table captioned as a figure (p. 2). | Either draw it as a schematic or make it a table. A table renumbers every table, so it is a structural call. | writer |
| O2 | Med | **Fixed in upgrade PR 1.** Box numbers are hard-coded in the box titles and in the text ("Box~1/2/3"). The labels `box:searchone` and `box:hallucination` exist but are not used for numbering, so reordering the boxes desyncs them silently. | Add a box counter inside the `explainer` environment and refer to boxes with `\cref`. | format |
| O3 | Med | Figures 4, 9 and 10 are `\fbox` placeholders. The text cites "Figure 4a" and "Figure 4b" panels that do not exist yet. Page 14 stacks four pending floats. | Replace them with real figures before submission. Keep the panel letters in step with the final figure. | writer |
| O4 | Low | **Fixed in upgrade PR 1** (date pinned to October 2026). `\date{Draft --- \today}` changes the committed PDF on every rebuild day. That defeats byte and text identity checks across days. | Pin the draft date, or drop it. | writer/format |
| O5 | Low | **Fixed in upgrade PR 3** (display equations 4–5). The §6.5 writer objective, with $\sum_{w\in G_p}$, is inline math. It opens up the line spacing in that paragraph. | Make it a display equation; the wording is unchanged. | writer |
| O6 | Low | Reference entries are inconsistent: DOI versus URL, "(NeurIPS 2017)" in some book titles but not others, and full author lists of up to about 58 names (refs [19], [66], [114]). | Pick one link policy. Consider truncating long author lists ("et al." beyond ~10), which needs a `.bst` or a biblatex switch. The Wolf month fix is still pending. | refs.bib owner |
| O7 | Low | Links are coloured. That suits a draft, but camera-ready IEEE and ACM versions are normally black. | Use `hidelinks` at submission. | format |
| O8 | Low | The appendices come before the references. That is IEEE order; NeurIPS and ICML put the references first. | Choose per venue at submission. | format |
| O9 | Low | Algorithm 1 has an extra vertical gap after the nested `\EndFor`, an artefact of `noend`, between lines 7 and 8. | Cosmetic. Leave as is, or adjust `algpseudocode` spacing. | format |
| O10 | Low | The title block has no affiliation or contact, and the abstract is styled as a section heading. | The venue template will replace both. | — |

## Clarity and structure (for the writer)

- **Float pressure on pp. 13–14.** Figures 6–8 belong to §6.8, but Figure 8
  floats into the Results pages. Together with the four pending Results floats,
  pp. 13–14 hold seven floats. Once the real Results exist, consider moving
  Figure 8 to an appendix or merging it with Figure 7.
- **Section 3 length.** It runs about 3 pages, longer than §4–§6. Each road has
  the same "What EurekaZero takes / The question carried forward" coda, which
  works. A reader looking for the method reaches §6 only on p. 10.
- **References outside the cleveref chain.** Every cross-reference resolves
  through cleveref with consistent capitalisation ("Section", "Appendix",
  "Equation (n)"). The only references outside that chain are the hard-coded
  Box references (O2).

## Full upgrade passes (owner addendum t2052)

The owner widened the scope to a full upgrade, carried out in small sequential PRs. Results (Section 7) changes only in layout. Every claim elsewhere stays supported by the repository and research records.

| Pass | Scope | Status |
|---|---|---|
| 1 | Title and subtitle ("EurekaZero: Engineered Intelligence for Algorithm Invention by Self-Play"), PDF metadata, pinned draft date (O4), numbered boxes referenced with `\cref` (O2) | merged (#770) |
| 2 | Abstract rewritten: problem, game, design, contributions, evidence status, and one TBD for the headline result. Introduction restructured: an "Engineered intelligence" paragraph ties the subtitle to the four pillars; an explicit contributions list; a "What is and is not established" paragraph replaces the introduction's TBD. Figure 1 redrawn as a TikZ diagram (O1). | merged (#772) |
| 3 | Method (§5–6): the §6.5 objective set as display equations (O5). Three TBDs become accurate statements: exploration in the reference procedure (§5.4, from §6.5/§6.7); the checkpoint acceptance record as an evaluation-contract input (§6.5); and what the admission CSVs do and do not record (§6.8). The regressor identity TBD (§6.2) stays for the team. | this PR |
