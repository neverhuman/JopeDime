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
| O3 | Med | The independent-comparison, repeated-attempt and pilot-cost figures remain placeholders. Their panel descriptions are plans, not supplied results; the development plots use retained log data. | Supply the missing comparison, attempt-curve and pilot-cost data before replacing those placeholders; keep panel labels aligned with the final figures. | writer |
| O4 | Low | **Fixed in upgrade PR 1** (date pinned to October 2026). `\date{Draft --- \today}` changes the committed PDF on every rebuild day. That defeats byte and text identity checks across days. | Pin the draft date, or drop it. | writer/format |
| O5 | Low | **Fixed in upgrade PR 3** (display equations 4–5). The §6.5 writer objective, with $\sum_{w\in G_p}$, is inline math. It opens up the line spacing in that paragraph. | Make it a display equation; the wording is unchanged. | writer |
| O6 | Low | **Decided (owner t0352): keep the current bibliography style.** Reference entries are inconsistent: DOI versus URL, "(NeurIPS 2017)" in some book titles but not others, and full author lists of up to about 58 names (refs [19], [66], [114]). | Given names now use initials with standard `abbrvnat`, retaining all authors and source metadata. The Wolf publication month was corrected in [#779](https://github.com/neverhuman/JopeDime/pull/779). DOIs and supplied URLs are both clickable, with existing metadata retained. The 13 NeurIPS entries use the canonical proceedings title, with volume and year in separate fields. Any author-list truncation remains open. | refs.bib owner |
| O7 | Low | **Fixed in upgrade PR 18** (`hidelinks`, owner decision t0352). Links are coloured. That suits a draft, but camera-ready IEEE and ACM versions are normally black. | Use `hidelinks` at submission. | format |
| O8 | Low | **Decided (owner t0352): keep the order on `main`; revisit per venue.** The appendices come before the references. That is IEEE order; NeurIPS and ICML put the references first. | Choose per venue at submission. | format |
| O9 | Low | **Fixed in upgrade PR 7** (a preamble patch drops the empty line that `noend` leaves in place of each end statement). Algorithm 1 has an extra vertical gap after the nested `\EndFor`, an artefact of `noend`, between lines 7 and 8. | Cosmetic. Leave as is, or adjust `algpseudocode` spacing. | format |
| O10 | Low | **Decided (owner t0352): keep the current template; revisit per venue.** The title block has no affiliation or contact, and the abstract is styled as a section heading. | The venue template will replace both. | — |
| O11 | Low | **Fixed in upgrade PR 7** (`flushend`). The two columns of the last page are unbalanced: on p. 23 the left column reaches the bottom margin and the right column stops short, by an amount that shifts with every upstream edit. | Balance the last page only; `balance.sty` was rejected because it also shortens every full bibliography page. | format |
| O12 | Low | **Fixed in upgrade PR 9** (2 pt row spacing). In Table 9 (Appendix A) wrapped cells run together: "Matrix multiplication" sits directly on "Sorting", and "Machine-learning research" on "Code" (p. 17). | `\setlength{\extrarowheight}{2pt}` separates rows; the widths are unchanged, and no float or section moves. | format |
| O13 | Low | **Fixed in upgrade PR 11** (2 pt row spacing). Table 4 (§4, p. 7) has the same run-together rows: "Legal move" spans three lines with "Illegal move" directly beneath it. | `\setlength{\extrarowheight}{2pt}`, as in Table 9; the widths are unchanged. After pass 10 the §7 float gap moves only from 2 to 3 (Figure 11 is one page after §7.4). A full-width `table*` still cascades. | format |
| O14 | Low | **Fixed in upgrade PR 12** (2 pt row spacing). Table 2 (§2, p. 5) has the same run-together rows: "Judgment" sits directly on "Retention", and "Where taste lives" on "Deployment budget". | `\setlength{\extrarowheight}{2pt}`, as in Tables 4 and 9. Moving from `\footnotesize` to `\small` was tested and rejected: it creates a bad box. No float or section moves. | format |
| O15 | Low | **Fixed in upgrade PR 13** (labels at 7.2 pt). Figure 2's 18 annotation spans printed at 6.3–6.4 pt, the smallest non-subscript text in any figure; Figures 3 and 4 print at 7.3–8.5 pt after L1. | `make_schematics.py`: Figure 2 labels at 7.2 pt; the incumbent label is split onto two lines and two boxes are nudged so nothing overlaps. Same figure size, so no reflow. | format |
| O16 | Low | **Fixed in upgrade PR 14** (legends at 7 pt). Figure 7's legend text printed at 6.0 pt; its tick labels were already 7.0 pt. The middle panel's legend also washed out its top data point (round 1322, 214 s). | `make_flywheel_figs.py`, Figure 7 only: legend `fontsize` 6 → 7; the middle label is split onto two lines (same wording) and `handlelength=1.0`, so every point is visible. Same figure size and data. Figures 5–6 are addressed separately in O17. | format |
| O17 | Low | **Fixed in writer pass 68.** Figures 5–6 legends and the train–validation gap annotation printed at about 6.45 pt. | `make_flywheel_figs.py` uses 7.1 pt source text, about 7.05 pt at the paper's column scale. The Figure 6 legend moves into the empty space between the curves, avoiding the larger annotation. Plot arrays, wording and data are unchanged; Figure 7 remains byte-identical. Five inherited script lint findings are also cleared. | writer |
| O18 | Low | **Fixed in upgrade PR 15.** Figure 7 (§6.8 operating measurements) was never referenced in the text; Figures 5, 6 and 8 were. | One pointer sentence in §6.8 restates the caption's scope ("operating measurements from the same snapshot"); no new claim. | format |
| O19 | Low | **Fixed in upgrade PR 15.** Single paragraph lines were stranded at column breaks: an orphan first line at the bottom of p. 13's left column ("Candidate programs are evaluated on two training-side"), and a one-line start of reference [117] at the bottom of p. 23's left column. | `\clubpenalty`, `\widowpenalty` and `\displaywidowpenalty` set to 10000 in `macros.tex` (with `\raggedbottom` already on). Only pp. 13, 14 and 23 change layout; no float moves. | format |
| O20 | Low | **Fixed in upgrade PR 18** (disclosure boundary, owner decision t0352). The promotions figure's provenance (`figures/data/program_promotions.provenance.json`) and its generator recorded a host name and absolute host paths. | Host name, host paths and inode numbers are removed from the JSON, and the script no longer writes them. File names, sizes, mtimes and all four SHA-256 bindings are unchanged. | format |

## Clarity and structure (for the writer)

- **Float pressure (fixed in pass 10).** Figures 5–8 belong to §6.8. Defined after
  the §6.8 text, Figures 6–8 landed among the §7 tables and placeholders, and
  the §7 floats sat on a knife edge: about 18 pt of extra height above §7 pushed
  Table 5 and Table 8 one to three pages from their subsections. That happens
  because floats keep their order within each type: Tables 6–8 wait behind the
  full-width Table 5, and Figures 9–11 wait behind Figures 7–8.

  Pass 10 defines Figures 5–7 at the top of §6.8, so they are placed before §7
  begins; numbers and text are unchanged. The summed page gap between each §7
  float and its subsection falls from 7 to 2, and under the same +18 pt
  perturbation from 13 to 3.

  Rejected alternatives:
  - a `\FloatBarrier` after §6.8 adds two pages of half-empty float pages;
  - `[tb]` placement and `[htb]` on the §7 floats change nothing;
  - a full-width Table 4 cascades further.

  Once the real Results exist, moving the development telemetry to an appendix
  remains an option.
- **Section 3 length.** It runs about 3 pages, longer than §4–§6. Each road has
  the same "What EurekaZero takes / The question carried forward" coda, which
  works. A reader looking for the method reaches §6 only on p. 10.
- **References outside the cleveref chain.** Every cross-reference resolves
  through cleveref with consistent capitalisation ("Section", "Appendix",
  "Equation (n)"). Box references joined the chain when O2 was fixed (#770).

## Full upgrade passes (owner addendum t2052)

The owner widened the scope to a full upgrade, carried out in small sequential PRs. Results (Section 7) changes only in layout. Every claim elsewhere stays supported by the repository and research records.

| Pass | Scope | Status |
|---|---|---|
| 1 | Title and subtitle ("EurekaZero: Engineered Intelligence for Algorithm Invention by Self-Play"), PDF metadata, pinned draft date (O4), numbered boxes referenced with `\cref` (O2) | merged (#770) |
| 2 | Abstract rewritten: problem, game, design, contributions, evidence status, and one TBD for the headline result. Introduction restructured: an "Engineered intelligence" paragraph ties the subtitle to the four pillars; an explicit contributions list; a "What is and is not established" paragraph replaces the introduction's TBD. Figure 1 redrawn as a TikZ diagram (O1). | merged (#772) |
| 3 | Method (§5–6): the §6.5 objective set as display equations (O5). Three TBDs become accurate statements: exploration in the reference procedure (§5.4, from §6.5/§6.7); the checkpoint acceptance record as an evaluation-contract input (§6.5); and what the admission CSVs do and do not record (§6.8). The regressor identity TBD (§6.2) stays for the team. | merged (#777) |
| 4 | Results presentation only, with every sentence, number and TBD of §7 preserved:<br>• "Evaluation plan at a glance" table (Table 5).<br>• Earlier diagnostics moved into their own closing subsection (§7.7).<br>• Worked-game fields moved from §4.6 into a §7.6 record table; §4.6 becomes "The analogy's limits".<br>• Pilot-cost figure moved from §5.2 into §7.4.<br>All results-dependent placeholders now sit in §7. | merged (#786) |
| 5 | Discussion, Conclusion and appendices:<br>• §8 becomes "Discussion", with each subsection ending in an explicit outcome-to-claim rule. Four per-subsection TBDs become pointers to the §7 subsections that will settle them; the opening TBD stays as the single tracked marker.<br>• New "Limitations" subsection (§8.5) and a §9 Conclusion.<br>• Appendix A becomes a domain-by-domain referee inventory (Table 9), restating referees already cited in §3–§4.<br>• Receipt TBDs in Appendices B and C point to the §7.1 evaluation record.<br>The regressor licence TBD stays. | merged (#789) |
| 6 | Fixes from an independent camera-ready review (dispositions below): scope stated in the abstract and introduction; honest evidence status; ratchet, map and loss wording matched to §6 and the code-level records; exploration programs added; drafting-process language removed; pointer, typography and Results-presentation slips fixed. | merged (#792) |
| 7 | Camera-ready preamble polish, no text change: equal-height columns on the last page (O11) and no blank line after a nested `\EndFor` in Algorithm 1 (O9). Algorithm 1 is two lines shorter, so text on pp. 13–17 moves up with every float on its original page, and p. 23 is rebalanced. The extracted text keeps the same words and characters. | merged (#896) |
| 8 | H56 paper-vs-code recheck (below): all 12 dimensions are FIXED or ALREADY-OK. One §6.8 framing sentence states that the inspected loop source is not yet bound to the run behind the retained logs. `\brokenpenalty=10000` (in `macros.tex`) stops words hyphenating across a column or page break; this removes the one on main ("Sup-/ply", §7.4) and the one this reflow would otherwise create ("sec-/comp", Appendix C). The float-pressure note is updated with the layout fixes tested. | merged (#923) |
| 9 | Appendix A: Table 9 rows separated with 2 pt row spacing (O12); no text change, every float and section stays on its page. Table 4 gets the same fix later (O13), because extra height above §7 currently cascades into the Results float placement. | merged (#931) |
| 10 | §6.8 float order: Figures 5–7 defined at the top of §6.8 (numbers and text unchanged), so they are placed before §7 begins and the §7 floats stay with their subsections. Float-to-text gap 7 → 2 pages; under a +18 pt upstream change, 13 → 3. | merged (#940) |
| 11 | §4: Table 4 rows separated with 2 pt row spacing (O13), the same fix as Table 9; no text change. §5 now starts on p. 9, and the §7 float gap moves from 2 to 3. | merged (#944) |
| 12 | §2: Table 2 rows separated with 2 pt row spacing (O14). No text change; no float or section moves, and only p. 5 changes layout. Tables 2, 4 and 9, the single-column tables with multi-line cells outside Results, now all separate their rows. | merged (#949) |
| 13 | Figure 2 legibility: annotation labels from 6.3–6.4 pt to 7.2 pt (O15), with no overlaps. Regenerated deterministically (`SOURCE_DATE_EPOCH` pinned to the committed figure's creation date; the unchanged script regenerates it byte-identical). Same figure size, no text change. | merged (#955) |
| 14 | Figure 7 legibility: legend text from 6.0 to 7.0 pt (O16), so all of Figure 7's text is at least 7 pt. A narrower middle legend uncovers the round-1322 point. Regenerated deterministically (`SOURCE_DATE_EPOCH` pinned; the unchanged script reproduces all three flywheel figures byte-identical); data, values, axes and figure size unchanged. | merged (#959) |
| 15 | Whole-paper layout audit on main `b2f271ce`. Fixed: the unreferenced Figure 7 (O18) and the widow/orphan lines (O19). Checked, nothing to fix: 0 overfull or underfull boxes; every float is on its first reference's page or the next, except Table 1 (2 pages after an introduction forward reference) and Figure 11 (§5's forward references to the Results figure, by design); all tables use consistent left-aligned `p`/`X` columns. `refs.bib`: no two keys share a DOI, arXiv id, URL or normalized title. The only near-pair, the AlphaZero preprint (`silver2017chess`) and Science paper (`silver2018general`), is co-cited once on purpose: the preprint's Table S3 is the recorded source of "44 million games" and the paper of "about nine hours", so both are kept. | merged (#971) |
| 16 | Camera-ready prose pass on §8–§9: a redundant sentence and the Limitations closing paragraph (which restated the Conclusion) cut, §8.2 tense and wording fixed, and source comments trace uncited claims to repository paths. No new numbers; TBDs unchanged. "Needs Jepson/team" gains the overnight H80–H97 closeouts as candidates only. | merged (#976) |
| 17 | Introduction terminology: the engineered-intelligence definition now reads "aim, judge and memory", matching the Conclusion and the §2 pillar name ("objective" before). No other change. | merged (#980) |
| 18 | Owner decisions t0352 applied: black links (O7); regressor named "Foundation", version `foundation_enc.v1.tabpfn`, "proprietary, available on request" (two TBDs filled); comparators pinned to receipted model ids (`claude-opus-5-5`, `gpt-6-astra-wm`), with Gemini and KumoRFM dropped (two Table 6 TBDs removed); host name and host paths removed from the promotions provenance (O20); template, appendix order and bibliography style kept (O6, O8, O10). TBDs 32 → 28; `RESULTS_NEEDS.md` renumbered. | merged (#982) |
| 19 | Flow and format pass on §2–§6, from wp_opus_layout_t0355. §2.3 no longer defines *taste* a second time (§2 defines it). "license" now matches the paper's US spelling (§6.2, Appendix C). §1 ("The Zero move") hyphenates "single-candidate budget". This report's closing paragraph is brought up to the post-t0352 count (28). §4 was read with nothing to change. Spelling, hyphenation, dash and repeated-word scans of §2–§6 are otherwise clean. | merged (#986) |
| 20 | wp_opus_layout_t0420. **H56 fold:** the native trainer's list sampling (H56 row 5 above); the encoder blocks (512 + 1,024 + 512 in `dope_encoder.rs`) and the map's separation from the writer were rechecked and match. **`refs.bib`:** no work under two keys (by arXiv id, DOI and normalized title), every `\cite` key resolves, and the four same-author, same-year key pairs are distinct works; no change. **Flow:** Figure 3 (the game loop) is now defined before the formal tuple, so it prints at the top of p. 8 where it is cited, not on p. 9; only §4.6's heading moves (p. 8 to 9). Section openings and closings were read with nothing to change. | merged (#992) |
| 21 | wp_opus_layout_t0450, a page-by-page formatting pass on the rendered PDF. Fixed: §3's closing sentence no longer hangs alone under the key-idea box at the foot of p. 6, so the box now ends the section; Table 6's comparator column is widened so both model identifiers set on one line; Table 8's field column is widened so "Hidden-fold score of a valid fit" no longer strands "fit"; Appendix C's inline percentile formula breaks after the solidus instead of inside "(\|G_p\| − 1)", which also stops "Bradley–Terry" splitting at its dash. Figure 4's writer-supervision box now says "each sampled move" (was "every scored move"), matching §6.5 after #992; regenerated with `SOURCE_DATE_EPOCH` pinned to its committed CreationDate. Checked with nothing to change: float placement against first citations, widows and orphans on every page, caption style (bold title sentence; tables above, figures below), heading spacing, display-equation breaks, references pages. One label move: Appendix B's heading, p. 18 to 17, opening under Table 9 with several lines. | merged (#995) |
| 22 | Owner call wp_opus_layout_t0510: the regressor licence in §6.2 and Appendix C now reads "proprietary, available on request; redistribution prohibited", matching the bundle manifest (`LicenseRef-Jain-ThirdParty-Provenance-Pending`, redistribution "prohibited"). This closes the confirmation request raised in pass 18. `RESULTS_NEEDS.md` and the owner-decision entries above are updated. With this, every upgrade-brief item outside Results is done; O3 (the Results placeholders) is the only open row. | this PR |

## H56 paper-vs-code recheck (pass 8)

[`h56_audit/SUMMARY.md`](../../research/flywheel_selfplay/h56_audit/SUMMARY.md) (#608, merged at `db20a45a`) compares the paper with the code in 12 dimensions. Each one was rechecked against the current paper and code:
- **ALREADY-OK** means the paper already avoided the overclaim when the audit merged. Some audit rows describe source drafts or earlier text, not the merged paper.
- **FIXED** means a PR after the audit resolved it.

None of the 12 leaves the paper overclaiming the writer or the map relative to the live loop.

| # | Dimension | Status | Where the paper stands now |
|---|---|---|---|
| 1 | Neural writer | FIXED (#791; §5 scope #639, #792) | §6.8: "this loop does not optimize a neural program writer"; heuristic edits, retrieval and grammar sampling supply candidates. §5.3 confines the 8B–1.5T writer figures to language-model writers and comparators; the reference network has 27.8M parameters and at most 128 tokens. |
| 2 | Compiler repair | FIXED (#688) | §5.1 counts a repair that emits a revised candidate as another attempt, with no separate repair term. Box 1, §5.1 and §5.6 state that the single-candidate protocol has no repair; §5.3's repair turns concern language-model writers. |
| 3 | Problem encoder | FIXED (#814); §6.2 ALREADY-OK | §6.2's 2,048-value encoder belongs to the reference checkpoint. §6.8 states that the inspected loop zeroes separate context and prior inputs. |
| 4 | Map / value network | ALREADY-OK in §6 (#590); FIXED in §4 (#792) | §6.1, §6.4 and §7.5: the map does not condition, screen or steer the writer. #792 removed §4's claim that also-rans become hard negatives for the map. |
| 5 | Training objective | ALREADY-OK (#590); live adapter FIXED (#814); list sampling FIXED (pass 20) | Eq. 4's rank weights $0.05+\rho^4$ match the native `Σ (0.05 + p⁴)` (`crates/jope-engine/src/training/objective.rs`); #824 adds the native smoothing and normalization. The audit's three-tier RL reward appears only in a source draft. §6.8 describes the adapter's regression-plus-listwise objective. Pass 20: §6.5 said the writer "imitates every scored program", but the native trainer (`sample` in `crates/jope-engine/src/training.rs`, sampler `family_uniform_champion_plus_seven_quantiles_v2`) trains on eight-program comparisons, each a list's best program plus one program from each of seven equal rank bands, with problem families given equal mass. §6.5 now says "scored programs sampled from the lists", and Appendix C states the sampler. The coefficients, row weights, Bradley–Terry boost and $\tau$ were rechecked against `objective.rs` and `neural.rs` and match. |
| 6 | Revealed trajectory | FIXED (#791) | §6.8: the pool utility uses the recorded scores and gate outcomes of the first four candidates as revealed history, and selects from the pool including those entries. |
| 7 | Referee folds | FIXED (#814) | §6.8: two training-side folds in the loop; the historical reference procedure specifies five. |
| 8 | Pool admission | FIXED (#814) | §6.8: ordinary admission keeps only programs that pass the gates and strictly improve their context's incumbent; evaluations are logged separately. |
| 9 | Incumbent ratchet | ALREADY-OK (#602) | Figure 5 marks accepted ties. §6.8 reports 4 acceptances, of which 2 are strict gains, in 695 rounds and a late plateau, and claims no continuous improvement. The audit's rollback-on-reject observation is not described in the paper. It was not re-verified against the current loop source, so no sentence was added. |
| 10 | Diversity archive | FIXED (#634) | §5.4 lists quality-diversity archives only among possible exploration controls. The reference procedure's exploration is 64 drafts, 16 random programs and seeded verified search. |
| 11 | Deployment budget | ALREADY-OK (#599); FIXED (#792, #791) | §6.7: $B=1$ is a protocol constraint, and its quality is an empirical question. §5.4 and Algorithm 1 show the training-side 64 + 16 candidates. §6.8 calls the loop's pool selection "a conditional ranking task", separate from a single new program. |
| 12 | Telemetry continuity | ALREADY-OK | At the audit's merge the plotted data already covered one run only: rounds 775–1469 (695 admission rows) and 1342–1469 (25 summaries). The audit's "rounds 1–1332" row describes an earlier draft. The later promotions slice (rounds 1838–1857, #892) is shown separately. |

**Closure check, pass 21 (t0450).** All 12 dimensions remain closed on main after #992. Two further writer statements were checked against code: §6.5's training sampler (fixed in #992, and now in Figure 4) and the player ratchet. §4.4 and §6.5 state the player ratchet's admission conditions (16 held-out cuts × 3 seeds; mean log error ratio favoring the challenger; wins ≥ losses; no more failures) as necessary conditions. They match promotion rule F10 in `docs/adr/ADR-0005-canonical-learning-loop.md` (decision 6), whose further conditions the paper does not claim to be exhaustive. No section claims that the live adapter loop of §6.8 trains, conditions or screens the writer.

## Needs Jepson/team

[`RESULTS_NEEDS.md`](RESULTS_NEEDS.md) is the per-marker checklist: every compiled `\TBD`, its location, what resolves it and the order of work.

Everything below needs data, receipts or decisions from the team. None of it was invented or changed. Unless a row says otherwise, it lives in §7.

### Evidence handoff for Results

Use the following packet to supply the evidence requested below. Each component
should identify its source commit, artifact location and content hash, generating
command and evaluation protocol. Freeze the evaluation contract before opening
outcomes. Keep unavailable fields and unresolved outcomes explicit; failed
attempts remain in the accounting.

| Component | Minimum contents | Used in |
|---|---|---|
| Frozen evaluation contract | Problem and family inventory; dataset identities, splits and exclusions; permitted inputs and known training exposure; writer, reference, comparator, encoder and evaluator identities; seeds, metrics and budgets. Link the provenance and acceptance receipts listed below. | §7.1 and every comparison |
| Paired first-proposal record | Every predeclared task/family/seed pairing, both checkpoint identities and emitted outputs, scores, terminal statuses and costs. Attach evidence of optimizer updates and parameter changes. | §7.2 |
| Fixed-method comparison record | Paired outcomes, ties, failures and exclusions; originating model versions, prompts and dates; implementation checks; only comparators whose receipts record the designing model (owner decision t0352). Bind Elo summaries to their comparison population and rating protocol. | §7.3, Table 6 and Figure 9 |
| Attempt and cost record | Every attempt, including repairs and failures, its terminal status and incurred encoding, generation, fitting and prediction costs. Identify unreached targets, unresolved cases and settings with no verified win. | §7.4 and §7.6, Figures 10–11 |
| Map, example and diagnostic records | Disjoint ranking/similarity data and program-only controls; a declared training influence path for a causal ablation, or ranking-only scope. Include the preselected worked example and its program/outcomes, and pin each historical diagnostic's own inventory, checkpoint and protocol. | §7.5–§7.7, Tables 7–8 |
| Reproducible summaries and figure inputs | Commands and input identities for paired family-level outcomes, uncertainty and costs; counts of ties, failures, exclusions and unknowns; the records underlying every table cell and plotted panel. | Results tables/figures, then the abstract and Discussion |

Trace each reported result back through its summary command to the frozen
inventory and attempt records. Keep the retained development plots and the
historical diagnostics bound to their own populations when preparing the final
comparisons.

### Results content that is missing

| Where | What is missing |
|---|---|
| Abstract | One sentence with the headline first-proposal result and its uncertainty. |
| §7 opening | Independent evaluation evidence for any performance claim. A source-bound benchmark for any deployment-throughput claim. |
| §7.1 contract | The final problem inventory, dataset identities and content hashes, row splits, family assignments and exclusions, component provenance, evaluator, checkpoint, seeds, task metrics and failure accounting. All of it must be frozen before outcomes are opened. |
| §7.2 effect of training | The matched first-proposal comparison: the trained writer against the frozen reference writer, paired first draws under predeclared seeds, all draws reported. Source-bound evidence of optimizer updates and parameter changes. This is the central claim of the paper. |
| §7.3 fixed methods (Table 6, Figure 9) | Paired outcomes with uncertainty for the frozen algorithms designed by `claude-opus-5-5` and `gpt-6-astra-wm`, including ties, failures and exclusions. The prompts, dates and implementation checks. (Gemini and KumoRFM were dropped for lack of a model-identity receipt; owner decision t0352.) The leaderboard's methods, paired outcomes, rating protocol and uncertainty. |
| §7.4 cost of attempts (Figures 10–11) | Attempt-versus-performance and cost curves for agents allowed successive attempts, with matched budgets, repeated seeds and uncertainty. Pilot cost data: per-stage failure shares, and cost per verified win by writer size and reasoning effort, with seeds and family-level uncertainty. Settings with no verified win must be identified without assigning them a finite cost. |
| §7.5 map (Table 7) | A held-out ranking study, a problem-similarity study and program-only-control evaluations. Either an implemented route by which the map influences training, so a causal ablation is possible, or a decision to limit the claim to ranking. A failure-class comparison. |
| §7.6 failures (Table 8) | Attempt inventories and costs by terminal outcome class, with unresolved cases identified. The worked example's selection rule, chosen before inspection, and its record: problem, encoding time, emitted output, check and fitting outcomes, and hidden-fold and incumbent scores (or the failure and its cost). |
| §7.7 diagnostics | Pinned inventories, checkpoints, seeding, timing and ranking protocols for the historical diagnostics. |
| §5.4 to §7 | The measured exploration quality–cost trade-off, if an exploration study is run. |

### Results evidence that exists but is weak

- **The earlier plotted adapter admission record (§6.8).** Over 695 rounds there are two strict validation gains, then a plateau, and a train–validation gap of 0.282–0.290 on 25 summaries. It is not an untouched test and cannot support the transfer claim.
- **The later program-promotion figure is development telemetry (§6.8).** Its [input CSV](figures/data/program_promotions.csv) and [provenance](figures/data/program_promotions.provenance.json) align journaled training-program events with candidate validation utility in one retained summary window. Strict promotion is a numerical gate; the journal supplies no confidence interval. Writer training and transfer still require the independent comparisons above.
- **The historical diagnostics (§7.7) do not show a learned proposer outperforming search.**
  - 3,227 of 3,268 paired searches ended in exact ties.
  - 98.84% of reported winners were already in their seed populations.
  - An earlier model's ranking accuracy, 0.506 [0.441, 0.568], includes chance.
- **§7.2 has no comparison at all yet.** Until it does, the abstract, the introduction and §8 can only state what would be tested.

### Overnight research closeouts (H80–H97): candidates only

Snapshot: `main` at [a470804b](https://github.com/neverhuman/JopeDime/tree/a470804b5bc0e0d5f46f42d300ee4b70519f463f/research/flywheel_selfplay) (2026-10-31). Statuses summarize the study READMEs and protocol files present in that source tree. **None supplies the §7.2 matched first-proposal comparison or any other `\TBD` in [`RESULTS_NEEDS.md`](RESULTS_NEEDS.md).** All are development-data studies of the adapter loop or its context models. If used at all, they belong with the development diagnostics (§6.8, §7.7), bound to their own populations and protocols.

| Study | What it tested | Status on `main` | Nearest paper slot | Why it cannot fill a marker |
|---|---|---|---|---|
| [H80](../../research/flywheel_selfplay/h80_long_es/README.md) | Long direct-utility ES on the r882 adapter: plain (ES) and harm-penalized (ESH) | Not admitted, either arm. Median Elo vs r882: ESH +60 (harm 19.5%), ES +17 (harm 29.3%); both mean LCB95 negative | §6.8 / §7.7 | Adapter training on development folds, scored against r882; not a first proposal from the reference writer |
| [H81](../../research/flywheel_selfplay/h81_w4_prospective_2plus2/README.md) | W4 active reveal, prospective 2+2 | Blocked before scoring: comparator-strength guard failed; no fits spent | — | No outcome |
| [H83](../../research/flywheel_selfplay/h83_w6_cross_family_cached/README.md) | W6 cross-family transfer, cached screen | NO_ADVANCE: both primaries fail the harm-tail rule; no native fits | §7.5 similarity | Cached screen of a W6 context model, not the reference map |
| [H84](../../research/flywheel_selfplay/h84_w6_exact_id_coverage/README.md) | W6 exact program-id coverage | Coverage metadata; Elo and U4 null | — | No outcome |
| [H85](../../research/flywheel_selfplay/h85_w1_cpu_remediation_and_models/README.md) | W1 CPU remediation; step-0 Elo panel against r882 continuity | Gate calculator at step 0: NO_ADVANCE | — | No optimizer update |
| [H86](../../research/flywheel_selfplay/h86_w3_outcome_status_census/README.md)–[H89](../../research/flywheel_selfplay/h89_w3_completion_admission/README.md) | Completed-status provenance for W3 inputs | Signature status mask not accepted as completed-status provenance; W3 input admission blocked | §7.6 outcome classes | Shows the admitted packet lacks per-row completion status |
| [H91](../../research/flywheel_selfplay/h91_w9_closeout_readiness/README.md) | W9 causal-closeout readiness | CPU preparation complete; W9 unadmitted | — | No execution |
| [H92](../../research/flywheel_selfplay/h92_w6_downside_controlled/README.md) | W6 hedged context learner, cached held-family screen | ADVANCE_CACHED_SCREEN on adaptive development data (3 of 4 primaries clear the common gate) | §7.5 program-only control | Adaptive and cached; the prospective follow-up (H94) did not advance |
| [H94](../../research/flywheel_selfplay/h94_w6_prospective_hedge/README.md) | The H92 learner on never-evaluated programs, against the program-only prior | NO_ADVANCE: −0.0105 U4 and −51.9 Elo against the program-only prior | §7.5 program-only control | Negative, on development families, and for the W1/W6 context model, not the reference map |
| [H95](../../research/flywheel_selfplay/h95_w8_hwm_resplit_confirm/README.md) | Resplit confirmation of H94's 45 apparent high-water marks | 30 confirmed (27 material); none admitted or promoted | §6.8 / §7.7 | Training-program incumbents, not proposals by a frozen writer |
| [H82](../../research/flywheel_selfplay/h82_sam_steps/PREREG.md), [H90](../../research/flywheel_selfplay/h90_candidate_diversity/PREREG.md), [H93](../../research/flywheel_selfplay/h93_lookahead_noise/PREREG.md), [H97](../../research/flywheel_selfplay/h97_reg_ema/PREREG.md) | SAM steps, candidate diversity, lookahead noise, regularization EMA | Preregistrations and runners only; no outcome on `main` | — | No outcome |

H96 was absent at this snapshot. H98 and H99 already had source on `main` and are outside the H80–H97 range covered by this table.


### Outside Results (provenance and receipts)

- **§6.2 / Appendix C. Resolved (owner decision t0352; licence wording t0510).** The regressor is named as on `main`: "Foundation", version `foundation_enc.v1.tabpfn`; licence "proprietary, available on request; redistribution prohibited". No LICENSE file covers it. The wording matches its bundle manifest, which records `LicenseRef-Jain-ThirdParty-Provenance-Pending` with redistribution "prohibited"; the earlier confirmation request is closed.
- **§6.5 / Appendix C.** The reference checkpoint's acceptance record, its held-out promotion receipt and a pinned runtime.
- **§6.8.** Bind the retained observations to exact loop source and input identities, adapter configuration, score normalization, evaluation population, candidate pools and revealed-history inputs. The implementation is now described; run qualification and cross-model comparisons still require those records.
- **Appendix B.** The final pre-registration and audit receipts: the inner-fold label-poison test, the sealed canary and the train-only input-closure audit.
- **Appendix C.** Isolation and parity receipts from the final evaluator.

### Owner decisions

- **Disclosure boundary.** **Decided (owner t0352):** disclose only material merged on `main` and backed by receipts, at method level (the typed-GP wording stays). No host names, IPs, S3 paths, sealed or holdout identities, unmerged experiments or kernel-level implementation detail. Pass 18 applied it (O20). Background: `deploy/ci/ip-scrub.sh` states that "nothing public may say how the program is found". The script scans only top-level prose, not this paper. Even so:
  - §3.2 and §3.5 discuss genetic programming and evolutionary search as history.
  - §4.3 cites strongly typed GP for the program language.
  - Appendix A calls "typed genetic programming and evolutionary search" milestones "directly needed for our argument".

  Decide whether the white paper falls under the boundary. If it does, the Appendix A wording and the §4.3 citation should change.
- **Regressor licensing.** **Decided (owner t0352; wording t0510):** "Foundation", version `foundation_enc.v1.tabpfn`, "proprietary, available on request; redistribution prohibited", matching the bundle manifest (see §6.2 / Appendix C above). Background: the §6.2 editor note says the encoder bundle lists TabICL-v2 artifacts as "provenance pending, redistribution prohibited, policy-ineligible". The legacy training source describes a TabPFN-v3-compatible architecture. Confirm which model it is, its version and licence, and whether that limits releasing the checkpoint.
- **Comparator identities.** **Decided (owner t0352):** pin the identifiers recorded in receipts and drop comparators without one:
  - **Kept:** `claude-opus-5-5`, configured (`crates/jope-bench/regressors/README.md`); `gpt-6-astra-wm`, as its session reported (`crates/jope-bench/submissions/gpt6-astra-max/provenance.json`; the receipt notes the session was not isolated).
  - **Dropped:** the Gemini-designed method (its receipt records no served model) and KumoRFM (no receipt).

  §1, the abstract, Table 5, §7.3 and Table 6 now say this. "Codex" no longer appears as a comparator, which removes the clash with the 2021 Codex model.

### More team items surfaced by the review

- **Weak evidence.** The code-level record (`docs/internal/asm-manual/sections/deep-stack.md`) says the recorded neural winner "descends from exploration candidates, not from a neural_policy proposal". If that still holds, the development record shows no proposal by the writer winning.
- **Checkpoint acceptance (§6.5, Appendix C).** The training history says all 48 indexed admission decisions of 11 September were inconclusive, and cites no admission pass for the step-60,000 checkpoint. The paper now says this evidence is not documented here.
- **Objective identity (§6.5, Appendix C).** The native fixed aggregate coefficients match Eq. 5 (`crates/jope-engine/src/training/objective.rs`); uncertainty weighting belongs to a separate objective and must not be substituted for this one. Native sequence-loss smoothing and normalization are documented in #824. The checkpoint-specific objective identity/configuration remains required evidence.
- **Figure 7 units — fixed in #806.** Throughput counts distinct training contexts per completed-round wall second. Program promotions count strict improvements to training-program incumbents over a rolling window of up to twenty completed rounds; the 4 adapter acceptances in 695 rounds count different events. Exact retained-run binding remains open.
- **The 63,108 evaluations — fixed in #819.** This is the total across 42 datasets; a best test-scored outcome was selected for each dataset. The entry remains excluded from all comparative claims.

### Review dispositions (pass 6)

**Fixed:**
- **Abstract.** It no longer equates the adapter log with the writer's admission test. It names the null diagnostics alongside the log, states the tabular domain and the typed language, and calls KumoRFM a relational model. The map is tested for ranking against a program-only control rather than "ablated".
- **Introduction.**
  - The retention sentence matches the actual ratchets.
  - "Engineered intelligence" has a single definition, "designed to be audited".
  - The scope is stated.
  - Contribution 3 is a cost model.
- **§3 and §4.**
  - "Much of it", and "evidence from Go".
  - Table 3's feedback source now matches the within-list ranking.
  - Table 4's caption no longer contradicts §4.6.
  - Hedges on learning claims.
  - §4.4 no longer claims an incumbent-to-map path that the losses do not implement.
  - Pointers fixed.
- **§5.**
  - LLM-writer scope stated.
  - The 16 exploration programs added.
  - The duplicate Move 37 anecdote removed from Box 3.
  - The "supplied cost scenarios" sentence removed.
- **§6.**
  - The writer and map are stated as separate branches.
  - The rank-based weighting is stated correctly.
  - L_BT, L_runtime and L_prog are defined.
  - Algorithm 1 includes the exploration programs.
  - The checkpoint acceptance is stated honestly.
- **§8 and §9.** "Designed to learn", and the decisive comparison is named.
- **Appendices.** Pointer, typography and provenance wording fixed.
- **§7 (presentation only).** "Planned for", and the Table 5 cell.

**Still required from the team or owner:** regressor identity/licence,
checkpoint acceptance and objective/runtime identities, comparator versions,
independent results and the disclosure-policy decision above. (The regressor identity/licence, the comparator versions and the disclosure policy were later decided by owner call t0352.) Figure 7 units
and the evaluation-count unit are resolved by #806 and #819.

**Editorial dispositions:**
- Figure 2 keeps conceptual positions. Its caption explicitly says they are
  not measured limits and that the single proposal remains an empirical question.
- Figures 3 and 4 remain separate views of the game feedback contract and the
  reference network dataflow; merging them is not required for those distinct roles.
- Cuts and the mean log error ratio are defined in #839 from the admission
  contract, separately from the single-program comparison.

## Writer follow-up and remaining evidence

The follow-up PRs preserve Results numbers and claims. Source inspection and a
successful document build are distinct from qualification of a training run or
checkpoint.

| Follow-up | Landed change | Evidence limit |
|---|---|---|
| #779 | Wolf publication month corrected; 159 keys in the compiled `refs.bib` checked with no exact or case-folded duplicates | Broader link/author-list style policy remains open (O6) |
| #791 and #814 | Historical writer, inspected evaluation-score ranking adapter and conditional pool metric separated; actual adapter inputs, objective, two-fold evaluation and pool admission described | The retained run still needs exact source/input/normalization and population bindings |
| #806 | Existing real-log plots identify adapter pool utility, retained-data gaps, throughput and program-promotion units | No new experiment, one-proposal benchmark or transfer result |
| #811 | Historical training specifications attributed to supplied source notes | Checkpoint acceptance and runtime qualification remain required |
| #819 | Evaluation-count unit corrected in Method and Appendix B, with before/after images | Test-selected entry stays excluded |
| #824 | Schematic writer equation distinguished from native smoothing and normalization; fixed aggregate coefficients source-checked | Checkpoint objective identity/configuration remains required |
| #839 | Admission cuts, nested averaging and ratio direction defined | The checkpoint admission record and independent single-program test remain separate requirements |

The paper retained 32 compiled TBD markers until the owner decisions of
t0352: the full-upgrade snapshot had 31, and #791 added an explicit
retained-run binding requirement. Pass 18 (#982) filled the regressor
identity and removed the two comparator rows without receipts, leaving 28
(see `RESULTS_NEEDS.md`). Existing independent results, acceptance, parity,
sandbox and input-closure receipts remain team inputs. The owner decisions above
are preserved; these writing passes do not change disclosure rules or qualify a
model for deployment.
