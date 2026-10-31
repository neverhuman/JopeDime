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
| O6 | Low | Reference entries are inconsistent: DOI versus URL, "(NeurIPS 2017)" in some book titles but not others, and full author lists of up to about 58 names (refs [19], [66], [114]). | Pick one link policy. Consider truncating long author lists ("et al." beyond ~10), which needs a `.bst` or a biblatex switch. The Wolf publication month was corrected in [#779](https://github.com/neverhuman/JopeDime/pull/779); the link and author-list style choices remain open. | refs.bib owner |
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
| 3 | Method (§5–6): the §6.5 objective set as display equations (O5). Three TBDs become accurate statements: exploration in the reference procedure (§5.4, from §6.5/§6.7); the checkpoint acceptance record as an evaluation-contract input (§6.5); and what the admission CSVs do and do not record (§6.8). The regressor identity TBD (§6.2) stays for the team. | merged (#777) |
| 4 | Results presentation only, with every sentence, number and TBD of §7 preserved:<br>• "Evaluation plan at a glance" table (Table 5).<br>• Earlier diagnostics moved into their own closing subsection (§7.7).<br>• Worked-game fields moved from §4.6 into a §7.6 record table; §4.6 becomes "The analogy's limits".<br>• Pilot-cost figure moved from §5.2 into §7.4.<br>All results-dependent placeholders now sit in §7. | merged (#786) |
| 5 | Discussion, Conclusion and appendices:<br>• §8 becomes "Discussion", with each subsection ending in an explicit outcome-to-claim rule. Four per-subsection TBDs become pointers to the §7 subsections that will settle them; the opening TBD stays as the single tracked marker.<br>• New "Limitations" subsection (§8.5) and a §9 Conclusion.<br>• Appendix A becomes a domain-by-domain referee inventory (Table 9), restating referees already cited in §3–§4.<br>• Receipt TBDs in Appendices B and C point to the §7.1 evaluation record.<br>The regressor licence TBD stays. | merged (#789) |
| 6 | Fixes from an independent camera-ready review (dispositions below): scope stated in the abstract and introduction; honest evidence status; ratchet, map and loss wording matched to §6 and the code-level records; exploration programs added; drafting-process language removed; pointer, typography and Results-presentation slips fixed. | merged (#792) |

## Needs Jepson/team

Everything below needs data, receipts or decisions from the team. None of it was invented or changed. Unless a row says otherwise, it lives in §7.

### Results content that is missing

| Where | What is missing |
|---|---|
| Abstract | One sentence with the headline first-proposal result and its uncertainty. |
| §7 opening | Independent evaluation evidence for any performance claim. A source-bound benchmark for any deployment-throughput claim. |
| §7.1 contract | The final problem inventory, dataset identities and content hashes, row splits, family assignments and exclusions, component provenance, evaluator, checkpoint, seeds, task metrics and failure accounting. All of it must be frozen before outcomes are opened. |
| §7.2 effect of training | The matched first-proposal comparison: the trained writer against the frozen reference writer, paired first draws under predeclared seeds, all draws reported. Source-bound evidence of optimizer updates and parameter changes. This is the central claim of the paper. |
| §7.3 fixed methods (Table 6, Figure 8) | Paired outcomes with uncertainty for algorithms designed by Claude, Codex and Gemini and for KumoRFM, including ties, failures and exclusions. The originating models, prompts, dates and implementation checks. The KumoRFM configuration and permitted inputs. The leaderboard's methods, paired outcomes, rating protocol and uncertainty. |
| §7.4 cost of attempts (Figures 9–10) | Attempt-versus-performance and cost curves for agents allowed successive attempts, with matched budgets, repeated seeds and uncertainty. Pilot cost data: per-stage failure shares, and cost per verified win by writer size and reasoning effort, with seeds and family-level uncertainty. Settings with no verified win must be identified without assigning them a finite cost. |
| §7.5 map (Table 7) | A held-out ranking study, a problem-similarity study and program-only-control evaluations. Either an implemented route by which the map influences training, so a causal ablation is possible, or a decision to limit the claim to ranking. A failure-class comparison. |
| §7.6 failures (Table 8) | Attempt inventories and costs by terminal outcome class, with unresolved cases identified. The worked example's selection rule, chosen before inspection, and its record: problem, encoding time, emitted output, check and fitting outcomes, and hidden-fold and incumbent scores (or the failure and its cost). |
| §7.7 diagnostics | Pinned inventories, checkpoints, seeding, timing and ranking protocols for the historical diagnostics. |
| §5.4 to §7 | The measured exploration quality–cost trade-off, if an exploration study is run. |

### Results evidence that exists but is weak

- **The plotted development evidence concerns an adapter admission record (§6.8).** Over 695 rounds there are two strict validation gains, then a plateau, and a train–validation gap of 0.282–0.290 on 25 summaries. It is not an untouched test and cannot support the transfer claim.
- **The historical diagnostics (§7.7) do not show a learned proposer outperforming search.**
  - 3,227 of 3,268 paired searches ended in exact ties.
  - 98.84% of reported winners were already in their seed populations.
  - An earlier model's ranking accuracy, 0.506 [0.441, 0.568], includes chance.
- **§7.2 has no comparison at all yet.** Until it does, the abstract, the introduction and §8 can only state what would be tested.

### Outside Results (provenance and receipts)

- **§6.2 / Appendix C.** The pretrained tabular regressor behind the 1,024-value block needs a name, version and licence.
  - The repository calls the component "Foundation" (`starlight_core_regressor`).
  - Its training source describes a TabPFN v3 checkpoint-compatible architecture.
  - The asset manifests say licence declarations vary by artifact.
  - Please confirm whether it is third-party or locally trained, and under which licence it ships.
- **§6.5 / Appendix C.** The reference checkpoint's acceptance record, its held-out promotion receipt and a pinned runtime.
- **§6.8.** Bind the retained observations to exact loop source and input identities, adapter configuration, score normalization, evaluation population, candidate pools and revealed-history inputs. The implementation is now described; run qualification and cross-model comparisons still require those records.
- **Appendix B.** The final pre-registration and audit receipts: the inner-fold label-poison test, the sealed canary and the train-only input-closure audit.
- **Appendix C.** Isolation and parity receipts from the final evaluator.

### Owner decisions

- **Disclosure boundary.** `deploy/ci/ip-scrub.sh` states that "nothing public may say how the program is found". The script scans only top-level prose, not this paper. Even so:
  - §3.2 and §3.5 discuss genetic programming and evolutionary search as history.
  - §4.3 cites strongly typed GP for the program language.
  - Appendix A calls "typed genetic programming and evolutionary search" milestones "directly needed for our argument".

  Decide whether the white paper falls under the boundary. If it does, the Appendix A wording and the §4.3 citation should change.
- **Regressor licensing.** The §6.2 editor note says the encoder bundle lists TabICL-v2 artifacts as "provenance pending, redistribution prohibited, policy-ineligible". The legacy training source describes a TabPFN-v3-compatible architecture. Confirm which model it is, its version and licence, and whether that limits releasing the checkpoint.
- **Comparator identities.** "Claude, Codex and Gemini" in §1 and §7.3 need exact model versions and dates. "Codex" also collides with the 2021 Codex model cited in §3.1.

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
independent results and the disclosure-policy decision above. Figure 7 units
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

The paper retains 32 compiled TBD markers: the full-upgrade snapshot had 31,
and #791 added an explicit retained-run binding requirement. Existing independent
results, comparator identities, encoder identity/licence, acceptance, parity,
sandbox and input-closure receipts remain team inputs. The owner decisions above
are preserved; these writing passes do not change disclosure rules or qualify a
model for deployment.
