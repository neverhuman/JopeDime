# Manuscript source still needed

The manuscript has **32 `\TBD` markers**. Jepson must supply the remaining source for the abstract, independent empirical Results, Discussion and Conclusion. Closed-out development diagnostics fill only R28--R30, and a development reference board fills R6--R10, below; archived prose and repository notes do not fill the remaining placeholders. This inventory supersedes older marker counts in historical layout logs.

The Results section retains its evaluation protocol and planned exhibits. The fixed-method subsection reports a development reference board (R6--R10); independent first-proposal outcomes remain placeholders. The earlier-development diagnostics now report the merged, closed-out adapter studies authorized by the owner. All Discussion/Conclusion prose and the abstract remain placeholders. The provenance item in Section6.8 also remains pending.

Freeze the evaluation contract before opening final outcomes. Bind evidence to source, inputs, artifact hashes, commands and protocol; report every draw, failures, ties and unknown outcomes. Use paired family-level outcomes and uncertainty. Keep development diagnostics separate from independent final comparisons.

## Every current placeholder

| Marker | Source | Supplied source must cover |
| --- | --- | --- |
| A1 | [00_abstract.tex:2](sections/00_abstract.tex#L2) | Supply the problem statement, proposed method, supported task scope, contributions and evaluation summary from Jepson's source. |
| A2 | [00_abstract.tex:5](sections/00_abstract.tex#L5) | State the headline first-proposal result with its uncertainty once the independent evaluation is complete. |
| R1 | [07_results.tex:48](sections/07_results.tex#L48) | Add independent evaluation evidence before making a performance claim. |
| R2 | [07_results.tex:49](sections/07_results.tex#L49) | Supply a source-bound benchmark before claiming deployment throughput. |
| R3 | [07_results.tex:68](sections/07_results.tex#L68) | Document the final problem inventory, dataset identities, row splits, family assignments and exclusions, component provenance, evaluator, checkpoint, seeds, task metrics and failure accounting. |
| R4 | [07_results.tex:84](sections/07_results.tex#L84) | Supply the matched first-proposal comparison and training-update evidence before claiming transfer. |
| R5 | [07_results.tex:108](sections/07_results.tex#L108) | Supply the one-proposal writer's matched outcomes against these comparators on untouched families. |
| R11 | [07_results.tex:159](sections/07_results.tex#L159) | Supply attempt curves, matched budgets, repeated seeds and uncertainty. |
| R12 | [07_results.tex:165](sections/07_results.tex#L165) | Search curve: performance versus attempts and compute, including failed attempts and unreached targets. |
| R13 | [07_results.tex:191](sections/07_results.tex#L191) | Supply problem-similarity and program-only-control evaluations. |
| R14 | [07_results.tex:174](sections/07_results.tex#L174) | Pilot cost measurements: supply per-stage failure shares and cost per verified win. |
| R15 | [07_results.tex:180](sections/07_results.tex#L180) | Supply pilot data, seeds and family-level uncertainty; identify settings with no verified win without assigning a finite cost per win. |
| R16 | [07_results.tex:208](sections/07_results.tex#L208) | held-out ranking study |
| R17 | [07_results.tex:209](sections/07_results.tex#L209) | matched causal ablation |
| R18 | [07_results.tex:210](sections/07_results.tex#L210) | failure-class comparison |
| R19 | [07_results.tex:219](sections/07_results.tex#L219) | Identify an implemented influence path or limit the study to ranking. |
| R20 | [07_results.tex:230](sections/07_results.tex#L230) | Supply attempt inventories and costs by outcome class, identifying unresolved cases. |
| R21 | [07_results.tex:235](sections/07_results.tex#L235) | Supply the example selection rule, actual program and verified outcomes, with no replacement of failed attempts. |
| R22 | [07_results.tex:252](sections/07_results.tex#L252) | Identify the problem, task and metric. |
| R23 | [07_results.tex:253](sections/07_results.tex#L253) | time |
| R24 | [07_results.tex:254](sections/07_results.tex#L254) | program or unfinished draft and size |
| R25 | [07_results.tex:255](sections/07_results.tex#L255) | outcomes and time |
| R26 | [07_results.tex:256](sections/07_results.tex#L256) | score |
| R27 | [07_results.tex:257](sections/07_results.tex#L257) | score |
| D1 | [08_meaning.tex:4](sections/08_meaning.tex#L4) | Supply a source-grounded discussion of the completed evaluation from Jepson. Reconcile every conclusion below with those outcomes and their uncertainty. |
| D2 | [08_meaning.tex:8](sections/08_meaning.tex#L8) | Use Jepson's supplied output and archive comparisons, with program behavior checks, to distinguish selection from useful new composition. |
| D3 | [08_meaning.tex:12](sections/08_meaning.tex#L12) | Interpret transfer of first-proposal quality across problems using the matched conditions and paired family-level uncertainty supplied by Jepson. |
| D4 | [08_meaning.tex:16](sections/08_meaning.tex#L16) | Discuss referee validity, contamination, admission versus generalization, and documented failure limits from Jepson's source. |
| D5 | [08_meaning.tex:20](sections/08_meaning.tex#L20) | State which level of self-improvement is supported by source-bound training changes and independent proposal outcomes supplied by Jepson. |
| D6 | [08_meaning.tex:24](sections/08_meaning.tex#L24) | Describe limits on task coverage, the program language, referee reliability and measured cost using Jepson's supplied source. |
| D7 | [08_meaning.tex:28](sections/08_meaning.tex#L28) | Supply the conclusion and next steps justified by Jepson's results and their limits; reserve stronger claims until their required evidence is supplied. |
| P1 | [flywheel_status.tex:36](sections/06_method/flywheel_status.tex#L36) | Bind the retained snapshot to the exact loop source, adapter, candidate pools, revealed-history inputs and utility normalization. |

The marker table follows the current source lines. Update it whenever a placeholder is added, moved, filled or removed; keep the compiled count and source inventory consistent.

## Filled development diagnostics

R28 (outcomes and uncertainty), R29 (conditions and source pins), and R30 (scoped interpretation and limits) are filled from the merged H195, H197, H198 and H201 reports. The [source inventory](development_results.sources.json) pins every report, protocol, compact metrics and cohort record by full source commit, Git blob and SHA-256. H195 is a candidate, not adopted; H197 is NO_ADVANCE; H198 is NO_ADVANCE (inert treatment); H201 is formally INCONCLUSIVE with its primary attempt INCOMPLETE. H199 remains excluded until #1682 merges. None fills an independent first-proposal, throughput, abstract or Discussion/Conclusion marker.

## Filled development reference board

R6--R9 (the comparison rows of `tab:headtohead`) and R10 (`fig:leaderboard`) are filled from the `dev91-audit-v1` reference board on `v1/next`: the board ladder at `7f7ff32833` and the H414 FINAL study at `45008196`, which holds `XSTACK_2L`'s paired contrasts with all 30 other entrants. [`make_leaderboard_dev91.sh`](figures/scripts/make_leaderboard_dev91.sh) writes the table, the figure and [`leaderboard_dev91.json`](figures/data/leaderboard_dev91.json) from those two Git blobs after checking their SHA-256; `--check` reproduces all three. No entrant was fitted, scored or rerun. The row is a locked search-assisted stack rather than a first proposal, and the board is a reuse cohort (H257 tuned on it; H381 and H410 read five of its contexts), so R5, the one-proposal outcomes on untouched families, stays open.
