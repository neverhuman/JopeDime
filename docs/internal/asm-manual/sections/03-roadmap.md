# Roadmap: critical order, task index and owner decisions

## Purpose and scope

This checklist is the requested property-list of remaining work. `roadmap.json` is its machine-readable source, with stable IDs, priorities, evidence levels, dependencies, commands, required receipts, budgets and source references. It describes source 68f56404 and the stopped training64, product28, warm-target and winner-export follow-on patches ([§](02-reading-this-manual.md#names-and-codenames)). PR83 is merged: remote main `a75e543d2166017cb0139e0a251d51d9a0c98e08` has the reviewed source tree, and main-guard 35889731910 attempt 1 succeeded. As of 2026-09-23 22:00 UTC no later change, including this manual, has merged to `main` ([§](02-reading-this-manual.md#implementation-snapshot-and-this-manuals-own-identity)). Clean canonical main was verified before the documentation branch. LIFE-01 records the completed readback; later source work still needs its own lifecycle.

P0 blocks the specified downstream proof or integration; it does not mean every item is a newly discovered defect. P1 is required for the full requested demonstration or manual. P2 is explicit subsequent work. Commands are unexecuted acceptance templates unless their named evidence says otherwise. Variables must resolve to actual final-source artifacts and emitted identities. An empty command list records a missing truthful interface rather than inventing one.

There are 43 primary items. Five of them (DOC-05, DOC-06, PACK-01, FOCUS-01 and LOOP-01) respond to the user’s training-history, continuation, flywheel, focused-search and packing questions. Documentation work does not wait for implementation; implementation and numerical follow-ons preserve the original budgets and frozen protocol.

## Critical order

Verified reviewed main → guarded source integration → actual tests, host qualification and final packaging → fresh qualified contexts and preregistered real teachers → all-eligible family-disjoint corpus → one GPU-hour training with continuation → explicit new-checkpoint consumption → real fitted warm demonstration → unchanged thirty-dataset campaign, selected-state backend qualification inside its remaining window, and report → eligible motif reuse.

Library parity fixtures precede the full panel. Actual selected-winner exports require complete confirmations and must still fit the original execution window. They cannot consume its reporting reserve or silently run after expiry. Documentation can proceed alongside source preparation; actual numerical examples need an allocated execution window.

Do not spend the GPU hour until the regression consumer and its actual fixture tests are ready. Do not freeze the final official panel configuration until its checkpoint is chosen and consumed. Separate proof allowances suggested in older plans are proposals, not new budget authorization.

## Compact task index

| ID | Priority | Stage | Workstream | Current state | Dependencies |
|---|---|---:|---|---|---|
| LIFE-01 | P0 | 0 | coordination | complete with retained readback | None |
| DOC-01 | P1 | 0 | manual | canonical private manual assembled | LIFE-01 |
| DOC-02 | P1 | 0 | manual examples | complete at named recipe/refit scope | DOC-01 |
| DOC-03 | P1 | 0 | manual figures | figures rendered; final PDF inspection pending | DOC-01 |
| DOC-04 | P1 | 0 | manual quality | rendered; final publication gates pending | DOC-01, DOC-02, DOC-03 |
| INT-01 | P0 | 0 | serial integration | stopped unmerged packets | LIFE-01 |
| INT-02 | P0 | 0 | integration qualification | pending | INT-01 |
| BENCH-01 | P1 | 1 | frozen benchmark foundation | implemented; retain during integration | INT-02 |
| BENCH-02 | P1 | 1 | external comparators | refresh at final environment | INT-02 |
| EXEC-01 | P1 | 2 | bounded workers | implemented; CI coverage gap retained | INT-02 |
| EXEC-02 | P1 | 2 | fault regression tests | test gap: drain-and-fixtures-F2 | INT-01 |
| EXEC-03 | P1 | 2 | receipt replay regression tests | test gap: resume-fix-F2 | INT-01 |
| EXEC-04 | P1 | 2 | cutoff semantics regression | test gap: regressions-F2 | INT-01 |
| EXEC-05 | P1 | 2 | accounting and unknown outcomes | pending execution | INT-02 |
| NEURAL-01 | P0 | 3 | context qualification | requires final-source binding | INT-02 |
| TEACH-01 | P0 | 6 | HWM comparison identity | not runtime qualified | INT-02 |
| TEACH-02 | P0 | 6 | label exposure readers | residual eligibility-B6 | TEACH-01 |
| TEACH-03 | P0 | 3 | real teacher collection | blocked on fresh admitted evidence | BENCH-01, BENCH-02, NEURAL-01, TEACH-01, TEACH-02 |
| TEACH-04 | P0 | 3 | real corpus materialization | blocked on two real families | TEACH-03 |
| TRAIN-01 | P0 | 3 | actual trainer tests | residual training-F3 | INT-02 |
| TRAIN-02 | P0 | 3 | native supervised training | host proof pending | TRAIN-01 |
| TRAIN-03 | P0 | 3 | bounded real GPU training | blocked on admitted data and consumption readiness | TEACH-04, TRAIN-02, NEURAL-02 |
| TRAIN-04 | P0 | 3 | checkpoint continuation and export | requires actual campaign | TRAIN-03 |
| NEURAL-02 | P0 | 3 | trained-checkpoint consumer readiness | compile/runtime fixture pending | INT-02, TRAIN-01 |
| NEURAL-03 | P0 | 3 | live newly trained consumption | newly trained consumption pending | TRAIN-04, NEURAL-01 |
| INFER-01 | P0 | 4 | fitted state and privacy | qualification pending | INT-02 |
| INFER-02 | P1 | 4 | exact selected-winner portability | actual export/parity unrun | INFER-01, PANEL-01 |
| INSTALL-01 | P1 | 5 | private installation | cold/live path pending | INT-02 |
| INSTALL-02 | P1 | 5 | final input/runtime preparation | pending final source | INSTALL-01, BENCH-01, BENCH-02 |
| DEMO-01 | P1 | 5 | measured warm fitted demonstration | real trained warm proof pending | NEURAL-03, INSTALL-02 |
| PANEL-01 | P1 | 5 | unchanged thirty-dataset campaign | pending final trained configuration | BENCH-01, BENCH-02, NEURAL-03, DEMO-01, INFER-01, INSTALL-02 |
| REPORT-01 | P1 | 5 | complete measured report | full campaign report pending | PANEL-01, EXEC-05 |
| MOTIF-01 | P1 | 6 | eligible reusable fragments | mining front door incomplete | TEACH-01, TEACH-02, TEACH-03 |
| MOTIF-02 | P1 | 6 | reuse validation | live reuse pending | MOTIF-01 |
| MOTIF-03 | P2 | 6 | new operators | future conditional work | MOTIF-02 |
| IP-01 | P1 | 0 | public privacy boundaries | residual D-IP4 and future PR text work | INT-01 |
| MAINT-01 | P2 | 0 | broad diagnostic qualification | not closed by fast CI | LIFE-01 |
| FUTURE-01 | P2 | 4 | CUDA inference and serving promotion | deferred | INFER-02, REPORT-01 |
| DOC-05 | P1 | 0 | real training and target history | Original training and target values verified; final composed-document inspection required | DOC-01 |
| DOC-06 | P1 | 0 | versions, KPIs and continuation runbook | Current and pending interfaces source-audited; independent chapter review complete | DOC-01 |
| PACK-01 | P2 | 3 | packed training consumer and incremental loading | Legacy binary store exists; pending Python category route still materializes JSONL | TEACH-04, TRAIN-01 |
| FOCUS-01 | P2 | 6 | sustained single-dataset improvement | Current bounded proof exists; no incumbent-seeded attack front door | TEACH-01, TEACH-02, TEACH-03 |
| LOOP-01 | P2 | 3 | repeatable measured learning flywheel | New complete improved-target-to-improved-search cycle not yet qualified | NEURAL-03, TEACH-04, TRAIN-04 |

## Proposed owner decisions and external dependencies

The property list is self-contained: every proposed dependency resolves to a primary item, a coordinated proposal, or the registry below. The original contributor proposal is preserved as `data/coordinated-roadmap-proposal.json`, SHA256 `334ae97446ef29963bce47582f1550c7d369f0b329ef143c128685991422ffd4`. Its recommendations are contributor suggestions, not fresh user instructions. Already explicit user decisions require no reconfirmation.

| Reference | Meaning | Current disposition |
| --- | --- | --- |
| OD-1 | Confirm the name ASM in writing (reported by Codex 16:07Z) | ASM naming is already explicit user direction; no reconfirmation needed. Any protocol rename remains a separately versioned change. |
| OD-2 | Confirm HWM train-side OOF admission and the test-to-development relabel with burn receipts (audit D3) | Train-side OOF admission and exposed-test-as-development treatment are already explicit user direction; no reconfirmation needed. |
| OD-3 | D-IP1: ban 'tree' in emitted code | Unresolved contributor proposal; not a newly imposed approval prerequisite and not execution authorization. |
| OD-4 | D-IP2: extra_trees naming | Unresolved contributor proposal; not a newly imposed approval prerequisite and not execution authorization. |
| OD-5 | D-IP3: publication boundary | This private manual is explicitly requested and authorized. Broader public-release boundary changes remain separate; this item does not block the manual. |
| OD-6 | D-IP4: PR 83 identifiers | Unresolved contributor proposal; not a newly imposed approval prerequisite and not execution authorization. |
| OD-7 | B6 fit fence and data rows | Unresolved contributor proposal; not a newly imposed approval prerequisite and not execution authorization. |
| OD-8 | Teachers from panel families 230_machine_cpu and 712_chscase_geyser1 | Prospective small families already selected from provenance; exposure labels require actual evidence. Unknown must not become known-seen solely from this recommendation. |
| OD-9 | W2 scale-out CPU (about 4 h) | Proposed additional compute outside the current allocation; no new hours or deadline are granted. |
| OD-10 | 3,755 unknown-licence catalog datasets | Unresolved data-use proposal. Unknown license metadata does not establish permission for any proposed use. |
| OD-11 | Lineage-B corpora | Unresolved contributor proposal; not a newly imposed approval prerequisite and not execution authorization. |
| OD-12 | Multiclass data (the catalog has none) | Unresolved contributor proposal; not a newly imposed approval prerequisite and not execution authorization. |
| OD-13 | Sealed one-shot evaluation | Outside current execution scope: sealed evaluation remains untouched. |
| OD-14 | Cost scenario constants | Unresolved contributor proposal; not a newly imposed approval prerequisite and not execution authorization. |
| OD-15 | CUDA timing | Unresolved contributor proposal; not a newly imposed approval prerequisite and not execution authorization. |
| OD-16 | Second GPU hour | Proposed additional compute outside the current allocation; no new hours or deadline are granted. |
| OD-17 | The proposed 300 s proof windows | Per-proof resource files must identify actual admission and remaining allowance; an example duration does not create a new budget. |
| product28 merged | Product28 stopped overlay has completed its own reviewed merge; part of INT-01, currently pending. | External prerequisite, explicitly unresolved unless a separate retained receipt closes it. |
| product28 on main | Same product28 reviewed-main prerequisite; currently pending. | External prerequisite, explicitly unresolved unless a separate retained receipt closes it. |
| training64 merged (eligibility API) | Training64 eligibility interface has completed its own reviewed merge; part of INT-01, currently pending. | External prerequisite, explicitly unresolved unless a separate retained receipt closes it. |
| freeze passed | Final implementation/runtime configuration and required acceptance are frozen; documentation freeze alone does not satisfy this. | External prerequisite, explicitly unresolved unless a separate retained receipt closes it. |
| compute window | A concrete finite resource allocation is admitted; no new allowance is implied. | External prerequisite, explicitly unresolved unless a separate retained receipt closes it. |
| DATA-01 for part b | The metadata inventory proposal DATA-01 is complete for the catalog-generation part of DOCFIX-01. | External prerequisite, explicitly unresolved unless a separate retained receipt closes it. |

The dispositions above are the roadmap data's own and are not edited here. Their state at 2026-09-23 22:00 UTC:

- training64 and product28 remain stopped patches outside `main`, so the rows “product28 merged”, “product28 on main” and “training64 merged (eligibility API)” stay open.
- OD-8: both proposed teacher families, `230_machine_cpu` and `712_chscase_geyser1`, are regression datasets of the frozen panel. If they are used to manufacture teachers, later evaluations on them must be labelled exposed/adaptive ([§](13-flywheel.md#freezing-the-experiment-and-choosing-families)).
- DATA-01: Claude's metadata-only packet was released at 19:30:23 UTC (manifest `SHA256SUMS`, `4bebee5a…`, 148 entries), but its sealed-name gate failed on one raw file. A later clarification (`RELEASE-CLARIFICATION.md`, `380f3402…`, with manifest `SHA256SUMS.release-2`, `a4b0c436…`) limits the release to eight aggregate files and keeps `raw/`, `scratch/` and the full HANDOFF in custody. Codex imported only an aggregate subset (stopped subset `558548be…`): the summary `e4dbe075…` and the release pointer `366fa000…`, present here as `data/data-estate-extension-summary.json` and `data/data-estate-extension-release.json`. No retained receipt closes “DATA-01 for part b”, and DOCFIX-01 remains a separate follow-up documentation PR.

## Stopping and completion rules

An unsuccessful prerequisite remains a recorded refusal or failure. Do not substitute synthetic real-data teachers, another winner, easier datasets or a new deadline. Source completion does not close a runtime task. A plan, help output or checkpoint file is not the corresponding live receipt.

Full delivery requires all thirty outcomes, losses and gaps, backend limitations, measured unique-predictor/recipe costs, honest training category scope and actual newly trained consumption. Ten wins per task and an under-sixty-second warm result remain declared goals, not promises. No win-counting rule is ratified: [§](02-reading-this-manual.md#glossary) gives the proposed PANEL-WINS definition, pending owner ratification, and [§](13-flywheel.md#win-counting-and-a-proposed-acceptance-record) the separate win columns. The one-hour/ten-million-token comparison must retain unknown hardware, workload, token and currency quantities.

The document gate requires one readable PDF plus matching section Markdown, reproducible figure inputs, source/lifecycle references and this maintained checklist. The maintainer-private examples are authorized documentation; their existence does not expand public inference bundle contents.
