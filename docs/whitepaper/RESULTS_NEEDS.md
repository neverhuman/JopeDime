# What Results still needs

A checklist for Jepson and the team. Status: `main` at `5fed0c9f`, 2026-10-31.

The compiled paper has **32 `\TBD` markers**:
- 27 in Results (§7);
- 2 that wait on Results (the abstract and the Discussion);
- 3 provenance items that every result inherits (§6.2, §6.8 and Appendix C).

Writers cannot fill any of them from the repository. Each one needs new evaluation outputs, receipts or an owner decision. This file changes no claim or number in the paper. [`LAYOUT_REDTEAM.md`](LAYOUT_REDTEAM.md#needs-jepsonteam) holds the background, and its evidence-handoff table defines the packet components named below.

## Order of work

1. **Freeze the evaluation contract (§7.1).** Every comparison depends on it. Freeze it before any final outcome is opened.
2. **Run the matched first-proposal comparison (§7.2).** This is the paper's central claim. The abstract's headline sentence and the Discussion wait on it.
3. **Run the remaining comparisons (§7.3–§7.6):** fixed methods, the cost of more attempts, the map, failures and the worked example.
4. **Pin the historical diagnostics (§7.7)** to their own inventories and protocols.
5. **Then write the abstract's result sentence**, resolve the two markers at the start of §7, and reconcile the Discussion.

## Every marker, by location

Source lines are as of `5fed0c9f`; P3's line was updated when §6.8 gained a sentence (layout pass 8). "Packet" names the component of the evidence-handoff table that supplies the item.

### Results (§7): 27 markers

| # | Where | Marker asks for | Packet |
|---|---|---|---|
| R1 | §7 opening ([07_results.tex:9](sections/07_results.tex#L9)) | Independent evaluation evidence before any performance claim. It is resolved by the completed comparisons in §7.2–§7.4. | Reproducible summaries |
| R2 | §7 opening ([:10](sections/07_results.tex#L10)) | A source-bound benchmark before any deployment-throughput claim; without one, make no throughput claim. | None yet: add a benchmark record bound to its source commit and inputs |
| R3 | §7.1 contract ([:65](sections/07_results.tex#L65)) | The final problem inventory, dataset identities, row splits, family assignments and exclusions, component provenance, evaluator, checkpoint, seeds, task metrics and failure accounting. | Frozen evaluation contract |
| R4 | §7.2 effect of training ([:81](sections/07_results.tex#L81)) | The matched first-proposal comparison: the trained writer against the frozen reference writer, with paired first draws under predeclared seeds and every draw reported. Also source-bound evidence of optimizer updates and parameter changes. | Paired first-proposal record |
| R5–R8 | Table 6 ([:104–107](sections/07_results.tex#L104)) | A paired evaluation with uncertainty for each comparator: the Claude-, Codex- and Gemini-designed methods and KumoRFM, including ties, failures and exclusions. This needs exact model versions and dates (see the owner decisions below). | Fixed-method comparison record |
| R9 | Figure 9 ([:114](sections/07_results.tex#L114)) | The leaderboard: evaluated methods, paired outcomes, rating protocol and uncertainty. Elo summarizes a stated comparison population and does not replace paired results. | Fixed-method comparison record |
| R10 | §7.4 cost of attempts ([:130](sections/07_results.tex#L130)) | Attempt curves with matched budgets, repeated seeds and uncertainty. | Attempt and cost record |
| R11 | Figure 10 ([:136](sections/07_results.tex#L136)) | Performance against attempts and compute, including failed attempts and unreached targets. | Attempt and cost record |
| R12 | Figure 11 ([:145](sections/07_results.tex#L145)) | Pilot cost: per-stage failure shares and cost per verified win. | Attempt and cost record |
| R13 | Figure 11 caption ([:151](sections/07_results.tex#L151)) | Pilot data, seeds and family-level uncertainty. Settings with no verified win are named, not given a finite cost per win. | Attempt and cost record |
| R14 | §7.5 map ([:162](sections/07_results.tex#L162)) | Problem-similarity and program-only-control evaluations, on data disjoint from fitting the map. | Map, example and diagnostic records |
| R15 | §7.5 map ([:170](sections/07_results.tex#L170)) | A **team decision**: implement a route by which the map changes training, or limit the map study to ranking. In the described checkpoint the map neither conditions nor screens the writer's draw. | Map, example and diagnostic records |
| R16–R18 | Table 7 ([:182–184](sections/07_results.tex#L182)) | A held-out ranking study; a matched causal ablation, which presupposes the training route in R15; and a failure-class comparison. | Map, example and diagnostic records |
| R19 | §7.6 failures ([:198](sections/07_results.tex#L198)) | Attempt inventories and costs by terminal outcome class, with unresolved cases identified. | Attempt and cost record |
| R20 | §7.6 worked example ([:203](sections/07_results.tex#L203)) | The selection rule, fixed before inspection; the actual program; and its verified outcomes, with no replacement of failed attempts. | Map, example and diagnostic records |
| R21–R26 | Table 8 ([:220–225](sections/07_results.tex#L220)) | The selected game's problem, task and metric; encoding time; the emitted program or unfinished draft and its size; check and fitting outcomes and time; the hidden-fold score; and the incumbent's score. If there is no valid fit, the failure and its incurred cost replace the scores. | Map, example and diagnostic records |
| R27 | §7.7 diagnostics ([:246](sections/07_results.tex#L246)) | Pinned inventories, checkpoints, seeding, timing and ranking protocols for the historical diagnostics, kept separate from the final comparisons. | Map, example and diagnostic records |

### Waiting on Results: 2 markers

| # | Where | Marker asks for | Unblocked by |
|---|---|---|---|
| D1 | Abstract ([00_abstract.tex:25](sections/00_abstract.tex#L25)) | One sentence with the headline first-proposal result and its uncertainty. | R4 |
| D2 | Discussion ([08_meaning.tex:9](sections/08_meaning.tex#L9)) | Every conclusion in §8 reconciled with the completed evaluation. §8.1–§8.4 already end with the outcome that decides each claim. | R1–R27 |

### Provenance every result inherits: 3 markers

| # | Where | Marker asks for | Notes |
|---|---|---|---|
| P1 | §6.2 ([06_method.tex:44](sections/06_method.tex#L44)) | The name, version and licence of the pretrained tabular regressor behind the 1,024-value block. | An owner decision; see "Regressor licensing" in `LAYOUT_REDTEAM.md`. |
| P2 | Appendix C ([appendix_stubs.tex:73](sections/appendix_stubs.tex#L73)) | The same component's name, version and licence. | One answer fills P1 and P2. The two markers spell it "licence" and "license"; use one spelling when filling them. |
| P3 | §6.8 ([06_method/flywheel_status.tex:41](sections/06_method/flywheel_status.tex#L41)) | The retained snapshot bound to the exact loop source, adapter, candidate pools, revealed-history inputs and utility normalization. | Concerns the §6.8 development record (the retained adapter-loop snapshot), not the §7 comparisons. |

## Evidence that exists but cannot fill these markers

These records are already described in the paper. None of them is an untouched first-proposal test.

- **The adapter admission record (§6.8, Figures 5–6).** Over 695 rounds it shows two strict validation gains, then a plateau, and a train–validation gap of 0.282–0.290 over 25 retained summaries. Validation outcomes inform admission.
- **The program-promotion telemetry (§6.8, Figure 8).** Strict promotion is a numerical gate in one retained window, and the journal supplies no confidence interval.
- **The historical diagnostics (§7.7).**
  - 3,227 of 3,268 paired searches ended in exact ties.
  - 98.84% of reported winners were already in their seed populations.
  - An earlier model's ranking accuracy is 0.506 [0.441, 0.568], an interval that includes chance.
- **The code-level record** (`docs/internal/asm-manual/sections/deep-stack.md`) says the recorded neural winner "descends from exploration candidates, not from a neural_policy proposal". If that still holds, no writer proposal has won in the development record.

## Decisions needed before the results can be written

Each of these is described under "Owner decisions" in [`LAYOUT_REDTEAM.md`](LAYOUT_REDTEAM.md#owner-decisions).

- **Comparator identities (R5–R9).** The exact model versions and dates behind "Claude, Codex and Gemini". "Codex" also collides with the 2021 Codex model cited in §3.1.
- **Regressor identity and licence (P1–P2).** The licence may also limit releasing the checkpoint.
- **The map's role (R15–R18).** Either a training route, or a ranking-only claim.
- **Disclosure boundary.** Whether the paper falls under `deploy/ci/ip-scrub.sh`'s rule. This affects the §4.3 and Appendix A wording, not the Results.
- **Reference checkpoint (§6.5, Appendix C).** Its acceptance record, held-out promotion receipt and pinned runtime. No marker carries these, but §6.5 names the acceptance record and a pinned runtime as required inputs to the §7.1 evaluation contract.

## How to hand evidence in

Follow the evidence-handoff table in `LAYOUT_REDTEAM.md`:
- Bind every component to its source commit, artifact location, content hash, generating command and evaluation protocol.
- Freeze the contract before opening outcomes.
- Report every draw.
- Keep failed generations and executions in the denominator.
- Keep unknown outcomes explicitly unknown.
- Report paired family-level outcomes first, with Elo only as a summary of a stated comparison population.

## Keeping this file current

Remove a row in the same PR that removes its `\TBD`. The row count must match the compiled count:

```sh
cd docs/whitepaper && make pdf && pdftotext eurekazero_whitepaper.pdf - | grep -o 'TBD' | wc -l   # 32 today
grep -rn --exclude-dir=sources '\\TBD{' sections   # each marker's starting line (32 today); sources/ holds verbatim drafts
```
