# Evidence index, reviewed delivery and glossary

## Source and lifecycle identity

This edition describes the implementation reviewed at `68f56404c3e199e06413bfd645d3fe0803d7b96f`. PR83 merged on 23 September 2026 at 16:34:47 UTC. The resulting remote main is `a75e543d2166017cb0139e0a251d51d9a0c98e08`. Both revisions have Git tree `176b75bc36d76c7ce4eaeb7234677ec013468889`. Rebase merge changed commit identity; it did not change the reviewed source tree.

The manual itself is a later documentation change. Its publication identity must be read from its own Git history and CI receipt, not confused with the implementation snapshot. Follow-on patches described as pending are outside the implementation snapshot even if they are preserved on this host.

| Gate | Exact observation |
| --- | --- |
| PR | `veox-ai/JopeDime#83`, merged |
| Manual CI dispatch | Run 35887284772, attempt 1, source `68f56404…`, success |
| PR CI | Run 35887255898, attempt 1, source `68f56404…`, success |
| Required CI jobs | Eight lanes and aggregate succeeded in both runs |
| CI test inventory | 937 Rust passed, 14 skipped; 221 Python passed per run |
| First independent native review | 5293748183, `neverhuman`, APPROVED, exact full reviewed head, 16:29:01 UTC |
| Second independent native review | 5293788772, same eligible account, APPROVED, same head, 16:32:43 UTC; includes residual disclosures |
| Author eligibility | All twelve commits associated with `jepsontaylor` as author and committer; reviewer differs |
| Merge | Guarded rebase with `--match-head-commit` at the reviewed head |
| Remote-main verification | Fetched main and compared complete Git tree; canonical clean main observed before documentation branch |
| Main guard | Run 35889731910, attempt 1, success at resulting main; guard job 107278860508 |

Two review processes overlapped during coordination. Both were independent of the source writer and both approvals are retained; they are not two different human reviewers. One process's supplementary disclosures identify outstanding work rather than silently treating it as fixed. The writer did not select another credential to approve its own source.

The new full-main diagnostic run 35889732032 was cancelled while queued to preserve the admitted local allocation and let main-guard run. It did not pass or execute. Earlier full-main run 35852022466 failed and remains historical diagnostic evidence requiring repair. Passing fast CI does not replace that outstanding broad qualification.

## Important local receipts

The durable custody root is `/home/ubuntu/jope-rescue-20260921/asm-integration-20260923/`. These are host-local receipt locations, not promises that every reader has the same files. The structured documentation evidence identifies the source observations copied into this edition. Weight bodies, source dataset bodies and sealed labels are not embedded in the manual.

| Relative custody path | Meaning |
| --- | --- |
| `neural-integration/lifecycle-68f56404c3e199e06413bfd645d3fe0803d7b96f/` | Full author/committer inventory, CI, native reviews, merge and main-guard readbacks |
| `neural-integration/pr83-final-review-68f56404c3e199e06413bfd645d3fe0803d7b96f/` | Independent review prompt, source/proof inspection and native approval receipt |
| `neural-integration/source-match-68f56404c3e199e06413bfd645d3fe0803d7b96f.json` | All 2,855 Git blobs match the immutable build input inventory |
| `neural-integration/native-context-68f56404c3e199e06413bfd645d3fe0803d7b96f/` | Eight native encoder qualification cases under unchanged numerical policy |
| `neural-integration/origin-run-68f56404.json` | Final operational run command, timing and exit status |
| `neural-integration/origin-report-68f56404.json` | Independent recovered-report command and timing |
| `agent-final-acceptance/measured-origin-68f56404/audit.json` | Source/report/worker inventory and independently bound supervisor accounting |
| `neural-integration/final-head-pipeline-bench-68f56404.json` | Local 53 benchmark and 106 pipeline test results |
| `neural-integration/final-head-host-ten-68f56404.json` | Ten actual privileged host cases and binary identity |
| `agent-final-acceptance/failed-origin-d0f5748b/audit.json` | Preserved failed prior run; not an accepted current score |
| `agent-maintainer-manual/` | Section drafts, structured data inventory, example preparation and documentation observations |

Treat a receipt's command, source SHA, input hashes, runtime and original deadline as one evidence unit. Copying a number out of that unit can lose its scope. A local path alone proves neither continued availability nor integrity: verify the retained hash and the contents required for the claim.

## Reproduction boundaries

Rendering this manual uses its committed section Markdown, structured data, figure inputs and Python rendering tools. It should not fetch a corpus, load a model, run a fit or contact the sealed store. Plotting historical evidence is distinct from rerunning historical experiments. The included figure data keeps missingness, eligibility and snapshot identities.

The private toy example has its own synthetic input and output receipt. Even a successful five-language recipe/refit comparison cannot qualify a different selected fitted predictor. The pending fitted-inference acceptance must authenticate the retained winner and compare complete saved-state predictions with its original numerical policy.

Operational replay needs more than the document: the exact source/build, immutable workers, model/checkpoint/vocabulary, runtime dependencies, allowed data inputs, original plan and journals, and an available resource allocation. Reproduction that resumes an expired campaign cannot start fresh jobs by assigning a new deadline. A new experiment needs a separately recorded plan; it does not repair the old receipt.

## Glossary

| Term | Meaning in this repository |
| --- | --- |
| ASM | Algorithm Synthesis Model: learns to generate and optimize executable algorithms from data |
| Algorithm / recipe / program | Executable typed procedure and configuration; reusable across fits |
| GP | Genetic programming: search over executable programs using variation and measured selection |
| Fitted predictor | One recipe plus preprocessing and model state learned from exact training inputs |
| Checkpoint | Serialized neural proposal-model parameters and compatibility metadata; distinct from a fitted tabular predictor |
| Context | Authenticated numerical dataset/target representation supplied to the proposal policy |
| Dataset encoder | Learned or specified mapping from permitted dataset structure/rows to context features |
| Target encoder | Representation of allowed target information; task-specific and restricted to permitted fitting rows |
| Vocabulary | Token identities, grammar and executable restrictions associated with a checkpoint |
| Generation zero | Initial candidate slots before evolutionary descendants |
| Exploration candidate | An admitted generic candidate in the neural arm; not falsely labelled neural-origin |
| Offspring | Mutation or crossover descendant with authenticated parent identities and actual fit evidence |
| OOF | Out-of-fold: predict rows using a fit that excluded those rows; adaptive recipe selection can still expose them at another stage |
| Cut identity | Frozen identity of the training/scoring row assignment and evaluator context |
| Development data | Data visible to optimization or recipe/model selection, regardless of a file's name |
| Sealed evaluation | Separately protected evaluation outside the learning and optimization closure |
| HWM | High-water mark: best retained result under a complete comparison identity |
| Teaching admission | Eligibility decision permitting a measured example into policy training |
| Motif / macro | Typed reusable program fragment with eligible provenance; not automatically a new primitive operator |
| Numerical policy | Declared arithmetic, probability, ordering, tolerance and runtime assumptions fixed before scoring |
| Alias | Alternate spelling of an existing recipe; not another independent method fit |
| Unsupported | A backend or task deliberately refuses a capability outside its admitted contract |
| Unavailable | A dependency or required runtime is missing; different from a model failure |
| Failed | An attempted operation ended unsuccessfully; retain its reason and measured/unknown costs |
| Skipped | A known test or operation did not execute; not passing coverage |
| Journal | Incremental durable started/finished/observation records for audit and original-window recovery |
| Receipt | Evidence binding command, source, inputs, runtime, resources, times, status and artifacts |
| Provenance | The chain of source, data, model, recipe and execution identities supporting an artifact |
| Elo | Relative Bradley–Terry rating on the arena's declared method universe and eligible pairwise scores |
| Warm demo | Demonstration after verified preparation; report separately from cold installation |
| Promotion | Explicit qualification of an exact artifact/model/backend for its intended serving use; a merge alone is insufficient |

## Questions that should remain answerable after a maintainer handoff

Which reviewed source and runtime produced this artifact? Which exact rows and original labels were available? Was this a recipe, a checkpoint or a saved fitted predictor? Which score selected it, under what comparison identity? Which failures, missing dependencies and cancelled jobs were retained? Did any nominal test labels influence selection? Did resume preserve the original deadline? Does the export predict from saved state, and which backend actually passed? Did training perform positive updates and change parameters? Can another maintainer locate and authenticate the receipt without reading credentials or sealed data?

The roadmap translates the unanswered parts into named acceptance items. Unknowns stay explicit until the corresponding evidence exists.
