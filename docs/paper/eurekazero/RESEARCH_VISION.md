# EurekaZero research vision

This is the plan behind the north-star draft: what we believe, what the paper must show, and the
work that turns each synthetic result into a measurement. It is written for the people and agents
who will do that work.

## The north star

> **Every problem whose requirements can be encoded, whose solutions can be written in a typed
> language and whose quality an independent verifier can measure is a game — and a network trained
> on verified outcomes of search can make a strong first move in it without searching.**

AlphaGo Zero's raw network, with no search at all, played at roughly professional strength: most of
what search had found was amortized into one forward evaluation. EurekaZero asks for the same
property where every game is a new board: a dataset that needs a learning procedure, an optimization
history that needs its next experiment, a specification that needs a part. If it works, innovation
that today needs a search for every problem needs it once per family, and the first proposal arrives
while the problem is still being described.

## The innovation game

| Element | Go | Learning procedure | Next experiment | Engineering part |
| --- | --- | --- | --- | --- |
| State | board position | training rows, objective, budget | evaluated points, bounds, horizon | loads, interfaces, envelope, material, process |
| Move | one stone | one complete procedure | one query | one CAD construction program |
| Rules | rules of Go | typed procedure language | typed search space | typed CAD language, validity checks |
| Verifier | game result | fit, then error on held-out rows | the objective's measured value | build, mesh, finite-element load test |
| Opponent | the previous network | incumbent procedure for that dataset | incumbent decision for that history | incumbent design for that specification |

Four conditions make a problem an innovation game: an encodable state, typed moves, an independent
verifier and a declared budget. The paper tests three games and claims nothing about the rest
(claim EZ-C12 is *do not assert*). Many more qualify (Section 12, Table 13): molecules for a target,
reaction conditions, compiler transformations, analog circuit sizing, formal lemmas, control policies,
materials. Each is a candidate fourth game once the first three succeed.

## The interface

```text
z = E_{phi,d}(s)          game_state_encoding   domain adapter: typed tokens, invariances, masks
h ~ P_theta(. | z)        solution_embedding    shared proposer: latent slots; protected stage
a = D_{psi,d}(h)          solution              typed decoder under a grammar mask; emitter
```

*Zero* is a protocol, not an absence of knowledge: one candidate is committed (K = 1) before any
candidate-specific execution (E = 0), with no revision after feedback (R = 0), as recorded by an
isolated supervisor. Four tracks keep claims honest: C0 cold start, CH frozen history, SEQ sequential
control, S search at a declared budget.

`solution_embedding` replaces what earlier documents called the kernel encoding layer. It is the
abstract, protected stage: the paper specifies its contract (inputs, outputs, determinism,
decodability, no verifier access, interventions that change the decoded move) and never its internal
construction. The teacher that generates training experience is described only as *verified search*
with declared budgets, interfaces and outputs. See [EDITORIAL_DECISIONS.md](EDITORIAL_DECISIONS.md).

## What the paper must show

The draft fixes these in advance. Each is a claim in `evidence/claims.json` and each synthetic number
behind it is a gap in `northstar/gaps.json`.

1. **First proposals win** (EZ-C04 to EZ-C06; gaps EZ-G01, G03, G05). One committed proposal beats the
   strongest predeclared non-search control on sealed families in all three games, Holm-adjusted, with
   failures scored −1 and kept.
2. **One proposal is worth search** (gaps EZ-G02, G04, G06). The search-equivalent budget is tens of
   evaluations, measured on a grid and never extrapolated.
3. **It is not retrieval, a catalog or capacity** (EZ-C07; gaps EZ-G09, G10, G12). Shuffled-state,
   retrieval, catalog-bound, direct-decoder, imitation-only and Native-encoder controls.
4. **Frontier models and the amortized literature lose at one attempt** (gaps EZ-G07, G08).
5. **One shared model plays all three games** (EZ-C08; gap EZ-G11), with transfer limits reported.
6. **Fresh verified outcomes improve the next model** (EZ-C09; gaps EZ-G13, G14). This is the result
   the program has so far failed to produce (Appendix I), and the single most important gap.
7. **Some winning moves are new** (EZ-C03, EZ-C10; gaps EZ-G15 to G17). The novelty ladder N0–N6, with
   prospective dossiers, removal and replacement interventions, prior-work review and replication.
8. **It pays** (EZ-C11; gap EZ-G18). Metered costs, quality-matched break-even and its sensitivity.

## Work packages

| WP | Deliverable | Closes gaps | Exit criterion |
| --- | --- | --- | --- |
| WP0 Reconcile and rename | this draft; corrected historical record (Appendix I) | — | builds; old evidence traceable; unsupported claims removed |
| WP1 Freeze the interface | typed state, output and receipt schemas; adapters for three games | EZ-G21 (in part) | the three example task types serialize and validate without regression-specific assumptions |
| WP2 Commitment harness | isolated proposer, post-commitment verifier, supervisor counters | supports EZ-C02 | retries, hidden scoring and post-feedback replacement are detected and logged |
| WP3 Strong controls | retrieval, tree selector, catalog decoder, direct generator, standard tools, learned-optimizer and frontier arms | EZ-G02, G07, G08 | reproducible development comparisons with complete failure and cost records |
| WP4 Open synthesis, learning procedures | compositional open-support decoder; verified experience store | EZ-G01, G12 | valid out-of-archive procedures; usefulness against strong controls |
| WP5 Next-experiment adapter | frozen-history (CH) and closed-loop (SEQ) benchmarks | EZ-G03, G04 | clean CH/SEQ separation; compatible learned-optimizer baselines |
| WP6 CAD adapter and verifier | specifications, editable program decoder, geometry and physics checks | EZ-G05, G06, G19 | end-to-end valid parts and qualified independent verification on development designs |
| WP7 Outcome and latent ablations | matched study with direct-decoder, imitation, objective-intervention and improvement-round controls | EZ-G09, G10, G13, G14, G21 | evidence for or against the representation and training mechanism |
| WP8 Shared-model study | joint and separate models with matched accounting | EZ-G11 | transfer and its limits quantified |
| WP9 Lock confirmation | preregistration, rosters, final checkpoints, evaluator versions | EZ-G20 | no unresolved protocol ambiguity or test-informed selection |
| WP10 Confirmation and dossiers | complete result records; prospective dossiers | EZ-G01–G08, G15–G18 | every attempt accounted for; claims match the measured outcome |
| WP11 Reproduce and write | replication package; physical tests; final text | EZ-G17, G18 | central claims trace to reproducible evidence; title and abstract match scope |

Start small. The first decision gate is cheap and decisive: **on development families for learning
procedures, does one EurekaZero proposal at least match retrieval (A3) and the catalog-bound decoder
(A2)?** If not, stop and diagnose — encoder information loss, teacher noise or decoder support —
before building the other two games. Three games on top of a proposer that loses to retrieval is the
expensive way to learn the same thing.

## Stop and pivot rules

- If a strong selector matches the generative model, test whether synthesis adds value before scaling it.
- If the solution embedding does not help, keep the framework and drop the claim that it is the essential ingredient.
- If valid outputs are rare, improve the language and the experience before adding games.
- If gains vanish when failures are included, say so and redesign the objective prospectively.
- If the shared model shows negative transfer, publish separate models and do not call it general.
- If independent CAD verification reverses a result, revise the verifier and reclassify the result as development evidence.

These change the claim or the next registered study. They never alter a completed primary.

## How to replace a synthetic result

Each gap is a contract. To close one:

1. **Claim the work** under the repository's agent contract (AGENTS.md): lock, claim, one lane.
2. **Run the measurement the gap's acceptance test describes** — on the declared families, with the
   declared comparator, budget and evaluator identity — and keep its receipt (source commit, command,
   exit status, inputs, resource use).
3. **Replace the synthetic value** in the manuscript with the measured one: remove the `\syn` wrapper,
   add the `\est` or `\pre` badge the evidence supports, and cite the receipt. Replace the synthetic CSV
   behind a figure with a measured CSV outside `data/synthetic/`, and remove that figure's
   `\synfigtag`.
4. **Delete the gap** from `northstar/gaps.json` and the generator's corresponding output, and update
   the claim's status in `evidence/claims.json`. If the measurement contradicts the draft, the text
   changes to match the measurement; the draft is a target, not a prediction we are entitled to keep.
5. **Run** `make -C docs/paper/eurekazero check self-test pdf`.

The paper is ready to submit when `make submission-check` passes: no `\syn`, no `\synbadge`, no
`\synfigtag`, no synthetic data input, and `\EZsyntheticfalse` in `main.tex`.

## Objections we must answer

| Objection | Answer the paper needs |
| --- | --- |
| "This is ordinary amortized optimization." | The innovation-game formulation, three output roles under one protocol, and wins over the closest pretrained optimizers |
| "The encoder–latent–decoder structure already exists." | Cite the Latent Programmer and its relatives; isolate what the solution embedding adds (EZ-G09) |
| "One inference hides a large search." | Supervisor counters, decoder steps, every surrogate or critic call, full compute |
| "It only chooses a stored artifact." | Catalog and retrieval controls; out-of-archive outputs with provenance checks |
| "Archive novelty is not novelty." | Agreed; that is why N5 and N6 exist |
| "Three models are not one model." | The matched joint-versus-separate study |
| "A next query is not an invention." | Separate action quality from artifact or mechanism novelty |
| "The CAD results are renderings." | Editable programs, independent simulation, physical tests |
| "Failures were dropped." | Failures score −1 in every denominator; tails shown |
| "Training cost dwarfs the savings." | Quality-matched break-even with its deployment volume and sensitivity |
| "The dossier was cherry-picked." | Prospective lowest-hash selection rule; every opportunity reported |
| "The old null results were omitted." | Appendix I keeps them, with their scope |
| "The title promises all innovation." | It does not: three games are tested and EZ-C12 is never asserted |

## Venue shape

The draft is the long version. A nine-page conference cut keeps Sections 1–3 (game, protocol), a
compressed Section 4, the primary results (Table 10, Figures 5 and 6), the mechanism ablations
(Figure 8) and one dossier, and moves training details, compute, the full outcomes and the dossiers
to the appendix. The north-star figures are designed so that Figure 1 (innovation as a game) and
Figure 3 (the network) carry the idea on their own.
