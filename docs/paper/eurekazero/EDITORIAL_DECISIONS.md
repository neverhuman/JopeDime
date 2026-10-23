# Editorial decisions for the EurekaZero north-star draft

This draft turns the supplied EurekaZero paper specification into a complete manuscript whose results
are synthetic placeholders. No model was trained, no campaign was run and no discovery was made to
prepare it.

## Source

| Supplied planning document | SHA-256 of supplied bytes |
| --- | --- |
| `EurekaZero_Paper_Specification.md` (prepared 23 October 2026 against repository snapshot `f9543b85b356675d377e38d75c4e3ecea8b2d302`) | `66356a25fd6c42a06ebf4d7ac4dc67345b2ad78b3cace8e002472cda28e3c7c2` |

Its repository facts were rechecked against the snapshot; corrections are recorded under
*Historical record* below and in Appendix I.

## Decisions adopted

- **Name and title.** The model and framework are *EurekaZero*; the title is *Learning to Innovate
  in a Single Proposal: EurekaZero*, the specification's first-ranked title for the eventual
  empirical paper. "Proposal" is used rather than "inference" or "forward pass" because the decoder is
  autoregressive. Prior uses of "Eureka" (reward design, multimodal reasoning, an industrial
  discovery-machine vision) are cited in Section 11 so that no affiliation or priority is implied.
  This is not a trademark clearance.
- **Zero is a protocol.** Defined at first use as K = 1, E = 0, R = 0 under supervisor counters. The
  draft never says "no priors", "without human knowledge" or "from scratch": the state encoding uses
  pretrained components and the solution languages are designed by people.
- **Innovation as a game.** The specification advised using "game" formally only where the structure
  justifies it. The draft does exactly that: Definition 1 gives state, move, rules, verifier and
  opponent (the incumbent), Figure 1 maps them for Go and the three games, and the text
  says plainly that each game is single-player against a rising bar, closer to a contextual bandit
  than to Go. "Self-play" is reserved for that rising bar. The innovation-game hypothesis is stated
  as a hypothesis, its four conditions are explicit, and Section 12 lists what is not a game.
- **`solution_embedding` replaces the kernel encoding layer.** The interface is
  `game_state_encoding -> solution_embedding -> solution`. The solution embedding is the single
  protected stage: Figure 3 draws it as a dark box, Section 4.2 and Appendix B state its contract and
  four tested guarantees, and nothing describes how it is built internally.
- **The teacher is "verified search".** Training experience comes from a verified search at declared
  budgets. Its internals are protected (Appendix J); its budgets, interfaces, outputs and costs are
  disclosed. The draft never says how candidate programs are found, and `scripts/check.sh` enforces
  the repository's restricted vocabulary (the same pattern as `deploy/ci/ip-scrub.sh`) on every
  manuscript file, ledger and Markdown file in this directory. Third-party titles in the bibliography
  are printed as published. Exactly one printed title contains a restricted word: AutoML-Zero, the
  closest prior "Zero" system for discovering learning algorithms, kept because omitting it would be
  a scholarly red flag (owner decision point). The CMA-ES baseline is cited through its reference
  implementation, whose title does not expand the acronym. Two prior-work entries whose titles
  describe the protected method class directly are not cited.
- **Three games, three output roles.** Learning procedures (artifact, C0), next-experiment decisions
  (action, CH and SEQ) and engineering parts (artifact, C0). The closed-loop result is reported as
  sequential optimization at a budget, never as "one inference".
- **Failure-inclusive utility.** `U = clip(log(b/e), -1, 1)` with failures at −1 and in every
  denominator; feasibility first, then quality among feasible moves.
- **Comparators.** The primary comparator in each game is the strongest non-search control named on
  development data before confirmation. TabPFN and other amortized models are performance references
  with different output objects, not forced into the primary. Frontier models get the byte-identical
  packet, no tools and one attempt.
- **Novelty ladder.** N0–N6; N4 is a statement about a frozen archive, and only N5/N6 dossiers with
  prospective selection, interventions, prior-work review and replication may support a discovery
  claim. Dossier mechanisms in this draft are illustrative placeholders.
- **No universal claim.** EZ-C12, "the approach generalizes to all innovation", is *do not assert*.
  The north star is stated as the research program, not as a result.

## Synthetic results

- Every synthetic number is `\syn[EZ-Gnn]{...}` (purple, hyperlinked to the gap ledger); every
  figure plotting synthetic data carries `\synfigtag`; every page carries a footer saying the purple
  results are synthetic, and a light watermark.
- Synthetic values are internally consistent: the generator derives table rows, macros and figure
  data from the same constants (for example, validity rates equal one minus failure rates; the
  novelty-ladder counts drive the percentages; break-even follows from the cost curve).
- Synthetic magnitudes are deliberately moderate and include unfavourable results: non-inferiority of
  the Native encoder is not established, positive transfer is not established, a small CAD transfer
  cost cannot be excluded, and search overtakes one proposal at large budgets.
- `make submission-check` refuses the draft. It can pass only when no synthetic element remains and
  `\EZsyntheticfalse` is set.

## Historical record (corrections applied)

- The policy context is 2,048 values: a 512-value native block, a 1,024-value foundation encoding
  derived from a third-party pretrained tabular regressor, and a 512-value target encoding. The native
  block comes from a dimension-128 checkpoint with four pooled statistics, labelled with a later
  version string only to satisfy the loader; the InnovationZero paper's description of it as the later
  model is corrected. A second tabular foundation model used elsewhere is not in the state encoding.
- The third-party weights are provenance-pending with redistribution prohibited, hence the
  EurekaZero-Native arm.
- Unfavourable results are kept: policy-seeded search lost to retrieval seeding on 15 of 17 contexts
  at unequal budgets; continued fine-tunes raised regression alerts; the outcome head ranked fresh
  programs at chance; in the matched 36-model study, 31 of 36 monitor objectives worsened between
  updates 100 and 400 and ranking lift was not established.
- The Golden45 oracle envelope (Elo 2259.8) is never presented as a result.

## Not done in this PR

- No change to CI workflows. The draft is checked by its own `make` targets; wiring it into
  `.github/workflows/paper.yml` is a separate trust-surface change.
- No change outside `docs/paper/eurekazero/`.
