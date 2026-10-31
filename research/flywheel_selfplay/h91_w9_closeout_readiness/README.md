# H91 — W9 causal-closeout readiness

**CPU preparation complete; W9 execution remains unadmitted.** H75's registered
factorial supplies part of the W9 design, but its fixed tree and two neural
memory replays do not cover the full W9 causal claim. This report supplies an
[acceptance matrix](acceptance.json) and conditional cost calculation. It does
not amend H75, select a development winner, or launch another campaign.

Owner `overnight19_t0135.md` directed the next unowned wave. W5's recorded
provenance preflight remains held; W7 has an existing assignment and upstream
dependencies. This disjoint W9 preparation preserves the H68 skeleton, H75
runner/support ownership, and all original protocols.

## Contract comparison

Inspected repository source: `cd4505314844a03ebc52b0935d477ca15594b944`.
The [W9 specification](../rd_breakthrough/PROPOSALS.md#w9--causal-flywheel-closeout)
and [H68 design](../h68_w9_causal_closeout/README.md) require both directions:
better guidance must acquire better data, and those data must improve a common
learner at matched cost. A stronger search oracle alone is insufficient.

| Requirement | H75 design supplies | W9 work still required |
| --- | --- | --- |
| Crossed intervention | A/B/C/D frozen/updated ranker × static/fresh memory; two cycles | Run-bound isolation, actual optimizer/parameter changes and complete cycle receipts |
| Strongest non-neural comparator | Fixed program412 tree G | Pin the selected strongest W1 policy/model and recipe; fixed G is not a substitute |
| Uniform data control | R in held-context terminal search | Uniform **fit-family acquisition memory**; never replay R's held-context outcomes |
| Common-learner replay | B and D memories into identical reset neural learners | Every A/B/C/D memory, including shadow logs, plus uniform memory through neural and strongest W1 tree learners |
| Causal inference | Seven H75 contrasts, including replay(D)−replay(B) | Uniform-memory comparisons, complete finite contrast set and adjustment frozen before W9 outcomes |
| Endpoint identity | Legacy transfer and confirmed U8 after four anchors | Preserve cycle-one U4, cycle increments and terminal identities; U8 is not U4 |

The [H75 preregistration](../h75_closed_loop_evolution/PREREG.md) deliberately
calls G fixed, rather than strongest. Its B/D neural replay remains a valid
planned narrower comparison. This audit does not retroactively change H75's
seven-contrast gate or deny its intended conditional development claim.

Under W9's full replay requirement there are five memory sources × two learner
classes = ten replay cells per fold/seed. H75 declares two of those cells;
eight are uncovered by that design. H75's status-stratified B/D matching must
be extended prospectively across the actual memory panel, with discarded rows
and effective exposures reported. No common support means untested. Match
updates within the neural class and the frozen training/cost recipe within the
tree class; tree fits are not neural optimizer steps or native candidate fits.

A small **synthetic algebra counterexample** illustrates why interaction alone
cannot pass the causal gate: A=0, B=1, C=−1, D=.5 gives interaction
`(D−C)−(B−A)=+.5` while D−B=−.5. These numbers are not study outcomes. Both
one-sided comparisons, the strongest baseline and common-learner data-value
comparisons still need their registered evidence.

## Budget reconciliation before an amendment

H75 already reserves the W9 envelope: **9,600 native calls, 14,400 GPU-seconds,
14,400 aggregate CPU-seconds**, deadline **2026-10-31T14:00:00Z**. This is one
reservation. Actual cumulative consumption was not audited here, so remaining
authorized resources are unknown, not reset to those ceilings.

The existing plan totals 7,200 native calls and 3,600 neural updates. Retaining
its exact five-fold/three-seed/four-context/two-cycle roster and 40-step replay
recipe yields this conditional calculation:

| Item | Conditional count |
| --- | ---: |
| Additional uniform fit-family acquisition: 5×3×2×4×4 | 480 native calls |
| Existing plan plus that acquisition | 7,680 native calls |
| Difference from 9,600 ceiling, before any new requirements or actual consumption | 1,920 calls |
| Updating C/D plus five neural memory replays per fold/seed | 5,400 optimizer updates |
| Five tree memory replays per fold/seed | 75 tree fits |

This assumes cached common-panel replay evaluation and no extra native calls
beyond the existing pilot, terminal search and confirmation allowances.
Additional tree training and neural updates still need timed CPU/GPU estimates.
The arithmetic is **not a feasible execution plan**, a new allocation, or a
license to spend 1,920 calls. Reconcile prior consumption and failures, freeze
all additional contrasts and the exact roster, then establish feasibility
inside the original envelope and deadline before any amendment or launch.

## Implementation and admission handoff

[acceptance.json](acceptance.json) records eleven obligations with source
locations and explicit statuses. No obligation is represented as an executed
W9 pass. The complete input/row/status admission, W1-selected recipe, exposure
and update receipts, common-learner integration, cost ledger and launch manifest
remain separate requirements. Unknown outcomes are never zero-filled.

The [H75 interfaces](../h75_closed_loop_evolution/INTERFACES.md) and
[scoring helpers](../h75_closed_loop_evolution/scoring/README.md) are reusable
support, not a qualified loop. At this audit's API readback,
[runner PR859](https://github.com/neverhuman/JopeDime/pull/859) was closed
unmerged at `d1a785412e7d8acdcfcfbc30a87fd225398db152`; the inspected H75 main
directory contained no `runner.py`. This does not assert that no runner source
exists elsewhere. OPUS retains integration ownership; H91 adds no runner.

Freeze exact source/evaluator/vocabulary, model, family/table/row/cut/program
identities and field-allowlisted completion provenance. Preserve training-only
closure and frozen DOPE. Pin the full policy/seed/fold/context roster before
scoring, report all failures and exclusions, and keep panels' Elo scales separate.
Report family Elo and raw utility/val_reg_inner with the appropriate track;
no Elo, utility improvement or breakthrough is measured by this preparation.

## Verification and S3 evidence

Three real S3 metadata objects were fetched at exact versions and SHA-256
verified: the original H75 custody manifest and protocol, plus W5's custody
manifest. Fifteen H75 scientific fields match current source; historical queue
changes are not treated as scientific changes. No new data prefix was inventoried.

Three deterministic source-check shards ran on xbabe1 (factorial, nice19),
xbabe3 (replay, nice10), and xbabe4's Linux/aarch64 Colima `jope-ci` container
(budget, one CPU, nice10, network disabled). All passed host capacity guards;
xbabe1 also passed the 150 GiB free-space guard. The initial Mac Docker command
was unavailable on PATH; correcting PATH allowed the Linux shard to run, with
no duplicate successful work. No remote source/data files were created: inputs
were sent on stdin, and the owned container removed after exit.

**48 assertions passed:** 27 source-hash checks across nine unique files and
21 contract/counterexample checks, including the one synthetic example above.
This is not 48 independent tests or evidence of a running learner. Summed
worker CPU time was 0.002035 seconds; interpreter/import startup, Git, SSH,
S3, Docker and publication overhead were not timed and are not asserted zero.
No native fit, optimizer update, GPU operation, model or protected payload read
occurred. [metrics.json](metrics.json) retains those limits and null metrics.

Create-only custody manifest:

`s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/W9/astra-h91-readiness-20261031T074030Z-b46531dc/manifest.json`

Version: `VIojyUs8NyX4i9HfrgHdozh9f.NZlK7V`

SHA-256: `6644198f6caabef16b6a1b1c2ee06a3cb2436c731ca7cbf2d0f3555cea638be3`

All four objects passed exact-version download and byte SHA-256 verification;
the restored evidence archive's twenty members were checked, with nineteen
source/receipt members matched to its inventory. It contains the exact input
packet, source hashes, checker, commands and each host's result. Checker source
is external and hash-pinned there, not a committed production validator. To
replay a shard after verifying the archive, pass `input_packet.json` on stdin to
`python3 -B worker.py --shard factorial|replay|budget --host HOST --sha256
97ef51f1aa789bddd2455f98763d31a872213a223ddc77f9c7546b209a1b16b0`
under the same host limits. Small metadata/source receipts remain in custody;
no bulk outputs or HWM candidate were produced.
