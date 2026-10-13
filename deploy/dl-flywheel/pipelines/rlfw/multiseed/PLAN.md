# J22: multi-seed baseline and separately assessed teachers

Code and planning only. This package launches no training, runs no evaluator,
and supplies no qualified reward or promotion decision. J22's operational gap
remains open until an owner-approved experiment has authenticated inputs,
actual optimizer-update receipts and qualified paired L6 evaluation. The
current training freeze remains in effect for this proposal.

The engineering spec G12 identifies two issues: v1 has one seed, and teachers
selected by maximising the same fold scores inherit selection optimism. These
require distinct checks. Reusing one v1 checkpoint with multiple sampling seeds
does not create independent training seeds. Subtracting an assumed noise bound
from a selected score does not create a verified teacher.

## Multi-seed proposal, pending owner approval

Propose three independently trained reproductions of the frozen **v1 recipe**,
with seed IDs 20261031, 20261032 and 20261033. These are proposed IDs, not runs.
Do not substitute three fine-tunes of the same v1 checkpoint or choose the
best-performing reproduction. A reproducible historical v1 recipe, its input
closure, initialization and initial optimizer-state protocol must be recovered
and pinned first. Their digests, update budget, resource limits and deadline
are **UNPINNED**; compute demand, power and between-seed variability are
**UNMEASURED**. The frozen comparison checkpoint is the existing v1 parameter
digest `ea078456690b0665592652b2ed5b04a0deb39b9e604de97a513dc885e467f9fb`.

Before any launch, preregister one equal update budget, a complete context and
parent-lineage roster, all seeds, deterministic failure handling, and exactly
one final checkpoint look. Partition at parent-lineage level; keep evaluation
parents out of all learning inputs. Previously inspected validation families
remain development evidence. They do not become untouched confirmation by
reshuffling folds. No sealed meta-test data enters this package or training.

Primary estimand: average over **all** preregistered seeds and contexts of
`seed_model_l6_cost - frozen_v1_l6_cost`. L6 is a loss; negative is better.
Report every seed and its sample standard deviation separately. The lowest
seed is descriptive only. Keep an additional parent-balanced sensitivity
estimate separate from the context-macro primary result.

`Study` fixes seed/context/recipe identities and an update budget.
`multiseed_report` refuses missing cells, duplicate cells, comparison or budget
drift, a changing reference, reused model/receipt digests, and relabelling the
reference checkpoint as an independent seed. Failure of one seed leaves the
panel incomplete; do not drop it or impute zero. Callers must authenticate
checkpoint and receipt bodies and verify the recipe and actual optimizer
updates. This library checks declarations and cannot establish those facts.

Its descriptive 95% interval resamples seeds and parent lineages independently;
one seed draw is shared across all parents and each drawn parent keeps all
its sibling contexts. This preserves the crossed design rather than counting
seed-by-parent cells as independent. An exhaustive toy verifies that a common
seed effect has variance proportional to 1/3, not 1/6, for three seeds and two
parents. This follows the design principle of the
[pigeonhole bootstrap](https://arxiv.org/abs/0712.1111), but is no finite-sample
coverage guarantee. Three seeds alone cannot establish power or a confirmatory
improvement claim. Family uncertainty, multiplicity and repeated looks require
the owner's separate preregistered qualification policy.

## Teacher selection without reusing selection scores as rewards

Commit `Selection` before exposing judge scores. It records all distinct
candidate program digests, inner scores and measurement digests, all fitting
rows (including normalizer/transform fitting), and all selection-exposed rows.
It selects only on inner scores with a stable digest tie-break. Include every
candidate tested, rather than only the survivor.

Fit on the declared inner data, select there, freeze the selected program and
fit artifacts, then assess that exact program and the incumbent on separate
judge rows under the same comparison protocol. `score_selected_teacher`
refuses changed commitments, programs, comparisons or fitting scopes, and any
judge row exposed to fitting or selection. It reports held-out advantage and
observed inner-minus-held-out optimism. It never promotes an inner score to a
reward. Hashes and row declarations do not authenticate bodies, preprocessing,
independent lineages, or comparable q; N-1 score-body closure and the production
HWM comparison contract remain caller requirements.

The exact two-candidate null toy has independent +/-0.1 inner noise and zero
true utility. Mean selected inner score is 0.05; independent held-out score
has mean zero. This is an analytic fixture, not measured GP performance or a
universal unbiasedness guarantee. Separate selection and assessment reflect
[Varma and Simon's nested-validation result](https://pmc.ncbi.nlm.nih.gov/articles/PMC1397873/).
Outer-test and sealed scoring are not performed here. A held-out teacher score
is learning-side evidence only if its role is authorized; untouched confirmation
scores must never feed teacher construction or training.

`optimism_bound(N, sigma=...)` exposes the spec's `sigma * sqrt(2 * log(N))`
diagnostic under a declared zero-mean sub-Gaussian noise assumption. Unknown
sigma produces UNMEASURED, not zero. Supplied sigma remains hypothetical,
distinct from measured SD or SE. No numeric reward correction is returned.

## Acceptance and remaining work

Run `python3 deploy/dl-flywheel/pipelines/rlfw/multiseed/run_multiseed_offline.py
--receipt <new-file.json>` from the repository root. The runner requires the
exact registered unittest identities, executes synthetic CPU fixtures, and
records source hashes, collection/execution/failure/skip counts and exit status.
No cloud, GPU, model loading, GP execution or optimizer updates occur.

Implemented: complete-panel validation, all-seed reporting, crossed descriptive
uncertainty, immutable inner-only teacher selection, held-out row isolation,
and transparent optimism diagnostics. Pending: authenticated v1 recipe and
artifacts, approved multi-seed training budget, production data-role and score
body verification, verified held-out teacher integration, and qualified paired
evaluation. Passing fixtures close the library proof only.
