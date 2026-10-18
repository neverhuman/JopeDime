# Prospective policy discovery qualification v1

**Status: specification only; not adopted and not execution authority.** This document
defines the fields and analysis rules for a future preregistration. It does not admit a
campaign, authenticate a roster, release labels, authorize evaluation, issue a seal, change
F10, or permit promotion or serving. D05 and D26 remain `HOLD_SCIENTIFIC_ADOPTION`.

The machine-readable proposal is
[`jope.policy-arena.v2.schema.json`](../../schemas/jope.policy-arena.v2.schema.json).
The schema describes a proposed manifest. A schema-valid manifest is not evidence that its
issuer, inputs, roster, or authorization are genuine.

## Estimand frozen by this protocol

The primary estimand is the equal-family mean of paired **policy-discovery linear-budget
q-AUC differences** between a preregistered challenger actor and its accepted incumbent
actor. A unit is an independent family. Within a family, first average its preregistered
context-by-seed cells with equal cell weight; then average family effects with equal family
weight. Unequal numbers of seeds or contexts across families therefore do not change the
weight of a family. The manifest records every family, context, seed, metric identity,
resource horizon, direction, and practical margin before any outer outcome is available.

For each cell, record the cumulative resource clock and the best eligible policy quality
`q` observed by that clock. `q` is a higher-is-better, finite scalar on the metric's declared
scale. The clock is cumulative CPU seconds charged to the same declared resource ledger for
both actors. Normalize the clock to `[0, 1]` using the preregistered common horizon, include
the preregistered baseline value at zero, and compute trapezoidal area over the complete
clock from zero through one. No extrapolation, per-actor stopping horizon, or endpoint
substitution is allowed. The cell effect is challenger q-AUC minus incumbent q-AUC. The
primary estimate is the arithmetic mean of those cell effects within each family followed
by the arithmetic mean across all preregistered families.

The primary direction is positive. The practical margin is a finite, positive value on the
q-AUC difference scale and must be fixed in the manifest before outcomes. A result is
practically positive only when its declared uncertainty lower bound exceeds that margin.
The q metric definition, units, version, resource ledger, horizon, uncertainty method,
confidence level, minimum family count, and minimum detectable effect/power inputs are
required manifest fields; this specification does not invent their owner-approved values.
They remain unresolved until an authorized issuer fills and adopts them prospectively.

The endpoint is intentionally distinct from terminal regularized log-MSE and selected
artifact log-RMSE AUC on a log-budget axis. Those are named diagnostics only. They cannot
replace the primary endpoint after outcomes are observed. This proposed q-AUC has no
calculator in the current public `promotion::decide` API. The protocol must not duplicate
its private bootstrap or quantile helpers, claim F10 equivalence, or use an incompatible
F10 report as a discovery verdict. Compatibility can be claimed only for a future adapter
that calls the public `promotion::decide` and proves exact metric, units, family,
resampling, and quantile compatibility. Otherwise the discovery estimate remains
unsupported by the current production evaluator and adoption stays held.

## Frozen comparison and controls

The manifest binds the exact accepted parent actor, challenger actor, artifact and build
identities, task and metric identities, initialization and ancestry, and a single primary
comparison. It separately records these contrasts:

1. **Program delta:** quality attributable to the selected program artifacts under the
   frozen evaluator. This is descriptive and does not establish an actor effect.
2. **Critic allocation:** quality under a control that changes critic allocation while
   holding the actor update budget, task mixture, initialization, and clock fixed.
3. **Actor policy delta:** challenger actor versus incumbent actor under the same update
   opportunity, initialization lineage, task mixture, and resource clock.
4. **Fresh-feedback contrast:** the challenger actor trained with newly admitted feedback
   versus a matched-update control that receives no fresh feedback. Optimizer steps,
   decay-only changes, or reused feedback do not count as fresh feedback.

The strongest simple baseline and a shuffled-context control are preregistered where their
support is meaningful. The no-fresh-feedback control is mandatory for a causal fresh-feedback
claim. Matching rules cover update count, compute, task/context support, initialization,
ancestry, and exposure. A mismatch or unsupported control is retained as a failure or
inconclusive outcome under the fixed rule; it is never silently dropped or credited as a
positive contrast.

## Families, failures, and missing contexts

The family roster, nested contexts, and seeds are fixed before outcomes. Each expected
family/context/seed cell has exactly one terminal disposition: observed, failed, or missing.
The complete expected-cell roster is the denominator. Failed and missing cells remain in
the record and are never imputed as zero, replaced, or removed to improve the estimate.
If any primary cell is failed or missing, the primary result is `inconclusive`; the report
still gives the full preregistered family and cell denominators and the observed subset as
clearly labelled descriptive information. There is no available-case primary mean and no
family deletion. This rule is specific to this proposed protocol and does not alter F10's
existing incomplete-family rule.

Families are the independent resampling and confirmation units. Contexts and seeds are
nested measurements, not additional independent families. The uncertainty procedure and
its seed, resample count, interval, and family-level input are pinned in the manifest before
outcomes. It must preserve the equal-family estimand. No context-weighted pool bootstrap or
post-outcome change of calculator is permitted. A new implementation may not access private
helpers in `promotion.rs`; until an authorized compatible public evaluator exists, no
confirmatory verdict is produced by production code.

## Multiplicity, looks, and untouched confirmation

One primary endpoint, one primary challenger/incumbent contrast, and one final efficacy
look are declared. Candidate development and budget selection use development evidence
only. Any selection among multiple candidates, budgets, hyperparameters, or interim looks
must be declared before outcomes and handled by either a fully specified alpha-spending
plan or an untouched outer-family holdout. The preferred confirmation route is an
independent outer-family holdout: its family IDs and exposure boundary are authenticated
before access, and no outer-family outcome may select a candidate, budget, metric, subgroup,
or analysis. Once used, the outer family cannot be reused for another confirmatory look.
If the selected multiplicity method, alpha allocation, or untouched roster is absent or
unverifiable, the result is inconclusive and no efficacy claim is made.

Nonprimary diagnostics are named before outcome release and reported without changing the
primary verdict. There is no best-of-B rescue, repeated peeking, unregistered subgroup,
post-hoc candidate reselection, or switching between q-AUC and terminal/log-budget metrics.

## Authority and independent evidence boundary

A future manifest must bind hashes for the accepted parent, candidate artifacts, source and
build identities, evaluator and metric definition, resource ledger, expected-cell roster,
exposure audit, raw results, and analysis implementation. A separate authorized issuer must
sign the exact manifest digest and identify the adopted ADR/roster, alpha/look allocation,
and permitted data boundary. The issuer must be independent of candidate training and
evaluation. A name, JSON Boolean, self-supplied digest, or schema-valid signature-shaped
string is not authentication.

The test fixtures in this change are synthetic contract checks. They establish no real
family independence, label isolation, power sufficiency, optimizer update, fresh feedback,
artifact quality, causal benefit, or scientific adoption. The current issuer, adopted
roster, endpoint margin, power sufficiency, and execution grant are `UNKNOWN`/`HELD`.

## Relationship to existing qualification

The accepted F10 rule and current `ArenaManifest`/`promotion::decide` behavior remain
unchanged. In particular, this proposal does not create an F10 alias, a new roster role,
an endpoint in the production gate, a seal, a promotion path, or a serving path. Existing
roster and promotion requests continue to refuse while their owner rulings are pending.
Any future adoption requires a separate authorized ADR and independently reviewed
implementation that consumes authenticated evidence. Until then, this document and schema
are preregistration design inputs only.
