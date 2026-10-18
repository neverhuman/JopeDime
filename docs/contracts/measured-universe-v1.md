# Measured universe v1

`jope.measured-universe/v1` is an exact, finite table over an explicit requested support. Each
key binds context, checkpoint, policy, program, evaluator, target, profile, fidelity, role, metric,
and (when the source contract declares one) a stable replicate ID. Joins require equality of the
whole key. A mismatched evaluator, execution profile, role, fidelity, or checkpoint remains
unjoined and is reported as a refusal or missing requested cell.

The table accepts immutable source views only through `UniverseEvidenceVerifier`. Production
implementations must authenticate the source receipt, full observation closure, lineage and
command/input provenance, plus the revocation snapshot. Complete values must also carry I02's
sealed `VerifiedMeasuredOutcome`; raw JSON, a `measured` boolean, a filename, or a digest-shaped
string cannot create measured authority. The verifier implementation itself must be rebound to
the accepted I27/I28 interfaces at integration.

Unavailable outcomes stay in the requested denominator and retain their source locators. Failed,
timed out/censored, rejected, unknown, conflicting, and missing cells remain separate. Two rows
with the same complete key and same status/value preserve both locators. Different status or exact
value for an equal key quarantines the whole cell; arrival order does not choose a winner and the
implementation never averages a conflict. Replicates remain separate only when they have distinct
stable source-declared replicate IDs.

Coverage is reported for every context/checkpoint/policy group with its requested cardinality,
observed and complete counts, each unavailable category, conflicts, missing support, and an
explicit full/partial flag. No value is imputed, predicted, or inferred outside the requested
support. Probability mass is `UNKNOWN` unless an adopted typed frozen-law DTO is supplied and
authenticated; this version does not infer mass from row counts. In particular, a complete table
over the requested finite support makes no claim about a wider deployment distribution.

The checked-in fixture is synthetic development evidence only. It has five requested keys across
three context/checkpoint groups, four observed outcomes, exact hand-checked values `0.75`, `0.25`,
and `0.5`, one explicit unknown outcome, and one missing key. It does not establish live corpus
completeness, scientific qualification, serving authority, or I03/I09/I27/I28 acceptance.

The canonical JSON schema and root module/export, coverage/inventory, and README registrations
are integrator-owned. No shared schema registry or legacy source adapter is changed by this worker.
