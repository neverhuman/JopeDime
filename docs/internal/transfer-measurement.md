# Private regression transfer measurement contract

Current publication: **not yet measured**. This reporting-only implementation does not
admit execution, grant a GPU, reset a deadline, change teacher admission, or change
acceptance/serving thresholds. A future numerical phase needs its own admission.

## Commands and immutable inputs

```
jope benchmark transfer-plan --input specification.json --output plan.json
jope benchmark transfer-report --plan plan.json --freeze freeze.json \
  --release release.json --release-sha256 EXTERNALLY_PINNED_SHA256 --output report.json
```

Both commands perform zero fits. Output paths must be new, are private (0600 on Unix),
and are never overwritten. Paths in the input refer to local files relative to the
invocation directory, so absolute paths are preferable for durable custody. SHA256
identities bind their bytes. Large opaque checkpoint/encoder/features assets are
stream-hashed; decoded JSON evidence is capped at 128 MiB.

The typed, unknown-field-rejecting JSON contracts live in
`crates/jope-bench/src/panel/transfer.rs`. A `jope.transfer-spec/v1` freezes:

- Round name, full source commit, procedure and encoder assets, additional assets,
  parent and challenger checkpoints, and separately named optional controls.
- Unique dataset IDs/content identities, family assignments, ordered row IDs/groups,
  outer-fold assignments, unique seeds, and distinct evaluation/checkpoint/encoder
  exposure declarations with hashed supporting evidence.
- CPU/memory limits, maximum searches, candidate cap, per-fit seconds and total fit
  seconds. These are preflight estimates, not runtime enforcement or an allowance.

Labels are a hashed JSON object mapping every ordered row ID to one finite regression
label. Planning verifies bytes without parsing label values. Feature bytes, procedure
semantics and exposure evidence are opaque to this reporter: their truth, schema and
closure must be independently checked at execution admission. Merely declaring
`untouched` is not a historical proof. The current adaptive panel's known IDs/source
hashes are refused even if relabeled untouched; re-encoding/aliasing an exposed corpus
cannot make it unseen. Unknown encoder or checkpoint exposure blocks a complete score.

## Nested procedure audit

Every dataset/seed has five outer folds. Each outer training partition has five audit
folds. Each audit fold repeats the complete procedure, including context construction,
preprocessing, candidate generation and inner selection on its own training partition.
An additional independent search selects/refits on all outer-training rows and scores
the outer fold. Thus each arm needs **30 searches per dataset/seed** before candidate
fitting. Five inner candidate fits plus one final refit per search give the declared
work estimate; search, context and I/O overhead are extra.

Groups cannot cross outer partitions. Audit and inner folds deterministically assign
lexically sorted group IDs round-robin, retaining original row order within partitions.
Seeds change executor randomness, not these partitions. Any empty scoring/inner fold is an explicit feasibility failure. This version assumes
exchangeable groups; time-ordered or other constrained evaluation requires a separately
versioned protocol. Five outer and five audit folds do not mean 25 fixed-recipe refits:
all six searches per outer fold are independent, and their winning recipes may differ.

A `jope.transfer-procedure-audit/v1` prediction binds a search identity, procedure,
checkpoint, per-search winning recipe and `jope.transfer-nested-selection/v1` receipt.
Fitting and selection label dependencies must match the permitted training rows;
inner partitions must match the frozen plan. Scoring labels never belong to those
closures. This evidence is **not** the existing fixed-program teacher-admission receipt.

## Commitment and release boundary

The freeze commits every planned search/arm as a hashed prediction or an explicit
failed/unavailable outcome with its reason. Missing slots are enumerated. Any absent
or failed pair blocks the complete headline; no partial freeze releases score subsets.
Each terminal event precedes the freeze, which precedes the first scored audit/outer result. Inner candidate-selection scores remain permitted.

A `jope.transfer-release/v1` receipt binds the exact plan and freeze, executor source,
chronology and independently verified isolation evidence. Its digest must come from
separately admitted trusted custody, **not** be recomputed from an unreviewed bundle
and treated as approval. The reporter validates that attestation and declared closure;
it does not implement an isolated executor, establish the truth of opaque provenance,
or prevent a malicious producer from lying. A JSON boolean or sequence number alone
is not isolation evidence. Future execution must commit the entire round before
releasing any scores and prevent cross-fold adaptive feedback. Released outcomes are
exposed development evidence for subsequent work.

## Descriptive KPI

For each matched dataset/outer-fold/seed unit, average five separately normalized
audit gains `(RMSE_parent - RMSE_challenger) / RMSE_training_mean`, and compute the
corresponding outer gain. Each trivial prediction uses only its own permitted training
labels. Take `min(audit_gain, outer_gain)` **per unit**, preserving negative values.
Average units within datasets, datasets within families, then families equally;
display 100 times that value in percentage points of baseline RMSE.

The report includes parent/challenger absolute skill, both component gains,
worst-family contribution and outer gain, family distributions, planned/valid coverage,
per-unit points and an embedded reproducible `scatter_svg`. Sign agreement uses three
categories: negative, zero, positive; zero agrees only with zero. A positive relative
score without positive mean absolute skill on both components displays
**no absolute skill demonstrated**. Controls remain separate loss rows.

Zero/nonfinite denominators, missing pairs, contaminated exposure or invalid identities
prevent a complete global score. Validation failures preserve a failure report; counts
from an unauthenticated plan are explicitly marked unverified. No absent case becomes
zero or silently disappears. The score is conservative and descriptive, not a confidence
bound or causal proof; folds/seeds are not independent statistical replicates.

[Cawley and Talbot](https://www.jmlr.org/papers/v11/cawley10a.html) motivate auditing
selection rather than only fitting. [Bengio and Grandvalet](https://www.jmlr.org/papers/v5/grandvalet04a.html)
explain why overlapping folds invalidate naive uncertainty estimates.
[Dwork et al.](https://arxiv.org/abs/1506.02629) explain the risk from adaptive holdout reuse.
These papers do not validate this particular descriptive minimum score as an estimator.

## Verification scope

`cargo test -p jope-bench --lib panel::transfer` runs CPU arithmetic, receipt,
partition and deterministic toy selection fixtures. The toy selection test changes
only each search's own scoring labels and checks unchanged selected recipe/predictions.
It verifies the fixture's input boundary, not a qualified production search executor.
No synthetic fixture is a historical result or a numerical campaign. Historical native
512×6 and experimental Python 256×2 measurements stay separate from this unmeasured KPI.
