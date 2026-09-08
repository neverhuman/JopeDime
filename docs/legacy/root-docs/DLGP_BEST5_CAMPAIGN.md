# Best-five DL + GP regression campaign

`jope_dlgp_best5` is an isolated regression campaign. It does not read or
modify the partial `jope-strategy-screen` run, Search V3 rankings, accepted
checkpoints, production pointers, or production-learning inputs. Immutable
prediction objects may be reused only through the fully bound prediction key.

The freeze retains all 42 input-manifest datasets. The first 20 are the pilot,
including exactly nine datasets below 300 training rows; the remaining 22 are
confirmation. Dataset weights are equal. The checkpoint is exactly
`e0aef7d74afd75d3003bdb049fa0bac06653f66408c7bfc458b1aba5ae1b98cb`;
validation fails if its bytes or regression-head schema binding differ.

The six public commands form a sealed sequence. `freeze` consumes a
`JopeDlgpBest5CampaignSource.v1` (not a caller-assembled campaign), verifies
every frozen file, and assembles `JopeDlgpBest5CampaignFreeze.v1`:

The observation snapshot is a JSON envelope with `snapshot_sha256`,
`observations`, and optional `diversity_seeds`. The snapshot ID is the frozen
protocol binding carried by every observation; the enclosing artifact has a
separate raw-byte SHA-256, avoiding a self-referential file hash.

```text
jope_dlgp_best5 freeze --input campaign-source.json --output-root artifacts
jope_dlgp_best5 validate-freeze --campaign campaign.json
jope_dlgp_best5 run-search --campaign campaign.json \
  --worker-result search-bundle.json --output-root artifacts
jope_dlgp_best5 select-seeds --campaign campaign.json \
  --search-result seed101.json --search-result seed202.json \
  --output-root artifacts
jope_dlgp_best5 score-sealed --campaign campaign.json \
  --score sealed-bundle.json --output-root artifacts
jope_dlgp_best5 compare --campaign campaign.json --stage pilot \
  --scores paired-scores.json --output-root artifacts
```

`run-search` and `score-sealed` are protocol finalizers for attested SCQ worker
bundles; bare result or score JSON is rejected. Search finalization rebuilds
the row identities, atomic groups, eight search splits, and CV7 folds from the
hash-frozen training bytes, resolves every prediction receipt from the
content-addressed store, and recomputes archive and CV7 metrics. Search staging
contains training plus metadata and explicitly attests that no test artifact
was staged.

Sealed work is dataset-level. It atomically publishes one claim for the full
selection set before resolving `test.csv`, evaluates all six pilot predictors
(or confirmation winner plus control), and represents deployable winners as a
single pipeline, residual composite, or one-to-four-member simplex portfolio.
Finalization verifies the frozen test hash and publishes each score plus one
dataset receipt.

Set-like worker evidence is canonicalized before publication, so retries,
cache hits, and shuffled completion order produce the same artifact root.

The SCQ plan contains one all-42 proposal job, one control proof, a protected
24-job wave, the remaining 216 pilot identities, and—only after all pilot
selection freezes—20 dataset-level sealed jobs. Confirmation is added only for
the unchanged pilot winner and control. Submission remains blocked until
doctor, evidence freshness, reviewed acknowledgement, an unchanged frozen
project weight, and at least 2% free disk on every eligible worker all pass.
Free disk below 5% warns but does not block. The project policy is temporarily
weight 10 with one GPU available for the proposal job; CPU search jobs use four
cores, 8 GiB, 30-second candidate deadlines, two-hour job limits, and two
same-identity retries.

The internal `jope_dlgp_best5_worker` binary is SCQ-only:

```text
jope_dlgp_best5_worker propose-all ...
jope_dlgp_best5_worker search-one ...
jope_dlgp_best5_worker sealed-one ...
jope_dlgp_best5_worker emit-plan ...
```

`propose-all` publishes a deterministic job alias before any retry can be
accepted; after successful publication it loads and validates the existing
42×64 batch and never invokes GPU inference again. `emit-plan` declares the
worker, campaign, evaluator/policies, proposal or selection inputs, and exact
dataset files in each SCQ envelope, runs `scq check` for every spec, and has no
submission path. Search specs do not declare test files or test hashes.

The search worker materializes the arm schedules rather than treating them as
advisory: lineage UCB uses paired frozen-founder baselines, MAP-Elites copies
protected niche leaders and realizes its 3:1 schedule, and surrogate search
generates 320 unique semantic candidates before evaluating the exact 3/60/17
selection. Every logical evaluation, including a validated cache hit, receives
one contiguous prediction receipt.
