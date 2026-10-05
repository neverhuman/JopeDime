# S1 regression teacher protocol

The current source implements the October 5 owner-granted S1 challenger.
The historical record below describes the frozen September 29 binary, not the
new source or a new qualified campaign. Original hashes and records remain in
`deploy/lambda/SNAPSHOT.json`, Git and custody; changed tree hashes are recorded
separately with the owner grant as their reason.

S1 uses `jope.s1-request/v1`, `jope.s1-evaluation/v1`, and
`jope.island-scored-evaluation/s1-v1`. Requests declare
`input_semantics=unscaled_fit_rows_v1`; JTT1 values must come from the admitted
unscaled input producer. The handler rejects the old request and storage prefix.
Each fold fits min/max feature and target scaling on its own fitting rows.
Scoring applies those constants and returns predictions in original target units.
The official-test fit uses all outer-train rows. Preprocessing receipt hashes
and model hashes remain separate; these score records are not exported models.

The eight islands contain eight individuals each (64 total), matching the
research campaign rather than the imported prototype's 896-member population.
At most 96 G0 evaluations seed the population; if they cannot supply 64 eligible
unique seeds, record an incomplete context rather than buying extra evaluations.
Migration is every three 64-completion equivalents. These engine counters describe
completed work; **the durable attempt ledger controls the resource ceiling**.

Fitness is `0.80*mean(log(CV ratio)) + 0.10*tail + 0.10*sd + 0.0005*complexity`
after the retained CV gate. Failed-gate fitness keeps its declared CV penalty.
CV-gated/CV-stable comparators, novelty, lexicase and semantic deduplication read
only CV/OOF products. The semantic hash contains OOF predictions only; the
64-value signature projects OOF residuals only; all 192 case indices are OOF.
Official-test metrics and the status test leader are diagnostics, never inputs
to selection or stopping. `selection_protocol=jope.s1-inner-cv-selection/v1`
is required for semantic cache and scored resume records. Missing/legacy tags
refuse before their old test-derived fitness or signatures can be used. The
trusted controller must also bind actual input/source bytes; this tag by itself
is not provenance authentication.

`s1_budget.py` carries channel 4 into tracked code: all contexts continue through
18 fixed equal generations without consulting HWM/test fields. A frozen plan
pins all 4,873 context dispositions, worker, engine and input manifest. Slots
0..95 are G0; slots 96..1247 are eighteen blocks of 64 evolution attempts.
Reservations are atomic, persistent and never refunded; single-use dispatch
claims refuse duplicate/unknown replay. Failed, cancelled, lost and retried
attempts count inside the **6,081,504** ceiling. A retry needs a free slot; there
is no extra allowance. The original deadline is October 10 at 00:00Z.
The receiving supervisor must integrate reservation and dispatch claims before
any real evaluation; the supplied CLI performs offline coordination only.

The owner typed receipt is
`2ca6a40c3128c07024c835d3a0b5cbf325e4df810242d13307db3a7451c136ff`.
This source change does not launch a campaign or qualify G22. Historical
`t1_split`, coreset/kmeans exposure and rights-held contexts remain visible;
no result becomes fully train-only through the new transforms. Deployment,
GPU/S3 publication, teacher admission and optimizer learning require their
separate receiving evidence.

## Historical September 29 protocol — retained record

# Semantic island campaign schema

This directory is the complete source closure for `airfoil-r0f0-islands-v2-20260929-v1`.
The Lambda evaluator remains the qualified Rust binary identified by evaluation context
`a8b2aaaf59a8b893915975c0a98601717f809086b94b80949ded366fa2e6e412`. The Rust island engine owns
fitness calculation, semantic fingerprints, populations, parent selection, migration, and replay.
Python only transports immutable requests and results, records exact timestamps, and enforces the
carried `$1,000` budget ledger.

## Versioned records

| Schema | Purpose | Compatibility rule |
|---|---|---|
| `jope.test-adaptive-evaluation/v1` | Rust worker's seven inner-fold fits and official outer-test fit | Frozen by the evaluation-context hash |
| `jope.teacher-semantic-evaluation/v1` | Compact control row with exact prediction hash, 64-value signature, and 192 sampled case errors | Rebuild only from immutable v1 evaluation packs |
| `jope.island-campaign/v1` | Population, policy, gate, migration, cost, worker, and evidence closure | A changed policy or hash algorithm requires a new campaign ID |
| `jope.island-scored-evaluation/v1` | One terminal proposed program, island identity, raw objectives, semantic identity, lineage, selection result, timing, and AWS report | Append-only; one row per proposal hash in a campaign |
| `jope.island-engine-status/v1` | Current leaders, population invariants, root diversity, and island progress | Mutable local status; immutable when embedded in a state object |
| `jope.island-engine-snapshot/v1` | Population hashes, archive tails, and exact counters | Diagnostic/restart evidence; scored evaluation JSONL remains the replay source |
| `jope.island-campaign-state/v1` | Engine snapshot plus budget and throughput status | Immutable S3 generation checkpoint |

## Identity hierarchy

The hierarchy is `dataset source hash -> official split metadata hash -> row membership hash ->
encoding hash -> data-context hash -> worker binary hash -> evaluation-context hash`. A fold number
alone is never an identity. The evaluation-context hash is the namespace for exact program claims and
results. Adding folds creates new membership and evaluation contexts without colliding with existing
work.

`program_sha256 = SHA256(canonical GP DSL UTF-8 bytes)`. The global conditional claim at
`contexts/<evaluation-context>/claims/<first-two>/<program-sha256>.json` prevents the same syntax from
ever being fit twice in that context, across campaigns and hosts.

`full_prediction_sha256 = SHA256(little-endian f64(inner-OOF predictions in fold order followed by
official-test predictions))`. This exact behavior hash collapses syntactically different trees that
make identical predictions. Active island membership is globally unique by both program hash and full
prediction hash.

The 64-value semantic signature is a deterministic Rademacher projection of normalized signed OOF and
test residuals. The 192 case vector contains 128 deterministic OOF and 64 deterministic test absolute
errors for epsilon lexicase selection. The projection bytes, sample manifest, and control semantic
index are content-addressed S3 objects referenced by the campaign manifest.

## S3 layout

```text
experiments/test-adaptive-teacher/v1/
  contexts/<evaluation-context>/
    context.json
    claims/<hh>/<program-sha256>.json
    evaluations/<hh>/<program-sha256>.json.gz
  attempts/<attempt-id>.json.gz
  objects/
    attempt-receipts/sha256/<sha>.json
    semantic-indexes/sha256/<sha>.jsonl.gz
    semantic-projections/sha256/<sha>.f32
    semantic-samples/sha256/<sha>.json
    island-source-manifests/sha256/<sha>.json
    island-orchestrators/sha256/<sha>.json
    island-qualifications/sha256/<sha>.json
    concurrency-receipts/sha256/<sha>.json
  campaigns/<campaign-id>/
    manifest.json
    states/completed-<count>-<sha-prefix>.json
    ledgers/events/through-<count>-<sha-prefix>.jsonl.gz
    ledgers/evaluations/through-<count>-<sha-prefix>.jsonl.gz
    final/<sha>.json
```

The context evaluation pack is the authoritative detailed scientific record. Campaign evaluation
rows are compact indexes with objective values and lineage. Generation state objects support fast
plots; they never replace detailed records. Every S3 write above is immutable or a conditional create.

## Selection and claims

Official test labels intentionally influence the balanced and test-gated islands. Every selected
tree must also pass the training-only gate: geometric mean CV RMSE ratio at most `0.90` and wins on at
least five of seven folds. These outputs are teacher-target mining evidence and cannot be presented as
an unbiased benchmark comparison.

The eight 112-member islands use balanced selection twice, test-gated selection, CV-stability
selection, epsilon lexicase twice, novelty-quality selection, and age/root-diversity selection.
Six quality/novelty migrants rotate per island after every five global generation-equivalents.
`histgbm2` and `gpr` are filtered before dispatch because the control campaign showed high timeout
rates and no competitive result. More than three winsor stages and complexity outside 3-10 open
parentheses are also rejected before paid evaluation.

## Secrets and cost

`transport/.env` is mode `0600`, ignored, and contains only a local AWS profile name and pinned paths.
No AWS keys appear in source, S3 manifests, events, or result packs. The new runtime begins with an
exact copy of the stopped parent SQLite admission ledger. A call reserves its full 120-second,
256-MiB ARM Lambda upper bound before the S3 claim. This makes the `$1,000` total ceiling conservative
across both campaigns and across coordinator crashes.
