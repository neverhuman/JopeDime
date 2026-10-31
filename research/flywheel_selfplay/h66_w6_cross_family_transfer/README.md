# H66 / W6: cross-family outcome transfer CPU preparation

## Hypothesis and registered advance gate

W6 tests whether conditional preferences across recurring programs and structural
neighbors improve new-family proposal selection beyond the program-only prior.
The registered arms are the r882 continuity control/current raw-lm12 retrieval,
fit-family program mean, W1 program-only prior, and a regularized
dataset-feature × program-feature residual with a context-shuffled control.
Exact-program and feature-neighbor transfer are reported separately; unseen IDs
use features, never fabricated zero outcomes.

The common gate from
[`rd_breakthrough/PROPOSALS.md`](../rd_breakthrough/PROPOSALS.md) requires a
positive adjusted family-mean lower bound above `+0.001`, median family delta at
least zero, and no more than 10% of families losing more than `0.01`, versus the
strongest matched control. `+5 Elo` is a practical target, not a substitute for
the raw utility gate. W6 must advance over both the generic prior and current
retrieval at matched budgets. Kill if benefit disappears on new templates, the
context-shuffled control matches, or coverage cannot identify context
interaction. This preparation does not execute W6, establish an advance, authorize
GPU use, or authorize serving promotion.

## Frozen protocol pins

- Proposal SHA-256: `e07688c1d999bf9d10b20da62703102fb9aa3f291a1e4b5eb8f0829c3c60e199`.
- Packet manifest file SHA-256: `99a53569fcaf59f9309e69329c830ae1684adff382da3ae22e0b2645c4ebc4d4`.
- Packet identity: `3eb4a62acea512f528500b043299a5968e1c9b1d1b1759bbf77a426b9fa1ecc3`.
- Roster: 1,219 DL-Train contexts, 205 families, regression only.
- Five family folds, seed `4900`; experiment seeds `701`, `702`, `703` are kept
  together within each family and are not independent families.
- At score time, exclude same-family and physical-table-alias donors. Preserve
  original training labels and row envelopes. No DL-Val, outer-test, sealed,
  holdout, or sibling-view outcome reads. DOPE remains frozen at
  `c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814`.

## Input admission and current limit

`prepare_folds.py` accepts only a small, already-admitted `jope.w6.train-roster/v1`
metadata export. The export contains context, family, and physical-table hashes
only, and pins both packet identities above. It rejects all extra fields and any
row outside `DL-Train` regression. It never opens pool, table, outcome, checkpoint,
or score-body files. The export must be produced from an authorized training-only
source; the full `selected.json` file is not an acceptable input because it also
contains development metadata.

The published S3 admission table authorizes the v2/v3 corpus manifests for census
and exact joins only, and states that v3 pool payloads are absent from the
approved prefix. A read-only CPU inspection streamed and parsed the 244,173-byte
v2 manifest in memory to inspect its envelope. It emitted only envelope keys, row
count, and field names; the schema contains an `outer_test` field. No individual
values were emitted or used for folds, model work, or metrics. This was a body
read, so it is not described as zero-read. The manifest is disqualified as a W6
roster. No safe training-only S3 roster was available during this original prep
pass, so that pass produced no real fold map or real-data smoke. The approved S3
persistence receipts are provenance counters only and are not substituted for
training data.

### H72 roster follow-up (2026-10-31)

H72 now supplies the narrow W6 roster from the already admitted H50 packet's
top-level `DL-Train` partition and per-dataset canonical physical-table hashes.
It contains 1,219 contexts, 205 families, 257 version-pinned source-object
references, and the seed-4900 fold manifest. The builder stops parsing before
the DL-Val array and decodes metadata only for training-referenced dataset IDs.
The full source files were hashed for identity, but unselected rows were not
parsed or used. No S3 source object bodies, outcome values, pools, tables,
checkpoints, or scores were opened. See
[`H72 W6 roster preparation`](../h72_w6_train_roster_prep/README.md) for the
create-only S3 custody keys and readback hashes.

This closes roster preparation only. It does not provide W1 model outputs, W6
outcome history, fit evidence, or a scientific verdict; the H66 numerical study
remains unstarted and its registered budgets are untouched.

## CPU checks

Use the folder's synthetic unit tests to verify deterministic grouping and
fail-closed role/schema handling. H72 provides the admitted roster; the study
remains preparation-only until its outcome inputs and W1 comparator are
available:

```sh
nice -n 10 python3 -m unittest discover \
  -s research/flywheel_selfplay/h66_w6_cross_family_transfer \
  -p 'test_*.py' -v
nice -n 10 python3 \
  research/flywheel_selfplay/h66_w6_cross_family_transfer/prepare_folds.py \
  --help
```

With the admitted external roster, write the fold manifest to a durable receipt
directory outside the source tree:

```sh
nice -n 10 python3 \
  research/flywheel_selfplay/h66_w6_cross_family_transfer/prepare_folds.py \
  --roster "$W6_TRAIN_ROSTER" --out "$W6_FOLD_RECEIPT"
```

The output contains metadata hashes and fold assignments only. It is a CPU
preparation artifact, not W6 numerical evidence.
