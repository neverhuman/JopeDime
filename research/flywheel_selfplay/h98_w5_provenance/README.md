# H98 — W5 data-provenance gate

**FAIL / W5 HELD.** Verified manifest and pool bytes do not establish the exact
parent-child, original-row or global sealed-family exclusion proofs required
by [W5](../rd_breakthrough/PROPOSALS.md#w5--parent-child-value-and-bounded-evolutionary-lookahead).
The requested CPU smoke plan is **not added**, because the owner made it
conditional on provenance PASS. No model, fit, GPU work or study scoring ran.

This refresh extends the [earlier W5 preflight](../rd_breakthrough/w5_provenance/README.md)
with the v3 census, all materialized pool hashes and an identity-only alias
check. It preserves [H71's preparation-only status](../h71_w5_parent_child_critic/README.md).
Source is pinned to `433e50a2c6486d4edcc7173b4b2c6552c1621f8b`; exact file hashes and
input versions are in [metrics.json](metrics.json).

## Checked evidence

| Check | Observation | Disposition |
|---|---|---|
| Approved S3 source prefix | Complete census of 83 objects, no added/removed keys versus the previous census; five source HEAD versions unchanged | PASS for this bounded snapshot |
| Source and prior custody | 14 exact-version objects downloaded and byte-SHA verified: three retention counters, v3 manifest, previous custody manifest and its nine objects | PASS for these bytes |
| v3a materializations | 241 pools, 122 recorded families, 430 winner occurrences; all local and H50 parent hashes match | PASS identity only |
| v3b materializations | 254 pools, 123 recorded families, 504 winner occurrences; all local and H50 parent hashes match | PASS identity only |
| Known sealed markers | Zero candidate-context overlaps against 86 known sealed alias components | Checked-scope result only |
| Global sealed meta-test families | Authoritative complete roster/exclusion proof is not supplied | FAIL: global overlap unknown |
| Other protected roles | 55 contexts connect to Track 2 non-training-role components; 66 connect to historical `st_val_gate_32` components | Reconcile before admission |
| Exact parent/operator/child and measured outcomes | Not supplied by the materialized-pool manifest | FAIL: eligible pair count unknown |
| Original-training/strat300 row closure | No independently bound original-row/selected-row ledger | FAIL: containment unverified |

The two revisions overlap: **495 pool-file checks cover 254 unique contexts**,
not 495 independent contexts or 934 unique winners. All candidate identities
match the pinned H50 DL-Train roster, including dataset, family, view and source
split. The 1,219-context roster was decoded only through its first train array;
DL-Val metadata following it was not decoded. Pool files were hashed as opaque
bytes; no pool arrays or numeric tables were decoded.

The source prefix contains only the v3 manifest under its v3 subprefix, with
zero pool-payload keys. Local verified pool availability and S3 payload custody
are therefore separate facts. No claim is made about copies outside the bounded
source prefix, and this audit does not upload legacy pools as a new admitted export.

## Exclusion scope and source limitations

The exclusion check uses six hash-pinned metadata sources from the existing
HWV-06 custody: catalog identities, H15 ledger/verification, corpus identities,
historical roles and Track 2 identities. It joins recorded families, contexts,
dataset aliases and physical-table identities. The candidate set is nonempty,
all candidate identities resolve, and it has zero overlaps with the supplied
DL-Val, DL-Holdout and sealed-marker components.

That is **not a global sealed-family PASS**. The existing
[metadata builder](../input_manifest.py) explicitly records
`global_sealed_roster_available: false`; its sealed scope is catalog exclusions,
H15 roles and marked BeyondArena origins. Global overlap remains JSON `null`.
The 55/66 other-role matches are conservative family/physical-alias quarantine
signals, not proof that a particular protected outcome was used. They may overlap;
do not add them. Exact training-context hashes and scope are retained in the
receipt, so the bindings can be reconciled without opening protected data.

The [v2 producer](../verify_flywheel_winner_pool_integration_v2.py) serializes
`verified_full_jtt1_outer_test_ratio` and `verified_full_jtt1_cv_test_agreed` into
its manifest. Its body was not fetched or decoded here. The exact S3 HEAD/version
was checked; the historical v2 byte hash remains a recorded upstream pin, not a
newly verified body hash. The v3 manifest's parent-v2 hash matches that pin.

The [v3 producer](../verify_flywheel_wider_train_extension_v3.py) combines phase-6
v2 winners with phase-7 winners. Its materialized-pool records retain child
program hashes and pool ancestry, but not the required exact same-context
parent-program/operator/outcome join. A `parent_pool_sha256` is not a parent AST.
The phase-7 `run_outer_test=False` setting does not establish the complete
historical ancestry or completion provenance of every imported row.

The pinned [strat300 source](../rust_evaluator.py) selects training indices but
can retain outer feature rows while zeroing their targets; its cache key uses
the source path and row limit rather than the source-content hash. It emits no
original-row identity ledger. These are source-level limitations, not proof that
the historical artifacts used those exact source bytes or that an actual table
is contaminated. No table was opened to manufacture a stronger conclusion.

## What would allow a new gate decision

1. A complete authoritative identity-only sealed family/alias registry, or an
   independently verified complete exclusion receipt bound to this exact roster.
2. Resolution of the protected/non-training alias matches, followed by a frozen
   cohort and protocol comparability check if the cohort changes.
3. A clean immutable parent/operator/child export with matching context, original
   table/cut, evaluator, metric/gate and vocabulary; source version/hash and
   original unrounded completed full-outcome receipts for both programs.
4. Run-bound original-training and strat300 row-ID/content-hash ledgers proving
   the permitted subset, with caches bound to actual source bytes.

After these checks pass, the W5 CPU smoke plan can be prepared under its existing
budget and dependency gates. No new compute budget or launch authority is implied
by this report. Elo, U4 and val_reg_inner are null.

## Verification, resources and custody

Four deterministic metadata proof shards ran on xbabe1/2/3 and xbabe4 inside
Linux Docker/Colima, one worker per host. Load/headroom was checked below 85%;
xbabe1 retained at least 150 GiB free. Actual nice values were 19 on xbabe1/2 and 10
on xbabe3/4. The final four workers used
0.270876 CPU-seconds; including the initial pass before adding
the historical role map, total worker CPU was
0.544213 seconds against the 120-second audit ceiling.
Preparation CPU is recorded separately; transport/import/control/publication
overhead is unmeasured, not zero. There is no seed or family-fold experiment.

All **19 synthetic checker controls passed**, including alias transitivity,
known-overlap refusal, incomplete-global-roster refusal, non-vacuous overlap,
unresolved identities and source-hash tampering. These validate the checker;
they do not turn the real-data admission FAIL into a PASS. Source pins, JSON,
Markdown links and the publication diff were checked. No hosted CI was run
under the owner's focused local-check policy.

Create-only S3 manifest: `s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/W5/astra-h98-provenance-20261031T093300Z-dd7fed25/manifest.json`

Version: `K_87tbL0dazl0FgTSxmUszw4CWYMBDS.`
SHA-256: `86679a26544da2e81a5fa88426f3f95ef5644645ea1ed71575086dacc561cdc9`

[custody.json](custody.json) lists all four output objects. Exact-version GET and
byte SHA checks passed for all four; every one of the
68 evidence-archive members was checked. The archive retains source,
checkers, input metadata/projections, pool-hash receipts, initial/final shard
receipts and controls. It contains no numeric tables or legacy pool payloads.

To reproduce the bounded checks, retrieve the recorded archive version, verify
its SHA and extract outside Git. Run `nice -n 10 python3 -B controls.py` (19 controls)
and `nice -n 10 python3 -B dispatch.py` (all four hosts, fresh preflight), then
`combine.py`. The result remains `FAIL`/`HELD` for these frozen inputs. These are
archived standalone checkers, not new repository test registrations. Restoring
the metadata replay does not recover the local legacy pools; rehashing those
requires their original read-only custody paths as specified in `prepare.py`.
Small audit receipts remain local; owned bulk archive/download/metadata copies
were removed after verified preservation. Original inputs and peer work remain.
