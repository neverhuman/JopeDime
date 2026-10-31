# W5 provenance and row-closure preflight

Owner: `sol_v1_obj02`, owner ruling `gb2136_lanes.md`. This is the admission
preflight for [W5](../PROPOSALS.md#w5--parent-child-value-and-bounded-evolutionary-lookahead),
under [S3 admission and row closure](../PROPOSALS.md#s3-admission-and-row-closure).
It does not start W5, fit a model, call native evaluation, or use a GPU.

## Admission boundary

Only `s3://veox-jopedime-learning-use2/experiments/flywheel-selfplay-20261023/`
is inventoried. `verify.py --collect` performs a bounded metadata listing and
five HEADs. The **only body GETs** are the three retention-counter receipts,
at the exact versions and SHA256s admitted in [ASTRA's receipt](../astra/S3_RECEIPTS.md).
The original 806-byte bodies are preserved and rehashed during offline replay.
No credential bytes are recorded. ETags are not substituted for SHA256.

Raw retained-score and distilled bodies are ineligible: their source schema
includes `test_rmse_ratio` and `cv_test_agreed`. Corpus inventories are not numeric
training rosters. This preflight does not fetch either corpus-manifest body,
local pools/tables, checkpoints, new DL-Val, outer-test or sealed bodies, and
does not widen input scope to the current full loop. Listing a suggestively
named export does not admit it. Retention counters cannot supply parent-child
labels, completed-outcome status, pair coverage, or unbiased off-policy evidence.

No independently admitted clean parent-child export or original-training/strat300
row ledger is currently registered by this preflight. Consequently its command
**refuses with exit 1 and FAIL / W5 HELD**, including when counter verification
succeeds. It does not scan unadmitted bodies looking for those missing proofs.
This establishes a missing admission prerequisite, not that all possible custody
lacks pairs or that an actual strat300 table has been proved contaminated.

## Required evidence before any future W5 fitting

A separately prepared, field-allowlisted training-only export needs independent
admission with immutable source version and verified byte SHA256. Its closure
must establish all of the following, not merely self-declare them:

- Exact parent, operator and child IDs, same context, physical table, cut,
  evaluator, metric, gate and vocabulary; original unrounded outcome identity.
  Similar ASTs and generation/birth chronology cannot establish parenthood.
- Completed full measured outcomes, distinguishing genuine gate failures from
  missing, execution-failed and screened-only results. Finite `fitness=100,
  logratio=0` sentinels do not prove a measured outcome.
- Current DL-Train role/alias mapping and independent original row IDs. Every
  strat300 row must belong to the target's original training envelope at that
  exact cut/table. A sibling view is not an admissible source; `nq=0` is insufficient.
- Collection/retention policy and a pair-coverage census. Selected historical
  confirmations do not measure screen-reject recall or supply missing propensities.

`pair_errors` and `row_errors` implement metadata consistency controls for these
joins and row sets. Their positive unit fixtures are synthetic, not admitted
data. They neither authenticate a source nor qualify W5; future admission must
bind independently verified export/ledger bytes before applying them to data.
Unknown pair counts and coverage remain JSON `null`; unknown exposure remains
unknown, never asserted absent.

## Acceptance and receipts

From the repository root, one nice-15 CPU process, no Cargo build:

```sh
python3 -m unittest discover -s research/flywheel_selfplay/rd_breakthrough/w5_provenance -p 'test_*.py' -v
nice -n 15 python3 research/flywheel_selfplay/rd_breakthrough/w5_provenance/verify.py --collect --output research/flywheel_selfplay/rd_breakthrough/w5_provenance/receipts
```

The second command's exit 1 is the recorded data-admission FAIL, not a passing
training experiment. Replay without AWS or any source-body access:

```sh
python3 research/flywheel_selfplay/rd_breakthrough/w5_provenance/verify.py --inventory research/flywheel_selfplay/rd_breakthrough/w5_provenance/receipts/inventory.json --output /path/to/evidence/replay
```

The receipt inventory binds collector source commit, verifier file SHA256,
command/profile/host/time, actual S3 keys, pinned counter versions and byte
hashes, metadata-only HEAD versions, duration and collection errors. The verdict
binds the inventory SHA256. W5 stays held until actual source/row admission passes;
these receipts cannot establish a W5 quality, Elo, advancement or serving claim.
DOPE remains frozen at `c51a56d0`.

## Actual result, 2026-10-31T04:00:10Z

**FAIL / W5 HELD**, intentional exit 1. The complete approved-prefix metadata
census returned 83 objects; one LIST, five HEADs and three pinned counter GETs
completed without collection errors in 1.363 seconds including process startup.
All three counter byte hashes passed. No independently admitted parent/operator/
child export or original-training/strat300 row ledger was available to this
preflight. Actual pair joins and row containment therefore remain **unverified**,
and pair count/coverage remain `null`.

See [inventory](receipts/inventory.json), [verdict](receipts/verdict.json),
[execution receipt](receipts/execution.json), and [protocol pin](receipts/protocol.json).
The executed collector source is `ca549a23e34b0338ff238d806bc65be990e43102`;
its file hash is recorded in both inventory and verdict. Subsequent documentation
and receipt commits preserve the verifier bytes. Twelve synthetic unit controls
passed; the actual data-admission verdict remains FAIL.

Long-term create-only S3 custody and verified download hashes are recorded in
[custody](receipts/custody.json). The manifest was written after every listed
object passed byte readback; this output prefix grants no additional input scope.
