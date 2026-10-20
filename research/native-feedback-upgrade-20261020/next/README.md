# Measured feedback journal: engineering candidate

This additive packet prepares the fresh-GP wave in `../NEXT_WAVE.md`. It does
not modify the completed historical replay, either native backbone, production
aliases or a high-water mark. No new GP fit or optimizer update was run here.

`feedback_targets.py` provides `Journal.reserve`, `Journal.finish` and
`Journal.snapshot`. Reserve before dispatching an evaluation. The transaction
charges one logical context/program opportunity against the frozen manifest,
including failures and unresolved attempts. The same key after restart returns
the same reservation; it cannot refill the budget. A retry or another numerical
seed requires a separately declared experiment. Real compute costs must still
be measured by the executor; this journal alone cannot establish equal cost.

Finish accepts exact result bytes with an S3 URI, VersionId, size and SHA256.
The caller must fetch that exact version; this module does not contact S3 or
prove an object exists. Successful comparable measurements retain ranking
evidence. Only strict improvement over the bound baseline creates a prospective
CE candidate. Ties, worse values, rejected metadata, failures and censored
outcomes remain visible. Failed outcomes never supply a numerical reward.
Snapshots recompile targets from retained result bytes and reject detected
metadata, hash or target drift. Interrupted commits retain reserved liabilities.

Native comparison fields and metric directions were read from these files at
`ca019e7c2c514f2875d64e12db1f28e4760d60a4` (unchanged from its main parent):

| File / symbol | Source SHA256 |
| --- | --- |
| `crates/jope-learning/src/record.rs`: `HwmIdentity`, `ComparisonContext` | `4360d96a4a6c0a27f5ba78fbcab4f5be10a3a800f56acee6379ef805b3815d61` |
| `crates/jope-learning/src/hwm.rs`: `decide` | `36009bf4103bc21892057ae7b593124caae2127b3f209de576283e3f3372c95b` |
| `crates/jope-learning/src/metric.rs` | `3f8832761dcaf1761dab63672a99bcc7971b2dbda45c382870b74d96725c1e2f` |

These fields are an explicit research envelope, not a drop-in native teaching
record. Every target and snapshot says `native_teaching_qualification=pending`,
`training_admitted=false` and `hwm_published=false`. A manifest's declared TRAIN
role does not independently prove dataset rights, family isolation or baseline
measurement provenance. `reported_fresh_fit` is a producer report, not verified
numerical-fit accounting. Do not count it as established fresh fits.

## Validation observed 2026-10-20

Seventeen tests passed, none failed or skipped, in 0.855 seconds. Fixtures cover
restart-safe failed/duplicate accounting, strict metric direction, identity and
program mismatch, immutable conflicting results, failure rewards, exact object
references, malformed/nonfinite input, budget exhaustion, crash rollback,
manifest loss, target corruption and competing connections. One regression
imports the original `FeedbackLedger` and reproduces its duplicate timeout
charge. The frozen completed-outcome replay did not exercise that path and is
unchanged. This regression requires the existing NumPy/PyTorch environment;
the new journal itself uses only Python's standard library.

From the capsule directory, put transient databases outside Git:

```sh
rtk proxy env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  OPENBLAS_NUM_THREADS=2 JOPE_FEEDBACK_TEST_ROOT=/absolute/evidence/next-fixtures \
  python3 -m unittest discover -s next -p test_feedback_targets.py -v
```

There is no independent exact-head review or CI acceptance for this packet yet.
Synthetic fixture success is not measured GP lift or an admitted training run.

## Ordered next work and ownership

1. Existing fleet resource owner qualifies a policy-bound launch path with
   60–65% operating margin, one-second host measurements and peak records,
   reduction/pause of only owned work before the 70% ceiling, and retained
   RAM/swap/temperature/storage guards. The read-only audit stopped at
   `cpu70-audit/CPU70_RESOURCE_POLICY_AUDIT_STOPPED.json`, SHA256
   `fd0471252541d9c0569b7f1fb24596df73f0ddceaa38654324913f29e4a1bec5`.
   Existing shared-slice R4 ownership prevents unilateral controller changes.
   No fresh GPU/GP work is qualified by the current target-only controller.
2. This research owner, after renewing its exact disjoint claim, adds
   `next/native_result_bridge.py` and `next/test_native_result_bridge.py`.
   Bind native measured result, actual Kumo context and baseline provenance to
   this envelope; run the existing native teaching verification before any
   target is eligible for training. Prove exact-version GET and corrupted-byte
   rejection, role/parent/rights closure, evaluator/cut/vocabulary equality,
   program identity, failed opportunity retention and native round-trip
   compatibility. Do not infer compatibility from matching field names.
3. Freeze `next/fresh_wave.manifest.json` only after concrete native result
   identity and resource binding resolve. Follow `../NEXT_WAVE.md` for both
   frozen native controls, matched continuation/uniform/targeted arms,
   independent parent units, practical lift threshold, costs and stop rules.
   Keep fresh numerical fits, cache hits and historical replay separate.
4. Existing PR387 integration owner retains the one-PR queue. Queue this small
   packet for actual independent exact-head review and CI after that lifecycle
   clears. Record ordinary merge and receiving-main verification separately.

Stop on an unqualified resource path, source/runtime/data identity mismatch,
missing native teaching proof, sealed/outer-role leakage, exhausted declared
budget or conflicting ownership. Preserve each failed result and cost. Do not
change R4, shared controller services, foreign jobs, the original trial files,
or production/HWM aliases. Receipt and source publication live under the
existing evidence custody, with no new checkout or numerical worker.
