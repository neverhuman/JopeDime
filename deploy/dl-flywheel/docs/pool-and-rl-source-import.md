# Native CPU pool and RL audit source import

This additive package is based on PR277 head `61e712cfc1ee8341776161eb6886ad73d3c094b3`. It contains the reviewed cardinality-aware native pool and a portable finite-action RL mathematical audit. It does not change the active model, scientific evaluator, parity tolerance, permanent execution registry or foreign RLFW source.

`pipelines/hpc_pool_v1/{operations_pool.py,native_build.py,test_pool.py}` remain byte-exact copies of reviewed source. The8/12/24-worker input chooses explicit cardinality; no silent downsizing occurs. Actual physical resource and custody gates still run before freeze, every preflight and each dispatch. The12-worker attempt was held before its clock started. Dated input review is not a current resource pass.

`docs/rl-spec-reconciliation.md` is the byte-exact finalized research reconciliation. Its operational counters, source heads, foreign-package observations, loss and STOP descriptions are dated author-reported evidence, not claims proved by these math fixtures. Refresh exact operational/source receipts before relying on them as current status. The document describes future actor work; neither this package nor the finite examples implements or qualifies an actor-learning runtime, CUDA update, native GP campaign, win-rate lift or generalization improvement.

`pipelines/rl_spec_audit/provenance/` retains the exact original audit script as text and the original numerical receipt. The portable `audit_math.py` preserves the mathematical helpers and all eight fixed cases, extracts `audit_cases()`, removes the private attachment dependency, and requires an explicit create-only `--output`. The recorded attachment SHA is a historical identifier only; the private attachment is neither distributed nor reverified. Runtime source, source-output directories and retained receipts are not written by the audit.

## Portable qualification

Run each family in its isolated process from the repository root; output directories must be new:

```sh
python3 -B deploy/dl-flywheel/run_native_pool_offline.py --output /tmp/jope-pool-proof-UNIQUE
python3 -B deploy/dl-flywheel/run_rl_math_offline.py --output /tmp/jope-rl-math-proof-UNIQUE
python3 -B deploy/dl-flywheel/pipelines/rl_spec_audit/audit_math.py --output /tmp/jope-math-audit-UNIQUE.json
```

The pool family contains21 fixtures. The math family contains eight estimator/counterexample fixtures and one CLI portability/create-only custody fixture, nine total. Only these30 new packaged fixtures were executed for this import. The original587 identities and runner sequence remain unchanged. The full offline runner now wires617 identities; this is a wiring count, not a claim that all617 were rerun. There are zero GP executions, cloud requests, optimizer updates or production queue mutations in these fixtures.

Coverage and research classifications start from the reviewed union of PR277 and main `717451249` metadata, retaining both independent RLFW families and both Python-lane evidence/check lists. That union describes source merged separately by the release owner; this package does not rewrite foreign code. All new Python entries are classified research. Hashes in `SOURCE_MANIFEST.json` describe the packaged files; historical source/review receipts retain their original paths and hashes without pretending relocation preserved path identity.

## Static inventory schema correction

The native `jope-records verify-test-inventory` command defines its expected keys solely from test-bearing CI coverage items (Python test families and test-enabled targets). The prior inventory contained119 production/helper sources in addition to those families; the error displayed only the first five. No source hashes caused that failure. This import moves the119 exact role/reason values into `RESEARCH_SOURCE_CLASSIFICATIONS.json`, adding exact source bytes/hashes there. All494 existing test-bearing families, all dispositions and both main-only RLFW families remain in the test inventory unchanged; the two new test families yield496. New production/helper sources are recorded separately, giving125 non-test research source records.

`verify_source_inventory.py --repo <checkout>` is an additional static key/role/hash check. It does not replace native `jope-records`, baseline-triage checks, coverage discovery or any tests. The source/test split avoids falsely labelling production helpers as test families merely to satisfy the validator. The release owner must run native `jope-records --repo <checkout> verify-test-inventory` against the final merged tree.

## Cold source revision2 and operational hold

The first source-import attempt stopped at `git diff --check` before commit/ref mutation: valid single-space context lines inside the retained unified diff triggered whitespace errors. Revision2 replaces only that new project artifact with `operations-pool-diff.json`, preserving its exact decoded bytes, length and SHA256. The original local diff, failed remote intent/private index and frozen revision1 payload remain intact. Runtime, scientific and fixture source bytes are unchanged; no tests were rerun for this encoding/documentation correction. Native `jope-records verify-test-inventory` subsequently passed with496 test families; that static result is not a full617-test execution claim.

The additional eight-worker launch remains HELD: an idle peer CI listener can claim48GiB and two QEMU services9GiB each, without an established shared-host lease. A sampled42.625GiB local threshold alone is insufficient. A current owner-bounded shared allocation is required before any new freeze/dispatch; no foreign services are controlled and the source guard is unchanged.
