# Browser prediction merge: inherited CI repairs — 2026-11-02

Manual full CI41383543203 at fa98944 failed three checks already broken on its base. No failed lane is claimed as passed. These repairs retain actual execution, scientific identities and existing security boundaries.

## Imported source custody

PR1702 and subsequent audit cleanups renamed variables or rewrote equivalent expressions in nine frozen producer files. The existing byte-for-byte verifier correctly refused them. Restore the eight production files EXACTLY to their unchanged producer bindings. A ninth, stale source-closure assertion needs the identified successor correction below. No numerical fixture/tolerance or production computation changes. These are historical imported sources, not new producer authorization. The general heuristic auditor's variable-name/empty-fixture findings are exact-file false positives here. Only the eight restored production files below are excluded from that heuristic; the corrected P3 test remains normally audited. The P3 row lists its preserved predecessor binding; its successor is recorded below. their stronger hash/syntax/custody checks and Gitleaks remain active. The exceptions last only while these bytes remain frozen and this heuristic applies.

- `deploy/dl-trainonly/lane/src/dl_training_v1/dltrain/encodings.py`: SHA-256 `e08ab26c0de16db05c6c37ba7005a55e1263e42b982935d8ea41410acdcf9427`; restored from `5619bf999a36730944a53428eb077ddc73a9ae2f^`.
- `deploy/dl-trainonly/lane/src/dl_training_v1/dltrain/holdout.py`: SHA-256 `6d25d3a4228ac073a6c17b0f1b78d73db79477c2108485e353203501dcac73d6`; restored from `5619bf999a36730944a53428eb077ddc73a9ae2f^`.
- `deploy/dl-trainonly/lane/src/dl_training_v1/dltrain/metafeatures.py`: SHA-256 `2b9ea218f5cbf7161edb012b69773e78faa10567362971281c5bf68461b861f2`; restored from `5619bf999a36730944a53428eb077ddc73a9ae2f^`.
- `deploy/dl-trainonly/lane/src/dl_training_v1/dltrain/train.py`: SHA-256 `5e42507aaab86ab5eb46646cab43305a78678eb60cd180eb3908f883315eaf09`; restored from `5619bf999a36730944a53428eb077ddc73a9ae2f^`.
- `deploy/dl-trainonly/lane/src/dl_training_v1/tests_p3/test_p3_verify.py`: SHA-256 `2c98d995424b02eed0da09170c94dc3ecce5b7ab2acc25f0831ec4ac419096f4`; restored from `5619bf999a36730944a53428eb077ddc73a9ae2f^`.
- `deploy/dl-trainonly/lane/src/dl_training_v1/tools/s1_outer_test_scorer.py`: SHA-256 `9ee08fbb91f8209007c12281559efc32e8ea19790e4b485238939b6a850a59c4`; restored from `1dc2320b8f3bbe4a7ebd7eb157341ed5adaf8fcc^`.
- `deploy/dl-trainonly/lane/src/dl_training_v1/tools/v2_rowsel_build.py`: SHA-256 `b1487969896224a733a9af9f8106c6d1ff879526fca9d0e17719196828631a00`; restored from `1dc2320b8f3bbe4a7ebd7eb157341ed5adaf8fcc^`.
- `deploy/dl-trainonly/lane/src/dl_training_v1/tools/v2_rowsel_raw_coverage.py`: SHA-256 `0382899423a1b7407e213983a366a94673c4e27bd21ff20b33752fbfec4e0585`; restored from `1dc2320b8f3bbe4a7ebd7eb157341ed5adaf8fcc^`.
- `deploy/dl-trainonly/lane/src/dl_training_v1/tools/v2_rowsel_raw_helper.py`: SHA-256 `13e68a9dda3cd346e7332ac9daefd036fd670100f54f295d06274ac659cca733`; restored from `1dc2320b8f3bbe4a7ebd7eb157341ed5adaf8fcc^`.

## Frozen benchmark split environment

The application now locks sklearn1.9.1, but benchmarks/prepare.py explicitly requires1.7.2 for its frozen split identity. Keep the application lock untouched. Run every benchmark preparation test in its own locked Python3.12 project with sklearn1.7.2, preserving a separate identity-bound nonzero execution receipt and full JUnit. The parent Python receipt continues to count its own application/adaptor pytest run once. Benchmark lock and project are bound in CI receipt inputs. No splitter/labels/partitions change, no version guard is relaxed.

## Authenticated CLI HTTPS follow-through (A20)

The original candidate fenced HTTPS behind `client` using exact package exceptions. Main PR1796
subsequently made ureq/zip optional behind `client` and HTTPS an explicit `cloud-tls` capability.
Adopt that merged architecture and withdraw the candidate's now-superseded dependency-policy
and parser exceptions; the original inference fences remain unchanged. Connected development
source builds select `--features cloud-tls`. The existing verified development installation is
not replaced by an unqualified build.

## Local authoritative base

This host retains an older forge `origin/main` alongside authoritative GitHub `github/main`. Add an explicit `JOPE_CUSTODY_BASE` argument for the local custody-additions comparison; `JOPE_CUSTODY_BASE=github/main just required` checks the same real PR base as hosted CI. Defaults and hosted base selection remain unchanged. No protection, file-size threshold or custody refusal changes. The direct check against GitHub main passes; the old local origin compares unrelated historical custody imports and correctly refuses them.

## Identified source-closure correction (no numerical change)

After restoring the imported bytes, the original P3 receipt assertion fails because its expected dependency list omits jtt1.py and quarantine.py, which the original static-import closure already includes. The earlier audit cleanup had corrected the assertion without a custody successor, causing the original hash refusal. Preserve the prior current manifest byte-for-byte under `predecessor/source-closure-correction-20261102/`, retain the original test bytes in historical/, and create successor `9afa6c81dc7e4ce0df2d45b76b7d871fe0ea5f660c932f4bb1e6f4553ca49ef7` that changes ONLY this stricter expected dependency enumeration. The verifier retains the full predecessor chain, and its tests require177 current/285 historical with the old177-file closure reconstructible. No producer source behavior, rows/labels/splits, numerical test or tolerance, host exclusions or offline test counts change. This is a separately identified source correction with reproducing failure and passing strict assertion, not blanket repinning to make edited sources pass. See SOURCE_CUSTODY.md and the manifest source_correction record.

## Local/hosted Python parity

`just python-check` now explicitly syncs the same locked CPU/dev/benchmark/foundry/custody extras as the hosted Python lane, then uses `uv run --no-sync` for Ruff, pytest and both research fixture runners. A fresh checkout previously had neither Ruff nor the selected CPU extras, and ambient system Python was used for Torch fixtures. This changes environment preparation only; tests/markers remain intact and no application dependency lock is changed.

## Native compatibility and local test isolation

All13 compatibility surfaces compare successfully in the documented identity-only mode; only source/lock/lineage metadata changes, with old/new identities announced before regeneration and a recorded transition. No numerical or emitted/model fixture is rewritten. The local full suite also exposed default-engine interference and invalid fixture placement when TMPDIR was inside a registered checkout. The typed in-process API unit now owns a private temporary engine instead of opening a user's running default engine. Test scratch is staged outside registered checkouts and parallel fixtures are bounded. Qualified RES07 fixtures are materialized from the README's independently SHA-pinned catalog a16efb5c3a8bd09ea08caaaf030c199f5d2d733153874d112b928a3f1cd2bb86, using original read-only custody objects from xbabe2, never fabricated or regenerated numerical fixtures. These are environment corrections and stronger isolation, not skipped assertions.

## Measured development-package gate

The old A18 fixture size gate also failed unchanged pre-connected-CLI main. Paired same-host dist-ci builds measure8144488/8267461B code/package before PR1659 and8144648/8268941B for this candidate. Only160B code and1480B net package growth are new; compressed dependency notices account for the net package delta, with model bytes unchanged. A21 records exact source/code/package hashes and explicitly unattributed inherited main growth. Only the development fixture baseline updates, keeping5% tolerance and every release profile/ceiling unchanged. The candidate passes development-key cryptographic verification, the updated size gate and the network-isolated clean-host fit/predict check.

## Native-only immutable runner and canceled-audit recovery

The shared runner image has no wasm32 standard library. The runtime builder now provisions the same pinned Rust toolchain/target in the lane target cache when needed; it never edits the immutable shared image. An actual read-only native-only runner-container build with no capabilities passes, followed by all11 real native/WASM parity/tamper/no-import checks.

Canceling superseded audit41388179289 left its read-only source copy in xbabe1's CI workspace. Subsequent checkout removed writable metadata but could not remove that source tree. Recovery archived the remaining copy, verified all13645 files byte-for-byte against the historical d1537a7eef1552d50225cda965fae03da720481c Git blobs (zero mismatches), and restored owner-write permission only on that own canceled snapshot. No source bytes, peer checkout, service configuration or provenance/admission rule changed. CI's audit now mirrors full-main's existing always-run post-upload permission restoration; the real immutable-source filesystem test covers both workflows. Normal cancellation is used for subsequent superseded runs so their cleanup can run. Recovery proof and archive remain under the browser-canceled-audit evidence paths outside the lane.

## Reusable paper workflow concurrency

The paper source-check reusable workflow used `github.workflow-ref` as its concurrency group. Called by manual ci.yml, `github.workflow` is the caller name ci, so its key equals the parent ci-ref group. Manual41398271392 stayed queued with active children and no paper job; official GitHub reusable workflow documentation warns against this caller/callee collision. Prefix only the reusable paper key with paper-, retaining cancellation of genuinely superseded paper checks and every gate. An existing runner-contract test evaluates the real caller/callee expressions for manual, PR and trusted-main contexts: it fails before the correction and passes after. No paper typesetting is restored. See https://docs.github.com/en/actions/reference/workflows-and-actions/reusing-workflow-configurations.

## Captured effective audit policy fingerprint

Manual41399605228/PR41399614338 correctly refused reports at the verifier's old default-policy
fingerprint. Pinned Jankurai1.7.5 audit/outcome.rs fingerprints canonical effective settings plus
SHA-256 of the complete repository policy source. Independently applying that algorithm to the
captured e15 policy derives the actual c96d07ea30c035ef0f338d8246e23c55e063a0a20af78b2f159a6ab012d353ec.
Derive the expected fingerprint only after validating the complete immutable Git-bound input,
using captured policy bytes and pinned tool/schema/standard constants. Floor85, critical/high,
standard mode, source/archive/blob/mode checks and every failure decision remain mandatory.
The new repository-policy positive case reproduces failure before the fix; all17 contract tests
pass afterward, including invalid fingerprint/advisory and existing corruption/refusal controls.
A read-only replay of the actual retained CI artifact verifies all13707 files and reports
artifact_valid:true, audit_passed:false, score64/85,204hard findings. The existing owner-approved
below-floor report-only disposition remains separate from the failed audit; no new finding is
waived or marked fixed. New source retains main's existing P37 exception verifier and expiry checks.

## Latest upstream reconciliation

Current main includes the default external RES07 inventory's explicit opt-in guard and isolated
dashboard test evidence/browser fonts. Preserve those authored fixes. The browser successor now
follows the already-merged CPU-affinity and authority-lineage identity transitions, with only its
two metadata fixtures regenerated at b794bf1026c11441fe17495f41249c0fccd234cd. All13 behavioral
compatibility cases and three historical joins pass; numerical fixtures and known identities
remain unchanged. The app's original source-pinned runtime artifact is unchanged.

## Optional classifier ancestor status

Main split PR and manual source classification into two optional jobs and an always-run fan-in. In PR41450376934 the real path plan selected every lane, but all dependent jobs were skipped because their implicit success guard inherited the skipped alternate classifier. Give each of the nine direct changes dependents an explicit !cancelled() status guard plus successful-classifier condition, preserving every original path expression. The regression checks both event routes against the real job graph, reproducing skipped selected jobs before the fix and retaining failure/cancellation refusals afterward. No selected lane is accepted as skipped-by-path. Manual runs use the existing native runner option and default source provenance binding; no admission check is removed.
