# Browser prediction merge: inherited CI repairs — 2026-11-02

Manual full CI41383543203 at fa98944 failed three checks already broken on its base. No failed lane is claimed as passed. These repairs retain actual execution, scientific identities and existing security boundaries.

## Imported source custody

PR1702 and subsequent audit cleanups renamed variables or rewrote equivalent expressions in nine frozen producer files. The existing byte-for-byte verifier correctly refused them. Restore the eight production files EXACTLY to their unchanged producer bindings. A ninth, stale source-closure assertion needs the identified successor correction below. No numerical fixture/tolerance or production computation changes. These are historical imported sources, not new producer authorization. The general heuristic auditor's variable-name/empty-fixture findings are exact-file false positives here. Only the nine files below are excluded from that heuristic; their stronger hash/syntax/custody checks and Gitleaks remain active. The exceptions last only while these bytes remain frozen and this heuristic applies.

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

## Authenticated CLI HTTPS (A24)

PR1659 introduced the explicitly requested cloud login and model sync. Its mandatory TLS dependencies violated the older no-TLS appliance fence. Make rustls optional and select it plus ureq TLS only through the existing client feature. The no-default-feature and inference-only GPU selections retain featureless ureq and no TLS/native-crypto closure; attempting HTTPS without client fails explicitly. The normal client keeps its verified HTTPS behavior and the existing workspace AWS-LC provider, with no second provider.

The dependency policy retains global denies and adds only exact version, exact root-selection, capped-feature exceptions for ureq3.5.0, rustls0.23.46, aws-lc-rs1.18.1/aws-lc-sys0.45.0 and their three build helpers. Native permission is separate and these helpers also have exact package caps. Empty root selection is represented explicitly by `[[]]`, matched as an exact set, never a wildcard. All other root selections, versions/features, research frameworks, C++ predictor and CUDA bans remain. The test asserts both connected defaults and minimal no-TLS boundaries; existing mutation controls reject new versions/features and native permission cannot be granted by a package exception alone.

This does not qualify an official release, production deployment or every model capability. The browser runtime remains a standalone prediction-only component with no host imports or fitting exports.

## Local authoritative base

This host retains an older forge `origin/main` alongside authoritative GitHub `github/main`. Add an explicit `JOPE_CUSTODY_BASE` argument for the local custody-additions comparison; `JOPE_CUSTODY_BASE=github/main just required` checks the same real PR base as hosted CI. Defaults and hosted base selection remain unchanged. No protection, file-size threshold or custody refusal changes. The direct check against GitHub main passes; the old local origin compares unrelated historical custody imports and correctly refuses them.

## Identified source-closure correction (no numerical change)

After restoring the imported bytes, the original P3 receipt assertion fails because its expected dependency list omits jtt1.py and quarantine.py, which the original static-import closure already includes. The earlier audit cleanup had corrected the assertion without a custody successor, causing the original hash refusal. Preserve the prior current manifest byte-for-byte under `predecessor/source-closure-correction-20261102/`, retain the original test bytes in historical/, and create successor `9afa6c81dc7e4ce0df2d45b76b7d871fe0ea5f660c932f4bb1e6f4553ca49ef7` that changes ONLY this stricter expected dependency enumeration. The verifier retains the full predecessor chain, and its tests require177 current/285 historical with the old177-file closure reconstructible. No producer source behavior, rows/labels/splits, numerical test or tolerance, host exclusions or offline test counts change. This is a separately identified source correction with reproducing failure and passing strict assertion, not blanket repinning to make edited sources pass. See SOURCE_CUSTODY.md and the manifest source_correction record.
