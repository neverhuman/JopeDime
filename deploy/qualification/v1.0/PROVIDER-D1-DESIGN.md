# REL-07 D1 provider

D1 qualifies two native artifacts built from the same frozen source revision: the signed
Linux x86-64 full candidate and a Darwin arm64 full-profile binary. The artifacts have
different bytes and therefore different SHA-256 digests. The D1 result records each platform's
source, target triple, and artifact digest separately; top-level `artifact_sha256` remains the
Linux signed candidate digest for compatibility with the gate runner. The Darwin digest must be
provided from the signed release manifest, not inferred from an untrusted receipt.

## Native Darwin build and receipt

Run on the isolated Darwin arm64 release host, with a clean checkout at the exact RC2 source SHA
and a new external evidence directory (never `/tmp`):

```bash
nice -n 19 bash deploy/qualification/v1.0/build-d1-darwin-arm64.sh \
  <40-lowercase-hex-source-sha> /absolute/evidence/REL-07/D1-macos-arm64
```

The recipe uses the existing lane Cargo target, `cargo-artifact`, locked dependencies, the `dist`
profile, and exactly `embedded,engine,mcp,client` with default features disabled. It refuses a
wrong host, dirty or wrong-source checkout, reused evidence path, non-arm64 executable, wrong
embedded source/target/features, or unexpected dynamic library. It emits a build record
`darwin-build.json` with the source, target, full feature set, base executable SHA-256, and
successful architecture/source/feature/link checks. This is pre-seal build evidence only; it is
not the final platform package digest, a D1 execution receipt, or a qualification PASS.

The binary is platform-specific and is not a replacement for the signed Linux candidate. Package
and sign it through the owner-authorized RC2 process; the final Darwin package digest must be
included in the same owner-signed manifest as the full and fused-cpu artifacts. The signed Darwin
package must then run through the macOS clean-host leg to produce a
`jope.rel07.d1-platform-leg/v2` receipt before D1 can pass. This recipe does not sign, stage, or
qualify an RC.

## Native DEV-BUILD rehearsal

To exercise D1 before an RC2 exists, build the repository's native development fixture package.
The fixture has a deterministic development-key signature and is never a release-signed
candidate. Run from a clean native arm64 checkout at the exact source SHA; keep all generated
output in an external evidence directory. The existing `test-clean-host-macos.sh --build-fixture`
path builds and verifies the fixture package in the lane's reused Cargo target. Copy the resulting
`jopedime` file and current `d1_macos_dev.py` helper into the external evidence directory, then
run:

```bash
nice -n 19 python3 -B deploy/appliance/d1_macos_dev.py package \
  --source-sha <40-lowercase-hex-source-sha> \
  --fixture-package /absolute/evidence/REL-07/D1-fixture/jopedime \
  --out /absolute/evidence/REL-07/D1-dev-package
nice -n 19 python3 -B deploy/appliance/d1_macos_dev.py run \
  --candidate /absolute/evidence/REL-07/D1-dev-package \
  --out /absolute/evidence/REL-07/D1-dev-run
```

`package` requires `--repo-root <checkout>` before the subcommand when staged outside the
source checkout. It verifies the native arm64 package's embedded source identity and development-key
trust, then labels the copied package `DEV-BUILD` with `release_signed: false` and
`development_signed: true`. `run` executes network-denied discovery, synthetic local fit,
prediction and non-empty Python, Rust and JavaScript exports with no toolchain on `PATH`. Its
v2-shaped receipt is labeled `DEV-BUILD`, has `qualification: NOT_QUALIFIED`, and is rejected by
the signed-candidate provider (`providers/d1.py` requires `status: PASS`). It does not sign or
qualify RC2.

## Aggregate invocation

Pass the external Mac receipt and both independent bindings in the release gate command manifest:

```text
python3 -B deploy/qualification/v1.0/providers/d1.py \
  --macos-receipt /absolute/path/D1-macos-arm64.json \
  --macos-receipt-sha256 <sha256-of-receipt-bytes> \
  --macos-artifact-sha256 <darwin-artifact-sha-from-signed-manifest>
```

Direct invocation may use `JOPE_REL07_D1_MACOS_RECEIPT`,
`JOPE_REL07_D1_MACOS_RECEIPT_SHA256`, and `JOPE_REL07_D1_MACOS_ARTIFACT_SHA256`. The provider
rejects a receipt with the wrong schema, source, target, profile, feature set, artifact binding,
or any missing/false clean-host assertion. It verifies the signed Linux candidate independently,
runs `deploy/appliance/test-clean-host-prestage.sh` with `--require-golden-path`, and writes a D1
PASS only if both platform legs pass. A bad, missing, stale, or mismatched leg never writes a
result.
