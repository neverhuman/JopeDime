# REL-07 D1 provider

The provider binds both supported clean-host legs to the same signed candidate and source
revision. It runs the Linux x86-64 golden-path harness locally, then consumes the native macOS
arm64 receipt produced on xbabe4. Windows is waived by the owner. A result is written only
after both legs pass; a missing macOS receipt is reported as `BLOCKED`, while a bad hash,
binding, assertion, or platform run is a failure.

## macOS receipt input

Pass the external receipt path and its independently transferred SHA-256 digest as command-line
arguments in the complete release-gate manifest:

```text
python3 -B deploy/qualification/v1.0/providers/d1.py \
  --macos-receipt /absolute/path/d1-macos-arm64.json \
  --macos-receipt-sha256 <64-lowercase-hex>
```

The receipt must be an absolute regular non-symlink JSON file with schema
`jope.rel07.d1-platform-leg/v1`, `gate_id: D1`, `platform: macos-arm64`, `status: PASS`, and
the exact `source_sha` and `artifact_sha256` supplied to this provider. Its `checks` object must
set `network_disabled`, `toolchain_absent`, `discovery`, `prediction`, and
`link_audit_os_libraries_only` to `true`, with `exports` exactly `{python: true, rust: true,
javascript: true}`. The supplied digest must match the receipt bytes. The provider records the
receipt digest and candidate bindings in its result.

Direct invocation may use `JOPE_REL07_D1_MACOS_RECEIPT` and
`JOPE_REL07_D1_MACOS_RECEIPT_SHA256` instead. `release-gates` gives providers a restricted
environment, so its command manifest must carry these values as argv options.

The expected receipt shape is:

```json
{
  "schema": "jope.rel07.d1-platform-leg/v1",
  "gate_id": "D1",
  "platform": "macos-arm64",
  "status": "PASS",
  "source_sha": "<full source SHA>",
  "artifact_sha256": "<candidate SHA-256>",
  "checks": {
    "network_disabled": true,
    "toolchain_absent": true,
    "discovery": true,
    "exports": {"python": true, "rust": true, "javascript": true},
    "prediction": true,
    "link_audit_os_libraries_only": true
  }
}
```

The Linux leg uses `deploy/appliance/test-clean-host-prestage.sh` with
`--require-golden-path`; the provider independently checks its receipt for candidate/source
identity, loopback-only network isolation, toolchain absence, discovery, all three run exports,
prediction rows and the fused-CPU OS-library link audit. Although that harness labels its raw
receipt `pre-stage`, the provider emits D1 PASS only after its stricter candidate signature
verification, full Linux assertions and matching macOS receipt all succeed. No candidate,
signing action, protected dataset, GPU, or hosted CI is part of the provider controls.
