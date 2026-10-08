# Target-maker executable artifact authority

Artifact SHA-256 pins bind the bytes loaded by joblib. They do not independently
authenticate a producer: replacing both an artifact and its self-reported pin
must not grant permission to deserialize executable Python state.

Persisted fused bundles now require `trusted_manifest_sha256` from a separately
trusted producer record. The caller must obtain that identity outside the bundle
under test. The consumer snapshots `manifest.json`, verifies its complete digest
before parsing, and admits only the artifact pins in that verified snapshot.
Each joblib load requires this in-memory producer authority and still verifies
the exact artifact snapshot before loading it through `BytesIO`.

```python
encoder = load_fused_encoder(
    bundle_dir, device="cpu", trusted_manifest_sha256=approved_producer_manifest_sha256
)
```

The encode/score/file/evaluation wrappers forward the same keyword. A missing or
mismatched outer pin refuses before any model loader runs. Never derive the
approved root by hashing a bundle that has not already been admitted; that would
restore the original self-authorization problem. This change supplies no signing
service, credentials, keys or automatic producer approval.

Classical training records in-memory authority after writing its own fitted
model and uses it for ensemble reload. A report containing a foreign model path
and a matching self-reported hash is refused. Authority is process-local and
never serialized into the JSON report. Persisted classical reports do not gain
an implicit replay route.

Trusted producer code and its output custody remain prerequisites. Privileged
Python code can construct or mutate module objects; this contract protects
untrusted artifact/manifest data, not hostile code already executing in-process.
The outer trusted root authorizes joblib identities. It does not certify model
quality, row isolation, external producer infrastructure or every neural
checkpoint load route.

Focused acceptance:

```sh
PYTHONPATH=python/src python3 -m pytest python/tests/test_target_artifact_pins.py -q
```

Controls retain good-pin loads, original missing/mismatch and same-byte snapshot
checks. Added controls cover missing producer authority, forged data-shaped
authority, a changed artifact plus recomputed manifest pin against the original
trusted root, foreign classical reports, replacement before manifest parsing,
and propagation through all public fused loader wrappers. No Cargo build is
required. A deployment must supply its independently trusted producer root
before using the persisted fused-bundle route.
