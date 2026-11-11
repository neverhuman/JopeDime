# Carbon Zero-Knowledge & Security Red-Team Receipt

- **Target Endpoint**: `https://lchlyzekra3i75jvurq3msfnde0jlcfl.lambda-url.us-east-2.on.aws`
- **Region**: `us-east-2`
- **Timestamp (epoch ms)**: `1794415028304`
- **Coverage**: `16/16` adversarial vectors passed
- **Suite Verdict**: **PASS**

| Vector | Finding | Attack Scenario | HTTP | Wall (ms) | Verdict | Evidence |
| --- | --- | --- | ---: | ---: | --- | --- |
| `V-01` | `RT-02` | Unauthenticated upload prepare rejected | `401` | 63.3 | **PASS** | POST /v1/carbon/uploads/prepare without token -> HTTP 401 |
| `V-02` | `RT-02` | Forged capability HMAC signature rejected in constant time | `401` | 63.0 | **PASS** | Tampered ctk1 signature -> HTTP 401 |
| `V-03` | `RT-02` | Free-to-Pro capability tier escalation refused | `403` | 65.7 | **PASS** | Free token requesting tier=pro -> HTTP 403 |
| `V-04` | `RT-02` | Cross-tenant IDOR access refused | `403` | 65.2 | **PASS** | Token B reading Tenant A run -> HTTP 403 |
| `V-05` | `RT-08` | Path-traversal & non-injective tenant_id rejected | `400` | 137.0 | **PASS** | tenant_id='a..b' -> 400, '..%2Fsecret' -> 400 |
| `V-06` | `RT-04` | Presigned PUT byte-size / SHA-256 inflation rejected | `403` | 332.7 | **PASS** | Inflated PUT (+64B) to size-pinned URL -> HTTP 403 |
| `V-07` | `RT-01` | Cleartext upload refused on zero-knowledge cloud cell | `400` | 64.9 | **PASS** | encrypted_cuen=false -> HTTP 400 |
| `V-08` | `RT-01` | CUEN payload slot-swap (HoldoutJmat -> LandmarkJlmb) rejected | `400` | 72.3 | **PASS** | Wrong CUEN slot tag in landmark_bundle_b64 -> HTTP 400 |
| `V-09` | `RT-01` | Tampered CUDK v1 wrapped U_DEK envelope rejected by engine | `502` | 193.5 | **PASS** | Flipped GCM auth byte in CUDK -> HTTP 502 |
| `V-10` | `RT-03` | Duplicate run_id state clobber refused via If-None-Match | `409` | 89.5 | **PASS** | Second POST /v1/carbon/runs for run_01M75QGHYM212XXRY31D2C118M -> HTTP 409 |
| `V-11` | `RT-13` | Oversized (>16 KiB) landmark wire payload refused at edge | `400` | 124.4 | **PASS** | 20,000-byte landmark_bundle_b64 -> HTTP 400 |
| `V-12` | `RT-01` | Cleartext /predict rows refused on encrypted run | `400` | 90.0 | **PASS** | Plaintext rows on encrypted run -> HTTP 400 |
| `V-13` | `RT-01` | Zero-cleartext E2E /predict wire inspection verified | `200` | 2041.3 | **PASS** | Raw JSON predictions.len()=0, client ECDH decrypted 8B matching sha256 3173c14d2054bf7c |
| `V-14` | `RT-10` | Free-tier per-run /predict model-extraction quota enforced | `429` | 4208.4 | **PASS** | 21st /predict call on Free run -> HTTP 429 |
| `V-15` | `RT-11` | Spot-hybrid advisor telemetry restricted to Pro/Admin | `403` | 122.3 | **PASS** | Unauth -> HTTP 401, Free -> HTTP 403 |
| `V-16` | `RT-06` | Cryptographic DELETE shred purges blobs/state & blocks resurrection | `404` | 606.3 | **PASS** | deleted=7 objects, post-shred GET -> 404, recreate -> 409 |
