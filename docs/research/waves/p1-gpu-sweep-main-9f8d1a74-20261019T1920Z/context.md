# Wave `p1-gpu-sweep-main-9f8d1a74-20261019T1920Z` (`v1`)

## Provenance & S3 CAS Archive Locator (`592c4e2d`)
- **Bucket**: `veox-jopedime-learning-use2`
- **Key**: `staging/fleet-rnd-20261018T1140Z/evidence/blobs/sha256/bc6ad1688b4d118d2153d6224d53eec39cadcc1d8295d98fff48b4bbcb7f51da`
- **URI**: `s3://veox-jopedime-learning-use2/staging/fleet-rnd-20261018T1140Z/evidence/blobs/sha256/bc6ad1688b4d118d2153d6224d53eec39cadcc1d8295d98fff48b4bbcb7f51da`
- **VersionId**: `0r8bziZHXOZXBQ0mE7SGUr6a5gFZtJxs`
- **Compressed Bytes (`tar+zstd`)**: `231,127` (`sha256: bc6ad1688b4d118d2153d6224d53eec39cadcc1d8295d98fff48b4bbcb7f51da`)
- **Decoded Tar Bytes**: `19,097,600` (`sha256: 5ee8b74612dce9ee4d80ff3f669646dfde3577dd9ba455fc0f0313fd9ae5ca28`)
- **Verified Regular Members**: `163` files (`154` `pass_*.json` documents + `9` logs/manifests), totaling `18991580` bytes.
- **Deduplicated Binary & Cases Reference Archive**: `s3://veox-jopedime-learning-use2/staging/fleet-rnd-20261018T1140Z/evidence/blobs/sha256/0d91b397ea6912e879d7ae04a135c2f30781c478e38c9cac2c380d4aa4e37df4` (`VersionId: CCdA4QL1zQqspQyblWJH0_YbqfI3V91S`, `1,002,870` B).

## Classifier & Mixed-Binary Identity Disclosure
1. **Classifier Correction**: The initial preservation script in `592c4e2d...` checked nonexistent keys `cases` and `envelope_ok`, falsely marking all `154` `pass_*.json` files as incomplete. All `154` `pass_*.json` files have `"rows": [9 cell objects]` and `"envelope_verified": true`.
2. **Mixed Binary Provenance**:
   - **25 documents on `xbabe1`** (`pass_01`..`pass_25`) declare `executable_sha256: e39bca1917ba1e26ad74c02f64fe8d0ebcf4c175ecabb096cfff20d24c809490` (built from `main` `9f8d1a74ab08fc200214a82beaf2ee96e316061a`).
   - **129 documents** (`49` on `xbabe1`, `6` on `xbabe2`, `74` on `xbabe3`) declare `executable_sha256: c20a2dc8917c755f02b94049356320fe1f62ac7d9f8994731f807d9df39302da`, whose source commit qualification is unproven. Do not claim `c20a2dc8...` outputs as `e39bca19...` runs.
3. **Strict Scope Boundary**: Non-neural GP CUDA kernel verification (`9`-cell `--verify-envelope`) only. Zero neural updates, zero learner exposures, and no promotion authority.
