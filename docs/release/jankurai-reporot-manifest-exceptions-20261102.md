# Repo-rot dispositions for pinned engine manifests

The full reports `rerun/j_ffba82986.json` and `rerun/j_27137b5fb.json` record these six HLT-040 findings:

| Fingerprint | Exact path | Current SHA-256 |
| --- | --- | --- |
| `sha256:704e72cf20a080e9408bfb9e1f040eab635cb62ac2ad91c7177422d1882fd08f` | `crates/jope-engine/manifests/hyperion-concurrent-v30.sha256` | `98a2dfff5b5a5a4bd684c213780df590ac8ec2cae76d64ef517608e714604ee1` |
| `sha256:3a7fb9e29819e543732e391777bb33064c93c9a130c22d6dd476b6a9b3a60fe8` | `crates/jope-engine/manifests/opportunistic-proposals-v44.json` | `2e2bb373d11c6c84a7d0c403d36898895a55bdd3a174a4d5b793d59b19ded9a0` |
| `sha256:98b3ec04de237e0203c5c49adfee01b9010bd07508413e4e568130d9e9113812` | `crates/jope-engine/manifests/opportunistic-proposals-v44.sha256` | `dceca18b1528a69913e416571987a3ce2a381f74e015590097b80bbd4ee843c9` |
| `sha256:207ba054c98c30dd37a16ee13f9326b61ba012a008d22c55d64ace08e0ce32a8` | `crates/jope-engine/manifests/opportunistic-proposals-v53.json` | `ccca20be93f74471c55e2f67eb60d011aff7bb8710af2b057ffacff533131c90` |
| `sha256:8d62b1c2766e42b6582260dfb31706d227fb9df1af56c32986dc9fc331a565ff` | `crates/jope-engine/manifests/opportunistic-proposals-v53.sha256` | `75bc8045972ca9204a0cafaa8213bfcc6a036752aee0629ef47cfcbf456fb78b` |
| `sha256:05b639e645eb0fdd1b57c7206423e4a7fa445137d567797770d997434abf0252` | `crates/jope-engine/manifests/champion-import-v44.sha256` | `4977c3d84c5d86baefe89acac30d1df094d71932e59ead3fc1f2d133066bffd9` |

These files are versioned integrity evidence, not parked implementation source. The
engine manifest registry describes `hyperion-concurrent-v30.sha256` as binding the
v30 executable, specification and generated job manifest. It describes the v44/v53
JSON and checksum pairs as pins for recorded proposal cycles and requires migration
by adding a new version. The v30 checksum is consumed by installed service
templates through `sha256sum --check`; the champion-import v44 checksum is consumed
by its systemd `ExecStartPre` integrity check. The v44 and v53 checksum files bind
their respective JSON records and archived service artifacts.

All six source files and their recorded values are preserved byte-for-byte. Only
the six exact paths above are excluded from Jankurai scoring in
`agent/audit-policy.toml`; tests and other security scans continue to include them.
Revisit the dispositions only when the corresponding replay/custody records are
retired or superseded by a new version. This does not exclude adjacent manifest
files, directories, or other findings.
