# Variety-b exact-path disposition

The full audit `rerun/j_ffba82986.json` reports HLT-001 fingerprint
`sha256:f73fa835aa14978110912c4b5f983a4c5d931f11e2d12d2c3cb903ef8ac99980`
at `deploy/dl-flywheel/pipelines/telemetry/trace_archive.py:304`. The word
`old` labels the prior `jopedime-hpc-archive-20261009-v1` service whose stopped
state the code verifies (`MainPID=0` and an empty control group). It describes
the service being checked; it is not a dead-code or stale-source marker.

The source is byte-identical (SHA-256
`c8a13033c6a93bb97d14d3ccb4de0d2fae01be0928a75f654cfd79f32d835c69`) to
`trace_recovery_v2/trace_archive_v1_c8a13033.py` and
`trace_recovery_v3/trace_archive_v1_c8a13033.py`. The v2 provenance validation
records this source hash; v3 `SOURCE_PACKET.json` and `validation.json` pin it.
Those versioned copies and all provenance remain unchanged.

The active root source now names the value `retired_unit_status`, preserving
the `MainPID=0` and empty-control-group checks while describing the actual
`systemctl show` result. The root path remains in the Jankurai scan.

The full audit also reports exact-copy HLT-043 fingerprint
`sha256:3aef41a1a406561ee6e024b8e2c1f8b163bce0d2f215e8ba86d29fea6e3e70ae`
for the root and versioned files. All three original files had SHA-256
`c8a13033c6a93bb97d14d3ccb4de0d2fae01be0928a75f654cfd79f32d835c69`. Only
the two versioned paths are excluded from Jankurai scoring: v2 is pinned by
`config.template.json`; v3 is pinned by `SOURCE_PACKET.json`, `validation.json`,
and versioned provenance. The source bytes and provenance remain unchanged.
The policy exclusions are exact-path only and do not alter Python tests or
other security scans. Revisit them when those frozen records are retired or the
pinned hashes change.
