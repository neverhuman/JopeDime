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

Only the exact active root path is excluded from Jankurai scoring in
`agent/audit-policy.toml`. This does not change runtime code or other scans.
Revisit the exception if the old-service check is removed or the source identity
changes; any other path or fingerprint needs separate review.
