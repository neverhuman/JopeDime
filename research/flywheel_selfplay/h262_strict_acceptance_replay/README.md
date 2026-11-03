# H262 — Strict acceptance replay

**PREREGISTERED; no outcomes read.** Test whether a paired family-bootstrap
acceptance margin improves a final offline adapter incumbent over r882 and the
observed live comparator, which accepts ties. Three seeds, ten rounds per rule,
40 H170 CONTROL updates per round; 2,400 primary updates, two smoke updates.

See the [frozen protocol](PREREG.md), [source audit](current_rule_source.json),
[split](fold_manifest.json), [schedule](schedules.json), [manifest](manifest.json)
and [metrics](metrics.json). DL-Val is excluded from all acceptance decisions;
all final checkpoints lock before six final trained-policy evaluations.

CURRENT and STRICT_LCB share round-indexed random streams; NEVER retains r882.
A secondary-only win is an acceptance-rule finding, not checkpoint lift. The
final-only scoring limit permits trajectory-level accept-then-final-worse rates;
per-round DL-Val harm remains unmeasured. TOP4_PICK is descriptive test-time
search. No serving promotion or live-loop modification.

CPU48,50-52 on xbabe2, nice19, one numeric thread, RSS4GiB/A1/load guards.
Prereg16:40Z, compute stop18:15Z, closeout19:00Z. Publication is v1/next only;
hosted CI is deferred under the owner split.
