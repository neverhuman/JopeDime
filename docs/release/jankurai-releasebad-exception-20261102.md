# Jankurai release behavior test exception

**Owner:** Jepson, via the Grok Bot P41 assignment (2026-11-02). **Auditor:** Jankurai 1.7.6.

`deploy/release/test-release.sh` is an authored test harness, not a release publisher. Its
synthetic bare-remote control deliberately attempts a normal, non-forced tag push against an
existing conflicting tag and asserts that the push fails and the original tag object remains
unchanged. The release publisher separately refuses a tag already present locally or on origin
before it creates a tag, and `push_release_tag` has no force option. The release behavior finding
on this test-only command is therefore a false positive. The exact-path exception suppresses only
Jankurai scoring for this test file; release tests and other security checks still run. Remove the
exception if the fixture or publishing behavior changes.
