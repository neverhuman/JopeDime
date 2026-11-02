# HLT-027 HWM comment false positive

The full audit `j_161a8216d` reported
`sha256:7985f857419f6e5c3a56385852c62b79490492fea81d320002f5b8f7533ea211`
at `crates/jope-learning/src/hwm.rs:276`. At audited source commit
`161a8216daa3d257ca2e6d27a89cb725f098e385`, the file SHA-256 is
`c1e88702c0413087f157f501608258731a86006c2ba8e0e4a226861ae29fee83`, and the
reported line SHA-256 is
`2987377b8566681e7470f41fefc58b8380afa6516e232bd82f9fac5673a70a32`.

The finding points into the documentation for `refuse_unrepresentable`. That
comment explains that non-finite scores cannot be serialized and that replacing
one with a finite score would fabricate measured data. It makes no claim that a
human reviewed the code or that external proof exists. This is the same detector
failure documented for the file-reader comment under the HLT-027 `io-comment`
disposition in `docs/release/jankurai-v1.md`.

The policy exception is restricted to this exact Rust source path because the
auditor currently excludes paths rather than individual HLT-027 fingerprints.
The source and its behavior are unchanged. Revisit the exception if the auditor
adds fingerprint-scoped dispositions or if the comment becomes a review claim.
