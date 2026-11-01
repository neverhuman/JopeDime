# REL-07 S2 native determinism provider

S2 runs the maintained ignored service test
`discover::r#loop::tests::real_native_workers_preserve_projection_across_capacity_pause_and_sigkill`.
The test executes the same fixed-plan native discovery with one and eight live
workers, pauses and resumes an in-flight wave, and SIGKILLs and recovers an
in-flight run. It compares the champion/model identity and full durable event
projection at each recovery point and requires zero limit-hit outcomes. This
matches the S2 criterion in spec §14 and the deterministic projection defined
in §5.6.

Before running the assertion, the provider verifies the supplied regular
candidate's full, non-development envelope and embedded source SHA with
`jope-dist`, and requires the working source checkout and candidate bytes to
remain pinned throughout the test. It records PASS only for exactly one
executed, passing test with no skips, failures or exclusions. The coordinator
retains the command log and the provider result under the standard S2 gate ID.
The provider first builds `jope-service --all-targets` because the test links
the real native worker against that library; a failed build stops before the
test and cannot produce a result.
Because the assertion includes native SIGKILL and recovery, the provider refuses
platforms other than Linux and macOS instead of silently omitting that clause.
Its Cargo and test child processes use a private temporary directory under the
reused checkout `target/`, so the test's generated worker executable and runtime
files do not go under `/tmp`.

The focused provider controls use synthetic candidate bytes and mocked child
processes. They verify command selection, source/artifact binding, exact test
counts, failure propagation and refusal of zero, skipped or changed-source
results. Those controls are not candidate-bound S2 evidence. No candidate
qualification is claimed until the coordinator runs this provider against an
approved full candidate.
