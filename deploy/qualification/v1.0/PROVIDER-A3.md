# A3 native POSIX CPU provider

The release coordinator can bind A3 to:

```json
{"id":"A3","argv":["python3","-B","deploy/qualification/v1.0/providers/a3.py"],"timeout_seconds":300}
```

It consumes the five standard `JOPE_REL07_*` variables from GATE-RUNNER.md.
The candidate must be a non-development, production-trust embedded full CPU
package (`fused-cpu`, or the legacy `full` profile), bound to the exact artifact
SHA256 and both compiled/package source SHAs. Independent `jope-dist verify`
checks every byte before the candidate executes. No key or signature is produced.
A missing full candidate remains a refusal, never a fixture substitution.

Six actual checks run against a newly owned native daemon: the daemon record is
owner-only, authenticated system access succeeds, missing/wrong tokens return
catalogued 401 problems, and a foreign Host or browser Origin returns a
catalogued 400 problem. Responses must expose no CORS surface. The record must
identify the exact child PID and numeric loopback address. The provider records
statuses and boolean assertions, without persisting tokens or daemon output.
Only its own child is stopped and reaped; the coordinator retains process-group
timeout control. Evidence counts describe all six checks, and the artifact hash
is checked again before publication. Existing result files are never replaced.

This provider qualifies this local-security criterion on native POSIX CPU builds;
it does not establish Windows ACL security, CUDA qualification or the other27
gates. Use the coordinator's CPU/disk admission and nice10 environment. No
training dataset, network service outside loopback, model evaluation or GPU is
needed. `cargo run -p jope-dist` reuses the checkout's existing Cargo target.

Manual controls:

```sh
nice -n 10 python3 -B deploy/qualification/v1.0/test_gate_provider_a3 -v
```

Those controls use deliberately insecure local fixture children and mocked
release-verifier responses. They test provider sensitivity and refusals; they
are not release candidate PASS rows. The actual release campaign requires the
full signed candidate and all28 distinct artifact-bound gate results.
