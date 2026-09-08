**The next deliverable is a verified source mirror plus repeatable compatibility
and research checks. It does not replace the running v44 campaign.**

Every subsequent change should create a new content-derived bundle containing the
runtime source, complete vendored kernel, helper dependencies, these instructions,
probe manifests and existing evidence. Mirror that bundle to
`/home/ubuntu/jope-dime/parallel/kernel-work/<bundle-id>` on
`ubuntu@158.101.20.226`. Never synchronize development files into the active remote
`src`, `assets/source`, `tools`, `bin`, vocabulary, corpus or campaign directories.
The staged source can be reviewed and built while the old executable routes remain
unchanged. Original datasets, historical artifacts and checkpoints are immutable.

The machine-readable acceptance definitions are in
[`verification-gates-v1.json`](../research/algorithm-kernel-regimes/verification-gates-v1.json).
Each verification receipt binds the bundle, source release, gate manifest, compiler,
commands, logs and outcome. Deferred work is recorded as deferred; no partial
profile qualifies deployment.

| Action/change | Executable verification | Pass condition |
|---|---|---|
| Freeze and mirror the complete work | `kernel_bundle.py pack/verify`; `kernel_mirror.py` | Local and remote file hashes agree; unexpected files and symlinks reject verification; live service invocation, source, route, fixed manifest and vocabulary identities are unchanged |
| Preserve historical behavior | `kernel_verify.py --profile full` | Existing freeze and all golden tests pass without changing expected hashes |
| Verify safety repairs and exact values | `kernel_verify.py --profile core` | Invalid clip/missing-column cases preserve whole-expression fallback; historical unsafe C++ is regenerated before loading; emitted Python/Rust/C++ execute; sanitizer cases pass; typed values retain their bits |
| Verify scalar evolution | Included in every profile | 100,000 expression children and 100,000 pipeline children pass their scoped structural/width checks; compiled expressions match interpretation; this is not yet full task/effect/resource admission |
| Verify useful approximation counterexamples | Included in every profile | Exact sensitivity/abstention fixtures replay; all 40 actual-learner MSE values reproduce under the same frozen context; dropped, duplicated, nonfinite or context-mismatched observations fail |
| Verify node protection | Scheduler cancellation/timeout tests plus before/after node snapshots | Reclaim kills our entire subprocess group; the node-wide lock releases; no new GPU execution; no active service restart or executable/source change |
| Verify observed node health | `kernel_remote_state.compare_health` in every new mirror receipt | Fresh GP/cycle/hardware telemetry, no new failed views, no backwards cumulative counters within the same invocation, no hardware errors, and no queued running GP with zero active workers; missing evidence is inconclusive |

Run these from `/home/ubuntu/jope-dime` (substitute the printed bundle path):

```bash
rtk proxy python3 -B -m unittest discover -s parallel/tools -p 'test_kernel_*.py' -v
rtk proxy python3 -B parallel/tools/kernel_release.py sync-mirror
rtk proxy python3 -B parallel/tools/kernel_bundle.py pack --repository . --output parallel/kernel-work
rtk proxy python3 -B parallel/tools/kernel_verify.py --bundle <bundle-path> --output <fresh-local-check-directory> --profile core
rtk proxy python3 -B parallel/tools/kernel_verify.py --bundle <bundle-path> --output <fresh-local-golden-directory> --profile full --command-timeout 1800
rtk proxy python3 -B parallel/tools/kernel_mirror.py --bundle <bundle-path> --receipts <fresh-mirror-receipt-directory>
```

The mirror tool uses the supplied SSH key by default. It uploads into a unique
incoming directory, checks bytes against the local manifest, then publishes by
rename. An existing identical bundle is verified and reused. It never replaces
an existing bundle or edits a systemd unit. It records live state before and after,
and stores the preservation evidence on both nodes. If another actor changes a
protected identity during staging, it reports drift and does not silently undo
their work.

The mirror command also requires the operational-health check to pass. Both the
source-preservation result and health evidence are copied before it reports an
unsuccessful health check. Healthy counters over this observation interval do not
establish a throughput or model-quality result. A service invocation change makes
counters incomparable; a counter reset must not hide failures.

On the remote node, use the verifier from the staged release:

```text
python3 -B <remote-bundle>/runtime/<release-id>/source/tools/kernel_verify.py
  --bundle <remote-bundle>
  --output /home/ubuntu/jope-dime/parallel/runs/kernel-verification-<unique-id>
  --node-root /home/ubuntu/jope-dime/parallel
  --profile smoke
```

This enters a transient system service with a one-CPU quota, 6 GiB memory cap,
no GPU devices and an 1800-second lifetime. It obtains the existing spare-CPU
lease, uses one spare CPU and polls campaign admission while its child runs. A
phase transition, stale required telemetry, or resource reclaim stops its own
process group and yields a deferred receipt. Shared source is read-only; only its
fresh output directory, scheduler lock and Cargo cache are writable. Cargo runs
offline with locked dependencies and a separate target directory. No live queue
receives test jobs. Full performance tests wait for exclusive scheduled resources.

**Research changes that need a separate experiment before adoption.**

1. Add a rejected-candidate audit to a new isolated search profile. Select 10%
   reproducibly using a separate RNG keyed by evaluation identity, without
   advancing historical search RNG. Save inclusion probability, candidate and
   context identity, screen decision, actual learner result and total cost. Audit
   all learner strata, including far-below-cutoff candidates. Estimate missed
   improvements using the known sampling probabilities; do not infer a global
   rejection rate from only candidates near the cutoff. Keep exploration
   promotions and evaluate the actual learning procedure at the first meaningful
   rung. Do not change the production screen based only on the synthetic probe.
2. Extend the task generator with mechanism holdouts: identical feature and target
   marginals but different interactions, near-singular directions, rare decisive
   effects and diffuse dependencies. The existing linear/product probe does not
   match target marginals and cannot pass this stronger gate. Compare ordinary
   context features and algorithm selection before adding a new regime encoder.
3. Measure complete cost against a frozen control with ten paired seeds under
   both equal evaluation and wall-clock budgets. Include encoding, code
   generation, compilation, diagnostics, audits, correction and fallback. Keep
   per-task results and incumbents. Throughput may degrade by at most 5%; the upper
   paired family-level 95% bound on relative primary-loss degradation must be at
   most 1%, with the original 0.005 absolute margin for bounded accuracy/F1 losses.
   The expression microbenchmark does not pass this gate. Include small batches
   so expression compilation overhead is visible.
4. Introduce an approximation only with a reference identity, preserved quantity,
   formal-versus-empirical evidence label, diagnostic input contract, abstention
   condition and fallback identity. A fit/schema check is not a quality
   certificate. Unavailable evidence must not become a fabricated fitness score;
   diagnostics must not read held-out targets. Keep final acceptance independent
   of the adaptive proposal and verification policy.

**Activation remains a later gate.**

Source parity and smoke tests establish a safe mirror, not a safe runtime upgrade.
The incomplete programmable core, structured tasks, shared backends, exact neural
literals/corpus/checkpoint migration and full admission contracts remain listed in
the implementation status document. Root application integration also needs the
missing `router_contracts_v2` source reconciled. Do not invent that module or remove
the caller to manufacture a green build.

After all required gates pass, verify the current campaign's terminal receipt and
saved checkpoints, stop further old-format producers and drain its finite queue.
Create a fresh namespace and compatible state, switch every executable route
together, and demonstrate identity-based rollback. Preserve the original 8 → 32 →
104 worker stages with 100 successful jobs each, actual eight-GPU rank recovery,
and a new 24-hour integrated qualification. An observed counter or service restart
is not evidence that a queue drained or that this release passed qualification.
