# Optional landmarks: closeout verification checkpoint

Code source `6b396365d7b0663501d6694c4ffc2167087e484b` adds two narrow repairs while
retaining published Dope `2535a36c`. Exact source/log hashes and outcomes are in the
[execution record](optional-landmarks-closeout-checkpoint.json).

Read-only native proof retention again refuses a symlinked immediate parent, using metadata
without creating, staging or syncing files. The regression fails before the fix; all 19
native-graph tests pass afterwards. Public predecessor validation already refused an existing
parent symlink. This restores the retention-time helper invariant across awaited work; it does
not establish an atomic path-race defense or confer execution/training authority.

The full local gate at `16fcf9ef0` passed hygiene, security, formatting and workspace Clippy,
then found eight actual GP compiler-probe assertion failures with `RUST_TEST_THREADS=1`.
Pretty libtest output prefixes the first numeric record with its test status. The same retained
binary produces 119 parsed values in that mode and all 120 exact expected bit patterns in terse
mode. Probe execution now requests terse output. Emitted model bytes, parsing, exact fault/value
counts and numerical tolerances are unchanged. The first shard's eleven programs pass against
the native reference through Rust, Python and both strict C++17 compilers.

The already-failed gate was gracefully interrupted: 946 passed, eight assertions failed, two
tests were interrupted and 2780 were unrun. Its actual exit was 100. That is failed evidence,
not an aggregate pass. The complete required gate and fresh exact-head CI remain pending at
this dated checkpoint; old-head green CI cannot qualify a changed head.

The production flywheel gap is now mapped more precisely. The trainer emits v2 and lacks a
validated successor context and fresh publication fence. A v3 producer must retain immutable
exact pre-publication journal prefixes. Its checkpoint binds consumed/predecessor lineage;
ACKs, committed credit and resulting lineage follow publication to avoid self-hash cycles.
Canonical I13 commit evidence, numeric-epoch/native-epoch mapping, common attempt/window IDs,
GlobalPolicyTraining exposure and actual consumption of compatibility pins still need explicit
authenticated bindings. No permissive verifier or dormant success-shaped emitter was added.

Device accounting remains explicit: table preparation, native landmarks and conditioning
assembly run on CPU; the pretrained forward and optional Jope fusion/context run on CUDA.
The coordinated upstream proposal to omit discarded output heads must preserve the whole
checkpoint and existing APIs, prove first-512 parity and show a measured benefit before adoption.
It does not establish GPU landmark computation or satisfy the cold-latency gate.

No second acquisition, training campaign, key/grant/adoption, service, production root, governed
distribution or release was performed. Trained registration, both-consumer predictive quality,
the final trained package, matched latency and authenticated continuous flywheel health remain
separate exits in the [completion plan](../optional-landmarks-hpc-plan.md).
