# REL-07 CPU gate preparation

This is the owner-approved gb2112 CPU step for spec §§11 and14. It runs source
diagnostics on one frozen origin/main commit and writes qualification_receipt.json.
The receipt always leaves all28 release gates and Q1 not-qualified, the signed
candidate and OWNER approval null, and release_qualified false. A successful
unsigned CPU run does not approve release bytes, learned guidance or a platform.

The design reuses each full CPU family's command from deploy/ci/coverage.json.
The literal full-main.py GROUPS must partition those families exactly. Independent
source hygiene, formatting, workspace check, Clippy, Python, audit, path and
dependency checks also run. A failing gate leaves its original exit and the next
gate still runs. No KI failure is silently changed to PASS.

## Plan and execute

Fetch origin/main first. Use an owned, clean main checkout and one reused target.
Planning accepts a feature checkout and records that it is not clean at main;
execution refuses it. The driver has its own SHA256 in addition to the tested
source/tree and command-input hashes.

    python3 deploy/qualification/v1.0/cpu_gates.py plan \
      --repo "$OWNED_MAIN" --expected-main-sha "$MAIN_SHA" \
      --output "$NEW_EXTERNAL_PLAN"

Expensive execution goes through the host owner's remote_test.sh auto dispatcher.
Provision the committed Cargo external entry point on eligible Linux CPU workers; a
symlink into the worker's Cargo executable search path can point to the committed
cargo-qualification-cpu file. The selected worker must have the explicitly named
owned main checkout. A missing or stale checkout refuses before gates run.

    nice -n 10 ~/bin/remote_test.sh auto "$MAIN_SHA" qualification-cpu run \
      --repo "$WORKER_OWNED_MAIN" --expected-main-sha "$MAIN_SHA" \
      --output "$NEW_EXTERNAL_EVIDENCE"

The Cargo entry point accepts the dispatcher's appended -j argument and caps
actual child jobs at8. It executes no Rust build itself. CUDA visibility is empty,
only basic locale and toolchain paths are forwarded; ambient credentials, fixture routing and GPU settings are not inherited. Math threads are1, incremental is0, and dev debug is line-tables-only. It checks
CPU load<=85% and disk use<95% before each command; declared family timeouts still
apply and the controller caps an owned process group at30min by default.
The script never acquires qualification.lock or probes/allocates a GPU.
Do not run an expensive matrix directly on the shared coordinator.

## Receipt and acceptance

The JSON Schema is qualification_receipt.schema.json. The receipt records:

- full origin/main, HEAD and tree hashes, driver and gate-input hashes;
- machine/Python identity, bounded environment, exact argv, original exit,
  timestamps, wall time, diagnostics and CPU/disk admission;
- collected/executed/passed/failed/skipped/excluded counts, with unknowns null;
- regular output lengths/SHA256, including command log, JUnit and collection;
  evidence symlinks are listed and never followed;
- every CPU gate, including unavailable, timed-out or failed gates.

Existing results.py parses execution summaries and cross-checks JUnit. Nextest
also needs a matching fresh collection. Zero or unknown test execution fails;
skipped/excluded coverage remains incomplete. Output directories must be new;
source HEAD, main ref and dirty state are rechecked after all gates. The receipt
is incrementally retained if execution is interrupted.

Focused controller acceptance (stdlib children, no matrix/Cargo/GPU jobs):

    python3 -m unittest discover -s deploy/qualification/v1.0 \
      -p test_cpu_gates.py -v

Tests exercise actual successful and failing child processes, timeout continuation,
zero/missing/skipped evidence, source/output refusal, inventory drift, complete
schema validation and the schema's refusal to promote a CPU result to a release.
Set JOPE_QUALIFICATION_TEST_TMP to external custody if desired; defaults reuse the
lane target and never put Git fixtures or build output under /tmp.

Signing, signed-byte gate reruns, native clean-host matrices, named client runs,
reference-laptop measurements, Q1 and OWNER release approval remain future work.
This PR does not claim that the complete CPU matrix has already run.
