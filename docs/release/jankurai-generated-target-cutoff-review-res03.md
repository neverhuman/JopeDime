# Generated-target cutoff SQL audit disposition

This records a narrow review of the `HLT-033-PYTHON-BAD-BEHAVIOR` finding on
[`cutoff.py:78`](../../deploy/dl-flywheel/pipelines/generated_target_admission_v1/cutoff.py#L78) from
the verified repository audit at source `8466bdefda2d0235e864d407fa053f20bb0e9b4b`.
The finding fingerprint is
`sha256:7c89da7c2fd186a17da55da646549baaa774c20f88db59c330b3bbffd69ea3a4`.

## Review

The reported statement is:

```python
for table, bound in [('events', max_events), ('emissions', max_rows), ('batches', max_rows)]:
    row = db.execute(f'SELECT count(*),coalesce(sum(length(body)),0) FROM {table}').fetchone()
```

`table` can only take one of three string literals in this function. `bound` is
an integer bound and is not interpolated into SQL. Neither value is read from
the database, caller input, environment, or a file. The database is opened as
an immutable read-only snapshot before this query; missing tables or invalid
schemas fail the cutoff instead of changing the selected identifier. SQL
parameters cannot bind identifiers, and an identifier allowlist would repeat
the existing literal-only constraint without reducing an untrusted input path.

The finding is therefore a static-pattern false positive for SQL injection in
this call. No scanner rule is suppressed and no source change is made. The
source file is pinned by the package's [`SOURCE_PINS.json`](../../deploy/dl-flywheel/pipelines/generated_target_admission_v1/SOURCE_PINS.json): `cutoff.py` is
10,540 bytes with SHA-256
`bbc939e79b5a37b2afd0395a470771cab6dbe826dd4441f2a62a5821f7556490`;
`SOURCE_PINS.json` has SHA-256
`045c8f5838920d323dcc3ae8d6ae1a44baf7cdeb629d0e062e2260a33d19978c`.
The file is byte-identical between the audit source and current `origin/main`
`86ba2a170eba184466a7b0dd426c65e473e6f09f`.

## Verification

At `86ba2a170eba184466a7b0dd426c65e473e6f09f`, the package's pinned
[`run_offline.py`](../../deploy/dl-flywheel/pipelines/generated_target_admission_v1/run_offline.py)
completed all 28 registered synthetic fixtures with zero failures,
errors, skips, or unexpected successes. This includes the cutoff tests for
snapshot identity, sidecar refusal, row bounds, event replay, ACK closure, and
read-only behavior. The runner reported `source_unchanged=true`, zero GP
executions, and zero admission commits. Its full receipt is retained outside
the repository at
`/home/ubuntu/jope_plan/evidence/res03-t0105/generated-target-admission-offline-86ba2a.json`
(SHA-256 `28d80bcf946e54d9e6552a884496c7b640530cb8da94bd3c06500dfc03e388f0`). The run completed at `2026-10-31T07:17Z` in
41.93 seconds, with zero live database accesses.

Command, from the repository root:

```sh
nice -n 10 rtk proxy python3 -B deploy/dl-flywheel/pipelines/generated_target_admission_v1/run_offline.py \
  --receipt /home/ubuntu/jope_plan/evidence/res03-t0105/generated-target-admission-offline-86ba2a.json
```

This disposition covers only the one finding above. It does not close the
remaining findings in audit batch 4, qualify a live export, or establish any
model, data, or release capability.
