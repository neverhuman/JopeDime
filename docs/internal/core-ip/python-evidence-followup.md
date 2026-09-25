# Full Python fixture evidence

Observation date: 2026-09-25 UTC. Full-main run 36188806213 at
`d825daf802f042994cc5f9ab8f3726511aa0dd62` executed 329 Python tests successfully,
with 15 host-only exclusions, but its inventory rejected
`pytest-tmp/test_recovery_double_read_reje0/link.csv` as unsafe evidence.
The test execution and a valid retained evidence inventory are separate results.

Fast CI already retained pytest fixtures in a non-dereferenced tar archive. The
full Python family now uses the same shared helper. Both workflows preserve their
test selection, JUnit output and proof gates. The fixture directory is a fresh
sibling of the receipt directory; only the archive enters uploaded evidence.
File, directory and dangling symlinks remain link entries inside it. The inventory
continues rejecting loose symlinks and altered evidence. Archive consumers must
inspect link targets before extracting an archive into a filesystem.

The helper preserves a failed pytest command's exit code, makes archive failure
fail an otherwise successful command, and refuses an existing fixture path before
pytest can clear it. It does not delete the original fixture directory. The helper
is explicitly hashed among receipt inputs.

Reproduce the wrapper/inventory checks with:

```sh
python3 deploy/ci/test-full-main.py
python3 deploy/ci/test-receipt.py
python3 deploy/ci/test-aggregate.py
shellcheck -S warning deploy/ci/pytest-evidence.sh deploy/ci/fast-lane.sh deploy/ci/full-lane.sh
```

Focused cases cover loose-link rejection, archive inventory acceptance, retained
link identity, tampering, original test failure, archive failure and stale-fixture
refusal. Local and hosted receipts are retained under
`/home/ubuntu/jope-rescue-20260921/python-fixture-evidence-20260925/`.
This repair does not qualify the remaining full-default, scratch or neural gaps,
and does not change training, corpus admission or GPU timing budgets.
