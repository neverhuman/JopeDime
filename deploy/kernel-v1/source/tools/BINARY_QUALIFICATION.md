# Binary qualification workflow

`binary_prepare.py` reconstructs recorded training rows, freezes source-family
aliases, and prepares up to 20 distinct grouped, stratified cuts per admitted
source. Cut artifacts contain references, membership indices, and preprocessing
recipes; they do not copy source tables. Final tests are not opened. Sonar's
unresolved acquisition grouping and qualification-only admission are retained.

For a newly prepared experiment, use serial Hyperion dispatch when exact replay
is required:

```sh
env -u JAIN_HYPERION_NO_NORMALIZE -u JAIN_HYPERION_FULL_CANVAS \
  python3 tools/binary_encode.py --root "$PWD" \
  --experiment "$PWD/runs/NEW_BINARY_EXPERIMENT" \
  --binary "$PWD/bin/FROZEN_NEURAL_BINARY" --serial-hyperion
```

The flag freezes dispatch in `encoding-run.json`; an existing concurrent run
cannot switch policy in place. Each wave assigns at most one cut per GPU rank,
preserving the original job index modulo eight. The existing native binary,
weights, sweep seed, 24,000-cell cap, and projection stay fixed. The command
above explicitly clears the two native semantic overrides. The scheduler shares
one spare 16-CPU allocation with comparisons and yields mandatory campaign GPU
blocks. `--initial-only` saves the initial cohort before expansion.

The September 8 experiment is saved in `runs/binary-qualification-v1/`:

- `report.json`: original 80 binary contexts and 240 raw components.
- `reproducibility/verification.json`: independent real-20 replay of all three
  encoders and projection. Target and foundation are exact; Hyperion and context
  fail byte equality. The failed result remains immutable.
- `reproducibility/numerical-review.json`: observed errors, existing qualification
  tolerance, and source-supported diagnosis of timing-dependent cross-cut
  batching. Original per-forward batch history was not retained.
- `deterministic-v2/pair-qualification.json`: two fresh serial Hyperion passes
  and projected contexts agree exactly on all 20 cuts, with tolerance zero.
- `deterministic-v2/index.json`: 80 separately versioned contexts, with launch
  policy and qualification hashes. All 80 Hyperion components were recomputed
  serially; the original target/foundation components and projection are reused.
- `deterministic-v2/local-artifact-verification.json`: full 80-context audit and
  explicit v1-to-v2 differences. Exact pair replay was checked on 20 cuts.
- `deterministic-v2/restart-verification.json`: completed stages and artifacts
  survive restart unchanged, with no worker launches.

`binary_serial.py` performs the bounded serial qualification and publication in
a separate directory. Its frozen source and helper files are saved with the
result. Its strict pair gate never increases a tolerance or rounds outputs.
Neither workflow writes the production champion store or changes the campaign
encoder selection.
