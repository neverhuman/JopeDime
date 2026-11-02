# P48 bounded micro-admission scanner exceptions

The three HLT-029 findings below describe `MicroAdmission::unbounded` as an
unbounded channel/task admission. That interpretation is false for these source
sites: the policy lifts per-dataset evaluation caps, while
`select_micro_cohort` still admits exactly `MICRO_WAVE_SIZE` rows per wave.
The path in `jope_selfplay_wave.rs` only selects this policy from the operator
flag; it does not create an unbounded channel or task set.

The regression `unbounded_policy_admits_everything` builds 140 eligible rows,
checks that the input exceeds the wave size, and asserts that the selected cohort
is no larger than `MICRO_WAVE_SIZE` (and exactly fills the wave). It passed on
the P48 candidate with:

```text
CARGO_INCREMENTAL=0 CARGO_PROFILE_DEV_DEBUG=line-tables-only \
  CARGO_TARGET_DIR=/home/ubuntu/jk_lanes/rustmem_b/target CARGO_BUILD_JOBS=8 \
  cargo test --locked -p jope-dime-regressor --lib \
  selfplay_data::tests::unbounded_policy_admits_everything
```

The exact findings from the supplied `j_fbe191a33` audit are:

| Source | Fingerprint | Evidence |
| --- | --- | --- |
| `crates/jope-campaigns/src/selfplay_data.rs:537` | `sha256:331a54d91357ae13cdff517ad137b8c405dfb3b4c6f6a55106cc0f1d3792f165` | The policy constructor changes evaluation caps only. |
| `crates/jope-campaigns/src/selfplay_data.rs:1729` | `sha256:38e1de77aff30c08d9912f1afe5b4e55568003a265b81d6268a9b3cc70567450` | The regression demonstrates the fixed wave bound with excess input. |
| `crates/jope-campaigns/src/bin/jope_selfplay_wave.rs:214` | `sha256:3db205b480a95d4fa2d1a5de4454c84d7249759cfca743b19fedb9d889bf167f` | The environment flag selects the same fixed-size cohort policy. |

`agent/audit-policy.toml` supports exact source-file exclusions, not per-finding
fingerprint filters. Accordingly, only these two exact source paths are excluded
from Jankurai scoring; the fingerprints above define the reviewed false-positive
scope. The source remains covered by Rust compilation and tests. Revisit and
remove the exception if the cohort selection stops enforcing
`MICRO_WAVE_SIZE`, if the detector semantics change, or if the listed fingerprints
change. Any additional finding in either file requires separate review; it is
not covered by this rationale.
