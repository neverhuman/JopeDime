# Ordered difference Experimental search admission

This closes the held B08 profile/generation/EEC/utility slice after native
routing #182 and recipes/fitted-export #185. The public
`DifferenceSearch` entry admits one optional causal SampleTime wire-59 stage,
orders 1–4 and fixed steps from 0.25, 0.5, 1, 2 and 4. Its explicit
`ExperimentalProfile` switch has a finite prior in [0,0.25]. The profile's
version-2, 17-byte codec preserves the switch and exact prior words, including
signed zero, and its identity binds those words to the bounded search contract.
Unknown versions or flags, malformed lengths and invalid priors refuse.

Admission checks the complete shape through the merged recipe-closure hook:
bounded width/input metadata, a plain numeric Ridge downstream program and
minimum native/fitted codec feasibility. IID tasks, opaque unsupported
transforms and oversized metadata refuse before generation. The constructor
requires an explicit ordered task and input identity. This opt-in family
leaves old profile tables, generator streams, operator IDs and parsing intact.
The generic wire reservation manifest remains closed; this specific shipped
Experimental entry is the bounded generator.

A prior miss returns the same plain child. Grow inserts the stage, Change
changes its order while preserving step/input/downstream identity, and Drop
returns the candidate's own downstream child. Disabling the profile prevents
creation/modification and still allows removal and lossless reading of stored
programs, including unsupported opaque children. Created and edited candidates
round-trip through versioned bytes and lossless canonical text.

EEC executes seeded Grow and Change candidates through Rust, Python, C++17
training recipes and fitted C++, comparing prediction words, stage faults and
raw continuation bytes with the native training-prefix fit. It also compiles
and executes the removed plain child in all four forms. The existing recipe
fixture contains nonfinite boundary words and group-local train/test carry.
The shortest-tree witness compares the existing within-row lag/roll axis
against two identical current rows with distinct prior observations: wire 59
returns different causal values while the existing row-only tree returns the
same current-row result.

The immutable utility corpus contains six named synthetic training datasets:
local_velocity, local_acceleration, local_jerk, group_shocks, pulse_carry and
level_control. Each has 128 ordered, two-column observations and explicit
categorical groups. Each dataset uses five seeds and two paired profile arms,
64 evaluations per arm, two inner folds [64,96] and [96,128], an archive of 16,
maximum 15 nodes and depth 4. Labels beyond each inner training prefix cannot
enter that fit. Dataset/corpus hashes, profile identities and fold identities
are recorded. The two arms share their downstream parameter draws and
fitness-fit budget. Resource-admission prototype fits are separately counted.

Receipts record CV loss, top 10 median loss, evaluations to 95% of the final
observed loss improvement, rejects, invalid and unique children, elite token
use and sizes, domain/saturation cell rates, evaluation time, and Rust/Python/
C++/fitted code-token and byte costs with paired deltas. Each distinct selected
winner is compiled and executed against its recorded prefix 96 native model
in all four forms. Artifact fits and recipe-admission prototype fits are
post-selection work outside the fitness budgets. Six test cases preserve the
same total 3840 evaluations/7680 fitness fits while keeping each compiled
winner proof within the existing test timeout.

Acceptance uses the registered `finite_difference_v2` target, exact legacy
freeze, strict feat-math Clippy, lane-only audit, PR tier and the final card
full suite under the shared heavy lock, followed by exact-head automatic CI
and independent native approval. Before shipping registration, external
preparation at source 36b6 passed all six utility cases (128 artifacts), and
separate EEC preparation passed 20 artifacts. These are preparation receipts,
not claims of qualification on the eventual shipping commit. Final exact-head
commands/results are recorded in B08_REMAINDER/profile-implementation-01.

Per-op definition of done (plan §3/B08): existing kernels/routing/recipes supply
native/interpreted/compiled/emitted twins, guarded numerical faults, bounded
arity/width, train-only fit and state replay, saturation, version rejection,
old freeze, IP scrub and no new crate dependency. This slice supplies the
shortest-tree witness, bounded Experimental prior, removal without loss of
parsing, actual GP create/change/remove EEC, six-dataset equal-budget utility
and token/byte deltas. No Baseline promotion or measured product lift is
claimed. Fixed-cohort W §17 G-A through G-E qualification remains required
before promotion; synthetic fixture measurements do not establish those gates.
