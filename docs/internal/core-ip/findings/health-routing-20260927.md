# CPU diagnostic repair and explicit test routing — 2026-09-27

This change repairs test execution and records its limits. It does not qualify a GPU
campaign, replace private corpus inputs with synthetic data, or establish a complete
regression learning loop.

## Reproduced baseline

Full-main diagnostic `36275127911`, attempt 1, source
`8a55c4fef407db014c4d84cf3980178f936bacd5`, remained red:

| Family | Executed | Passed | Failed | Skipped |
|---|---:|---:|---:|---:|
| Default | 3,433 | 3,425 | 8 | 44 |
| Scratch | 2,118 | 2,110 | 8 | 19 |
| Neural | 807 | 731 | 76 | 16 |

Original collections, JUnit and logs are retained under
`/home/ubuntu/jope-rescue-20260921/claude-resume-20260927/baseline-full-main/`.
The older W0 baseline remains unchanged in `baseline.json`.

## Receiving paths and routing

- The 74 neural failures caused by the deliberately unavailable CPU fit-service
  binding remain collected and active in both default and scratch. They are ignored
  only under `neural`, with explicit reasons and an exact inventory in
  `deploy/ci/neural-cpu-only-tests.txt`. The routing guard verifies that CPU cases are
  selected, not merely present. The active neural refusal test remains in place.
- The two remaining neural failures previously called public campaign and ranked
  proposal routes that deliberately refuse unavailable GPU grants. Public-route
  refusal tests now accompany private post-admission fixtures. The ranked proposal
  fixture still uses fresh processes and checks its six controls, resume and shared
  tail. No production grant issuer, service fence or bypass flag was added.
- Eight private-corpus/deferred cases remain in
  `deploy/ci/real-corpus-tests.txt`. Credential-free CI explicitly ignores them;
  `just test-real-corpus CUSTODY_ROOT` executes exactly that list. The wrapper binds
  all fixture-root overrides to custody, rejects writable/link/special files and
  verifies the entire regular-file closure against its manifest before execution.
  Absent historical context and deferred classification inputs remain unavailable.
- Telemetry fixtures use one explicit observation clock, including freshness
  boundaries. Resident readiness fixtures wait for observed polls instead of a
  guessed sleep, retaining the original deadline.

## Local evidence and limits

All commands and outcomes below are retained under
`/home/ubuntu/jope-rescue-20260921/claude-resume-20260927/health/`.
Numerical/test batches have 4 CPU, 16 GiB and 900-second limits, no visible CUDA
GPU, and the lane's reused `target/`. They are local fixtures, not hosted CI.

| Evidence | Outcome |
|---|---|
| `cpu-directed*` | All 98 directed CPU cases passed: the 74 inventoried service contracts, 23 resident-protocol tests and one telemetry-clock test. 287 seconds including build; 1,600 unrelated cases excluded by the directed filter. |
| `neural-selected2*` | Four directed neural refusal/post-admission tests passed; 102 seconds including build. |
| `full-neural*` | First full-family attempt exceeded 16 GiB at 16 test threads; retained failure, not assertion qualification. |
| `full-neural2*` | Two-thread attempt hit the original 900-second cap: 358 passed, two running tests terminated, 374 not run, 91 skipped. Incomplete. |
| `host-corpus*` | Eight selected cases executed: five passed; the historical context-catalog case and two deferred binary42 cases failed as unavailable. 231 unrelated cases excluded by the declared selection. |
| `routing-proof/` and `test-test-routing.sh` | Actual collected routing inspected; three valid controls and 11 adversarial refusals. |
| `test-real-corpus-lane.sh` | Wrapper/root-binding control and five custody refusals; Cargo alone is mocked, no private-data qualification claimed. |

The first selected-neural launcher failed to find Cargo in the systemd environment;
its failure is retained separately. The corrected launcher records its explicit PATH.
No cancelled, unexecuted or excluded test is counted as passing.

The host input copy `host-inputs-v2/` has manifest SHA-256
`7a48a408dbb29ab8cb56cac774a8fc81d20b0ce570932120214d5e258ef8c44e`.
It corrects only the original custody manifest's construction-order mismatch for
`PROVENANCE.txt`, preserving that manifest and provenance verbatim as `ORIGINAL-*`.
All original dataset/run/registry bytes verified unchanged. This is a separately
identified custody copy, not a new dataset release or recreated historical evidence.

## Landing on main `cee1fc8bd`

The routing was replayed onto `cee1fc8bd242688972c352133847793c6927279f` (after PR #114).
PR #114 added six more CPU fit-service contracts. At the replayed head the local
full-neural family failed exactly those six, each with the designed refusal `CPU fit
authority requires a non-neural evaluator build`. The same six passed in full-default
and full-scratch at that head:

| Added by | Contract |
|---|---|
| `82006d600` (W1-T05 R2) | `gp::fit_contract_tests::the_public_fit_view_projects_under_its_declared_root_and_never_resolves_an_ancestor` |
| `82006d600` (W1-T05 R2) | `gp::fit_contract_tests::the_legacy_isolated_export_never_resolves_a_pointer_view_from_its_ancestors` |
| `82006d600` (W1-T05 R2) | `projected_export::shipped_fit_view_projects_under_its_declared_root_before_the_fence` |
| `4aa4aca8a` (W5-T01) | `gp::fit_contract_tests::a_seeded_search_freezes_the_inputs_its_lineage_is_verified_against` |
| `4aa4aca8a` (W5-T01) | `gp::fit_contract_tests::recorded_search_inputs_verify_without_caller_paths_and_refuse_substitution` |
| `c900459dd` (W5-T05) | `gp::fit_contract_tests::helpers_do_not_survive_a_killed_search_controller` |

They are routed the same way as the original 74. Each is ignored only under `neural`
and listed in `deploy/ci/neural-cpu-only-tests.txt`, which now names 80 contracts. The
routing guard still requires every listed contract to be selected in default and
scratch. No refusal, fence or assertion changed. The failing reproduction is retained
as `health-landing-full-neural-20260927T172425Z`. The post-routing family runs are the
later `health-landing-*` directories. Both sit under
`/home/ubuntu/jope-rescue-20260921/regression-remediation-20260926/claude-w0-w12/runs/`.

One local full-default failure at that head is environmental, not a product result.
`jope-paths tests::home_resolution_is_independent_of_the_build_directory` requires a
temporary directory with no `.jope-home` ancestor. The host rule places `TMPDIR`
inside the lane checkout, which carries the tracked `.jope-home` marker, so that
precondition fails. The branch does not change `crates/jope-paths`. The test passed
on the hosted baseline diagnostic `36275127911`.

Exact-head CI and the next full-main diagnostic remain distinct verification gates.
A credential-free diagnostic passing after these explicit exclusions must not be
reported as complete private-corpus or GPU coverage.
