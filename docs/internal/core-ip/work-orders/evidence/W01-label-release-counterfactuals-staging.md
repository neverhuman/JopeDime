# W1-T04, W1-T06 and W1-T07 (source/fixture scope) — label release, counterfactual proofs, historical staging

Branch `rr/w1-inputs-2`, base `38f83f47c` (`feat/regression-remediation`). Owner: Claude
implementation subagent, lane `.agent/worktrees/demo`. Inputs: the committed
[consumer map](W01-consumer-map.md) (routes R7, R13, R14) and
[release/preparation record](W01-release-and-preparation.md).

Observed states, each scoped:

| Task | State | Not claimed |
|---|---|---|
| W1-T04 fit inputs / scoring labels / release | reproduced (3 routes); implemented; fixture-proven locally | No OS isolation of the scorer, no release CLI, no live worker (W7); not CI-verified |
| W1-T06 counterfactual receiving-path proofs | 3 defects reproduced by the harness, fixed; 13 mutation receipts; fixture-proven locally | Legacy registry path and GP search only; context producer, encoders, teacher and trainer are not covered |
| W1-T07 historical staging | recorded from custody evidence; corrected protocol identity and replay verifier implemented; custody replay rejects the old closure | Not wired into the jope-bench planner; no corrected run exists; producer attestations are declared, not proven |

Nothing here is `CI-verified` or `live-qualified`, and no register entry is closed. The
original findings stay open with their original hashes.

Evidence lives under
`~/jope-rescue-20260921/regression-remediation-20260926/claude-w0-w12/runs/`. Each `w1b-*`
directory holds `HEAD`, `diff.patch` (plus `untracked.patch` for new files), `status.txt`,
`source.txt` (argv, toolchain, environment, exit code), `nextest.log` (command output) and
`summary.txt`. Receipt directories are listed per task.

## W1-T04 — Separate fit inputs, scoring labels and result release

### Routes reproduced, then closed at the receiver

| Route | Reproduction (failing first) | Fix | Commit |
|---|---|---|---|
| R7: `audit.json` published the sealed test label value set | `w1b-t04-prefix-audit-search-20260926T073551Z`: audit.json carried all ten sealed label strings | The audit projects the sealed split to feature columns as it parses; label-derived statistics are withheld and declared in a new `sealed_test_access` object | `98839ca01` |
| R13: `--qualification` searched a sealed `meta_test` family | same run: the search succeeded | `search_inner` refuses `meta_test` before any view load, reservation or fit claim | `98839ca01` |
| R14: `score` scored any caller-named label file and published directly | `w1b-t04-prefix-score-20260926T092242Z`: caller-chosen labels scored, mae/rmse published | New outer release authority (below); `score` needs its grant | `13dc48cc1` |

`w1b-t04-prefix-score-20260926T092213Z` is a build failure of the change's own refactor,
not a reproduction.

### Contract (`crates/jope-engine/src/gp/outer_release.rs`)

- **Permitted consumer.** Sealed labels (role `sealed_final`) have one consumer: an outer
  scorer holding a grant.
- **Grant.** Issued only for an authenticated fitted subject:
  - the predictor exported from a sealed, independently replayed selection
    (`handoff::verify_seal`);
  - its final refit verified against its fold fit (`lineage::verify_final_refit`).

  The grant binds the subject, its preparation, and the preparation's own sealed-test
  identity (`Prepared.original_test_sha256`) at a root-relative path with no symlinks. There
  is one grant per subject and label set, and the sealed file is named, not opened.
- **Score.** The scorer runs these steps in order:
  1. Verifies the recorded grant, the exact subject bytes and the granted path.
  2. Records the exposure (`score_started`) before the first label byte. So a grant admits
     one attempt, and a retry or crash cannot get an unrecorded look.
  3. Parses exactly the hashed bytes (`Table::parse`).
  4. Writes the result only to the authority's withheld store, and returns a receipt
     without metrics.
- **Release.** `Authority::release` is the only path to visibility, and it works once.
- **Ledger.** Every grant, refusal (with `labels_opened`), exposure, score and release is
  appended to a hash-chained `ledger.jsonl`. A removed event breaks the chain.
- **CLI.** `gp::score` (the shipped `score` command) routes a grant to the authority named by
  `JOPE_OUTER_RELEASE_AUTHORITY`. A frozen selection receipt alone is refused before the
  test path is opened.

### Acceptance tests

- `label_release::audit_never_parses_or_publishes_sealed_test_labels`
  - Forbidden-only: sealed labels changed and rows reordered leave audit.json
    byte-identical.
  - Authorized control: sealed-feature overlap is still detected.
- `gp::fit_contract_tests::qualification_search_of_a_sealed_meta_test_family_is_refused_before_any_view_or_fit`
- `gp::fit_contract_tests::a_selection_receipt_alone_neither_reads_caller_chosen_labels_nor_publishes_a_score`
- `gp::fit_contract_tests::outer_scores_need_a_grant_read_only_the_granted_labels_and_release_once`
  - Denied upstream access: the sealed file does not exist while search, selection and
    export run.
  - Refusals, each before any label read (`labels_opened=false`):
    - at grant: symlinked label path, unexported subject, duplicate grant;
    - at score: forged grant, re-encoded model, byte-identical copy at another path.
  - The valid score goes through the shipped CLI entry and is withheld; a retry is refused
    as consumed.
  - One release, whose RMSE equals an independent recomputation (relative tolerance 1e-12;
    same formula and operation order). A second release is refused.
  - The exact ledger event sequence is checked, and no fit starts.

Retained receipts: `w1b-t04-postfix-score-20260926T092803Z/receipts/outer-release/`. They are
`ledger.jsonl`, `grant.json`, `scored-withheld-receipt.json`, `released-outer-score.json`,
`refused-label-read-forged-grant.json`, `refused-label-read-other-file.json` and
`refused-retry.json`. The same set is in `w1b-slice2-regression-20260926T092904Z`.

### Scope and limits

- **Development search vs fold-pure context.** A GP search is task-level development
  search. The cut's validation labels are its adaptive selection labels ("cut heldout
  optimized repeatedly"). Fold-pure isolation is proven for the cut's *fitting state*
  (T06), not for the search's selection.
- **Projection before worker launch.**
  - Holds already: GP evaluation workers receive a sealed memfd of the projected
    training/validation rows.
  - Not changed: the fitted-view export worker loads its own training view from the
    prepared path (`main.rs`, outside this lane). That is recorded as open.
- **Behavior change.** The CLI `score` no longer accepts a selection receipt as authority to
  read sealed labels. Release is library-only until a CLI entry is added in `main.rs`.

## W1-T06 — Counterfactual receiving-path proofs

**Harness.**
- Integration test: `crates/jope-engine/tests/counterfactual_boundaries.rs`. It runs over
  the fixture `tests/support/legacy_dataset.rs`: generated backing source → scaled
  historical train/test → `audit::run` → `prepare::run` → `load_view_in_root`.
- Search-level proof: `gp::fit_contract_tests::search_proposals_fits_and_scores_depend_only_on_their_permitted_rows`.

**What is frozen.**
- Source and fixture generator.
- `SEED`. Folds and caps depend only on row positions and feature identities.
- Single-threaded execution.
- Every permitted byte except the perturbation.

**Tolerance: zero.** Comparisons are exact SHA-256 equality of produced bytes, because the
producers are deterministic. The search proof compares recorded identities and f64 bits.

**Receipts.** One receipt per mutation, in `w1b-slice3-t06-regression-20260926T093641Z/receipts/counterfactual/`.
Earlier copies are in `w1b-t06-postfix-*`.

| Mutation (receipt) | Class | Result |
|---|---|---|
| `sealed-labels-changed`, `sealed-labels-shuffled`, `sealed-rows-reordered`, `sealed-categories-changed` | forbidden-only | Every audit/prepared/cut/view digest is identical. Permitted differences: `transfer.json` and `prepared.json` `original_test_sha256` |
| `sealed-split-denied-read` | forbidden-only | With test.csv removed after the audit, every prepared byte and view is identical: preparation and views never open it |
| `cut0-heldout-labels-changed`, `-shuffled`, `-categories-changed` | forbidden for cut 0's fitting | Cut 0's fitting entry, preprocessing and training view are identical. Cut 0's validation view and every other cut's training view change (authorized) |
| `cut0-training-label-changed` | authorized | Cut 0's fitting state changes |
| `stale-cached-preparation-state` | refusal | Baseline state reused after the label change is refused for both partitions |
| `unreferenced-backing-row-changed` | forbidden-only | The old preparation is refused (`backing table changed`). Re-preparation differs only in the backing identity |
| `historical-training-rows-reordered` | new identity | `original_train_sha256` differs; both preparations remain loadable |
| `group-alias-feature-identical-rows` | grouping | Rows with identical features never straddle any cut |
| `duplicate-join` | refusal | **defect fixed** |
| `symlink-escapes` | refusal | Symlinked backing source refused at the view. Symlinked `test.csv`/`train.csv` are **defects fixed** |
| `search-unreferenced-backing-rows-changed` | forbidden-only | Proposals, fit keys, fitted models, scores and predictions identical; zero new fits (all reused) |
| `search-heldout-labels-changed` | authorized selection labels | Slot-0 fit key identical (reused), score changes; later proposals recorded, not asserted |
| `search-training-label-changed` | authorized fitting input | Fit key and fitted model change; new fits |

**Defects reproduced by the harness.** The prefix run `w1b-t06-prefix-20260926T092507Z` had
10 tests: 6 passed, 4 failed. Three of the failures were defects, fixed in `bb186aaf8`:

1. **Duplicate-expanding join.** A historical row repeated onto one backing row was
   prepared. Fix: `check_cut_membership` (receiver and producer) refuses two rows of one
   membership that resolve to one backing row.
2. **Genuine duplicates collapsed.** Reconstruction mapped equal historical rows to the first
   equal backing row only. Fix: it now assigns distinct equal backing rows in source order,
   so genuine source duplicates stay admissible
   (`identical_backing_rows_map_to_distinct_pointers`).
3. **Symlink escape.** The audit read `train.csv`/`test.csv` through symlinks, and
   preparation read `meta.json`/`train.csv` through them. Fix: both refuse symlinked leaves.
   The audit records the refusal as an issue; preparation refuses before writing.

The fourth failure was a wrong test expectation, not a defect: the audit's training summary
is order-independent. The expectation was corrected and the correction recorded in that
receipt.

The first post-fix run `w1b-t06-postfix-20260926T093342Z` had 1 failure. Its search
projection compared `seeds_sha256`, an identifier of the changed preparation. That field is
now excluded and declared in the receipt.

**Behavior-change risk** (fail-closed): an on-disk pointer view whose membership repeats a
backing pointer is now refused at load.

**Not covered.**
- Context producer, encoders, teacher and trainer receivers. They are outside this lane or
  cannot dispatch (`require_original_gpu_grant`).
- The ASM `jope-pipeline` receiver.
- The backing reader still digests and tokenizes the whole backing file. The proofs show
  that no output depends on rows outside the membership, not that those bytes are unread.

## W1-T07 — Historical staging, without rewriting it

**Source.** `crates/jope-datasets/src/staging_disposition.rs`, CLI
`jope-datasets staging-disposition {derive,verify}`. Committed records are in
`crates/jope-datasets/releases/staging-disposition/`:

| Document | Identity | sha256 |
|---|---|---|
| `historical-gpu-staging-20260925.json` (`jope.historical-staging-disposition/v1`) | `disposition_id` `fc59d3be553870627acdf0db9d31d8c69b1b08d9186f18dbb06a3b17fcfdca32` | `0573e611aeee5187302390e8bbdf7a56b88ec00c21ad673cdcbf9cbcf0d03266` |
| `fitting-only-staging-protocol-v1.json` (`jope.fitting-only-staging-protocol/v1`) | `protocol_id` `8f5d19fa9a71c730f383cf677887a4ff39840e00c42a16583743f7a586a1218b` | `c293194ae79b6646e20df0a75a084a3b3d28559dc313b7b41b88bb1f5bb5586b` |

**What the disposition records.** Everything comes from the retained evidence only. It was
derived from custody `core-ip-20260925/study-final/` and the published
`docs/internal/core-ip/data/*.json` records.

Before recording anything, the derivation cross-checks the evidence for:
- one staging source, `802fb464`;
- one admitted window, experiment `8c059199…`;
- identical published and custody deviation/closeout records;
- binary and plan digests matching the study manifest;
- all 25 retained projected objects re-hashed against the historical manifest `e0256090…`.

What the evidence states:
- **Observed.**
  - Outer (cut-0) scoring labels were parsed during staging.
  - No sealed table was opened, no outer score was executed, no held look was consumed,
    and no outer target was in the worker payload.
  - Status `controller_failed_before_cuda_qualification`: exit 1, no direct request,
    full-GP arm or prediction batch.
- **Affected descendants (28).** The manifest, 10 projected fitting inputs, 15 inner-fold
  inputs, the v2 plan, and the v3 local plan proof that accepted this closure. All are
  retained and excluded from corrected execution and from headline measurements.
- **Unknowns.**
  - Whether the parsed labels influenced any retained byte.
  - Other readers during staging.
  - Encoder/teacher/checkpoint exposure.
- **Classification.** Contamination is *not* inferred. The run is not protocol-compliant
  and not a qualified measurement, and the result is not reclassified. Validation refuses
  any document that upgrades these, even with a recomputed identity.

**Corrected protocol.** A new identity, derived purely from the disposition. It forbids:
- the historical projection source;
- the 26 staging products (label-free frozen split memberships are not forbidden);
- the historical window.

A corrected run (`jope.fitting-only-staging-run/v1`) has its own `run_id` and its own window,
or none until admission. It must pin its manifest bytes and declare a fitting-only
projection. The declaration is recorded and required, not independently proven.

**Replay verifier (custody).** Receipt `w1b-t07-custody-replay-20260926T093927Z`:
- Re-derivation from custody is byte-equal (`rederived_bytes_equal: true`).
- The retained historical closure `retained-fitting-inputs.json` (`e0256090…`) is
  **rejected** (exit 1): "cannot be replayed under the corrected fitting-only protocol".
- The repaired-planner proof's injected manifest (`15db373f…`) is **rejected** for its
  historical projection source.

The derivation receipt is `w1b-t07-derive-20260926T093903Z`.

**Pre-existing gap.** This is a new contract, so it had no failing receiver to run. The
retained `planner-local-proof.json` records that the v3 planner accepted this historical
closure (`positive_plan_exit: 0`), and nothing rejected it for a corrected protocol. That
evidence was not re-executed.

**Tests** (`jope-datasets --test staging_disposition`, 6):
- `committed_disposition_records_the_retained_evidence_without_upgrading_it`: the published
  evidence digests are pinned.
- `the_committed_protocol_is_a_new_identity_derived_from_the_disposition`.
- `the_replay_verifier_rejects_the_historical_input_closure`, with a positive control.
- `a_corrected_fitting_only_run_gets_a_new_identity_and_its_own_window`.
- `historical_and_corrected_documents_cannot_be_substituted`. It covers:
  - schema cross-opening;
  - the historical window;
  - a run bound to the historical identity;
  - other closure bytes;
  - a parsed-labels attestation;
  - an edited run;
  - a protocol derived from another history.
- `derivation_cross_checks_the_retained_evidence_and_is_deterministic`, on a synthetic
  custody tree. Six damages are refused: missing stderr, changed projected input, a second
  source, a second window, differing published records and a symlinked plan.

**Open.**
- Wire the verifier into `jope benchmark inference-plan` (jope-bench, outside this lane).
- The corrected run itself, with its window and W11 admission.

## Commits and verification

| Commit | Content | Pre-commit run |
|---|---|---|
| `98839ca01` | T04 audit and meta_test routes | `w1b-slice1-regression-20260926T091847Z` 130/130; `w1b-slice1-datasets-20260926T092057Z` 57/57 |
| `13dc48cc1` | T04 outer release authority | `w1b-slice2-regression-20260926T092904Z` 132/132; `w1b-slice2-check-20260926T093043Z` (`cargo check --workspace --all-targets`) |
| `bb186aaf8` | T06 harness and three receiving-path fixes | `w1b-slice3-t06-regression-20260926T093641Z` 143/143; `w1b-slice3-datasets-20260926T093828Z` 57/57 |
| T07 commit | staging disposition, protocol, verifier, CLI | `w1b-slice4-*` (see the commit message) |

**Baseline.** `w1b-baseline-engine-20260926T073035Z` had 104/104 over the lib `gp::` tests
and view_contracts, gp_contracts, fit_boundary, preparation_semantics and
experiment_isolation. `w1b-baseline-datasets-20260926T073343Z` had 57/57. Every regression
run's target set contains the baseline set, and no previously passing test fails.

Coverage keys added:
- `target:jope-parallel/test/label_release`
- `target:jope-parallel/test/counterfactual_boundaries`
- `target:jope-datasets/test/staging_disposition`

`deploy/ci/coverage.py validate` reported 0 unassigned. No Python was added.
