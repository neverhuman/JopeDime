# DL lane source custody

## 2026-11-02 source-closure assertion correction

The source-custody successor is `9afa6c81dc7e4ce0df2d45b76b7d871fe0ea5f660c932f4bb1e6f4553ca49ef7`:177 current files,285 historical versions. The previous current manifest `cc9609693ada637ed1cc3308c514ecb52de9d35aa8ed1d45a17103c432a76414` is retained byte-for-byte at `lane/predecessor/source-closure-correction-20261102/MANIFEST.json`; its entire current closure is reconstructible. The original test bytes are retained as `lane/historical/test_p3_verify.py.2c98d995424b02eed0da09170c94dc3ecce5b7ab2acc25f0831ec4ac419096f4`.

Only the stale expected dependency list in `tests_p3/test_p3_verify.py` changes: the receipt already includes `jtt1.py` and `quarantine.py`, and the corrected test explicitly requires them. The original assertion reproduces a failure on the original imported source; the corrected strict assertion passes. Production source, data, labels, numerical methods/tolerances, host exclusions and offline counts remain unchanged. This is an identified source correction and receipt, not a reauthorization of historical experiments. Eight heuristic-linter variable renames elsewhere are restored to the unchanged producer bytes; the historical manifest/pins are preserved rather than edited.

The records below describe the retained predecessor and its historical review.

The ROOT correction for PR #251 review 5444902657 is the current source-custody head `cc9609693ada637ed1cc3308c514ecb52de9d35aa8ed1d45a17103c432a76414`:
177 current files and 284 historical versions. It retains the prior producer manifest
`e88de3c63c574a133b0742f7c179a217f26f2dee5cffc14f7b82189c63a06143` byte for byte at
`lane/predecessor/root-pr251-resume-20261007T1640Z/MANIFEST.json`; all older predecessor pins
remain unchanged. Four replaced source versions are retained byte for byte in `lane/historical/`.
The new `dltrain/replacement.py` dependency is bound by each affected builder's code inventory.
The three completed-context boundaries compare replacement metadata before skipping `.done`,
and their dataset shortcuts validate the recorded admitted/refused set. Stale completed trees
are preserved and recovered through a new destination: [RESUME_RECOVERY.md](RESUME_RECOVERY.md).
The five additional acceptance cases bring the offline inventory to 729 collected / 688 executed
and the same 41 named private-host exclusions; this inventory is not a runtime qualification.
No existing tests, exclusions, numerical tolerances or publication gates are weakened.

The following import and review record describes the retained predecessor, not a new producer
campaign or live adoption of this correction.

This directory retains the DL lane's 2026-10-06 source handoff for review and reproducible offline fixtures. It does not install, dispatch, adopt or authorize the staged entrypoints.

`lane/MANIFEST.json` follows the exact producer manifest `e88de3c63c574a133b0742f7c179a217f26f2dee5cffc14f7b82189c63a06143` (restage `20261007T142247Z`, the merge restage of PR #251 onto main 8b7476fdf: FOLD-1 and the partial-refusal replacement record on top of main's PR #249 custody head): 176 current files and 280 historical versions. Its predecessor chain is retained byte for byte under `lane/predecessor/`, newest first:

- `restage-20261007T080755Z/MANIFEST.json` (`f7caed54a6d78a1a7da92b61db3c47bc79e86e15d5ce50e1228aa6396a747f43`), PR #249's merged handoff (admission fail-closed), imported byte for byte from main;
- `restage-20261007T054519Z/MANIFEST.json` (`fb2140bc50deb66057f009f42cf6690cca20e3862e8f552199e292068d9ba754`), PR #246's merged handoff (RAW-GROUP-PR with the held-out-target isolation fix);
- `restage-20261007T034806Z/MANIFEST.json` (`985cb6288a26565ae22e331f92f1d28aeb459dd2c72e31fa26a60b6e8f96dacc`), the first RAW-GROUP-PR head, superseded by the leak fix;
- `restage-20261006T153320Z/MANIFEST.json` (`4419ff430ec81a079829e67b935da8b16633e238c76887bd209c6ca9c4d0712d`), PR #224's payload-strip handoff;
- `restage-20261006T131001Z/MANIFEST.json` (`23d3a8a7fd6d15f056e6d5465a9e524ae46f5b69fd9d0e6c6f4721251ce70413`), PR #218's merged handoff;
- `restage-20261006T114844Z/MANIFEST.json` (`253e593b7537cdecf75a89be24a83804bc63d7e2596e02cc0c28d780a50f2b9e`);
- `restage-20261006T082918Z/MANIFEST.json` (`365ec3cecab4e647a160c1f5c2a6baec1f45c5ea102e3180922173ed463f628f`);
- `restage-20261006T073946Z/MANIFEST.json` (`6314ef6f33f01878a149cdb36823a78c96d6ae3e2c45cf409d756711c71d6619`);
- `restage-20261006T035515Z/MANIFEST.json` (`009532038cee11ea20075a7291e4b247c7df8e97806946a61bd8dce7dd635802`);
- the original handoff `MANIFEST.json` (`4b8c54da7e4b11cc2f30b02aa88f159d5240134bf4962e670d2bfcddbcd699ad`).

Each manifest's `supersedes_sha256` names the next one; the original names none.

Three predecessor current files that no later manifest retains are preserved there byte for byte:

- the original's `test_v2_rowsel_raw.py` (`2ae3cf61`), from the independently verified predecessor archive;
- the `00953203` version of `test_v2_publish.py` (`bbb8e5de`), which is also the original's;
- the `00953203` version of `test_v2_rowsel_raw.py` (`f33f3d2b`).

The latter two come from the retained `restage-20261006T035515Z` staging and equal the blobs committed here at that time. Every current file of all ten predecessors can be restored from the verifier's per-manifest reconstruction map. All 276, 274, 274, 262, 204, 190, 88, 79, 58 and 46 of their historical payloads remain available.

The predecessor manifest `253e593b` records `restage-20261006T113426Z` under `abandoned_unpushed_stagings` (`e1ca4e355a5092bc8a7d97a90352ffd35c5d96f1a902c7a411ddca23c9a57ca5`). That staging was never pushed because it picked up the out-of-scope P8 file `dltrain/exclusions.py`. It is not in `lane/`, and the verifier refuses any abandoned staging that appears in the chain, by manifest sha or by path. The current manifest likewise records `restage-20261007T032743Z` (`0388e6a75c0f54038661c61f35e18e1225f1386a2b40abbb43b2ff25b3381987`): its staged tests failed 1 of 578 because `tests/test_split_defaults.py` was kept at prior bytes and rejects the raw v2 `target_linear` block; it was never pushed.

The producer staging kept history at a shared `../historical/` path. This import normalizes exactly the manifest-listed history into `lane/historical/`; it does not rename historical results into current evidence. The producer's README inside `lane/` is the `20261006T131001Z` restage note and describes the original staging location. Current and historical source files and all ten producer manifests are unchanged.

This handoff carries the owner rulings NONRAW-SPLIT-A and GROUP-UNION-B, the publisher hardening, and the fixes for the PR #218 review findings.

The `20261007T034806Z` restage adds only the raw-route group hook (owner RAW-GROUP-PR, owner-decisions-20261006T103543Z): `tools/v2_rowsel_raw.py` `45dbc458` (rowsel-split-v2-groups on the raw route with the target-tie admission and empty-fold gates, and the LINEAR-DETERMINED gate), the new `dltrain/target_linear.py` `e2dbebba`, the new `tests/test_v2_rowsel_raw_v2g.py` and `tests/test_target_linear.py`, and `tests/test_split_defaults.py` `6294d720` scoped to the raw route. These are the bytes the `rowsel-raw-v2gsplit` build receipts pin. The other 15 changed lane files are kept at their prior bytes (`kept_prior_bytes`) and 22 new lane files are excluded (`excluded_new_files`); 12 code-custody versions are new in `lane/historical/` (the manifest's `historical_added` lists 1, because the abandoned staging had already copied the other 11 into the shared producer store). The `20261007T054519Z` restage (PR #246 review 5437848458) changes `tools/v2_rowsel_raw.py` to `48829e82` (the group target-tie guard and LINEAR-DETERMINED now decide each context on its own outer-train rows only; dataset admission before the split reads features only), the docstring of `dltrain/target_linear.py` (`f2c22410`), `tests/test_v2_rowsel_raw_v2g.py` (`72ae2cc1`), and adds `tests/test_v2_rowsel_raw_fold_isolation.py` (`a3480f5d`); the replaced versions are in `lane/historical/`. The `20261007T081144Z` restage (owner ruling FOLD-1, ENG:watcher 07:22Z) changes `tools/v2_rowsel_build.py` and `tools/v2_build_local.py` (target-tie and LINEAR-DETERMINED per context after the split on its own outer-train rows; the census and planning read features only; the L7 split gate refuses fewer than two components), adds `tools/v2_g3_append_build.py` (the G3 catalog builder, moved into the lane from tools-gen/g3_context_build.py), rewrites the tests that encoded the superseded #218 finding-8 order (`tests/test_v2_nonraw_split_v2g.py`, `tests/test_v2_rowsel_build_split_v2g.py`, `tests/test_split_defaults.py`; `tests/test_v2_rowsel_build_split_v2.py` at the lane's bytes), and adds `tests/test_v2_rowsel_build_fold_isolation.py`, `tests/test_v2_build_local_fold_isolation.py`, `tests/test_v2_nonraw_target_linear.py` and `tests/test_v2_rowsel_build_fallback.py`; the 11 other changed and 20 other new lane files stay out of scope (`kept_prior_bytes`, `excluded_new_files`). The `20261007T125037Z` restage (PR #251 review 5441124504) keeps a context-specific refusal from blocking the admitted folds: the T1, L7, G3 and raw builders record `built_folds`, `admitted_folds` and `refused_folds` in `epoch0_replaced` (with `new_contexts` = the written contexts) when any fold is refused, `tools/v2_rowsel_raw_manifest.py` publishes exactly the admitted set bound to the source build tree, and `tests/test_v2_partial_refusal_manifest.py` covers builder to manifest; `gpu_manifest_cut.py` pins the new manifest tool bytes. The `20261007T142247Z` merge restage supersedes main's `f7caed54` (PR #249) and re-applies PR #251's branch-only stagings `d921942a` and `b1505023` on top of it; those two were pushed on the PR branch only, never on main, and are not chain members (their bytes stay in the branch history and the producer store). The lane source also carries PR #249's `dltrain/admission.py` (`59c4f10c`) and `tests/test_admission_fail_closed.py` (`e3c16b5e`). Hashes are listed in the table below.

| file | sha256 | role |
| --- | --- | --- |
| `tools/v2_rowsel_build.py` | `94fcc4b1` | non-raw split hook |
| `tools/v2_build_local.py` | `8c3c9a6a` | non-raw split hook |
| `tools/v2_build_l7a.py` | `0b0bc458` | non-raw split hook |
| `dltrain/rowsel_split_v2_groups.py` | `792aec76` | group split, built on `dltrain/rowsel_raw_split_v2.py` (`97340df3`) |
| `tools/v2_publish.py` | `e97d74c7` | final publisher |
| `tools/v2_provenance.py` | `b777ed2e` | provenance resolver |

The other publish tools are `v2_pubset`, `v2_superseded`, `v2_holdout_denylist`, `v2_s1_input_list`, `v2_holdout_supplement` and `v2_rowsel_raw_manifest`.

Split defaults:

- The raw CLI now defaults to `rowsel-raw-split-v2` (tools/v2_rowsel_raw.py:970; coverage tool tools/v2_rowsel_raw_coverage.py:421). The old feature-ordered split in `rowsel_raw.py` (`18eac154`) runs only when v1 is passed explicitly, for historical replay.
- The T1, local and L7a builders require `--split-version` (tools/v2_rowsel_build.py:2136, tools/v2_build_local.py:1538, tools/v2_build_l7a.py:574), so `registry` runs only when it is passed.

Source custody still grants no runtime authority. Changing a default again requires a separately identified source correction and a new receipt.

## Review dispositions (PR #218 comment 6013410292 and owner pre-review F1–F6)

`RUN` is the lane's private custody root, `/home/ubuntu/scratch/jopedime-dl-training-v1`. This directory does not import or qualify those receipts.

- **Findings 1 and 2** (v2/v2g fold assignment reads no target; scaler fit on outer-train rows): confirmed, no change.
- **Finding 3:** the v1 port lineage check is the only all-row target read. It runs only on an explicit registry replay (`load_dataset(..., v1_port_check=True)`; `None` is refused). The v2 and v2g routes record `null` plus the reason. A before/after reproduction leaves `view_id`, context sha, encoder/teacher/A0 bytes, `cut_id` and the gates unchanged.
- **Findings 4, 5 and 6:**
  - `encoder.jtt1.gz` and `teacher.jtt1.gz` still contain the outer-test block. Train-side use is through `jtt1.train_view` (train range only), enforced by `tests.test_jtt1` (including `test_counterfactual_test_row_change`, now in the offline selection) and the `train_rows_sha256` gates.
  - The S1 inner-CV-only must-fail control passed (`RUN/receipts/trainonly-v2/s1-control/receipt.json`, `714eb90e71a4253fb94f6605711709f0c2e1eb09146ba06e86da34d4c86d4014`), which closes the teacher-input gate previously listed here as open.
  - Owner ruling 06:52 MT (2026-10-06): the train_view/flip-test route is accepted for #218's merge and for local training, but NOT for publication. Nothing that carries outer-test targets goes to S3. Published encoder and teacher payloads are stripped to JTT1 `n_test = 0`, and outer-test labels move to a local-only scorer artifact. PR #224 implements this; see "Payload strip" below.
- **Finding 7:** test-side categorical missing/unseen counts are recorded counts, not inputs to the scaler, the fold assignment or A0.
- **Finding 8 (F4, F5, F6):**
  - **F4:** `target_tied_group` refuses a group column that is bytes-equal to the target, an affine copy (relative RMSE ≤ 1e-10), a function of it, binned with disjoint target ranges, or nearly determined by it (≥ 99%). Module mutants are killed 24/24, including 11 leak mutants, and 0 of the 26 T1 group datasets are refused.
  - **F5:** `split()` requires `partition` and `group_values`.
  - **F6:** target_tied_group reads all-row y ONLY for the dataset-level admit/refuse bit. On admission, y never reaches split()/components() and the split identity is unchanged. On refusal, no context is built.
  - **Plan-path order fix** (grok_sit_watch red-team FAIL at `eb91ab02`, issue comment 6016730510): on the plan path, `registry_group_evidence` ran `component_partition` on a candidate column before `target_tied_group`. A tied column's values therefore reached the union-find, and its evidence kept `v2g_components` and `component_structure_sha256`, although `group_disposition` then returned `not_built` and no context was built. In `tools/v2_rowsel_build.py` `94fcc4b1`:
    - `target_tied_group` runs first.
    - A tied column records only its tie and registry-port fields. Field grouping, `v2g_components` and `component_structure_sha256` are `null` with a not-computed marker, and its values never reach `component_partition` or `components`.
    - `identical_components_across_reproducing` is `null` when a reproducing column has no structure.
    - `group_disposition` checks ties first and requires identical components to be exactly `True`.

    Evidence:
    - Fix receipt: `RUN/receipts/trainonly-v2/build/pr218-finding8-tiefirst-fix-20261006T130400Z.json` (`75d4690e`). The must-fail tests FAILED on `8b056648` (the spy saw `component_partition`/`components` on the tied values, and the old disposition said "ambiguous") and pass on the fix. Mutants are killed 7/7, including 5 leak mutants.
    - The port-free `EvidenceOrderNoPort` class in `tests/test_v2_rowsel_build_split_v2g.py` (`b5a40f2f`) has 5 tests, which run in offline CI. They FAILED on `8b056648` (`failures=4`) and pass on `94fcc4b1`, and they alone kill the same 7 mutants (logs `RUN/tmp/pr218-tiefirst/noport-mustfail-old-8b056648.log`, `noport-new-94fcc4b1.log` and `mutants-noport-result.json`).
- **Finding 9 (F1, F2, F3; PROVENANCE-01):**
  - Code hashes bind by full sha256 only, and the resolver fails closed: exit 2, with `--publish` refused before any client is created.
  - Every published row is scanned, `gpu_encode_versions` are counted, and missing hashes are reported as `MISSING CODE HASH`.
  - Tests are `tests/test_v2_provenance.py` (`32c8da6f`). Mutants are killed 10/10 (`RUN/receipts/trainonly-v2/build/mutants-v2-provenance-f123-20261006T104404Z.json`), with one equivalent mutant explained in addendum `8093c042`.
  - A live dry run reports 0 unresolved hashes and 119 custody files.
- **Finding 10:** the split defaults described above. Explicit paths are byte-identical to custody. Guard mutants are killed 17/17, including 12 leak mutants (`RUN/receipts/trainonly-v2/build/pr218-review-fixes-20261006T113311Z.json`, `fdb0ba29`).

The following are outside this custody and go to a RAW-GROUP follow-up:

- the raw-route v2-groups hook;
- the hyperion_v6all runner and publisher;
- the outer-test scorer adapter;
- the multi-column OLS target-copy gate (must-fail `2a5429bf`);
- the T1 fallback for the 23 drifted rights-held datasets.

Original dated plans, runbooks, dependency inventories, runtime defaults and command examples inside the custody tree are preserved observations. They are not current compute, credentials, publication or pool-control grants.

The lane's red-team, scope, equivalence and mutant receipts are retained in the lane's private custody:

- `receipts/trainonly-v2/build/redteam-*-publish*/receipt.json`;
- `receipts/trainonly-v2/split-scope-nonraw-20261006T034028Z.json`;
- `receipts/trainonly-v2/build/publisher-equivalence-*.json`;
- `receipts/trainonly-v2/build/mutants-*.json`;
- `receipts/trainonly-v2/code-custody/`.

Fresh production input-list admission, G0 provenance/controller integration, S1 pilot, publication, policy training and serving promotion remain independently qualified gates.

## Payload strip (PR #224, owner ruling 06:52 MT)

The ruling: nothing that carries outer-test targets goes to S3, and the published encoder and teacher payloads hold no outer-test target values. Outer-test labels move to a separate scorer-only artifact with its own sha256, read only by the outer-test scorer adapter.

- **Stripped payload.** `tools/v2_strip_outer_test.py` writes a create-only publish copy of each tree. Each `encoder.jtt1.gz` and `teacher.jtt1.gz` is rewritten as JTT1 with `n_test = 0`: the same header fields otherwise, the same inner-fold labels and the same train block. `jtt1.train_view` and `train_rows_sha256` are byte-identical, so A0 and every GPU input binding are unchanged.
- **Scorer artifact.** The outer-test rows and targets move to a scorer-only artifact (`dltrain/scorer_labels.py`, magic `JTS1`, which no JTT1 decoder accepts). It is written to a separate local scorer tree, never to a publish tree, and never to S3.
- **Readers.** `tools/s1_outer_test_scorer.py`, the S1 outer-test adapter, and the strip tool are the only modules allowed to import or name the artifact; a static import test enforces this. On 24 sampled contexts, including the 4 must-fail contexts, the adapter reproduces the stored `official_test` bytes exactly.
- **Publisher gate.** `tools/v2_publish.py` refuses any row-level payload with `n_test > 0`, a header `n_train` that differs from the context's, or the scorer magic. A preflight scan of the whole batch fails closed before any AWS session or client exists, and `publish_context` re-checks the bytes before every PUT.
- **New keys.** Stripped payloads publish only under the new `inputs/jtt1-encoder-s0/` and `inputs/jtt1-teacher-s0/` keys.
- **Quarantine.** The 6,560 unstripped `inputs/jtt1-encoder|teacher/` objects published before the ruling (627,957,830 bytes, 2026-10-05 21:00Z to 2026-10-06 02:15Z) are listed in a local quarantine document (`tools/v2_quarantine_unstripped.py`, `dltrain/quarantine.py`). The publisher, `s3io.get_exact` and `jtt1.load_raw` refuse any quarantined or old-scheme key before any request or cache read. Nothing is deleted; their retention is the owner's decision.
- **Evidence** (lane receipts, not imported here):
  - strip mutants 80/80 killed, 44 of them leak mutants;
  - guard mutants for the s3io, jtt1 and publisher-preflight additions 3/3 killed;
  - a 7-family pilot PASS;
  - the S1 adapter equivalence check, 24/24;
  - the provenance dry run, 0 unresolved.
- **History.** `historical/` grows to every code-custody version recorded so far, including P8 and RAW-GROUP custody bytes. These are inert history, not current lane files.
- **Scope.** Neither this custody nor this PR runs the full strip or a publish batch.

## Checks

Run the stdlib custody verifier and refusal fixtures from the repository root:

```sh
python3 deploy/dl-trainonly/verify_lane.py
python3 deploy/dl-trainonly/test_verify_lane.py
python3 deploy/dl-trainonly/test_offline_guard.py
```

The verifier uses committed expected manifest pins for the handoff and its ordered predecessor chain. It:

- checks every `supersedes_sha256` link;
- refuses abandoned unpushed stagings inside the chain;
- checks every payload and filename;
- rejects extra files and symlinks;
- reconstructs every predecessor's current files and historical versions;
- parses Python syntax without importing the staged code.

These are source custody checks, not executed-family coverage or producer authentication for operational consumers.

The Python CI lane additionally runs exactly 21 preserved unittest modules through `run_lane_offline.py`: 555 collected identities, of which 517 synthetic cases execute and 38 named cases stay explicitly deferred.

What the runner allows and refuses:

- It uses the existing locked Python environment, hides GPUs, and removes ambient credential configuration.
- It refuses Python credential-file reads and every Python socket audit event, including socket creation and UDP sends.
- It refuses direct Python subprocess/spawn/fork/exec launches. The only exception is an exact read-only `/sbin/ldconfig -p` library-cache probe through Popen or posix_spawn, if the CPU dependency loader needs it.
- The offline fixture process disables urllib3's import-time IPv6 capability probe; socket creation remains refused, including IPv6.
- During collection and execution, private-lane path existence checks report unavailable without statting those paths, including optional L7b ancestor pointers.
- It also refuses private-lane input reads.

These Python audit controls are not OS sandboxing or native-extension isolation. Twelve independent guard fixtures exercise these boundaries before any network traffic or child launch, covering refusals, authorized probes and restoration of the existence function after execution.

The runner excludes only these named identities, each with its reason in `HOST_CASES`:

- **2 cases from `tests.test_v2_rowsel_build`** (private scope and rights metadata): `Scope.test_real_scope_is_the_admitted_class` and `FrozenScope.test_real_frozen_scope`.
- **26 cases from `tests.test_v2_rowsel_build_split_v2`, `tests.test_v2_rowsel_build_split_v2g` and `tests.test_split_defaults`.** They read pinned private lane inputs:
  - the NONRAW-SPLIT-A/GROUP-UNION-B owner decisions and relay;
  - the registry port `gsk.py` (`Evidence` class setup, including its two finding-8 order tests; the port-free `EvidenceOrderNoPort` class is not excluded);
  - scale-check sources (`Scope` class setup);
  - earlier builder bytes (`ed00684a`, `c0b989b6`, `fcfe3828`, `28be7f98`).

  The builder bytes are retained in `lane/historical/`, but the preserved cases are not rewritten to read them.
- **All 10 cases of `tests.test_v2_provenance`.** They run a scratch copy of `tools/v2_provenance.py` as a child process, and the guard refuses subprocess launches. The provenance fix is therefore exercised by the producer's runs and mutants, not by this offline lane.

Missing or duplicated identities and any other skip or exclusion fail. The original cases and producer bytes remain intact, and no substitute metadata is created. Synthetic fake S3 clients and scratch ledgers are fixture data; no live object or campaign ledger is accessed.

Four CPUs, a 16 GiB address-space ceiling and a 900-second process timeout bound this check. Its separate receipt records actual case IDs, counts, dependency versions, forbidden attempts, source hashes before/after and the log digest; it is not folded into the existing pytest count.

```sh
cd python
uv run --no-sync python ../deploy/dl-trainonly/run_lane_offline.py --output /durable/custody/new-run
```

The output directory must not exist. Neither it nor the custody checkout may lie under a private-lane input root, because the guard refuses those reads. Other newly discovered lane families have explicit deferred dispositions pending dependency/side-effect audit, bounded host fixtures or separate qualification. Their static presence is not a pass. Every pre-existing CI mapping and test remains intact.

Local evidence for the original import is retained in `/home/ubuntu/jope-evidence/s1-twofold-bootstrap-20261006/`. The original 191-member predecessor archive has SHA256 `24c5404bf0b0ec4f8d40e96008dcb1db40bc36c5954be86aa6e96a8d91899528`. The restage and its predecessors are retained in `/home/ubuntu/jope-evidence/dl-lane-handoff/`, indexed by `RESTAGES.jsonl`.

The producer ran the same 21 modules on the staged bytes with private host inputs present: 555 tests, exit 0 (`tests_run_on_staged_copy` in the manifest). The initial offline harness's library-probe refusal, missing CI dependency, refused urllib3 IPv6 import probe and CI private-path collection and execution refusals are separate retained outcomes. Earlier 329-, 454-, 508- and 515-identity receipts cover the earlier eleven-, sixteen- and nineteen-module selections and do not establish current host qualification. `just required` has an unchanged-main Clippy baseline; this import does not claim a passing complete CPU diagnostic.
