# DL lane source custody

This directory retains the DL lane's 2026-10-06 source handoff for review and reproducible offline fixtures. It does not install, dispatch, adopt or authorize the staged entrypoints.

`lane/MANIFEST.json` is the exact producer manifest `253e593b7537cdecf75a89be24a83804bc63d7e2596e02cc0c28d780a50f2b9e` (restage `20261006T114844Z`): 158 current files and 190 historical versions. Its predecessor chain is retained byte for byte under `lane/predecessor/`, newest first:

- `restage-20261006T082918Z/MANIFEST.json` (`365ec3cecab4e647a160c1f5c2a6baec1f45c5ea102e3180922173ed463f628f`), the manifest this directory carried before;
- `restage-20261006T073946Z/MANIFEST.json` (`6314ef6f33f01878a149cdb36823a78c96d6ae3e2c45cf409d756711c71d6619`);
- `restage-20261006T035515Z/MANIFEST.json` (`009532038cee11ea20075a7291e4b247c7df8e97806946a61bd8dce7dd635802`);
- the original handoff `MANIFEST.json` (`4b8c54da7e4b11cc2f30b02aa88f159d5240134bf4962e670d2bfcddbcd699ad`).

Each manifest's `supersedes_sha256` names the next one; the original names none.

Three predecessor current files that no later manifest retains are preserved there byte for byte:

- the original's `test_v2_rowsel_raw.py` (`2ae3cf61`), from the independently verified predecessor archive;
- the `00953203` version of `test_v2_publish.py` (`bbb8e5de`), which is also the original's;
- the `00953203` version of `test_v2_rowsel_raw.py` (`f33f3d2b`).

The latter two come from the retained `restage-20261006T035515Z` staging and equal the blobs committed here at that time. Every current file of all four predecessors can be restored from the verifier's per-manifest reconstruction map. All 88, 79, 58 and 46 of their historical payloads remain available.

The manifest's `abandoned_unpushed_stagings` records `restage-20261006T113426Z` (`e1ca4e355a5092bc8a7d97a90352ffd35c5d96f1a902c7a411ddca23c9a57ca5`). That staging was never pushed because it picked up the out-of-scope P8 file `dltrain/exclusions.py`. It is not in `lane/`, and the verifier refuses any abandoned staging that appears in the chain, by manifest sha or by path.

The producer staging kept history at a shared `../historical/` path. This import normalizes exactly the manifest-listed history into `lane/historical/`; it does not rename historical results into current evidence. The producer's README inside `lane/` is the `20261006T114844Z` restage note and describes the original staging location. Current and historical source files and all five producer manifests are unchanged.

This handoff carries the owner rulings NONRAW-SPLIT-A and GROUP-UNION-B, the publisher hardening, and the fixes for the PR #218 review findings. Hashes are listed in the table below.

| file | sha256 | role |
| --- | --- | --- |
| `tools/v2_rowsel_build.py` | `8b056648` | non-raw split hook |
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
  - Stripping the payloads is not part of this custody. It remains an open owner decision; no owner ruling accepts the current route in its place.
- **Finding 7:** test-side categorical missing/unseen counts are recorded counts, not inputs to the scaler, the fold assignment or A0.
- **Finding 8 (F4, F5, F6):**
  - **F4:** `target_tied_group` refuses a group column that is bytes-equal to the target, an affine copy (relative RMSE ≤ 1e-10), a function of it, binned with disjoint target ranges, or nearly determined by it (≥ 99%). Module mutants are killed 24/24, including 11 leak mutants, and 0 of the 26 T1 group datasets are refused.
  - **F5:** `split()` requires `partition` and `group_values`.
  - **F6:** target_tied_group reads all-row y ONLY for the dataset-level admit/refuse bit. On admission, y never reaches split()/components() and the split identity is unchanged. On refusal, no context is built.
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

The Python CI lane additionally runs exactly 19 preserved unittest modules through `run_lane_offline.py`: 508 collected identities, of which 472 synthetic cases execute and 36 named cases stay explicitly deferred.

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
- **24 cases from `tests.test_v2_rowsel_build_split_v2`, `tests.test_v2_rowsel_build_split_v2g` and `tests.test_split_defaults`.** They read pinned private lane inputs:
  - the NONRAW-SPLIT-A/GROUP-UNION-B owner decisions and relay;
  - the registry port `gsk.py` (`Evidence` class setup);
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

The producer ran the same 19 modules on the staged bytes with private host inputs present: 508 tests, exit 0 (`tests_run_on_staged_copy` in the manifest). The initial offline harness's library-probe refusal, missing CI dependency, refused urllib3 IPv6 import probe and CI private-path collection and execution refusals are separate retained outcomes. Earlier local329 and 454-identity receipts cover the earlier eleven- and sixteen-module selections and do not establish current host qualification. `just required` has an unchanged-main Clippy baseline; this import does not claim a passing complete CPU diagnostic.
