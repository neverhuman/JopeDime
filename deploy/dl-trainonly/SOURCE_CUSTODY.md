# DL lane source custody

This directory retains the DL lane's 2026-10-06 source handoff for review and reproducible offline fixtures. It does not install, dispatch, adopt or authorize the staged entrypoints.

`lane/MANIFEST.json` is the exact producer manifest `009532038cee11ea20075a7291e4b247c7df8e97806946a61bd8dce7dd635802`: 147 current files and 58 historical versions. The original handoff `4b8c54da7e4b11cc2f30b02aa88f159d5240134bf4962e670d2bfcddbcd699ad` is retained under `lane/predecessor/MANIFEST.json`. Its omitted old `test_v2_rowsel_raw.py` is preserved there byte for byte from the independently verified predecessor archive. Every original current file can be restored from the verifier's reconstruction map, and all 46 original historical payloads remain available.

The producer staging kept history at a shared `../historical/` path. This import normalizes that exact manifest-listed history into `lane/historical/`; it does not rename historical results into current evidence. The producer's README inside `lane/` describes the original staging location. Current and historical source files and both producer manifests are unchanged.

The current raw builder supports explicit `--split-version rowsel-raw-split-v2`, implemented in `rowsel_raw_split_v2.py` at `97340df3`. The old feature-ordered split in `rowsel_raw.py` (`18eac154`) remains for historical replay; the v2 must-fail fixture demonstrates why it cannot satisfy the new invariant. The preserved CLI still defaults to v1 (tools/v2_rowsel_raw.py:951), so source custody must not be treated as a new safe operational default. A qualified v2 invocation must explicitly select v2; changing the preserved default requires a separately identified source correction and new receipt. Original dated plans, runbooks, dependency inventories, runtime defaults and command examples inside the custody tree are preserved observations. They are not current compute, credentials, publication or pool-control grants. Nonraw grouping policy, fresh production input-list admission, live rebuild red-team, G0 provenance/controller integration, S1 pilot, policy training and serving promotion remain independently qualified gates.

## Checks

Run the stdlib custody verifier and refusal fixtures from the repository root:

```sh
python3 deploy/dl-trainonly/verify_lane.py
python3 deploy/dl-trainonly/test_verify_lane.py
python3 deploy/dl-trainonly/test_offline_guard.py
```

The verifier uses committed expected manifest pins, checks every payload and filename, rejects extra files and symlinks, reconstructs the predecessor closure, and parses Python syntax without importing the staged code. These are source custody checks, not executed-family coverage or producer authentication for operational consumers.

The Python CI lane additionally runs exactly 11 preserved unittest modules (329 collected identities): 327 synthetic cases execute, and two genuine private-metadata cases remain explicitly deferred for host qualification through `run_lane_offline.py`. It uses the existing locked Python environment, hides GPUs, removes ambient credential configuration, refuses Python credential-file reads, every Python socket audit event (including socket creation and UDP sends), and direct Python subprocess/spawn/fork/exec launches. It permits only an exact read-only `/sbin/ldconfig -p` library-cache probe through Popen or posix_spawn if the CPU dependency loader needs it. Independent guard fixtures exercise these boundaries before any network traffic or child launch. The offline fixture process disables urllib3's import-time IPv6 capability probe; socket creation remains refused, including IPv6. These Python audit controls are not OS sandboxing or native-extension isolation. During import, private-lane path existence checks report unavailable without statting those paths. The runner excludes only `Scope.test_real_scope_is_the_admitted_class` and `FrozenScope.test_real_frozen_scope` from `tests.test_v2_rowsel_build`; missing or duplicated identities and any other skip/exclusion fail. The original cases and producer bytes remain intact; no substitute metadata is created. The audit also refuses private-lane input reads. Synthetic fake S3 clients and scratch ledgers are fixture data. No live object or campaign ledger is accessed. Four CPUs, a 16 GiB address-space ceiling and a 900-second process timeout bound this check. Its separate receipt records actual case IDs, counts, dependency versions, forbidden attempts, source hashes before/after and the log digest; it is not folded into the existing pytest count.

```sh
cd python
uv run --no-sync python ../deploy/dl-trainonly/run_lane_offline.py --output /durable/custody/new-run
```

The output directory must not exist. Other newly discovered lane families have explicit deferred dispositions pending dependency/side-effect audit, bounded host fixtures or separate qualification. Their static presence is not a pass. Every pre-existing CI mapping and test remains intact.

Local evidence is retained in `/home/ubuntu/jope-evidence/s1-twofold-bootstrap-20261006/`; the original 191-member predecessor archive has SHA256 `24c5404bf0b0ec4f8d40e96008dcb1db40bc36c5954be86aa6e96a8d91899528`. The initial offline harness's library-probe refusal, missing CI dependency, refused urllib3 IPv6 import probe and CI private-path collection refusal are separate retained outcomes. Earlier local329 receipts included327 synthetic cases plus two private-metadata checks; their all-synthetic scope field is superseded by this correction, and those checks do not establish current host qualification. `just required` has an unchanged-main Clippy baseline; this import does not claim a passing complete CPU diagnostic.
