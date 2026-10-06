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
```

The verifier uses committed expected manifest pins, checks every payload and filename, rejects extra files and symlinks, reconstructs the predecessor closure, and parses Python syntax without importing the staged code. These are source custody checks, not executed-family coverage or producer authentication for operational consumers.

The Python CI lane additionally runs exactly 11 audited offline unittest modules (329 cases) through `run_lane_offline.py`. It uses the existing locked Python environment, hides GPUs, removes ambient credential configuration, refuses credential-file reads and network connections, and permits only an exact read-only `/sbin/ldconfig -p` library-cache probe if the CPU dependency loader needs it. All other subprocesses fail. Synthetic fake S3 clients and scratch ledgers are fixture data. No live object or campaign ledger is accessed. Four CPUs, a 16 GiB address-space ceiling and a 900-second process timeout bound this check. Its separate receipt records actual case IDs, counts, dependency versions, forbidden attempts, source hashes before/after and the log digest; it is not folded into the existing pytest count.

```sh
cd python
uv run --no-sync python ../deploy/dl-trainonly/run_lane_offline.py --output /durable/custody/new-run
```

The output directory must not exist. Other newly discovered lane families have explicit deferred dispositions pending dependency/side-effect audit, bounded host fixtures or separate qualification. Their static presence is not a pass. Every pre-existing CI mapping and test remains intact.

Local evidence is retained in `/home/ubuntu/jope-evidence/s1-twofold-bootstrap-20261006/`; the original 191-member predecessor archive has SHA256 `24c5404bf0b0ec4f8d40e96008dcb1db40bc36c5954be86aa6e96a8d91899528`. The initial offline harness's library-probe refusal and subsequent bounded replay are separate retained outcomes. `just required` has an unchanged-main Clippy baseline; this import does not claim a passing complete CPU diagnostic.
