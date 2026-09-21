# assets/regression — provenance

Every file here is a byte-exact copy (or a lossless compression) of a committed artefact in one of
the read-only source trees the WP12 brief names. Nothing was edited; the SHA-256 values below are of
the **originals** and are re-checked by `crates/jope-arena/tests/elo_regression.rs` and by
`deploy/ci/regression-suite.sh`.

Source trees (read-only at the time of the port, 2026-09-18/19):

| Tree | Role | Identity |
|---|---|---|
| `/home/ubuntu/jain-split/jope-dime/sotarena/` | nested benchmark clone of `github.com/neverhuman/sotarena` | commit `e5db30a07c093661c8f6940d3e22192511f8388b` ("Upgrade Golden45 verification protocol and replay", 2026-07-25) |
| `/home/ubuntu/jain-split/jope-dime/` | the historical jope-dime workspace | commit `c9513e8288434a37ceedd4535b38f702bf35ad7f` (`main`, 2026-09-18) |
| `/home/ubuntu/sotarena/` | the Rust Elo reporter (`src/main.rs`, `config.json`, `data/`) | code ported into `crates/jope-arena`; no data copied |
| `/home/ubuntu/sotarena_pull/` | local score-emission toolkit | its `SotarenaPullArtifact.v1` emission schema is read by `jope-arena emit-scores`; no data copied |

## `sotarena-tabular-regression-v2/` — a minimal `SOTARENA_ROOT`

The inputs `jope-arena elo --track tabular-regression-v2` needs, laid out exactly as in the clone so
the directory is a valid `SOTARENA_ROOT`. `scores.csv` (3,401,605 B) exceeds the 500 KiB fixture cap
and is committed as `scores.csv.zst` (`zstd -19`, 492,076 B); the resolution ledger as `.gz`.
`jope_arena::io` reads `.zst`/`.gz` spellings transparently.

| Fixture path | Original (relative to the clone) | Bytes | SHA-256 of the original |
|---|---|---|---|
| `tracks/tabular-regression-v2/track.json` | same | 1,214 | `c6540865dbbd29e3b9163a3a6fa9aa941939683b5c0485b73826ce598ff1eb22` |
| `registry/datasets.tabular-regression-v2.json` | same | 73,172 | `5ec3d62df5900c3e91d64fa940775e4761ae8f68a346afc379723c249a3b0804` |
| `registry/methods.json` | same | 102,120 | `69c88d86d9ab5c0923a4a272e4e07f68ae84f68a5e5b7addf28811594cbf6930` |
| `results/tabular-regression-v2/scores.csv.zst` | `results/tabular-regression-v2/scores.csv` | 3,401,605 | `5ec868987c2f4286ae44728fde49215ec6e55e7d27e62100d3823744e2957db6` |
| `results/tabular-regression-v2/resolution-ledger.jsonl.gz` | `results/tabular-regression-v2/resolution-ledger.jsonl` | 161,105 | `d28833254eff6787b45c2280f46fc4db0d3dd14a944677c06565a903288d5cd3` |
| `results/tabular-regression-v2/leaderboard.csv` | same (the published ladder) | 56,721 | `2d28ffbd8b5830c6adc39874dfcf98498419ef8c1df6ac5eb135c660013fb24f` |
| `results/tabular-regression-v2/aspfm-acceptance-manifest.json` | same | 3,984 | `963b6c55988959837a351d86a2d135dcccecda6df1fef9cca85d28e068d4a190` |

Not copied (too large or not needed by the ladder): `results/tabular-regression-v2/evidence-manifest.json`
(`1a5031cf6fec153353683a596c0999a51250b1c21c6f45d1741c6baa99040825`, 397 listed files — verified in
place with `jope-arena verify-results`: ok=397 missing=0 mismatched=0), `dataset-metadata/`,
`external-references/`, `source-audits/`, `terminal-receipts/`, `aspfm-replay-receipts/`.

## `elo-tabular-regression-v2.json`

`jope-arena elo --track tabular-regression-v2 --root assets/regression/sotarena-tabular-regression-v2 --json …`:
the recomputed ladder (`jope.arena-elo-snapshot/v1`). Its `leaderboard_sha256` equals the committed
`leaderboard.csv` above, i.e. the recomputation is byte-identical: **jope-dime rank 1, Elo 2259.8,
11 dataset wins, 45/45**, 305 methods, 8,763 rankable rows. The same command with
`--root /home/ubuntu/jain-split/jope-dime/sotarena --check` reported `check=ok`; so did
`--track tabular-regression-v1` (Golden42 development track: jope-dime 2259.9).

## `golden42-baseline.json`

Per-dataset RMSE of the newest committed Golden42 campaign cell: the 42 `JopeDimeDatasetReport.v3`
reports the frozen `jope-dime-regressor score-panel` campaign (panel `grok4p5-42`, kernel
`legacy-v2`, one seed per dataset = `BASE_SEED ^ dataset_hash`) published, as imported into the
benchmark clone:

* reports: `results/external-references/golden42/<dataset_hash>/jope-dime-regressor.json` (42 files;
  each file's SHA-256 is recorded per row as `report_sha256`);
* they are the `method_id = jope-dime-regressor` rows of `results/tabular-regression-v2/scores.csv`
  (`5ec86898…`, 42 rows, `source_kind = reproducible-run-report`) and of
  `results/tabular-regression-v1/scores.csv` (`b943def7e3f4ac49047a46ebcbb3ce9f2fbeeb1b623cb1d73aedd90ce94a6167`);
  the generator asserted `value == metrics.rmse` and `source_sha256 == report_sha256` for all 42;
* panel: `manifests/panel.txt` of the workspace (`b86c8ea5f5ab1051e8aafb4368f8dabdcfd8c740c63eaa1d086ddb72fae0ba75`,
  the constant `PANEL_SHA256` in `crates/jope-campaigns/src/main.rs`), 42 hashes; the same 42 are the
  `DATASETS` array of that binary.

Searched and rejected as baselines: `manifests/advanced-v3/legacy-v2-performance-baseline.v1.json`
(timings, not RMSE), `manifests/advanced-v3/golden42-v3-max-prechange-20260722.v1.json` (one dataset),
`docs/paper/**/claim_ledger.json` (aggregates), `results/tabular-regression-v1/dataset-coverage.csv`
(`jope_best_rmse` is the evolution-wide best, not one campaign cell).

## `seed-population/`

Three real `parallel.seed_population.v1` files with the `parallel.proposal_generation.v1` receipt that
produced each, from the workspace's `parallel/runs/regression-cycle-1/neural/` (the neural-policy
proposal run of regression cycle 1):

| Fixture | Original | Bytes | SHA-256 |
|---|---|---|---|
| `52138ee74df9c896-cut-0/seeds.json` | `parallel/runs/regression-cycle-1/neural/52138ee74df9c896/cut-0/seeds.json` | 11,213 | `da8761cbba2180f71b79263f89aec0ba9dcc2df686f5880a5cc9fb046c90fa4a` |
| `52138ee74df9c896-cut-0/generation.json` | `…/52138ee74df9c896/cut-0/generation.json` | 540 | `b1043efcd2a7d12b5be6f18293d75d018d1ae412a8f5e3060b8f155c2f1fd2ee` |
| `04fcd71f2ccf8242-cut-0/seeds.json` | `…/04fcd71f2ccf8242/cut-0/seeds.json` | 11,949 | `1720ae74a287fb02a1fbb8e5902cec67aea75683c806b0cae76db3ca3a9f8a0d` |
| `04fcd71f2ccf8242-cut-0/generation.json` | `…/04fcd71f2ccf8242/cut-0/generation.json` | 540 | `6942ec1a9bb11df72dbd5c1259fe4e2226e52377d5393899e8f0ee18ab4cc512` |
| `353ae247c133a94a-cut-1/seeds.json` | `…/353ae247c133a94a/cut-1/seeds.json` | 11,593 | `f0340e6548607cb373efec12f2b745e90d07bde9803542197d04f670cea5c629` |
| `353ae247c133a94a-cut-1/generation.json` | `…/353ae247c133a94a/cut-1/generation.json` | 540 | `8fc439ed95eed20bc7b5615b9b08397905328b25b5d7da57834e6722a3059831` |

Every `generation.json` carries `seeds_sha256` equal to its `seeds.json`; all three bind
`checkpoint_sha256 = caa22d011f9744824a0d3ef8c3cbd0037ae7c5a7a65a6fc2ee310c223aa729f2`, which is the
SHA-256 of `parallel/runs/regression-cycle-1/train-1/checkpoints/step-300/state.json` (the run's
`latest.json` / `complete.json` pointer; trainer run `parallel.training_run.v1`, corpus
`4becfff26be84323e1c0965ae076db6bff9817e5c496de4728e1741509cdf72d`, world 8, microbatch 256).
**These are not step-60000 proposals**: no `seeds.json` in `parallel/runs` or `parallel/assets`
references the serving checkpoint (`state.json` `8ac80c26…`); the step-60000 state plane is in the
learning store, not the workspace. The policy-parity step therefore fails precisely until
`JOPE_POLICY_CHECKPOINT` resolves to a checkpoint whose `state.json` hashes to the fixture's
`checkpoint_sha256`; replacing the fixtures with step-60000 ones is a one-line change to the same
step once WP11 publishes them.

## `bench-baseline.json`

No criterion bench exists in the tree (`cargo bench -p feat-math` has nothing to run), so the
baseline is the emitter throughput of `deploy/ci/regression-suite.sh bench`: the release
`feat-math` `emit` example run repeatedly on the committed champion
`crates/feat-math/assets/champions/8944cbdf.dsl`. Host, git sha, rustc and the per-run timings are
inside the file; the suite refuses to compare against a baseline recorded on another host.
