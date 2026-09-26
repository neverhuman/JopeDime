# W1-T01 — Regression input consumer map and release-source inventory

Task: W1-T01. Branch `rr/w1-inputs`, base `7216cad21`. Owner: Claude implementation
subagent, lane `.agent/worktrees/honesty`. The same branch carries the W1-T02 release
contract and the W1-T03 receiving-path checks; those are recorded in
[W01-release-and-preparation.md](W01-release-and-preparation.md).

Observed states:
- **Consumer map: mapped by source reading only.** No access instrumentation was run,
  so nothing here is an execution trace or a proof that a route is unused.
- **Release-source inventory: verified against the custody bytes** by
  `jope-datasets regression-release verify`. The receipt paths are in §2.
- Nothing here is `CI-verified` or `live-qualified`, and nothing closes a register entry.
  The original findings stay open with their original hashes.

Anchors are `path:line`. Files this branch did not modify match base `7216cad21`.
For the modified files:
- `crates/jope-engine/src/prepare.rs`: anchors below line 1040 are unchanged from
  base; later anchors are post-change and marked *(post)*.
- `crates/jope-datasets/src/blobs.rs`: anchors are **base** (`ensure` changed in
  `5b7e591c1`).
- `crates/jope-datasets/src/cache.rs`: the cited lines 98–145 are unchanged.

Anchors marked *(read)* were confirmed by the author. Unmarked anchors come from two
delegated read-only source reads of the same base; treat them as leads, not proof.

## 1. Two input universes

1. **ASM private panel `asm-private-30-v1`.** The ten-regression release is derived here.
   - Input: `benchmarks/panel.json` (frozen, `jope.benchmark-panel/v1`).
   - Preparation: `benchmarks/prepare.py`, which writes the custody root
     `~/.cache/jope/asm-private-30-v1/{downloads,metadata,development,splits,sealed}`.
   - Receiver: `crates/jope-pipeline/src/prepared_data.rs`, which opens the development
     partition only.
   - Consumers: `crates/jope-bench/src/panel/*` and the `jope-learning` OOF scorer.
2. **Legacy "parallel" registry.**
   - Catalog: `registry-real-v1`, 3,859 rows. Custody copy at
     `~/jope-rescue-20260921/core-ip-20260925/catalog.jsonl`, sha256 `ab9fda8d…1120`.
   - Flow: `jope-datasets fetch/link` writes `JOPE_REGISTRY/<hash>/{train,test}.csv,meta.json`;
     then jope-engine `audit`, `prepare`, pointer views, and finally GP, encoding,
     teaching and training.

The universes share datasets. Nine of the ten release members appear by display name
in the historical `registry-real-v1` catalog (§2).
- Four of those rows are capped copies (1,999 rows each).
- Five have exactly the original PMLB row count (209, 500, 250, 323, 222).
- `auto_insurance_price` has no display-name match. Two same-source candidates exist
  under other names.

Catalog listing is **availability**, not a read, encoder, teacher or optimizer record.
It is recorded separately from exposure (§2.3).

## 2. Release-source inventory and membership proof

### 2.1 Frozen inputs (unchanged by this branch)

| Object | sha256 | Last change |
|---|---|---|
| `benchmarks/panel.json` (30 entries, 5 folds, seed 20260922) | `ba7dace4cd535687002cc95538dbc7f7aaf77ec973177c7159b3c6d22471eee0` | `417b3a65d` |
| `benchmarks/panel-selection.json` | `373b537cd87cfdfcf0e6ba4898fdb5bf0ae3a40d3472cdebb848e5c9d2511f48` | `417b3a65d` |
| `benchmarks/prepare.py` (protocol `asm-grouped-development-v1/sklearn-1.7.2`) | `7600de95f0769309d814d80c6723d65f0588124b3e0d1a9875d8155fcc4a91c9` | base |
| `registry-real-v1` catalog (custody) | `ab9fda8d2dea46067b70a42812e9d3c1d9dc6ba025780100df34377e81aa1120` | custody |

**Membership rule:** every frozen panel entry whose original task is `regression`, in
frozen panel order. Nothing is substituted, added or chosen from outcomes. The panel
itself enforces ten entries per task (`prepare.py:258-262`). The release validator
re-checks 30 entries and exactly 10 regression entries
(`crates/jope-datasets/src/regression_release.rs:886`).

### 2.2 Members

All ten are byte-verified against custody: the raw object, source metadata, split,
development partition, and the sealed partition (hashed only, never parsed).

| # | Dataset | Raw sha256 (prefix) / bytes | Rows total / dev / sealed | Fold rows | Group columns | Features | Catalog (`registry-real-v1`) | Relations resolved / unresolved |
|---|---|---|---|---|---|---|---|---|
| 0 | `197_cpu_act` | `d00fd6bac2ed` / 381809 | 8192 / 6553 / 1639 | [1311, 1311, 1311, 1310, 1310] | — | 21 (0 cat.) | listed `fc3812567b5fdd64` (1999 rows) | 0 / 3 |
| 1 | `225_puma8NH` | `0be6203e167c` / 657333 | 8192 / 6553 / 1639 | [1311, 1311, 1311, 1310, 1310] | — | 8 (0 cat.) | listed `e08f0b9931f93686` (1999 rows) | 0 / 0 |
| 2 | `230_machine_cpu` | `937b0a23076f` / 1628 | 209 / 169 / 40 | [34, 32, 37, 33, 33] | — | 6 (0 cat.) | listed `cba8393879afd1dc` (209 rows) | 0 / 3 |
| 3 | `522_pm10` | `ecbabaa5a989` / 13373 | 500 / 402 / 98 | [81, 81, 91, 67, 82] | `day` | 7 (0 cat.) | listed `aebf5d861a8365b4` (500 rows) | 0 / 0 |
| 4 | `529_pollen` | `2e9414f0216e` / 134956 | 3848 / 3078 / 770 | [616, 616, 616, 615, 615] | — | 4 (0 cat.) | listed `c649a4cee87c4321` (1999 rows) | 0 / 1 |
| 5 | `579_fri_c0_250_5` | `d32597bb1043` / 14473 | 250 / 200 / 50 | [40, 40, 40, 40, 40] | — | 5 (0 cat.) | listed `c30ff9405c916718` (250 rows) | 0 / 1 |
| 6 | `auto_insurance_price` | `244c05ca3338` / 3835 | 201 / 162 / 39 | [35, 32, 31, 31, 33] | — | 23 (10 cat.) | not listed by name | 2 / 2 |
| 7 | `690_visualizing_galaxy` | `5fe909e08a62` / 8764 | 323 / 258 / 65 | [52, 52, 52, 51, 51] | — | 4 (0 cat.) | listed `9312d813229df035` (323 rows) | 0 / 1 |
| 8 | `712_chscase_geyser1` | `9231e5c6bb08` / 1031 | 222 / 172 / 50 | [32, 34, 38, 35, 33] | — | 2 (0 cat.) | listed `590210721973399a` (222 rows) | 0 / 2 |
| 9 | `503_wind` | `2cc1e040bdc2` / 198141 | 6574 / 5238 / 1336 | [1066, 1074, 1030, 1032, 1036] | `year`, `month` | 14 (0 cat.) | listed `f24e9383331c43e2` (1999 rows) | 0 / 1 |

**Exclusions.** Twenty entries, all kept in the old panel, none altered. Each is recorded
in the release as `task_not_regression:<task>`.
- Binary, panel indices 10–19: `breast_cancer_wisconsin_diagnostic`, `credit_approval_germany`,
  `congressional_voting_records`, `haberman`, `ionosphere`, `sonar`, `spambase`, `phoneme`,
  `tic_tac_toe`, `horse_colic_surgery`.
- Multiclass, 20–29: `iris`, `wine_recognition`, `car_evaluation`, `balance_scale`,
  `dermatology`, `segmentation`, `vehicle`, `optdigits`, `penguins`, `tae`.
- **Time series: none.** The panel's task census is 10/10/10/0; PMLB has no
  time-series task.

**Receipts.** The release is committed at
`crates/jope-datasets/releases/regression-10-v1/`.
- `release.json`: sha256 `34c94de6…4e3`, `release_id` `859ce689…4d4a`.
- `family-registry.json`: sha256 `757ee4d2…8e70`.

Custody receipts live under
`~/jope-rescue-20260921/regression-remediation-20260926/claude-w0-w12/runs/`:
- `w1-release-derive-20260926T054743Z/` holds `derive.out`, `verify.out` and `source.txt`,
  including the argv and the binary sha256. `verify` re-derived byte-identical release
  bytes with `members_verified: 10`, `sealed_partitions_parsed: false` and
  `network_access: false`.
- `w1-release-verify-20260926T060255Z/` re-verifies the committed artifacts at `5b7e591c1`.

### 2.3 Family and exposure registry (summary)

Encoder, teacher and checkpoint exposure are **`unknown`** for all ten.
- **Basis.** No retained read/use log exists, and the panel records `checkpoint_exposure`
  as "unknown; no unseen-family or uncontaminated-checkpoint claim".
- **Never-seen claims.** Every member's never-seen-family claim is blocked. The blocker
  counts reported by `verify` are 3–6 per member.

Relations:
- **Resolved:** `auto_insurance_price` has two alternate targets, `auto_insurance_losses`
  and `auto_insurance_symboling`. They come from the sha-bound PMLB `metadata.yaml`
  text and are kept inside the family's exclusion unit.
- **Unresolved:** fourteen candidates found from catalog display-name tokens only. Names
  neither prove nor disprove a relation (S02-G5), so they stay unresolved:
  - `197_cpu_act` ↔ `573_cpu_act`, `227_cpu_small`, `562_cpu_small`
  - `230_machine_cpu` ↔ `561_cpu`, `cpu.with.vendor`, "Performance of Computer CPUs"
    (all 209 rows)
  - `529_pollen` ↔ "Pollen Data Challenge"
  - `579_fri_c0_250_5` ↔ `603_fri_c0_250_50`
  - `auto_insurance_price` ↔ `_deprecated_207_autoPrice`, `_deprecated_195_auto_price`
  - `690_visualizing_galaxy` ↔ `678_visualizing_environmental` (shared collection
    prefix only)
  - `712_chscase_geyser1` ↔ two "Old Faithful Geyser Data" rows
  - `503_wind` ↔ "TidyTuesday 2018: Us Wind" (domain word only)
- The generator script and all evidence are retained with the derive receipt
  (`make_registry.py`).

## 3. Consumer map — ASM panel (release universe)

| Stage | Receiver | Files opened | Parent receipts / checks | Cache / retry / env |
|---|---|---|---|---|
| Acquisition (outside any worker) | `benchmarks/prepare.py` `prepare_one` `:117` *(read)* | `downloads/<sha>`, `metadata/<id>.yaml` (frozen mode `:125-131`) | Digest of raw and metadata against the panel `:131`; `keep` refuses changed immutable files `:36` | **Network fetch when a staged blob is absent** `:128`, `:130`, then digest-checked. Output must be outside the checkout `:255`. |
| Preparation | `prepare.py` `:150-208` *(read)* | Writes `development/`, `sealed/`, `splits/<id>.json` `:207-208` | Frozen mode requires the result to equal the frozen entry `:243` | sklearn 1.7.2 pinned `:84-86` |
| Release derivation (**new**) | `regression_release.rs` `load_staged` `:1128`, `read_exact_object` `:1075`, `derive` `:932`, `verify_staged` `:413` | Exactly `downloads/<sha>`, `metadata/<id>.yaml`, `development|splits|sealed/<id>.json`, for regression members only | Each object's bytes are hashed against the panel; symlink components are refused; the opened inode must be the inspected one | **No fallback**: a missing object fails. No cache, no network. |
| Development receiver | `jope-pipeline/src/prepared_data.rs` `load_development` `:76-106` *(read)* | `development/<id>.json` only | Symlink refusal `:79-91`; sha256 **before** parse `:114-118`; schema/source/features `:127-150`; ids, groups and folds `:172-212` | Checks the symlink then opens without `O_NOFOLLOW`; the sha check bounds the damage. The release is **not yet consumed** here (open, §6). |
| Cut / fitting rows | `DevelopmentData::cut` `:314`, `partition` `:337` *(read)* | In memory | Exact fold assignment; no recut | Hold labels stay in the parent process `:337-343`. |
| Context producer | `jope-policy/src/context_producer/input.rs` `TrainingInput::prepare` `:84-217` *(read)* | Payload JSON (`worker.rs:218`) | **Self-certified** row/label identity (`:90-114`); dataset and prepared sha are format-checked only `:13-24`. Controller pins in `admission.rs`. | Training labels only; `deny_unknown_fields` blocks holdout fields. |
| GP fit / score / teacher | `jope-bench/src/panel/inference_inputs.rs` `:37-44`, `:110-113`; `origin_proof.rs` `:379-380`, `:441-446`; `teaching.rs` `:43` | Fitting manifests and assignments; `prepared_root` from configuration | `Artifact::verify` hashes by path, then callers re-read the file | Fitting membership excludes outer fold 0 and sealed rows. |
| OOF scorer / learning | `jope-learning/src/eligibility.rs` `:216-252`, `:960-965` | Development blob from `$JOPE_STATE` (re-hashed on read) | Held-out fold targets are scored in the scorer | Without an inner partition, the context fitting scope includes the scored holds (`eligibility/context.rs:63-68`). This is W3 scope. |

## 4. Consumer map — legacy parallel path (jope-engine)

| Stage | Receiver | Files opened | Parent receipts / checks | Cache / retry / env-derived roots |
|---|---|---|---|---|
| Catalog / acquisition | `jope-datasets` `Fetcher` (`cache.rs:98-145` *(read)*, `blobs.rs:105-140` *(read)*) | `by-id/<hash>/manifest.json`, `catalog/LATEST`, `catalog/<v>/catalog.jsonl`, `blobs/sha256/<sha>` | Blobs are sha-checked on download | **Manifest is cache-first** `:98-103`. **Remote `LATEST` beats local** `:122-129`. **No LATEST → whatever manifests are cached** `:130-131`. **Cached catalog read with no hash** `:133-135`. **Blob cache hit is size-only** `blobs.rs:129-134`. **Network download on miss** `:137-140`. Root: `JOPE_HOME` → `XDG_CACHE_HOME` → `HOME` → relative `.cache` (`paths.rs:6-17` *(read)*). `JOPE_DATASETS_URL` may be `file://` or a bare path (`store.rs:38-66` *(read)*). |
| Registry roots | `jope-paths` `jope_home` `:80`, `jope_registry` `:124`, `quant_root` `:161` *(read)* | — | — | **Falls back to the current directory** when no marker is found; `JOPE_REGISTRY` is taken verbatim. |
| Audit | `jope-engine/src/audit.rs` `inspect` `:104-114` *(read)* | `train.csv` **and `test.csv`** (parsed `:107-108`) | — | **`stats` persists the test label value set (≤256 distinct) into `audit.json` `test.target_values` `:95-99`.** No Rust consumer reads the field; only the struct declares it (`manifest.rs:161`). |
| Family admission | `families.rs` `binding` `:353-380` *(read)*; `prepare.rs` `:917-922` *(read)* | Registry JSON | Registry validation re-verifies growth only | A binding error is swallowed with `if let Ok` and drops the candidate; the result is a refusal unless another identity matches. |
| Preparation | `prepare.rs` `run_with_admission` `:896` *(read)* | `audit.json`, `meta.json` (parsed `:971`, hashed `:973`), `train.csv` (**parsed unhashed** `:982`), `transfer.json`, backing manifest, backing CSV | Metadata vs audit; backing sha in the allowlist `:1020-1028` | **`mirrored_backing_source` re-roots a missing recorded path under env-derived `quant_root()/../hyper/data`** `:553-561`, `:576-580`. `test.csv` is never opened. **New:** view semantics `:1044-1049` *(post)* and cut membership `:1067` *(post)*, both before output. |
| View receiver | `load_view_with_root` `:1301` *(post)* → `load_raw_view_in_root` `:1365` *(post)* | `prepared.json`, `cut-N/preprocessing.json`, backing CSV | `resolve` refuses symlinks per component (`manifest.rs:71-82` *(read)*); preprocessing hash; backing hash; final transformed-view hash against the cut. **New:** membership, column semantics, target-copy and learned-state binding (§5). | **`load_view` without a root walks up to the nearest `manifests/`+`Cargo.toml` ancestor** (`workspace_root` `:1281` *(post)*). Hash-then-reopen of preprocessing and backing. `prepared.json` is not hashed inside the receiver; callers bind it (`label_admission.rs:254-258` *(read)*). |
| Encoder / context | `encoding.rs:97`, `components.rs:104`, `:430`, `hyperion.rs:161` *(read, grep)* | Training views | `task_contract::require_original_gpu_grant` always refuses (`task_contract.rs:88-92` *(read)*), so encoding/contexts cannot dispatch | `hyperion.rs:161` uses `load_view` with no root even though a root is in scope. The Hyperion output cache is trusted by its own receipt. |
| GP / fitter | `gp.rs:158` fit-view, `:421-422` evaluate-view, `:1339-1340` search *(read, grep)*; `native_scq.rs:451` | Training (+ validation) views | Seeds bound to prepared and training hashes; the worker re-hashes the memfd | **Qualification mode accepts any partition, including `meta_test`**: `gp.rs:1322-1325` *(read)*. |
| Scorer | `gp.rs` `score` `:347-401` *(read)* | Selection receipt, model, **sealed test CSV** | Selection receipt and evidence; test hash against `receipt.final_test_sha256` `:366-368` | The test path and its hash come from caller-supplied files and are not bound to `Prepared.original_test_sha256`. Hash, then `Table::read` reopens `:369-370`. |
| Teacher | `label_admission.rs` `verify_inputs` `:240-268` *(read)*; `teaching_store.rs` `:571-612` | `prepared.json` (hashed), training **and validation** views | Search receipt, completion and authority checks `:428-563` | Shard reuse is checked against its own `shard.json` (`teaching_store.rs:142-147`). |
| Trainer | `training.rs:806-841`; `corpus.rs:196` | Corpus index, `records.bin`, vocabulary, checkpoints | Corpus hash; meta_train-only gradient | `JOPE_TEACHING_CACHE` socket; resident tensor caches keyed on claimed hashes. |
| Worker input cache | `native_input_cache.rs` `bound_hash` `:224-284` *(read)*; `enter` is called only at `smartcluster_worker.rs:494` *(read, grep)* | Declared inputs | A hit reuses the digest from the stat fingerprint without re-reading bytes: ext4 only, read-only, ctime/mtime at least 2 s old `:209-223` | Scope ends when verification returns. |

## 5. Substitution and access routes

Status key: **explained** = the route is identified here; **enforced** = this branch
refuses it at the receiver; **open** = still reachable.

| # | Route | Anchor | Kind | Status / owner |
|---|---|---|---|---|
| R1 | Missing recorded backing path re-rooted under env-derived `quant_root` | `prepare.rs:553-561`, `:576-580` | Alternate location | Explained; open. The result is recorded and must appear in the backing allowlist. W1-T05. |
| R2 | `load_view` root found by walking up from the prepared path | `prepare.rs:1281` *(post)* | CWD/ancestry root | Explained; open. Callers that pass a root are safe. W1-T05. |
| R3 | Blob cache hit on size only | `jope-datasets/src/blobs.rs:129-134` (base) | Stale cache | **Enforced by this branch** (`5b7e591c1`, W1-T05 fixture scope). A hit is re-hashed; a stale entry is removed, re-downloaded online and refused offline; its consumer copy is not left servable. The jope-models bundle links remain open. |
| R4 | Catalog and manifest are cache-first, not pinned; `LATEST` resolution; fallback to cached rows | `jope-datasets/src/cache.rs:98-145` | `latest` pointer / stale cache | Explained; open. Acquisition-only unless it runs in a worker (unproven; W7). |
| R5 | Network fetch on cache miss | `blobs.rs:137-140`, `cache.rs:104-111`, `:136-144` | Network fallback | Explained; open. `--offline` and `JOPE_OFFLINE` disable it. There is no worker isolation proof (W7). |
| R6 | `prepare.py` frozen-mode network fetch when a staged blob is absent | `prepare.py:128`, `:130` | Network fallback (acquisition) | Explained; digest-checked at the acquiring component. The release derivation refuses absent objects. |
| R7 | `audit.json` persists sealed test label value sets | `audit.rs:95-99`, `:107-108` | Outer labels in a shared artifact | Explained; **open**. No Rust reader; it is an exposure surface for anything reading the audit. W1-T04. |
| R8 | Hash, then reopen by path | `prepare.rs:971-973`; backing `:1365ff` *(post)*; `label_admission.rs:477-481`; `gp.rs:366-370` | Replacement race | Explained. Views are mitigated by the final transformed hash and, now, by the learned-state refit check. Consuming validated bytes remains open (W1-T05). |
| R9 | Worker input cache reuses digests by stat fingerprint | `native_input_cache.rs:224-250` | Cache hit without re-read | Explained; documented safeguard with existing tests. W1-T05 still needs cache hit/miss/resume receiver tests. |
| R10 | `JOPE_DATASETS_URL` (`file://`, bare path); credential chain | `jope-datasets/src/store.rs:38-66`; `credentials.rs` | Configurable source / broad credentials | Explained; open. W1-T05 credential removal. |
| R11 | `jope_home` falls back to the current directory; `JOPE_REGISTRY` taken verbatim | `jope-paths/src/lib.rs:80-126` | Env-derived roots | Explained; open. W1-T05. |
| R12 | Context-producer training identity is self-certified | `jope-policy/.../input.rs:13-24`, `:90-114` | Unbound identity | Explained; open. Binding it to the release's `development_alignment_sha256` is W2 (outside this partition). |
| R13 | Qualification search accepts sealed `meta_test` | `gp.rs:1322-1325` | Sealed partition reachable | Explained; open. W5 / W1-T04. |
| R14 | `score` test path and hash are not bound to `Prepared` | `gp.rs:347-401` | Caller-chosen outer file | Explained; open. W1-T04. |
| R15 | Starforge `data.rs` reads `test.csv`, imputes unparsable values to 0, no hashing | `jope-encoders/src/data.rs:40-121` | Unverified inputs | Explained; open. Not on the release path. |
| R16 | Preparation accepted target-as-feature, target copies, ambiguous headers, duplicated/reordered/overlapping membership and learned state not fitted on training membership | `prepare.rs` receiver and producer | Semantics | **Enforced by this branch** (W1-T03). 11 failing reproductions now pass. |

Acceptance status against W1-T01: every route above is **explained** with an owner.
Several are **not closed**; W1-T05 is still blocked until they are. W1-T03 was
implemented only where its receiver is in this partition (§6).

## 6. Held-out and outer label read sites

| Label kind | Sites |
|---|---|
| Final / sealed test | `gp.rs:366-391` (`score`); `audit.rs:107-108` (parsed and value-set persisted); starforge `data.rs:42`, `:68` |
| OOF hold folds | `jope-learning/src/eligibility.rs:960-965`, `:510-514`; `jope-bench/src/panel/origin_proof.rs:379-380` |
| Cut validation | `gp.rs:422`, `:988-1013`; `label_admission.rs:266`; `native_genome.rs:349`; `prepare.rs` validation views |
| `meta_validation` families | `training.rs:910-914`, `:1516-1526`; `monitoring.rs:67-76`; `auxiliary_training.rs:301-320`; `pipeline.rs:609-619`; `teaching_store.rs:1142` |
| Sealed `meta_test` | `gp.rs:1322-1325` (qualification); `jope-learning/src/sources/champions.rs:67-75` |

## 7. Open items this map hands on

1. **The release receiver.** `jope-pipeline/src/prepared_data.rs` does not yet check
   the release's `development_alignment_sha256` or its family/exposure scope. It already
   checks exact bytes. Integrating it means editing `jope-pipeline`, which is outside
   this lane's partition. `RegressionRelease::verify_development` is the ready entry point.
2. **R7, R13, R14** belong to W1-T04 (score and label projection).
3. **R1, R2, R4–R6 and R8–R11** belong to W1-T05 (worker closure); R3 is fixed in the
   fixture scope. Live proof is W7.
4. The **unresolved relations** in §2.3 need an independent review, which may be
   content-level (row overlap or schema fingerprints). The committed registry leaves
   them unresolved, and any resolution is a new registry version and a new release
   identity.

## 8. Files read for this map

By the author:
- `crates/jope-engine/src/{prepare,families,label_admission,native_input_cache,manifest,audit(55-114)}.rs`
- `crates/jope-engine/tests/{data_contracts,view_contracts(1-270),gp_admission}.rs`
- `crates/jope-engine/tests/support/fit_authority.rs` (360-424)
- `crates/jope-pipeline/src/prepared_data.rs`
- `crates/jope-policy/src/context_producer/input.rs`
- `crates/jope-datasets/src/{cli,hash,blobs(100-150),cache(95-150),paths(1-25)}.rs`
- `benchmarks/{panel.json,panel-selection.json,prepare.py}`
- the custody metadata YAMLs and catalog
- `gp.rs` (1316-1330, 345-372) and `task_contract.rs` (85-95)

Delegated read-only source reads (anchors unmarked above):
- the jope-datasets and jope-paths modules
- the engine's audit, families, encoding, hyperion, gp, service, pipeline,
  smartcluster_worker, native_loop, native_genome, teaching_store, corpus, training
  and champions modules
- jope-policy's context producer
- jope-encoders' data, cli and bundle modules
- jope-bench `panel/*`
- jope-learning's eligibility, state and hwm modules
