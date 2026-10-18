# Algorithmic Self-Play Additive Wire-Contract & Identity Catalogue (`I02`)

- **Normative Envelope Contract**: `jope.canonical-json/v1`
- **Schema File**: `schemas/jope.canonical-json.v1.schema.json` (`$id`: `https://github.com/neverhuman/JopeDime/schemas/jope.canonical-json.v1.schema.json`)
- **Executable Seam**: `crates/jope-records/src/wire_contract.rs` (`jope_records::wire_contract`)
- **Conformance Suite**: `crates/jope-records/tests/wire_contract.rs` + `crates/jope-records/tests/fixtures/wire-contract-v1/`
- **Merged Base Pin (`I01`)**: `1a8a66c502ca2cede77c197b4f5b5bbc24ff398e` (PR `#357`)
- **Adoption Status**: **FROZEN / ADOPTED FOR RING-1 (`I03`, `I07`, `I46`, `I63`) AND DOWNSTREAM SELF-PLAY DAG**
- **Coordinator & Sole Serializer**: `2830191.jopedime_agy`

---

## 1. Executive Summary & Adoption Receipt

Task `I02` freezes one additive wire-contract catalogue, canonical UTF-8 JSON byte encoding (`jope.canonical-json/v1`), length-prefixed domain-separated SHA-256 identity projection, adapter matrix (`native`, `inference`, `storage`), and outcome-authority boundary (`VerifiedMeasuredOutcome` vs `PredictedOutcome`, `SyntheticOutcome`, and `MissingOutcome`).

### 1.1 Zero Legacy Rehash & Zero ADR-0004 Identity Drift (`I02-B1`)

1. **Zero `Cargo.lock` or `Cargo.toml` Mutation**: `crates/jope-records/src/wire_contract.rs` is implemented in pure Rust using `jope-records`'s existing `sha2` dependency. Neither `Cargo.toml` nor `Cargo.lock` is modified, so the ADR-0004 evaluator identity (`crates/feat-math/tests/fixtures/compat_freeze/evaluator_identity.v1.json`) and lineage source hashes remain bit-for-bit unchanged.
2. **All 29 Existing Schemas & Historical Hash Schemes Preserved**: Every existing `*.schema.json` file (`14` in `schemas/` and `15` in `crates/` and `deploy/`), `crates/jope-learning/src/canon.rs`, `crates/jope-learning/src/record.rs`, `crates/jope-records/src/pyjson.rs`, and `crates/jope-records/src/claims.rs` remain read-only with their byte digests and historical IDs verified by `crates/jope-records/tests/wire_contract.rs`.

### 1.2 Frozen `I02` Artifact Digests

| Artifact Path | Role | SHA-256 Digest |
| --- | --- | --- |
| `schemas/jope.canonical-json.v1.schema.json` | JSON Schema Draft 2020-12 envelope schema (within `crates/jope-datasets/src/schema.rs` supported vocabulary) | `04668c4457a044d9caf7beeea7e319b5e52303c25366208d98316c2a03a59ed9` |
| `crates/jope-records/src/wire_contract.rs` | Strict lexer, canonicalizer, `DomainSeparatedId`, `ExactF32`/`ExactF64`, domain adapters, and `VerifiedMeasuredOutcome` gate | `7aa274a7a2c0e8c9873ed3221d5f833f5d517c4110593bbaa438c947b5c1cfc5` |
| `crates/jope-records/tests/wire_contract.rs` | Rust integration & cross-language conformance test suite (`8` integration tests + `4` doctests including `3` `compile_fail` proofs) | `42b4989e48e7172d5fda1b4c992fd4880589d61b8613cc223246124c5b354b92` |
| `crates/jope-records/tests/fixtures/wire-contract-v1/positive_vectors.json` | `10` positive golden byte and domain-separated SHA-256 conformance vectors (`V01`..`V10`) | `89da4c16a90f9052d79c0748862a9c4417aaeafe14dc59388df9db306fdf7f68` |
| `crates/jope-records/tests/fixtures/wire-contract-v1/negative_vectors.json` | `18` negative lexical, numeric, cycle, and authority refusal vectors (`N01`..`N18`) | `f43d3f75a39cd33bd112eadea02505782cb347bcba0a2dd7e470263bbf4c352a` |
| `crates/jope-records/tests/fixtures/wire-contract-v1/verify_vectors.mjs` | Independent Node.js ESM recursive-descent parser, canonicalizer, and SHA-256 verifier | `a4a724b986f3596127614183d7a2feecdd322578f692c276ed074222f49e0d4d` |

---

## 2. Inventory of Existing Committed Schemas & In-Code Contracts

### 2.1 Top-Level Product Schemas (`schemas/*.schema.json` — 14 Pre-Existing + 1 New Additive)

| Schema Path | `$id` | Discriminator | Owning Rust Module / Type | Hash / Identity Rule | Readers & Writers | Pinned SHA-256 |
| --- | --- | --- | --- | --- | --- | --- |
| `schemas/jope.benchmark-panel.v1.schema.json` | `<none>` | `schema: "jope.benchmark-panel/v1"` | `crates/jope-datasets/src/regression_release.rs` | File SHA-256 manifest pin | `jope-datasets`, `jope-learning::eligibility` | `bb641dfb8190053c10dcc5484bcc58b484048a2b5178ea0163fb152f010d5369` |
| `schemas/jope.dataset-catalog.v1.schema.json` | `https://veox.ai/schemas/jope.dataset-catalog.v1.schema.json` | `schema: "jope.dataset-catalog/v1"` | `crates/jope-datasets/src/catalog.rs` (`Catalog`) | File SHA-256 + per-entry `sha256` | `jope-datasets`, `jope-bench` | `6a226b627c9d28fb04b7ca520d37ec3c8def6ac6a2d14d6b131b0a8ed08aa563` |
| `schemas/jope.dataset-collection.v1.schema.json` | `https://veox.ai/schemas/jope.dataset-collection.v1.schema.json` | `schema: "jope.dataset-collection/v1"` | `crates/jope-datasets/src/catalog.rs` (`CollectionManifest`) | File SHA-256 + member digests | `jope-datasets`, `jope-records` | `34f85f5a076beb7f0daaaad25ba9e63c905e3d062538adabfba4bed8efb75b29` |
| `schemas/jope.evaluation-request.v1.schema.json` | `<none>` | `schema: "jope.evaluation-request/v1"` | `crates/jope-pipeline/src/experiment.rs` | Request payload SHA-256 | `jope-pipeline` | `7f2fbb3c8a1e55a405690fda421836be499c0e7388759fb96a163de0c70556b1` |
| `schemas/jope.evaluation-result.v1.schema.json` | `<none>` | `schema: "jope.evaluation-result/v1"` | `crates/jope-pipeline/src/experiment.rs` | Result payload SHA-256 | `jope-pipeline` | `b539f787dd163267c09b0edd2c7988af63ae7b88018b4739eac39dfe17346858` |
| `schemas/jope.evaluation-result.v2.schema.json` | `<none>` | `schema: "jope.evaluation-result/v2"` | `crates/jope-pipeline/src/experiment_v2.rs` | Result v2 payload SHA-256 | `jope-pipeline` | `c29d719be53781abad28b046298ea9c7046584434f56aa75f492116159387925` |
| `schemas/jope.experiment.v1.schema.json` | `<none>` | `schema: "jope.experiment/v1"` | `crates/jope-pipeline/src/experiment.rs` | Experiment manifest SHA-256 | `jope-pipeline`, `jope-learning::eligibility` | `84c3309caeca4226bcb1a0aa16ca194b1194a3e114efbf958bc153c6af604f62` |
| `schemas/jope.hwm.v1.schema.json` | `https://veox.ai/schemas/jope.hwm.v1.schema.json` | `schema: "jope.hwm/v1"` | `crates/jope-learning/src/record.rs` (`HwmRecord`, `HwmIdentity`) | Plain `SHA256(canonical_json(HwmIdentity))` (`crates/jope-learning/src/canon.rs`) | `jope-learning` (`hwm.rs`, `record.rs`) | `65a11eafdca127d5c6d14dc4a041f135a155526fb051b91df205457985267f30` |
| `schemas/jope.model-bundle.v2.schema.json` | `https://github.com/veox-ai/JopeDime/schemas/jope.model-bundle.v2.schema.json` | `schema: "jope.model-bundle/v2"` | `crates/jope-models/src/manifest.rs` | Bundle manifest + asset SHA-256 | `jope-models`, `feat-math` | `886f20d17290c9c31b163785543a240c6befd71d144a0fc5e48ab5c899032a75` |
| `schemas/jope.pipeline-receipt.v1.schema.json` | `https://github.com/veox-ai/JopeDime/schemas/jope.pipeline-receipt.v1.schema.json` | `schema: "jope.pipeline-receipt/v1"` | `crates/jope-pipeline/src/receipt.rs` | Receipt file SHA-256 | `jope-pipeline` | `d7e34e5835f87ff7203f9818d1c91c162b7741896f730c7b90945f50febb35a2` |
| `schemas/jope.pipeline-receipt.v2.schema.json` | `https://github.com/veox-ai/JopeDime/schemas/jope.pipeline-receipt.v2.schema.json` | `schema: "jope.pipeline-receipt/v2"` | `crates/jope-pipeline/src/receipt.rs` | Receipt file SHA-256 | `jope-pipeline`, `jope-learning::sources::pipeline` | `830f424e09ec8ea13f2506cd47fc9773ac1a85e028b14b01f22f1ff6611d98b4` |
| `schemas/jope.pipeline-receipt.v3.schema.json` | `https://github.com/veox-ai/JopeDime/schemas/jope.pipeline-receipt.v3.schema.json` | `schema: "jope.pipeline-receipt/v3"` | `crates/jope-pipeline/src/receipt.rs` | Receipt file SHA-256 | `jope-pipeline` | `1f5909a1ed5ff9f92e24ef7644f014104b9ec6a260406484d4a104ea70dfa19f` |
| `schemas/jope.pipeline-receipt.v4.schema.json` | `https://github.com/neverhuman/JopeDime/schemas/jope.pipeline-receipt.v4.schema.json` | `schema: "jope.pipeline-receipt/v4"` | `crates/jope-pipeline/src/kernel2/receipt.rs` | Receipt file SHA-256 | `jope-pipeline::kernel2` | `d71906664e8e0818831af228416dbd763350d670b68092b4f2c9aa3547c3501b` |
| `schemas/jope.score.v1.schema.json` | `https://veox.ai/schemas/jope.score.v1.schema.json` | `schema: "jope.score/v1"` | `crates/jope-learning/src/record.rs` (`ScoreRecord`) | Plain `SHA256(canonical_json(HwmIdentity))` (`hwm_id`) | `jope-learning` (`record.rs`, `schema_agreement.rs`) | `4d897333f6cd85348a1f062cddde8c1a7efc99c45a36145739019bac949632b4` |
| `schemas/jope.canonical-json.v1.schema.json` (**NEW, `I02`**) | `https://github.com/neverhuman/JopeDime/schemas/jope.canonical-json.v1.schema.json` | `envelope_schema: "jope.canonical-json/v1"` | `crates/jope-records/src/wire_contract.rs` (`EnvelopeFrame`, `CanonicalPayload`, `DomainSeparatedId`) | `SHA256(schema_utf8 \|\| 0x00 \|\| u64_be(len) \|\| canonical_payload_bytes)` | `jope-records::wire_contract` (`NativeEnvelopeAdapter`, `InferenceEnvelopeAdapter`, `StorageEnvelopeAdapter`) | `04668c4457a044d9caf7beeea7e319b5e52303c25366208d98316c2a03a59ed9` |

### 2.2 Additional Committed Schemas Outside `schemas/` (15 Files, `F8`)

| Schema Path | `$id` | Discriminator | Owner / Surface | Pinned SHA-256 |
| --- | --- | --- | --- | --- |
| `crates/smartcluster/schemas/jain.appliance-role-v2.schema.json` | `jain.appliance-role/v2` | Role configuration object | Vendored SmartCluster (`crates/smartcluster/`) | `c3439ddfc4a1875bca7e525e3c43cd591881be9a8cdc842f723271ced73f22a4` |
| `crates/smartcluster/schemas/jain.node.storage-config-v1.schema.json` | `jain.node.storage-config/v1` | `schema: "jain.node.storage-config/v1"` | Vendored SmartCluster (`crates/smartcluster/`) | `7592184b9f9334f2861f45820d859a1d88559275420c3aa63a39cc7fe98212cb` |
| `crates/smartcluster/schemas/jain.node.storage-control-v1.schema.json` | `jain.node.storage-control/v1` | Storage control command | Vendored SmartCluster (`crates/smartcluster/`) | `ec361ad1e8208d1110fe6c51305928d66b51e8c8f931d7259926924e29a9d337` |
| `crates/smartcluster/schemas/jain.node.storage-telemetry-v1.schema.json` | `jain.node.storage-telemetry/v1` | Storage telemetry payload | Vendored SmartCluster (`crates/smartcluster/`) | `220984a1cdc1f8b5c44557fce6251ca4c81cf2f7761c382c63aa726ebb18a255` |
| `crates/smartcluster/schemas/jain.storage-fragment-receipt-v1.schema.json` | `jain.storage-fragment-receipt/v1` | Storage fragment receipt | Vendored SmartCluster (`crates/smartcluster/`) | `bd658840555f80d2f22a76eb5337faf4900dd932d73d59cf6d344f54802279c3` |
| `deploy/dl-trainonly/lane/src/dl_training_v1/schemas/jope.dl-gp-exposed-record.schema.json` | `jope.dl-gp-exposed-record/v1` | `origin`, `split_role` | Historical DL train-only lane (`deploy/dl-trainonly/`) | `919dc3b7b5315c04f0c468861df680f4388727d4a91d6e9175774c53ddcf9a7b` |
| `deploy/dl-trainonly/lane/src/dl_training_v1/schemas/jope.dl-gp-lineage-record.schema.json` | `jope.dl-gp-lineage-record/v1` | `backend_source`, `origin`, `split_role` | Historical DL train-only lane (`deploy/dl-trainonly/`) | `e6000473cd6e3fac550353368963008cc812e0140543daaa365ec560f1a52e11` |
| `deploy/dl-trainonly/lane/src/dl_training_v1/schemas/jope.dl-gp-refused-record.schema.json` | `jope.dl-gp-refused-record/v1` | `reason`, `split_role` | Historical DL train-only lane (`deploy/dl-trainonly/`) | `49f2ddf2733344e79c4a5fe156a0342dc940ff7a9359b3116e7535d83487c459` |
| `deploy/dl-trainonly/lane/src/dl_training_v1/schemas/jope.dl-gp-train-context.schema.json` | `jope.dl-gp-train-context/v1` | `cohort`, `label_status`, `split_role` | Historical DL train-only lane (`deploy/dl-trainonly/`) | `c0b57a0d112447f3e77c4af9812f6c16a84c2805a4d223cb6a448ef44c328898` |
| `deploy/dl-trainonly/lane/src/dl_training_v1/schemas/jope.dl-gp-train-record.schema.json` | `jope.dl-gp-train-record/v1` | `backend`, `split_role` | Historical DL train-only lane (`deploy/dl-trainonly/`) | `783473e4f2302a8917fd0212dc7f1e14bd8816467add8d98de28a5c80d4442af` |
| `deploy/lambda/regression/schemas/jope.gp-context-event.v1.schema.json` | `urn:jope.gp-context-event/v1` | `schema: "jope.gp-context-event/v1"` | Lambda regression worker (`deploy/lambda/regression/`) | `512fc5582b8eed3f7e80dafc28f166a4e165964170d3afd18e25e06616759d23` |
| `deploy/lambda/regression/schemas/jope.gp-context-head.v1.schema.json` | `urn:jope.gp-context-head/v1` | `schema: "jope.gp-context-head/v1"` | Lambda regression worker (`deploy/lambda/regression/`) | `716ca69694525af9d946b4f98fdf40b8426160864041452d30bfc05b2b326e0b` |
| `deploy/lambda/regression/schemas/jope.gp-evaluation.v1.schema.json` | `urn:jope.gp-evaluation/v1` | `schema: "jope.gp-evaluation/v1"` | Lambda regression worker (`deploy/lambda/regression/`) | `db7db63bcb237057f1ac230bd56480c174b569e0f7a8b0d6f99a6bed0400bba6` |
| `deploy/lambda/regression/schemas/jope.regression-fit.v1.schema.json` | `urn:jope.regression-fit/v1` | `schema: "jope.regression-fit/v1"` | Lambda regression worker (`deploy/lambda/regression/`) | `96ca44fa8176094cab43d3c7d48743561eb877fe6cf753f5cbbe60b760c8163a` |
| `deploy/lambda/regression/schemas/jope.regression-invocation.v1.schema.json` | `urn:jope.regression-invocation/v1` | `schema: "jope.regression-invocation/v1"` | Lambda regression worker (`deploy/lambda/regression/`) | `be132a986c4f24913fbe424788c394c598a0d9e54c1b42dd9ec0193dff2ca1a3` |

### 2.3 Schema-Less In-Code Contracts & Legacy Hash Conventions

| Contract Identifier(s) | Owning Rust Module | Serialization & Digest Rule | Status Under `I02` |
| --- | --- | --- | --- |
| `jope.hwm/v1`, `jope.score/v1` | `crates/jope-learning/src/record.rs`, `crates/jope-learning/src/canon.rs` | `serde_json` shortest float roundtrip + sorted keys; plain `SHA256(canonical_json)` (`LegacyHashScheme::PlainCanonicalSha256`) | **Retained read-only**; golden digest `fd049cd0c44a38699df78b7e3c5690b8447b06f772706f5862e05692fa3d6326` verified in `crates/jope-learning/tests/schema_agreement.rs` and `crates/jope-records/tests/wire_contract.rs`. |
| `jope.family-partition/v1` | `crates/jope-cli/src/train/corpus.rs` (line 238) | Separator-only `SHA256("jope.family-partition/v1\0" \|\| canonical_bytes)` without length prefix (`LegacyHashScheme::SeparatorOnlyNullTerminated`) | **Retained read-only**; isolated to `jope.family-partition/v1` and never shares a `schema_id` with length-prefixed `jope.canonical-json/v1`. |
| `jope.progress.work-id/v1`, `jope.progress.head/v1`, `jope.progress.config/v1`, `jope.progress.run-manifest/v1`, `jope.progress.event/v1`, `jope.progress.candidate-evidence/v1`, `jope.progress.hwm-decision/v1`, `jope.progress.corpus-manifest/v1`, `jope.progress.exposure/v1`, `jope.progress.checkpoint/v1`, `jope.progress.commit/v1`, `jope.progress.summary/v1`, `jope.progress.summary-index/v1`, `jope.progress.outbox-record/v1`, `jope.progress.outbox-ack/v1` | `crates/jope-learning/src/progress/inventory.rs`, `crates/jope-learning/src/progress/outbox.rs`, `crates/jope-learning/src/progress/product.rs`, `crates/jope-learning/src/progress/rebuild.rs` | Progress ledger canonical JSON + work-ID derivation | **Retained read-only**; bridged to `I46` via `LegacyBridgeRecord` when referencing historical progress objects. |
| `jope.selfplay-round-qualification/v1`, `jope.selfplay-round1-prereg/v1`, `jope.j9-folds/v1`, `jope.j9-pool/v1`, `jope.j9-pool-manifest/v1`, `jope.j9-selection-receipt/v2` | `crates/jope-learning/src/clover/round_qual.rs`, `crates/jope-learning/src/pools/pool.rs`, `crates/jope-learning/src/pools/evaluate.rs` | Strict duplicate-key-rejecting JSON + file SHA-256 pins (`SPQ-1`) | **Retained read-only**; offline verifier semantics preserved. |
| `jope.core-ip-claims/v1`, `jope.core-ip-conformance/v1`, `jope.core-ip-conformance-pack/v1`, `jope.core-ip-elo-histories/v1`, `jope.regression-findings/v1`, `jope.regression-finding-dispositions/v1`, `jope.regression-finding-evidence/v1`, `jope.regression-task-status/v1`, `jope.regression-test-inventory/v1`, `jope.coverage/v1`, `jope.experiment-record/v1`, `jope.unified-workplan.file-map/v2` | `crates/jope-records/src/claims.rs`, `crates/jope-records/src/conformance.rs`, `crates/jope-records/src/elo_history.rs`, `crates/jope-records/src/findings.rs`, `crates/jope-records/src/dispositions.rs`, `crates/jope-records/src/task_status.rs`, `crates/jope-records/src/test_inventory.rs`, `crates/jope-records/src/coverage.rs`, `crates/jope-records/src/paper_trace.rs`, `crates/jope-records/src/selfplay_baseline.rs` | `crates/jope-records/src/pyjson.rs` Python-compatible document-order parser/serializer (last-key-wins in historical Python records) | **Retained read-only** for historical core-IP and hygiene records; `pyjson` is never used as the strict `jope.canonical-json/v1` identity parser. |
| `jope.proposal-ledger/v1`, `jope.policy-semantic-manifest/v1`, `jope.policy-semantic-manifest/v2`, `jope.cpu-context/v1`, `jope.cpu-context-handoff/v1`, `jope.cpu-context-qualification/v1`, `jope.cpu-context-reference/v1`, `jope.cpu-context-verification/v1` | `crates/jope-policy/src/proposal_ledger.rs`, `crates/jope-policy/src/semantic.rs`, `crates/jope-policy/src/context_producer/admission.rs`, `crates/jope-learning/src/eligibility/context.rs` | `serde_json` + SHA-256 receipt digests | **Retained read-only**. |
| `jope.real-training-corpus-request/v1`, `jope.real-training-corpus-result/v1`, `jope.training-input-closure/v1`, `jope.training-family-partition/v1`, `jope.training-context-index/v1`, `jope.training-context-producer/v1`, `jope.native-training-plan/v1`, `jope.native-training-admission/v1`, `jope.native-training-invocation/v1`, `jope.native-training-window/v1`, `jope.native-training-outcome/v1`, `jope.native-training-error/v1` | `crates/jope-cli/src/train.rs`, `crates/jope-cli/src/train/corpus.rs` | `serde_json` + input-closure SHA-256 | **Retained read-only**. |
| `jope.bench-report/v2`, `jope.bench-ablation/v1`, `jope.bench-adapter/v1`, `jope.bench-adapter/v2`, `jope.panel-campaign/v2`, `jope.panel-admission/v2`, `jope.panel-report/v2`, `jope.inference-benchmark/v2`, `jope.inference-benchmark/v3`, `jope.transfer-spec/v1`, `jope.transfer-report/v1`, `jope.warm-demo/v1`, `jope.worker-execution/v2` | `crates/jope-bench/src/report.rs`, `crates/jope-bench/src/ablation.rs`, `crates/jope-bench/src/adapter.rs`, `crates/jope-bench/src/panel/campaign.rs`, `crates/jope-bench/src/panel/inference_benchmark.rs`, `crates/jope-bench/src/panel/transfer.rs`, `crates/jope-bench/src/panel/demo.rs` | `serde_json` + benchmark report SHA-256 | **Retained read-only**. |

---

## 3. Domain-Separation Formula Resolution (`S5.md:1242` vs `S1.md:1548` / `S6.md:1137`)

The reference specifications in `docs/plans/algorithmic-selfplay-20261017/reference-specs/` state two incompatible domain-separation formulas:

1. **`S5.md:1242` (Length-Prefixed Domain Separation)**:
   $$\text{SHA256}(\text{UTF8}(\text{schema\_id}) \,\|\, \texttt{0x00} \,\|\, \text{u64\_be}(\text{len}(\text{canonical\_payload\_bytes})) \,\|\, \text{canonical\_payload\_bytes})$$
2. **`S1.md:1548` and `S6.md:1137` (Separator-Only Domain Separation)**:
   $$\text{SHA256}(\text{UTF8}(\text{schema\_id}) \,\|\, \texttt{0x00} \,\|\, \text{canonical\_payload\_bytes})$$

### 3.1 Binding Resolution

- **Adopted Normative Formula for All `jope.canonical-json/v1` Envelopes**: **`S5.md:1242` (`sha256(schema_utf8||0x00||u64_be(len)||canonical_payload_bytes)`)**, implemented in `DomainSeparatedId::compute` (`crates/jope-records/src/wire_contract.rs`).
- **Rationale**:
  1. Validating that `schema_id` contains no `0x00` byte and framing `canonical_payload_bytes` with its 8-byte big-endian byte length `u64_be(len)` makes the preimage bijective over `(schema_id, canonical_payload_bytes)`.
  2. Keeping the explicit `formula` constant `"sha256(schema_utf8||0x00||u64_be(len)||canonical_payload_bytes)"` inside every `DomainSeparatedId` wire object prevents silent confusion with legacy plain (`LegacyHashScheme::PlainCanonicalSha256`) or separator-only (`LegacyHashScheme::SeparatorOnlyNullTerminated`) hashes.
  3. No new `jope.canonical-json/v1` payload may reuse a legacy `schema_id` (`jope.hwm/v1`, `jope.score/v1`, `jope.family-partition/v1`). Cross-version provenance uses `LegacyBridgeRecord` without rehashing legacy bytes.

---

## 4. `jope.canonical-json/v1` Lexical, Numeric, Envelope & Authority Rules

### 4.1 Lexical & UTF-8 Canonicalization Rules

1. **Encoding**: UTF-8 without byte-order mark (`0xEF 0xBB 0xBF` is rejected with `WireContractError::ByteOrderMark`; invalid UTF-8 is rejected with `WireContractError::InvalidUtf8`).
2. **No Insignificant Whitespace**: Canonical byte output contains zero whitespace outside string literals. `CanonicalPayload::from_canonical_bytes` rejects any leading, trailing, or interior whitespace with `WireContractError::NonCanonicalByteEncoding`.
3. **Duplicate Object Keys Rejected Before Map Construction**: `StrictParser` in `crates/jope-records/src/wire_contract.rs` rejects duplicate object keys at every nesting depth (`WireContractError::DuplicateObjectKey { path, key }`) while parsing tokens.
4. **Key Ordering & Unicode Preservation**:
   - Object keys are sorted lexicographically by Unicode scalar value sequence (identical to UTF-8 byte order).
   - Array element order is preserved (`[1,2,3]` and `[3,2,1]` produce distinct canonical bytes and distinct `DomainSeparatedId`s).
   - **No Unicode Normalization**: Combining sequences (`"cafe\u{0301}"`, `U+0065 U+0301`) and precomposed characters (`"caf\u{00e9}"`, `U+00E9`) remain distinct byte sequences and produce distinct `DomainSeparatedId`s (`V02` vs `V03`).
   - **Surrogates**: Unpaired UTF-16 surrogates (`\uD800`..`\uDFFF` not forming a valid high/low surrogate pair) and unescaped control characters (`0x00`..`0x1F`) are rejected (`WireContractError::UnpairedSurrogate`, `WireContractError::UnescapedControlChar`).
5. **Canonical String Escaping (`S5.md:1240`)**:
   - `"` (`U+0022`) $\to$ `\"`
   - `\` (`U+005C`) $\to$ `\\`
   - Control characters `U+0000`..`U+001F` $\to$ lowercase six-character `\u00xx` (including `\u0008`, `\u0009`, `\u000a`, `\u000c`, `\u000d`)
   - `/` (`U+002F`) $\to$ literal `/`
   - All other Unicode scalar values `>= U+0020` $\to$ literal UTF-8 bytes.

### 4.2 Numeric Identity Rules

1. **Safe Integers (`SAFE_INT_MIN..=SAFE_INT_MAX`)**:
   - Integer fields inside canonical identity payloads use minimal base-10 ASCII spelling in `[-(2^53 - 1), +(2^53 - 1)]` (`[-9007199254740991, +9007199254740991]`).
   - Negative zero integer (`-0`) is rejected (`WireContractError::NegativeZeroInteger`).
   - Leading zeros (`01`, `-01`) are rejected (`WireContractError::LeadingZeroNumber`).
   - Integers outside `±9007199254740991` are rejected (`WireContractError::IntegerOverflow`); 64-bit seeds and SplitMix64 stream identifiers use fixed-width 16-character lowercase hex (`HexU64`, e.g. `"seed_u64_hex": "ffffffffffffffff"`).
2. **Exact IEEE-754 Floating-Point Bit Tags (`ExactF32`, `ExactF64`)**:
   - Raw JSON floating-point literals (`0.875`, `1e-5`) are forbidden in canonical identity payloads (`WireContractError::RawFloatInIdentityPayload`).
   - Floating-point metrics and weights use tagged bit-pattern objects:
     - `ExactF32`: `{"f32_bits": "<8 lowercase hex chars>"}`
     - `ExactF64`: `{"f64_bits": "<16 lowercase hex chars>"}`
   - `+0.0` (`0000000000000000`) and `-0.0` (`8000000000000000`) and subnormals (`0000000000000001`) remain distinct and exact (`V06` vs `V07`).
   - `NaN` and `±Inf` bit patterns (`(bits & 0x7ff0_0000_0000_0000) == 0x7ff0_0000_0000_0000` for `f64`, `(bits & 0x7f80_0000) == 0x7f80_0000` for `f32`) are rejected (`WireContractError::NonFiniteFloatBits`).
   - Optional human-readable `"display_f64"` annotations inside tagged float objects are validated as finite numbers during `parse_and_canonicalize` and **stripped from the canonical identity projection** so display formatting differences never alter `CanonicalPayload` bytes or `DomainSeparatedId`.

### 4.3 Envelope Separation & Cycle Prohibition

1. **Separated Envelope Structure (`EnvelopeFrame`)**:
   - `envelope_schema`: `"jope.canonical-json/v1"`
   - `domain`: `"native" | "inference" | "storage"`
   - `schema_id`: matches `payload.schema`
   - `payload`: `CanonicalPayload` (identity preimage)
   - `own_id`: `DomainSeparatedId` computed strictly over `(schema_id, payload.as_bytes())`
   - `signature`: optional `SignatureEnvelope` binding `signed_own_id_sha256 == own_id.sha256`
   - `display`: optional human-only note outside identity
2. **No Self-Reference or Forward Cycles**:
   - Any occurrence of `"own_id"`, `"signature"`, `"later_git_commit"`, or `"later_assessment_id"` at any depth inside `payload` is rejected with `WireContractError::ForbiddenCycleField`.
   - Changing `domain`, `signature`, or `display` never changes `payload.as_bytes()` or `own_id`.

### 4.4 Outcome Authority Boundary (`S8.md:423-437`)

1. **Four Disjoint Outcome Representations**:
   - `VerifiedMeasuredOutcome` (sealed private fields; cannot be deserialized from JSON or constructed via struct literal).
   - `PredictedOutcome` (`authority = "predicted"`, carries `model_bundle_sha256`).
   - `SyntheticOutcome` (`authority = "synthetic"`, carries `synthetic_fixture_id`).
   - `MissingOutcome` (`authority = "missing"`, carries typed `MissingReason`: `not_applicable`, `unscheduled`, `budget_exhausted`, `failed_closed`, `sealed_holdout`; never coerced to `0.0` or `null`).
2. **Two-Step Verification for Measured Authority**:
   - Parsing wire JSON with `"authority": "verified_measured"` yields only `ParsedOutcomeWire::UnverifiedMeasured(ParsedMeasuredClaim)`.
   - Upgrading `ParsedMeasuredClaim` to `VerifiedMeasuredOutcome` requires `VerifiedMeasuredOutcome::verify_from_claim(&claim, &proof)` with an independent `MeasuredEvidenceProof` (`independent_verifier_passed == true`, matching non-zero 64-hex `receipt_sha256`, and matching `MeasuredEvidenceKind`).
   - `accept_measured_claim(&VerifiedMeasuredOutcome)` rejects `&PredictedOutcome`, `&SyntheticOutcome`, and `&ParsedMeasuredClaim` at **compile time** (verified by three `compile_fail` doctests in `crates/jope-records/src/wire_contract.rs`).

---

## 5. Old/New Reader Matrix & Future Self-Play DAG Contract Catalogue

### 5.1 Old/New Reader & Adapter Matrix

| Surface / Consumer | Accepted Input Schemas | Hash Verification Rule | Unknown Field / Version Policy | Unknown / Missing Value Policy |
| --- | --- | --- | --- | --- |
| Legacy `jope-learning` HWM & Score readers (`crates/jope-learning/src/record.rs`) | `jope.hwm/v1`, `jope.score/v1` | `LegacyHashScheme::PlainCanonicalSha256` (`canon::canonical_sha256`) | Strict schema validation via `crates/jope-datasets/src/schema.rs` | Unchanged historical behavior |
| Legacy `jope-cli` family-partition reader (`crates/jope-cli/src/train/corpus.rs`) | `jope.family-partition/v1` | `LegacyHashScheme::SeparatorOnlyNullTerminated` (`"jope.family-partition/v1\0"`) | Rejects mismatched schema string | Unchanged historical behavior |
| `NativeEnvelopeAdapter` (`crates/jope-records/src/wire_contract.rs`) | `envelope_schema = "jope.canonical-json/v1"`, `domain = "native"` | `DomainSeparatedId` (`sha256(schema_utf8\|\|0x00\|\|u64_be(len)\|\|canonical_payload_bytes)`) | Fails closed on duplicate keys, unknown envelope/outcome fields, unknown enums, or `domain != "native"` | Requires explicit `MissingOutcome` (`MissingReason`); rejects `NaN`/`±Inf` |
| `InferenceEnvelopeAdapter` (`crates/jope-records/src/wire_contract.rs`) | `envelope_schema = "jope.canonical-json/v1"`, `domain = "inference"` | Identical `DomainSeparatedId` and identical `payload.as_bytes()` as `NativeEnvelopeAdapter` | Fails closed on duplicate keys, unknown fields/enums, or `domain != "inference"` | Emits `PredictedOutcome` for model surrogates; never emits `VerifiedMeasuredOutcome` |
| `StorageEnvelopeAdapter` (`crates/jope-records/src/wire_contract.rs`) | `envelope_schema = "jope.canonical-json/v1"`, `domain = "storage"` | Identical `DomainSeparatedId` and identical `payload.as_bytes()` as `NativeEnvelopeAdapter` | Fails closed on duplicate keys, unknown fields/enums, or `domain != "storage"`; verifies optional `SignatureEnvelope` | Preserves exact payload bytes; relates legacy records via `LegacyBridgeRecord` without rehashing |
| Ring-1 `I03` (`jope-engine` CV/fold outcome consumer) | `jope.selfplay.cv-outcome/v1` inside `jope.canonical-json/v1` (`domain = "native"`) | `DomainSeparatedId::compute` | Rejects unknown versions (`/v2+`) and unknown fields; missing optional folds use `MissingOutcome` (`MissingReason`) | Never coerces missing/failed fold to `0.0` or `NaN` |
| Ring-1 `I07` (`feat-math` / `jope-pipeline` profile & registry consumer) | `jope.selfplay.profile-registry/v1` inside `jope.canonical-json/v1` | `DomainSeparatedId::compute` | Binds only to the 8 registered `(id, revision)` pairs in `crates/feat-math/src/kernel/profile.rs`; rejects unknown profile IDs/revisions | Absent optional strategy fields remain explicit `None`/typed status |
| Ring-1 `I46` (`jope-learning` progress/storage outbox consumer) | `jope.selfplay.verified-object-ref/v1`, `jope.selfplay.cost-reservation/v1` inside `jope.canonical-json/v1` (`domain = "storage"`) | `DomainSeparatedId::compute` + `LegacyBridgeRecord` for existing `jope.progress.*` v1 objects | Rejects unknown versions and unverified object refs; preserves existing `jope.progress.*/v1` bytes untouched | Refuses live S3 default construction in tests; requires explicit file/memory store |

### 5.2 Reserved Self-Play DAG Contract Families (`I03`..`I66`)

| Contract Family | Reserved `schema_id` | Owning Crate & Reader Seam | Primary DAG Tasks | Scientific / Decision Disposition |
| --- | --- | --- | --- | --- |
| Role & Exposure Envelope | `jope.selfplay.role-exposure/v1` | `jope-learning` / `jope-records::wire_contract` | `I04`, `I05`, `I11` | Enforces outer-test and sealed meta-test non-exposure. |
| Draw / Request / Attempt / Edit | `jope.selfplay.draw-request/v1`, `jope.selfplay.attempt-edit/v1` | `jope-policy` / `jope-records::wire_contract` | `I12`, `I13`, `I16`, `I17` | Seeds encoded as `HexU64`; ordered edit operations. |
| Typed CV & Outcome Envelope | `jope.selfplay.cv-outcome/v1`, `jope.selfplay.outcome-envelope/v1` | `jope-engine` / `jope-records::wire_contract` | `I02`, `I03`, `I06`, `I08` | `VerifiedMeasuredOutcome` vs `PredictedOutcome` vs `SyntheticOutcome` vs `MissingOutcome`. |
| Cost & Budget Reservation | `jope.selfplay.cost-reservation/v1` | `jope-learning` / `jope-records::wire_contract` | `I09`, `I10`, `I46` | Safe integer counters (`SAFE_INT_MIN..=SAFE_INT_MAX`). |
| Profile & Strategy Registry | `jope.selfplay.profile-registry/v1` | `feat-math` / `jope-pipeline` | `I07`, `I26` | Binds the 8 `(id, revision)` pairs in `crates/feat-math/src/kernel/profile.rs`. |
| Graph & Operator Effects | `jope.selfplay.graph-effects/v1` | `feat-math` / `jope-policy` | `I14`, `I15`, `I20` | Exact IEEE-754 `ExactF32`/`ExactF64` bit tags. |
| Checkpoint Ancestry & Optimizer Epoch | `jope.selfplay.checkpoint-ancestry/v1` | `jope-learning` / `jope-models` | `I27`, `I28`, `I29`, `I30` | **HOLD on live training (`training_authorized = false`)**; ancestry graph is data-only until authorized. |
| Replay & Execution Intent | `jope.selfplay.replay-intent/v1` | `jope-learning` / `jope-pipeline` | `I04`, `I27`, `I31`, `I32` | Deterministic seed + input closure binding. |
| Verified Object & Storage Reference | `jope.selfplay.verified-object-ref/v1` | `jope-learning` (`StorageEnvelopeAdapter`) | `I41`, `I44`, `I46`, `I48` | File/memory store fixtures only; no live S3 writes. |
| Snapshot, Assessment & Promotion Gate | `jope.selfplay.snapshot-assessment/v1` | `jope-learning` / `jope-records` | `I63`, `I64`, `I66` | **HOLD on `D05` / `D26` scientific promotion endpoints**: no endpoint choice or live promotion is authorized by `I02`. |
| Capsule & Wave Context | `jope.selfplay.capsule/v1`, `jope.selfplay.wave-context/v1` | `jope-records` | `I63`, `I66` | References immutable `docs/plans/algorithmic-selfplay-20261017/PACKET_MANIFEST.v4.json` without mutating frozen v4 files. |

---

## 6. Conformance Vector Catalogue (`V01`..`V10` & `N01`..`N18`)

All vectors below are stored in `crates/jope-records/tests/fixtures/wire-contract-v1/positive_vectors.json` and `crates/jope-records/tests/fixtures/wire-contract-v1/negative_vectors.json` and verified by both Rust (`crates/jope-records/tests/wire_contract.rs`) and the independent Node.js verifier (`crates/jope-records/tests/fixtures/wire-contract-v1/verify_vectors.mjs`).

### 6.1 Positive Golden Vectors (`V01`..`V10`)

| Vector ID | `schema_id` | Canonical Byte Length | Domain-Separated SHA-256 (`own_id.sha256`) | Invariant Proved |
| --- | --- | --- | --- | --- |
| `V01_REORDERED_NESTED_KEYS_AND_ESCAPES` | `jope.selfplay.draw-request/v1` | `242` | `1a5085a2fa9b41d46ebcb5569e8a663d7714c14c5eab1c40fb44fac6a6ef92fb` | Recursive key sorting, quote/backslash/literal-slash escaping, `\u0000`/`\u0009`/`\u000a` control escaping, `±9007199254740991` safe-int bounds, and `HexU64` (`ffffffffffffffff`). |
| `V02_UNICODE_PRECOMPOSED_E_ACUTE` | `jope.selfplay.unicode-test/v1` | `63` | `a3259f9cf9be28d1dfffeef3b1e3dff7ef9bb39749dff011a8255f259fbdfaea` | Precomposed `U+00E9` (`café`) + surrogate pair `U+1F680` (`🚀`) emitted as literal UTF-8. |
| `V03_UNICODE_COMBINING_E_ACUTE` | `jope.selfplay.unicode-test/v1` | `64` | `cffbb38cf8793ec6292571a00edecaa1267ab674b99f432119c5756c2b5b9862` | Combining `U+0065 U+0301` (`café`) is not NFC-normalized and has a distinct digest from `V02`. |
| `V04_ARRAY_ORDER_FORWARD` | `jope.selfplay.array-order/v1` | `57` | `586b9070c5e6ac3cd6493e457a8031529cb44fe182e7016bb36fa0f7db96182d` | Ordered array `[1,2,3]`. |
| `V05_ARRAY_ORDER_REVERSED` | `jope.selfplay.array-order/v1` | `57` | `82b74b76846a392ed83e715daffb5db8db70e5f9d01eb3b89b474f296ebd654c` | Reversed array `[3,2,1]` produces a distinct digest from `V04`. |
| `V06_EXACT_FLOAT_POSITIVE_ZERO_WITH_DISPLAY` | `jope.selfplay.float-bits/v1` | `121` | `d362ee7fe32b5f33be3792de47d2fc4548ee173f8dd20c2faec3ec7cf8761ce8` | `ExactF64` `+0.0` (`0000000000000000`) and `ExactF32` `+0.0` (`00000000`); `display_f64` stripped from identity projection. |
| `V07_EXACT_FLOAT_NEGATIVE_ZERO_AND_SUBNORMAL` | `jope.selfplay.float-bits/v1` | `169` | `476df903f7adc3685dae2b5d6bd0ecda784b789d9d74c88d11d19cb628b57b64` | `ExactF64` `-0.0` (`8000000000000000`), smallest `f64` subnormal (`0000000000000001`), and `ExactF32` `-0.0` (`80000000`) distinct from `V06`. |
| `V08_OUTCOME_VERIFIED_MEASURED` | `jope.selfplay.outcome-envelope/v1` | `280` | `147271f350d03356576d84e55baf825e43c3fbe73d00d7660fa0bccb13cdc7db` | Measured claim wire representation (`3fec000000000000`, `measured_cpu_cv`), parsed as `ParsedMeasuredClaim` requiring `MeasuredEvidenceProof`. |
| `V09_OUTCOME_PREDICTED` | `jope.selfplay.outcome-envelope/v1` | `234` | `57cf1c3a9d8e17fdd02d883c07541b1f19c8c19d36438bdf6d5d6ca4a703cfc8` | Predicted outcome (`authority = "predicted"`, `model_bundle_sha256`). |
| `V10_OUTCOME_MISSING_TYPED_REASON` | `jope.selfplay.outcome-envelope/v1` | `147` | `c91986165f658918ecb9c3f1254e1c334e28eb085d057f76f84292a13e5a0e5f` | Missing outcome (`authority = "missing"`, `missing_reason = "sealed_holdout"`). |

### 6.2 Negative Refusal Vectors (`N01`..`N18`)

| Vector ID | Category | Expected `WireContractError` Variant |
| --- | --- | --- |
| `N01_DUPLICATE_TOP_LEVEL_KEY` | `duplicate_key` | `DuplicateObjectKey` |
| `N02_DUPLICATE_NESTED_KEY` | `duplicate_key` | `DuplicateObjectKey` |
| `N03_UNPAIRED_HIGH_SURROGATE` | `unicode_surrogate` | `UnpairedSurrogate` |
| `N04_UNPAIRED_LOW_SURROGATE` | `unicode_surrogate` | `UnpairedSurrogate` |
| `N05_NEGATIVE_ZERO_INTEGER` | `numeric_integer` | `NegativeZeroInteger` |
| `N06_LEADING_ZERO_INTEGER` | `numeric_integer` | `LeadingZeroNumber` |
| `N07_INTEGER_OVERFLOW_ABOVE_MAX_SAFE` | `numeric_integer` | `IntegerOverflow` |
| `N08_INTEGER_OVERFLOW_BELOW_MIN_SAFE` | `numeric_integer` | `IntegerOverflow` |
| `N09_RAW_FLOAT_IN_IDENTITY_PAYLOAD` | `numeric_float` | `RawFloatInIdentityPayload` |
| `N10_NONFINITE_F64_NAN_BITS` | `numeric_float` | `NonFiniteFloatBits` |
| `N11_NONFINITE_F64_POS_INF_BITS` | `numeric_float` | `NonFiniteFloatBits` |
| `N12_NONFINITE_F32_NEG_INF_BITS` | `numeric_float` | `NonFiniteFloatBits` |
| `N13_UPPERCASE_HEX_IN_F64_BITS` | `hex_encoding` | `InvalidHexEncoding` |
| `N14_FORBIDDEN_CYCLE_OWN_ID_IN_PAYLOAD` | `envelope_cycle` | `ForbiddenCycleField` |
| `N15_FORBIDDEN_CYCLE_LATER_ASSESSMENT_IN_PAYLOAD` | `envelope_cycle` | `ForbiddenCycleField` |
| `N16_UNKNOWN_OUTCOME_AUTHORITY_ENUM` | `unknown_enum` | `UnknownEnumVariant` |
| `N17_UNKNOWN_OUTCOME_FIELD` | `unknown_field` | `UnknownField` |
| `N18_MISSING_OUTCOME_WITH_NUMERIC_METRIC_COERCION` | `authority_violation` | `InvalidOutcomeAuthority` |
