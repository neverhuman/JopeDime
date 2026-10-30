# FUP-02 H-wave evidence inventory and recovery

Snapshot date: 2026-10-30. This is a custody inventory, not a rerun or a new scientific
qualification. It covers H01–H50 and records where the available reports, receipts, configs,
and metrics stand. Existing reviewable study records are hashed in `source-hashes.sha256`; files
recovered into this record are preserved under `recovered/` with `SHA256SUMS`. Original S3 hashes
for the three redacted JSON files are in `recovered/S3_SOURCE_SHA256SUMS`.

## Scope and handling

- The H03 recipe is excluded in full per the pending Jepson ruling. The table records only the
  already-published high-level outcome summary; it does not copy, hash, inspect, or reconstruct
  recipe material.
- No model/checkpoint weights, dataset bodies, prediction arrays, or sealed/holdout payloads were
  copied. This work did not score models or recompute metrics. DOPE was not changed.
- H29–H34 uses the corrected HWV-02 custody record: 40 public-research documents are present and
  the source receipts verified 41/41 digests; the 41st item is a pointer, not a recovered artifact.
  The earlier `EVIDENCE_MISSING` status in that history is superseded.
- S3 downloads were limited to the `evidence/flywheel-h40-h45-20261026/` reports, receipts,
  budgets, and run ledgers listed in `recovered/SHA256SUMS`. S3's H41–H44 `results.json` is
  `DEEP_ARCHIVE`; a read-only `HeadObject` showed 178,506 bytes, but S3 refused download because
  the object is archived. No restore request was made.
- Three recovered JSON records contain an `auth_packet_sha256` field that the repository's
  full-history secret scanner classified as a generic API key. Those field values are redacted in
  the committed copies; the exact unmodified S3 object hashes and keys are preserved separately
  in `S3_SOURCE_SHA256SUMS`. Unredacted source copies remain in the external FUP-02 evidence
  staging area and were not added to Git.
- H50 was in progress in the accessible status snapshot. Its evolving run output was not copied;
  no completion or terminal-selection claim is made here.

## Wave-by-wave inventory

“Present” means a hashable report/result/receipt exists in the cited study record. It does not
imply that the wave qualified, that every raw log/config was preserved, or that metrics were
independently recomputed. “Gap” names evidence not present in the bounded inventory or a specific
restriction on recovery.

| Wave | Present record and disposition | Remaining evidence gap |
|---|---|---|
| H01 | Early-history outcome summary: exploratory gain on reused development data; inconclusive. | Full original recipe/config and independently bound run-level metrics are not in the consolidated public-history record. |
| H02 | Early-history summary: rejected; reported confidence intervals include zero. | Original execution receipt/config not located in the consolidated record. |
| H03 | High-level summary only: early positive-CI signal, inconclusive. | Recipe explicitly excluded by ruling; no recipe recovery or hash attempted. Detailed run receipt remains unverified. |
| H04 | Early-history inventory summary: data/candidate inventory only. | No model-fit result exists for this inventory wave. |
| H05 | Early-history summary: concept only, never run. | No execution config, receipt, or metrics exist in the searched record. |
| H06 | Early-history summary: rejected; randomized/shuffled controls include harm. | Original run-level receipt/config not located. |
| H07 | Early-history summary: rejected; nested conditional Ridge underperformed trees. | Original run-level receipt/config not located. |
| H08 | Early-history summary: rejected; frozen readouts did not beat random/A0. | Original run-level receipt/config not located. |
| H09 | Early-history summary: queued, zero fits; never run. | No execution receipt or metrics. |
| H10 | Early-history metadata binding only. | No numeric result or run-level config. |
| H11 | No artifact in the consolidated history; recorded as missing. | Config, receipt, metrics, and source record remain unrecovered. |
| H12 | Early-history metadata binding only. | No numeric result or run-level config. |
| H13 | Early-history summary: archive qualification failed (91/147 contexts); closed. | Full raw qualification log/config not in the consolidated record. |
| H14 | Early-history summary: secondary diagnostic, inconclusive. | Detailed receipt/config not independently located. |
| H15 | Research-continuation summary: canonical-frontier/audit work; zero numeric admissions. | Underlying audit evidence is not a numeric training result; no admission metrics to recover. |
| H16 | Research-continuation summary: metadata/audit only; zero numeric admissions. | Per-wave standalone receipt/config not identified in the consolidated record. |
| H17 | Research-continuation summary: metadata/audit only; zero numeric admissions. | Per-wave standalone receipt/config not identified in the consolidated record. |
| H18 | README, source manifest, results, curves, and runtime ledger are present. Outcome: no established lift. | No known missing public summary artifact; raw inputs were intentionally not inventoried/copied. |
| H19 | Plan, README, and results are present; diagnostic completed. | No standalone runtime ledger identified. |
| H20 | Plan, README, curves, and results are present; rejected (both primary CIs include harm). | No known missing public summary artifact. |
| H21 | README, plan, source bridge/manifest, phase ledger, and qualification/results records are present; metadata/source qualification. | No numeric admission result; record remains qualification/audit evidence. |
| H22 | README and results are present. | No standalone protocol or runtime ledger identified. |
| H23 | README and results are present; continuation plan also present. | No standalone runtime ledger identified. |
| H24 | README, results, and next metadata packet are present. | No standalone runtime ledger identified. |
| H25 | README, results, source pins, and cost ledger are present. | No standalone runtime ledger identified. |
| H26 | README, fold-identity correction, source-contract summary, and L1/L2 records are present. | No standalone final execution ledger identified. |
| H27 | Qualification, probe, source receipt, plan, and NN diagnostic/training records are present. Routing lift remained untested. | No downstream routing outcome exists; this is an explicit scientific gap, not lost evidence. |
| H28 | README, results, source receipt, and arithmetic verification are present; diagnostic arithmetic only. | No model-run config or training metrics expected for this arithmetic check. |
| H29 | README and metadata summary present; corrected custody verifies public artifacts. | The source bundle is custody provenance; the referenced `derive_exact_gains.py` item is pointer-only. |
| H30 | README, protocol, results, curves, storage/cost ledgers, and source receipts present; 41/41 custody check includes this wave. | No missing public artifact reported by corrected custody. |
| H31 | README, protocol, results, curves, failure/cost records, and source receipts present; 41/41 custody check includes this wave. | No missing public artifact reported by corrected custody. |
| H32 | Results, mechanism notes, source pins, and synthetic callback curve present; 41/41 custody check includes this wave. | `derive_exact_gains.py` is a pointer only, not an executable recovered result artifact. |
| H33 | README, results, protocol, training curves, and source receipts present; 41/41 custody check includes this wave. | No missing public artifact reported by corrected custody. |
| H34 | README, results, curves, protocol, cost ledger, and source receipts present; 41/41 custody check includes this wave. | No missing public artifact reported by corrected custody. |
| H35 | Combined H35–H45 report recovered from S3; diagnostic ablation/search summary. | No separate H35 run-level config/receipt in the reviewed source set. |
| H36 | Combined H35–H45 report recovered from S3; diagnostic summary. | No separate H36 run-level config/receipt in the reviewed source set. |
| H37 | Combined H35–H45 report recovered from S3; diagnostic summary. | No separate H37 run-level config/receipt in the reviewed source set. |
| H38 | Results, reconciled results, and budget ledger present; H38–H41 custody digests verified in HWV-02. | No new independent Elo result in this inventory. |
| H39 | Results and execution/reservation ledgers present; H38–H41 custody digests verified in HWV-02. | No new independent Elo result in this inventory. |
| H40 | Results, budget ledger, and timing closeouts present; custody digests verified in HWV-02. | H40B is separately indexed as cached-pool replay; no independent fit or non-reused holdout evidence is claimed. |
| H40B | Master index and H35–H45 report summarize cached-pool replay, zero native fits, and reused holdout. | Dedicated H40B standalone result/config was not present in the reviewed repo inventory; do not treat the replay as independent qualification. |
| H41 | Results and reconciled closeouts present locally; combined H41–H44 S3 README, budget, and runtime ledger recovered. | S3 combined `results.json` is archived (`DEEP_ARCHIVE`) and was not restored; standalone H42–H44 results are grouped. |
| H42 | Combined H41–H44 S3 report and accounting records recovered; SPQ-1 tilt is documented as violating the KL gate. | Standalone per-wave config/result is not separated from the combined H41–H44 result; archived combined metrics were not downloaded. |
| H43 | Combined H41–H44 S3 report and accounting records recovered. | Standalone per-wave config/result is not separated from the combined H41–H44 result; archived combined metrics were not downloaded. |
| H44 | Combined H41–H44 S3 report documents the in-sample QHat diagnostic. | Standalone per-wave config/result is not separated from the combined H41–H44 result; archived combined metrics were not downloaded. |
| H45 | S3 H45 cluster README, result, bounded budget, and runtime ledger recovered and SHA-256 recorded. | No new qualification: both generations were rejected by the ratchet; no independent promotion claim. |
| H46 | README and metrics present; complete, no selected win, objective mismatch/no final gain or negative arm. | No missing public summary artifact identified. |
| H47 | README and metrics present; complete, no win; all final median deltas negative. | No missing public summary artifact identified. |
| H48 | README and metrics present; complete, no strict win; ES lead failed family-CI/Elo checks. | No missing public summary artifact identified. |
| H49 | README and metrics present; complete, no strict win; fold-1 winner's curse and control-seed pseudoreplication recorded. An xbabe1 owner status snapshot is also preserved under `recovered/xbabe1/`. | No missing public summary artifact identified. The older owner snapshot predates terminal results; use the later repo metrics as closeout evidence. |
| H50 | Completed H50-S `sensitivity.json` and 320-candidate `directions.jsonl` were copied from xbabe2 to `recovered/xbabe2/h50-s/`, with stable source size/mtime across copy and recorded SHA-256. | The search arms were still active; their evolving outputs were not copied. Final arm metrics, terminal validation disposition, and a closed wave receipt remain pending. |

## Recovered S3 files

All files below came from `s3://veox-jopedime-archive-use2/evidence/flywheel-h40-h45-20261026/`.
Their byte sizes and committed-copy SHA-256 values are in `recovered/SHA256SUMS`. Unmodified
source-object hashes are in `recovered/S3_SOURCE_SHA256SUMS` where redaction was required.

- H00–H45 master index and H35–H45 combined report.
- H40–H45 S3 custody receipt.
- H41–H44 README, bounded additive budget, and a runtime ledger with the secret-like digest
  withheld.
- H45 cluster README, results, bounded additive budget, and runtime ledger. The result and
  runtime JSON copies have the flagged digest withheld.

## Recovered host files

- xbabe1 H49 owner status snapshot (historical preregistration/progress only).
- xbabe2 completed H50-S sensitivity report and direction metrics; the source size/mtime were
  unchanged during copying. No active H50 search-arm output or checkpoint was copied.

H41–H44 `results.json` remains an identified but unrecovered archive object because it is in
`DEEP_ARCHIVE`; the copy request returned the storage-class restriction. No archive restore was
requested. No H40B checkpoint or any `checkpoints/` key was fetched.

## Search and verification boundary

The filename-only host scan covered the authorized wave record roots on xbabe1, xbabe2, and xbabe3,
pruning build targets and binary/input/pool/checkpoint paths. It found an xbabe1 H49 status snapshot
path, but no additional H01–H17 annex or H49/H50 per-wave checkout record outside the listed
sources. The exact H15–H17 source annex name was not found on those hosts or under the S3 evidence
prefix. H29–H34 is resolved by the newer merged HWV-02 correction and must not be called missing.

Complete source sets were not recovered for every wave: H11 remains absent; H01–H03 and H13–H17
have summary-only or restricted evidence; H35–H37 and H42–H44 have grouped rather than standalone
records; H40B has only the indexed replay summary; H41–H44 metrics are archived; and H50 was still
running. These are the unresolved gaps at this snapshot, not zero-valued metrics.
