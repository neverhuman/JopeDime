# Funding and required-learning original evidence recovery

Observed 2026-09-11; remote custody completed approximately 21:53:52 UTC. This report records bounded original-file recovery, not execution authorization. No model or optimizer tensor was loaded, no numerical work ran, and no grant, experiment selection, balance, or continuation qualification was created.

The original optimizer tensor and its metadata exist, and their exact bytes now have verified remote custody. Four original charge-receipt byte joins also verify. Earlier wording that the optimizer itself was absent is incorrect. The remaining blockers are incomplete funding provenance and ledger coverage, unavailable production adapters, and missing evidence for the required experiment and compatible continuation.

## Scope and retained evidence

Inspected named recovery reports, their explicit original paths, selected small trial/checkpoint metadata, and one named source member in the retained source archive. There was no broad payload scan, whole inventory load, unrelated credential read, or application execution. Missing-file observations therefore apply to those paths and observation times; they are not a complete namespace census.

Existing local observations and review:

| Evidence | SHA-256 |
| --- | --- |
| `/home/ubuntu/jope-dime/parallel/runs/recovery-execution-20260911/funding-capture-v3/local.json` | `11d12fdcae60f3ad7fad26dddcf193b9a3d79f9a52439b99c9fa9e1beb8b7401` |
| `/home/ubuntu/jope-dime/parallel/runs/recovery-execution-20260911/funding-capture-v3/remote.json` | `739d8054c3b0b4634cac752ce7126c7e8642e205c4e4a16d5005df2a2c8c09cc` |
| `/home/ubuntu/jope-dime/parallel/runs/recovery-execution-20260911/authority/funding-independent-review-v3/receipt.json` | `53d38ad538a1b7fcd89f2654aec524f748ba27aafb55a29a4df313f8ffb5f3ce` |

That review retained 20 observation origins, five legacy snapshot origins, four distinct snapshot hashes, 170 reservation-row appearances, 47 names and 48 versions. `operation-observer-v3` has multiple versions. Its stated scope did not reopen original remote budget or completion files. The present work closes the named remote budget and four completion-byte joins below; it does not convert row appearances into additive charges or establish complete ledger coverage.

## New original custody

On GPU host `158.101.20.226` (`158-101-20-226`), the new-only custody directory is:

`/home/ubuntu/jope-dime/parallel/runs/all-gap-closure-20260911/authority-recovery-originals`

| Artifact in that directory | Bytes | SHA-256 |
| --- | ---: | --- |
| `originals.tar` | 222,597,120 | `57189444ac5150bcf5530107769528137897d75cc97acd26c329b03323a2bf2e` |
| `manifest.json` | Recorded in archive receipt | `2e5873f3f23b33227697b923c9b29cfabac762c10b4123496cfe45d12ca6e1bb` |

`archive-receipt.json` identifies the archive and manifest, ten files and 222,553,802 original bytes. The uncompressed archive contains `manifest.json` and `originals/` with paths relative to `/home/ubuntu/jope-dime/parallel`. The remote source and each copied object were hashed independently. The manifest records source device, inode, mode, owner, size, mtime, ctime and digest before and after copying; the complete ten-file source set remained unchanged at the final stat check. Opened sources were regular files without symlink ancestors. Copied objects and archive were made read-only. These are controller-produced custody records, not an independent funding attestation.

Before staging, free capacity was 4,787,457,773,568 bytes. The required reserve was 624,366,621,082 bytes, the larger of 250 GiB and 10% of the 6,243,666,210,816-byte filesystem. The check additionally reserved 2,592,591,252 bytes for two copies of all originals plus 2 GiB staging headroom. The recorded post-copy free capacity is 4,787,235,123,200 bytes; this manifest observation precedes archive construction.

Root subsequently mirrored the archive through the shared limiter and independently verified all ten originals on xbabe2. The inspected local receipt [local-verification.json](/home/ubuntu/jope-dime/parallel/runs/all-gap-closure-20260911/authority-recovery-originals/local-verification.json) is 2,710 bytes, SHA-256 `0701784decf3d68a07d5275f6f5748559ca5fb2640ce0464f44bb1eda626b624`, observed at Unix time 1789163985.5220692. It binds the same manifest and every original byte count/hash. This closes the ten-file storage join; canonical funding, application resume and original three-seed selection remain explicitly false in that receipt.

## Checkpoint and optimizer correction

The exact original checkpoint directory is:

`/home/ubuntu/jope-dime/parallel/runs/regression-campaign64-qualification-v1/expansion-20260909-source-reviewed/cycle-0000/train-2/checkpoints/step-3000`

| Original file under that directory | Bytes | SHA-256 and verification |
| --- | ---: | --- |
| `state.json` | 2,502 | `c2f229c9b8c582e11e0e62ec011a9ccb066820fbcee5e2da9d0ad5d63780160e`; reopened, matches trial reference, copied and verified |
| `optimizer.json` | 166 | `7dc5ae8f897728c69344cb4231d131a12e734fbd91662b2aaf615b7c00ecea5f`; reopened, matches state reference, copied and verified |
| `optimizer.ot` | 222,453,019 | `2dc05d1ab76793fd4744c9fe1ec3ae5f4e1e1fc42a24719682f9de13fa04bea2`; original and copy hashes verified against state and optimizer metadata |
| `model.ot` | 111,218,527 | State declares `e2cc5d9bceb39850c96eb6e4aa609b25ea7a435774c41c90efffaecb0da5fb8e`; this pass checked file existence/stat only, did not hash, load or copy it |

The state declares schema `parallel.checkpoint.v1`, step 3000, seed 20260907, world 8, global batch 2048, `ranking_backward=dense_pairwise_v2`, `runtime_objective=huber_log1p_seconds_v2`, parameter hash `5f8357e5ca9409487dd7c3bc6f73e6d96e1e5bb934ea534ca5e21b60a4545453`, and corpus hash `11e60a43fc02c9b9452ea22126a017299b5ecb8d050466b15710d6f62853360d`. Optimizer metadata declares backend `cuda_fused_fp32_v1`, weight decay 0.01, learning rate 0.0001 and step 3000.

These are verified metadata declarations, not proof that the saved state resumes identically. Named `sampler.json` and `rng.json` were absent; the inspected state lacks explicit microbatch, `partition_sha` and `objective_sha`. No compatible original sampler/RNG/objective-partition evidence or numerical continuation comparison was recovered. A metadata match in [task_contract.rs](/home/ubuntu/jope-dime/parallel/src/task_contract.rs:22) alone cannot supply those originals. The model's state-declared digest still needs verification under its existing custody route before a complete checkpoint claim.

## Four original charge receipts

Each file below was reopened on the GPU host, its bytes matched the legacy budget's stored reference, and those bytes were copied into the new custody archive.

| Exact original path | Bytes | Verified SHA-256 | Limited interpretation |
| --- | ---: | --- | --- |
| `/home/ubuntu/jope-dime/parallel/runs/flywheel-20260909/qualification-budget-audit-v1/complete.json` | 140 | `c32253d3e5ea94db1ac6b8d75a8fab4e28dabc7e7406caa29a87083ecfe2d2e5` | Historical accounted receipt declares 48,000 core-seconds |
| `/home/ubuntu/jope-dime/parallel/runs/flywheel-20260909/strict-replay-v1/orchestration-budget-audit.json` | 504 | `ac3e08ab0b51d72fbb0bc540d6df4470ec9e8445b0ed6c1bdae2dd0abd299a98` | Historical audit declares 2,100 core-seconds and zero node-seconds; created after dispatch, no refund implied |
| `/home/ubuntu/jope-dime/parallel/runs/regression-gain-20260909/trial-v2/broad_carry/qualification-training.json` | 1,546 | `fe1ad3b33df1fe3d0fcf2d8699762272ef0755398dcffc8e69846319b7360eca` | Historical trained receipt reports approximately 94.1375 seconds; does not qualify the required trial |
| `/home/ubuntu/jope-dime/parallel/runs/flywheel-20260910/operation-observer-v3/complete.json` | 46,350 | `d1fc07b29d1c60855e3eeeec55e2e7c891562ec4d3fdd4365182aefad1cbb8bc` | `operation_observer_progress.v3` original; byte provenance alone does not establish a useful hour |

The joins establish file identity. They do not establish authoritative charging policy, unique cross-ledger consumption, independent pre-dispatch authorization, or remaining balance. Unknown consumption and overlapping historical rows must remain visible and must not be refunded by this recovery.

## Legacy budget versus missing canonical authority

The reopened GPU-host original `/home/ubuntu/jope-dime/parallel/runs/regression-gain-20260909/trial/budget.json` is 17,482 bytes, SHA-256 `b9a3c52324aeec8342ebba2ed58bf05ffae6fdd8f6e7b7f84e1c25c53ae0c9d5`, with 47 reservation names. Its unchanged bytes are in the archive. The same logical path on xbabe2 is a different six-row, 2,077-byte version, SHA-256 `109c236d3216cd7ac78241740937810966eb9e5cc66ef3aea28a0964fd6f35c4`. Both origins must be retained; one must not overwrite the other as a supposed repair.

At the bounded recheck near 21:48 UTC, these four exact paths were absent on both hosts:

- `/home/ubuntu/jope-dime/parallel/runs/qualification-budget/authorization.json`
- `/home/ubuntu/jope-dime/parallel/runs/qualification-budget/domain.json`
- `/home/ubuntu/jope-dime/parallel/runs/qualification-budget/journal.jsonl`
- `/home/ubuntu/jope-dime/parallel/runs/qualification-budget/budget.json`

Absence at those paths is not evidence that an original v2 grant once existed and was lost. The retained source archive `/home/ubuntu/jope-dime/parallel/runs/all-gap-closure-20260911/remote-original-custody/originals.tar` contains the exact member `parallel/runs/flywheel-20260909/native-v5-cpu/source/parallel/tools/qualification_budget.py` (2,456 bytes; SHA-256 `f7e4ecb61814155e7c132740082ec8f6bf63ff63f5e829e936701d5de6100c26`). Its lines 10–13 create a budget under each supplied output directory; lines 20–24 initialize a v1 reservations object and use hardcoded 64 CPU-hours/two node-hours ceilings; lines 31–43 attach completion references and preserve ceilings. That implementation did not create a canonical authorization, domain or journal. It supports the inference that the missing v2 objects may never have been produced. Hardcoded historical ceilings are not independently authenticated grants.

There are also explicit unfinished adapter gates. Current [qualification_budget.py](/home/ubuntu/jope-dime/parallel/tools/qualification_budget.py:20) refuses the canonical production domain even if files are planted; [initialization](/home/ubuntu/jope-dime/parallel/tools/qualification_budget.py:42) is restricted to private synthetic qualification because complete two-host history is unavailable. Its existing [reservation verifier](/home/ubuntu/jope-dime/parallel/tools/qualification_budget.py:159) defines a concrete domain/identity/task binding and one-time consumption interface, but does not make production initialization available. Native [require_original_gpu_grant](/home/ubuntu/jope-dime/parallel/src/task_contract.rs:50) is an explicit unconditional hold, not an implemented positive verifier. Recovering bytes alone will not clear either gate.

The source-only archive inventory covers 435 source files; absence of funding/trial originals from that inventory is not a ledger census. Prior priority/source inventories likewise do not cover all application evidence. No complete original authorizer/capability provenance or cross-output ledger closure was identified in the named sources inspected here.

## Original experiment candidates

| Original GPU-host path | Bytes | Verified SHA-256 | Classification |
| --- | ---: | --- | --- |
| `/home/ubuntu/jope-dime/parallel/runs/regression-gain-20260909/trial/manifest.json` | 15,865 | `796a9030bfbcba875b1b0b7e849b9757b4e2f4f36e41660b40108ed3bdf40722` | Copied original: broad/fresh × reset/carry, four conditions, 1000 steps |
| `/home/ubuntu/jope-dime/parallel/runs/regression-gain-20260909/trial-v2/manifest.json` | 16,228 | `9d68d6d68ab94f155110a699b0f32feb9f92e5f5fc3a2bf2369170eb62d4676b` | Copied historical retry: same four conditions, parent reference, 300-second retry ceiling |
| `/home/ubuntu/jope-dime/parallel/runs/flywheel-20260909/strict-replay-v1/manifest.json` | 39,896 | `d0c3ca5457ed694b9a7e915f43e9a4f29f90b25fccd7c82c60ea7ecb5464c4a9` | Read and hashed, not in this archive: same four conditions, 1000 steps, 180-second ceiling |
| `/home/ubuntu/jope-dime/parallel/runs/seed-quality-experiment-v1/manifest.json` | 1,609,220 | `165b5e9269e062a686b9ccbb7b1daa7db548ded7acf1af693616ec4736cd3810` | Existing local and remote bytes match; different proposal-quality experiment, seed 20260908, cycle-0010/train-1 checkpoint and 11 panels |

The original trial's four small policy references were reopened and matched their hashes. Carry and reset both declare eligibility. Broad and fresh policy sampling seeds are respectively 14200484795610075786 and 2853096593380960880, two lineage-derived sampling seeds. `policy.seeds_per_cut=3` describes the qualification GP policy; it does not identify three selected training seeds. The strict-replay policy payloads exceeding the bounded 128 KiB read limit were not inspected. None of these candidates establishes the original selected three training seeds and support/replay condition membership required by the current task.

The [required-learning status](/home/ubuntu/jope-dime/parallel/tools/qualify_regression_gain.py:175) therefore remains unavailable before opening references or publishing state. The original constraints remain global batch 2048, 1000 **new committed** steps, a 300-second bound, compatible recovery lineage and the originally selected three seeds/conditions. The [legacy matrix launcher](/home/ubuntu/jope-dime/parallel/tools/qualify_regression_gain.py:202) is retired. Neither recovered four-arm manifests nor historical retry ceilings authorize resetting the optimizer, renewing an expired deadline or inventing a new selection.

## Closure dependencies and acceptance

1. **Preserve and mirror this evidence — completed for the ten named files.** Root transferred the archive under the existing aggregate limiter and verified the member hashes against the manifest; the local receipt is cited above. Preserve host-specific legacy budget versions and original receipt identities. This acceptance is a storage result only, not complete application or workspace custody.
2. **Close the historical accounting denominator.** Complete bounded two-host ledger discovery with explicit coverage and unresolved objects, identify copied versions and reservation identities without double charging or accidental refunds, and join original charge evidence. A defensible reconciliation must keep unknown ceilings charged and state why every included/excluded origin belongs to the same funding domain. The current four receipt joins do not meet this requirement.
3. **Resolve authority provenance before implementing admission.** Recover independently attributable original authority or obtain an explicitly new authorized funding decision through the proper owner, without describing it as recovered history. Implement production verification against that evidence, including capabilities, expiry, full historical charges and remaining ceiling. Reuse the existing typed domain/identity/task reservation contract; make native start consumption atomic and single-use. Tests must reject planted self-hashed files, copied domains, changed task identities, duplicate consumers, expired original windows and incomplete history; an independent real positive receipt is required before enabling execution.
4. **Recover the actual experiment selection and continuation contract.** Identify the original selected three training seeds/conditions and compatible checkpoint, sampler, RNG, objective and partition references. Verify the model bytes through their custody route. Byte-compatible optimizer state is now a candidate input, not missing storage. If the original selection or exact continuation evidence cannot be recovered, report it unavailable; any newly authorized experiment must be explicitly distinguished from the required original trial.
5. **Qualify execution only after those dependencies.** Under a valid original task grant and resource/deadline reservation, verify exact continuation and the required batch/step/condition identities with independently retained evidence. The 1000-step count must measure new committed updates after the original baseline, and retries must consume the original deadline/ceiling. No funded numerical acceptance was attempted by this recovery pass.

No whole-workspace absence, aggregate consumption, remaining balance, canonical grant, qualified continuation, completed required trial, or production execution authority is claimed.
