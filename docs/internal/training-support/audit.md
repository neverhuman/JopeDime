# JopeDime training support audit

**25 September 2026 · Source audit and implementation plan · Page 1 of 5**

Source: `1216b79cba73594b3d1c3a6d9b21011dee7daf5c`, branch `feat/training-support-multinode`, checkout `/home/ubuntu/JopeDime/.agent/worktrees/codex-training-support`. Remote main matched this SHA when checked. Audit scope: repository training surfaces, kernel/GP improvement, S3 storage, and a path across xbabe2, xbabe1 and xbabe3. No training, promotion, S3 writes or merge was performed.

## Decision and entrypoint map

**The building blocks exist; a qualified three-host training loop does not follow from them.** The canonical Rust lineage, the newer experimental Python campaign, and historical distributed Python trainers have different admission, checkpoint and orchestration contracts. Connecting them requires explicit interfaces and proofs, not a blanket launch script.

| Surface | Entry and implementation | Current interpretation |
|---|---|---|
| Canonical lineage B | `crates/jope-engine/src/main.rs` → `controller.rs`, `training.rs`, `neural.rs`; `python/src/jope/tools/continuous_learning.py` | tch optimizer, GP teaching corpus, checkpoint/recovery and generation control. Public historical GPU dispatch is held. |
| Admitted native training | `crates/jope-engine/src/smartcluster_worker.rs`; `training.rs:537` | Opaque verified native task carries authority, inputs and lifetime. Training currently calls numerical implementation with rank 0, world 1. |
| Bounded experimental training | `crates/jope-cli/src/train.rs` → `python/src/jope/policy/training_campaign.py` | Prepare, plan, run, resume, compare, verify and export. Strict optimizer continuation and closed worker inputs; explicitly single process. |
| Policy-v3 distributed mechanics | `python/src/jope/policy/train_v3.py`, `ddp.py`, `model_v3.py`, `losses.py` | Rank/local-rank separation, sharding, manual gradient averaging and epoch coordination exist. This is not evidence that the admitted campaign supports distributed execution. |
| Historical and research trainers | `python/src/jope/train.py`, `self_play.py`; `research/test_exact_train.py`; `chimera/elo_finetune/train_elo.py`, `chimera/router/train_trunk_reg.py` | Separate model/checkpoint families and launch assumptions. Preserve as reference; do not silently substitute for canonical policy training. |
| Auxiliary learning | `python/src/jope/dime_score/{train,broad_train}.py`, `target_maker/train.py`; engine `auxiliary_training.rs` | Distinct auxiliary objectives; catalog separately from policy qualification. |
| Local demonstration | `crates/jope-cli/src/champions.rs:69` | `train-step --local` fits a centroid router. It does not update policy weights or write S3. |

## Fresh evidence and limits

Read-only SSH and `nvidia-smi` succeeded: **xbabe2 = RTX 3090 / 24,576 MiB; xbabe1 = RTX 4090 / 24,564 MiB; xbabe3 = RTX 3090 / 24,576 MiB**. These are heterogeneous GPUs. Driver compatibility, free capacity, interconnect throughput and scheduler grants were not qualified. xbabe1 resolved by hostname; xbabe3's SSH configuration resolved to `192.168.68.54`. Do not infer local xbabe2 from the host's misleading SSH alias.

Local service readback showed native observer, SmartCluster qualification, a full CI runner, and a qualification placeholder service active. Service names are not proof of a live training allocation. GitHub run **36083244749/1** at the audited SHA completed **failure**; **main-guard 36083244595/1** succeeded. Prior ledger failure counts were not independently recounted. PR94 remains another team's open implementation PR; no duplicate lane or review was started.

<!-- pagebreak -->

# Kernel/GP improvement and S3 paths

**Source audit · Page 2 of 5**

## What “push higher” actually traverses

The intended loop is: admitted dataset and cut → verified contexts → policy proposals and GP search → fitted/evaluated programs → strict per-identity champion → eligible teaching corpus → optimizer updates → matched comparison → separately admitted promotion. A better score on one dataset does not establish cross-dataset transfer or authorize serving promotion.

| Stage | Owning source | Audit finding |
|---|---|---|
| Numeric kernels and GP search | `crates/feat-math/src/`; engine `gp.rs`, `service/`, `proposals.rs`, `proposal_ranking.rs` | Kernel evaluation and genetic search are numerical work, not S3 publication. |
| Winner persistence | `crates/jope-engine/src/champions.rs:123` | Durable observations, strict winners, immutable events, publication outbox/recovery and frozen snapshots exist. Preserve this evidence chain. |
| Shared HWM | `crates/jope-learning/src/hwm.rs:196`, `record.rs`, `sources/` | Ingest maps champions, legacy genebank and snapshots. Direction-aware strict improvement is bound to comparison identity; ties retain incumbents. |
| Teaching admission | `crates/jope-learning/src/eligibility.rs`; CLI `train.rs`; policy `training_input_closure.py` | Diagnostic HWM freeze and admitted teaching are different exports. Sealed/outer-test authority must remain outside the worker input closure. |
| Transfer evidence | `python/src/jope/policy/training_comparison.py`; `tools/promotion_tournament.py` | Sibling comparisons and historical tournament logic are separate qualification routes. Training completion alone cannot advance serving. |
| Kernel release transport | `python/src/jope/tools/kernel_bundle.py`, `kernel_mirror.py`, `kernel_remote_publish.py` | Hash-bound bundles, SSH/rsync and verified local rename. “Remote publish” here is not an S3 API or learning improvement. |

HWM comparison identity includes task, dataset hash, cut, evaluator, vocabulary and protocol. Labels stay intact. Corpus inclusion must retain family/split roles and evaluation ancestry; flattening all winners into a training folder would bypass these boundaries.

## S3 implementation map

- **Datasets:** `crates/jope-datasets/src/{repo,blobs,cache,ingest,store}.rs`. Content-addressed blobs, manifests and catalogs; default product bucket `veox-jopedime-datasets-use2`. Selection still needs an explicit admitted manifest and local verified cache.
- **Learning:** `crates/jope-learning/src/{mirror,sync,state,ledger}.rs`. Local state is authoritative for this writer; optional `--store` / `JOPE_LEARNING_URL` mirrors objects. Immutable writes use create-only semantics and verify existing bytes. Pointer replacement is compare-then-unconditional-put, unsafe for concurrent cross-host publishers.
- **Models:** `crates/jope-models/src/publish.rs:23`. Verify local payloads, upload blobs create-only, publish immutable bundle manifest last. Default `veox-jopedime-models-use2`. Bundle publication does not itself prove training quality or promotion.
- **Credentials:** `crates/jope-datasets/src/credentials.rs` supports explicit profile resolution through AWS CLI and guards absent providers. It is credential plumbing, not least-privilege authorization. The host contract records an administrator provider, no provisioned restricted training writer and no AWS credentials in runner containers; these IAM facts were not revalidated live in this audit.

The dated `catalogs/s3.json` inventory reports only staging copies in the learning bucket and T1 datasets in the dataset bucket. **No live object inventory was performed**; staging must not be assumed canonical. Product buckets and preservation custody buckets are distinct. The top-level CLI champion S3 route explicitly refuses operation (`champions.rs:12`); that refusal does not mean the separate dataset/model/learning crates lack S3 support.

<!-- pagebreak -->

# Gaps, dead paths and organization

**Prioritized findings · Page 3 of 5**

| Priority | Finding and evidence | Consequence / required change |
|---|---|---|
| P0 | `mirror.rs:92–101` reads expected bytes then invokes unconditional `Store::put`. | Two publishers can both pass the check and overwrite the pointer. Keep one publisher until atomic conditional update and stale-writer tests exist. This is a documented limitation, not a newly observed live loss. |
| P0 | `task_contract.rs:88` deliberately always rejects route-only GPU dispatch; admitted `training.rs:630` fixes rank/world to 0/1. | Environment flags or inherited files cannot unlock historical training. Extend verified native admission to a bounded multi-host allocation. |
| P0 | `training.rs:779` accepts world 1/2/4/8; device uses `Cuda(rank)` at 1058; `neural.rs:1034` uses a shared `nccl.id` file. | Three hosts with one GPU each cannot use the existing canonical topology. Separate global rank, local device and host identity; implement rendezvous/receipt transport. |
| P0 | `training_campaign.py:1230` forces `WORLD_SIZE=1`; device checks at 743 require one visible RTX 3090. | Experimental campaign is not a torchrun-ready three-node launcher; xbabe1's 4090 also fails its current device contract. Version the contract explicitly. |
| P1 | Global batch 2048 with equal rank partitions (`task_contract.rs:28`; `controller.rs:495`). | 2048 is not divisible by three. Adding 3 to an allowlist is insufficient; new partition/objective semantics and weighted reductions need qualification. |
| P1 | Rank checkpoints and completion receipts are read from common local paths (`training.rs:218`); local GPU leases transfer file descriptors (`gpu_worker.rs`). | Neither local filesystem visibility nor Unix descriptor ownership spans SSH hosts. Need per-host leases plus coordinated checkpoint commit and cleanup. |
| P1 | `controller.rs:202–290` has an eight-rank proposal path and CPU bindings; `self_play.py:38–44` hardcodes network settings. | Parameterize node inventory, CPU sets, devices and interface selection; training-only changes leave proposal/orchestration constraints behind. |
| P1 | S3 source layout and restricted credentials remain unqualified operationally. | Pin live inventory and permission receipts before data rollout; do not copy administrator credentials to workers. |
| P1 | CLI/documentation surfaces diverge. ADR describes `jope learn ...`; current top-level CLI has `train`, while `jope-learning` owns HWM/scores. | Generate a command/status map from actual parsers. Do not advertise the ADR command list as executable acceptance instructions. |
| P2 | `neural.rs:677` defines `preference_dpo`; audited engine source search found no caller. | Label it unused implementation. Do not claim active DPO or policy-gradient reinforcement learning. |

## Organization recommendation

Keep source in its current owning crates for the first fixes. Create one maintained training index with **entrypoint, lineage/model, input schema, checkpoint schema, authority, device topology, storage writer and proof status** for each route. Classify `deploy/kernel-v1/source/` as a release snapshot with its owning regeneration procedure, not a second editable implementation. Treat self-play units under `deploy/systemd/selfplay/` and `jope-campaigns` as historical control-plane code unless current admission and state are established. ADR-0005 records lost lineage-A state; surviving binaries cannot reconstruct its authority.

This is an architecture and selected-path source audit, not exhaustive correctness review of every research trainer. Missing end-to-end qualification is distinguished from missing code. No kernel performance claim, live score improvement, collective benchmark or fresh optimizer proof was produced.

<!-- pagebreak -->

# Multi-node design for xbabe2, xbabe1 and xbabe3

**Proposed architecture; not a compute grant · Page 4 of 5**

## Start with independent workers, then synchronous training

Use **xbabe2 as coordinator and sole learning-store publisher**, with independently admitted dataset/GP jobs on xbabe1 and xbabe3. Stage immutable admitted inputs on all participating hosts, collect immutable worker results, and let one coordinator validate evidence and ingest strict improvements. This uses all three hosts without pretending that three-way synchronous training is already qualified. Every job still requires its own original resource contract.

For synchronous DL, first qualify a **two-host, one-GPU-per-host** topology on the two 3090s. That is a proposal for numerical qualification, not permission to run. World 2 fits the existing global batch arithmetic, but rank mapping, rendezvous, distributed authority and checkpoints still require implementation. Bring the 4090 into a separate heterogeneous three-rank experiment only after those mechanics pass.

## Required manifest and lifecycle

A source-bound launch manifest should name: run/attempt ID; source and executable hashes; model and optimizer parent; corpus/vocabulary/closure hashes; host IDs; global and local ranks; GPU UUIDs; CPU/memory ceilings; original absolute deadline; native allocation/participant identities; rendezvous endpoint; numerical partition version; and immutable result destinations. Host configuration belongs in data, not Python constants.

1. **Preflight:** native authority verifies all host participants before any CUDA initialization. Check device compatibility and availability, runtime hashes, clock skew, TCP reachability, disk space, corpus hashes, split fences and closed input directories. Missing one participant refuses the whole synchronous attempt.
2. **Sync and shards:** reuse manifest-verified staging/transport primitives. Download once into external content-addressed caches using scoped read access. Derive deterministic global example assignments from run seed and step; preserve grouped comparisons and family balance. Retain a hash of each rank's actual exposure. Transfer only the required closure, never entire custody trees or sealed evaluation material.
3. **Rendezvous and execution:** global rank drives collectives and sampler identity; local rank selects the device. Use run-specific bounded rendezvous, rank heartbeat and collective timeout. Failure, lost authority or deadline kills and drains every participant; a remote restart cannot reset the clock.
4. **Checkpoint commit:** ranks acknowledge matching parameters and completed update count. Coordinator commits a manifest only after required model, optimizer, scheduler, RNG, sampler and rank receipts are durable and verified. Orphaned partial attempts remain incomplete. Resume retains the original allocation and numerical identity.
5. **Publication:** workers emit immutable result objects; the coordinator alone moves HWM heads until server-side conditional writes are implemented. Serving promotion stays a separate verified decision.

## Batch and heterogeneous GPU decision

Three equal partitions cannot exactly consume global batch 2048. Either adopt a newly versioned divisible batch and accept changed optimization semantics, or implement unequal deterministic partitions with mathematically appropriate weighted loss/gradient reductions and a new reproducibility proof. Rank-local ranking groups make this more than changing an averaging divisor. Neither option is exact resume of the existing objective. Benchmark the 4090/3090 combination under an approved short qualification window before choosing it for throughput.

## S3 layout and grants

Preserve existing dataset/model blob namespaces and current learning schemas. Proposed new run namespace: `runs/<run-id>/inputs/manifest.json`, `attempts/<attempt>/ranks/<rank>/`, `checkpoints/<step>/manifest.json`, and immutable receipts. This is a design addition, not a deployed prefix. Checkpoints reference content-addressed payloads; rank outputs never overwrite shared HWM heads. Scoped dataset readers, per-run artifact writers and a separate promotion publisher need provisioned policies and readback evidence. SmartCluster must bind all participants and their local leases to one original bounded allocation; local locks or a user-controlled grant JSON are insufficient.

<!-- pagebreak -->

# Concrete change sequence and acceptance

**Reviewable implementation plan · Page 5 of 5**

No implementation PR was opened while PR94 occupies the active slot. The following are focused candidate changes in this worktree after that lifecycle resolves; they are not claims of implemented behavior.

| Order | Concrete patch scope | Acceptance evidence required |
|---|---|---|
| 1. Storage concurrency | Add version-bearing reads and conditional create/update to `crates/jope-datasets/src/store.rs`; pass the observed version through learning `mirror.rs` and `sync.rs`. Preserve immutable object behavior and gap reconciliation. | Deterministic two-writer race with one winner and one conflict; absent-pointer race; stale update rejection; existing reconciliation/tamper tests; separate bounded S3 proof after scoped credentials exist. |
| 2. Inventory and dry-run launcher | New training topology/manifest module and CLI plan command beside current `train.rs`; reuse `transport.rs`, native build receipts and input-closure verification. Do not create another generic SSH campaign framework. | CPU-only plan fixtures for three host identities; duplicate ranks/devices refused; wrong source, missing shard, sealed input and expired grant refused; dry-run causes no remote writes or launch. |
| 3. Native multi-host lifetime | Extend `smartcluster_worker.rs`, native provisioner and task descriptors for explicit participants, host-local GPU leases, aggregate deadlines and drain receipts. Read nested SmartCluster AGENTS before editing those crates. | Positive authorized path plus missing/revoked/expired participant tests; one-rank crash drains all hosts; reconnect cannot mint authority, extend budget or replay a consumed start. |
| 4. Rank and checkpoint transport | Update `training.rs`, `neural.rs`, `controller.rs`, `task_contract.rs`; separate global rank/device, replace file-only rendezvous, collect verified per-rank receipts. | World-1 regression parity; two-host collective correctness; actual optimizer updates with changed parameters; equal rank parameter hashes; interrupt/resume matches declared numerical tolerance and original step window. |
| 5. Three-rank partition | Version sampler, grouped-loss reduction and checkpoint execution partition; remove hardcoded device-class assumptions only through a qualified runtime contract. | No missing or double-counted global examples; correct weighted gradient reference; family/group invariants; 3090/4090 resource/throughput receipts; incompatible exact resume refused. |
| 6. Closed learning loop | Bind distributed completion to existing teaching admission, HWM comparisons, model export and separate promotion evidence. Correct stale CLI/runbook examples through their owners. | Admitted teacher → training updates → trained proposals → bounded fit/evaluation under the same comparison identity. Retain failures and unknowns; no held-label leakage; improvement and serving claims require their own evidence. |

## Checks completed versus outstanding

Completed: Git/worktree/PR and shared-claim inspection; remote main SHA; two GitHub run readbacks; source inspections and targeted call-site searches; read-only three-host SSH/GPU inventory; local service-state readback. Hardware probes collected one GPU per host. No training tests were executed: **zero optimizer updates, zero GPU allocations, zero S3 mutations**. No campaign budget was consumed or reset. Build/CI testing is intentionally deferred because this change is an audit document, and full exact-head CI is still mandatory before any future merge.

Document acceptance: Markdown/PDF parity, exactly five rendered pages, nonempty extracted text per page, source-path existence checks, and `git diff --check`. The PDF is a formatted sibling of this report; the worktree copy is `docs/training-support-audit.md`. A generation receipt records source SHA, output hashes, render time and page checks.

## Handoff and evidence interpretation

Audit owner: Codex `/root`. Canonical shared AGENT_CHAT and lane AGENT_CHAT receive locked append-only findings. Existing source owners retain their paths and PR94 lifecycle. No index mutation, commit, push, merge, IAM change or service restart is part of this audit.

The next useful milestone is **a tested, manifest-bound plan plus storage concurrency fix**, followed by a bounded two-host numerical qualification. A configured launcher, green fixture tests, successful hosted CI, live distributed training, cross-dataset improvement and serving promotion remain six separate milestones. Current hardware reachability satisfies only a small part of the live-training prerequisites.
