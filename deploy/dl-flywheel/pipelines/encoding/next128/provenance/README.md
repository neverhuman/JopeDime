# Next128 revision2: four fresh Kumo Small packets on xbabe3

Prepared locally on 2026-10-09. **No live claim, GPU run, S3 PUT, or campaign
clock freeze has been performed by this preparation.** The selection is the
root-collected read-only metadata snapshot, not current-role or fresh-S3 proof.

## Objective and exact scope

Encode sequences938–1065, exactly128 new variant identities in four fixed32
packets. Use Kumo Small alone, output2051, with the exact successful fresh32
numerical worker `0883dd4085e01cc0e1a3353396fd7004854c4ae2e68868f1eb79a053a6cb2a42`.
Retain its full six-check/four-call qualification for every variant.

Every packet must pass current-role checks before claiming/staging, live
resource admission, exact source and dataset VersionId/SHA custody, deterministic
input replay, actual CPU preflight, full GPU qualification/sealed evidence,64
create-only S3 objects with exact-version readback, then a fresh-role additive
stage CAS. No GP labels, teaching permission, or training exposure is claimed.

| Packet | Sequences | Immutable packet ID |
|---|---|---|
|0|938–969|`5b2ff87d02155846a90305370faa8411bef5dcb5f6a3d9146eb92711ba44804a`|
|1|970–1001|`7e5f2ce6b37992b62569cad1400b5b756c319a1430deaa575a3d2765b34765aa`|
|2|1002–1033|`d658c6479af6b1af692b4a33facda0dacaba5c2b42f513af21baadfecfeeab7c`|
|3|1034–1065|`b88b82d928fe768ca0e07e038a42d5a65439f5bc6dd1be1e1323a638a436d9fa`|

This is a newly authorized128-variant campaign. It does **not** borrow/reset the
old production2a process's four-packet/128-variant counter. The existing
all-history128-packet-row ceiling is an additional conservative DB guard, not
original production2a budget accounting. No old sources, counters, deadlines,
claims or stage receipts are changed. The prior failed held32 and its19 vectors,
and the now-complete fresh32 sequences906–937, remain excluded from this support.

## Read first and review

1. `selection-readonly.json` (326033B, SHA
   `947ff1dd0269b82dfa2cf4e3a53ceb052fa4c498b327429410529012499fbd5b`).
   Its observation reports25 historical packet rows (24complete,1held) and768
   stages. These are dated selection-time facts, not a live completion claim.
2. `validation.json` and `revision2.diff`.
3. `run_next128.py`, `freeze_next128.py`, `config.template.json`.
4. `packet0/kumo_controller.py`, `kumo_queue.py`, `kumo_intake.py`,
   `kumo_commit.py`, `campaign.py`, and the child template. Other three child
   controllers differ only in fixed identities/source roots/support bounds.
5. The pinned successful fresh32 source/qualification already reviewed by root.

The reviewed root template SHA is
`ec0e27b35498121091e85c9d7214233ec9c713588e5e755d1a75e73cce66f5e7`.
The independent review must be a JSON object containing at least
`{"state":"passed","template_sha256":"ec0e27b35498121091e85c9d7214233ec9c713588e5e755d1a75e73cce66f5e7"}`.
Pin the actual review file SHA. The freeze helper verifies both pins before
creating any clock file.

## Resource and lifecycle boundary

- One shared1800s wall deadline **and** local monotonic deadline for all four.
  Every child receives the identical values; none computes a fresh clock.
- Original global ceiling remains `1791592146.380285` (2026-10-10
  00:29:06.380285 UTC). Freeze fails unless the full1800s fits beneath it.
- The immutable intake reserves the last600s for publication/commit; GPU
  qualification cannot pass shared-start+1200s. A new child is refused when
  less than660s remains, rather than acquiring work with no useful GPU window.
- Sequential child packets only. GPU units: CPUQuota800%,MemoryMax12G,
  MemorySwapMax0,TasksMax128,Nice10,CUDA0,BLAS/OMP1,KillModecontrol-group.
  CPU role/intake/publication/commit units: CPU100%,512M,Tasks32,Nice19,noCUDA.
  Each retains its inherited tighter180/600s operation cap and shared remaining
  budget. Live worker resource policy is unchanged.
- Failure, blocked claim, empty fixed support, unknown child outcome, missing
  publication proof, clock expiry, or source drift stops the whole controller.
  No next packet, automatic retry, resumption or counter/deadline reset exists.
- Local `run/` and `frozen/` directories are exclusive once-only markers. Keep
  partial outputs. Process cleanup targets only the local session this
  controller created. Remote cgroup limits remain independently enforced.

## Ownership and integration

The local proposal is
`/Users/bentaylor/Documents/Codex/2026-10-06/on-x20/work/heartbeat-20261009-1203/kumo-xbabe3-next128/revision2`.
All frozen absolute paths are deployment evidence for this host/workspace, not
portable defaults. Freeze and execution must use the same local clock host.

Owned source runtimes, on both staging hosts:
`/home/ubuntu/JopeDime/.agent/proposals/dl-flywheel-kumo-xbabe3-next128-20261009/revision2/packet<N>/runtime`.
Create-only artifact roots:
`/home/ubuntu/jopedime_probe_exp/ofm-20261008/dl_final/flywheel_kumos_packet_next128_xbabe3_<packet-id-first16>`.
The controller streams bounded immutable stage files xbabe2→xbabe3; it does not
archive the raw corpus on the Mac. Expected publication is256 objects total,
64 per successfully completed child, in the existing reviewed new-variant
dataset/receipt namespaces. Original metadata remains untouched.

Likely repository integration is additive under
`deploy/dl-flywheel/pipelines/encoding/next128/` with its own isolated offline
runner, fixtures and research classification. This preparation makes no source
repository/PR/manifest/CI edits. Do not alter or repin the earlier fresh32 source
integration payload while adding this successor.

## Launch handoff (root only, after independent review)

Read/reconcile the live shared handoff and claim this exact scope before launch.
Confirm no other xbabe3 GPU launch is being authorized concurrently. The child
itself still checks live resource admission before each claim and CUDA start.

From the local proposal directory, freeze once:

```sh
rtk proxy python3 -B freeze_next128.py --review /absolute/path/to/independent-review.json --review-sha EXACT_REVIEW_SHA --template-sha ec0e27b35498121091e85c9d7214233ec9c713588e5e755d1a75e73cce66f5e7
```

The returned descriptor pins `frozen/config.json`; launch immediately with that
exact SHA through the root's persistent local controller mechanism:

```sh
rtk proxy python3 -B run_next128.py --config /Users/bentaylor/Documents/Codex/2026-10-06/on-x20/work/heartbeat-20261009-1203/kumo-xbabe3-next128/revision2/frozen/config.json --config-sha EXACT_RETURNED_CONFIG_SHA
```

Never invoke `build_packet.py` over this existing packet: its create-only source
writer intentionally refuses existing directories. Never freeze twice.

## Verification, receipts and completion criteria

25 **revision2 successor-only** tests passed in0.272s; the closed fresh32's37 tests were
not rerun. Tests cover shared clocks/backward wall time, original global ceiling,
review/source/support drift, refusal versus empty, held/duplicate ownership,
current-role rollback, no replay, stop-after-uncertain failure, retained earlier
completion, owned process cleanup and exact numerical/controller AST boundaries.

`validation.json` SHA
`52a0c2a647362157b0c701c80fbe346065a039b3205054fe7db33f464a5331c0`
binds source/template/test identities and local-only limits. This is CPU/source
qualification only; it does not replace real host preflight.

Runtime receipts are append/create-only:

- `frozen/clock.json`, four child configs, `frozen/config.json`.
- Root `run/window.json`, per-child intent/stdout/stderr/verified receipts,
  `run/progress-0.json` through `progress-3.json`, then `run/complete.json`
  **or** `run/failure.json`.
- Each child retains `packet<N>/runs/<packet-id>/` with intake, source stage,
  role proof, CPU preflight, GPU intent/result, publication and stage-CAS proof.
- Remote manifest/progress/phase timing/seal/publication files remain under
  each new artifact root; unit name prefix remains `jope-kumo3fresh-*` with
  distinct packet-id suffixes, inherited from the proven child controller.

Completion requires four verified32-row stage commits and256 exact-version
verified objects, all before the one shared deadline. Any incomplete campaign
must report the verified completed subset and retained claimed/unknown subset;
never relabel those as completed or start a retry automatically.

Revision2 also checks the660s reserve immediately before dispatching actual intake and inside the xbabe2 BEGIN transaction immediately before INSERT. Its new delay fixtures prove that both boundaries refuse without a live claim. The prior23-test packet remains byte-for-byte preserved and unapproved. `build_packet.py` is pinned historical source-packaging provenance; do not run it to reconstruct revision2.
