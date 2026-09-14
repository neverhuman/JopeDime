# Continuous learning restored — 9 September 2026

Continuous operation passed live acceptance at 13:15 UTC. The original recovery,
the saved follow-up cycle, and one fully automatic GP → corpus → 3,000-step
training handoff completed. GP, packing and training ran concurrently. The
supervisor continues until explicitly paused; individual jobs remain bounded.

## Running configuration

- `jope-parallel-campaign.service`: hash-bound Python supervisor, unlimited service
  runtime, durable journal, checkpoint continuation, retry backoff, exclusive lock.
- `jope-parallel-gp.service`: 104 workers, one thread each, CPUs 8–59 and 70–121,
  `--wait-for-work`; empty queues retain a fresh waiting heartbeat.
- Training: original v44 neural executable, eight ranks, batch 256/GPU,
  3,000-step blocks, checkpoints every 1,000 steps, 192/256-GiB host limits.
- Packing: verified atomic CPU publication, private scan cache, complete provenance,
  CPUs 60/61. Snapshots trigger after 104 new completed searches or ten minutes
  with changed labels. The newest verified snapshot trains after the current block.
- New tournament jobs bind their generation, checkpoint, population and RNG.
  All comparison arms receive matched RNG and 1,000-evaluation/600-second budgets.
- `jope-parallel-console.service`: persistent, hash-verified tmux session `jope`,
  independent of training and recreated after reboot.

Run the normal console from the repository root:

```bash
rtk proxy bash parallel/tools/tui-remote.sh
```

Detach with **Ctrl-B, then D**. Reconnect attaches to the same session. Normal
launcher, resizing, navigation, cold history loading and reboot reconnection passed.

## Verified results and preservation

The original expired cycle completed at 11:09:34 using its exact v44 arguments,
420 saved searches and 140 predictor exports. Its missing training block initialized
from the selected auxiliary step-300 checkpoint `80dc36c5…`. The historical
04:10 UTC qualification deadline and original initialization argument stayed intact.
Subsequent work is recorded as continuation, not success before that deadline.

The saved follow-up cycle completed at 12:52:45. Its launch omitted retained source
arguments. Before packing, its parent was suspended while the qualified cached
packer published a recorded corpus from all prior sources plus its 420 searches.
The exact v44 executable and cycle arguments stayed unchanged. All 140 predictor
exports passed; the original parent resumed after reconciliation.

Exact checks preserved all **437 retained ledgers and 65,294 older labels**, rejecting
withdrawn, out-of-cohort and sealed-test supervision. The corrected follow-up corpus
has 6,658,401 records. A production-size interrupted/resumed cache qualification
matched the original v44 corpus byte-for-byte, including its index.

The first automatic snapshot contained **6,667,769 records**, incorporating 35 new
searches at the ten-minute threshold. Training completed 3,000 steps from the prior
committed checkpoint. Another snapshot was requested after **104 additional searches**.

Accepted automatic checkpoint:
`runs/continuous-learning-v2-20260909/cycle-000000/train-2/checkpoints/step-3000`

State SHA:
`a34a813b2f4531f2800bb78e9830fe692da156bfaa955dcd6ba159257eeeff17`

Corpus index SHA:
`0df00208c4af99bc52774522f0f5d84ce891a7418b3ad5eb94319325d5f19d2b`

All 83 original runs / 185,906 records remain. Acceptance found **197,906 records**
across 88 discovered runs, with checkpoint lineage, invalidated branches and the
empty failed auxiliary attempt preserved.

## Validation is an experimental outcome

On the same frozen 2,048-example adaptive validation panel, sequence NLL was
1.39449847 after original recovery, 1.41777979 after the follow-up cycle, and
1.42137299 after the first automatic block. These blocks **did not improve that
monitored loss**. Training loss, frozen validation, seed quality, retained teachers
and GP improvements are reported separately. No final tests were opened.

The first continuous generation also contains the recorded reboot recovery test;
individual search receipts retain their resume and conservative search-clock
metadata. Nominal search budgets and paired RNG bindings stayed unchanged.

## Restart and failure evidence

A live supervisor SIGKILL test restarted automatically without interrupting GP;
all prior committed receipts survived. A real reboot then restored GP, supervisor
and console automatically, preserving the checkpoint, manifest, 34 committed
receipts and 104 ledger prefixes. Completed journals were verified against their
final observation snapshots. No prior-boot workers survived.

Nine supervisor tests passed on both machines. Native GP/queue/tournament code is
unchanged from the previously qualified release; CPU and neural tests, default GP
behavior, empty-queue wakeup, cohort/RNG rejection, durable publication, cache
agreement and full-history TUI tests passed. Healthy waits for GP or packing are
explained; unexplained inactivity with ready work warns at two minutes and becomes
an incident at five minutes.

The first attempted continuous generation was interrupted by a competing v44 GP
service that ignored explicit RNG fields. Its 420 queue entries were isolated and
its outputs quarantined, with ledgers and incident evidence preserved. It is never
used as valid comparison data or teaching input. `AGENT_CHAT.md` records exclusive
ownership and the subsequent coordinated recovery.

## Immutable inputs, controls and rollback

Active manifest: `runs/continuous-recovery-20260909/continuous-manifest-v3.json`

SHA: `de8a16caef54f66a0715438e2015bb15c2518dbb4ebc5e854fba3b2d9ffc71a5`

Supervisor source release:
`kernel-v1-3f706bf6422c64abd4f463c4bee3b6c594a611a766b39f83196b4e51ceae30b3`

Native CPU and observer source release:
`kernel-v1-130acc93dcc7d024108ba2bc783a36200637e2f0627392bafe08633bed416299`

Native CPU/observer SHA:
`054c1c50f341bcd3d3b4194380da41b3ff78dfa971365f05fd1aa7c97dcaa984`

Sources are immutable and both-host hashes match. The supervisor-only reporting
update has a recorded manifest/state migration; checkpoint and job identities did
not change.

On the GPU node, use the active release's `tools/continuous_learning.py` with
`--root /home/ubuntu/jope-dime/parallel` and the absolute active manifest path:
`--check` is read-only; `--pause` and `--resume` control durable operation.
Do not resume the archived invalidated generation or its old manifest.

Earlier service overrides and reversible `99-continuous.conf` changes remain in
`runs/continuous-recovery-20260909/service-activation*`. The v2 `rollback.sh` stops
continuous services, restores prior overrides and leaves the historical incident
paused. Obsolete cycle/encoding units retain boot guards against competing owners.

Main acceptance evidence: `runs/continuous-recovery-20260909/restoration-acceptance.json`.
Related evidence includes `reconciliation.json`, `cycle-0001-corrected-publication`,
`live-controller-restart-v2`, `live-reboot-v2`, `live-generation-v2-verification.json`,
`idle-reporting-migration`, `full-history-restored.json` and preserved attempt logs.
