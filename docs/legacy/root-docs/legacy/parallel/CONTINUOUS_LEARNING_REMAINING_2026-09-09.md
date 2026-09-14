# Continuous learning: remaining work and remote monitoring

Evidence window starts 2026-09-09 20:18 UTC on GPU node `158.101.20.226`.
This inventory covers the restored continuous-learning runtime and the still-open
qualification work recorded in the prior progress review. Historical project-wide
roadmaps include other campaigns; those are not asserted complete by this report.

## Current evidence

At 20:19 UTC, controller PID 2370 and GP PID 2342 had run since the 13:05 reboot
without service restarts. GP had 104 active workers, 322 queued views, 27,904 total
completed views, and zero failed views. The total includes restored historical
work; it is not the count produced during this observation window. Disk usage was
12%, with approximately 5.1 TB available. No warning-priority journal entries were
returned for these two services since 18:15 UTC.

The controller had completed 51 automatic 3,000-step blocks, with block 52 active.
The latest published snapshot contained 7,516,955 records, 849,186 more than the
first automatic snapshot. All eight GPUs subsequently showed 46-75% utilization;
a zero-utilization sample at block startup did not establish a stalled learner.

The latest completed fixed-panel result at that observation was cycle-000050:

| Measure, lower is better | Recovery baseline | Cycle-000050 | Change |
| --- | ---: | ---: | ---: |
| Sequence NLL | 1.394498497 | 1.412826672 | +0.018328175 |
| Pairwise Bradley-Terry loss | 0.533676382 | 0.666974910 | +0.133298527 |
| Listwise ranking loss | 2.262484282 | 2.512873352 | +0.250389069 |
| Runtime Huber loss | 0.076231269 | 0.062914201 | -0.013317069 |

All 52 observed continuous monitoring receipts used the same 2,048-example panel,
corpus, and evaluator. Best continuous NLL was 1.396643236 at cycle-000025, still
above recovery. Runtime prediction improved while sequence and ranking quality
regressed. This is an adaptive validation diagnostic, not independent evaluation.

Across the first twelve reported GP generations, exact-comparison neural exclusive
wins over both generic and shuffled controls were
`10, 10, 13, 8, 7, 7, 4, 7, 4, 6, 5, 3` out of 140 matched triplets each.
These mix 124 training and 16 validation cuts. They must not be presented as 140
independent held-out datasets. The new observer reports those partitions separately
and uses an explicit `1e-9` relative / `1e-12` absolute comparison tolerance.

## Work closed by the recovery, preserved as evidence

- Original and follow-up cycles completed with saved searches and predictor exports.
- All 437 retained ledgers and 65,294 older labels were reconciled into the corpus.
- Production-size interrupted/resumed cached packing matched the original artifacts.
- Native GP waits for new work and honors explicit generation, RNG and cohort bindings.
- The conflicting legacy generation remains quarantined with its evidence preserved.
- Automatic GP -> verified snapshot -> training completed without manual cycle launches.
- Supervisor crash and actual node reboot recovery passed with saved work preserved.
- Full training history and normal TUI reconnect were restored.

Evidence: `runs/continuous-recovery-20260909/restoration-acceptance.json` and
`docs/CONTINUOUS_LEARNING_2026-09-09.md`. Older progress-review statements that the
retained-label recovery or cached-packer activation were outstanding are superseded
by those later receipts.

## Immediate monitoring deliverable

`tools/continuous_status.py` is a standalone observer. It reads controller/GP
heartbeats, task attempts, complete training metric lines, snapshot counts, GPU
utilization, fixed-panel receipts and completed tournament reports. It checks
manifest and checkpoint-state bindings, rejects mixed evaluation identities,
separates training/validation denominators, and does not impute absent seed scores.
It tolerates an incomplete final metrics write while rejecting corrupt complete
lines. It reports regressions separately from process failures.

The dedicated `jope-parallel-health.timer` invokes it every minute. Its service
can write only `runs/continuous-health-20260909/status` and checks its frozen script
hash before execution. Output is `latest.json` plus compact append-only
`history.jsonl`. This is local reporting on the remote node, not an external
notification subscription or an automatic checkpoint-selection policy.

Alerts cover stale/future heartbeats, changed GP executable, GP failures/retries,
paused operation, overdue/retried tasks, stalled training or packing, missing
monitoring, lack of GP completions, GPU count, and fixed-panel regression.
A receipt error produces an explicit incident instead of a fabricated healthy
result. A current alert may coexist with healthy GP and training throughput.

## Prioritized remaining work

1. **P0: Keep operational and learning status distinct.** Complete the live observer
   deployment and capture successive samples showing real GP/optimizer progress.
   Acceptance: timer active, hash check succeeds, reports refresh, both production
   controller PIDs remain unchanged, and regression alerts remain visible.

2. **P0: Validate observer failure handling.** Exercise changed panel/evaluator/corpus,
   nonfinite losses, missing or duplicated arms, missing seed scores and concurrent
   metrics writes. Acceptance: each produces either explicit rejection or the
   specified incomplete-data status, with no false win or independent-gain claim.

3. **P0: Track the current in-flight generation and block to completion.** Inspect
   committed receipts, not only busy processes. Acceptance: completed-block and GP
   counters advance, current attempts remain bounded, next automatic work starts,
   and no new incident or unexplained inactivity is hidden by GPU memory allocation.

4. **P1: Separate training continuation from proposal-checkpoint promotion.**
   `Supervisor.finish_task('train')` currently advances the only checkpoint pointer
   after structural validation. Introduce separately journaled candidate and serving
   checkpoint identities, a declared acceptance rule, and retained rejected candidates.
   Acceptance: a deliberately worse candidate cannot change proposal authority;
   restart cannot double-apply a promotion; continuing training still has explicit lineage.

5. **P1: Freeze the promotion evaluation contract before selecting on it.** The
   current monitor explicitly declares `training_selection_used=false`. Do not
   silently turn those same receipts into a selection claim. Freeze an additive
   policy covering metrics, tolerances, evaluation identities and data partitions.
   Acceptance: selection receipts disclose adaptation, distinguish diagnostics from
   decisions, and leave sealed tests unopened until their declared final evaluation.

6. **P1: Choose the candidate initialization through a controlled comparison.**
   Recovery beats every observed continuous checkpoint on NLL and has much better
   ranking loss, but a rollback is not proven to improve downstream seeds. Compare
   recovery, best continuous NLL and latest candidates on the same declared budget.
   Acceptance: matched evidence supports the chosen proposal checkpoint, with all
   candidate checkpoints retained and no historical receipt rewritten.

7. **P1: Right-size updates to fresh teaching information.** Each snapshot currently
   causes another 3,000 steps even when only a small fraction of records changed.
   Measure novel valid program/cut labels, improved teachers and affected families;
   distinguish those from repeated source observations. Compare shorter blocks,
   delayed training and the current schedule at equal resource or example budgets.
   Acceptance: a declared rule improves validation/downstream results without starving
   newly improved teachers, and unchanged labels cannot trigger redundant retraining.

8. **P1: Investigate the ranking-loss deterioration.** Inspect fixed-panel family
   slices, teacher/comparator label distributions, reward ties and head gradients.
   Existing NLL, pairwise and listwise objectives are already present; another loss
   term is not justified by utilization alone. Acceptance: a reproducible diagnosis
   and a small matched intervention improve ranking without losing seed quality.

9. **P1: Qualify optimizer behavior across corpus transitions.** Checkpoint receipts
   confirm fresh FP32 AdamW and a reset sampler at each new corpus. Compare that
   declared policy against retained moments and a controlled learning-rate schedule,
   preserving vocabulary/model/corpus bindings. Acceptance: restart within one run
   remains exact and any cross-corpus optimizer change has separately bound evidence.

10. **P1: Measure direct neural seed quality separately.** Report initial evaluated
    neural slots, retained-teacher comparisons, final GP gains and missing coverage
    for training and validation families. Acceptance: a GP improvement is never
    attributed to direct neural generation unless the initial seed receipt supports it.

11. **P1: Establish downstream gains over both controls.** Run candidate checkpoints
    with generic and shuffled-context arms, matched RNG, frozen populations and equal
    budgets. Report per-family wins/losses/ties, RMSE effect sizes and repeat variation.
    Acceptance: stronger neural seeds have evidence beyond shared retained teachers
    and repeated adaptive cuts; no result is relabeled independent evaluation.

12. **P1: Preserve a stable best-known proposal population.** After the promotion
    contract is qualified, prevent an unaccepted training checkpoint from replacing
    useful proposals or teachers. Acceptance: exact program/cut/evaluator identities
    survive transitions, old scores never transfer to modified programs, and rejected
    candidates remain available for diagnosis.

13. **P2: Measure time spent packing, starting ranks, training and monitoring.**
    The restored system works, but GPU gaps remain part of its schedule. Capture
    phase durations and useful labels per GPU-hour; distinguish allocated memory
    from execution. Acceptance: a measured bottleneck breakdown supports the next
    optimization and observer overhead is included.

14. **P2: Reduce cumulative packing cost if it dominates.** The cache avoids repeated
    parsing, but every snapshot still includes the retained source set and global merge.
    Profile hashing, merging, publication and verification separately. Acceptance:
    any incremental representation reproduces records, labels, vocabulary and provenance,
    and interruption cannot publish a partial corpus.

15. **P2: Finish a clean 256-versus-1024 batch comparison.** Prior equal-example trials
    favored 1024 on both validation losses, but the baseline timing was externally
    contended. Production remains 256. Add complete startup/training/finish ownership
    evidence, then run uncontended paired blocks. Acceptance: quality, actual elapsed
    resource cost and downstream gains all support any change; 2048 remains rejected
    by its recorded pairwise result unless a new qualification supersedes it.

16. **P2: Bring every competing launcher under ownership rules.** Audit optional
    encoders, qualification runners, continuation wrappers and ad hoc systemd units
    for shared leases, bounded children and parent-death cleanup. Acceptance: a
    competing launch cannot take training GPUs or write the live queue while another
    controller owns them; cancellation leaves no surviving workers.

17. **P2: Test candidate/serving recovery when that new mechanism exists.** Existing
    crash/reboot evidence covers the current controller only. Add interruption points
    around evaluation, decision journaling and pointer publication. Acceptance:
    restart produces one consistent serving identity and retains all training work.

18. **P2: Plan checkpoint and evidence retention.** At one block every several minutes,
    checkpoints and monitor history continue growing. Inventory model/optimizer sizes,
    lineage references and daily growth before proposing cleanup. Acceptance: retained
    parents, selected candidates, rollback checkpoints and evidentiary receipts remain
    resolvable; any archival policy has a tested restore path.

19. **P2: Verify an off-node recovery copy.** Local disk has ample headroom, which does
    not protect against node loss. Inventory what is already mirrored and identify
    missing current checkpoints, manifests and evidence. Acceptance: a selected live
    checkpoint and its complete dependencies can be reconstructed from a hash-bound
    copy without using the original node. This observation does not assert no backup exists.

20. **P2: Keep source and running releases unambiguous.** The remote top-level source,
    local working tree and immutable active release need not match. Extend the inventory
    of active script/binary hashes, configuration bindings and rollback paths.
    Acceptance: a deployment never assumes directory synchronization proves runtime
    identity, and no mutable tree overwrites an active release.

21. **P2: Keep the TUI consistent with the new health receipt.** The separate observer
    provides alerts now; decide how to surface the same fixed-panel baseline, latest
    result and training/validation arm denominators in the established TUI. Acceptance:
    all five views, resize and reconnect preserve history and distinguish training,
    monitoring and promotion. No TUI deployment is required for the JSON observer.

22. **P3: Finish kernel quality/performance qualification.** Prior remote smoke tests
    passed contracts and screening, but full kernel quality/performance is still open
    in the recorded review. Acceptance: intended workloads, parity, resource limits and
    quality thresholds pass in a separately bound candidate release before activation.

23. **P3: Resolve Sonar entity/acquisition grouping.** Exact binary replay passed;
    source grouping suitability remains unresolved. Acceptance: documented grouping
    supports leakage-resistant splits before promoting beyond encoding qualification.

24. **P3: Revisit the larger-pool selector only with new evidence.** Its three matched
    repeats were approximately 0.18% worse than pure_first64, so it remains undeployed.
    Acceptance: a materially changed, frozen proposal policy beats the existing control
    under the same quality and interruption-accounting rules.

25. **P3: Recover the external disk before claiming a fresh inventory.** The prior
    review records a kernel-I/O metadata hang on `/mnt/easystore`. Its current condition
    was not probed during remote GPU monitoring. Acceptance: coordinated storage
    recovery permits a fresh hash inventory; preserved catalogs remain explicitly
    unverified against that external copy until then.

26. **P3: Close the session-resume incident separately.** Earlier investigation found
    an active Codex writer lock and terminated the identified writer chain. Resume
    success was not demonstrated in this monitoring task. Acceptance: one chosen
    client resumes the intended thread without a second writer, with its history intact.

## Execution order and boundaries

Finish items 1-3 in this monitoring pass. Items 4-12 are the next meaningful learning
milestone: prevent unqualified proposal replacement, then demonstrate stronger seeds.
Items 13-21 address throughput and durable operations. Items 22-26 retain known
qualification and environment gaps without confusing them with the restored pipeline.

Production remains authorized to run until explicitly paused. The observer neither
pauses it nor changes training, proposal authority, manifests, queue entries or data.
Checkpoint policy changes require a new source-bound release and concrete acceptance
evidence; throughput alone does not close the learning-quality gap.
