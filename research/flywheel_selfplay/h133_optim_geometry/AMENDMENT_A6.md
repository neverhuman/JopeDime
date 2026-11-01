# H133 Amendment A6 — supervised load and memory pause

Date: 2026-10-31 UTC. Owner authorizations: h133_t1630.md and h133_t1715.md. Parent preregistration #1186 is merged at `dbd88c910f30742a93e839289597beab452c2879`; this prospective amendment must merge before the next H133 run.

## Frozen science

A6 changes launcher/resource handling and the seed roster only. The H133 hypothesis, frozen r882 comparator, DOPE c51a56d0, train-only data closure, family split and fold seed 133100, arms CONTROL / LOWLR_LONG / SGDM, optimizers and learning rates, update counts, batch generation, assessment fold, family-bootstrap/Holm/harm/Elo calculations, and every admission gate remain unchanged.

Seeds are 133, 134, and 138. Seed 138 replaces 135 because H135 owns seeds 135–137; H133 has not used seed 135. SGDM remains fit-drift-calibrated at LR 0.003 using 160 fit-only calibration updates; the main SGDM cell remains 40 updates. Report its realized 40-update adapter drift.

## Resource behavior

Run CPU only at nice 19, no more than four threads, no qualification.lock, and verify DOPE SHA c51a56d0 before and after each cell. The four-hour wall budget includes startup, computation, and paused time.

The A6 worker delegates only load and available-memory pause decisions to its pinned parent launcher:

- The launcher sends SIGSTOP when load1 > 86.7 or MemAvailable < 30 GiB.
- The launcher sends SIGCONT only after load1 < 80 and MemAvailable >= 34 GiB continuously for 60 seconds.
- Record every SIGSTOP/SIGCONT pair in pause_events.jsonl and resource snapshots in resources.jsonl.

All other resource guards remain hard aborts: RSS > 6 GiB, more than four process threads, disk >= 93.8%, H133 scratch >= 2 GiB, or host CPU > 85%. CUDA remains hidden and uninitialized. The worker refuses load/memory-supervised mode unless launched by the pinned parent process.

The preferred host is xbabe1 when the complete required input closure is under 2 GB. H50 input bytes measured locally at 457,391,900; xbabe1 currently has 175,187,219 pool bytes, but lacks train tables and the r882 checkpoint. Stage the missing pinned inputs and source/runtime to xbabe1 only after the PR merge, verify all hashes there, and record the host and input-closure size. If staging or any pinned hash fails, do not launch.

## Exact regression gate and launch

After merge, run the unchanged-r882 comparison first. It must reproduce the owner-pinned family utility rounded to seven decimals as 1.1377942, and the paired prediction, utility, CONTROL fit loss, and fit-gradient differences must be exactly zero. Attempts 008–012 produced 1.1255360736621804 under their then-pinned environment, so this is a material preflight discrepancy that must be resolved by the merged-source regression gate. No treatment cell may start unless all expected regression values match. Launch exactly once under the four-hour monitor. No resume or automatic retry. A mismatch is an incomplete attempt with no arm scoring, and a previous cell from a differing baseline is not mixed into the A6 outcomes.

The monitored launcher and worker are pinned in manifest.json:
- launcher.py SHA-256 `d5cbb5f7fb62e6853847b4c85b62fab5bdd1522c6c05225e0e88aa0276c3794e`
- runner.py SHA-256 `0143f42efc3d20f962fe02065c601e0ddde69bedd4bbf1cb024fa6a045b966c1`
- source parent A5 runner SHA-256 `9c9ab6b1b753362b134d224a935f8ba81e6d6fe9edcef612c7b06093a2b7230a`

The runner writes into a fresh run root and refuses to resume existing cells. Before it starts, the launcher requires a post-merge receipt recording the merged PR and current main readback, then verifies that receipt's runner, launcher, and manifest hashes against the staged source. The launcher records the merge receipt, host, command, source/data paths, measured input-closure size, resource snapshots, pause intervals, elapsed time, worker status, and metrics hash.

## Attempt disclosure through attempt 012

A6 carries forward all voids; none are erased or included as scored treatment outcomes.

- Attempts 001–007 are preserved in startup_attempt_001.json through startup_attempt_007.json. They stopped on import, thread-count, CUDA-helper-task, or CPU-landmark-thread preflight failures before regression/treatment scoring; each receipt reports zero treatment optimizer updates.
- Attempts 008–009 passed the then-pinned r882 regression but stopped before the first treatment update (missing helper, then a fifth CUDA helper task). Neither produced a treatment outcome.
- Attempt 010 performed 40 CONTROL seed-133 fit updates, then the fit-only predictor failed on the first fold-0 context before any assessment model prediction. Those 40 updates are unscored and void.
- Attempt 011 stopped at the load guard before input reads, predictions, or optimizer updates.
- Attempt 012 passed the then-pinned regression and completed CONTROL seed 133 (40 updates, adapter drift 0.4389823302707865, fold-0 family utility 1.0960967447286205). LOWLR_LONG seed 133 then performed 100/160 fit-only updates; load1 was 424.1279 and MemAvailable 21.64 GiB, so the hard guard stopped it before assessment scoring or a cell receipt. Those 100 updates are unscored and void. Attempt 012 held qualification.lock, a protocol deviation; it released on exit. A6 and attempt 013 use no qualification lock.

No main outcome is inferred from the partial attempt. A complete closeout remains contingent on all nine newly executed cells and will report family medians, family-cluster bootstrap CIs, Holm results, harm fractions, and family-paired Elo, alongside all voids and realized drift.
