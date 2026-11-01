# H189 — plain weight averaging and EMA on an r882 continuation

Owner: `luna_ns_hbaseline`; assignment `~/nudges/hwave_t0800_h189_hbaseline.md`; go ruling `~/nudges/hwave_t0805_h189_go.md`. Diagnostic, preregistered CPU research only. H189 cannot alter the live loop, adopt a checkpoint, or authorize serving.

## Owner decision and lane exception

The 2026-11-01 08:05 owner decision explicitly permits one H189 research lane while other owners' six PRs remain open. It also clarifies the disk guard: compute `100 * used / (used + avail)` from `df -B1 --output=size,used,avail /`; rounded `df -h` Use% is not the gate. H189 reuses the just-freed `h182-hbaseline` lane path, so the H182-to-H189 lane transition is net-zero. This exact owner exception supersedes the otherwise blocking queue/capacity observations for H189 only; no other lane or PR is modified.

## Question and fixed arms

Does averaging the private r882 adapter weights from a single 80-update continuation recover any of the family-utility gain seen with fresh-adapter ensembles, without adding inference cost?

- `R882`: zero-update reference.
- `CTRL80`: the step-80 weights from the common continuation; descriptive control only and excluded from selection and Holm correction.
- `SWA80`: primary; uniform elementwise mean of the 29,211 adapter parameters at steps `40,45,50,55,60,65,70,75,80`.
- `EMA80`: secondary; adapter-only exponential moving average with decay `0.95` over all 80 post-update states. Initialize at step 1 and apply `ema_t = 0.95*ema_(t-1) + 0.05*theta_t` for steps 2–80.

Each `(outer fold, seed)` has one shared continuation trajectory and one H176/H156 batch plan. Thus CTRL80, SWA80, and EMA80 are paired transformations of the same fitted path, not independent runs. Only the private 47-tensor/29,211-parameter adapter changes; the r882 anchor, DOPE, and every other model tensor remain frozen.

## Data, folds, and training

Use the pinned real-S3 TRAIN cohort only: 1,219 contexts from 205 families. Use the 91-context/19-family `DL-Val` set only for the exact frozen-r882 baseline gate; do not use it to tune or select treatments. Do not read sealed, holdout, or meta-test inputs. The live flywheel remains read-only. H189 inherits H182's real-catalog and numeric-input custody closure, H176 continuation loss/optimizer, H156 batch construction, and H182's merged-source guarded inference engine.

Use five family folds from `family_folds(seed=189140)`. For outer fold `j`, `A=fold[j]`, `B=fold[(j+1) mod 5]`, and FIT is the remaining 123 families. Every TRAIN family appears exactly once in pooled B and once in pooled A over the five outer folds; FIT/B/A are disjoint in every cell. For each of three outer seeds (`189101`, `189102`, `189103`), run one continuation in each outer fold, for 15 trajectories and exactly 1,200 optimizer updates total. The bootstrap seed is `189141`.

The continuation starts from a private copy of the exact r882 adapter, with fresh AdamW state (`lr=1e-4`, `weight_decay=1e-4`, betas `(0.9,0.999)`, default epsilon), global gradient clipping L2=1, and the pinned H146/H156 SmoothL1 plus `0.1` KL objective. Each of the 80 updates samples eight FIT contexts and, within each, the four lowest raw-fitness candidates plus 20 sampled without replacement. No B, A, or DL-Val labels enter updates or minibatch construction. The exact batch stream is deterministic and common to all three reported arms in a cell.

Before any TRAIN outcome pool is opened, verify the exact native 91-context r882 `DL-Val` utility vector bit-for-bit and mean `1.1377941898164525`. If it fails, stop with zero optimizer updates. Then train all cells using FIT only; do not open B or A outcomes until all 15 trajectories and their model states are complete. Score pooled B for R882/SWA80/EMA80, compute decisions, and fsync the full B lock before opening any A outcome. CTRL80 is evaluated on A only as a descriptive control.

## Pooled selector and confirmatory rule

H189 reuses H177's family-wise aggregation, shared 10,000-draw paired-family bootstrap, strict `+0.001` margin, one-sided bootstrap p-value construction, and durable stage-order-lock pattern. H177's committed `pooled_b_select` is hard-coded to four arms and Holm `m=4`; H189 must honor the explicitly assigned Holm `m=2` over only SWA80 and EMA80, so H189 implements the same pooled statistic for exactly these two prespecified treatment arms. CTRL80 and R882 do not enter the multiplicity family. Ties reject. B admits a candidate only when its equal-family gain is strictly greater than `0.001`, its paired-family bootstrap LCB95 is strictly greater than zero, and its Holm-adjusted one-sided p-value is strictly below `0.05`.

The committed H178 receipt is preserved and checked as provenance (`COMPLETE`, null/harmful admission rate `0.0`, H177 analysis SHA matching the pinned H177 source). It reports a four-arm `m=4` replay; H189 does not mislabel it as an independent calibration of H189's `m=2` family. H189's selector rule and two-arm multiplicity are fixed here before outcomes.

After the pooled B lock is fsynced, score both candidate arms on pooled A without changing selection. Report equal-family utility gain versus R882 by seed, median-of-seed gain, 10,000-draw paired-family bootstrap 95% interval, positive-seed count, and Holm-adjusted B p-values. `ADVANCE_CANDIDATE` requires B admission and all three pooled-A seed gains strictly positive with median gain and its bootstrap LCB95 strictly above zero. Otherwise report `NO_ADVANCE`. A result remains diagnostic and cannot be adopted by this run. Report Elo and utility only; no AUC or speed claims.

For interpretation only, disclose the supplied pooled-selector `MDE80=+0.0258` heuristic. It is a planning reference, not a measured H189 power result and not an additional pass/fail condition.

Persist complete per-context B cost vectors, choices/utilities, seed/fold mappings, pooled panels, and the fsynced B decision lock for independent replay. Large arrays and run artifacts stay outside Git and go to versioned S3 custody. Keep only compact metadata and metrics in the repository.

## Integrity, resources, and launch gate

Verify DOPE composite SHA-256 `c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814` before and after. Verify adapter inventory and all non-adapter state throughout. Record opened/refused/protected reads; any protected read or FIT/B/A overlap invalidates the attempt.

Run once on `xbabe2`, CPU only, under `nice -n 19`; Torch/math threads one, aggregate supervisor-plus-worker process-tree threads at most four, aggregate RSS strictly below 6 GB. Stop on host CPU above 85%, MemAvailable below 16 GiB, disk free below 40 GB, `df -B1 /` used fraction at least 93.8%, RSS at least 6 GB, threads above four, or a four-hour wall limit. Never use `xbabe0`; do not use GPU or modify the live flywheel. Recheck every guard before launch and continuously in the monitor.

The preregistration PR carries the data-free tests, source/input/fold pins, runner and monitor. Run focused local checks and all five `jope-records` verifiers on the exact PR head. The standing owner exception permits admin squash merge after these local gates; GitHub CI is not a wait condition. Bind one attempt to the merged source, exact launch JSON, and reviewed PR head. Close out separately with README, compact `metrics.json`, one canonical overnight-index row, checksummed/versioned S3 archive and manifest, and no large Git artifacts. Reconcile the final metrics and custody receipt before removing the reused lane under the preservation procedure.
