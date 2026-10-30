# H53b preregistration — extended learn-from-zero wave

**Preregistered before any H53b training cell.** Branch `research/h53b-wave`; base `62327dd06f1f5eb2d8504be73fcdc593349a601d`; the six-cell wave is copy-only and never attaches to or reads a live loop. H53 is a three-update pilot and is inconclusive about 882-round convergence.

## Question, arms, and inputs

Can a fresh random 29,211-parameter outer adapter, trained from zero with frozen DOPE, approach or exceed the r882 reference after a longer horizon? Run six cells: scales 1.0 and 0.1 crossed with seeds 12, 13, and 14. Each cell starts from a fresh initialization; no candidate adapter or optimizer state transfers from r882 or another cell. The r882 checkpoint is used only to load the architecture/reference and score the fixed comparison baseline; its adapter is discarded before candidate training. DOPE c51a56d0 is frozen and guarded before and after every cell.

Pinned offline inputs: source-manifest file SHA-256 `99a53569fcaf59f9309e69329c830ae1684adff382da3ae22e0b2645c4ebc4d4`, manifest identity `3eb4a62acea512f528500b043299a5968e1c9b1d1b1759bbf77a426b9fa1ecc3`, r882 comparator file SHA-256 `0084479aa0703667a0f3ccfe2beda60741c204795ec57b7c08542401ec6cdb64`, train roster 1,219 contexts/205 families, DL-Val roster 91 contexts/19 disjoint families. Holdout and outer payloads stay unopened.

## Update unit and training protocol

One H53b **round** is one live-loop adapter update unit. Each unit has `k=2` AdamW optimizer steps, `lr=1e-4`, `weight_decay=1e-4`; batch size is 1,219 fixed-cohort presentations per step, slate size 192, and sampling uses the live `batch_seed(seed, round, step)` SHA-256 stream, Python `random.Random(step_seed).choices` over the round-rotated training roster, and the same `training_indices` slate sampler. The candidate loss is the fused live-loop objective: gated target value 2.0; row Smooth-L1 plus 0.1 × KL over negative-fitness softmax at temperature 0.2; mean across batch. Optimizer state persists within a cell and resets between cells. Pin the exact implementation hashes in the cell receipts.

The offline cohort is fixed at 1,219 contexts. The live r882 loop accumulated data over time; therefore rounds-to-reach either reference are descriptive and **not round-for-round comparable**. Record the exact k/LR/sampler and objective source hashes in each receipt.

## Horizon, timing pilot, and resource rules

Each cell runs up to 300 rounds or 20 minutes, whichever occurs first. First run one seed12/scale1.0 timing pilot with a 10-minute hard cap solely to measure exact-unit throughput and evaluation overhead. Stop before other cells if six-cell projected wall time exceeds 2.5 hours. The pilot cap is not an early efficacy stop. During a cell, score DL-Val only every 25 rounds and at the terminal round; these intermediate scores are diagnostic and cannot select a checkpoint, stop a cell early, or choose an arm. Capture training/evaluation time separately.

GPU-first: acquire `~/jope_plan/qualification.lock` exclusively for each CUDA cell and release it between cells; recheck current GPU availability before each cell. If the lock remains busy for over 20 minutes, CPU fallback is allowed only with `nice -n 10`, OMP threads <=8, host CPU <=85%, MemAvailable >=20 GiB, RSS <6 GiB, and the full swap condition recorded. Per owner ruling, one dedicated lane is authorized despite the general eight-lane ceiling; no other owner's lane may be touched. Use only an immutable copy of the offline flywheel packet.

## Primary outcome and reporting

Primary endpoint: final-round DL-Val **family utility**, per seed and scale. Report per-scale seed median and seed-bootstrap 95% interval for final utility, Elo, and gaps; ties fail. Also report the prospective family mean-LCB/no-harm diagnostic: one-sided family bootstrap LCB95 > 0.001 and median family gain >=0. It is a diagnostic on this fixed familiar cohort, not an independent admission claim. No checkpoint/arm selection on the 25-round curves.

Compare final utility separately to (a) harness-scored r882 `1.1536152769912449`; (b) live-loop target `1.1377942`, `evaluation_identity=e79a7715...`; and (c) the post-r882 live-candidate distribution from the authorized read-only watcher summary: n=543, mean 1.13393, SD 0.00212, p50 1.13443, p99/max 1.13771, none >=1.1377942. The live target is a maximum over about 650 DL-Val reads and r882 tied r859 at `1.1377941898164525` under `accept_tie`; these are distinct identities. Report rounds-to-cross the harness value and live target with the fixed-vs-growing-data caveat.

## H53 correction carried forward

H53 is **pilot only: inconclusive at three updates**. Final-round outcome is primary; a max over rounds is selection on validation even though all six H53 best values equaled their final values. Include H53 per-seed final utility/Elo and per-context W/T/L aggregates (91 contexts each). For 0.1x, seeds 12/13/14 W/T/L are 27/48/16, 28/45/18, and 27/46/18; corresponding Elo gains are +40.41, +36.71, +33.01 while final family utility gaps to harness r882 are -0.044348, -0.045300, -0.042826. Positive Elo does not imply higher mean utility. Family-grouped top-five most-negative 0.1x dataset mean contributions are `fe93a534092f0948` -0.6240092534, `3c81d7475323daff` -0.4885376928, `c1c828dee4613b77` -0.0521676814, `90ae3fa51293ea1a` -0.0118055284, `7f3420ce132b63aa` -0.0050667200; these are not per-context raw effects.

## Closeout

The wave folder will contain `SUMMARY.md`, `metrics.json` (<50 KB), and per-cell receipts outside the source tree. Add one index row and open one small PR with only summary/compact metrics and the minimal runner/protocol needed to reproduce. Run targeted local checks; ignore GitHub CI/protection per owner policy. After merge/closure preserve and verify source/evidence, remove the owned lane using `git worktree remove`, inspect `target/` before any cleanup, and report `df -h /` in comms.


## Amendment before any cell — source and sampler pins

The fully prepared cell runner/coordinator/audit and exact source dependencies are pinned in `code-manifest.sha256` before cell 1. In particular, `run_fused_closed_loop.py` defines the live objective and limits full-cohort updates to at most two optimizer steps; `dev_incumbent.py` defines `batch_seed`; `automatic_promotion.py` defines `training_indices`. H53b fixes k=2, LR=1e-4, B=1219, slate=192, and uses these pinned routines. This is an offline fixed-roster application of the live sampler, not a claim that live data grew at the same rate.


## Dated pre-training correction (2026-10-30)

Pilot attempt 01 acquired CUDA but reached its 10-minute wall cap while preparing the model/packet; it completed **zero optimizer updates** and produced no score receipt. Its empty per-cell log and `wave.json` are preserved under `~/jope-data/h53b/runs/`. The child was stopped by its owner after observing the coordinator's pilot timeout bug. Before any training update, the coordinator was amended so the pilot's 10-minute cap includes CPU/memory checks, lock wait, setup, and execution; each later cell's 20-minute cap likewise includes setup and lock wait. The original wave deadline remains fixed at 2026-10-30T21:18:50Z (150 minutes from the first coordinator start); it is not reset by this retry. Phase timing is printed as setup proceeds. The OpenBLAS thread limit is eight (within the OMP<=8 rule); the process remains nice 10 from the coordinator (the child inherits a lower priority). Current source hashes are pinned in `code-manifest.sha256` and this amendment is committed before any retry.


## Dated pre-training amendment after timing attempts (2026-10-30)

Attempt 02 reached the 10-minute cap with model/data setup complete but no completed two-step update unit and no terminal receipt; its log and wave receipt are preserved in `~/jope-data/h53b/attempt-02/`. No state is resumed or transferred. The two timing attempts establish that a full update unit cannot be timed within the 10-minute pilot cap; the remaining authorized horizon is therefore the preregistered per-cell 20-minute wall cap. Run each seed/scale from fresh initialization, preserving one fixed 2.5-hour deadline anchored at the first coordinator start (2026-10-30T18:48:50Z), not reset by retries. GPU lock waiting may take up to 20 minutes before authorized CPU fallback; that wait is charged to the fixed overall deadline. Each actual cell receives at most 20 minutes after lock/fallback acquisition and cannot extend the overall deadline. If a cell does not produce a complete update and terminal summary within that cell budget, stop the wave without interpreting partial in-memory state.


## Overseer pilot-fix amendment (2026-10-30T19:15Z; infra-only)

Attempts 01 and 02 produced no outcome data; attempt 03 was stopped during setup after the 19:15Z nudge identified the same deadline race. All three logs/wave receipts are retained outside the checkout; no adapter state transfers. The runner now receives `wall_cap - 90s`, reserving at least 90 seconds for final DL-Val, Elo, and atomic summary; the coordinator hard-kills at the actual 10/20-minute cap. It emits flushed per-step slate-build/forward-backward timings and per-round train/cumulative timings without utility, so there is no intermediate selection signal. On cap inside an update unit, any partial-unit parameter changes are discarded and a partial receipt records the last complete round (or no endpoint if none).

Pure implementation optimization: `training_indices` sorts each immutable context fitness vector on every presentation. H53b caches each context's `argsort` once, while retaining the identical `np.random.choice` calls/order and return construction. A fixed-seed sampler-equivalence preflight compares cached and canonical indices over 32 contexts before the first update. The NumPy stream is seeded once per live round and advances across its k steps, exactly as the pinned live loop does. No k/LR/batch/slate/objective/cohort/cap changed. All source hashes are refreshed in `code-manifest.sha256` and committed before the next pilot.


## Pilot outcome and remaining-cell handoff (2026-10-30T19:26Z)

The corrected time-only pilot is also the preregistered cell 1 (seed12, scale1.0); it completed 93 full rounds/186 AdamW steps in 516.1s coordinator wall and discarded one partial next-round step. No outcome-based stopping or arm selection occurred. It reached the 10-minute pilot cap; its round-93 final score is retained as that cell's terminal score, with its shorter budget disclosed. Measured average training time was about 4.9s/round; 300 rounds therefore exceed 20 minutes, so the remaining five cells use their fixed 20-minute wall caps and report actual completed rounds. A separate continuation coordinator copies and hash-verifies cell1's compact summary; cells2–6 keep the original order. The original 21:18:50Z wave end remains fixed; if it arrives before all cells finish, remaining arms are reported incomplete with no reordering or cap extension.


## GPU occupancy fallback rule clarification (2026-10-30T19:28Z)

Before waiting on `qualification.lock`, query current free GPU memory. If less than 20 GiB is free, treat the GPU as busy and use the preregistered guarded CPU fallback immediately without waiting on or disturbing the lock holder. Otherwise acquire the lock, then recheck free memory; if occupancy rises below 20 GiB, release the lock and fall back. If the GPU has headroom but its lock remains held, retain the original 20-minute busy-lock wait before CPU fallback. CPU guard receipt includes `SwapTotal`/`SwapFree`; RSS remains capped at 6 GiB. This changes coordinator resource routing only, not any arm/update/metric protocol.


The coordinator records the child process's observed Linux nice value (the host parent is already nice 15, and the child adds 10, so it may appear as NI 19). It continues to enforce host CPU <=85%, OMP/MKL <=8, MemAvailable >=20 GiB, and RSS <6 GiB; CPU fallback also records swap occupancy.


## Owner GO amendment — equal-budget six-cell wave (2026-10-30T19:30Z)

This amendment supersedes the earlier pilot-as-cell and prior fixed-deadline text. Citing original prereg commits `8940054927087db70e9edb5c1e3913f77112540d` (author date 2026-10-30T18:47:08Z) and `a41d61a9140e166e2fd945d8705508f88ec0c09e` (author date 2026-10-30T18:48:41Z): the 10-minute seed12/scale1.0 run is **timing-only** and excluded from H53b wave outcomes. Attempts 01–03 and the interrupted continuation cell2 were infrastructure-only, with no completed endpoint. Run all six wave cells fresh at equal per-cell budgets, in order: seeds12/13/14 at scale1.0, then seeds12/13/14 at scale0.1; each up to 300 rounds or 20 minutes. Cell1 is a fresh seed12/scale1.0 initialization; do not reuse or transfer the timing pilot's adapter or optimizer state. Use the full runner cap-minus-90s terminalization margin and coordinator hard cap at20m. Do not reorder or extend a cell.

The new overall deadline is anchored to the **new six-cell coordinator start plus 150 minutes**. The coordinator records its actual UTC start and deadline in `wave.json`; no prior failed-attempt time is charged to this re-anchored budget. The owner-provided approximate finish `21:35Z` is a planning estimate; the literal +150m deadline is authoritative (a 19:30Z start gives 22:00Z). Projected six-cell compute at the measured 20m caps is about 121m, under the 150m limit.

Final-round family utility remains primary. Round-25 reads are diagnostic only; the separate timing pilot's curve (rounds0/25/50/75/93: 1.091686/1.092344/1.132931/1.106626/1.136392) demonstrates substantial read-to-read swing, so never select a best round. Ties fail. For every cell, preserve final family mean gain and family LCB alongside Elo and per-context W/T/L; a positive Elo with a negative mean must be reported together with the families driving the loss.

Separate timing-pilot diagnostic (not part of H53b seed bootstrap): final mean family gain vs harness r882 -0.0172231, mean-LCB95 -0.0836601, median gain 0.000000; Elo +29.33, W/T/L 27/45/19. Worst family gains are `fam-9f67f1449b77152cbf20a6c82926f90496dacd7f05a222376edf0fdfb45f667f` -0.6236002, `fam-7d4e2dc9f16cc0145caf3ff803564a61818542d6ab30dfe68270c4fe24b542e8` -0.0380964, `fam-598bbb77441667f9a37a8f024d6675afd597dec08fef83af86f36d67ee2f9564` -0.0307308, `fam-94f2455bd049dc8fd80a311445ae104f4d5bff163877f0f7ffa12758465eac1f` -0.0159064, and `fam-280b1a133e098dea8ae2dc71cb0c19fa2560533c1e9c3fcbdb64ef1d56007103` -0.0156955. These are family utility gains, not per-context scores.


## Infrastructure-only attempt and owner GO chronology

After the timing-only pilot completed, a continuation seed13 CPU process started at 19:27:57Z when the GPU was busy. At 19:28:33Z, it was stopped before any optimizer update after the owner GO ruling required equal fresh budgets and a fresh seed12/scale1.0 cell first. Its setup-only log and `wave.json` are preserved under `~/jope-data/h53b/attempt-03/remaining/`; it is not an outcome. The final source amendment is committed before the next training cell. The new coordinator anchors its 150-minute deadline to its own UTC start and stores start/deadline in `wave.json`.
