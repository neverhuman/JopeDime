# H296: SWA / Polyak averaging of offline TRAIN-only adapter trajectories

**Status: complete. Verdict: NOT_ADVANCED.** There is no serving or live change.

**The result.** Weight averaging did **not** escape the plateau.
- **Every trained candidate is below r3175:** SWA_CYCLE, its paired controls and EMA_POLYAK all fall short of
  1.1389572403748425.
  - **SWA_CYCLE:** −0.0064, −0.0089 and −0.0088 by seed; median of seeds −0.0088, 95% CI [−0.0193, 0.0].
  - **SWA_CYCLE against its paired control:** better on seeds 29601 (+0.0020) and 29602 (+0.0029); worse on 29603
    (−0.0004).
- **SOUP_RET (zero-train r3100 + r3175 average) comes closest:** 1.13837 (−0.0006), changing only 2 of 91 contexts
  (W/T/L 1/89/1).
- **The functional plateau persists:** after 16 updates, the paired controls of seeds 29601 and 29603 land on
  **exactly the same dev91 mean and W/T/L**, and EMA_POLYAK-29603 matches them too.

## Run record

- **Prereg:** published on `origin/v1/next` as `53a39590` at 03:02:36Z, before any H296 arm was trained or scored
  (same-agent review receipt sha256 `11dfb67c…`).
- **Trajectories:** xbabe1, 03:03–04:42Z: 48 updates, 0 pauses, all 3 workers exited 0.
- **Scoring:** dev91 scoring ran 04:42–04:46Z, once per candidate, after every candidate had been frozen and hashed.
- **Integrity:** DOPE `c51a56d0` and every frozen tensor were verified after every update. Only the 29,211 adapter
  parameters changed. Only copies were used. The live run directory, the writer and the observer were untouched.

## Hypothesis

Weight averaging over a longer offline trajectory escapes the discrete argmin plateau seen in t1820/t1845, where
2-update copy trials change only 1–3 of 91 choices.

## Setup

- **Harness.** t1845's `experiment.py`/`trials.py` functions in [h296.py](h296.py), with:
  - its own copy directory (`xbabe1:~/jope-data/H296-opus`, 475 MB, every file rehashed);
  - helpers pinned from t1845's source commit `e5c5b852` ([pinned/](pinned));
  - t1845's pinned scorer extension (`93acd308`).
- **Regressions, both exact:**
  - the r3175 copy replays **1.1389572403748425** with 91/91 choices;
  - t1845's control seed 184501 reproduces **1.1354387884248622** with identical selections.
- **Trajectories** (seeds 29601–29603; logical round = seed):
  - **Start:** t1845's control start (r3175 plus its Adam moments, then `warm_restart_adapter`). The restart LRs were
    2.55e-5, 8.01e-5 and 3.32e-5.
  - **Updates:** **16 TRAIN-only updates**, each a full family-blocked pass of 8,190 presentations over the 1,219 TRAIN
    contexts with slate 192, SmoothL1 + 0.1 KL, microbatch 32, clip 1 and AdamW.
  - **Schedule:** **4 cosine-restart cycles of 4 updates**, from lr_max down to 0.1 × lr_max.
- **Candidates.**
  - **SWA_CYCLE:** the mean of the 4 cycle-end iterates.
  - **CONTROL:** the 16th iterate.
  - **EMA_POLYAK:** decay 0.8 over the 16 iterates.
  - **SOUP_RET:** the r3100 + r3175 adapter average, computed once.
- **Rule.** [analysis.py](analysis.py), strict; ties never win. SWA_CYCLE must pass on 3/3 seeds:
  - dev91 > 1.1389572403748425 + 0.001;
  - above its paired control;
  - family-grouped (19 families) bootstrap LCB95 > 0 (10,000 draws, seed 296150);
  - top 4 contexts < 80% of the positive gain.

## Outcome (dev91, 91 contexts / 19 families; paired against the r3175 copy)

| Candidate | Dev91 mean | Lift vs r3175 | Family-bootstrap 95% CI | W/T/L | Changed | Elo (r3175 = 1500) [95%] |
|---|---:|---:|---|---|---:|---|
| **SWA_CYCLE-29601** | 1.132522 | −0.00644 | [−0.0163, 0.0] | 0/88/3 | 3 | 509 [339, 1500] |
| **SWA_CYCLE-29602** | 1.130101 | −0.00886 | [−0.0195, −0.0004] | 2/83/6 | 8 | 1310 [420, 1430] |
| **SWA_CYCLE-29603** | 1.130129 | −0.00883 | [−0.0195, 0.0] | 0/87/4 | 4 | 459 [318, 1500] |
| CONTROL-29601 | 1.130496 | −0.00846 | [−0.0192, 0.0] | 1/86/4 | 5 | 1260 [339, 1740] |
| CONTROL-29602 | 1.127233 | −0.01172 | [−0.0259, −0.0006] | 3/79/9 | 12 | 1310 [388, 1500] |
| CONTROL-29603 | 1.130496 | −0.00846 | [−0.0192, 0.0] | 1/86/4 | 5 | 1260 [339, 1740] |
| EMA_POLYAK-29601 | 1.130129 | −0.00883 | [−0.0195, 0.0] | 0/87/4 | 4 | 459 [318, 1500] |
| EMA_POLYAK-29602 | 1.127301 | −0.01166 | [−0.0257, −0.0006] | 3/80/8 | 11 | 1330 [420, 1500] |
| EMA_POLYAK-29603 | 1.130496 | −0.00846 | [−0.0192, 0.0] | 1/86/4 | 5 | 1260 [339, 1740] |
| SOUP_RET | 1.138367 | −0.00059 | [−0.0019, 0.0] | 1/89/1 | 2 | 1500 [1500, 1500] |

- **Elo note:** an arm with no wins has its Elo set by the BT prior (0.01). The intervals are wide because only a few
  contexts change.

**The primary rule per seed:**

| Seed | Above stored + 0.001 | Above paired control | LCB95 > 0 | Not concentrated |
|---|---|---|---|---|
| 29601 | no | **yes** | no | no |
| 29602 | no | **yes** | no | no |
| 29603 | no | no | no | no |

**Trajectory losses** (first → 16th update): 3.630 → 3.684 (29601), 3.662 → 3.635 (29602), 3.671 → 3.645 (29603).
The training loss barely moves, and dev91 selections move in a few contexts, all net-negative.

**Resources:** peak process RSS 4.07 GB (workers share pages copy-on-write), minimum MemAvailable 14.7 GB, maximum host
CPU 82.9%, and no pauses.

## Verdict

**NOT_ADVANCED.**
- **No averaging scheme lifted dev91 above the retained r3175 (1.1389572)**, on any seed.
- **Averaging is mildly better than the final iterate.** SWA beats its paired control on 2/3 seeds, consistent with
  averaging damping the drift. But the trajectories themselves move dev91 selections in the wrong direction.
- **The plateau is functional.** Distinct seeds' controls collapse onto one selection vector after 16 updates.
- **SOUP_RET** (no training) stays nearly at r3175 and changes only 2 contexts.

## Next step

- **Stop investing in weight averaging.** Averaging and longer trajectories under the same SmoothL1 + KL surrogate do
  not move the argmin in a helpful direction. The surrogate–selection alignment is the likelier bottleneck: the
  training loss is nearly flat while selections drift negatively.
- **The selection layer is the productive path.** The frozen rerankers (H294's ENS_VOTE2, CONFIRMED_FRESH) improve the
  pick without moving the adapter. Adapter-training waves should change the objective (for example, a ranking or
  argmin-aware loss) rather than the optimiser trajectory.

## Deviations

- **Cores.** The assigned cores 56–59 are outside this session's xbabe1 cpuset, so the idle cores 20, 21, 28 and 29
  were used.
- **Workers.** xbabe1's system torch is now a generic-BLAS build, at about 6 minutes per update. The 3 seeds therefore
  ran as forked 1-thread workers (≤ 4 threads), all declared in the prereg. The regressions ran with 4 threads.
- **Regression attempt 0.** The first regression attempt failed at import: the `~/.local` jope no longer provides
  `FullFlywheelReporter`. It was re-run with t1845's pinned extension loaded explicitly. No result was read in the
  failed attempt.
- **Slates.** t1845's slate hash was not bit-reproduced (scorer-build RNG drift), but its terminal control result was
  reproduced exactly.

## Files and custody

- **In this directory:**
  - [metrics.json](metrics.json): 8 KB, holding every candidate's statistics, the per-seed rule, the trajectories
    (lr, loss, grad norm), the regressions, resources and pointers;
  - [PREREG.md](PREREG.md), [h296.py](h296.py), [pinned/](pinned), [analysis.py](analysis.py),
    [test_h296.py](test_h296.py) and [manifest.json](manifest.json).
- **S3:** `s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/H296/attempt-001/` holds the 10 candidate
  checkpoints (sha-verified against the xbabe1 copies), the per-candidate JSONs, the events, resource samples,
  regression and baseline records, the analysis, the sources and `s3_manifest.json` (VersionId `42SwgGrS…`).
- **Copy directory:** deleted at close after the archive was verified.
