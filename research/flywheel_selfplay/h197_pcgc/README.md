# H197 closeout: per-context gradient clipping

**Status: INCOMPLETE (runner import failure before science).** H197 produced no scientific result and cannot be labeled `ADVANCE` or `NO_ADVANCE`.

## Hypothesis and registered setup

The preregistered hypothesis was that per-context adapter-gradient clipping (PCGC40) could reduce the influence of a small number of high-gradient, low-loss contexts and improve held-family utility over the exact r882 reference. The registered comparison used 3 independent TRAIN-family fold seeds, five cyclic A/B/FIT folds per seed, four 40-update arms (CTRL40, PCGC40, GCLIP40, PCGC_SHUF40), and fixed step 40 with no B-based step selection. Only the adapter was eligible to train; the native loss and kernel were frozen. See [PREREG.md](PREREG.md) and [fold_manifest.json](fold_manifest.json).

## Attempt and outcome

The preregistration was squash-merged in [PR #1616](https://github.com/neverhuman/JopeDime/pull/1616) at `33aa7fd2544cb4395d1c06f3e083be87a7502960`. One guarded attempt was claimed on xbabe2 on 2026-11-01. The three 5-second prelaunch samples passed: disk use was below 93.8%, free disk exceeded 40 GB, available memory exceeded 16 GiB, CPU and load were under their limits, and the monitor started at nice 19.

The worker exited with code 1 after 1.77 seconds, before creating the output directory, loading the checkpoint or corpus, evaluating the reference, or performing an optimizer update. The captured exception was `ModuleNotFoundError: No module named 'h182_base'`: the repository-root calculation in `runner.py` walked one parent too far, so its read-only H189 helper import pointed outside the checkout. The monitor reported no resource violation. No model or outcome data was read, no B lock or science panel was produced, and no metrics can be estimated.

The one-attempt limit is consumed. The repository-root calculation is corrected in the closeout source so a future separately authorized wave can import the pinned helper; H197 was not rerun. The corrected runner's `--help` import check passes, and all data-free tests and record gates pass. This engineering correction does not change H197's attempt status or supply science evidence.

## Metrics and decision

There are no per-seed gains, median/interval, bootstrap p-value, harm rate, pick counts, clip statistics, or Elo results. These quantities are **not computed**, rather than zero. The registered decision is **not evaluable** because no reference replay or science scoring occurred. No adoption or serving action is supported.

## Evidence

The attempt claim, launch binding, stdout traceback, resource samples, and monitor summary are in `/home/ubuntu/jope_plan/ns/receipts/H197_attempt1.claim`, `/home/ubuntu/jope_plan/ns/receipts/H197_attempt1_launch.json`, and `/home/ubuntu/jope_plan/ns/results/H197_attempt1.*`. The launch used the merged source commit and exact tree recorded in the launch receipt. Large outcomes were not created, so no S3 outcome custody exists.
