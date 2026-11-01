# H197 closeout: per-context gradient clipping

**Status: INCOMPLETE (two authorized attempts failed before science).** H197 produced no scientific result and cannot be labeled `ADVANCE` or `NO_ADVANCE`.

## Hypothesis and registered setup

The preregistered hypothesis was that per-context adapter-gradient clipping (PCGC40) could reduce the influence of a small number of high-gradient, low-loss contexts and improve held-family utility over the exact r882 reference. The registered comparison used three independent TRAIN-family fold seeds, five cyclic A/B/FIT folds per seed, four 40-update arms (CTRL40, PCGC40, GCLIP40, PCGC_SHUF40), and fixed step 40 with no B-based step selection. Only the adapter was eligible to train; the native loss and kernel were frozen. See [PREREG.md](PREREG.md) and [fold_manifest.json](fold_manifest.json).

## Attempts and outcome

The preregistration was squash-merged in [PR #1616](https://github.com/neverhuman/JopeDime/pull/1616) at `33aa7fd2544cb4395d1c06f3e083be87a7502960`.

- **Attempt 1:** The monitor's three 5-second prelaunch samples passed. The worker exited with code 1 after 1.77 seconds because the repository-root calculation pointed outside the checkout and Python could not import `h182_base`. No model was loaded and no optimizer update occurred. Its fix was merged in [closeout PR #1617](https://github.com/neverhuman/JopeDime/pull/1617) at `ca3c938e1`.
- **Attempt 2:** ENG authorized a fresh attempt from current main. Its three 5-second prelaunch samples passed (disk 92.79%, free disk over 566 GB, available memory over 82 GiB, and CPU/load within limits). The worker exited with code 1 after 2.25 seconds when the reused H182 verifier checked H197's `PREREG.md` pin relative to the H189 helper directory. The verifier had hashed pinned files, but did not deserialize the checkpoint or load/train a model, score the reference, read S3 outcome panels, or perform optimizer updates. The monitor reported no resource violation.

Both attempt claims, launch bindings, stdout traces, resource samples, and summaries are retained in `/home/ubuntu/jope_plan/ns/receipts/H197_attempt{1,2}*` and `/home/ubuntu/jope_plan/ns/results/H197_attempt{1,2}*`. Attempt 2's supervisor was a hash-recorded copy of the merged monitor with only its checkout path and attempt claim filename changed.

The H197 local-source verifier now resolves pins from the H197 directory, while the reused parent verifier checks external inputs, assets, parents, and package versions. This closeout fix was not used to start a third attempt. The two authorized attempts are consumed; no further run is authorized here.

## Metrics and decision

There are no per-seed gains, median/interval, bootstrap p-value, harm rate, pick counts, clip statistics, or Elo results. These quantities are **not computed**, rather than zero. The registered decision is **not evaluable** because no reference replay or science scoring occurred. No adoption or serving action is supported.

## Verification

Data-free H197 fixtures, Ruff, the five `jope-records` checks, Python compilation, and `runner.py --help` passed for the closeout source. The source-pin helper has a fixture for correct-root resolution, mismatch rejection, and path traversal refusal. Large outcome files and S3 custody records were never created.
