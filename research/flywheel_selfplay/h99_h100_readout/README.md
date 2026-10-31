# H99/H100 terminal Elo publisher

Owner grant: `unstick_t0955_sol_v1_dsc08.md`. This CPU-only publisher turns the
workers' completed FIFO receipts and native statistical reports into JSON and
Markdown verdicts. A watcher polls every 30 seconds, so a complete native report
can publish within one poll. It opens the budget first and keeps incomplete or
in-progress panels UNKNOWN without opening their outcome files.

```sh
PYTHONDONTWRITEBYTECODE=1 nice -n 10 python3 \
  research/flywheel_selfplay/h99_h100_readout/readout.py \
  --wave H99 --watch --out-dir /home/ubuntu/jope_plan/evidence/H99/readout-RUN
```

Use `--wave H100` for H100. Default inputs are the corresponding
`~/jope-data/h99/h99-wave` and `~/jope-data/h100/h100-wave` roots; `--root` accepts
a retained receipt copy. Output must be outside the input root. Watch time is
bounded (default six hours, configurable up to 24), polls are 1–60 seconds, and
load above 85 delays receipt reads. Exit codes: 0 complete, 2 invalid, 3 pending
or timeout. Each publication uses fresh output names; exact replay is idempotent
and differing prior reports are retained. Watch logs expose readiness only.

H99 requires seven unique completed cells, including the NULL diagnostic, plus
the once-only Fold-B scoring action, `outcomes/fold_b.json` and
`outcomes/admission.json`. H100 requires twelve paired cells, the original timing
action and `metrics.json`. Every completed H100 cell and timing receipt is checked
against its FIFO SHA; the aggregate must contain the exact ledger, except that
the final owner re-arm's `state` may advance from `FIFO_REARM` to `COMPLETE`
after aggregate publication. Every other ledger field, including all attempts,
charges, receipt hashes and re-arm authority metadata, must remain identical.
This check never modifies either source receipt. H99 checks
the FIFO/scored-adapter digest chain, original source/fold/DOPE identities and
six once-scored candidates. JSON paths stay inside the input root and reject
symlinks, traversal and protected/live paths. All wall charges, including retained
pre-scoring failures, count against the original 9,000/2,700 second caps. Duplicate
successes, unknown scoring failures, mixed identities, nonfinite values, changed
snapshots and contradictory native verdicts yield UNKNOWN/INVALID.

The common output format preserves each wave's frozen scientific contract:

| Wave | Primary / strict LCB margin | Nonnegative point | Harm vs r882 | Native Elo ties |
| --- | --- | --- | --- | --- |
| H99 | r882 / >0.001, two-arm Holm | family median over paired seeds | <=10% of family seed-medians below -0.01 | abs(gain)<=1e-4 |
| H100 | CONTROL / >0.001546, three-arm Holm | median seed mean-family gain | <=10% of family seed-means below -0.01 | gain==0 |

Elo is anchored at r882=1500 with per-seed and median paired family-bootstrap
95% intervals. Family wins/ties/losses are separate. Native finite clipping and
`Infinity`/`-Infinity` strings are preserved; waves are not pooled. CIs are
published directly from the [H99 scorer](../h99_esh_harm/build_metrics.py) and
[H100 summary](../h100_asam_confirm/runner.py), with their original RNGs and
10,000 draws. The publisher checks primary points, harm and Holm/gate consistency;
it does not introduce a new statistical test or independently recompute CIs.
Every consumed receipt SHA and the publisher source SHA accompany the result.
See the [H99 preregistration](../h99_esh_harm/PREREG.md) and
[H100 preregistration](../h100_asam_confirm/PREREG.md).

Validation uses synthetic producer receipts, including direct compatibility
replays through both existing native scorers. Run
`nice -n 10 python3 -m pytest -q -p no:cacheprovider
research/flywheel_selfplay/h99_h100_readout/test_readout.py` plus Ruff lint/format
and syntax checks. The publisher uses only the Python standard library. Its
operations are receipt reads and separate report writes; training, model/data
loading, CUDA, FIFO/qualification-lock ownership and serving promotion belong
to the original workers and their separate authorization.
