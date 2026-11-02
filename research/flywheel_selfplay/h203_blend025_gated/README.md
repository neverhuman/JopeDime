# H203 — BLEND_025 gated by an outcome-free harm signal, flywheel native protocol, 0 updates

**Status: INCOMPLETE, with no verdict.** Both preregistered attempts, the run and its single resume, were stopped by
the monitor's host-load guard. The load came from other jobs on xbabe2. **No outcome was computed or seen.**
H203 is closed as the [PREREG](PREREG.md) requires ("a second stop closes H203 INCOMPLETE with no verdict").

## What was asked

H199's BLEND_025 lifts mean native utility over r882 (+0.0434, Elo 1514.0) but harms 24.9% of families. H203 asked
whether an outcome-free, cost-only signal could decide per context when to trust the blend's switch. The primary
signal is `zgap`, how strongly r882 disfavours the blend's pick, with a FIT-only threshold. When the signal says
"risky", the arm falls back to r882. The aim was to keep the native gain with harm ≤ 10%. The primary arm was
GATED_025 (m = 1): CANDIDATE iff LCB95 > 0, p < 0.05 and harm ≤ 10%.

## What happened

| | Run | Resume (the single allowed) |
|---|---|---|
| Start (UTC) | 2026-11-02 02:24 | 2026-11-02 05:37 |
| Stopped (UTC) | 02:43:37 | 06:10:17 |
| Guard | load1 120.9 > 108.8 (85% of 128 cores) | load1 154.6 > 108.8 |
| Fits done (of 45) | 4 | 15 in total (4 reused after hash checks, 11 new) |
| DL-Val R882 native 91/91 | bit-exact PASS | bit-exact PASS |

- **Only seed 203101's 15 fits finished.** No stage-1 lock (calibration and τ) was written, and no A, B or DL-Val
  candidate was scored. The fits are FIT-only HistGBM models.
- **H203's own footprint stayed inside its limits:** peak 3.94 cores, RSS 3.8 GB, root 92 MB, 0 optimizer updates,
  no GPU, and no sealed or holdout reads.
- **The load spikes were other jobs:** many `jankurai` processes at the first stop and about 35 `codex` processes at
  the second. Before the resume, rd_opus waited about 3 h for load to settle below 50; it rose again within
  35 min.
- **Disk** peaked at 93.754% used, against the 93.8% guard.

## Deviation from the prereg (host only)

The prereg admits xbabe1 only. The owner instructed a launch on xbabe2 (Jepson, 2026-11-01 7:18 PM MT, relayed in
`~/nudges/n1918_rd_opus_h203.md`). Run commit `f41b2f8d` therefore admits xbabe2 in [runner.py](runner.py) and
[monitor.py](monitor.py), under the lanes_t1615 disk guard (user-visible used < 93.8%, free ≥ 40 GB).

The launcher verified that the following are byte-identical to the gov-reviewed prereg head `c2d67cb4` (merged as
`272877f4` via #1709):
- PREREG.md, analysis.py, fold_manifest.json and manifest.json;
- the H124 and H139 runners;
- the DOPE freeze guard.

The deviation commit is self-reviewed. Its tests are in [test_h203.py](test_h203.py) (25 data-free tests). The
pipeline is CPU-only as preregistered, so the xbabe2 RTX 3090 was neither used nor claimed.

One incident during the run: between the two attempts, another session checked out `rd/h205-prereg` inside the
unlocked run lane. The first resume try at 05:36:52Z therefore loaded the old xbabe1-only monitor. It refused on the
host check before any claim or output, so the resume was not consumed. The actual resume ran from a new locked lane
at the same commit.

## Custody

The run root, logs, launch record and claims are uploaded create-only to
`s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/H203/run-20261102T0224Z/`. Object versions
and SHA-256 values are in [custody.json](custody.json); [metrics.json](metrics.json) holds the stop record.

## Next step

H203's question is still open. A retry needs a **new H number** with fresh seeds, because this one's single resume is
spent. It also needs a host whose load stays under the guard for about 2.5 h: xbabe1, as originally preregistered, or
a CPU reservation on xbabe2. Whether to retry is the driver's scheduling decision.
