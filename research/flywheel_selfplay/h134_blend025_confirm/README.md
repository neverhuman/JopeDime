# H134 — prospective confirmation of H129 BLEND_025 vs r882

**Status: INCOMPLETE, closed with no verdict.** The host disk filled twice, and both times the preregistered hard
disk guard (`/` ≥ 93.8%) stopped the run before all 45 fits existed:
- **First attempt, 22:45Z.** 39/45 fits; `/` at 93.97%.
- **The single authorized resume (Amendment A-resume, #1226), 23:30Z.** 42/45 fits; `/` at 93.81%.

Amendment A-resume says any further guard abort closes H134 with no retries. **No outcome, readout, verdict or Elo
was ever computed.** BLEND_025 is therefore **neither confirmed nor refuted** by H134, and H129's CANDIDATE stands
alone, with H135 still pending.

## What happened

| Attempt | Source | Fits done | Stop |
|---|---|---:|---|
| `run-20261031T2036Z` | prereg `7e4537dbf` / `11fba6d03` | 39/45 | disk 93.97% before `seed143-exclude12` |
| A-resume (once) | `90b2db52f`, after #1226 merged | 42/45 (39 sha-verified + 3 new) | disk 93.81% before `seed143-exclude23` |

- **What filled the disk.** Other activity on xbabe2 filled it. H134's own scratch stayed under 100 MB. The overseer
  freed about 85 GB before the resume, and use climbed back across the line within 10 minutes.
- **The guard held.** The 93.8% line was never relaxed. The run read zero sealed, holdout or meta-test payloads and
  decoded zero DL-Val cache rows. DOPE was `c51a56d0` before and after the first attempt's frozen pass. No GPU or
  qualification lock was used, and the live flywheel was never touched.

## Next step (owner)

H134 cannot be resumed under its own amendment. If the confirmation is still wanted, it needs a new preregistered
attempt (a new H number, or an explicit owner amendment), launched only when xbabe2 has sustained headroom, for
example `/` below 90%. Another option is a host with free disk. Until then, the adoption question for BLEND_025
rests on H129 (CANDIDATE, development evidence on DL-Train) and H135 (flywheel val).

## Custody

- **S3.** `s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/H134/`:
  - `opus-h134-20261031a-partial/` (85 objects);
  - `opus-h134-20261031a-final/` (9 objects).

  Both are create-only with a version-pinned readback.
- **Rescue copies.** `/home/ubuntu/jope-rescue-20260921/h134-rd_opus-20261031-{partial,final}/`.
- **[metrics.json](metrics.json).** It records the status and identities only; it holds no outcome values.

## Preregistration (unchanged)

See [PREREG.md](PREREG.md).

- **Primary.** One arm, BLEND_025 = r882 + 0.25·(GBM − r882), with λ fixed.
- **Fresh draw.** New family fold 134141, sampling seed 134142 and model seeds 141–143.
- **Gates.** All are required:
  - median seed gain > +0.001;
  - family-bootstrap LCB95 > 0;
  - all three seeds > 0;
  - harm ≤ 10%;
  - trimmed mean (dropping the 10 largest-|gain| families) > 0.
- **Elo.** Reported against r882 = 1500.
- **Code.** [runner.py](runner.py) imports the merged H129 runner (sha `e34d4a3f…`) and rebinds only the seeds and
  fold. [test_h134.py](test_h134.py) has 8 synthetic controls.
- **Caveat.** This re-splits the same DL-Train families, so it tests robustness, not new data.

## Amendment A-resume

The preregistered disk guard aborted the first attempt after 39/45 fits, before any outcome existed. Owner decision
(a) in `rd_opus_t1715` authorizes exactly one sha-verified resume of the 6 missing fits.
[AMENDMENT_A_RESUME.md](AMENDMENT_A_RESUME.md) records it, and [resume.py](resume.py) is the launcher. The protocol
and gates are unchanged.
