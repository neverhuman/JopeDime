# H420 — language exposure (in progress)

**This is a status log, not a closeout.** No H420 deploy on a scored panel has started; no AUDIT value has been read.
Plan: [`PREREG.md`](PREREG.md), addenda [1](PREREG_ADDENDUM_1.md) and [2](PREREG_ADDENDUM_2.md).

| Time (UTC, 2026-11-10) | Item | State |
| --- | --- | --- |
| 09:28–09:45 | Smoke search run GX, 2 contexts | 2 records; 334 worker runs each, 1 timeout |
| 09:45–10:13 | Replay check on the smoke | 644 results per source, 0 worker runs (GS, GE, GX) |
| 10:15 | GX search run on `DEV` started (4 processes) | 0 finished contexts after 40 minutes |
| 10:36 | GX search run on `FIT-SEL` (xbabe1) stopped, 0 finished contexts | deferred until a `DEV` test passes (addendum 1) |
| 10:54 | GX `DEV` restarted as 16 processes, same run directory and memo | — |
| **11:34** | GX `DEV` finished contexts | **0 of 115** |
| 12:02 | **xbabe2 rebooted (not by this wave).** GX `DEV` died at 37 of 115 finished contexts; the unseeded smoke had finished variant B and was inside TBOTH | no torn record; memo intact (85 context files) |
| 12:22 | GX `DEV` resumed from its memo, same stage and arguments | the 37 finished contexts are store hits; in-flight contexts replay from the memo |
| 10:15–12:02 | Unseeded deploy smoke (B, TBOTH; 2 contexts) | B finished (about 60 minutes); TBOTH interrupted by the reboot and not resumed: the seeded smoke runs B and TBOTH instead |

**Interruption and lost time (reboot of 12:02Z).**
- Lost: the 20 minutes between the reboot and the resume, plus the Python-side progress of the contexts in flight
  (16 processes × about 50 minutes since their last finished context at worst; their worker evaluations are in the
  memo and are not repeated). About 1.5 hours of wall-clock in all for this run.
- After the reboot the bubblewrap sandbox on xbabe2 no longer starts (`kernel.apparmor_restrict_unprivileged_userns`
  is back at 1). I did not change the setting. From 12:22Z the xbabe2 runs use the docker sandbox
  (`sandbox_h335_docker.sh`, the one used on xbabe0, xbabe1 and xbabe3), with the same staged code.

**Negatives.**
- Two errors of mine were caught before any scored deploy: a mistyped design checksum (the first smoke deploy
  stopped on it) and a 60 s deploy limit where the stored recipe uses 300 s (addendum 2).
- The planned same-host replica of the baseline and the `FIT-SEL` replication were dropped to avoid computation that
  H417 already stores (addendum 1). The paired baseline is H417's stored deploy.
- The deploys cannot start before H417's deploy finishes (about 14:40), so no H420 result is expected before the
  late afternoon UTC.
