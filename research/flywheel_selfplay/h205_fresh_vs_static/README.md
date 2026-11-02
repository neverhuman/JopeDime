# H205 — Fresh D4 families vs static 205-family pool and static teacher, 50 rounds from r882, H178 pooled selector

**Status: preregistered; launching on `xbabe1` RTX 4090 under `hwave_t2022_plan.md` unblock (c).** See [PREREG.md](PREREG.md).

- **Authority.** `"Jepson via Grok Bot, card answer 8:22 PM MT" (2026-11-01, t11s3): "Start H205 and H204 now with those unblocks; H206 after xbabe3 disk is cleared."`
- **Question.** Does supplying fresh nonsealed D4 `TRAIN` families + newly verified Rust-CV examples across 50 rounds (`FRESH`) break the 205-family static-pool ceiling (`STATIC-DATA`) or static-teacher ceiling (`STATIC-TEACHER`) from `r882` under equal compute?
- **Arms & Admission.**
  - Three active arms (`FRESH`, `STATIC-DATA`, `STATIC-TEACHER`) $\times$ `50` rounds from `r882` (`0084479aa0703667a0f3ccfe2beda60741c204795ec57b7c08542401ec6cdb64`) across seeds `205101–205103` (fold seed `205140`, boot seed `205141`).
  - Per-round admission via the `H178` calibrated `205`-family pooled cross-fit selector (`LCB95 > 0`, equal-family gain `> 0.001`, harm `<= 10%` of families).
- **Common Gate.**
  - Native `r882` `91/91` exact `DL-Val` parity (`1.1377941898164525`) before any candidate forward.
  - `205`-family pooled cross-fit `LCB95 > 0`, Holm $p < 0.05$ ($m = 2$), harm `<= 10%` of families.
  - Sealed D4 confirmation families (`76` families, `122` datasets) never enter training or per-round selection.
- **Execution & Guards.** Single GPU process on `xbabe1` `cuda:0` (`RTX 4090`, sharing with `PID 2297929`), `nice -n 19`, `<= 4` numeric threads (CUDA runtime/driver threads exempt), CPU `<= 85%`, disk `<= 93.8%`. `STATIC-DATA` + `STATIC-TEACHER` infra + `r882` parity run immediately; `FRESH` gates on `JRB-01 ADMISSION PASS`.
- **Code.** [runner.py](runner.py), [analysis.py](analysis.py), [monitor.py](monitor.py), [manifest.json](manifest.json), and [test_h205.py](test_h205.py).
