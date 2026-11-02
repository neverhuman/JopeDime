# QLT-04 Q1 Attempt-08 controller (source of record)

Byte-identical copies of the QLT-04 Q1 Attempt-08 controller code that is armed on xbabe2
(passive FIFO waiter behind the live JRB-01 GPU recovery). Merged to main per Jepson t200u
("merge all JRB-01/QLT-04/REL-07 code now; runs become validation"). The armed run keeps
executing its own copies under `~/jope_plan/evidence/QLT-04/qlt02-t0240/`; nothing there
was moved or deleted.

| File | Evidence source | SHA-256 |
| --- | --- | --- |
| `resume_controller.py` | `evidence/QLT-04/qlt02-t0240/attempt-08/resume_controller.py` (Gov04 PASS_SCOPED_CONTROLLER_REVIEW_NO_LAUNCH `abdb4009…`) | `94bce7a971ad5a39da4e23d99992ec9316745519d4bbb1062e5666ad32a0d339` |
| `sustained_load_gate.py` | `evidence/QLT-04/qlt02-t0240/attempt-08/sustained_load_gate.py` | `e1f029ad9dcc2aaafc6ecb72200bf8baf8247485c177402cdf4136e23b8bbe40` |
| `remote-engine-start.py` | `evidence/QLT-04/qlt02-t0240/remote-engine-start.py` | `13b0b3960a30526034eb767d7df2b271d1235cf4f5d11762c17b3025b906663f` |

Status: code only. Not built or tested in this merge; Q1 run outcome, REPORT.md and the
owner guidance default remain pending validation. The open QLT incomplete-panel question
(max 581/582 valid if the original invalid FINISHED row lacks durable-response provenance;
COORD 2026-11-02T18:07Z) is unresolved and is not changed by this copy.
