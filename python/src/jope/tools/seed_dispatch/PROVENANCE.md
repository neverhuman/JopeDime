# seed_dispatch lane (vendored)

Codex's seed-dispatch lane (`recovery/eight-a100-20260912/experiments/seed-dispatch`, untracked in git and present on the node under
`runs/eight-a100-recovery-20260912/experiments/seed-dispatch`) vendored so `successor_ledger.py` and its tests run from a checkout.
`dispatch.py` and `run_episode.py` are byte-identical to the node copies (sha256 below). `continuation.py` is Codex's v2 module
(node sha256 `f3497c341a946cd95275898f7b2aa00bda1fec83bc2365928a386f2b709144ef`) plus the additive `plan()` keywords `arm`, `role`, `seed_origin`, `replicate`, `protocol`, `generation`, `select`
(defaults reproduce the v2 identities byte for byte; `test_successor_ledger.PlanningTests` proves it). The modules import each other
by bare name; `successor_ledger.lane_module` loads them by path. Set `JOPE_SEED_DISPATCH_LANE` to use another lane copy.

| file | sha256 |
|---|---|
| continuation.py | `34174dc2aed647350cabbe85c8f02dd4d3223c577aa9cf37f5088e94b088993a` |
| dispatch.py | `818cdd52b787a04c84df6ff4ed569990883a340be4463c1268167886173b53a2` |
| run_episode.py | `809b2e0a8334d80ae6910d2d9e146f1e8b7c0b65d6548d473508d4a294093164` |

Vendored 2026-09-13 by Claude (Fable); never edit the node copies.
