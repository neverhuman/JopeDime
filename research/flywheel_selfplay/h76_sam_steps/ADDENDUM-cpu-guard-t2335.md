# H76 CPU-guard infrastructure retry (owner t2335)

Prospective operational addendum before any H76 dev outcome. The frozen
PREREG.md remains byte-identical (SHA256
`a338d290b5dbafcd7adc0459bf91178c9b3ac74c21dade28cd2ab33b5996a4a8`).
Owner instruction: `~/nudges/sol_v1_hwv_h76_cpu_guard_t2335.md`.

The first K8/seed12 timing attempt stopped at 2026-10-31 05:27:44Z because host
load reached 89.17/102 cores. Charge its full 65.85235959594138 seconds to the
original 2700-second cap and retain its 106.17447731108405 seconds of lock wait.
It is an infrastructure void, not an outcome. Although the child wrote a
complete timing-only receipt, neither its loss nor any utility was inspected or
used. No dev score was generated. The original raw timing receipt/log, budget
entry, incomplete.json and launch metadata remain untouched. Receipt SHA256:
`3cfeb22242f2060ccfbc6834a97787dedb701501f79dcc17201c6246861cd70a`.
The actual execution source was `36d87153e6fbc681aeb822b63a37bf6f70a3fece`,
whose H76/shared runner bytes equal CPU PR842's reviewed source.

The owner now explicitly permits exactly one timing retry. This narrowly
supersedes the preregistration's no-retry rule for this single infrastructure
void only. No outcome-cell retries, seed replacement, hyperparameter changes,
checkpoint selection or admission-rule changes are permitted. Resume requires
one fully accounted CPU-guard timing failure, no in-progress/uncertain launch,
no outcome artifacts, and the original preregistration and cap identity. It
retains the ledger and uses distinct attempt2 log/receipt names. A second timing
failure stops this wave. The retry and all 12 outcomes share the original cap;
feasibility is prior void wall + retry wall * 13 * 1.25 <=2700 seconds.

Before each FIFO ticket (and hence each cell), require two consecutive
one-minute-load checks <=86.7, 30 seconds apart; back off at most 30 minutes.
Checks/waits happen outside qualification.lock and are recorded. After ticket
wait, the helper's fresh preflight still applies before launching. Keep the
in-cell CPU/load, RSS, memory, disk and wall watchdogs unchanged. The two-check
wait also supplies the required >=30-second yield before every fresh ticket.
Keep FIFO/legacy backoff; never pause or kill anyone else's jobs.

Scientific context supplied by the owner: H61 closeout PR854 reports 0/240
accepts under its selection rules; H59 CLIP was least harmful (Elo1487 versus
CONTROL1479, both below r882). These observations motivate H76 priority only;
H76 arms and analysis remain frozen. They will be noted as context in the final
README, without importing their scores as H76 controls.
