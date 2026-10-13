# J26 — Legacy self-play reuse assessment

Status: assessment only. Nothing in `jope_selfplay_*` or `jope-arena` is adopted, deleted, scheduled, or promoted.
Source pin: `9ab56e980fa9d246b3ee2eb2d588bd3443424449` (inventory, `origin/main` after #305). Rebased onto `97ba24955143de07d39368f0c7f2cf22ca692908` (#306). The diff of `crates/jope-campaigns/src/bin`, `crates/jope-arena`, `docs/campaigns.md`, `docs/parity/capability-matrix.md`, `docs/adr/ADR-0005-canonical-learning-loop.md`, `deploy/systemd/selfplay`, and `deploy/dl-flywheel` between those two commits is empty.
Owner: `grok_gp_index`. Reviewer: `jopedime_clover`. Train slot 11. No PR until that slot.
L6 is a validation loss and lower is better. This stack does not compute L6. This document claims no improvement.

## Verdict

Do not reuse the `jope_selfplay_*` control plane as the JAILGUN expert-iteration loop.

That loop is lineage A. [ADR-0005](../../adr/ADR-0005-canonical-learning-loop.md) records it as the xbabe2 self-play flywheel of July–8 September 2026, and records its live state plane as gone. The canonical learning loop in that ADR is lineage B (step-60000 and `promotion_tournament.py`), which is also not this rlfw path.

The current flywheel tree does not call these binaries. A search of `crates/jope-learning`, `crates/jope-engine`, `crates/jope-pipeline/src`, and `deploy/dl-flywheel` at this pin finds no `jope_selfplay_` string. The SP-1 probe called `jope pipeline evolve` (`run_sp1.sh` in the SP-1 proposal). A search of that proposal tree finds no `jope_selfplay_` string.

Do not delete the binaries. `docs/campaigns.md` keeps the control plane with its tests and says it is not scheduled by default. This assessment does not remove code, units, or history.

`jope-arena` is a separate crate (SOTArena ladders and Elo). JAILGUN J19 defers Elo. `deploy/dl-flywheel` has no `jope-arena` reference. Do not use Elo as the flywheel score.

## What the 16 binaries are

`crates/jope-campaigns/src/bin/` contains 16 `jope_selfplay_*.rs` programs at this pin. The 2026-10-13 flywheel audit said 17. This tree has 16. The audit count is not repeated as fact.

| Binary | Role, from its module note |
|---|---|
| `jope_selfplay_wave` | Checkpoint-bound micro-wave planner. Freezes a 128-dataset cohort. No scheduler submit. |
| `jope_selfplay_supply` | Keeps a reserve of those waves. Invokes only `jope_selfplay_wave`. Does not submit SCQ jobs. |
| `jope_selfplay_queue` | Admits the newest 128-dataset wave for the active checkpoint. Submit is opt-in. |
| `jope_selfplay_freshness` | Single-writer checkpoint-freshness authority over a local socket. |
| `jope_selfplay_status` | Corpus ingestion, index export, operational status. |
| `jope_selfplay_corpus_freezer` | Advances the training-corpus watermark outside the promotion tournament. |
| `jope_selfplay_trainer_watchdog` | Validates inputs and submits one trainer spec. Python is confined to the payload. |
| `jope_selfplay_propose_driver` | Rebinds DL proposals when the lineage-A catalog changes. Eight direct-DL slots and 120 same-wave GP slots. |
| `jope_selfplay_expansion_driver` | Unlocks dataset tranches into the lineage-A catalog. Not GP program expansion. |
| `jope_selfplay_promotion` | Lineage-A promotion orchestration: tournament journal, gate spec, accepted pointer. Supply owns the next wave. |
| `jope_selfplay_shadow` | Audit surface for the causal shadow namespace. |
| `jope_selfplay_alarms` | Read-only health scan. Always exits 0. |
| `jope_selfplay_gc` | Run-dir garbage collector. Default is a plan, not a delete. |
| `jope_selfplay_bundle` | Content-addressed runtime bundler. |
| `jope_selfplay_mcp` | Stdio adapter for the freshness authority. |
| `jope_selfplay_tui` | Read-only operations console. |

`docs/campaigns.md` describes the chain as queue, freshness, supply, ingestion, corpus-freezer, trainer-watchdog, promotion. It points at `deploy/systemd/selfplay/*.service` and says 39 units. This pin has 27 `*.service.in` templates and 10 `*.timer.in` templates (37 files), not 39 installed units. They are templates. This assessment did not enable any of them.

`docs/parity/capability-matrix.md` lists the same 16 binaries as moved root binaries, with a pending WP5 collapse into one `jope-selfplay` multi-call binary. That collapse has not landed. It is not a reason to call the binaries from rlfw.

## Why this is not SP-1 or GB-06

SP-1 is a per-context ratchet: `jope pipeline evolve` for four rounds, HWM-seeded arm versus unseeded arm, parents 25–56. Clover's 2026-10-13T07:20Z readout is the status of that probe: 64/64 arm exits 0, selection-biased, NON-QUAL, no frozen-v1 L6 claim. This document does not rescore it. The probe is evidence that a new ratchet was run. It is not evidence that `jope_selfplay_*` ran.

GB-06 (`crates/jope-learning/src/clover/**`) is PolicyLM-to-GP seeding and GP expansion on the frozen TRAIN roster. The name `jope_selfplay_expansion_driver` is a catalog-unlock tool for lineage A. It is a different program. L5 (seal gate) is ordered not to edit `jope_selfplay_promotion.rs`. Promotion stays off the rlfw path. G36 is still a Jepson ruling, not a reuse of this binary.

`jope_selfplay_propose_driver` is the closest idea: a deep model seeds GP, and GP offspring fill the rest of a wave. The binding is the lineage-A catalog and checkpoint, with a 128-dataset wave and 64 proposals per selected dataset (`jope_selfplay_wave`). That is not the 2,051-d PolicyLM, the frozen-24 roster, or `jope pipeline evolve`. Reusing it would mean reviving lineage A, which ADR-0005 says has no live state plane on xbabe2.

## What not to do

- Do not schedule the systemd templates.
- Do not point rlfw, SP-1, GB-06, or R25 at these binaries.
- Do not treat `jope_selfplay_promotion` as the G36 champion gate.
- Do not treat `jope-arena` Elo as a substitute for paired L6 against frozen v1.
- Do not delete the crate. Retention is a separate decision. This document does not take it.

## Unknowns

The binaries were not executed. Live host state was not re-probed. ADR-0005's 2026-09-18 lost-state finding is cited, not re-verified on disk. Whether any off-repo scheduler still has a lineage-A unit enabled is unknown from this tree. Python research modules under `python/src/jope/research/` still name these binaries. They are the lineage-A bridge, not an rlfw caller.
