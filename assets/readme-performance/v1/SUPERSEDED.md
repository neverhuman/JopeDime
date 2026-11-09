# Superseded by v2 (2026-11-08)

The README's `current-performance` block is now rendered and checked by
[`../v2/render.sh`](../v2/render.sh). It shows the six-entrant AutoML Elo v3 panel
([`assets/automl-elo/v3`](../../automl-elo/v3/SOURCES.md)) and `kimik3p1_max`'s rank on
every panel it was scored on. `deploy/ci/check.sh` runs the v2 check.

This directory is kept unchanged as the provenance of the four-entrant block:

- `render.sh` rendered the block from [`assets/automl-elo/v1`](../../automl-elo/v1/SOURCES.md)
  and checked it against the paper's
  [exposure summary](../../../docs/paper/innovationzero/evidence/elo-summary.json) and the
  six v1 figure hashes.
- `replay.json` is the same-agent numerical replay of those v1 ratings.

**Do not run `render.sh --write` here.** It would overwrite the current README block with the
four-entrant table. Its `--check` now fails by design, because the README no longer holds that
block. The paper's evidence still binds the v1 result bytes, which this change does not touch.
