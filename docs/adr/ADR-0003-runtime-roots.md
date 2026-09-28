# ADR-0003 — Runtime roots resolved from the environment, never from `/home/ubuntu/…`

**Status:** accepted 2026-09-18; implementation in PR [#13](https://github.com/veox-ai/JopeDime/pull/13)
(WP4) · **Plan section:** "Path relocation (ADR-0003)", blockers B8, B12, B20, B27, "Verification"
item 8 · **Scope:** every crate, `python/`, `deploy/systemd`, `.github`

## Context

The historical tree hard-coded one machine's layout: 420 occurrences of `/home/ubuntu/jope-dime`,
193 of `/home/ubuntu/jope-flywheel`, 105 of `/home/ubuntu/remote_super`, 65 of
`/home/ubuntu/smart_cluster` in tracked sources (hotspots: `src/bin` 132, provenance manifests
131, `parallel/tools` 75, `parallel/systemd` 61, `src/test_adaptive` 42, `src/main.rs` 21 —
including **ten** clap `default_value = "/home/ubuntu/sotarena"`, which is the Rust Elo-*reporter*
repository, not the nested SOTArena benchmark clone; B20). `parallel/src` had no `jope-dime` path
but two other absolute roots; the grammar JSONs contain no absolute paths (their pins are keyed by
relative `jope/grammars/…`, preserved by keeping `python/src/jope/`). Crate-relative
`include_str!`/`include_bytes!` inputs (grammar, manifests, seeds, champions; B1–B9) are not paths
at all and must not be relocated.

## Decision

Five roots, each resolved **at call time, environment first**, in one zero-dependency crate
`crates/jope-paths` (re-exported as `jope_parallel::paths`; mirrored by `jope/paths.py`):

| Resolver | Variable | Default | Replaces |
|---|---|---|---|
| `jope_home()` | `JOPE_HOME` | nearest ancestor of the current directory, then of the executable, holding the committed empty marker `.jope-home`; else the current directory | `~/jope-dime` — the checkout: assets, manifests, seeds, `python/`, `models/`, `datasets/` |
| `jope_state()` | `JOPE_STATE` | `$XDG_STATE_HOME/jope`, else `$HOME/.local/state/jope`, else `<jope_home>/state` | `~/jope-flywheel/test_exact_gen0` and every other mutable tree (runs, queues, sqlite, pointers, logs). The restored `parallel/` tree lives here; lineage-A material under `legacy/` |
| `jope_registry()` | `JOPE_REGISTRY` | `<jope_home>/datasets/by-id` | `~/remote_super/quant/regression` — dataset ids resolve to `<hash>/{train.csv,test.csv,meta.json}`; `prepare.rs`'s `ends_with("remote_super/quant")` compares against this instead |
| `sotarena_root()` | `SOTARENA_ROOT` | `<jope_home>/benchmarks/sotarena` | the nested benchmark clone (an HTTPS submodule after WP4) |
| `sotarena_reporter_root()` | `SOTARENA_REPORTER_ROOT` | `<jope_state>/arena` | `~/sotarena`, the Elo-reporter data root the campaign CLIs read leaderboards from (served by `crates/jope-arena`, WP12) |

Two derived roots exist because the historical tree used them as roots of their own:
`scq_root()` / `JOPE_SCQ_ROOT` (default `<jope_state>/scq`; replaces `~/smart_cluster` — the
scheduler checkout whose `target/release/scq`, `var/runs`, `etc/projects.toml` keep their relative
layout under it; `scqd` listens only there), `quant_root()` (parent of the registry) and
`legacy_root()` (`<jope_state>/legacy`). Environment values are taken verbatim when non-empty;
relative values are not resolved against anything (systemd and shells hand over absolute paths).
Nothing is cached, so a test may set a variable, call a resolver and restore it.

Mechanics: clap defaults become `env = jope_paths::var::…`; `assets/manifests/provenance-*.json`
evidence paths are rewritten to relative `evidence/...`; the 39 + 22 + 1 systemd units become
`deploy/systemd/{selfplay,engine,pending}/*.service.in` templates (`@JOPE_BUNDLE@`, `@JOPE_STATE@`,
`@JOPE_USER@`) rendered by `install-services.sh`, with `tests/systemd_unit_paths.rs` running on
the rendered output. The lint `just lint-paths`
(`git grep -nE '/home/[a-z]+/' -- crates/*/src crates/*/tests python/src deploy/systemd .github`)
must be empty; `deploy/ci/check.sh` reports the count until WP4 lands and fails afterwards.

## Consequences

- One checkout runs on a laptop, in a container and on xbabe2 with at most five variables set;
  the xbabe2 layout is the *default* nowhere and a *configuration* everywhere.
- The two "sotarena" meanings are separated by name, which ends the B20 conflation; the Elo
  reporter's data become part of the learning store's local mirror under `JOPE_STATE`.
- Restore drills (WP12) become possible: `jope models fetch`, `jope datasets fetch`,
  `jope learn pull` populate `JOPE_HOME`/`JOPE_STATE` on a fresh machine.
- Documents under `docs/legacy/` keep their absolute paths verbatim; they are not operational and
  are excluded from the lint (as is `deploy/kernel-v1/source`, a frozen lineage).
- Python mirrors the same table (`jope/paths.py`), so a grammar pin, a trainer or a research
  driver never re-derives a root of its own.

## Verification

- PR #13: `crates/jope-paths` with 13 tests; `just lint-paths` = 0 hits over the Rust crates,
  `deploy/systemd` and `.github`; `cargo test -p jope-dime-regressor --test systemd_unit_paths`
  on rendered units; `git clone --recurse-submodules` of the repository succeeds in a clean
  container (WP4 acceptance).
- Plan "Verification" item 8: `git grep -nE 'git\.neverhuman\.org|127\.0\.0\.1:8787|/home/ubuntu/' -- crates python assets deploy .github`
  is empty (docs/legacy excluded).
- Today on `main` (before PR #13) the remaining absolute defaults are the campaign CLIs'
  `DEFAULT_SOTARENA_ROOT` and the Golden42 node-lease path, both replaced by the table above.

## Amendments

- 2026-09-28 (clean-base): `deploy/kernel-v1/source` was removed from the tree, so its lint
  exclusion under Consequences no longer applies to anything. Restore it with the command in
  [docs/archive/clean-base-manifest.json](../archive/clean-base-manifest.json). The decision
  above is unchanged.
