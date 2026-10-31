# jope-paths

Zero-dependency resolvers for JopeDime's runtime roots, plus the systemd
unit-template renderer behind `deploy/systemd/render.sh`.

Every resolver is environment-first and evaluated at call time (nothing is
cached; `once_str!` exists for call sites that want a `&'static str` once per
process).

| Resolver | Variable | Default | Historical value it replaces |
|---|---|---|---|
| `jope_home()` | `JOPE_HOME` | nearest ancestor of the current directory, then of the executable, holding a `.jope-home` marker; else the current directory | `/home/ubuntu/jope-dime` — the source checkout (assets, manifests, seeds, `python/`) |
| `jopedime_home()` | `JOPEDIME_HOME` | `$XDG_DATA_HOME/jopedime` or `~/.local/share/jopedime` on Linux; `~/Library/Application Support/jopedime` on macOS; `%LOCALAPPDATA%/jopedime` on Windows | — |
| `jope_state()` | `JOPE_STATE` | `$XDG_STATE_HOME/jope`, else `~/.local/state/jope`, else `<jope_home>/state` | `/home/ubuntu/jope-flywheel/test_exact_gen0` and every other mutable tree (runs, queues, sqlite, pointers, logs). The restored `parallel/` tree lives here; lineage-A material under `legacy/test_exact_gen0` |
| `jope_registry()` | `JOPE_REGISTRY` | `<jope_home>/datasets/by-id` | `/home/ubuntu/remote_super/quant/regression` (dataset id → directory) |
| `sotarena_root()` | `SOTARENA_ROOT` | `<jope_home>/benchmarks/sotarena` | the nested `sotarena` benchmark clone |
| `sotarena_reporter_root()` | `SOTARENA_REPORTER_ROOT` | `<jope_state>/arena` | `/home/ubuntu/sotarena` — the Rust Elo-reporter repo (not the benchmark clone) |

Additional derived roots the historical tree also used as anchors:

| Resolver | Variable | Default | Historical value |
|---|---|---|---|
| `scq_root()` | `JOPE_SCQ_ROOT` | `<jope_state>/scq` | `/home/ubuntu/smart_cluster` (`target/release/scq`, `var/runs`, `etc/projects.toml` keep their layout beneath it) |
| `quant_root()` | — | parent of `jope_registry()` | `/home/ubuntu/remote_super/quant` |
| `legacy_root()` | — | `<jope_state>/legacy` | `/home/ubuntu/jope-flywheel` and other retired per-host trees, by name |

`jope-engine` re-exports the crate as `jope_parallel::paths`; `jope-campaigns`
builds its campaign-specific locations on top of it in `paths.rs`.

## Unit templates

`deploy/systemd/{selfplay,engine,pending}/*.{service,timer}.in` carry four
placeholders — `@JOPE_HOME@`, `@JOPE_STATE@`, `@JOPE_BUNDLE@` (the installed
binaries directory), `@JOPE_USER@`. Render them with

```
deploy/systemd/render.sh --home /opt/jope/checkout --state /var/lib/jope \
    --bundle /opt/jope/bundle/bin --user jope --out /tmp/units
```

which wraps `cargo run -p jope-paths --bin jope-units-render`. Rendering is
plain substitution; an unknown `@NAME@` is an error.
