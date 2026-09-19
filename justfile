# JopeDime task runner. `just ci-pr` is what the hosted CI runs; `just ci-main` what xbabe2 runs on main.
set shell := ["bash", "-euo", "pipefail", "-c"]

default:
    @just --list

# one-time developer setup: git hooks, uv environment (CPU torch)
setup:
    git config core.hooksPath .githooks
    cd python && uv sync --extra dev --extra cpu

hygiene:
    bash deploy/ci/check.sh

fmt:
    cargo fmt --all -- --check

clippy:
    cargo clippy --workspace --all-targets --locked -- -D warnings

# engine + emitters, PR tier (no neural/gpu features; slow tests excluded by .config/nextest.toml)
test-pr:
    env OMP_NUM_THREADS=1 RAYON_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 JOPE_PARITY_STRICT=1 cargo nextest run -P pr --workspace --locked

# everything, main tier (xbabe2)
test-main:
    env OMP_NUM_THREADS=1 RAYON_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 JOPE_PARITY_STRICT=1 cargo nextest run -P main --workspace --locked -j 32

python-check:
    cd python && uv run ruff check . && uv run ruff format --check . && uv run pytest -q -m "not xbabe2"

# ADR-0003: no absolute home paths in code or units (python/src is informational until WP6 lands)
lint-paths:
    ! git grep -nE '/home/[a-z]+/' -- crates/*/src crates/*/tests deploy/systemd .github
    @git grep -cE '/home/[a-z]+/' -- python/src | awk -F: '{n+=$2} END {if (n) print n " hits in python/src (informational until WP6 lands)"}' || true

# render the unit templates into a directory (wraps crates/jope-paths' jope-units-render)
render-units home state bundle user out:
    bash deploy/systemd/render.sh --home {{home}} --state {{state}} --bundle {{bundle}} --user {{user}} --out {{out}}

# link every historical program name (deploy/systemd/BINARIES.md) to jope-selfplay / jope-campaigns in a bin directory
install-links bindir:
    bash deploy/systemd/render.sh --install-symlinks {{bindir}}

ci-pr: hygiene fmt clippy test-pr python-check

ci-main: ci-pr test-main

# kept for muscle memory: the historical gate name
required: ci-pr
