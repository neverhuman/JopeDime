# JopeDime task runner. `just ci-pr` is a local subset of the pull-request lanes (check.sh, fmt, the PR
# nextest tier, ruff and pytest) plus clippy, which no CI lane runs; the lanes are deploy/ci/fast-lane.sh.
# `just ci-main` adds the whole-workspace main tier; full-main-cpu runs deploy/ci/full-main.py, not this.
set shell := ["bash", "-euo", "pipefail", "-c"]

default:
    @just --list

# InnovationZero publication only: source/receipt checks and three typeset PDFs.
paper:
    make -C docs/paper/innovationzero check self-test pdf

# Validate imports, citations, immutable appendices, and deliberate refusal cases without TeX.
paper-check:
    make -C docs/paper/innovationzero check self-test

# one-time developer setup: git hooks, uv environment (CPU torch)
setup:
    git config core.hooksPath ops/git-hooks
    cd python && uv sync --extra dev --extra cpu

hygiene:
    bash deploy/ci/check.sh
    bash deploy/ci/check-api-schema.sh

# Full default-policy audit of immutable complete tracked input; fresh retained reports.
# Trusted-main execution uses full-main.yml; the dedicated hosted workflow remains opt-in.
audit:
    bash deploy/ci/install-jankurai.sh
    JANKURAI_NO_UPDATE_CHECK=1 JANKURAI_BIN="$PWD/target/jankurai/jankurai-1.7.5-x86_64-unknown-linux-gnu/jankurai" bash ops/ci/jankurai.sh "$PWD/target/jankurai/reports/local-$(date -u +%Y%m%dT%H%M%SZ)-$$"

fmt:
    cargo fmt --all -- --check

clippy:
    cargo clippy --workspace --all-targets --locked -- -D warnings

# `test-pr-deferred` is the heavy set a hosted runner cannot link (72 GB target); `test-pr-all` is
# the whole workspace. Both are below.
# PR tier: exactly what the hosted `rust-test` lane runs — the product crates, 617 tests
test-pr:
    env OMP_NUM_THREADS=1 RAYON_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 JOPE_PARITY_STRICT=1 bash -euo pipefail -c 's="$(bash deploy/ci/rust-test-scope.sh)"; read -ra p <<< "$s"; [ "${#p[@]}" -gt 0 ] || exit 1; exec cargo nextest run -P pr --locked "${p[@]}"'

# the crates deferred from the hosted lane; expect the failures in deploy/ci/known-broken-tests.txt
test-pr-deferred:
    env OMP_NUM_THREADS=1 RAYON_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 JOPE_PARITY_STRICT=1 bash -euo pipefail -c 's="$(bash deploy/ci/rust-test-scope.sh --deferred)"; read -ra p <<< "$s"; [ "${#p[@]}" -gt 0 ] || exit 1; exec cargo nextest run -P pr --locked "${p[@]}"'

# the binaries that spawn real daemons and must not race each other (deferred crates only, so this
# does not link the whole workspace to run 25 tests)
test-pr-serial:
    bash -euo pipefail -c 's="$(bash deploy/ci/rust-test-scope.sh --deferred)"; read -ra p <<< "$s"; [ "${#p[@]}" -gt 0 ] || exit 1; exec env OMP_NUM_THREADS=1 RAYON_NUM_THREADS=1 JOPE_PARITY_STRICT=1 cargo nextest run -P pr --locked "${p[@]}" --test-threads 1 -E "$(bash deploy/ci/rust-test-filter.sh --serial)"'

# everything in the PR tier, nothing excluded
test-pr-all:
    env OMP_NUM_THREADS=1 RAYON_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 JOPE_PARITY_STRICT=1 cargo nextest run -P pr --workspace --locked

# everything, main tier (xbabe2)
test-main:
    env OMP_NUM_THREADS=1 RAYON_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 JOPE_PARITY_STRICT=1 cargo nextest run -P main --workspace --locked -j 32

# host lane: the real-corpus and deferred-classification fixtures every CI family ignores
# (deploy/ci/real-corpus-tests.txt), run against a read-only custody copy of their inputs
test-real-corpus custody:
    bash deploy/ci/real-corpus-lane.sh {{custody}}

python-check:
    cd python && uv run ruff check . && uv run ruff format --check . && uv run pytest -q -m "not xbabe2"
    python3 deploy/dl-flywheel/pipelines/rlfw/run_rlfw_offline.py
    python3 deploy/dl-flywheel/pipelines/rlfw/multiseed/run_multiseed_offline.py

# ADR-0003: no absolute home paths in code or units (python/src is informational until WP6 lands)
# These two immutable evidence records retain literal custody locations, not runtime roots.
lint-paths:
    ! git grep -nE '/home/[a-z]+/' -- crates/*/src crates/*/tests deploy/systemd .github ':(exclude)crates/jope-appliance/tests/e2e/INT-01-ACCEPTANCE.md' ':(exclude)crates/jope-appliance/tests/system/artifact-stage-linux.json'
    @git grep -cE '/home/[a-z]+/' -- python/src | awk -F: '{n+=$2} END {if (n) print n " hits in python/src (informational until WP6 lands)"}' || true

# render the unit templates into a directory (wraps crates/jope-paths' jope-units-render)
render-units home state bundle user out:
    bash deploy/systemd/render.sh --home {{home}} --state {{state}} --bundle {{bundle}} --user {{user}} --out {{out}}

ci-pr: hygiene fmt clippy test-pr python-check

ci-main: ci-pr test-main

# kept for muscle memory: the historical gate name
required: ci-pr

# deploy/release/bump.sh does the work and says why; deploy/ci/test-bump.sh (hygiene lane) runs this
# recipe on a throwaway copy of the tree and checks every workspace afterwards.
# bump the Rust workspace version (workspace, jope-engine, jope-campaigns), Cargo.lock and the Lambda workspace locks that pin a root crate, then check them all as the release and the hygiene lane do; python/pyproject.toml is not bumped while no wheel ships
bump version:
    bash deploy/release/bump.sh {{quote(version)}}

# documentation checks: rustdoc builds (warnings counted), relative links in README/docs (not docs/legacy) resolve, cliff.toml renders
docs:
    #!/usr/bin/env bash
    set -euo pipefail
    # rustdoc: warnings are counted, not yet fatal — the pre-port feat-math/engine sources carry
    # intra-doc-link and unclosed-backtick warnings; the hosted `docs` lane (WP8) flips to
    # RUSTDOCFLAGS="-D warnings" once they are cleared crate by crate.
    cargo doc --workspace --no-deps --locked 2>&1 | tee target/rustdoc.log | grep -E '^(warning: `|error)' || true
    grep -qE '^error' target/rustdoc.log && { echo "rustdoc errors above"; exit 1; }
    echo "rustdoc warnings are advisory until the docs lane is blocking (see the per-crate counts above)"
    python3 - <<'PY'
    import os, re, sys
    files = ["README.md", "CONTRIBUTING.md", "SECURITY.md", "CODE_OF_CONDUCT.md"]
    for root, _, names in os.walk("docs"):
        if root == "docs/legacy" or root.startswith("docs/legacy/"):
            continue  # historical documents kept verbatim; excluded from every lint
        files += [os.path.join(root, n) for n in names if n.endswith(".md")]
    link = re.compile(r"\[[^\]]*\]\(([^)\s]+)\)")
    bad = 0
    for path in sorted(files):
        for target in link.findall(open(path, encoding="utf-8").read()):
            if target.startswith(("http://", "https://", "mailto:", "#")):
                continue
            rel = target.split("#", 1)[0]
            if not os.path.exists(os.path.normpath(os.path.join(os.path.dirname(path), rel))):
                print(f"{path}: broken link {target}")
                bad += 1
    print(f"{len(files)} markdown files, {bad} broken links")
    sys.exit(1 if bad else 0)
    PY
    if command -v git-cliff >/dev/null; then git cliff --config cliff.toml --unreleased --strip all > /dev/null && echo "cliff.toml ok"; else echo "git-cliff not installed (release.yml installs it)"; fi
