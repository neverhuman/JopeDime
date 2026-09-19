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

lint-paths:
    ! git grep -nE '/home/[a-z]+/' -- crates/*/src crates/*/tests python/src deploy/systemd .github

ci-pr: hygiene fmt clippy test-pr python-check

ci-main: ci-pr test-main

# kept for muscle memory: the historical gate name
required: ci-pr

# set the release version everywhere version-consistency.sh checks (workspace, jope-engine, jope-campaigns, python), refresh the lock, verify
bump version:
    sed -i -E '0,/^version = "[^"]+"/s//version = "{{version}}"/' Cargo.toml
    sed -i -E '0,/^version = "[^"]+"/s//version = "{{version}}"/' crates/jope-engine/Cargo.toml
    sed -i -E '0,/^version = "[^"]+"/s//version = "{{version}}"/' crates/jope-campaigns/Cargo.toml
    sed -i -E '0,/^version = "[^"]+"/s//version = "{{version}}"/' python/pyproject.toml
    cargo update --workspace --offline
    bash deploy/ci/version-consistency.sh
    @echo "bumped to {{version}}; commit, then tag v{{version}} to run release.yml"

# documentation checks: rustdoc -D warnings, relative links in README/docs (not docs/legacy) resolve, cliff.toml renders (git-cliff optional)
docs:
    #!/usr/bin/env bash
    set -euo pipefail
    RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps --locked
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
