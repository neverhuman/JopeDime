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

# self-hosted lanes (xbabe2 only; the scripts refuse or block with a precise reason elsewhere)
gpu-qualification *ARGS:
    bash deploy/ci/gpu-qualification.sh {{ARGS}}

e2e-smoke:
    bash deploy/ci/e2e-smoke.sh

kernel-v1-build *ARGS:
    bash deploy/ci/kernel-v1-product-build.sh {{ARGS}} target/kernel-v1

# the no-silent-skip assertion over the last `just test-main` run
no-silent-skip:
    cargo nextest list -P main --workspace --locked --message-format json > target/nextest-list-main.json
    python3 deploy/ci/no-silent-skip.py --lane full-rust --list target/nextest-list-main.json --junit target/nextest/main/junit-main.xml

# local images (never pushed)
runner-image:
    DOCKER_BUILDKIT=1 docker build -f deploy/docker/runner.Dockerfile -t jope-runner:$(git rev-parse --short=12 HEAD) -t jope-runner:current .
    DOCKER_BUILDKIT=1 docker build -f deploy/docker/kernel-builder.Dockerfile -t jope-kernel-builder:rust1.95-glibc2.35 .

ci-pr: hygiene fmt clippy test-pr python-check

ci-main: ci-pr test-main no-silent-skip gpu-qualification e2e-smoke

# kept for muscle memory: the historical gate name
required: ci-pr
