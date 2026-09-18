set shell := ["bash", "-eu", "-o", "pipefail", "-c"]

required:
    bash ops/ci/required.sh

fmt:
    cargo fmt --all -- --check

test:
    cargo test --locked --workspace
