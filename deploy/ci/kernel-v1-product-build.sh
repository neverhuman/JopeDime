#!/usr/bin/env bash
# shellcheck disable=SC1083  # the container script below is a heredoc whose \${VAR} escapes are expanded inside the container
# Glibc-bounded CPU build of the kernel-v1 product evaluator lineage.
#
# Usage: ops/ci/kernel-v1-product-build.sh OUT_DIR
#
# The frozen kernel-v1 source (parallel/releases/kernel-v1-8aa0affbf…/source)
# is built inside a glibc-2.35 builder image so the binary runs on the
# Debian bookworm appliance (glibc 2.36). Phases:
#
#   1. fetch   (host, network): `cargo fetch --locked` of the release's exact
#              Cargo.lock into OUT_DIR/cargo-home. This is the only step that
#              touches the network. Set KERNEL_V1_SEED_CARGO_HOME to an existing
#              CARGO_HOME (e.g. ~/.cargo) to copy its registry index + .crate
#              cache first, so an already-populated host needs no download.
#   2. build   (docker --network none): the source is mounted read-only,
#              copied to OUT_DIR/work/src, and built with
#              `cargo build --release --offline --locked --bin jope-parallel`
#              against the mounted CARGO_HOME. Runs as the invoking uid.
#   3. receipt (host): `jope-build-receipt` (built from this checkout) writes
#              a jope.product_evaluator_build.v1 receipt binding the git tree
#              of the release source, the builder's `rustc -vV`, and the
#              binary's ELF facts, and fails unless the binary needs at most
#              GLIBC_${KERNEL_V1_MAX_GLIBC:-2.36} and links no CUDA/LibTorch library.
#              The limits and the verdict are recorded under `bounds`; a
#              rejected build still leaves a receipt with `passed: false`.
#
# Toolchain override: the builder image sets RUSTUP_TOOLCHAIN (1.95.0 for the
# default image), which takes precedence over the release source's
# rust-toolchain.toml (channel 1.97.1); the rustc that actually ran is the
# image's. The image is kept as is: its glibc-2.35 userland is what bounds
# the binary. The receipt's `toolchain` object records the file channel, the
# builder's RUSTUP_TOOLCHAIN, the actual `rustc -vV` release, and
# `matches_file`, so the override is explicit rather than silent.
#
# Python reads Cargo artifact messages on the host. Not part of `just required`.
set -euo pipefail
export CARGO_INCREMENTAL=${CARGO_INCREMENTAL:-0}
export CARGO_PROFILE_DEV_DEBUG=${CARGO_PROFILE_DEV_DEBUG:-line-tables-only}

out=${1:?usage: kernel-v1-product-build.sh OUT_DIR}
repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
lineage=kernel-v1-8aa0affbf24ca68b4d6ca039ca8fbd9bffa22e3c44c1c327bf81b733a596809a
source_dir=$repo/parallel/releases/$lineage/source
image=${KERNEL_V1_BUILDER_IMAGE:-jeryu-builder:rust1.95-glibc2.35}
jobs=${KERNEL_V1_JOBS:-16}
max_glibc=${KERNEL_V1_MAX_GLIBC:-2.36}

[[ -f $source_dir/Cargo.lock ]] || { echo "missing $source_dir/Cargo.lock" >&2; exit 1; }
if [[ -n $(git -C "$repo" status --porcelain -- "$source_dir") ]]; then
    echo "release source differs from HEAD; refusing to build" >&2
    exit 1
fi

mkdir -p "$out"
out=$(cd "$out" && pwd)
stamp=$(date -u +%Y%m%dT%H%M%SZ)
run=$out/run-$stamp
cargo_home=$out/cargo-home
mkdir -p "$run/work" "$cargo_home"

# 1. fetch (network allowed; host toolchain only resolves/downloads crates).
if [[ -n ${KERNEL_V1_SEED_CARGO_HOME:-} && ! -d $cargo_home/registry ]]; then
    mkdir -p "$cargo_home/registry"
    cp -a "$KERNEL_V1_SEED_CARGO_HOME/registry/index" "$KERNEL_V1_SEED_CARGO_HOME/registry/cache" \
        "$cargo_home/registry/"
fi
CARGO_HOME=$cargo_home cargo fetch --locked --manifest-path "$source_dir/Cargo.toml"

# 2. build (no network, read-only source, glibc-2.35 userland).
build_cmd="cargo build --release --offline --locked --bin jope-parallel -j $jobs --message-format=json-render-diagnostics"
image_id=$(docker image inspect --format '{{.Id}}' "$image")
docker run --rm --network none \
    --user "$(id -u):$(id -g)" \
    -e HOME=/work -e CARGO_HOME=/cargo-home -e CARGO_TARGET_DIR=/work/target \
    -e CARGO_NET_OFFLINE=true -e CARGO_INCREMENTAL -e CARGO_PROFILE_DEV_DEBUG \
    -v "$source_dir:/src:ro" \
    -v "$cargo_home:/cargo-home" \
    -v "$run/work:/work" \
    "$image" \
    bash -euc "
        cp -r /src /work/src
        cd /work/src
        rustc -vV > /work/rustc-vV.txt
        printf '%s\n' "\${RUSTUP_TOOLCHAIN-}" > /work/rustup-toolchain.txt
        (ldd --version | head -n1) > /work/builder-libc.txt
        cargo metadata --offline --locked --no-deps --format-version 1 > /work/metadata.json
        $build_cmd > /work/build.jsonl
    " 2>&1 | tee "$run/build.log"

binary=$(python3 "$repo/deploy/ci/cargo-artifact.py" select --package jope-parallel --bin jope-parallel \
    --metadata "$run/work/metadata.json" --messages "$run/work/build.jsonl" --path-map /work "$run/work")
# The copy that was built must equal the git-tracked source it is receipted as.
diff -r "$source_dir" "$run/work/src" >/dev/null

# 3. receipt + bounds (host Rust tool from this checkout, no network).
cargo run --quiet --locked --offline --manifest-path "$repo/parallel/Cargo.toml" \
    -j "$jobs" --bin jope-build-receipt -- \
    --binary "$binary" \
    --source-dir "$source_dir" \
    --output "$run/receipt.json" \
    --lineage "$lineage" \
    --rustc-vv "$run/work/rustc-vV.txt" \
    --rustup-toolchain "$run/work/rustup-toolchain.txt" \
    --profile release \
    --command "docker run --network none $image ($(cat "$run/work/builder-libc.txt")): $build_cmd" \
    --builder-image "$image@$image_id" \
    --max-glibc "$max_glibc" \
    --forbid-gpu-libraries >/dev/null

echo "binary:  $binary"
echo "sha256:  $(sha256sum "$binary" | cut -d' ' -f1)"
echo "receipt: $run/receipt.json"
