#!/usr/bin/env bash
# shellcheck disable=SC1083  # the container script below is a heredoc whose \${VAR} escapes are expanded inside the container
# Glibc-bounded CPU build of the kernel-v1 product evaluator lineage (the `kernel-v1-build` lane
# of main.yml / nightly.yml, label xbabe2-kernel).
#
# Usage: deploy/ci/kernel-v1-product-build.sh [options] OUT_DIR
#   --builder-image <ref>     builder image (default jope-kernel-builder:rust1.95-glibc2.35,
#                             built from deploy/docker/kernel-builder.Dockerfile — the private
#                             jeryu-builder image is no longer a dependency, plan B17)
#   --frozen-source <s3 key>  build a lineage source tree fetched from the archive bucket
#                             (s3://veox-jopedime-archive-use2/<key>, a .tar.zst of the frozen
#                             `source/` directory whose name carries its sha256) instead of the
#                             committed deploy/kernel-v1/source; the receipt records which one
#   --lineage <name>          lineage label in the receipt (default: the committed kernel-v1 sha)
#   --jobs <n>                cargo -j (default 16; env KERNEL_V1_JOBS)
#   --max-glibc <ver>         bound checked by jope-build-receipt (default 2.36; env KERNEL_V1_MAX_GLIBC)
#
# Phases:
#   1. fetch   (host, network): `cargo fetch --locked` of the source's exact Cargo.lock into
#              OUT_DIR/cargo-home. Only step that touches the network. KERNEL_V1_SEED_CARGO_HOME
#              copies an existing registry index + .crate cache first.
#   2. build   (docker --network none): the source is mounted read-only, copied to
#              OUT_DIR/work/src and built with `cargo build --release --offline --locked --bin
#              jope-parallel` against the mounted CARGO_HOME, as the invoking uid.
#   3. receipt (host): `jope-build-receipt` (built from this checkout, crates/jope-engine) writes
#              a jope.product_evaluator_build.v1 receipt binding the source tree, the builder's
#              `rustc -vV`, the image id and the binary's ELF facts, and fails unless the binary
#              needs at most GLIBC_<max-glibc> and links no CUDA/LibTorch library. The receipt's
#              `toolchain` object records the source's rust-toolchain.toml channel (1.97.1), the
#              builder's RUSTUP_TOOLCHAIN (1.95.0) and the actual `rustc -vV`, so the override
#              is explicit rather than silent.
#
# No Python is invoked. Not part of `just ci-pr`.
set -euo pipefail

repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
committed_lineage=kernel-v1-8aa0affbf24ca68b4d6ca039ca8fbd9bffa22e3c44c1c327bf81b733a596809a
image=${KERNEL_V1_BUILDER_IMAGE:-jope-kernel-builder:rust1.95-glibc2.35}
jobs=${KERNEL_V1_JOBS:-16}
max_glibc=${KERNEL_V1_MAX_GLIBC:-2.36}
frozen_key=""
lineage=""
archive_bucket=${KERNEL_V1_ARCHIVE_BUCKET:-veox-jopedime-archive-use2}
out=""

while [[ $# -gt 0 ]]; do
  case $1 in
    --builder-image) image=${2:?--builder-image needs a ref}; shift ;;
    --frozen-source) frozen_key=${2:?--frozen-source needs an s3 key}; shift ;;
    --lineage) lineage=${2:?--lineage needs a name}; shift ;;
    --jobs) jobs=${2:?--jobs needs a number}; shift ;;
    --max-glibc) max_glibc=${2:?--max-glibc needs a version}; shift ;;
    -h|--help) sed -n '3,33p' "$0"; exit 0 ;;
    -*) echo "unknown option $1" >&2; exit 2 ;;
    *) [[ -z $out ]] || { echo "unexpected argument $1" >&2; exit 2; }; out=$1 ;;
  esac
  shift
done
[[ -n $out ]] || { echo "usage: kernel-v1-product-build.sh [options] OUT_DIR (--help)" >&2; exit 2; }

for tool in docker cargo sha256sum; do
  command -v "$tool" >/dev/null || { echo "missing tool: $tool" >&2; exit 2; }
done

mkdir -p "$out"
out=$(cd "$out" && pwd)
stamp=$(date -u +%Y%m%dT%H%M%SZ)
run=$out/run-$stamp
cargo_home=$out/cargo-home
mkdir -p "$run/work" "$cargo_home"

# ---- source selection -----------------------------------------------------------------------
if [[ -z $frozen_key ]]; then
  source_dir=$repo/deploy/kernel-v1/source
  source_kind="committed deploy/kernel-v1/source"
  lineage=${lineage:-$committed_lineage}
  [[ -f $source_dir/Cargo.lock ]] || { echo "missing $source_dir/Cargo.lock" >&2; exit 1; }
  if [[ -n $(git -C "$repo" status --porcelain -- "$source_dir") ]]; then
    echo "release source differs from HEAD; refusing to build" >&2
    exit 1
  fi
  source_sha256=$(git -C "$repo" rev-parse "HEAD:deploy/kernel-v1/source")
  source_id_kind="git-tree"
  source_ref="git tree $source_sha256 at $(git -C "$repo" rev-parse HEAD)"
else
  command -v aws >/dev/null || { echo "--frozen-source needs the aws CLI (OIDC role JopeDimeCiRead)" >&2; exit 2; }
  # The archive key is <lineage>/source-<sha256>.tar.zst; the digest in the name is verified
  # against the download before anything is unpacked.
  base=$(basename "$frozen_key")
  expected=$(sed -n 's/^source-\([0-9a-f]\{64\}\)\.tar\.zst$/\1/p' <<<"$base")
  [[ -n $expected ]] || { echo "--frozen-source key must end in source-<sha256>.tar.zst, got $base" >&2; exit 2; }
  lineage=${lineage:-$(basename "$(dirname "$frozen_key")")}
  mkdir -p "$run/frozen"
  aws s3 cp --only-show-errors "s3://$archive_bucket/$frozen_key" "$run/frozen/$base"
  actual=$(sha256sum "$run/frozen/$base" | cut -c1-64)
  [[ $actual == "$expected" ]] || { echo "frozen source digest mismatch: expected $expected, got $actual" >&2; exit 1; }
  mkdir -p "$run/frozen/source"
  tar --zstd -xf "$run/frozen/$base" -C "$run/frozen/source" --strip-components=1
  source_dir=$run/frozen/source
  source_kind="frozen s3://$archive_bucket/$frozen_key"
  source_sha256=$expected
  source_id_kind="sha256"
  source_ref="archive object sha256 $expected"
  [[ -f $source_dir/Cargo.lock ]] || { echo "frozen source has no Cargo.lock" >&2; exit 1; }
fi
echo "source:  $source_kind ($source_ref)"
echo "lineage: $lineage"
echo "builder: $image"

# ---- 1. fetch (network allowed; host toolchain only resolves/downloads crates) --------------
if [[ -n ${KERNEL_V1_SEED_CARGO_HOME:-} && ! -d $cargo_home/registry ]]; then
  mkdir -p "$cargo_home/registry"
  cp -a "$KERNEL_V1_SEED_CARGO_HOME/registry/index" "$KERNEL_V1_SEED_CARGO_HOME/registry/cache" \
    "$cargo_home/registry/"
fi
CARGO_HOME=$cargo_home cargo fetch --locked --manifest-path "$source_dir/Cargo.toml"

# ---- 2. build (no network, read-only source, glibc-2.35 userland) ---------------------------
build_cmd="cargo build --release --offline --locked --bin jope-parallel -j $jobs"
build_start=$(date +%s)
image_id=$(docker image inspect --format '{{.Id}}' "$image") \
  || { echo "builder image $image is not present: docker build -f deploy/docker/kernel-builder.Dockerfile -t $image ." >&2; exit 1; }
docker run --rm --network none \
  --user "$(id -u):$(id -g)" \
  --cap-drop ALL --security-opt no-new-privileges --pids-limit 4096 \
  -e HOME=/work -e CARGO_HOME=/cargo-home -e CARGO_TARGET_DIR=/work/target \
  -e CARGO_NET_OFFLINE=true \
  -v "$source_dir:/src:ro" \
  -v "$cargo_home:/cargo-home" \
  -v "$run/work:/work" \
  "$image" \
  bash -euc "
    cp -r /src /work/src
    cd /work/src
    rustc -vV > /work/rustc-vV.txt
    printf '%s\n' \"\${RUSTUP_TOOLCHAIN-}\" > /work/rustup-toolchain.txt
    (ldd --version | head -n1) > /work/builder-libc.txt
    $build_cmd
  " 2>&1 | tee "$run/build.log"

binary=$run/work/target/release/jope-parallel
[[ -x $binary ]] || { echo "build produced no binary" >&2; exit 1; }
# The copy that was built must equal the source it is receipted as.
diff -r "$source_dir" "$run/work/src" >/dev/null

# ---- 3. receipt + bounds (host Rust tool from this checkout, no network) --------------------
cargo run --quiet --locked --offline --manifest-path "$repo/crates/jope-engine/Cargo.toml" \
  -j "$jobs" --bin jope-build-receipt -- \
  --binary "$binary" \
  --source-dir "$source_dir" \
  --output "$run/receipt.json" \
  --lineage "$lineage" \
  --rustc-vv "$run/work/rustc-vV.txt" \
  --rustup-toolchain "$run/work/rustup-toolchain.txt" \
  --profile release \
  --command "source=$source_kind; docker run --network none $image ($(cat "$run/work/builder-libc.txt")): $build_cmd" \
  --builder-image "$image@$image_id" \
  --max-glibc "$max_glibc" \
  --forbid-gpu-libraries >/dev/null

# Side receipt: which source was built (the jope-build-receipt schema has no source-kind field).
printf '{"schema":"jope.kernel-v1-source/v1","source":"%s","source_id":"%s","source_id_kind":"%s","lineage":"%s","builder_image":"%s@%s","rustc":"%s","builder_libc":"%s","build_seconds":%s,"receipt":"%s"}\n' \
  "$source_kind" "$source_sha256" "$source_id_kind" "$lineage" "$image" "$image_id" \
  "$(sed -n 's/^release: //p' "$run/work/rustc-vV.txt")" "$(cat "$run/work/builder-libc.txt")" "$(( $(date +%s) - build_start ))" \
  "$run/receipt.json" > "$run/source.json"

echo "binary:  $binary"
echo "sha256:  $(sha256sum "$binary" | cut -d' ' -f1)"
echo "receipt: $run/receipt.json"
echo "source:  $run/source.json"
