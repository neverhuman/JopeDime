# syntax=docker/dockerfile:1.7
# deploy/docker/kernel-builder.Dockerfile — glibc-2.35 builder for the kernel-v1 product
# evaluator lineage (deploy/ci/kernel-v1-product-build.sh). Replaces the private image
# jeryu-builder:rust1.95-glibc2.35 (sha256:008faf89…, built 2026-09-10 with no Dockerfile — plan
# blocker B17). Same contract: Ubuntu 22.04 userland (glibc 2.35, so the binary needs at most
# GLIBC_2.35 < the 2.36 appliance bound), rustc 1.95.0 selected through RUSTUP_TOOLCHAIN (which
# overrides the frozen source's rust-toolchain.toml channel; the receipt records both).
#
#   docker build -f deploy/docker/kernel-builder.Dockerfile -t jope-kernel-builder:rust1.95-glibc2.35 .
#
# The build script runs this image with --network none as the invoking uid, mounting CARGO_HOME
# and the source read-only; nothing in the image is writable by that uid except what is mounted.
FROM ubuntu:22.04@sha256:281c5745f657873d78e5531fc5ba8575f46ab7769b94550ac99543f122679986

ARG DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8 TZ=UTC
RUN apt-get update -qq && apt-get install -y -qq --no-install-recommends \
        ca-certificates curl build-essential pkg-config git \
    && rm -rf /var/lib/apt/lists/* \
    && ldd --version | head -1 | grep -F "2.35"

ARG RUSTUP_VERSION=1.29.1
ARG RUSTUP_INIT_SHA256=dda7234360b7f578ca8b0ddcb80145646fa61a67c1720a5abc7051b35c9fcb71
ARG RUST_TOOLCHAIN=1.95.0

COPY deploy/docker/fetch.sh /usr/local/bin/jope-fetch
RUN chmod 0755 /usr/local/bin/jope-fetch

ENV RUSTUP_HOME=/opt/rust/rustup CARGO_HOME=/opt/rust/cargo RUSTUP_TOOLCHAIN=${RUST_TOOLCHAIN}
ENV PATH=/opt/rust/cargo/bin:${PATH}
RUN set -eu; \
    jope-fetch "https://static.rust-lang.org/rustup/archive/${RUSTUP_VERSION}/x86_64-unknown-linux-gnu/rustup-init" "${RUSTUP_INIT_SHA256}" /tmp/rustup-init; \
    chmod +x /tmp/rustup-init; \
    /tmp/rustup-init -y --no-modify-path --profile minimal --default-toolchain "${RUST_TOOLCHAIN}"; \
    rm -f /tmp/rustup-init; \
    rustc -vV | grep -F "release: ${RUST_TOOLCHAIN}"; \
    cargo --version | grep -F "cargo ${RUST_TOOLCHAIN}"; \
    chmod -R a+rX /opt/rust

# The build script passes --user "$(id -u):$(id -g)" and HOME=/work; no user is baked in.
WORKDIR /work
CMD ["bash"]
