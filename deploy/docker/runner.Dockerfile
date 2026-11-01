# syntax=docker/dockerfile:1.7
# deploy/docker/runner.Dockerfile — the self-hosted GitHub Actions runner image (WP8).
# Built locally on whichever xbabe host runs the lanes, never pushed to a registry:
#   docker build -f deploy/docker/runner.Dockerfile -t jope-runner:<git sha> .
#
# Base: the CUDA *12.8* devel image, not 13.0.1. deploy/receipts/libtorch.json (the WP8
# precondition) pins LibTorch 2.13.0+cu126, whose libtorch_cuda.so needs libcudart.so.12,
# libcublas.so.12, ... — the CUDA 13 image ships only the .13 runtime, so the `neural` feature
# could not link or run in it. 12.8.1 is the same toolkit major/minor as the xbabe2 host that
# produced the receipt (nvcc 12.8.93). The tag nvidia/cuda:13.0.1-devel-ubuntu24.04 exists
# (amd64 sha256:84e5f33efdccadc21599978ea7e6ed80cbe19bb432f9c339b15a66c4025ac983) and is the
# base to move to when the receipt is regenerated for a cu130 LibTorch.
#
# Every download is pinned by sha256. The neural SDK (LibTorch + the five NVIDIA wheels) is NOT
# in the image: it is a read-only named volume provisioned by deploy/runner/provision.sh from the
# receipt (docs/runbooks/neural-sdk.md). No credentials, no runner registration: the runner is
# configured per job by deploy/runner/run.sh with a JIT config passed on stdin.
FROM nvidia/cuda:12.8.1-devel-ubuntu24.04@sha256:4b9ed5fa8361736996499f64ecebf25d4ec37ff56e4d11323ccde10aa36e0c43

ARG DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8 TZ=UTC

# Compilers and interpreters match the hosted ubuntu-24.04 runner and the xbabe2 host that made
# the receipts: gcc 13.3, clang 18.1, python 3.12. libicu/krb5/lttng are the actions-runner
# (.NET) runtime dependencies (bin/installdependencies.sh installs the same set). iproute2 provides
# `ip` for the Linux clean-host pre-stage's loopback-only namespace checks.
RUN apt-get update -qq && apt-get install -y -qq --no-install-recommends \
        ca-certificates curl git git-lfs iproute2 jq unzip xz-utils zstd \
        build-essential cmake g++-13 clang-18 libclang-rt-18-dev lld-18 pkg-config libssl-dev libsqlite3-dev \
        python3.12 python3.12-venv python3-pip shellcheck \
        libicu74 libkrb5-3 zlib1g liblttng-ust1t64 libssl3 \
        openssh-client \
        libasound2t64 libatk-bridge2.0-0t64 libatk1.0-0t64 libatspi2.0-0t64 \
        libcairo2 libcups2t64 libdbus-1-3 libdrm2 libgbm1 libglib2.0-0t64 \
        libnspr4 libnss3 libpango-1.0-0 libx11-6 libxcb1 libxcomposite1 \
        libxdamage1 libxext6 libxfixes3 libxkbcommon0 libxrandr2 \
        fonts-liberation fonts-noto-color-emoji \
    && rm -rf /var/lib/apt/lists/* \
    && update-alternatives --install /usr/bin/clang++ clang++ /usr/bin/clang++-18 100 \
    && update-alternatives --install /usr/bin/clang clang /usr/bin/clang-18 100 \
    && test "$(g++ --version | head -1 | grep -o '13\.[0-9]*\.[0-9]*')" != "" \
    && test "$(clang++ --version | head -1 | grep -o '18\.[0-9]*\.[0-9]*')" != "" \
    && test "$(python3 --version | grep -o '3\.12\.[0-9]*')" != ""

# Non-root runner user: uid/gid 1001 (docker run --user 1001; the named volumes are chowned to
# it by deploy/runner/provision.sh). No sudo in the image.
RUN groupadd -g 1001 runner && useradd -u 1001 -g 1001 -m -d /home/runner -s /bin/bash runner

# ---- pinned tool versions (bump here; the lanes assert several of them, so a bump that
# skips .github/workflows/ci.yml will fail rather than drift) ----
ARG ACTIONS_RUNNER_VERSION=2.337.0
ARG ACTIONS_RUNNER_SHA256=70920811a4f8ad4328818682bca5c6469c1c942fab52448868071d0063816613
ARG RUSTUP_VERSION=1.29.1
ARG RUSTUP_INIT_SHA256=dda7234360b7f578ca8b0ddcb80145646fa61a67c1720a5abc7051b35c9fcb71
ARG NEXTEST_VERSION=0.9.133
ARG NEXTEST_SHA256=a9f992321e8759818400d93abb9477b4b11422d18d216e8d208505bd73454103
ARG CARGO_DENY_VERSION=0.20.2
ARG CARGO_DENY_SHA256=9f12ed4c49936e09b48bf862b595cde2fe64fcbd9d74dfacac6131ca824c8d5f
ARG CARGO_AUDIT_VERSION=0.22.2
ARG CARGO_AUDIT_SHA256=7fb9497f8594b389e5fce5ef9b92db08432996895b2e0c5a0167a69ed445c428
ARG CARGO_SBOM_VERSION=0.10.0
ARG CARGO_SBOM_SHA256=860a50d3e009c72fefc12506869cc11ea34d240c65da65cda2c491ed2aaa34f3
ARG CARGO_SWEEP_VERSION=0.8.0
ARG JUST_VERSION=1.58.0
ARG JUST_SHA256=4a5cc2f53e6f0f8c59092a6cc38291eb729d46a7dd95d3ae582008881b84931d
ARG GITLEAKS_VERSION=8.21.2
ARG GITLEAKS_SHA256=5bc41815076e6ed6ef8fbecc9d9b75bcae31f39029ceb55da08086315316e3ba
ARG ACTIONLINT_VERSION=1.7.8
ARG ACTIONLINT_SHA256=be92c2652ab7b6d08425428797ceabeb16e31a781c07bc388456b4e592f3e36a
ARG SYFT_VERSION=1.40.0
ARG SYFT_SHA256=f551cd16da3a5456f5245bb8045b98594263a678a9d2a07b462a05be0357b795
ARG UV_VERSION=0.11.28
ARG UV_SHA256=e490a6464492183c5d4534a5527fb4440f7f2bb2f228162ad7e4afe076dc0224
# Node 26.1.0: the version on xbabe2 that produced the emitter parity receipts (Ubuntu 24.04's
# apt nodejs is 18.19, older than every hosted runner); pinned from nodejs.org SHASUMS256.txt.
ARG NODE_VERSION=26.1.0
ARG NODE_SHA256=9fc6f21b6c4a62439727123e510e9c39febb2f563738f4927cd3e0b288c9b3c9
# AWS CLI v2 (the ci-receipts upload; `awscli` is not in the CUDA image's apt sources). AWS
# publishes no checksum file: the sha256 below was computed from the zip on 2026-09-19.
ARG AWSCLI_VERSION=2.36.49
ARG AWSCLI_SHA256=6b51b62d232217350e72d8f2493f47d3461239d011d1d5eef960dfcd8cb07069
# GitHub CLI: .github/workflows/main-guard.yml is three `gh api` / `gh issue create` calls, and
# the CUDA base image has no gh, so that guard fails on every push to main without it. sha256
# from cli/cli's published gh_<version>_checksums.txt.
ARG GH_VERSION=2.101.0
ARG GH_SHA256=9bca2d1c16825f109907a23307628a2f0698fbf99662b73a5cf0b020293072b8

# fetch <url> <sha256> <dest>: download, verify, fail closed.
COPY deploy/docker/fetch.sh /usr/local/bin/jope-fetch
RUN chmod 0755 /usr/local/bin/jope-fetch

# ---- Rust: rustup pinned by sha256, toolchain taken from rust-toolchain.toml so the image can
# never disagree with the repository (the check at the end fails the build otherwise). ----
ENV RUSTUP_HOME=/opt/rust/rustup CARGO_HOME=/opt/rust/cargo
ENV PATH=/opt/rust/cargo/bin:/opt/node/bin:${PATH}
COPY rust-toolchain.toml /tmp/rust-toolchain.toml
RUN set -eu; \
    channel=$(sed -n 's/^channel = "\(.*\)"/\1/p' /tmp/rust-toolchain.toml); \
    test -n "$channel"; \
    jope-fetch "https://static.rust-lang.org/rustup/archive/${RUSTUP_VERSION}/x86_64-unknown-linux-gnu/rustup-init" "${RUSTUP_INIT_SHA256}" /tmp/rustup-init; \
    chmod +x /tmp/rustup-init; \
    /tmp/rustup-init -y --no-modify-path --profile minimal --default-toolchain "$channel" --component rustfmt --component clippy; \
    rm -f /tmp/rustup-init /tmp/rust-toolchain.toml; \
    rustc --version | grep -F "rustc $channel"; \
    cargo --version | grep -F "cargo $channel"

RUN set -eu; \
    jope-fetch "https://github.com/nextest-rs/nextest/releases/download/cargo-nextest-${NEXTEST_VERSION}/cargo-nextest-${NEXTEST_VERSION}-x86_64-unknown-linux-gnu.tar.gz" "${NEXTEST_SHA256}" /tmp/nextest.tgz; \
    tar -xzf /tmp/nextest.tgz -C /opt/rust/cargo/bin cargo-nextest; \
    jope-fetch "https://github.com/EmbarkStudios/cargo-deny/releases/download/${CARGO_DENY_VERSION}/cargo-deny-${CARGO_DENY_VERSION}-x86_64-unknown-linux-musl.tar.gz" "${CARGO_DENY_SHA256}" /tmp/deny.tgz; \
    tar -xzf /tmp/deny.tgz -C /opt/rust/cargo/bin --strip-components=1 "cargo-deny-${CARGO_DENY_VERSION}-x86_64-unknown-linux-musl/cargo-deny"; \
    jope-fetch "https://github.com/rustsec/rustsec/releases/download/cargo-audit%2Fv${CARGO_AUDIT_VERSION}/cargo-audit-x86_64-unknown-linux-musl-v${CARGO_AUDIT_VERSION}.tgz" "${CARGO_AUDIT_SHA256}" /tmp/audit.tgz; \
    tar -xzf /tmp/audit.tgz -C /opt/rust/cargo/bin --strip-components=1 "cargo-audit-x86_64-unknown-linux-musl-v${CARGO_AUDIT_VERSION}/cargo-audit"; \
    jope-fetch "https://github.com/psastras/sbom-rs/releases/download/cargo-sbom-v${CARGO_SBOM_VERSION}/cargo-sbom-x86_64-unknown-linux-gnu" "${CARGO_SBOM_SHA256}" /opt/rust/cargo/bin/cargo-sbom; \
    chmod 0755 /opt/rust/cargo/bin/cargo-sbom; \
    rm -f /tmp/*.tgz; \
    cargo nextest --version | grep -F "cargo-nextest ${NEXTEST_VERSION}"; \
    cargo deny --version | grep -F "cargo-deny ${CARGO_DENY_VERSION}"; \
    cargo audit --version | grep -F "${CARGO_AUDIT_VERSION}"; \
    cargo sbom --version | grep -F "${CARGO_SBOM_VERSION}"
# cargo-sweep publishes no binary release: it is compiled from crates.io with the crate's own
# Cargo.lock (--locked), so every dependency is pinned by the registry checksums in that lockfile.
RUN cargo install cargo-sweep --version "${CARGO_SWEEP_VERSION}" --locked \
    && cargo sweep --version | grep -F "${CARGO_SWEEP_VERSION}" \
    && rm -rf /opt/rust/cargo/registry /opt/rust/cargo/git /opt/rust/cargo/.package-cache

# ---- other tools ----
RUN set -eu; \
    jope-fetch "https://github.com/casey/just/releases/download/${JUST_VERSION}/just-${JUST_VERSION}-x86_64-unknown-linux-musl.tar.gz" "${JUST_SHA256}" /tmp/just.tgz; \
    tar -xzf /tmp/just.tgz -C /usr/local/bin just; \
    jope-fetch "https://github.com/gitleaks/gitleaks/releases/download/v${GITLEAKS_VERSION}/gitleaks_${GITLEAKS_VERSION}_linux_x64.tar.gz" "${GITLEAKS_SHA256}" /tmp/gitleaks.tgz; \
    tar -xzf /tmp/gitleaks.tgz -C /usr/local/bin gitleaks; \
    jope-fetch "https://github.com/rhysd/actionlint/releases/download/v${ACTIONLINT_VERSION}/actionlint_${ACTIONLINT_VERSION}_linux_amd64.tar.gz" "${ACTIONLINT_SHA256}" /tmp/actionlint.tgz; \
    tar -xzf /tmp/actionlint.tgz -C /usr/local/bin actionlint; \
    jope-fetch "https://github.com/anchore/syft/releases/download/v${SYFT_VERSION}/syft_${SYFT_VERSION}_linux_amd64.tar.gz" "${SYFT_SHA256}" /tmp/syft.tgz; \
    tar -xzf /tmp/syft.tgz -C /usr/local/bin syft; \
    jope-fetch "https://github.com/astral-sh/uv/releases/download/${UV_VERSION}/uv-x86_64-unknown-linux-gnu.tar.gz" "${UV_SHA256}" /tmp/uv.tgz; \
    tar -xzf /tmp/uv.tgz -C /usr/local/bin --strip-components=1 uv-x86_64-unknown-linux-gnu/uv uv-x86_64-unknown-linux-gnu/uvx; \
    jope-fetch "https://nodejs.org/dist/v${NODE_VERSION}/node-v${NODE_VERSION}-linux-x64.tar.xz" "${NODE_SHA256}" /tmp/node.txz; \
    mkdir -p /opt/node; tar -xJf /tmp/node.txz -C /opt/node --strip-components=1; \
    jope-fetch "https://github.com/cli/cli/releases/download/v${GH_VERSION}/gh_${GH_VERSION}_linux_amd64.tar.gz" "${GH_SHA256}" /tmp/gh.tgz; \
    tar -xzf /tmp/gh.tgz -C /usr/local/bin --strip-components=2 "gh_${GH_VERSION}_linux_amd64/bin/gh"; \
    jope-fetch "https://awscli.amazonaws.com/awscli-exe-linux-x86_64-${AWSCLI_VERSION}.zip" "${AWSCLI_SHA256}" /tmp/awscli.zip; \
    unzip -q /tmp/awscli.zip -d /tmp/awscli; /tmp/awscli/aws/install --bin-dir /usr/local/bin --install-dir /opt/aws-cli; rm -rf /tmp/awscli; \
    rm -f /tmp/*.tgz /tmp/*.txz /tmp/*.zip; \
    just --version | grep -F "just ${JUST_VERSION}"; \
    gitleaks version | grep -F "${GITLEAKS_VERSION}"; \
    actionlint -version | head -1 | grep -F "${ACTIONLINT_VERSION}"; \
    syft version | grep -F "${SYFT_VERSION}"; \
    uv --version | grep -F "uv ${UV_VERSION}"; \
    node --version | grep -F "v${NODE_VERSION}"; \
    gh --version | grep -F "gh version ${GH_VERSION}"; \
    aws --version | grep -F "aws-cli/${AWSCLI_VERSION}"

# ---- actions-runner (JIT/ephemeral use only; never `config.sh`-registered in the image) ----
RUN set -eu; \
    mkdir -p /opt/actions-runner; \
    jope-fetch "https://github.com/actions/runner/releases/download/v${ACTIONS_RUNNER_VERSION}/actions-runner-linux-x64-${ACTIONS_RUNNER_VERSION}.tar.gz" "${ACTIONS_RUNNER_SHA256}" /tmp/runner.tgz; \
    tar -xzf /tmp/runner.tgz -C /opt/actions-runner; \
    rm -f /tmp/runner.tgz; \
    chown -R 1001:1001 /opt/actions-runner; \
    grep -qF "${ACTIONS_RUNNER_VERSION}" /opt/actions-runner/bin/Runner.Listener.deps.json

COPY --chmod=0755 deploy/runner/entrypoint.sh /usr/local/bin/jope-runner-entrypoint

# Runtime layout (deploy/runner/run.sh mounts these; the root filesystem is read-only):
#   /home/runner        tmpfs: the per-job copy of /opt/actions-runner + _diag
#   /home/runner/work   volume jope-work: the job workspace (checkout)
#   /var/cache/jope/*   volumes jope-cargo (CARGO_HOME), jope-target (CARGO_TARGET_DIR), jope-uv
#   /opt/jope-sdk       volume jope-sdk, read-only: LibTorch + NVIDIA wheels (JOPE_NEURAL_SDK_ROOT)
#   /var/lib/jope/models volume jope-models: pinned model bundles (JOPE_MODELS_ROOT)
ENV CARGO_HOME=/var/cache/jope/cargo \
    UV_CACHE_DIR=/var/cache/jope/uv \
    JOPE_NEURAL_SDK_ROOT=/opt/jope-sdk \
    JOPE_MODELS_ROOT=/var/lib/jope/models \
    CUDA_HOME=/usr/local/cuda \
    RUNNER_MANUALLY_TRAP_SIG=1 \
    ACTIONS_RUNNER_PRINT_LOG_TO_STDOUT=1 \
    DOTNET_CLI_TELEMETRY_OPTOUT=1
USER 1001:1001
WORKDIR /home/runner
ENTRYPOINT ["/usr/local/bin/jope-runner-entrypoint"]
