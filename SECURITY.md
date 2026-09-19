# Security policy

## Reporting a vulnerability

Do not open a public issue for a vulnerability. Email the owner named in
[.github/CODEOWNERS](.github/CODEOWNERS); once the repository is public, GitHub private
vulnerability reporting (<https://github.com/veox-ai/JopeDime/security/advisories/new>) is the
second channel. Include the commit sha, the
component (crate, Python package, workflow, S3 layout), a reproduction, and the impact you see.
You will get an acknowledgement within three working days and a fix or a documented decision within
30 days for anything exploitable; credit is given in the release notes unless you prefer otherwise.

## Scope

- Rust crates under `crates/`, the Python package under `python/`, the emitted-program drivers
  under `deploy/ci/emitters/`, systemd templates under `deploy/systemd/`, and every workflow under
  `.github/workflows/`.
- The S3 layouts described in [docs/datasets.md](docs/datasets.md) and the models/learning-store
  documents (landing with WP7/WP11): bucket policies, OIDC trust, content-addressed integrity.
- Emitted programs (`model.py`, `model.rs`, `model.cpp`, `model.js`): they are generated from a
  genome and never contain network, filesystem or environment access — a finding that shows an
  emitted artefact doing any of those is a vulnerability in the emitter.

Out of scope: the vendored `crates/smartcluster/*` workspace (report upstream to jain-smartcluster;
a copy of the report here is welcome), and the archived legacy documents under `docs/legacy/`.

## What is already enforced

- No credentials in the tree: `*.env`, `*.pat`, `*.pem`, `*.key` are ignored by `.gitignore`;
  `gitleaks detect --redact` runs in the hygiene lane with [`.gitleaks.toml`](.gitleaks.toml);
  the history was filtered before the first push (ADR-0001) and scanned for
  `github_pat_`/`ghp_`/`AKIA`/private-key markers.
- Workflows run with `permissions: contents: read` by default; every third-party action is pinned
  to a commit SHA; pull-request lanes have no cloud credentials and no self-hosted runner label
  (`id-token: write` exists only in main/nightly/release workflows).
- Supply chain: `Cargo.lock` committed and `--locked` everywhere; `cargo deny check bans licenses
  sources` (`allow-git = []`, unknown registries denied) in every PR; Dependabot for cargo, pip and
  actions; SBOMs and build-provenance attestations on every release
  ([release.yml](.github/workflows/release.yml)).
- Binaries: the workspace lint table sets `unsafe_code = "deny"`; today the vendored smartcluster
  crates inherit it (`[lints] workspace = true`) and the product crates opt in as they are
  hardened (`battle-gpu`, the raw CUDA driver FFI, stays exempt by design). Release binaries are
  receipted with a glibc bound and a no-GPU-library check (`jope-build-receipt`).
- Cloud: OIDC roles `JopeDimeCiRead`, `JopeDimeTrainer`, `JopeDimeRelease` are bound to exact
  `sub` claims; buckets are versioned, SSE-S3, Block Public Access, Object Lock where the plan says
  so. Rotation procedure: [docs/runbooks/rotate-credentials.md](docs/runbooks/rotate-credentials.md).

## Supported versions

Only the latest `v*` release and `main` receive fixes. Model bundles and dataset collections are
versioned independently (`models/PINNED`, `datasets/manifest.json`); a security fix to a weight
file or a dataset is a new bundle/collection version, never an in-place overwrite (blobs are
content-addressed and write-once).
