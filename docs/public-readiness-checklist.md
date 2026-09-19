# Public-readiness checklist

The repository is private (owner decision 1) and is built clean-room so that going public is a
switch. This is the switch. Every line is either **done** (with where the evidence is),
**owner** (only the owner can do it) or **gate** (CI fails until it holds). Flip nothing until
every gate is green and every owner item is signed. Plan sections: "Owner decisions", "Models and
weights: Licences", "Datasets: Licence rule", "Security", "Self-hosted runner: GitHub settings",
"Risks".

## 1. Licences of the code

| Item | State |
|---|---|
| `LICENSE` (Apache-2.0) and `NOTICE` at the root; every crate declares `license = "Apache-2.0"` and `publish = false`; `python/pyproject.toml` `license = "Apache-2.0"` | done (`7c855cb`, `858450d`) |
| Third-party Rust dependencies within the `deny.toml` allow-list (`cargo deny check licenses` in every PR) | gate, green today |
| Python dependencies: `uv run pip-licenses --allow-only "Apache-2.0;MIT;BSD;ISC;PSF"` in the `python` lane | gate, lands with WP8 |
| Vendored `crates/smartcluster/*` (jain-smartcluster @ `715128f8`) carries its upstream licence headers and is named in `NOTICE` | owner: confirm the upstream licence permits redistribution before the switch (it is a private-forge project today) |
| `docs/legacy/` documents mention internal hosts and people | done: excluded from lints; owner: skim for anything not meant to be public (names, customer identifiers) |

## 2. Licences of model weights (`models/bundles/*.json`, PR #9)

`jope models fetch` prints each file's `jain.model-license/v1` block and refuses
`redistribution: prohibited` without `--accept-restricted`. The manifest's licence block is what a
public release audits.

| Bundle | Files | Licence blocks | Public-readiness |
|---|---|---|---|
| `policy-step-60000` | policy + context transforms | `LicenseRef-Jain-Internal-Model` (permitted) | owner: decide the public licence of the trained policy (it is own-trained; the encoders it was trained on are below) |
| `hyperion-v7` | 2 | `LicenseRef-Jain-Internal-Model` (permitted) | owner: same decision |
| `jope-prime-gen31-v2` | 17 | 7 `LicenseRef-Jain-Internal-Model`, 4 `LicenseRef-Jain-Redistributable-Model` (hyperion v5 arm B, target encoder fused v1), **3 `LicenseRef-Jain-ThirdParty-Provenance-Pending` with `redistribution: prohibited`** — `foundation/tabicl-classifier-v2-chimera-shuffled.safetensors`, `foundation/tabicl-regressor-v2-20260212.safetensors`, `starforge/starlight_core_regressor.safetensors` (TabICL / TabPFN-derived) | **gate**: these three stay `restricted/prohibited` until provenance is established; a public bundle may not contain them. Options: (a) resolve provenance and record the upstream licence + evidence URL, (b) publish a bundle variant without them (the foundation encoder is then fetched from its upstream by the user), (c) keep the whole bundle private. Upstream TabPFN `.ckpt` files never enter the repo or a public bundle |
| Parity references, recovery checkpoints, lineage-A warm start | learning/models buckets | internal | stay private; not part of any public bundle |

## 3. Licences of datasets (`datasets/manifest.json`, PR #11)

Rule (enforced by `jope datasets verify --catalog`): `status ∈ {recorded, unknown, restricted}`;
`recorded` needs an SPDX expression or `LicenseRef-*` plus an `evidence_url`; `unknown` is legal
only in private collections; `restricted` never in a public collection; **a public collection with
any non-`recorded` member fails CI**.

| Item | State |
|---|---|
| The ten committed test datasets (`test-10`): all `recorded` with evidence — PMLB ×3 MIT, agridat MIT (registry said GPL; CRAN `DESCRIPTION` corrected it), stevedata GPL-2.0-only, wooldridge GPL-3.0-only, parkinsons / openml-29 / openml-23 CC-BY-4.0, socrata CDC US-Gov public domain; `LICENSE` text committed per dataset | done in PR #11; `test-10` is `visibility: private` until the switch |
| GPL-licensed *data* in a public repository | owner: confirm that redistributing the two GPL Rdatasets CSVs verbatim with their licence text is acceptable (data, not linked code); otherwise swap them for two other rank-1 datasets before the switch |
| `golden42`/`golden46` collections | gate: private until the 24 blank licences are backfilled (WP-D6) |
| `registry-real-v1` (1,416 unknown of 2,613 real datasets) | stays private; backfill is ongoing, not a public-switch blocker because the collection is not public |
| Excluded on purpose: hash-only datasets without provenance, `ZTS_small`, both NOMAD releases (restricted encrypted bundle) | done |
| `meta_sha256_missing` rejects (27,055 synthetic-regression directories) | owner decision, [docs/datasets.md](datasets.md); not a public-switch blocker |

## 4. Secrets and credentials

| Item | State |
|---|---|
| `gh.env` (a GitHub PAT) and `aws.env` (a static AWS access key) were untracked in the historical tree and are out of the new tree; `*.env`, `*.pat`, `*.pem`, `*.key` ignored; `.gitleaks.toml` rule `jope-env-files` | done (ADR-0001 step 0) |
| **Rotate the `gh.env` PAT and the `aws.env` static key** | **owner** — they were exposed on a shared host; rotation has not been receipted. Procedure: [runbooks/rotate-credentials.md](runbooks/rotate-credentials.md). Gate for the switch: a note in the checklist with the rotation date |
| History scan: zero `github_pat_`/`ghp_`/`AKIA`/private-key matches in `git log --all -p`; no `*.env/*.pem/*.key/*.pyc` tracked | done at the first push (ADR-0001 step 5); re-run before the switch on the current history |
| `gitleaks detect --redact` in every PR; `gitleaks protect --staged` in `.githooks/pre-commit` via `just setup` | gate (hygiene), green; the hook lands with the `.githooks` directory |
| Runner secrets: the self-hosted runner's registration PAT is root-0600 on xbabe2, never mounted; no host credentials in containers; AWS only via OIDC roles `JopeDimeCiRead` / `JopeDimeTrainer` / `JopeDimeRelease` bound to exact `sub` claims | done for the roles (WP0/D0); runner provisioning lands with WP8 |
| Static AWS keys anywhere in CI | none; gate: `.github/workflows/*` contain no `AWS_SECRET_ACCESS_KEY` (actionlint + review) |

## 5. GitHub organisation and repository settings

| Item | State |
|---|---|
| Upgrade `veox-ai` to the **Team plan** (branch protection on private repositories, environments, required reviewers, 3,000 hosted minutes) | owner (decision 5), at go-live |
| Branch protection on `main`: required checks `hygiene`, `rust-check`, `emitters`, `python` (then `rust-engine`, `msrv`, `docs`, `parity-matrix`, `test-inventory`, `coverage` as they land); 1 approving review by a different identity; conversation resolution; linear history (squash/rebase only); delete branch on merge; no force-push, no deletion | owner, once on Team; until then `main-guard.yml` (PR #7) files an issue for any commit on `main` without a merged PR |
| `CODEOWNERS`: `* @neverhuman`, `crates/feat-math/ @neverhuman`, `crates/jope-learning/ @neverhuman`, `deploy/ @neverhuman` — enforced by "require review from code owners" | done (file); owner: enable enforcement with protection |
| Actions settings: `sha_pinning_required: true`; `allowed_actions: selected` = GitHub-owned + `Swatinem/rust-cache`, `astral-sh/setup-uv`, `aws-actions/configure-aws-credentials`, `EmbarkStudios/cargo-deny-action`, `taiki-e/install-action`, `anchore/sbom-action`, `actions/attest-build-provenance`; default workflow permissions read-only | owner; every workflow in the tree already pins SHAs and declares `permissions: contents: read` |
| Fork-PR policy: fork pull-request workflows **off** while private; when public, "require approval for all outside collaborators" (first-time and every run), PR lanes stay credential-free and without self-hosted labels so a fork can never reach xbabe2 or S3 | owner at the switch; the workflow design already satisfies it |
| Private vulnerability reporting + security advisories enabled; `SECURITY.md` names the channel | owner at the switch |
| Dependabot (cargo monthly grouped, pip monthly grouped, actions weekly, limit 2, majors excluded) | done |
| Repository description, topics, website; social preview; issue templates | owner, cosmetic |

## 6. Content review before the switch

- `git grep -nE 'git\.neverhuman\.org|127\.0\.0\.1:8787|/home/ubuntu/' -- crates python assets deploy .github` is empty (plan "Verification" 8; `deploy/ci/check.sh` enforces the first two today).
- `docs/legacy/` and `deploy/kernel-v1/source` are the only places allowed to mention historical hosts; skim them once for personal data.
- `NOTICE` names every third-party model and dataset source the public artefacts depend on.
- `README.md` quickstart re-run from a fresh clone on a machine without CUDA (< 15 min budget in the plan; measured 2026-09-18: build 1 m 57 s, feat-math tests 13 m 01 s including compilation, emitters 20 s).
- A `v0.1.0` tag through `release.yml` with attested `jope` binary, wheel, SBOMs and receipt (plan "Verification" 7).

## Sign-off

| Who | What | Date |
|---|---|---|
| owner | credentials rotated (PAT + AWS key) | |
| owner | Team plan, branch protection, actions allow-list, fork policy applied | |
| owner | weight-licence decision for the three provenance-pending files and the internal models | |
| owner | GPL-data decision for the two Rdatasets test sets | |
| maintainer | gates green on `main` at the switch commit (`hygiene`, `rust-check`, `emitters`, `python`, `datasets verify --catalog` public rule) | |
| maintainer | `v0.1.0` release published and attested | |
