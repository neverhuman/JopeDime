# ADR-0001 — History migration: filtered lineage in GitHub, complete history in S3

**Status:** accepted 2026-09-18 · **Plan section:** "History migration (ADR-0001)", owner decision 3,
"Verification" item 1 · **Scope:** the git history of `veox-ai/JopeDime`

## Context

`~/jain-split/jope-dime` was a 293 GB working tree with 118,719 tracked files and a 10.3 GiB
pack, of which the *source* (engine, campaigns, feat-math, grammars, tools, Python) is ≈ 45 MB.
The bulk — `parallel/recovery` (88,998 files), kernel snapshots, frozen release trees, evidence,
the `AGENT_CHAT`/`PR_AGENT_CHAT` wires — is evidence and snapshots that do not belong in a source
repository. The repository-wide history held 105 blobs over 20 MB (24.6 GB), all under evidence
and recovery paths; the history of the kept paths held **none**. Two credential files (`gh.env`,
`aws.env`) sat untracked in the tree. The family couplings (jain-split ledgers, Lane 2 custody,
redline locks, jankurai identity, `hosted-source-authority.sh`, the private forge at
`127.0.0.1:8787`) are explicit non-goals.

## Decision

1. **Preserve first.** `git bundle create --all` of the unfiltered repository (the 2026-09-16
   bundle, 11,094,342,823 B, and a fresh 2026-09-18 bundle), each with its sha256, uploaded to
   `s3://veox-jopedime-archive-use2/git/20260916-…` and `git/20260918-jope-dime-all-refs.bundle`
   under Object Lock. Work happens on a `git clone --no-local --mirror`, never on the checkout.
2. **Filter, do not squash.** `git filter-repo --paths-from-file keep-paths.txt` with a
   `--path-rename` for every move of the target layout (`vendor/feat-math → crates/feat-math`,
   `parallel/src → crates/jope-engine/src`, `parallel/assets/grammar → crates/jope-engine/assets/grammar`,
   `src → crates/jope-campaigns/src`, `manifests|seeds|fixtures → crates/jope-campaigns/…`,
   `parallel/releases/kernel-v1-8aa0affb…/source → deploy/kernel-v1/source`,
   `parallel/assets/source/jope-prime/src/jope → python/src/jope`, `scripts/*.py → python/src/jope/research`,
   `scripts/*.sh → xtools/ops-sh`, `systemd → deploy/systemd/selfplay`,
   `parallel/systemd → deploy/systemd/engine`, `ops/ci → deploy/ci`, root docs → `docs/legacy/root-docs`, …),
   `--strip-blobs-bigger-than 20M`, then an `--invert-paths` pass removing `*.pyc`, `__pycache__`,
   evidence, the ssh/node scripts, `parallel/systemd/xbabe2`, the chat wires and `*.lock` wires.
   Result: **124 commits, 2,079 files, 0 blobs > 20 MB**. Untracked inputs the filter cannot
   carry (`target-export-v1` parity files, the starforge/jain-core/smartcluster imports, the
   recovered exact-cv fixture) arrive as explicit `git add` commits with sha256 receipts.
3. **Refs.** `main` kept; `codex/*`, `engineering/exact-stage-cache-v2`, `four-task-closure`,
   `migration/xbabe2-native-final-20260915`, `phase5-recovery-candidate-8a4214c`,
   `research-untracked-20260914` renamed to `archive/<name>` (seven `archive/*` branches exist in
   the repository); `wip/*`, `local/preserve-*`, `worktree-agent-*`, `workspace-canonical`
   dropped (they live in the bundle). The two gitlinks (`sotarena`, `sotarena-refresh` → the
   private forge) removed; `benchmarks/sotarena` returns as an HTTPS submodule after its 24
   unpushed commits reach `github.com/neverhuman/sotarena` (WP4).
4. **Gate before the first push** (all must hold): zero blobs > 20 MB; `git log --all -p` has zero
   matches for `github_pat_[A-Za-z0-9_]{20,}|ghp_[A-Za-z0-9]{36}|AKIA[0-9A-Z]{16}|BEGIN (RSA|OPENSSH) PRIVATE KEY`;
   `git ls-files` has no `*.env`, `*.pem`, `*.key`, `*.pyc`; every tracked path is under the new
   top-level set. The `git filter-repo --analyze` report and `.git/filter-repo/{commit-map,ref-map}`
   are archived next to the bundle, because provenance documents cite old SHAs (for example
   `07304d6f` in `jope_dime/mod.rs`, `c9513e82` in `docs/ORIGIN.md`).
5. `gh.env` and `aws.env` move out of the tree (`~/.config/gh/`, the AWS profile), `*.env` is in
   `.gitignore` and `.gitleaks.toml`, and **both credentials are rotated** — the rotation is an
   owner action tracked in the [public-readiness checklist](../public-readiness-checklist.md).

## Consequences

- The GitHub repository is small (2,338 tracked files at this commit, no file over 5 MiB — the
  hygiene lane enforces the cap) and clones in seconds; `git blame` still reaches the 2026-07-31
  initial commit of the sources.
- Nothing is lost: any old path, branch or blob is recoverable from the bundle
  (`git clone <bundle>` restores byte-identically; that is the preservation rule of the family and
  it was applied here too), and old SHAs map to new ones through the archived commit-map.
- Historical documents that quote old paths (`parallel/…`, `vendor/feat-math`, `/home/ubuntu/…`)
  are kept verbatim under `docs/legacy/` and are excluded from the leftover lints; anything
  operational was rewritten to the new layout (ADR-0003 for paths).
- The family rituals are gone from the tree by construction, not by policy: no ledgers, no
  `hosted-source-authority.sh`, no forge remote (`deploy/ci/check.sh` fails on
  `git.neverhuman.org` or `127.0.0.1:8787` outside `docs/legacy` and the kernel-v1 snapshot).
- Re-running the filter is cheap (minutes for ~200 commits) if a mistake is found before the
  history is depended upon; after that, corrections are new commits.

## Verification

- History gate items 1–4 above held at the first push (`16b388e` "workspace: one Cargo workspace
  over the filtered lineage" and its predecessors).
- `git rev-list --count --all` = 170 in the hosted repository today; `git ls-files | grep -E '\.(env|pem|key|pyc)$'` is empty.
- The bundles' sha256 equal the `HeadObject` metadata of the archive objects (WP0 acceptance).
- `docs/ORIGIN.md` records the snapshot commit `c9513e8288434a37ceedd4535b38f702bf35ad7f` of the
  source lineage the filter started from.

## Amendments

- 2026-09-28 (clean-base): the kernel-v1 snapshot `deploy/kernel-v1/source` was removed from the
  tree, so the `deploy/ci/check.sh` leftover lint under Consequences no longer excludes it.
  Restore it with the command in
  [docs/archive/clean-base-manifest.json](../archive/clean-base-manifest.json). The decision
  above is unchanged.
