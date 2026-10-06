# Changelog

All notable changes to JopeDime. Generated with git-cliff from the commit history and curated by
pull request. Capability receipts distinguish implemented kernels, admitted profiles and live
qualification.

## [Unreleased]

## [1.0.0] - 2026-10-06

The GP language release: versioned expression kernels, protected numerical semantics, explicit
search profiles and internally generated Python, Rust and C++17 artifacts. C++20 compiles the
C++17 artifact. The capability manifest and each operator's receipt define its admitted routes.

<!-- Local release preparation at ca1f7570a, including merged #221/#223/#224/#225/#226/#227/#228/#229/#230/#231/#232/#233/#234/#236/#238.
Rebase onto final main and finalize the date before publication. GP-language #231 includes the
maintained Tanh mutation gate; audit dispositions are merged and linked below. Final verifier
binding, main qualification and owner dispositions remain prerequisites. B17/B18/B20 wiring is
merged; B19 wiring is tracked separately. -->

### feat-math and emitters

- Freeze historical values, emitted bytes, serialized models, evaluator lineage and generator
  streams; extend boundary and GP conversion witnesses before changing the kernel (#126, #138,
  #143, #144, #151, #152, #177).
- Add runtime/emitted parity and equal-budget search ablation (#127). The ablation harness now
  runs Baseline and Experimental kernel2 searches with explicit bounded refinement controls and
  paired per-dataset results (#221).
- Report scalar and stage fault classes, check scopes and fitted widths, and apply opt-in search
  fault policies. Checked prediction rejects width mismatches before stage execution (#128,
  #130-#133).
- Ship fitted C++17 runtime revision 3 with fitted-state validation, empty-row and ordered-tail
  behavior, and regeneration of stored revision-2 containers (#134).
- Declare revisioned capability manifests and search profiles; validate canonical profile children,
  versioned expression operators, generation witnesses and export admission (#135, #136, #153,
  #154, #166).
- Admit ordered axes, append-only version-2 operator identities, expression recipes and guarded
  zero-width Roll behavior; preserve historical semantics through explicit identity transitions
  (#139-#142, #145, #146, #149, #150).
- Add Box-Cox/Yeo-Johnson expression routes and certified probit, logit, expm1, log1mexp and
  opt-in asinh/stable-reduction kernels. Scalar-v2 execution, ridge models, train/predict recipes
  and fitted C++ export use versioned ingress (#154, #167-#174, #186, #190, #198, #201).
  B05-B07 profile production/admission, EEC and multi-dataset equal-budget utility follow-ups
  remain deferred; these implementations establish no automatic Baseline promotion.
- Add causal finite differences, row Savitzky-Golay filtering, sparse Goertzel banks, cumulative
  integration, irregular-grid backward differences and fixed-five realized path statistics,
  with their route-specific recipes, models and export bounds (#176, #182-#185, #191, #192,
  #196, #203).
- Persist opt-in scalar calibration winners and prove their recipe/export bits; add certified
  ridge scoring and opt-in epsilon-lexicase selection (#178, #188, #189, #207).
- Stage bounded LM constant fitting with a dual-number Jacobian, a versioned loss/complexity
  Pareto archive, bounded one-variable Remez compression and protected-semantics e-graph
  simplification as test-only cores (#195, #197, #200, #202). Production wiring is tracked
  separately from these core receipts.
- Wire opt-in damped least-squares literal refinement into kernel2 with `--refine-method lm`.
  Fit the stage surrogate on training rows, then accept its candidate only when whole-pipeline
  training loss strictly decreases. Retain coordinate refinement as the default and verify
  bounded parent parsing, objective charges and published trace consistency (#227).
- Publish the measured loss/cost Pareto front with opt-in `--pareto-archive`, canonical archive
  bytes and receipt digests. Rebuild the archive from bounded, re-derived observations when
  verifying evidence. The archive leaves ranking, parent selection and the selected program
  unchanged; default sidecar bytes remain pinned (#230).
- Offer deterministic protected-semantics expression rewrites for ranked parents with opt-in
  `--rewrite-simplify`. Admit and measure each new rewritten program within the shared evaluation
  allowance; publish the proposal records for receipt replay and retain profile refusals (#232).
- Route profile-aware pipelines through bounded search and refinement, caller-controlled trial
  limits, fitted C++ export and manifest-bound recipe artifacts. Refined children publish under
  their own observations (#155-#165, #208).

### GP red-team and determinism

- Preserve tree/wire/genome round trips and parameterized recipe conversion bits before emitted
  comparisons; compile numerical ABI boundary witnesses (#143-#145, #151, #152, #177, #207).
- Repair generated C++17 recipes for `-Wall -Werror` and Rust Min/Max and Clip signed-zero tie
  selection. Add the maintained scalar GP determinism and differential-compile gate: repeated
  and cross-process byte checks, Python/Rust/C++17 execution, strict g++ and clang compilation,
  native feature/fault comparisons and signed-zero regressions (#223). Lane A's C++17 warning
  reproduction is credited in that gate's record. Its bounded comparison contract and
  toolchain/platform limits are documented in `docs/parity/gp-emitter-determinism.md`.
- Differential-compile bounded GP math and ML constructions through Python, Rust, g++ and
  clang, with native output-word comparisons and independent reference checks. Retain nested
  expression and recipe conversion witnesses and an independent ridge PRESS proof (#231).
- Retain positive tiny tails in bounded exponential, softmax and log-sum-exp compositions,
  and test bounded RBF constructions against independent Decimal references. Compare native
  output and fault words across all recipe backends while preserving the direct exponential
  guard and existing reference limits (#238).
- Guard the protected-semantics e-graph's libm call shapes, including Tanh, with a maintained
  test that fails when the recorded unauthorized Tanh-fold mutation is applied (#231).

### Engine and S1

- Make S1 teacher selection independent of test scores, cap each attempt, restore two-fold
  teacher search and materialize a fresh G0 (#204, #214).
- Synchronize fit fixture identity with served readiness and make synthetic authority races
  deterministic under neural load (#215, #217).
- Add train-only GPU budgets, output evidence and encoder pins; reject out-of-range CPU affinity
  indices (#199, #222).
- Refuse oversized GPU copies before CUDA dispatch in release builds, retain empty-copy no-ops,
  restrict transfers to sealed scalar representations and make allocation fields private (#226).
- Validate mapped corpus record bounds before reading rows and record the resulting evaluator
  lineage through an explicit identity-only compatibility transition (#229).

### DL lane

- Preserve pinned train-only builder source custody and add non-raw re-split, group-split,
  publisher and publish-tool custody. These source and fixture records retain the train-only
  boundary and do not establish a new live training qualification (#216, #218).
- Record the owner's all-data DL research ruling while retaining the leak boundary (#220).
- Require published JTT1 payloads to omit outer-test blocks and keep outer-test labels in local
  JTS1 scorer artifacts. Refuse unstripped payloads before publication and quarantine old keys
  without deleting objects (#224).
- Add metadata-only S1 GP HWM publication with pinned family, cut and program lineage,
  rights-specific manifests, create-only writes, exact-version readbacks and resumable delivery.
  Omit outer-test data and scores; refuse output traversal before client or state creation.
  Local and fake-store qualification remains distinct from live S3 persistence (#236).

### CI, build and release

- Import frozen standalone teacher-v2 search/worker workspaces and their Python custody record
  (#129). Refresh only the `feat-math` version entry in affected Lambda working locks for 1.0.0,
  recording their current tree hashes while retaining the original frozen snapshot hashes.
- Bind runner provenance to the PR base, pin swap limits in systemd, validate base/digest SHAs,
  and bound shallow-lock recovery and fetches; document post-merge swap-pin deployment (#205,
  #206, #210, #211, #213).
- Fix manifest hash comparison and record an identity-only lineage transition (#212).
- Repair inherited Clippy findings in receiving libraries, alongside earlier behavior-preserving
  lint fixes (#209, #219).
- Remove the disabled duplicate final-CV loop and retain its positional input under an unused
  parameter name so strict warning checks pass. Active scoring and retained tests are unchanged
  (#233).
- Stabilize executor preparation, daemon polling, lost-ACK and durable execution fixtures while
  preserving work limits and retained deadlines (#175, #181, #187, #194).
- Stabilize durable recovery and cancellation fixtures without extending signed work limits;
  permit synthetic teaching tests to select a validated persistent fixture directory (#225).
- Bind SmartCluster participant tests to their own duplicated control descriptors and verify
  inherited regular-file lock refusal without releasing or changing the parent's lock (#228).
  Production authentication and runner behavior retain their existing contracts.
- Set CI evidence retention to seven days and align product workspace and Python metadata at
  1.0.0; vendored SmartCluster versions and frozen source/history pins remain unchanged (#179).

### Documentation

- Publish the [complete Jankurai finding register and release rationale](docs/release/jankurai-v1.md)
  for the pinned `2e0b7723` audit: 544 findings, 468 hard and 76 soft. Its score remains 47/85,
  with a failing audit exit; the differing auditor binary prevents treating the changed finding
  total as a source-health improvement (#234).
- Retain the documented unsafe-site and handle-threading proof gaps, uncapped legacy admission
  qualification, and executable-artifact loading limitations. These dispositions establish no
  clean audit, complete GPU safety, complete W1–W12 remediation, or qualified live campaign.
- Record operator-specific numerical, profile, recipe and export receipts and evaluator-identity
  transitions with their remaining admission and utility requirements.
- Document GP tree coverage and emitter determinism in `docs/parity/gp-language-tricks.md` and
  `docs/parity/gp-emitter-determinism.md`; these records bind the release's final qualification.
- Update release preparation, version consistency and the README's 1.0.0 source-build guidance.

### Breaking changes and compatibility

- Rust callers of `Model::predict`, `FittedHead::predict`, `PortfolioModel::predict`,
  `apply_op_train` and `apply_op` must handle `Result<_, WidthMismatch>` instead of the former
  bare `Vec<f64>` or `Rows` return. Propagate or handle the width error explicitly (#146).
- GPU callers use the `DeviceBuffer` pointer accessor rather than public allocation fields;
  transfers accept the sealed supported scalar types (#226).
- Profile-aware searches and scalar-v2 programs/models require their explicit versioned
  envelopes and selected capability manifests. Legacy parsing and profiles keep their documented
  historical route; newly implemented kernels are admitted per profile.
- Fitted C++ runtime revision 3 validates fitted widths/state; loading revision-2 containers
  regenerates current source. Use checked prediction where a width mismatch must return an error.
- Evaluator-identity changes carry lineage transition records. Compare or resume learning only
  under a compatible comparison identity; a successful conversion alone is not a new utility claim.
- B05-B07 private profile/EEC/utility work remains post-1.0. Final-main CPU, compatibility,
  red-team and audit receipts must qualify the candidate before tag publication.

## [0.2.0] - 2026-09-29

The first tagged release. Generated with git-cliff 2.14.1 and `cliff.toml` over the commits after
PR #90 (`7f2b2bf86`), then curated by pull request: #91 to #125.
Earlier history is not itemised here; see `git log`.

### feat-math and emitters

- hygiene: `jope emit --lang cpp20` through the pipeline's C++ source call, a 20-worker default cap for `jope-bench run`, and status text that cannot go stale (#95)

### Engine

- engine: project training corpora and refuse training contexts that were not admitted (#93)
- policy: native challenger export and binding for matched searches (#97)
- engine: an export never leaves unreceipted weights or partial files, and resolves `/tmp` (#98)
- engine: refuse dangling-link ancestors and stale temporaries before any write (#100)
- engine, policy, pipeline, bench, records: regression remediation W0 and every source-implementable slice of W1 to W12 (#111)
- preflight: a native, read-only host census for pinned plans (#113)
- engine, policy, pipeline, bench, records: harden the regression provenance, input, export and progress receiving paths (the W1 to W12 gap closure) (#114)
- pipeline: targeted single-dataset search in jope pipeline (#118)
- native-loop: the bounded multi-host native training loop (W7), qualified on source and fixtures: a pinned host and device census, versioned run plans and participant grants, authenticated cross-host transport with pinned certificate identity and an absolute deadline on every read and write, controller recovery, and SmartCluster dispatch of bounded ranks that prove their job before any work (#120)
- native-loop: W7's host-identity tests are hermetic: the host is observed through an explicit provider and the fixtures use a fixture host, so full-main-cpu no longer depends on the runner's `/etc/machine-id` (#122)

### Campaigns, self-play and learning

- bench: search rounds, per-arm incumbents and deferred held-out scoring (#94)
- motifs: the `jope-motifs` crate and a hidden `jope motifs` verb over the qualified learning State (#96)
- learning: condition pointer publication on observed versions, and record the multi-node training audit (#99)
- learning: encoder receipts and capability-bound teaching bindings (contract spike) (#101)
- training: validate distributed startup and prove two- and three-rank collectives (#105)
- campaigns: restore the campaign policy audit after the workspace migration (#106)
- campaigns, runner: restore campaign Elo parity against the maintained arena reference, and bind runner admission to the transferred repository name and its immutable identity (#107)

### Crates, models and datasets

- datasets: admit the forty deterministic synthetic fixtures (D5); fixture consumers read one verified snapshot and report unsupported fixtures (#121)

### Python

- python: port the CI and engine Python to Rust (`jope-ci cargo-artifact` and the build report, the `jope-records` hygiene guards, native family growth) and archive 703 files with no live consumer; ADR-0006 P1 (#117)

### CI, build and release

- ci: retain the full Python fixture evidence in both CI workflows without dereferencing links (#108)
- health: route the host-data and CPU fit-service contracts to their proof lanes, so full-main-cpu can pass (#116)
- clean base, part A: remove dead leftovers, retire the inactive `ops/ci` copies, pin `actions/checkout` to v7.0.1, stop feat-math's scratch-directory leaks and archive the stash records (#119)
- clean base, part B: retire the frozen kernel-v1 and `docs/legacy` trees (with an archived restore manifest), name `neverhuman/JopeDime` as the workspace repository, make the `jope` binary and its direct crates the workspace `default-members`, pin `actions/checkout` in the weekly workflow, and add a two-way test-routing check and the repository-layout guard to the hygiene lane; the guard reads every tracked Cargo manifest as TOML 1.0 and refuses, with no exceptions, a manifest it cannot read (`bad-manifest`) and a tracked path that is not UTF-8 (`bad-path`) (#123)
- release: `deploy/release/release.sh` cuts a release on xbabe2 from a verified `git archive` export, built in the digest-pinned `rust:1.97.1-bookworm` image (glibc 2.36), with a build receipt, SBOMs, a release manifest and a cosign signature over `SHA256SUMS` (owner decision D-D); syft and cosign are pinned by sha256 and reported version, host tools are looked up only in a fixed, read-only system `PATH` and must be root-owned files reached only through root-owned directories and symlinks, and no step writes under `/tmp`, in the builder containers and root's signing shell included; `deploy/release/test-release.sh` runs in the hygiene lane (#124)
- docs(release): `docs/release/README.md` explains how to verify a release, and `docs/release/signing.pub` is the release public key; hygiene refuses any tracked private-key block (#124)
- ci(release): `release.yml` is dispatch-only and fails closed at its first step; releases come from `release.sh` (#124)
- release: version 0.2.0 for the workspace and every product crate; `jope` prints `jope 0.2.0` (#124)
- lambda: import the serverless workers and terminal monitor (frozen snapshot) as standalone Cargo workspaces under `deploy/lambda`, outside the `jope` build, built and tested in the hygiene lane; their Python stays in custody with a tracked hash manifest (#124)
- lambda: import the regression worker and its offline assessment tool (frozen snapshot) as two more standalone Cargo workspaces under `deploy/lambda`, their locks at the 0.2.0 `feat-math`, built and tested in the hygiene lane; their Python stays in custody with the same tracked hash manifest (#125)

### Documentation

- docs: rebuild the learning pipeline manual and its evidence volumes (#91)
- docs: administrator AWS access and the restricted credentials still pending (#92)
- docs: an internal technical reference, and tooling for an auditable CUDA pipeline timing study with fitted export receipts (#104)
- records: preserve the regression audit findings and verify complete source coverage (#109)
- docs: detailed W1 to W12 regression remediation work orders (#110)
- docs: record the #111 landing (#112)
- docs: record the #114 landing (#115)
- docs: this changelog; the repository-layout guard allows `CHANGELOG.md` at the root (#124)
- docs(records): the v0.2.0 completion record: the #124 and #125 landings, the owner's decisions of 2026-09-29 and the regression Lambda run's closure (#125)

### Dependencies

- build(deps): cc 1.4.5 to 1.5.1, clap 4.6.6 to 4.6.7, crc32fast 1.5.1 to 1.5.2 (PR #5's intent, with the MSRV-aware resolver) (#124)
- build(deps): taiki-e/install-action 2.87.15 to 2.87.18 (#102)
- build(deps): aws-actions/configure-aws-credentials 4.0.2 to 6.3.0 (#103)

### Deferred to the next milestone

- regression trials v2, trimmed to its first live qualification, which now comes before the three-host demonstration (owner decisions O2 and O3 in `MASTER_PLAN.md`)
- the live three-host S3 training demonstration and the P6-0 throughput measurement
- the self-hosted `release.yml` lane (a bookworm release runner image and release-domain admission)
- Python removal P2 and P3 (ADR-0006), which also retires the wheel
- the open issues #47 (an audit of the secret-scan findings on custody refs, and a scheduled scan of every retained ref), #48 (the C++ and JavaScript emit binding) and #62 (live S3 from the command line), and W7's live cross-host checkpoint exchange and continuation routing (#120 landed their source and fixture foundation)
- the README's generated arena block (rendered by `jope-records readme-benchmarks` from a sha256-pinned corrections file) and `docs/learning.md` still cite Python files that #117 archived (Python removal P2)
- untracking `AGENT_CHAT.md`

<!-- generated by git-cliff -->
