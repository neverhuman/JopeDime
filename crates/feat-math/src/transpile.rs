//! battle-transpile — deterministic program → standalone-code emitters.
//!
//! Given a `GenomeIr` (the internal program representation), emits (1) a
//! single-file dependency-free Python 3 script and (2) a single-file std-only
//! Rust program, each exposing a Model with fit/predict and a main() that reads
//! `train.csv test_features.csv` and writes predictions. Emitted code must be
//! bit-identical in behavior to battle-runtime, human-readable, documented, and
//! free of any vocabulary that reveals how the program was found
//! ([`lint_shipped_text`]).
//!
//! PUBLIC API CONTRACT — battle-judge and battle-cli build against these
//! signatures; keep them stable.

pub use crate::GenomeIr;
use anyhow::{bail, Result};
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::time::Instant;

#[path = "emit_cpp.rs"]
mod emit_cpp;
#[path = "emit_js.rs"]
mod emit_js;

/// Bumped whenever emitted artifacts change byte-wise for the same genome.
/// v2: proper `class Model` / `struct Model` with fit/predict/score, a single
/// concise module docstring (no per-stage repetition), an honest `step`
/// indicator (no `*1e15` trick), and a single-buffer assembly path.
/// v2.1: composite-kernel Gaussian process (`gpr kernel="..."`), the
/// Nadaraya-Watson `ksmooth` head, and the deep-boosting `histgbm2` knobs
/// (rounds / row+column subsampling / min_leaf / holdout split / Huber /
/// MAE stopping) — histgbm2 defaults stay numerically bit-identical to v2.
/// The `(target kind=...)` pseudo-stage (normal-scores / shifted-log /
/// winsorized target transforms) was added without a bump: such genomes
/// previously failed to emit, and artifacts for every genome without a
/// target stage are byte-identical.
/// The self-tuning `ridge_gcv` head was likewise added without a bump:
/// genomes using it previously failed to emit, and artifacts for every
/// genome without a ridge_gcv head are byte-identical.
/// The compact `portfolio` head (v35 G3/G4) was also added without a bump:
/// such genomes previously failed to emit, and artifacts for every genome
/// without a portfolio head are byte-identical (verified against the
/// pre-G3 binary on the invariance fixture set).
/// v2.2: the `(sym affine ...)` docstring no longer cites the method
/// literature (its bytes change; every pipeline artifact is byte-identical
/// to v2.1), every emitter is linted with [`lint_shipped_text`], and the
/// manifest carries `program` / `program_sha256` beside the legacy `dsl` /
/// `genome_sha256` keys (readers switch, then the legacy keys go).
/// Bound Python JSON quotation has a no-bump repair for control/Unicode text that
/// previously emitted invalid Python (Rust Debug escapes such as `\\u{2028}`).
/// Printable ASCII bound output and all frozen recipe emitter bytes are unchanged.
/// The separate fitted-portable/1 public emitter pins its own template identities.
pub const EMITTER_VERSION: &str = "2.2.0";

/// Which transcendental math implementation an emission request selects.
///
/// - `PlatformLibm` (DEFAULT): transcendentals lower to the host math library
///   (`math.exp`, `f64::exp`, `std::exp`, `Math.exp`). Bit-parity between the
///   in-process interpreter and the emitted Python/Rust holds because they
///   share the same platform `libm` — but that parity does NOT extend to
///   JavaScript or CUDA, whose engine/device libm differ. V1's frozen artifact
///   bytes and receipts are all `PlatformLibm`; never change this default for
///   them.
/// - `Portable`: reserves the hand-rolled bit-reproducible `portable_math`
///   kernel identity (`portable-scalar-kernel-v1`) for the V2 tape contract.
///   Until every emitted transcendental is routed through that kernel, emit
///   fails closed instead of silently returning PlatformLibm bytes under the
///   portable identity. It is opt-in and is never used for V1.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum MathKernel {
    #[default]
    PlatformLibm,
    Portable,
}

impl MathKernel {
    /// Stable identity string for receipts / manifests.
    pub fn policy_id(self) -> &'static str {
        match self {
            MathKernel::PlatformLibm => "platform-libm-v1",
            MathKernel::Portable => "portable-scalar-kernel-v1",
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct EmitOptions {
    /// Human title woven into the module docstring (e.g. from meta.json source).
    pub title: Option<String>,
    /// Original column headers for the docstring (never used as identifiers).
    pub feature_names: Vec<String>,
    /// Extra context sentence for docs (dataset description etc.).
    pub context: Option<String>,
    /// Opt-in fast-runtime profile: the Rust artifact gains a row-parallel
    /// predict wrapper plus an optimized build recommendation, and the Python
    /// artifact a PyPy note. Predictions stay bit-identical to the default
    /// profile; artifact BYTES differ, so receipt-pinned emissions must keep
    /// the default (false).
    pub fast_runtime: bool,
    /// Requested transcendental math implementation. Default `PlatformLibm`
    /// keeps V1's frozen bytes. `Portable` is a distinct opt-in identity and
    /// fails closed until the emitted-language routing gate is complete.
    pub math_kernel: MathKernel,
    /// Versioned opt-in recipe profile. Programs containing Roll refuse an
    /// empty feature row before fitting or predicting. The legacy default
    /// keeps its source bytes for already pinned artifacts.
    pub guard_zero_width_roll: bool,
    /// Versioned opt-in recipe and bound-Python profile. Recipe programs
    /// containing `LowerBound` or `UpperBound` helpers (`QuantileBins`,
    /// `BinOnehot`, `TargetBinEncode`, `Rank`, `Blom`) and bound Python
    /// artifacts (`advanced_v3::emit::emit_bound`) emit the exact `(base, size)`
    /// `slice::partition_point` binary search instead of the legacy default
    /// helpers. The legacy default (`false`) keeps its source bytes and
    /// `emitter_version` labels for already pinned artifacts.
    pub partition_point_quantile_bounds: bool,
}

#[derive(Clone, Debug, Serialize)]
pub struct TokenCount {
    pub code_tokens: usize,
    pub doc_tokens: usize,
    pub bytes: usize,
}

#[derive(Clone, Debug, Serialize)]
pub struct EmitTiming {
    /// Pure string-assembly time, no I/O.
    pub emit_ns: u128,
    pub py_tokens: usize,
    pub rs_tokens: usize,
    pub tokens_per_sec: f64,
}

#[derive(Clone, Debug)]
pub struct Emitted {
    pub python: String,
    pub rust: String,
    /// Sidecar manifest JSON (genome round-trip lives here, never in the code).
    pub manifest: String,
    pub py_tokens: TokenCount,
    pub rs_tokens: TokenCount,
    pub timing: EmitTiming,
}

/// Additive identity for the independently qualified C++17 rung.
#[derive(Clone, Debug)]
pub struct Cpp17Emitted {
    pub source: String,
    pub source_sha256: String,
    pub tokens: TokenCount,
    pub math_policy: &'static str,
}

/// Additive identity for the independently qualified ES2020 rung.
#[derive(Clone, Debug)]
pub struct JavaScriptEmitted {
    pub source: String,
    pub source_sha256: String,
    pub tokens: TokenCount,
    pub math_policy: &'static str,
}

/// The sidecar manifest. `program` / `program_sha256` are the names; `dsl` /
/// `genome_sha256` are kept for the readers that still use them and are
/// dropped once every reader has switched. The manifest is the program's
/// round-trip record, so its `program` value is never linted; its keys and
/// every other value are ([`lint_shipped_text`] in the tests).
#[derive(Serialize)]
struct Manifest<'a> {
    schema: &'a str,
    program: String,
    program_sha256: String,
    dsl: String,
    genome_sha256: String,
    flavor: &'a str,
    emitter_version: &'a str,
    py_sha256: String,
    rs_sha256: String,
}

fn sha256_hex(data: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(data.as_bytes());
    let digest = hasher.finalize();
    let mut out = String::with_capacity(64);
    for byte in digest {
        out.push_str(&format!("{byte:02x}"));
    }
    out
}

pub(crate) fn validate_math_kernel(opts: &EmitOptions) -> Result<()> {
    if opts.math_kernel == MathKernel::Portable {
        bail!(
            "portable math emission is not available until every emitted transcendental is routed; use the default PlatformLibm policy"
        );
    }
    Ok(())
}

/// Emit both artifacts. Deterministic: same genome + options => byte-identical output.
pub fn emit(genome: &GenomeIr, opts: &EmitOptions) -> Result<Emitted> {
    validate_math_kernel(opts)?;
    let plan = match crate::spec::build_plan(genome) {
        Ok(plan) => plan,
        Err(missing) => bail!("unsupported program elements: {}", missing.join("; ")),
    };
    let start = Instant::now();
    let python = crate::emit_py::emit_python(&plan, opts);
    let rust = crate::emit_rs::emit_rust(&plan, opts);
    let emit_ns = start.elapsed().as_nanos().max(1);
    let is_portfolio = matches!(
        &plan,
        crate::spec::Plan::Pipeline {
            head: crate::spec::PlannedHead::Portfolio(_),
            ..
        }
    );
    for (lang, src) in [("python", python.as_str()), ("rust", rust.as_str())] {
        let mut hits = if is_portfolio {
            lint_vocabulary_portfolio(src)
        } else {
            lint_vocabulary(src)
        };
        hits.extend(lint_shipped_phrases(src));
        if !hits.is_empty() {
            bail!("banned vocabulary in {lang} artifact: {}", hits.join(", "));
        }
    }
    let py_tokens = crate::tokens::count_tokens(&python, crate::tokens::Lang::Python);
    let rs_tokens = crate::tokens::count_tokens(&rust, crate::tokens::Lang::Rust);
    let py_total = py_tokens.code_tokens + py_tokens.doc_tokens;
    let rs_total = rs_tokens.code_tokens + rs_tokens.doc_tokens;
    let tokens_per_sec = (py_total + rs_total) as f64 / (emit_ns as f64 / 1e9);
    let flavor = match genome {
        GenomeIr::Pipeline(_) => "pipeline",
        GenomeIr::Symbolic { .. } => "symbolic",
        // Unreachable today: build_plan rejects the explicit nested form
        // (emission lands in G5); kept for the manifest schema's sake.
        GenomeIr::Portfolio(_) => "portfolio",
    };
    let canonical = genome.canonical();
    let helpers = crate::spec::helper_set(&plan);
    let has_roll_guard = opts.guard_zero_width_roll
        && matches!(&plan, crate::spec::Plan::Pipeline { stages, .. }
            if stages.iter().any(|stage| matches!(stage.op, crate::spec::PlannedOp::Roll { .. })));
    let has_quantile_partition_point = opts.partition_point_quantile_bounds
        && helpers.iter().any(|h| {
            matches!(
                h,
                crate::spec::Helper::LowerBound | crate::spec::Helper::UpperBound
            )
        });
    let has_clip_safety = helpers.contains(&crate::spec::Helper::MathClip);
    let emitter_version = match (
        has_roll_guard,
        has_quantile_partition_point,
        has_clip_safety,
    ) {
        (true, true, true) => {
            "feat-math-roll-width-guard-v1+quantile-partition-point-v1+clip-safety-v2"
        }
        (true, true, false) => "feat-math-roll-width-guard-v1+quantile-partition-point-v1",
        (false, true, true) => "feat-math-quantile-partition-point-v1+clip-safety-v2",
        (false, true, false) => "feat-math-quantile-partition-point-v1",
        (true, false, _) => "feat-math-roll-width-guard-v1",
        (false, false, true) => "feat-math-clip-safety-v2",
        (false, false, false) => EMITTER_VERSION,
    };
    let manifest = serde_json::to_string(&Manifest {
        schema: "battle-artifact-v1",
        program: canonical.clone(),
        program_sha256: sha256_hex(&canonical),
        genome_sha256: sha256_hex(&canonical),
        dsl: canonical,
        flavor,
        emitter_version,
        py_sha256: sha256_hex(&python),
        rs_sha256: sha256_hex(&rust),
    })?;
    Ok(Emitted {
        python,
        rust,
        manifest,
        py_tokens,
        rs_tokens,
        timing: EmitTiming {
            emit_ns,
            py_tokens: py_total,
            rs_tokens: rs_total,
            tokens_per_sec,
        },
    })
}

/// Emit a complete native C17 model API bundle backed by the full Rust recipe.
pub fn emit_c17(genome: &GenomeIr, opts: &EmitOptions) -> Result<crate::c17::Emitted> {
    crate::c17::emit(genome, opts)
}

/// C17 uses exactly the Rust recipe coverage, without a second arithmetic backend.
pub fn supports_c17(genome: &GenomeIr) -> std::result::Result<(), Vec<String>> {
    supports(genome)
}

/// Check emitter coverage; Err lists unsupported ops (judge rejects such claims).
pub fn supports(genome: &GenomeIr) -> std::result::Result<(), Vec<String>> {
    match crate::spec::build_plan(genome) {
        Ok(_) => Ok(()),
        Err(missing) => Err(missing),
    }
}

/// Emit the independently qualified standalone C++17 training-recipe artifact.
///
/// This is additive to [`emit`]: the existing Rust/Python sources and manifest
/// remain byte-identical. The C++ rung deliberately reports its narrower
/// certified coverage instead of silently producing a different backend.
pub fn emit_cpp17(genome: &GenomeIr, opts: &EmitOptions) -> Result<String> {
    let plan = match crate::spec::build_plan(genome) {
        Ok(plan) => plan,
        Err(missing) => bail!("unsupported program elements: {}", missing.join("; ")),
    };
    let source = match emit_cpp::emit_cpp(&plan, opts) {
        Ok(source) => source,
        Err(unsupported) => bail!(
            "C++17 emitter does not yet support: {}",
            unsupported.join(", ")
        ),
    };
    let hits = lint_shipped_text(&source);
    if !hits.is_empty() {
        bail!("banned vocabulary in C++17 artifact: {}", hits.join(", "));
    }
    Ok(source)
}

/// Emit C++17 with its source, arithmetic, and exact token identities bound.
/// Counting the completed source is self-calibrating by construction: emitter
/// growth needs no hand-maintained token constants.
pub fn emit_cpp17_counted(genome: &GenomeIr, opts: &EmitOptions) -> Result<Cpp17Emitted> {
    let source = emit_cpp17(genome, opts)?;
    Ok(Cpp17Emitted {
        source_sha256: sha256_hex(&source),
        tokens: crate::tokens::count_tokens(&source, crate::tokens::Lang::Cpp),
        math_policy: opts.math_kernel.policy_id(),
        source,
    })
}

/// Check the independently certified C++17 emitter coverage.
pub fn supports_cpp17(genome: &GenomeIr) -> std::result::Result<(), Vec<String>> {
    let plan = crate::spec::build_plan(genome)?;
    emit_cpp::emit_cpp(&plan, &EmitOptions::default()).map(|_| ())
}

/// Emit the independently qualified standalone ES2020 training-recipe artifact.
///
/// This API is additive and intentionally separate from [`emit`], preserving
/// every frozen Rust/Python artifact and manifest byte.
pub fn emit_javascript(genome: &GenomeIr, opts: &EmitOptions) -> Result<String> {
    let plan = match crate::spec::build_plan(genome) {
        Ok(plan) => plan,
        Err(missing) => bail!("unsupported program elements: {}", missing.join("; ")),
    };
    let source = match emit_js::emit_js(&plan, opts) {
        Ok(source) => source,
        Err(unsupported) => bail!(
            "JavaScript emitter does not yet support: {}",
            unsupported.join(", ")
        ),
    };
    let hits = lint_shipped_text(&source);
    if !hits.is_empty() {
        bail!(
            "banned vocabulary in JavaScript artifact: {}",
            hits.join(", ")
        );
    }
    Ok(source)
}

/// Emit ES2020 with its source, arithmetic, and exact token identities bound.
pub fn emit_javascript_counted(genome: &GenomeIr, opts: &EmitOptions) -> Result<JavaScriptEmitted> {
    let source = emit_javascript(genome, opts)?;
    Ok(JavaScriptEmitted {
        source_sha256: sha256_hex(&source),
        tokens: crate::tokens::count_tokens(&source, crate::tokens::Lang::JavaScript),
        math_policy: opts.math_kernel.policy_id(),
        source,
    })
}

/// Check the independently certified ES2020 emitter coverage.
pub fn supports_javascript(genome: &GenomeIr) -> std::result::Result<(), Vec<String>> {
    let plan = crate::spec::build_plan(genome)?;
    emit_js::emit_js(&plan, &EmitOptions::default()).map(|_| ())
}

/// TypeScript facade over the standalone Rust kernel.
pub fn emit_typescript(
    genome: &GenomeIr,
    opts: &EmitOptions,
) -> Result<crate::typescript::TypeScriptEmitted> {
    crate::typescript::emit(genome, opts)
}

pub fn supports_typescript(genome: &GenomeIr) -> std::result::Result<(), Vec<String>> {
    supports(genome)
}

/// Emit a typed Go API and its complete WebAssembly build source.
pub fn emit_go(genome: &GenomeIr, opts: &EmitOptions) -> Result<crate::go::GoEmitted> {
    crate::go::emit(genome, opts)
}

/// Go shares the full Rust recipe coverage.
pub fn supports_go(genome: &GenomeIr) -> std::result::Result<(), Vec<String>> {
    supports(genome)
}

/// Banned-vocabulary lint run on every emitted artifact (emission fails on
/// hit). Extended for v35 with `nnls` and the portfolio member registry
/// identifiers — none of which any artifact (old or new) ever contained.
pub fn lint_vocabulary(src: &str) -> Vec<String> {
    let banned = [
        "genome",
        "genetic",
        "gp_",
        "evolut",
        "mutat",
        "crossover",
        "fitness",
        "candidate",
        "chromosome",
        "genotype",
        "phenotype",
        "allele",
        "population",
        "dsl",
        "zyal",
        "jmath",
        "newmath",
        "jmir",
        "oracle",
        "elo",
        "lexicase",
        "transpile",
        "/home/",
        "/tmp/",
        "http://",
        "https://",
        "nnls",
    ];
    let lower = src.to_lowercase();
    let mut hits: Vec<String> = banned
        .iter()
        .filter(|b| lower.contains(**b))
        .map(|b| b.to_string())
        .collect();
    // Member registry identifiers (spec §6.10) — data-driven from members.rs
    // so new registry versions are covered automatically.
    for def in &crate::members::CORE_V1 {
        if lower.contains(def.name) {
            hits.push(def.name.to_string());
        }
    }
    if lower.contains("core_v1") {
        hits.push("core_v1".to_string());
    }
    hits
}

/// Extended lint for portfolio artifacts (spec §6.10): additionally bans the
/// bare substring `oof`. It cannot join the global list because the legacy
/// `stack` head's emitted snippets use `oof` as a variable name and existing
/// artifacts must stay byte-identical; portfolio emissions say "held_out".
pub fn lint_vocabulary_portfolio(src: &str) -> Vec<String> {
    let mut hits = lint_vocabulary(src);
    if src.to_lowercase().contains("oof") {
        hits.push("oof".to_string());
    }
    hits
}

/// Phrases that reveal how a program was found and must never appear in
/// anything that ships: emitted code in every language, manifests, receipts,
/// CLI help. They extend [`lint_vocabulary`] (which bans the older single
/// words) at the phrase level, so ordinary machine-learning words keep their
/// ordinary meaning: "tree" in "boosted trees" is fine, "expression tree" is
/// not. Matched case-insensitively as substrings.
pub const SHIPPED_TEXT_BANNED: &[&str] = &[
    "keijzer",
    "koza",
    "evolv",
    "genetic prog",
    "gp ",
    "gp-",
    "generation",
    "expression tree",
    "parse tree",
    "syntax tree",
    "symbolic regression",
    "tree-based program",
    "island",
    "tournament",
    "elitism",
    "elites",
];

/// The phrase-level half of [`lint_shipped_text`].
pub fn lint_shipped_phrases(src: &str) -> Vec<String> {
    let lower = src.to_lowercase();
    SHIPPED_TEXT_BANNED
        .iter()
        .filter(|b| lower.contains(**b))
        .map(|b| b.to_string())
        .collect()
}

/// The lint every shipped artifact must pass: [`lint_vocabulary`] plus
/// [`SHIPPED_TEXT_BANNED`]. Returns every hit, empty when clean.
pub fn lint_shipped_text(src: &str) -> Vec<String> {
    let mut hits = lint_vocabulary(src);
    hits.extend(lint_shipped_phrases(src));
    hits
}

fn validate_recipe_target(
    genome: &GenomeIr,
    target: crate::artifact_manifest::Target,
) -> Result<()> {
    use crate::artifact_manifest::Target;
    // Go, TypeScript and C17 reuse the Rust numerical implementation and its admission.
    if Target::RECIPES.contains(&target)
        || matches!(target, Target::Go | Target::TypeScript | Target::C17)
    {
        if let GenomeIr::Pipeline(pipe) = genome {
            for stage in &pipe.stages {
                let row = crate::kernel::registry_v2::operator("legacy_recipe", &stage.op)
                    .ok_or_else(|| anyhow::anyhow!("unregistered legacy operation {}", stage.op))?;
                row.supports(target.registry_target())
                    .map_err(anyhow::Error::msg)?;
            }
        }
    }
    Ok(())
}

/// Opt-in target-complete manifest. Historical `emit` and its defaults stay unchanged.
/// Unsupported targets remain explicit; successful rendering binds exact source bytes.
pub fn emit_targets_v3(
    genome: &GenomeIr,
    options: &EmitOptions,
    requested: &[crate::artifact_manifest::Target],
) -> Result<crate::artifact_manifest::TargetEmission> {
    use crate::artifact_manifest::{ArtifactManifestV3, Target, TargetEmission, TargetRecord};
    validate_math_kernel(options)?;
    // Validate the independent request set before invoking an emitter.
    anyhow::ensure!(!requested.is_empty(), "empty requested target set");
    anyhow::ensure!(
        requested
            .iter()
            .collect::<std::collections::BTreeSet<_>>()
            .len()
            == requested.len(),
        "duplicate requested target"
    );
    let mut sources = std::collections::BTreeMap::new();
    let mut records = Vec::new();
    for &target in requested {
        if target == Target::C17 {
            let result =
                validate_recipe_target(genome, target).and_then(|()| emit_c17(genome, options));
            match result {
                Ok(bundle) => {
                    let inputs: Vec<_> = bundle
                        .files
                        .iter()
                        .map(|(p, s)| (p.as_str(), s.as_str()))
                        .collect();
                    records.push(TargetRecord::supported(
                        target,
                        "legacy_recipe",
                        crate::c17::EMITTER,
                        &inputs,
                        None,
                    ));
                    sources.extend(bundle.files);
                }
                Err(error) => records.push(TargetRecord::unsupported(target, format!("{error:#}"))),
            }
            continue;
        }
        if target == Target::TypeScript {
            let result = validate_recipe_target(genome, target)
                .and_then(|()| emit_typescript(genome, options));
            match result {
                Ok(bundle) => {
                    records.push(TargetRecord::supported(
                        target,
                        "legacy_recipe",
                        crate::typescript::EMITTER_VERSION,
                        &[
                            ("model.ts", &bundle.source),
                            ("model.kernel.rs", &bundle.kernel),
                        ],
                        None,
                    ));
                    sources.insert("model.ts".into(), bundle.source);
                    sources.insert("model.kernel.rs".into(), bundle.kernel);
                }
                Err(error) => records.push(TargetRecord::unsupported(target, format!("{error:#}"))),
            }
            continue;
        }
        if target == Target::Go {
            let result =
                validate_recipe_target(genome, target).and_then(|()| emit_go(genome, options));
            match result {
                Ok(bundle) => {
                    records.push(TargetRecord::supported(
                        target,
                        "legacy_recipe",
                        crate::go::EMITTER_VERSION,
                        &crate::go::sources(&bundle),
                        None,
                    ));
                    for (path, source) in crate::go::sources(&bundle) {
                        sources.insert(path.into(), source.into());
                    }
                }
                Err(error) => records.push(TargetRecord::unsupported(target, format!("{error:#}"))),
            }
            continue;
        }
        let result: Result<(String, String)> = (|| {
            validate_recipe_target(genome, target)?;
            Ok(match target {
                Target::Python | Target::Rust => {
                    let pair = emit(genome, options)?;
                    let identity: serde_json::Value = serde_json::from_str(&pair.manifest)?;
                    let version = identity["emitter_version"]
                        .as_str()
                        .unwrap_or(EMITTER_VERSION)
                        .to_owned();
                    (
                        if target == Target::Python {
                            pair.python
                        } else {
                            pair.rust
                        },
                        version,
                    )
                }
                Target::Cpp17 | Target::Cpp20 => (
                    emit_cpp17(genome, options)?,
                    format!("feat-math-cpp17/{EMITTER_VERSION}"),
                ),
                Target::JavaScript => (
                    emit_javascript(genome, options)?,
                    format!("feat-math-javascript/{EMITTER_VERSION}"),
                ),
                Target::Cpp17RecipeV2 => {
                    let e = crate::kernel::expression_recipe::emit_cpp17(genome, options)
                        .map_err(anyhow::Error::msg)?;
                    (e.source, e.profile.into())
                }
                Target::C17 => unreachable!("multi-file target handled above"),
                Target::TypeScript => unreachable!("handled as a complete bundle"),
                Target::Go => unreachable!("handled as a complete bundle"),
                Target::Cpp17Fitted => {
                    anyhow::bail!("fitted target requires explicit training rows")
                }
            })
        })();
        match result {
            Ok((source, emitter)) => {
                records.push(TargetRecord::supported(
                    target,
                    "legacy_recipe",
                    &emitter,
                    &[(target.path(), &source)],
                    None,
                ));
                sources.insert(target.path().into(), source);
            }
            Err(error) => records.push(TargetRecord::unsupported(target, format!("{error:#}"))),
        }
    }
    let manifest = ArtifactManifestV3::new(
        &genome.canonical(),
        "legacy_recipe",
        options,
        requested,
        records,
    )?;
    Ok(TargetEmission { manifest, sources })
}

#[cfg(test)]
mod math_policy_tests {
    use super::*;
    use std::fmt::Write as _;
    use std::fs;
    use std::process::Command;

    fn fixture() -> GenomeIr {
        GenomeIr::parse("(pipe X (robust_scale) (ridge))").expect("fixture parses")
    }

    #[test]
    fn explicit_platform_policy_preserves_default_artifact_bytes() {
        let genome = fixture();
        let default = emit(&genome, &EmitOptions::default()).expect("default emits");
        let explicit = emit(
            &genome,
            &EmitOptions {
                math_kernel: MathKernel::PlatformLibm,
                ..EmitOptions::default()
            },
        )
        .expect("explicit PlatformLibm emits");

        assert_eq!(default.python, explicit.python);
        assert_eq!(default.rust, explicit.rust);
        assert_eq!(default.manifest, explicit.manifest);
    }

    #[test]
    fn portable_policy_never_silently_emits_platform_math() {
        let options = EmitOptions {
            math_kernel: MathKernel::Portable,
            ..EmitOptions::default()
        };
        let error = emit(&fixture(), &options)
            .expect_err("portable emission must fail closed until every call is routed");

        assert!(error.to_string().contains("portable"));
        assert!(crate::tokens::try_estimate_tokens(&fixture(), &options).is_err());
    }

    #[test]
    fn cpp17_numeric_abi_is_bit_exact_and_has_no_data_access() {
        let genome = GenomeIr::parse("(pipe X (standardize) (ridge))").expect("fixture parses");
        let source = emit_cpp17(&genome, &EmitOptions::default()).expect("C++17 emits");
        for forbidden in [
            "fstream",
            "ifstream",
            "ofstream",
            "fopen(",
            "filesystem",
            "getenv(",
            "system(",
            "fprintf(",
            "std::exit(",
            "std::abort(",
            "train.csv",
            "test_features.csv",
        ] {
            assert!(
                !source.contains(forbidden),
                "emitted C++17 contains forbidden data-access marker {forbidden}"
            );
        }
        assert!(source.contains("model_fit_numeric"));
        assert!(source.contains("model_predict_numeric_batch"));

        let train = vec![
            vec![-2.0, 0.5],
            vec![-1.0, -0.25],
            vec![0.0, 1.5],
            vec![0.75, -2.0],
            vec![1.5, 0.25],
            vec![3.0, 2.0],
        ];
        let targets = vec![-4.25, -1.375, -0.75, 3.5, 2.875, 5.0];
        let test = vec![vec![-0.5, 0.0], vec![1.25, -1.0], vec![2.5, 3.0]];
        let expected = crate::model::fit(&genome, &train, &targets)
            .expect("Rust fit")
            .predict(&test)
            .unwrap();

        let mut driver = source;
        driver.push_str("\n#include <cstdio>\nint main() {\n");
        driver.push_str("  const double train[] = {");
        for row in &train {
            for value in row {
                write!(driver, "{value:?},").unwrap();
            }
        }
        driver.push_str("};\n  const double targets[] = {");
        for value in &targets {
            write!(driver, "{value:?},").unwrap();
        }
        driver.push_str("};\n  const double test[] = {");
        for row in &test {
            for value in row {
                write!(driver, "{value:?},").unwrap();
            }
        }
        driver.push_str(
            "};\n  double out[3] = {}; model_handle* model = nullptr;\n\
             if (model_fit_numeric(train, 11, 6, 2, targets, 6, &model) != 1 || model != nullptr) return 7;\n\
             if (model_fit_numeric(train, 12, 6, 2, targets, 5, &model) != 1 || model != nullptr) return 8;\n\
             if (model_fit_numeric(train, std::numeric_limits<std::size_t>::max(), std::numeric_limits<std::size_t>::max(), 2, targets, std::numeric_limits<std::size_t>::max(), &model) != 1 || model != nullptr) return 9;\n\
             if (model_fit_numeric(train, 12, 6, 2, targets, 6, &model) != 0) return 10;\n\
             if (model_predict_numeric_batch(model, test, 5, 3, 2, nullptr, 0, out, 3) != 1) return 11;\n\
             if (model_predict_numeric_batch(model, test, std::numeric_limits<std::size_t>::max(), std::numeric_limits<std::size_t>::max(), 2, nullptr, 0, out, std::numeric_limits<std::size_t>::max()) != 1) return 12;\n\
             if (model_predict_numeric_batch(model, test, 6, 3, 2, nullptr, 0, out, 3) != 0) return 13;\n\
             if (model_predict_numeric_batch(model, nullptr, 0, 0, 2, nullptr, 0, nullptr, 0) != 0) return 14;\n\
             for (double value : out) { std::uint64_t bits; std::memcpy(&bits, &value, 8); std::printf(\"%016llx\\n\", static_cast<unsigned long long>(bits)); }\n\
             model_unload(model); return 0; }\n",
        );

        let scratch = tempfile::Builder::new()
            .prefix("feat-math-cpp17-")
            .tempdir()
            .expect("create C++17 test directory");
        let dir = scratch.path();
        let source_path = dir.join("model.cpp");
        fs::write(&source_path, driver).expect("write generated C++17 source");
        let expected = expected
            .iter()
            .map(|value| format!("{:016x}", value.to_bits()))
            .collect::<Vec<_>>()
            .join("\n")
            + "\n";
        let profiles = [
            ("normal", vec!["-O2", "-Wall", "-Wextra", "-Werror"]),
            (
                "ubsan",
                vec![
                    "-O1",
                    "-fsanitize=undefined",
                    "-fno-sanitize-recover=undefined",
                ],
            ),
            (
                "asan",
                vec!["-O1", "-fsanitize=address", "-fno-omit-frame-pointer"],
            ),
        ];
        for (profile, extra_flags) in profiles {
            let binary_path = dir.join(format!("model-{profile}"));
            let compile = crate::cxx::command(None)
                .args(["-ffp-contract=off", "-fno-fast-math"])
                .args(extra_flags)
                .arg(&source_path)
                .arg("-o")
                .arg(&binary_path)
                .output()
                .expect("g++ is required for the C++17 gate");
            assert!(
                compile.status.success(),
                "g++ {profile} failed: {}",
                String::from_utf8_lossy(&compile.stderr)
            );
            let run = Command::new(&binary_path)
                .env("ASAN_OPTIONS", "detect_leaks=1:halt_on_error=1")
                .output()
                .expect("run C++17 artifact");
            assert!(
                run.status.success(),
                "C++17 {profile} artifact failed: {}",
                String::from_utf8_lossy(&run.stderr)
            );
            let actual = String::from_utf8(run.stdout).expect("hex output is UTF-8");
            assert_eq!(actual, expected, "C++17 {profile} prediction bits");
        }
    }

    #[test]
    fn cpp17_coverage_and_tokens_fail_closed() {
        let unsupported = GenomeIr::parse("(sym raw \"x\")").expect("symbolic parses");
        assert!(supports_cpp17(&unsupported).is_err());
        let portable = emit_cpp17(
            &fixture(),
            &EmitOptions {
                math_kernel: MathKernel::Portable,
                ..EmitOptions::default()
            },
        )
        .expect("portable C++17 emits");
        assert!(portable.contains("static double xexp(double x)"));
        assert!(portable.contains("static double xln(double x)"));
        assert!(!portable.contains("std::exp("));
        assert!(!portable.contains("std::log("));
        let first = emit_cpp17_counted(&fixture(), &EmitOptions::default()).unwrap();
        let second = emit_cpp17_counted(&fixture(), &EmitOptions::default()).unwrap();
        assert_eq!(first.source, second.source);
        assert_eq!(first.source_sha256, second.source_sha256);
        assert_eq!(first.tokens.code_tokens, second.tokens.code_tokens);
        assert_eq!(first.tokens.doc_tokens, second.tokens.doc_tokens);
        assert_eq!(first.tokens.bytes, first.source.len());
        assert_eq!(first.math_policy, "platform-libm-v1");
        let direct = crate::tokens::count_tokens(&first.source, crate::tokens::Lang::Cpp);
        assert_eq!(first.tokens.code_tokens, direct.code_tokens);
        assert_eq!(first.tokens.doc_tokens, direct.doc_tokens);
        assert_eq!(first.tokens.bytes, direct.bytes);
        let count = crate::tokens::count_tokens(
            "// two words\nextern \"C\" { int f(); }",
            crate::tokens::Lang::Cpp,
        );
        assert_eq!(count.doc_tokens, 2);
        assert!(count.code_tokens > 0);
    }

    #[test]
    fn cpp17_portable_body_and_shared_symbols_are_audited() {
        assert!(emit_cpp::portable_math_matches_verified_fixture());

        let genome = GenomeIr::parse("(pipe X (standardize) (ridge))").expect("fixture parses");
        let source = emit_cpp17(&genome, &EmitOptions::default()).expect("C++17 emits");
        let scratch = tempfile::Builder::new()
            .prefix("feat-math-cpp17-symbols-")
            .tempdir()
            .expect("create C++17 symbol-audit directory");
        let dir = scratch.path();
        let source_path = dir.join("model.cpp");
        let library_path = dir.join("libmodel.so");
        fs::write(&source_path, source).expect("write generated C++17 source");
        let compile = crate::cxx::command(None)
            .args([
                "-O2",
                "-ffp-contract=off",
                "-fno-fast-math",
                "-Wall",
                "-Wextra",
                "-Werror",
                "-fPIC",
                "-shared",
            ])
            .arg(&source_path)
            .arg("-o")
            .arg(&library_path)
            .output()
            .expect("g++ is required for the C++17 symbol audit");
        assert!(
            compile.status.success(),
            "g++ shared build failed: {}",
            String::from_utf8_lossy(&compile.stderr)
        );
        let nm = Command::new("nm")
            .args(["-u"])
            .arg(&library_path)
            .output()
            .expect("nm is required for the C++17 symbol audit");
        assert!(nm.status.success(), "nm failed");
        let forbidden = [
            "fopen", "fopen64", "open", "open64", "read", "pread", "pread64", "getenv", "system",
            "popen", "connect", "socket", "fprintf", "printf", "exit", "_exit", "abort",
        ];
        for line in String::from_utf8_lossy(&nm.stdout).lines() {
            let symbol = line
                .split_whitespace()
                .last()
                .unwrap_or("")
                .split('@')
                .next()
                .unwrap_or("");
            assert!(
                !forbidden.contains(&symbol),
                "emitted C++17 imports forbidden external symbol {symbol}"
            );
        }
    }

    #[test]
    fn cpp17_claimed_pipeline_corpus_matches_rust_bits() {
        let transforms = [
            "standardize",
            "robust_scale",
            "rank",
            "blom",
            "sigmoid",
            "step t=0.25",
            "winsor_clip q=0.1",
            "norm_row",
            "rff_cos n=4 gamma=0.7 seed=5",
            "rff_sin n=4 gamma=0.7 seed=5",
            "cheb degree=3",
            "hinge knots=3",
            "poly_interact",
            "quantile_bins bins=4",
            "bin_onehot bins=4",
            "pair_prod topk=3",
            "rand_proj k=4 seed=7",
            "row_stats",
            "row_diff_stats",
            "oblique_hinge n=4 seed=3",
            "landmark_dist k=4 gamma=0.7 seed=5",
            "target_bin_encode bins=4 prior=2",
        ];
        let heads = ["ridge", "elasticnet alpha=0.01 l1_ratio=0.3 sweeps=20"];
        let cases = transforms
            .iter()
            .flat_map(|transform| {
                heads
                    .iter()
                    .map(move |head| format!("(pipe X ({transform}) ({head}))"))
            })
            .collect::<Vec<_>>();
        let train = vec![
            vec![-2.0, 0.5, 1.25],
            vec![-1.0, -0.25, 2.0],
            vec![0.0, 1.5, -1.0],
            vec![0.75, -2.0, 0.25],
            vec![1.5, 0.25, -0.5],
            vec![3.0, 2.0, 1.5],
            vec![-0.75, 0.75, 3.0],
            vec![2.25, -1.5, -2.0],
        ];
        let targets = vec![-4.25, -1.375, -0.75, 3.5, 2.875, 5.0, -2.0, 4.25];
        let test = vec![
            vec![-0.5, 0.0, 1.0],
            vec![1.25, -1.0, 0.5],
            vec![2.5, 3.0, -1.5],
        ];
        let scratch = tempfile::Builder::new()
            .prefix("feat-math-cpp17-corpus-")
            .tempdir()
            .expect("create C++17 corpus directory");
        let dir = scratch.path();

        for (case_index, dsl) in cases.iter().enumerate() {
            let genome = GenomeIr::parse(dsl).expect("corpus fixture parses");
            let source = emit_cpp17(&genome, &EmitOptions::default()).expect("C++17 emits");
            let expected = crate::model::fit(&genome, &train, &targets)
                .expect("Rust fit")
                .predict(&test)
                .unwrap();
            let mut driver = source;
            driver.push_str("\n#include <cstdio>\nint main() {\n");
            driver.push_str("  const double train[] = {");
            for row in &train {
                for value in row {
                    write!(driver, "{value:?},").unwrap();
                }
            }
            driver.push_str("};\n  const double targets[] = {");
            for value in &targets {
                write!(driver, "{value:?},").unwrap();
            }
            driver.push_str("};\n  const double test[] = {");
            for row in &test {
                for value in row {
                    write!(driver, "{value:?},").unwrap();
                }
            }
            driver.push_str(
                "};\n  double out[3] = {}; model_handle* model = nullptr;\n\
                 if (model_fit_numeric(train, 24, 8, 3, targets, 8, &model) != 0) return 10;\n\
                 if (model_predict_numeric_batch(model, test, 9, 3, 3, nullptr, 0, out, 3) != 0) return 11;\n\
                 for (double value : out) { std::uint64_t bits; std::memcpy(&bits, &value, 8); std::printf(\"%016llx\\n\", static_cast<unsigned long long>(bits)); }\n\
                 model_unload(model); return 0; }\n",
            );
            let source_path = dir.join(format!("model-{case_index}.cpp"));
            let binary_path = dir.join(format!("model-{case_index}"));
            fs::write(&source_path, driver).expect("write generated C++17 corpus source");
            let compile = crate::cxx::command(None)
                .args([
                    "-O2",
                    "-ffp-contract=off",
                    "-fno-fast-math",
                    "-Wall",
                    "-Wextra",
                    "-Werror",
                ])
                .arg(&source_path)
                .arg("-o")
                .arg(&binary_path)
                .output()
                .expect("g++ is required for the C++17 corpus gate");
            assert!(
                compile.status.success(),
                "g++ failed for {dsl}: {}",
                String::from_utf8_lossy(&compile.stderr)
            );
            let run = Command::new(&binary_path)
                .output()
                .expect("run C++17 corpus artifact");
            assert!(
                run.status.success(),
                "C++17 failed for {dsl}: {}",
                String::from_utf8_lossy(&run.stderr)
            );
            let actual = String::from_utf8(run.stdout).expect("hex output is UTF-8");
            let expected = expected
                .iter()
                .map(|value| format!("{:016x}", value.to_bits()))
                .collect::<Vec<_>>()
                .join("\n")
                + "\n";
            assert_eq!(actual, expected, "C++17 prediction bits for {dsl}");
        }
    }

    #[test]
    fn cpp17_portable_fitted_sigmoid_matches_rust_reference_bits() {
        fn portable_sigmoid(value: f64) -> f64 {
            let clamped = value.clamp(-30.0, 30.0);
            1.0 / (1.0 + crate::portable_math::exp(-clamped))
        }

        let train = vec![
            vec![-20.0, 0.5],
            vec![-2.0, -0.25],
            vec![-0.0, 1.5],
            vec![0.75, -2.0],
            vec![3.0, 0.25],
            vec![20.0, 2.0],
        ];
        let targets = vec![-4.25, -1.375, -0.75, 3.5, 2.875, 5.0];
        let test = vec![vec![-30.0, 0.0], vec![1.25, -1.0], vec![30.0, 3.0]];
        let portable_train = train
            .iter()
            .map(|row| {
                row.iter()
                    .copied()
                    .map(portable_sigmoid)
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let portable_test = test
            .iter()
            .map(|row| {
                row.iter()
                    .copied()
                    .map(portable_sigmoid)
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let ridge = GenomeIr::parse("(pipe X (ridge))").expect("ridge parses");
        let expected = crate::model::fit(&ridge, &portable_train, &targets)
            .expect("portable-reference Rust fit")
            .predict(&portable_test)
            .unwrap();
        let sigmoid = GenomeIr::parse("(pipe X (sigmoid) (ridge))").expect("sigmoid parses");
        let mut driver = emit_cpp17(
            &sigmoid,
            &EmitOptions {
                math_kernel: MathKernel::Portable,
                ..EmitOptions::default()
            },
        )
        .expect("Portable C++17 emits");
        assert!(!driver.contains("std::exp("));
        assert!(!driver.contains("std::log("));
        assert!(!driver.contains("std::sin("));
        assert!(!driver.contains("std::cos("));
        driver.push_str("\n#include <cstdio>\nint main() {\n  const double train[] = {");
        for row in &train {
            for value in row {
                write!(driver, "{value:?},").unwrap();
            }
        }
        driver.push_str("};\n  const double targets[] = {");
        for value in &targets {
            write!(driver, "{value:?},").unwrap();
        }
        driver.push_str("};\n  const double test[] = {");
        for row in &test {
            for value in row {
                write!(driver, "{value:?},").unwrap();
            }
        }
        driver.push_str(
            "};\n  double out[3] = {}; model_handle* model = nullptr;\n\
             if (model_fit_numeric(train, 12, 6, 2, targets, 6, &model) != 0) return 10;\n\
             if (model_predict_numeric_batch(model, test, 6, 3, 2, nullptr, 0, out, 3) != 0) return 11;\n\
             for (double value : out) { std::uint64_t bits; std::memcpy(&bits, &value, 8); std::printf(\"%016llx\\n\", static_cast<unsigned long long>(bits)); }\n\
             model_unload(model); return 0; }\n",
        );
        let scratch = tempfile::Builder::new()
            .prefix("feat-math-cpp17-portable-model-")
            .tempdir()
            .expect("create Portable C++17 directory");
        let dir = scratch.path();
        let source_path = dir.join("model.cpp");
        let binary_path = dir.join("model");
        fs::write(&source_path, driver).expect("write Portable C++17 source");
        let compile = crate::cxx::command(None)
            .args([
                "-O2",
                "-ffp-contract=off",
                "-fno-fast-math",
                "-Wall",
                "-Wextra",
                "-Werror",
            ])
            .arg(&source_path)
            .arg("-o")
            .arg(&binary_path)
            .output()
            .expect("g++ is required for Portable C++17 parity");
        assert!(
            compile.status.success(),
            "g++ Portable build failed: {}",
            String::from_utf8_lossy(&compile.stderr)
        );
        let run = Command::new(&binary_path)
            .output()
            .expect("run Portable C++17 artifact");
        assert!(
            run.status.success(),
            "Portable C++17 failed: {}",
            String::from_utf8_lossy(&run.stderr)
        );
        let actual = String::from_utf8(run.stdout).expect("hex output is UTF-8");
        let expected = expected
            .iter()
            .map(|value| format!("{:016x}", value.to_bits()))
            .collect::<Vec<_>>()
            .join("\n")
            + "\n";
        assert_eq!(actual, expected, "Portable fitted prediction bits");
    }

    #[test]
    fn cpp17_declared_pipeline_corpus_is_bit_exact() {
        let train = (0..32)
            .map(|row| {
                (0..5)
                    .map(|column| {
                        let centered = (row * 17 + column * 11) as f64 / 31.0 - 9.0;
                        centered + ((row + column * 3) % 7) as f64 * 0.125
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let targets = train
            .iter()
            .enumerate()
            .map(|(row, values)| {
                1.7 * values[0] - 0.9 * values[1]
                    + 0.15 * values[2] * values[3]
                    + (row % 3) as f64 * 0.25
            })
            .collect::<Vec<_>>();
        let test = train[3..10].to_vec();
        let cases = [
            "(pipe X (standardize) (ridge))",
            "(pipe X (robust_scale) (ridge))",
            "(pipe X (rank) (ridge))",
            "(pipe X (blom) (ridge))",
            "(pipe X (sigmoid append=true) (ridge))",
            "(pipe X (step t=0.0 append=true) (ridge))",
            "(pipe X (winsor_clip q=0.05) (ridge))",
            "(pipe X (norm_row) (ridge))",
            "(pipe X (rff_cos n=8 gamma=0.5 seed=9 append=true) (ridge))",
            "(pipe X (rff_sin n=8 gamma=0.5 seed=9 append=true) (ridge))",
            "(pipe X (cheb degree=4) (ridge))",
            "(pipe X (hinge knots=3) (ridge))",
            "(pipe X (poly_interact) (ridge))",
            "(pipe X (quantile_bins bins=4) (ridge))",
            "(pipe X (bin_onehot bins=4) (ridge))",
            "(pipe X (pair_prod topk=3) (ridge))",
            "(pipe X (rand_proj k=6 seed=5) (ridge))",
            "(pipe X (row_stats) (ridge))",
            "(pipe X (row_diff_stats) (ridge))",
            "(pipe X (oblique_hinge n=6 seed=6 append=true) (ridge))",
            "(pipe X (landmark_dist k=6 gamma=0.5 seed=3) (ridge))",
            "(pipe X (target_bin_encode bins=4 prior=5.0) (ridge))",
            "(pipe X (elasticnet alpha=0.01 l1_ratio=0.5 sweeps=30))",
        ];
        let scratch = tempfile::Builder::new()
            .prefix("feat-math-cpp17-corpus-")
            .tempdir()
            .expect("create C++17 corpus directory");
        let dir = scratch.path();

        for (case_index, genome_source) in cases.iter().enumerate() {
            let genome = GenomeIr::parse(genome_source).expect("corpus genome parses");
            supports_cpp17(&genome).unwrap_or_else(|missing| {
                panic!("C++17 coverage rejected {genome_source}: {missing:?}")
            });
            let expected = crate::model::fit(&genome, &train, &targets)
                .expect("Rust corpus fit")
                .predict(&test)
                .unwrap();
            let mut driver =
                emit_cpp17(&genome, &EmitOptions::default()).expect("C++17 corpus emits");
            driver.push_str("\n#include <cstdio>\nint main() {\n  const double train[] = {");
            for row in &train {
                for value in row {
                    write!(driver, "{value:?},").unwrap();
                }
            }
            driver.push_str("};\n  const double targets[] = {");
            for value in &targets {
                write!(driver, "{value:?},").unwrap();
            }
            driver.push_str("};\n  const double test[] = {");
            for row in &test {
                for value in row {
                    write!(driver, "{value:?},").unwrap();
                }
            }
            write!(
                driver,
                "}};\n  double out[{}] = {{}}; model_handle* model = nullptr;\n\
                 if (model_fit_numeric(train, {}, {}, 5, targets, {}, &model) != 0) return 10;\n\
                 if (model_predict_numeric_batch(model, test, {}, {}, 5, nullptr, 0, out, {}) != 0) return 11;\n\
                 for (double value : out) {{ std::uint64_t bits; std::memcpy(&bits, &value, 8); std::printf(\"%016llx\\n\", static_cast<unsigned long long>(bits)); }}\n\
                 model_unload(model); return 0; }}\n",
                test.len(),
                train.len() * 5,
                train.len(),
                targets.len(),
                test.len() * 5,
                test.len(),
                test.len()
            )
            .unwrap();
            let source_path = dir.join(format!("case-{case_index}.cpp"));
            let binary_path = dir.join(format!("case-{case_index}"));
            fs::write(&source_path, driver).expect("write C++17 corpus source");
            let compile = crate::cxx::command(None)
                .args(["-O2", "-ffp-contract=off", "-fno-fast-math"])
                .arg(&source_path)
                .arg("-o")
                .arg(&binary_path)
                .output()
                .expect("g++ is required for the C++17 corpus");
            assert!(
                compile.status.success(),
                "g++ failed for {genome_source}: {}",
                String::from_utf8_lossy(&compile.stderr)
            );
            let run = Command::new(&binary_path)
                .output()
                .expect("run C++17 corpus artifact");
            assert!(
                run.status.success(),
                "C++17 failed for {genome_source}: {}",
                String::from_utf8_lossy(&run.stderr)
            );
            let actual = String::from_utf8(run.stdout).expect("hex output is UTF-8");
            let expected = expected
                .iter()
                .map(|value| format!("{:016x}", value.to_bits()))
                .collect::<Vec<_>>()
                .join("\n")
                + "\n";
            assert_eq!(actual, expected, "C++17 parity for {genome_source}");
        }
    }

    fn run_javascript_fixture(
        genome: &GenomeIr,
        options: &EmitOptions,
        train: &[Vec<f64>],
        targets: &[f64],
        test: &[Vec<f64>],
    ) -> String {
        let mut driver = emit_javascript(genome, options).expect("JavaScript emits");
        driver.push_str("\nconst train = new Float64Array([");
        for row in train {
            for value in row {
                write!(driver, "{value:?},").unwrap();
            }
        }
        driver.push_str("]);\nconst targets = new Float64Array([");
        for value in targets {
            write!(driver, "{value:?},").unwrap();
        }
        driver.push_str("]);\nconst test = new Float64Array([");
        for row in test {
            for value in row {
                write!(driver, "{value:?},").unwrap();
            }
        }
        let columns = train.first().map_or(0, Vec::len);
        write!(
            driver,
            "]);\nconst fitted = modelFitNumeric(train, train.length, {}, {}, targets, targets.length);\n\
             if (fitted.status !== 0 || fitted.handle === null) process.exit(10);\n\
             const out = new Float64Array({});\n\
             if (modelPredictNumericBatch(fitted.handle, test, test.length, {}, {}, null, 0, out, out.length) !== 0) process.exit(11);\n\
             const bitsBuffer = new ArrayBuffer(8); const bitsF64 = new Float64Array(bitsBuffer); const bitsU64 = new BigUint64Array(bitsBuffer);\n\
             for (const value of out) {{ bitsF64[0] = value; console.log(bitsU64[0].toString(16).padStart(16, '0')); }}\n",
            train.len(),
            columns,
            test.len(),
            test.len(),
            columns
        )
        .unwrap();
        let scratch = tempfile::Builder::new()
            .prefix("feat-math-javascript-fixture-")
            .tempdir()
            .expect("create JavaScript fixture directory");
        let dir = scratch.path();
        let source_path = dir.join("model.js");
        fs::write(&source_path, driver).expect("write emitted JavaScript source");
        let syntax = Command::new("node")
            .arg("--check")
            .arg(&source_path)
            .output()
            .expect("stock Node is required for JavaScript syntax validation");
        assert!(
            syntax.status.success(),
            "Node syntax check failed: {}",
            String::from_utf8_lossy(&syntax.stderr)
        );
        let run = Command::new("node")
            .arg(&source_path)
            .output()
            .expect("run emitted JavaScript model");
        assert!(
            run.status.success(),
            "JavaScript model failed: {}",
            String::from_utf8_lossy(&run.stderr)
        );
        String::from_utf8(run.stdout).expect("hex output is UTF-8")
    }

    fn run_cpp17_fixture(
        case_index: usize,
        genome: &GenomeIr,
        options: &EmitOptions,
        train: &[Vec<f64>],
        targets: &[f64],
        test: &[Vec<f64>],
    ) -> String {
        let mut driver = emit_cpp17(genome, options).expect("C++17 emits");
        driver.push_str("\n#include <cstdio>\nint main(){\n  const double train[]={");
        for row in train {
            for value in row {
                write!(driver, "{value:?},").unwrap();
            }
        }
        driver.push_str("};\n  const double targets[]={");
        for value in targets {
            write!(driver, "{value:?},").unwrap();
        }
        driver.push_str("};\n  const double test[]={");
        for row in test {
            for value in row {
                write!(driver, "{value:?},").unwrap();
            }
        }
        let columns = train.first().map_or(0, Vec::len);
        write!(
            driver,
            "}};\n  double out[{}]={{}}; model_handle* model=nullptr;\n\
             if(model_fit_numeric(train,{}, {}, {},targets,{},&model)!=0)return 10;\n\
             if(model_predict_numeric_batch(model,test,{}, {}, {},nullptr,0,out,{})!=0)return 11;\n\
             for(double value:out){{std::uint64_t bits;std::memcpy(&bits,&value,8);std::printf(\"%016llx\\n\",static_cast<unsigned long long>(bits));}}\n\
             model_unload(model);return 0;}}\n",
            test.len(),
            train.len() * columns,
            train.len(),
            columns,
            targets.len(),
            test.len() * columns,
            test.len(),
            columns,
            test.len()
        )
        .unwrap();
        let scratch = tempfile::Builder::new()
            .prefix("feat-math-javascript-cpp-parity-")
            .tempdir()
            .expect("create JavaScript/C++ parity directory");
        let dir = scratch.path();
        let source_path = dir.join(format!("case-{case_index}.cpp"));
        let binary_path = dir.join(format!("case-{case_index}"));
        fs::write(&source_path, driver).expect("write generated C++17 parity source");
        let compile = crate::cxx::command(None)
            .args(["-O2", "-ffp-contract=off", "-fno-fast-math"])
            .arg(&source_path)
            .arg("-o")
            .arg(&binary_path)
            .output()
            .expect("g++ is required for JavaScript/C++ parity");
        assert!(
            compile.status.success(),
            "g++ parity build failed: {}",
            String::from_utf8_lossy(&compile.stderr)
        );
        let run = Command::new(&binary_path)
            .output()
            .expect("run C++17 parity artifact");
        assert!(
            run.status.success(),
            "C++17 parity artifact failed: {}",
            String::from_utf8_lossy(&run.stderr)
        );
        String::from_utf8(run.stdout).expect("C++17 hex output is UTF-8")
    }

    #[test]
    fn javascript_numeric_abi_is_bit_exact_and_has_no_data_access() {
        let genome = GenomeIr::parse("(pipe X (standardize) (ridge))").expect("fixture parses");
        let first = emit_javascript_counted(&genome, &EmitOptions::default()).unwrap();
        let second = emit_javascript_counted(&genome, &EmitOptions::default()).unwrap();
        assert_eq!(first.source, second.source);
        assert_eq!(first.source_sha256, second.source_sha256);
        assert_eq!(first.tokens.code_tokens, second.tokens.code_tokens);
        assert_eq!(first.tokens.doc_tokens, second.tokens.doc_tokens);
        assert_eq!(first.tokens.bytes, first.source.len());
        assert_eq!(first.math_policy, "platform-libm-v1");
        assert!(first.source.contains("function modelFitNumeric("));
        assert!(first.source.contains("function modelPredictNumericBatch("));
        for forbidden in [
            "require(",
            "import ",
            "process.",
            "fetch(",
            "XMLHttpRequest",
            "WebSocket",
            "Deno.",
            "Bun.",
            "train.csv",
            "test_features.csv",
            "Math.fround",
        ] {
            assert!(
                !first.source.contains(forbidden),
                "emitted JavaScript contains forbidden marker {forbidden}"
            );
        }

        let train = vec![
            vec![-2.0, 0.5],
            vec![-1.0, -0.25],
            vec![0.0, 1.5],
            vec![0.75, -2.0],
            vec![1.5, 0.25],
            vec![3.0, 2.0],
        ];
        let targets = vec![-4.25, -1.375, -0.75, 3.5, 2.875, 5.0];
        let test = vec![vec![-0.5, 0.0], vec![1.25, -1.0], vec![2.5, 3.0]];
        let expected = crate::model::fit(&genome, &train, &targets)
            .expect("Rust fit")
            .predict(&test)
            .unwrap()
            .iter()
            .map(|value| format!("{:016x}", value.to_bits()))
            .collect::<Vec<_>>()
            .join("\n")
            + "\n";
        let actual =
            run_javascript_fixture(&genome, &EmitOptions::default(), &train, &targets, &test);
        assert_eq!(actual, expected, "JavaScript prediction bits");
    }

    #[test]
    fn javascript_coverage_and_portable_body_fail_closed() {
        assert!(emit_js::portable_math_matches_verified_fixture());
        let unsupported = GenomeIr::parse("(sym raw \"x\")").expect("symbolic parses");
        assert!(supports_javascript(&unsupported).is_err());
        let portable = emit_javascript(
            &fixture(),
            &EmitOptions {
                math_kernel: MathKernel::Portable,
                ..EmitOptions::default()
            },
        )
        .expect("Portable JavaScript emits");
        assert!(portable.contains("function xexp(x)"));
        assert!(portable.contains("function xln(x)"));
        assert!(!portable.contains("function grid()"));
        assert!(!portable.contains("function mExp(x){return Math.exp(x);}"));
        assert!(!portable.contains("function mLog(x){return Math.log(x);}"));
    }

    #[test]
    fn javascript_typed_boundaries_and_pcg_vectors_are_pinned() {
        let mut source = emit_javascript(&fixture(), &EmitOptions::default()).unwrap();
        source.push_str(
            r#"
function expect(condition, code){ if(!condition) process.exit(code); }
const values=new Float64Array([1.0,2.0,3.0,4.0]);
const targets=new Float64Array([1.0,2.0]);
expect(modelFitNumeric([1.0,2.0,3.0,4.0],4,2,2,targets,2).status===1,10);
expect(modelFitNumeric(values,3,2,2,targets,2).status===1,11);
expect(modelFitNumeric(values,4,2,2,new Float64Array([1.0]),1).status===1,12);
expect(modelFitNumeric(values,4,Number.MAX_SAFE_INTEGER,2,targets,2).status===1,13);
expect(modelFitNumeric(values,4,2,2,new Float64Array([1.0,NaN]),2).status===1,14);
const fitted=modelFitNumeric(values,4,2,2,targets,2);
expect(fitted.status===0&&fitted.handle!==null,15);
expect(modelPredictNumericBatch(fitted.handle,[5.0,6.0],2,1,2,null,0,new Float64Array(1),1)===1,16);
expect(modelPredictNumericBatch(fitted.handle,new Float64Array([5.0]),1,1,2,null,0,new Float64Array(1),1)===1,17);
expect(modelPredictNumericBatch(fitted.handle,new Float64Array([5.0,6.0]),2,1,2,null,0,new Float64Array(2),2)===1,18);
expect(modelPredictNumericBatch(fitted.handle,new Float64Array(0),0,0,2,null,0,new Float64Array(0),0)===0,19);
const rng=new Pcg32(42n,54n);
const expected=[0xa15c02b7,0x7b47f409,0xba1d3330,0x83d2f293,0xbfa4784b,0xcbed606e];
for(let i=0;i<expected.length;i++)expect(rng.nextU32()===expected[i],20+i);
modelUnload(fitted.handle);
"#,
        );
        let scratch = tempfile::Builder::new()
            .prefix("feat-math-javascript-boundary-")
            .tempdir()
            .expect("create JavaScript boundary directory");
        let dir = scratch.path();
        let source_path = dir.join("boundary.js");
        fs::write(&source_path, source).expect("write JavaScript boundary source");
        let run = Command::new("node")
            .arg(&source_path)
            .output()
            .expect("run JavaScript boundary checks");
        assert!(
            run.status.success(),
            "JavaScript boundary/PCG check failed (status {:?}): {}",
            run.status.code(),
            String::from_utf8_lossy(&run.stderr)
        );
    }

    #[test]
    fn javascript_portable_claimed_corpus_matches_cpp17_bits() {
        let transforms = [
            "standardize",
            "robust_scale",
            "rank",
            "blom",
            "sigmoid",
            "step t=0.25",
            "winsor_clip q=0.1",
            "norm_row",
            "rff_cos n=4 gamma=0.7 seed=5",
            "rff_sin n=4 gamma=0.7 seed=5",
            "cheb degree=3",
            "hinge knots=3",
            "poly_interact",
            "quantile_bins bins=4",
            "bin_onehot bins=4",
            "pair_prod topk=3",
            "rand_proj k=4 seed=7",
            "row_stats",
            "row_diff_stats",
            "oblique_hinge n=4 seed=3",
            "landmark_dist k=4 gamma=0.7 seed=5",
            "target_bin_encode bins=4 prior=2",
        ];
        let heads = ["ridge", "elasticnet alpha=0.01 l1_ratio=0.3 sweeps=20"];
        let train = vec![
            vec![-2.0, 0.5, 1.25],
            vec![-1.0, -0.25, 2.0],
            vec![0.0, 1.5, -1.0],
            vec![0.75, -2.0, 0.25],
            vec![1.5, 0.25, -0.5],
            vec![3.0, 2.0, 1.5],
            vec![-0.75, 0.75, 3.0],
            vec![2.25, -1.5, -2.0],
        ];
        let targets = vec![-4.25, -1.375, -0.75, 3.5, 2.875, 5.0, -2.0, 4.25];
        let test = vec![
            vec![-0.5, 0.0, 1.0],
            vec![1.25, -1.0, 0.5],
            vec![2.5, 3.0, -1.5],
        ];
        let options = EmitOptions {
            math_kernel: MathKernel::Portable,
            ..EmitOptions::default()
        };
        let mut case_index = 0;
        for transform in transforms {
            for head in heads {
                let dsl = format!("(pipe X ({transform}) ({head}))");
                let genome = GenomeIr::parse(&dsl).expect("portable corpus parses");
                let cpp = run_cpp17_fixture(case_index, &genome, &options, &train, &targets, &test);
                let javascript = run_javascript_fixture(&genome, &options, &train, &targets, &test);
                assert_eq!(javascript, cpp, "Portable fitted bits for {dsl}");
                case_index += 1;
            }
        }
        assert_eq!(case_index, 44);
    }
}
