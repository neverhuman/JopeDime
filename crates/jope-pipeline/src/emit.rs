//! Stage 6: emit standalone sources from the selected genome.
//!
//! This reproduces `crates/feat-math/examples/emit.rs`: `transpile::emit` for the
//! bit-identical Python/Rust pair plus the `battle-artifact-v1` manifest,
//! `emit_cpp17` and `emit_javascript` behind their own coverage gates, and a
//! `targets.txt` naming what was emitted. The legacy entrypoint uses
//! `EmitOptions::default()`; the explicit option entrypoint binds opt-in changes.
//! The title, feature names, context and `fast_runtime` flag all
//! change the Python and Rust bytes, and those bytes are what
//! `deploy/ci/emitters.sh` pins.
//!
//! There is exactly one C++ emitter. C++20 is a compile flag over the same
//! source, so `cpp17` and `cpp20` are two artifact rows sharing one `path` and
//! differing only in the recorded `build` command.
//!
//! A requested language the genome is not covered for is recorded as an
//! exclusion with the emitter's own reason, never silently dropped.

use anyhow::{bail, ensure, Context, Result};
use feat_math::{
    cxx::{self, CxxStd},
    genome::GenomeIr,
    transpile::{self, EmitOptions},
};
use std::path::{Path, PathBuf};

use crate::sha256_hex as hash;

pub mod fitted;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Language {
    Python,
    Rust,
    Cpp17,
    Cpp20,
    JavaScript,
    Go,
    TypeScript,
    Cpp17Fitted,
}

impl Language {
    pub const ALL: [Language; 5] = [
        Language::Rust,
        Language::Python,
        Language::Cpp17,
        Language::Cpp20,
        Language::JavaScript,
    ];

    pub fn as_str(self) -> &'static str {
        match self {
            Language::Python => "python",
            Language::Rust => "rust",
            Language::Cpp17 => "cpp17",
            Language::Cpp20 => "cpp20",
            Language::JavaScript => "javascript",
            Language::TypeScript => "typescript",
            Language::Go => "go",
            Language::Cpp17Fitted => "cpp17-fitted",
        }
    }

    pub fn parse(value: &str) -> Option<Language> {
        match value.trim().to_ascii_lowercase().as_str() {
            "python" | "py" => Some(Language::Python),
            "rust" | "rs" => Some(Language::Rust),
            "cpp17" | "c++17" => Some(Language::Cpp17),
            "cpp20" | "c++20" => Some(Language::Cpp20),
            "javascript" | "js" => Some(Language::JavaScript),
            "typescript" | "ts" => Some(Language::TypeScript),
            "go" | "golang" => Some(Language::Go),
            "cpp17-fitted" => Some(Language::Cpp17Fitted),
            _ => None,
        }
    }

    /// Legacy recipes, `cpp17-fitted`, or `all` (the legacy recipes). Duplicates collapse; the order is
    /// fixed so the emitted set does not depend on how the flag was written.
    pub fn parse_list(value: &str) -> Result<Vec<Language>> {
        let mut chosen = std::collections::BTreeSet::new();
        for part in value.split(',').map(str::trim).filter(|p| !p.is_empty()) {
            if part.eq_ignore_ascii_case("all") {
                chosen.extend(Language::ALL);
                continue;
            }
            let language = Language::parse(part).with_context(|| {
                format!(
                    "unknown emit target {part:?}; expected rust, python, cpp17, cpp20, js, ts, go, cpp17-fitted or all"
                )
            })?;
            chosen.insert(language);
        }
        if chosen.is_empty() {
            bail!("at least one emit target is required");
        }
        Ok(Language::ALL
            .into_iter()
            .chain([Language::Go, Language::TypeScript, Language::Cpp17Fitted])
            .filter(|l| chosen.contains(l))
            .collect())
    }
}

/// One emitted source file, as the receipt records it.
#[derive(Clone, Debug)]
pub struct Artifact {
    pub language: &'static str,
    pub path: String,
    pub sha256: String,
    pub bytes: u64,
    /// How the artifact is turned into a program, when that needs a compiler.
    pub build: Option<String>,
}

/// A requested language the genome is not covered for.
#[derive(Clone, Debug)]
pub struct Exclusion {
    pub language: &'static str,
    pub reason: String,
}

pub struct Emitted {
    pub canonical_genome: String,
    pub genome_sha256: String,
    pub artifacts: Vec<Artifact>,
    pub exclusions: Vec<Exclusion>,
}

/// Emit `genome` into `directory`, one file per covered language.
pub fn emit(genome: &str, languages: &[Language], directory: &Path) -> Result<Emitted> {
    emit_with_options(genome, languages, directory, &EmitOptions::default())
}

/// Explicit recipe-option route; legacy emit continues selecting frozen defaults.
pub fn emit_with_options(
    genome: &str,
    languages: &[Language],
    directory: &Path,
    options: &EmitOptions,
) -> Result<Emitted> {
    ensure!(
        !languages.contains(&Language::Cpp17Fitted),
        "cpp17-fitted requires training rows; use emit_with_training"
    );
    let parsed =
        GenomeIr::parse(genome.trim()).with_context(|| format!("parse genome {genome}"))?;
    std::fs::create_dir_all(directory)?;

    let mut artifacts = Vec::new();
    let mut exclusions = Vec::new();
    let mut targets: Vec<&str> = Vec::new();

    // emit() always produces the pair plus the manifest, or fails; the manifest is
    // the genome's identity sidecar and is always written.
    //
    // The test seam below makes the transpiler lower a *different* genome than the one requested.
    // Without it there is no way to write a failing test for the emit-to-selection binding: every
    // fixture genome is already canonical, so a check that compares the request with itself passes
    // whether or not the artifacts correspond.
    #[cfg(test)]
    let swapped: Option<GenomeIr> = seam::swapped_genome()?;
    #[cfg(test)]
    let render_target = swapped.as_ref().unwrap_or(&parsed);
    #[cfg(not(test))]
    let render_target = &parsed;
    let emitted = transpile::emit(render_target, options)?;
    write(directory, "manifest.json", &emitted.manifest)?;
    if languages.contains(&Language::Python) {
        artifacts.push(Artifact {
            language: Language::Python.as_str(),
            ..write(directory, "model.py", &emitted.python)?
        });
        targets.push("py");
    }
    if languages.contains(&Language::Rust) {
        artifacts.push(Artifact {
            language: Language::Rust.as_str(),
            ..write(directory, "model.rs", &emitted.rust)?
        });
        targets.push("rs");
    }

    let wants_cpp = languages.contains(&Language::Cpp17) || languages.contains(&Language::Cpp20);
    if wants_cpp {
        match cpp_source_with_options(&parsed, options)? {
            Ok(source) => {
                let artifact = write(directory, "model.cpp", &source)?;
                for (language, standard) in [
                    (Language::Cpp17, CxxStd::Cxx17),
                    (Language::Cpp20, CxxStd::Cxx20),
                ] {
                    if languages.contains(&language) {
                        artifacts.push(Artifact {
                            language: language.as_str(),
                            build: Some(build_command(standard)),
                            ..artifact.clone()
                        });
                    }
                }
                targets.push("cpp");
            }
            Err(reason) => {
                for language in [Language::Cpp17, Language::Cpp20] {
                    if languages.contains(&language) {
                        exclusions.push(Exclusion {
                            language: language.as_str(),
                            reason: reason.clone(),
                        });
                    }
                }
            }
        }
    }

    if languages.contains(&Language::JavaScript) {
        match transpile::supports_javascript(&parsed) {
            Ok(()) => {
                let source = transpile::emit_javascript(&parsed, options)?;
                artifacts.push(Artifact {
                    language: Language::JavaScript.as_str(),
                    ..write(directory, "model.js", &source)?
                });
                targets.push("js");
            }
            Err(unsupported) => exclusions.push(Exclusion {
                language: Language::JavaScript.as_str(),
                reason: format!(
                    "the JavaScript emitter does not cover {}",
                    unsupported.join(", ")
                ),
            }),
        }
    }

    if languages.contains(&Language::TypeScript) {
        let bundle = transpile::emit_typescript(render_target, options)?;
        artifacts.push(Artifact {
            language: "typescript",
            build: Some(feat_math::typescript::BUILD_COMMAND.into()),
            ..write(directory, "model.ts", &bundle.source)?
        });
        artifacts.push(Artifact {
            language: "typescript-kernel",
            ..write(directory, "model.kernel.rs", &bundle.kernel)?
        });
        targets.push("ts");
    }
    if languages.contains(&Language::Go) {
        let bundle = transpile::emit_go(render_target, options)?;
        for (path, source) in feat_math::go::sources(&bundle) {
            artifacts.push(Artifact {
                language: match path {
                    "go.mod" => "go-module",
                    "go.sum" => "go-checksums",
                    "model.go.kernel.rs" => "go-kernel",
                    _ => "go",
                },
                build: Some(feat_math::go::BUILD_COMMAND.into()),
                ..write(directory, path, source)?
            });
        }
        targets.push("golang");
    }

    write(
        directory,
        "targets.txt",
        &format!("{}\n", targets.join(" ")),
    )?;
    // The identity reported for this emission is read back out of the manifest that was just
    // written, NOT echoed from `parsed`. Deriving both sides of the comparison from the same
    // argument makes every downstream check a tautology: the emitted sources come from
    // `transpile::emit(&parsed, ..)` and were never consulted, so a transpiler that lowered a
    // different genome produced a receipt asserting the selected one, and every `ensure!` passed.
    // `transpile` builds the manifest from the IR it actually lowered, so reading it back is what
    // makes the identity a statement about the artifacts rather than about the request.
    let identity = read_identity(directory, &artifacts)?;
    Ok(Emitted {
        canonical_genome: identity.dsl,
        genome_sha256: identity.genome_sha256,
        artifacts,
        exclusions,
    })
}

/// Emit recipes and optionally a fitted C++17 predictor. Each training row is
/// finite numeric features followed by its target. Without `cpp17-fitted`, this
/// delegates to the unchanged recipe route and does not inspect `rows`.
pub fn emit_with_training(
    genome: &str,
    languages: &[Language],
    directory: &Path,
    rows: &[Vec<f64>],
) -> Result<Emitted> {
    if languages.contains(&Language::Cpp17Fitted) {
        fitted::emit(genome, languages, directory, rows)
    } else {
        emit(genome, languages, directory)
    }
}

/// The one C++ source for a parsed program: [`emit`] writes it to `model.cpp` for both `cpp17`
/// and `cpp20`, and `jope emit --lang cpp20` takes its bytes from here too. The inner `Err` is
/// the exclusion reason when the C++ emitter does not cover every element of the program.
pub fn cpp_source(parsed: &GenomeIr) -> Result<std::result::Result<String, String>> {
    cpp_source_with_options(parsed, &EmitOptions::default())
}

fn cpp_source_with_options(
    parsed: &GenomeIr,
    options: &EmitOptions,
) -> Result<std::result::Result<String, String>> {
    Ok(match transpile::supports_cpp17(parsed) {
        Ok(()) => Ok(transpile::emit_cpp17(parsed, options)?),
        Err(unsupported) => Err(format!(
            "the C++17 emitter does not cover {}",
            unsupported.join(", ")
        )),
    })
}

/// The subset of `manifest.json` that identifies what was emitted.
#[derive(Debug, serde::Deserialize)]
struct ManifestIdentity {
    dsl: String,
    genome_sha256: String,
    py_sha256: String,
    rs_sha256: String,
}

/// Read `manifest.json` back from `directory` and check it against the files beside it.
///
/// A manifest is only evidence about an emission if it describes the bytes actually written, so
/// its per-language digests are compared with the artifacts' own (which `write` takes from disk).
fn read_identity(directory: &Path, artifacts: &[Artifact]) -> Result<ManifestIdentity> {
    let path = directory.join("manifest.json");
    let bytes = std::fs::read(&path).with_context(|| format!("read back {}", path.display()))?;
    let identity: ManifestIdentity =
        serde_json::from_slice(&bytes).with_context(|| format!("parse {}", path.display()))?;
    ensure!(
        identity.genome_sha256 == hash(identity.dsl.as_bytes()),
        "{}: genome_sha256 {} does not hash its own dsl {}",
        path.display(),
        identity.genome_sha256,
        identity.dsl
    );
    for (name, declared) in [
        ("model.py", &identity.py_sha256),
        ("model.rs", &identity.rs_sha256),
    ] {
        let Some(artifact) = artifacts.iter().find(|a| a.path == name) else {
            continue;
        };
        ensure!(
            &artifact.sha256 == declared,
            "{}: declares {name} is {declared}, but the file written is {}",
            path.display(),
            artifact.sha256
        );
    }
    Ok(identity)
}

/// The command `deploy/ci/emitters.sh` compiles an emitted C++ source with: the
/// configured compiler, the requested standard, `-march` only when
/// `JOPE_CXX_MARCH` is set, and the two flags that make the bits reproducible.
fn build_command(standard: CxxStd) -> String {
    let mut parts = vec![
        cxx::compiler(),
        standard.flag().to_owned(),
        "-O3".to_owned(),
    ];
    parts.extend(cxx::march_flag(None));
    parts.extend(["-ffp-contract=off", "-fno-fast-math"].map(str::to_owned));
    parts.join(" ")
}

fn write(directory: &Path, name: &str, contents: &str) -> Result<Artifact> {
    let path: PathBuf = directory.join(name);
    std::fs::write(&path, contents).with_context(|| format!("write {}", path.display()))?;
    // Hash what is on disk, not the buffer we meant to put there. A digest taken from `contents`
    // describes this function's intent; the receipt claims to describe the emitted artifact.
    let written = std::fs::read(&path).with_context(|| format!("read back {}", path.display()))?;
    Ok(Artifact {
        language: "",
        path: name.to_owned(),
        sha256: hash(&written),
        bytes: written.len() as u64,
        build: None,
    })
}

/// Test-only hook: run `body` with the transpiler lowering `genome` instead of the one requested.
///
/// This exists so the emit-to-selection binding can be tested at all. It is compiled out of any
/// non-test build -- `emit()`'s `#[cfg(not(test))]` arm has no branch on it.
#[cfg(test)]
pub(crate) mod seam {
    use super::*;
    use std::cell::RefCell;

    thread_local! {
        static RENDER_INSTEAD: RefCell<Option<String>> = const { RefCell::new(None) };
    }

    pub(crate) fn swapped_genome() -> Result<Option<GenomeIr>> {
        let raw = RENDER_INSTEAD.with(|c| c.borrow().clone());
        match raw {
            Some(g) => Ok(Some(
                GenomeIr::parse(g.trim()).with_context(|| format!("parse swapped genome {g}"))?,
            )),
            None => Ok(None),
        }
    }

    /// Render `genome` instead of whatever `emit()` is asked for, for the duration of `body`.
    pub(crate) fn render_instead<T>(genome: &str, body: impl FnOnce() -> T) -> T {
        RENDER_INSTEAD.with(|c| *c.borrow_mut() = Some(genome.to_owned()));
        let out = body();
        RENDER_INSTEAD.with(|c| *c.borrow_mut() = None);
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The identity `emit()` reports must describe the artifacts it wrote, not the genome it was
    /// asked for. Reported on PR #44 by the independent reviewer, who demonstrated the gap by
    /// making the transpiler lower `(pipe X (ridge alpha=999.0))` while the receipt went on
    /// claiming the selected genome -- exit 0, `ok: true`, every `ensure!` satisfied and
    /// `verify-receipt` green, with `model.rs` containing a completely different model.
    #[test]
    fn the_reported_identity_follows_the_artifacts_not_the_request() {
        let dir = tempfile::tempdir().unwrap();
        let requested = "(pipe X (rank append=true) (gpr m=400 steps=60))";
        let actually_rendered = "(pipe X (ridge alpha=999.0))";

        // Control: with no swap, the reported identity is the requested genome.
        let honest = emit(
            requested,
            &[Language::Python, Language::Rust],
            dir.path().join("honest").as_path(),
        )
        .unwrap();
        assert_eq!(honest.canonical_genome, requested);

        // With the transpiler lowering something else, the reported identity must follow the
        // artifacts. Before this fix it echoed `requested` and the pipeline shipped happily.
        let swapped = seam::render_instead(actually_rendered, || {
            emit(
                requested,
                &[Language::Python, Language::Rust],
                dir.path().join("swapped").as_path(),
            )
        })
        .unwrap();
        assert_eq!(
            swapped.canonical_genome, actually_rendered,
            "emit() reported the genome it was asked for, not the one it wrote"
        );
        assert_ne!(
            swapped.genome_sha256, honest.genome_sha256,
            "two different emissions reported the same genome digest"
        );
    }

    /// A manifest is only evidence if it describes the files beside it.
    #[test]
    fn a_manifest_that_disagrees_with_the_file_it_names_is_refused() {
        let dir = tempfile::tempdir().unwrap();
        let out = dir.path().join("emit");
        let good = emit(
            "(pipe X (ridge alpha=1.0))",
            &[Language::Python, Language::Rust],
            out.as_path(),
        )
        .unwrap();

        // Rewrite model.py underneath a manifest that still names the original digest.
        std::fs::write(out.join("model.py"), "# not the emitted model\n").unwrap();
        let artifacts: Vec<Artifact> = good
            .artifacts
            .iter()
            .map(|a| Artifact {
                sha256: if a.path == "model.py" {
                    hash(b"# not the emitted model\n")
                } else {
                    a.sha256.clone()
                },
                language: a.language,
                path: a.path.clone(),
                bytes: a.bytes,
                build: a.build.clone(),
            })
            .collect();
        let err = read_identity(out.as_path(), &artifacts)
            .expect_err("a manifest was accepted over a file it does not describe");
        let msg = format!("{err:#}");
        assert!(
            msg.contains("model.py") && msg.contains("but the file written is"),
            "the refusal must name the file and both digests, got: {msg}"
        );
    }

    #[test]
    fn language_lists_parse_into_a_fixed_order() {
        assert_eq!(
            Language::parse_list("js,rust,cpp20,py,cpp17").unwrap(),
            Language::ALL.to_vec()
        );
        assert_eq!(Language::parse_list("all").unwrap(), Language::ALL.to_vec());
        assert_eq!(
            Language::parse_list("rs, rs ,rust").unwrap(),
            vec![Language::Rust]
        );
        assert_eq!(Language::parse_list("golang").unwrap(), [Language::Go]);
        assert!(Language::parse_list(" ").is_err());
    }

    #[test]
    fn the_bootstrap_genome_emits_every_language_and_shares_one_cpp_source() {
        let temp = tempfile::tempdir().unwrap();
        let emitted = emit(crate::select::BOOTSTRAP_GENOME, &Language::ALL, temp.path()).unwrap();
        assert!(emitted.exclusions.is_empty(), "{:?}", emitted.exclusions);
        let languages: Vec<&str> = emitted.artifacts.iter().map(|a| a.language).collect();
        assert_eq!(
            languages,
            ["python", "rust", "cpp17", "cpp20", "javascript"]
        );
        let cpp: Vec<&Artifact> = emitted
            .artifacts
            .iter()
            .filter(|a| a.path == "model.cpp")
            .collect();
        assert_eq!(cpp.len(), 2);
        assert_eq!(cpp[0].sha256, cpp[1].sha256);
        assert!(cpp[0].build.as_ref().unwrap().contains("-std=c++17"));
        assert!(cpp[1].build.as_ref().unwrap().contains("-std=c++20"));
        for artifact in &emitted.artifacts {
            let path = temp.path().join(&artifact.path);
            assert_eq!(std::fs::metadata(&path).unwrap().len(), artifact.bytes);
        }
        assert_eq!(
            std::fs::read_to_string(temp.path().join("targets.txt")).unwrap(),
            "py rs cpp js\n"
        );
        let typescript_dir = temp.path().join("typescript");
        let typescript = emit(
            crate::select::BOOTSTRAP_GENOME,
            &[Language::TypeScript],
            &typescript_dir,
        )
        .unwrap();
        assert!(typescript.exclusions.is_empty());
        assert_eq!(typescript.artifacts.len(), 2);
        assert_eq!(
            std::fs::read_to_string(typescript_dir.join("targets.txt")).unwrap(),
            "ts\n"
        );
        assert!(temp.path().join("manifest.json").is_file());
        assert_eq!(
            emitted.genome_sha256,
            hash(emitted.canonical_genome.as_bytes())
        );
        assert_eq!(
            emitted.canonical_genome,
            feat_math::genome::GenomeIr::parse(crate::select::BOOTSTRAP_GENOME)
                .unwrap()
                .canonical()
        );
        let go_dir = temp.path().join("go");
        let go = emit(crate::select::BOOTSTRAP_GENOME, &[Language::Go], &go_dir).unwrap();
        assert!(go.exclusions.is_empty());
        assert_eq!(go.artifacts.len(), 4);
        assert_eq!(
            std::fs::read_to_string(go_dir.join("targets.txt")).unwrap(),
            "golang\n"
        );
        assert!(temp.path().join("manifest.json").is_file());
        assert_eq!(
            emitted.genome_sha256,
            hash(emitted.canonical_genome.as_bytes())
        );
        assert_eq!(
            emitted.canonical_genome,
            feat_math::genome::GenomeIr::parse(crate::select::BOOTSTRAP_GENOME)
                .unwrap()
                .canonical()
        );
    }

    #[test]
    fn a_head_the_cpp_and_js_emitters_do_not_cover_is_recorded_not_dropped() {
        let temp = tempfile::tempdir().unwrap();
        let genome = "(pipe X (histgbm depth=5 lr=0.15 rounds=150))";
        let emitted = emit(genome, &Language::ALL, temp.path()).unwrap();
        let languages: Vec<&str> = emitted.artifacts.iter().map(|a| a.language).collect();
        assert_eq!(languages, ["python", "rust"]);
        let excluded: Vec<&str> = emitted.exclusions.iter().map(|e| e.language).collect();
        assert_eq!(excluded, ["cpp17", "cpp20", "javascript"]);
        assert!(emitted.exclusions.iter().all(|e| !e.reason.is_empty()));
        assert_eq!(
            std::fs::read_to_string(temp.path().join("targets.txt")).unwrap(),
            "py rs\n"
        );
    }
}

#[cfg(test)]
mod fitted_tests;
/// Explicit opt-in publication result. `expected` must be retained in the selected
/// program/run receipt; reacquiring it from mutable files loses the trust anchor.
pub struct EmittedV3 {
    pub expected: feat_math::artifact_manifest::ExpectedArtifactV3,
    pub manifest: feat_math::artifact_manifest::ArtifactManifestV3,
}

/// New target-complete writer; old `emit`/`emit_with_training` remain diagnostic
/// legacy routes. The independent requests are bound before disk publication.
pub fn emit_v3(
    genome: &str,
    requested: &[feat_math::artifact_manifest::Target],
    directory: &Path,
    rows: Option<&[Vec<f64>]>,
) -> Result<EmittedV3> {
    use feat_math::artifact_manifest::Target;
    let parsed = GenomeIr::parse(genome.trim()).context("parse v3 selected program")?;
    ensure!(
        rows.is_none() || requested.contains(&Target::Cpp17Fitted),
        "training rows require fitted target"
    );
    let mut emission = transpile::emit_targets_v3(&parsed, &EmitOptions::default(), requested)?;
    if requested.contains(&Target::Cpp17Fitted) {
        fitted::add_v3(
            &mut emission,
            &parsed,
            rows.context("fitted target requires explicit training rows")?,
        )?;
    }
    publish_v3(emission, directory)
}

/// Publish already rendered recipe or bound-model sources and verify disk bytes.
pub fn publish_v3(
    emission: feat_math::artifact_manifest::TargetEmission,
    directory: &Path,
) -> Result<EmittedV3> {
    use feat_math::artifact_manifest::{self, Domain, ExpectedArtifactV3, TargetState};
    emission.manifest.validate()?;
    let bound: std::collections::BTreeSet<_> = emission
        .manifest
        .targets
        .iter()
        .flat_map(|record| match &record.state {
            TargetState::Supported { files, .. } => {
                files.iter().map(|f| f.path.clone()).collect::<Vec<_>>()
            }
            TargetState::Unsupported { .. } => vec![],
        })
        .collect();
    ensure!(
        bound == emission.sources.keys().cloned().collect(),
        "source set disagrees with expected target closure"
    );
    let manifest_bytes = emission.manifest.envelope_bytes(Domain::Native)?;
    let expected = ExpectedArtifactV3::from_emission(&manifest_bytes, &emission.manifest);
    std::fs::create_dir_all(directory)?;
    ensure!(
        !std::fs::symlink_metadata(directory)?
            .file_type()
            .is_symlink(),
        "symlink output directory"
    );
    for (path, source) in &emission.sources {
        let destination = directory.join(path);
        if let Some(parent) = destination.parent() {
            std::fs::create_dir_all(parent)?;
            ensure!(
                !std::fs::symlink_metadata(parent)?.file_type().is_symlink(),
                "symlink artifact parent"
            );
        }
        write_v3_file(&destination, source.as_bytes())?;
    }
    write_v3_file(&directory.join("manifest.json"), &manifest_bytes)?;
    let manifest = artifact_manifest::verify_directory(directory, "manifest.json", &expected)?;
    Ok(EmittedV3 { expected, manifest })
}
fn write_v3_file(path: &Path, bytes: &[u8]) -> Result<()> {
    // create_new prevents following/replacing a prior file or symlink. A failed
    // publication is retained for diagnosis and requires a fresh output directory.
    use std::io::Write;
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .with_context(|| format!("create v3 artifact {}", path.display()))?;
    file.write_all(bytes)?;
    Ok(())
}

/// Consumption always requires the external root and independent request set.
pub fn read_identity_v3(
    directory: &Path,
    expected: &feat_math::artifact_manifest::ExpectedArtifactV3,
) -> Result<feat_math::artifact_manifest::ArtifactManifestV3> {
    feat_math::artifact_manifest::verify_directory(directory, "manifest.json", expected)
}

/// Task-local I09 adapter: fitted predictor with explicit requested target and profile.
/// Writes exact source, fitted codec and manifest; no native policy export is implied.
pub fn emit_numeric_v2(
    genome: &str,
    profile: feat_math::numeric_v2::NumericProfileV2,
    x: &feat_math::Rows,
    y: &[f64],
    options: feat_math::numeric_v2_emit::OptionsV2,
    directory: &Path,
) -> Result<feat_math::numeric_v2_emit::FittedArtifactV2> {
    use feat_math::{
        numeric_v2::NumericModelV2,
        numeric_v2_emit::{self, TargetV2},
    };
    let parsed = GenomeIr::parse(genome)?;
    let model = NumericModelV2::fit(profile, &parsed, x, y)?;
    let artifact = numeric_v2_emit::emit(&model, options)?;
    let filename = match artifact.options.target {
        TargetV2::Python => "model.py",
        TargetV2::Rust => "model.rs",
    };
    std::fs::create_dir_all(directory)?;
    std::fs::write(directory.join(filename), &artifact.source)?;
    std::fs::write(directory.join("model.codec.json"), &artifact.codec)?;
    std::fs::write(
        directory.join("numeric-manifest.json"),
        serde_json::to_vec_pretty(&artifact)?,
    )?;
    Ok(artifact)
}

/// Reload and verify exactly the persisted source and fitted codec before use.
pub fn reload_numeric_v2(
    directory: &Path,
    expected: &feat_math::numeric_v2_emit::OptionsV2,
    model_digest: &str,
) -> Result<feat_math::numeric_v2::NumericModelV2> {
    use feat_math::numeric_v2_emit::{FittedArtifactV2, TargetV2};
    let artifact: FittedArtifactV2 =
        serde_json::from_slice(&std::fs::read(directory.join("numeric-manifest.json"))?)?;
    let filename = match expected.target {
        TargetV2::Python => "model.py",
        TargetV2::Rust => "model.rs",
    };
    ensure!(
        std::fs::read(directory.join(filename))? == artifact.source.as_bytes(),
        "shipped source mismatch"
    );
    ensure!(
        std::fs::read(directory.join("model.codec.json"))? == artifact.codec.as_bytes(),
        "shipped codec mismatch"
    );
    artifact.reload(expected, model_digest)
}
