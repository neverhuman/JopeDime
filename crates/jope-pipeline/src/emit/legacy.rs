use super::*;

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
        match cpp_source_with_options(render_target, options)? {
            Ok(honest) => {
                let written = unbound_source(render_target, &honest, |genome| {
                    transpile::emit_cpp17(genome, options)
                })?;
                let declared = hash(honest.as_bytes());
                if languages.contains(&Language::Cpp17) {
                    bind_digest(directory, "cpp17_sha256", &declared)?;
                }
                if languages.contains(&Language::Cpp20) {
                    bind_digest(directory, "cpp20_sha256", &declared)?;
                }
                let artifact = write(directory, "model.cpp", &written)?;
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
        match transpile::supports_javascript(render_target) {
            Ok(()) => {
                let honest = transpile::emit_javascript(render_target, options)?;
                let written = unbound_source(render_target, &honest, |genome| {
                    transpile::emit_javascript(genome, options)
                })?;
                let declared = hash(honest.as_bytes());
                bind_digest(directory, "js_sha256", &declared)?;
                artifacts.push(Artifact {
                    language: Language::JavaScript.as_str(),
                    ..write(directory, "model.js", &written)?
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

    if languages.contains(&Language::Cuda) {
        for (path, source) in transpile::emit_cuda(render_target, options)?.files {
            artifacts.push(Artifact {
                language: Language::Cuda.as_str(),
                build: Some(feat_math::cuda::BUILD.into()),
                ..write(directory, &path, &source)?
            });
        }
        targets.push("cuda");
    }
    if languages.contains(&Language::C17) {
        for (path, source) in transpile::emit_c17(render_target, options)?.files {
            artifacts.push(Artifact {
                language: Language::C17.as_str(),
                build: Some(format!("cd c17 && {}", feat_math::c17::BUILD)),
                ..write(directory, &path, &source)?
            });
        }
        targets.push("c17");
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
///
/// `cpp17_sha256`, `cpp20_sha256` and `js_sha256` are absent on manifests
/// written before those targets were bound. A new emission that writes the
/// corresponding file always records the digest.
#[derive(Debug, serde::Deserialize)]
pub(super) struct ManifestIdentity {
    dsl: String,
    genome_sha256: String,
    py_sha256: String,
    rs_sha256: String,
    #[serde(default)]
    cpp17_sha256: Option<String>,
    #[serde(default)]
    cpp20_sha256: Option<String>,
    #[serde(default)]
    js_sha256: Option<String>,
}

/// Read `manifest.json` back from `directory` and check it against the files beside it.
///
/// A manifest is only evidence about an emission if it describes the bytes actually written, so
/// its per-language digests are compared with the artifacts' own (which `write` takes from disk).
pub(super) fn read_identity(directory: &Path, artifacts: &[Artifact]) -> Result<ManifestIdentity> {
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
    for artifact in artifacts {
        let Some(declared) = digest_for(&identity, artifact)? else {
            continue;
        };
        ensure!(
            artifact.sha256 == declared,
            "{}: declares {} is {declared}, but the file written is {}",
            path.display(),
            artifact.path,
            artifact.sha256
        );
    }
    Ok(identity)
}

/// Digest field for an artifact this schema binds. Other languages are not
/// `battle-artifact-v1` rows. A bound path with no digest is a refusal: the
/// old loop skipped a missing Python or Rust file and also skipped C++ and
/// JavaScript because the schema had no field for them.
fn digest_for(identity: &ManifestIdentity, artifact: &Artifact) -> Result<Option<String>> {
    let declared = match (artifact.path.as_str(), artifact.language) {
        ("model.py", _) => Some(identity.py_sha256.clone()),
        ("model.rs", _) => Some(identity.rs_sha256.clone()),
        ("model.cpp", "cpp17") => identity.cpp17_sha256.clone(),
        ("model.cpp", "cpp20") => identity.cpp20_sha256.clone(),
        ("model.js", _) => identity.js_sha256.clone(),
        _ => return Ok(None),
    };
    match declared {
        Some(digest) => Ok(Some(digest)),
        None => bail!(
            "cannot bind {} ({}); manifest.json has no digest for that artifact",
            artifact.path,
            artifact.language
        ),
    }
}

/// Record `digest` on the manifest that `transpile::emit` already wrote.
///
/// The original key order stays. The new field is appended so an old reader
/// that only knows `py_sha256` and `rs_sha256` still sees those values, and a
/// new reader can require the field when the artifact is present.
fn bind_digest(directory: &Path, field: &str, digest: &str) -> Result<()> {
    let path = directory.join("manifest.json");
    let mut text = std::fs::read_to_string(&path)
        .with_context(|| format!("read {} before binding {field}", path.display()))?;
    let trimmed = text.trim_end();
    ensure!(
        trimmed.ends_with('}'),
        "{} is not a JSON object",
        path.display()
    );
    ensure!(
        !trimmed.contains(&format!("\"{field}\"")),
        "{} already binds {field}",
        path.display()
    );
    text.truncate(trimmed.len() - 1);
    text.push_str(&format!(",\"{field}\":\"{digest}\"}}"));
    std::fs::write(&path, &text).with_context(|| format!("write {}", path.display()))?;
    Ok(())
}

/// Bytes written for C++ or JavaScript. Production writes the same source the
/// manifest digest names. The test seam can write a different genome.
fn unbound_source(
    honest_genome: &GenomeIr,
    honest_source: &str,
    emit_source: impl FnOnce(&GenomeIr) -> Result<String>,
) -> Result<String> {
    #[cfg(test)]
    if let Some(swapped) = super::seam::unbound_cpp_js()? {
        let _ = honest_genome;
        return emit_source(&swapped);
    }
    let _ = (honest_genome, emit_source);
    Ok(honest_source.to_owned())
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

pub(super) fn write(directory: &Path, name: &str, contents: &str) -> Result<Artifact> {
    let path: PathBuf = directory.join(name);
    std::fs::create_dir_all(path.parent().context("artifact parent")?)?;
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
