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
#[derive(Debug, serde::Deserialize)]
pub(super) struct ManifestIdentity {
    dsl: String,
    genome_sha256: String,
    py_sha256: String,
    rs_sha256: String,
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
