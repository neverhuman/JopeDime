use super::*;

/// Emit source for `genome` in `lang`, optionally writing artifacts under `output`.
pub fn emit_genome(genome: &str, lang: &str, output: Option<&Path>) -> Result<EmitReport> {
    emit_genome_with_train(genome, lang, None, output)
}

/// The fitted C++ route accepts a bounded numeric CSV whose final column is `target`.
/// Its model.cpp bytes are exactly the existing fitted emitter's output; the
/// optional standalone runner is a separate translation unit.
pub fn emit_genome_with_train(
    genome: &str,
    lang: &str,
    train: Option<&Path>,
    output: Option<&Path>,
) -> Result<EmitReport> {
    // Explicit version selection through the existing string-valued language
    // argument; no parser or legacy default changes are needed.
    if let Some(targets) = lang.strip_prefix("v3:") {
        let root = output.context("v3 emission requires --output to retain its bound bytes")?;
        let emitted = emit_genome_v3(genome, targets, train, root)?;
        let unsupported: Vec<_> = emitted
            .manifest
            .targets
            .iter()
            .filter_map(|r| match &r.state {
                feat_math::artifact_manifest::TargetState::Unsupported { reason, .. } => {
                    Some(format!("{}: {reason}", r.target.as_str()))
                }
                _ => None,
            })
            .collect();
        anyhow::ensure!(
            unsupported.is_empty(),
            "unsupported requested targets: {}; retained manifest {} sha256 {}",
            unsupported.join("; "),
            root.join("manifest.json").display(),
            emitted.expected.manifest_sha256
        );
        let mut artifacts = Vec::new();
        for record in &emitted.manifest.targets {
            if let feat_math::artifact_manifest::TargetState::Supported { files, .. } =
                &record.state
            {
                for file in files {
                    artifacts.push(Artifact {
                        lang: record.target.as_str().into(),
                        bytes: file.bytes as usize,
                        sha256: file.sha256.clone(),
                        path: Some(root.join(&file.path).display().to_string()),
                    });
                }
            }
        }
        let manifest_bytes = feat_math::artifact_manifest::read_artifact(root, "manifest.json")?;
        artifacts.push(Artifact {
            lang: "manifest-v3".into(),
            bytes: manifest_bytes.len(),
            sha256: emitted.expected.manifest_sha256,
            path: Some(root.join("manifest.json").display().to_string()),
        });
        return Ok(EmitReport {
            schema: "jope.emit.v3",
            ok: emitted.manifest.all_supported(),
            genome: emitted.manifest.dsl,
            artifacts,
        });
    }
    emit_genome_with_options_impl(
        genome,
        lang,
        train,
        output,
        &EmitOptions {
            title: Some("jope emit".into()),
            ..EmitOptions::default()
        },
        true,
    )
}

/// Explicit opt-in recipe options; root CLI flag allocation belongs to the integrator.
pub fn emit_genome_with_options(
    genome: &str,
    lang: &str,
    train: Option<&Path>,
    output: Option<&Path>,
    opts: &EmitOptions,
) -> Result<EmitReport> {
    emit_genome_with_options_impl(genome, lang, train, output, opts, false)
}

fn emit_genome_with_options_impl(
    genome: &str,
    lang: &str,
    train: Option<&Path>,
    output: Option<&Path>,
    opts: &EmitOptions,
    legacy_cpp20: bool,
) -> Result<EmitReport> {
    let parsed = GenomeIr::parse(genome).context("parse genome")?;
    if lang == "cpp17-fitted" {
        anyhow::ensure!(
            !opts.partition_point_quantile_bounds,
            "fitted C++ profile options unsupported"
        );
        let train = train.context("cpp17-fitted requires --train <numeric.csv>")?;
        let (x, y) = read_numeric_training_csv(train)?;
        let model = fit_legacy_numeric_bound(&parsed, &x, &y).context("fit C++17 model")?;
        let fitted = emit_fitted_cpp17(&model).context("emit fitted C++17")?;
        let mut artifacts = Vec::new();
        for (name, file, source) in [
            ("cpp17-fitted", "model.cpp", fitted.source.as_str()),
            (
                "cpp17-fitted-codec",
                "model.codec.json",
                fitted.codec.as_str(),
            ),
            (
                "cpp17-fitted-standalone",
                "standalone.cpp",
                FITTED_STANDALONE,
            ),
        ] {
            let path = if let Some(root) = output {
                fs::create_dir_all(root).with_context(|| format!("create {}", root.display()))?;
                let path = root.join(file);
                fs::write(&path, source).with_context(|| format!("write {}", path.display()))?;
                Some(path.display().to_string())
            } else {
                None
            };
            artifacts.push(Artifact {
                lang: name.into(),
                bytes: source.len(),
                sha256: digest(source),
                path,
            });
        }
        return Ok(EmitReport {
            schema: "jope.emit.v1",
            ok: true,
            genome: parsed.canonical(),
            artifacts,
        });
    }
    anyhow::ensure!(
        train.is_none(),
        "--train is only valid with --lang cpp17-fitted"
    );
    let mut artifacts = Vec::new();
    for name in languages(lang)? {
        if name == "c17" || name == "cuda" {
            let files = if name == "cuda" {
                feat_math::transpile::emit_cuda(&parsed, opts)?.files
            } else {
                feat_math::transpile::emit_c17(&parsed, opts)?.files
            };
            for (file, source) in files {
                let path = if let Some(root) = output {
                    let path = root.join(file);
                    fs::create_dir_all(path.parent().context("C17 output parent")?)?;
                    fs::write(&path, &source)?;
                    Some(path.display().to_string())
                } else {
                    None
                };
                artifacts.push(Artifact {
                    lang: name.into(),
                    bytes: source.len(),
                    sha256: digest(&source),
                    path,
                });
            }
            continue;
        }
        let source = match name {
            "python" | "rust" => {
                let emitted = emit(&parsed, opts).context("emit rust/python")?;
                if name == "python" {
                    emitted.python
                } else {
                    emitted.rust
                }
            }
            "cpp17" => emit_cpp17(&parsed, opts).context("emit C++17")?,
            "cpp17-recipe-v2" => {
                feat_math::kernel::expression_recipe::emit_cpp17(&parsed, opts)
                    .context("emit C++17 expression recipe")?
                    .source
            }
            // C++20 uses the same source and explicit options as C++17.
            "cpp20" => {
                if legacy_cpp20 {
                    match jope_pipeline::emit::cpp_source(&parsed).context("emit C++20")? {
                        Ok(source) => source,
                        Err(reason) => bail!("{reason}"),
                    }
                } else {
                    emit_cpp17(&parsed, opts).context("emit C++20")?
                }
            }
            "typescript" => {
                let bundle = feat_math::transpile::emit_typescript(&parsed, opts)?;
                for (target, filename, source) in [
                    ("typescript", "model.ts", &bundle.source),
                    ("typescript-kernel", "model.kernel.rs", &bundle.kernel),
                ] {
                    let path = if let Some(root) = output {
                        fs::create_dir_all(root)?;
                        let path = root.join(filename);
                        fs::write(&path, source)?;
                        Some(path.display().to_string())
                    } else {
                        None
                    };
                    artifacts.push(Artifact {
                        lang: target.into(),
                        bytes: source.len(),
                        sha256: digest(source),
                        path,
                    });
                }
                continue;
            }
            "go" => {
                let bundle = feat_math::transpile::emit_go(&parsed, opts)?;
                for (target, filename, source) in [
                    ("go", "model.go", &bundle.source),
                    ("go-kernel", "model.go.kernel.rs", &bundle.kernel),
                    ("go-module", "go.mod", &feat_math::go::MODULE.to_string()),
                    ("go-checksums", "go.sum", &feat_math::go::SUM.to_string()),
                ] {
                    let path = if let Some(root) = output {
                        fs::create_dir_all(root)?;
                        let path = root.join(filename);
                        fs::write(&path, source)?;
                        Some(path.display().to_string())
                    } else {
                        None
                    };
                    artifacts.push(Artifact {
                        lang: target.into(),
                        bytes: source.len(),
                        sha256: digest(source),
                        path,
                    });
                }
                continue;
            }
            "javascript" => emit_javascript(&parsed, opts).context("emit javascript")?,
            _ => unreachable!(),
        };
        let mut path = None;
        if let Some(root) = output {
            fs::create_dir_all(root).with_context(|| format!("create {}", root.display()))?;
            let file = root.join(filename(name));
            fs::write(&file, &source).with_context(|| format!("write {}", file.display()))?;
            path = Some(file.display().to_string());
        }
        artifacts.push(Artifact {
            lang: name.into(),
            bytes: source.len(),
            sha256: digest(&source),
            path,
        });
    }
    Ok(EmitReport {
        schema: "jope.emit.v1",
        ok: true,
        genome: parsed.canonical(),
        artifacts,
    })
}
