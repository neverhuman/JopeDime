//! Fit, emit, and a tiny offline search over feat-math GP genomes.

use crate::admission::validate_dataset;
use crate::schema::{Dataset, Kind, Split, Task, MAX_FILE_BYTES};
use crate::storage::Directory;
use anyhow::{bail, Context, Result};
use feat_math::transpile::{emit, emit_cpp17, emit_javascript, EmitOptions};
use feat_math::{emit_fitted_cpp17, fit_legacy_numeric_bound, r2, GenomeIr, Model, RawTable};
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::{fs, path::Path};

/// Default ridge pipeline used by fit, emit, and related CLI operations.
pub const DEFAULT_GENOME: &str = "(pipe X (ridge))";

const SEARCH_CANDIDATES: [&str; 3] = [
    "(pipe X (ridge))",
    "(pipe X (standardize) (ridge))",
    "(pipe X (robust_scale) (ridge))",
];

#[derive(Debug, Serialize)]
/// Training and validation scores for a fitted genome.
pub struct FitReport {
    pub schema: &'static str,
    pub ok: bool,
    pub dataset_id: String,
    pub genome: String,
    pub train_rows: usize,
    pub hold_rows: usize,
    pub train_r2: f64,
    pub hold_r2: f64,
}

#[derive(Debug, Serialize)]
/// Metadata for one emitted source artifact.
pub struct Artifact {
    pub lang: String,
    pub bytes: usize,
    pub sha256: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub path: Option<String>,
}

#[derive(Debug, Serialize)]
/// Summary of source artifacts emitted for a genome.
pub struct EmitReport {
    pub schema: &'static str,
    pub ok: bool,
    pub genome: String,
    pub artifacts: Vec<Artifact>,
}

#[derive(Debug, Serialize)]
/// Best candidate and validation score from the bounded offline search.
pub struct SearchReport {
    pub schema: &'static str,
    pub ok: bool,
    pub dataset_id: String,
    pub genome: String,
    pub hold_r2: f64,
    pub considered: usize,
}

/// Fit `genome` on the training split of a validated regression dataset.
pub fn fit_dataset(path: &Path, genome: &str) -> Result<FitReport> {
    let data = read_dataset(path)?;
    fit_validated_dataset(&data, genome)
}

/// Fit one already-validated snapshot; the caller never reopens its path.
pub(crate) fn fit_validated_dataset(data: &Dataset, genome: &str) -> Result<FitReport> {
    ensure_regression_support(data)?;
    let (train_x, train_y) = matrix(data, Split::Train)?;
    let (hold_x, hold_y) = holdout(data)?;
    let parsed = GenomeIr::parse(genome).context("parse genome")?;
    let model = feat_math::fit(&parsed, &train_x, &train_y).context("fit genome")?;
    Ok(FitReport {
        schema: "jope.fit.v1",
        ok: true,
        dataset_id: data.id.clone(),
        genome: parsed.canonical(),
        train_rows: train_x.len(),
        hold_rows: hold_x.len(),
        train_r2: score(&model, &train_x, &train_y)?,
        hold_r2: score(&model, &hold_x, &hold_y)?,
    })
}

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

fn read_numeric_training_csv(path: &Path) -> Result<(feat_math::Rows, Vec<f64>)> {
    let (x, y, _) = read_numeric_training_csv_with_schema(path)?;
    Ok((x, y))
}

fn read_numeric_training_csv_with_schema(
    path: &Path,
) -> Result<(feat_math::Rows, Vec<f64>, Vec<String>)> {
    let parent = path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    let filename = path
        .file_name()
        .and_then(|name| name.to_str())
        .context("training CSV needs a UTF-8 filename")?;
    let bytes = Directory::open(parent)
        .and_then(|directory| directory.read(filename, MAX_FILE_BYTES))
        .with_context(|| format!("read training CSV {}", path.display()))?;
    let csv = std::str::from_utf8(&bytes).context("training CSV is not UTF-8")?;
    let table = RawTable::from_csv(csv).context("parse training CSV")?;
    anyhow::ensure!(
        table.headers.len() >= 2 && table.headers.last().is_some_and(|name| name == "target"),
        "training CSV needs feature columns followed by target"
    );
    anyhow::ensure!(!table.rows.is_empty(), "training CSV has no rows");
    let width = table.headers.len() - 1;
    let mut x = Vec::with_capacity(table.rows.len());
    let mut y = Vec::with_capacity(table.rows.len());
    for (index, row) in table.rows.iter().enumerate() {
        let numbers = row
            .iter()
            .map(|cell| cell.parse::<f64>())
            .collect::<std::result::Result<Vec<_>, _>>()
            .with_context(|| format!("non-numeric cell on training row {}", index + 1))?;
        anyhow::ensure!(
            numbers.iter().all(|value| value.is_finite()),
            "non-finite value on training row {}",
            index + 1
        );
        x.push(numbers[..width].to_vec());
        y.push(numbers[width]);
    }
    Ok((x, y, table.headers[..width].to_vec()))
}

const FITTED_STANDALONE: &str = jope_pipeline::emit::fitted::STANDALONE;

/// Score the built-in small candidate set on a dataset's validation split.
pub fn search_dataset(path: &Path) -> Result<SearchReport> {
    let data = read_dataset(path)?;
    search_validated_dataset(&data)
}

/// Search one already-validated snapshot. Selection and its score use the validation split only.
pub(crate) fn search_validated_dataset(data: &Dataset) -> Result<SearchReport> {
    ensure_regression_support(data)?;
    let (train_x, train_y) = matrix(data, Split::Train)?;
    let (hold_x, hold_y) = holdout(data)?;
    let mut best_genome = String::new();
    let mut best_r2 = f64::NEG_INFINITY;
    for genome in SEARCH_CANDIDATES {
        let parsed = GenomeIr::parse(genome).context("parse search candidate")?;
        let model =
            feat_math::fit(&parsed, &train_x, &train_y).with_context(|| format!("fit {genome}"))?;
        let hold = score(&model, &hold_x, &hold_y).with_context(|| format!("evaluate {genome}"))?;
        if hold > best_r2 {
            best_r2 = hold;
            best_genome = parsed.canonical();
        }
    }
    anyhow::ensure!(!best_genome.is_empty(), "search produced no genome");
    Ok(SearchReport {
        schema: "jope.search.v1",
        ok: true,
        dataset_id: data.id.clone(),
        genome: best_genome,
        hold_r2: best_r2,
        considered: SEARCH_CANDIDATES.len(),
    })
}

fn languages(lang: &str) -> Result<Vec<&'static str>> {
    Ok(match lang {
        "c17" | "c" => vec!["c17"],
        "rust" => vec!["rust"],
        "python" => vec!["python"],
        "cpp17" => vec!["cpp17"],
        "cpp17-recipe-v2" => vec!["cpp17-recipe-v2"],
        "cpp20" => vec!["cpp20"],
        "javascript" | "js" => vec!["javascript"],
        "typescript" | "ts" => vec!["typescript"],
        "go" | "golang" => vec!["go"],
        "cuda" | "cuda-cpp" | "cu" => vec!["cuda"],
        "all" => vec!["python", "rust", "cpp17", "javascript"],
        other => bail!(
            "unknown language {other} (rust|python|cpp17|cpp17-recipe-v2|cpp20|javascript|typescript|go|c17|cuda|all)"
        ),
    })
}

fn filename(lang: &str) -> &'static str {
    match lang {
        "python" => "model.py",
        "rust" => "model.rs",
        "cpp17" | "cpp17-recipe-v2" | "cpp20" => "model.cpp",
        "javascript" => "model.js",
        _ => "model.txt",
    }
}

/// Read one dataset through the same bounded, no-symlink reader as admission (regular files
/// only, at most `MAX_FILE_BYTES`), then apply the full schema validation to those bytes.
pub(crate) fn read_dataset(path: &Path) -> Result<Dataset> {
    let parent = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    let filename = path
        .file_name()
        .and_then(|name| name.to_str())
        .with_context(|| format!("dataset path needs a UTF-8 filename: {}", path.display()))?;
    let bytes = Directory::open(parent)
        .and_then(|directory| directory.read(filename, MAX_FILE_BYTES))
        .with_context(|| format!("read dataset {}", path.display()))?;
    let data = serde_json::from_slice(&bytes)
        .with_context(|| format!("parse dataset {}", path.display()))?;
    validate_dataset(&data).with_context(|| format!("validate dataset {}", path.display()))?;
    Ok(data)
}

/// Why fit/search/route cannot use a dataset, or `None` when it is a supported regression
/// fixture with integer, non-null features.
pub(crate) fn unsupported_regression_reason(data: &Dataset) -> Option<String> {
    if data.task != Task::Regression {
        return Some(format!(
            "unsupported task {}: fit/search/route support regression only",
            data.task.name()
        ));
    }
    if data
        .columns
        .iter()
        .any(|column| column.kind != Kind::Integer || column.nullable)
    {
        return Some("fit/search/route currently admit integer, non-null features only".into());
    }
    None
}

pub(crate) fn ensure_regression_support(data: &Dataset) -> Result<()> {
    if let Some(reason) = unsupported_regression_reason(data) {
        bail!("{reason}");
    }
    Ok(())
}

/// The held-out score is the validation split. There is no fallback to test rows.
fn holdout(data: &Dataset) -> Result<(feat_math::Rows, Vec<f64>)> {
    matrix(data, Split::Validation)
}

fn matrix(data: &Dataset, split: Split) -> Result<(feat_math::Rows, Vec<f64>)> {
    ensure_regression_support(data)?;
    let mut xs = Vec::new();
    let mut ys = Vec::new();
    for row in &data.rows {
        if row.split != split {
            continue;
        }
        let mut features = Vec::with_capacity(data.columns.len());
        for feature in &row.features {
            features.push(
                feature
                    .as_i64()
                    .with_context(|| format!("non-integer feature in {}", data.id))?
                    as f64,
            );
        }
        xs.push(features);
        ys.push(row.target as f64);
    }
    anyhow::ensure!(!xs.is_empty(), "no {split:?} rows");
    Ok((xs, ys))
}

fn score(model: &Model, x: &feat_math::Rows, y: &[f64]) -> Result<f64> {
    let predictions = model.predict(x)?;
    anyhow::ensure!(
        predictions.len() == y.len() && predictions.iter().all(|value| value.is_finite()),
        "evaluation produced missing or non-finite predictions"
    );
    let score = r2(y, &predictions);
    anyhow::ensure!(score.is_finite(), "evaluation produced a non-finite R²");
    Ok(score)
}

fn digest(source: &str) -> String {
    format!("{:x}", Sha256::digest(source.as_bytes()))
}
/// Emit-only v3 adapter, selected with the existing `--lang v3:<targets>` string.
/// Existing language aliases, report schema and default emission stay compatible.
pub fn emit_genome_v3(
    genome: &str,
    lang: &str,
    train: Option<&Path>,
    output: &Path,
) -> Result<jope_pipeline::emit::EmittedV3> {
    use feat_math::artifact_manifest::Target;
    let requested = Target::parse_list(lang)?;
    anyhow::ensure!(
        train.is_none() || requested.contains(&Target::Cpp17Fitted),
        "--train requires cpp17-fitted"
    );
    let rows = if let Some(path) = train {
        let (x, y) = read_numeric_training_csv(path)?;
        Some(
            x.into_iter()
                .zip(y)
                .map(|(mut row, target)| {
                    row.push(target);
                    row
                })
                .collect::<Vec<_>>(),
        )
    } else {
        None
    };
    jope_pipeline::emit::emit_v3(genome, &requested, output, rows.as_deref())
}

#[cfg(test)]
mod artifact_v3_tests {
    use super::*;
    #[test]
    fn emit_adapter_preserves_default_route_and_binds_opt_in_outputs() {
        let dir = tempfile::tempdir().unwrap();
        let legacy = dir.path().join("legacy");
        emit_genome(DEFAULT_GENOME, "rust", Some(&legacy)).unwrap();
        assert!(!legacy.join("manifest.json").exists());
        let routed = dir.path().join("routed");
        let report = emit_genome(DEFAULT_GENOME, "v3:rs,js,cpp20", Some(&routed)).unwrap();
        assert_eq!(report.schema, "jope.emit.v3");
        assert!(report.ok);
        assert!(report.artifacts.iter().any(|a| a.lang == "manifest-v3"));
        for language in ["ts", "typescript", "v3:typescript"] {
            let output = dir.path().join(language.replace(':', "-"));
            let report = emit_genome(DEFAULT_GENOME, language, Some(&output)).unwrap();
            assert!(report.ok);
            assert!(output.join("model.ts").is_file());
            assert!(output.join("model.kernel.rs").is_file());
            assert!(report.artifacts.iter().any(|a| a.lang == "typescript"));
        }
        for language in ["golang", "go", "v3:go"] {
            let output = dir.path().join(language.replace(':', "-"));
            let report = emit_genome(DEFAULT_GENOME, language, Some(&output)).unwrap();
            assert!(report.ok);
            assert!(output.join("model.go").is_file());
            assert!(output.join("model.go.kernel.rs").is_file());
            assert!(report.artifacts.iter().any(|a| a.lang == "go"));
        }
        if matches!(
            std::env::var("JOPE_CUDA_PARITY").as_deref(),
            Ok("host" | "compile" | "device")
        ) {
            for language in ["cuda", "cu", "cuda-cpp", "v3:cuda"] {
                let output = dir.path().join(language.replace(':', "-"));
                let report = emit_genome(DEFAULT_GENOME, language, Some(&output)).unwrap();
                assert!(report.ok);
                for file in [
                    "CMakeLists.txt",
                    "model.cu",
                    "model.hpp",
                    "model_core.rs",
                    "model_native.rs",
                    "manifest.json",
                ] {
                    assert!(output.join("cuda").join(file).is_file());
                }
                assert!(report.artifacts.iter().any(|a| a.lang == "cuda"));
            }
        }
        let refused = dir.path().join("unsupported");
        let err = emit_genome("(pipe X (histgbm))", "v3:js", Some(&refused)).unwrap_err();
        assert!(err.to_string().contains("unsupported requested targets"));
        assert!(refused.join("manifest.json").is_file());
        let new = dir.path().join("v3");
        let out = emit_genome_v3(DEFAULT_GENOME, "rs,js,cpp20", None, &new).unwrap();
        assert!(out.manifest.all_supported());
        jope_pipeline::emit::read_identity_v3(&new, &out.expected).unwrap();
        assert!(emit_genome_v3(DEFAULT_GENOME, "rs,rust", None, &dir.path().join("bad")).is_err());
    }
    #[test]
    fn fitted_emit_adapter_binds_validated_training_rows() {
        let dir = tempfile::tempdir().unwrap();
        let train = dir.path().join("train.csv");
        fs::write(&train, "a,target\n0,1\n1,3\n2,5\n3,7\n").unwrap();
        let new = dir.path().join("v3");
        let out = emit_genome_v3(DEFAULT_GENOME, "cpp17-fitted", Some(&train), &new).unwrap();
        assert!(out.manifest.all_supported());
        jope_pipeline::emit::read_identity_v3(&new, &out.expected).unwrap();
        fs::write(new.join("cpp17-fitted/model.codec.json"), "{}").unwrap();
        assert!(jope_pipeline::emit::read_identity_v3(&new, &out.expected).is_err());
    }
}

/// Explicit fitted numeric profile route; does not change legacy emit defaults.
pub fn emit_numeric_v2(
    genome: &str,
    train: &Path,
    profile: feat_math::numeric_v2::NumericProfileV2,
    options: feat_math::numeric_v2_emit::OptionsV2,
    output: &Path,
) -> Result<feat_math::numeric_v2_emit::FittedArtifactV2> {
    let (x, y, schema) = read_numeric_training_csv_with_schema(train)?;
    anyhow::ensure!(
        profile.schema == schema,
        "numeric profile/CSV schema order mismatch"
    );
    jope_pipeline::emit::emit_numeric_v2(genome, profile, &x, &y, options, output)
}

#[cfg(test)]
mod numeric_v2_tests {
    use super::*;
    #[test]
    fn numeric_v2_kernel_export_and_option_selection() {
        let root =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/task-evidence/I10/cli-tests");
        fs::create_dir_all(&root).unwrap();
        let dir = tempfile::tempdir_in(root).unwrap();
        let csv = dir.path().join("train.csv");
        fs::write(&csv, "a,target\n0,-200\n1,0\n2,200\n").unwrap();
        let options = feat_math::numeric_v2_emit::OptionsV2::new(
            feat_math::numeric_v2_emit::TargetV2::Python,
        );
        let artifact = emit_numeric_v2(
            "(pipe X (ridge))",
            &csv,
            feat_math::numeric_v2::NumericProfileV2::new(vec!["a".into()]),
            options.clone(),
            dir.path(),
        )
        .unwrap();
        assert!(emit_numeric_v2(
            "(pipe X (ridge))",
            &csv,
            feat_math::numeric_v2::NumericProfileV2::new(vec!["wrong".into()]),
            options.clone(),
            dir.path()
        )
        .is_err());
        let model =
            jope_pipeline::emit::reload_numeric_v2(dir.path(), &options, &artifact.model_sha256)
                .unwrap();
        assert!(model.predict(&["a".into()], &vec![vec![2.]]).unwrap()[0] > 60.);
        let recipe = "(pipe X (quantile_bins bins=4) (ridge))";
        let old = emit_genome(recipe, "python", None).unwrap();
        let explicit = emit_genome_with_options(
            recipe,
            "python",
            None,
            None,
            &EmitOptions {
                title: Some("jope emit".into()),
                ..EmitOptions::default()
            },
        )
        .unwrap();
        assert_eq!(old.artifacts[0].sha256, explicit.artifacts[0].sha256);
        let new = emit_genome_with_options(
            recipe,
            "python",
            None,
            None,
            &EmitOptions {
                title: Some("jope emit".into()),
                partition_point_quantile_bounds: true,
                ..EmitOptions::default()
            },
        )
        .unwrap();
        assert_ne!(old.artifacts[0].sha256, new.artifacts[0].sha256);
        let cpp20 = emit_genome(recipe, "cpp20", None).unwrap();
        let parsed = GenomeIr::parse(recipe).unwrap();
        let historical = jope_pipeline::emit::cpp_source(&parsed).unwrap().unwrap();
        assert_eq!(cpp20.artifacts[0].sha256, digest(&historical));
    }
}
