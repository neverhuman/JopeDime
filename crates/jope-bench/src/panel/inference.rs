//! Allowlisted fitted inference exports and private qualification plans.
//! Exporting does not qualify a backend, refit a model, or replace the selected method.
use super::{
    data::RAW_CELL_ENCODING,
    methods::{digest, FittedRustArtifact},
    score::Predictions,
    synthesized::FittedSynthesizedArtifact,
};
use anyhow::{ensure, Context, Result};
use feat_math::advanced_v3::{
    emit_fitted_portable, emit_fitted_public_cpp17, fitted_javascript_unsupported,
    fitted_portable_templates, fitted_portable_unsupported, fitted_public_cpp_unsupported,
    lint_fitted_public_text, ColumnKind, PortableUnsupported,
};
use jope_datasets::catalog::Task;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

mod contract;
pub use contract::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", content = "detail", rename_all = "snake_case")]
pub enum UnsupportedReason {
    FittedState(PortableUnsupported),
    CppMatrix,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum ExportOutcome {
    Emitted {
        bundle: Box<PublicBundle>,
    },
    Unsupported {
        backend: Backend,
        reason: UnsupportedReason,
    },
}

/// Identifies complete fitted numeric fill/onehot state without claiming a particular recipe.
pub const FITTED_INPUT_POLICY: &str = "jope.fitted-input-state/v1";

/// Nonserializable borrowed view; constructors authenticate the complete private
/// envelope first. Proposal text and discovery identity cannot enter public export
/// through this view. Its private artifact digest is used only by qualification.
pub struct FittedInferenceView<'a> {
    model: &'a feat_math::advanced_v3::BoundModel,
    task: Task,
    classes: Vec<String>,
    input_policy: String,
    fitted_sha256: String,
}
impl<'a> FittedInferenceView<'a> {
    pub fn registered(artifact: &'a FittedRustArtifact) -> Result<Self> {
        artifact.validate()?;
        Ok(Self {
            model: &artifact.model,
            task: artifact.task,
            classes: artifact.classes.clone(),
            input_policy: artifact.input_policy.clone(),
            fitted_sha256: digest(&artifact.bytes()?),
        })
    }
    pub fn synthesized(artifact: &'a FittedSynthesizedArtifact) -> Result<Self> {
        artifact.validate()?;
        Ok(Self {
            model: &artifact.model,
            task: artifact.task,
            classes: artifact.classes.clone(),
            input_policy: FITTED_INPUT_POLICY.into(),
            fitted_sha256: digest(&artifact.bytes()?),
        })
    }
    fn validate(&self) -> Result<()> {
        self.model.validate_binding()
    }
    /// Digest of the complete private fitted envelope this view was built from.
    pub fn fitted_sha256(&self) -> &str {
        &self.fitted_sha256
    }
    fn predict_raw(
        &self,
        features: &[jope_pipeline::experiment::Column],
        rows: &[Vec<Option<String>>],
    ) -> Result<Predictions> {
        self.validate()?;
        ensure!(
            features.len() == self.model.schema.columns.len()
                && features
                    .iter()
                    .zip(&self.model.schema.columns)
                    .all(|(a, b)| a.name == b.name
                        && matches!(
                            (&a.kind, b.kind),
                            (
                                jope_pipeline::experiment::ColumnKind::Numeric,
                                ColumnKind::Numeric
                            ) | (
                                jope_pipeline::experiment::ColumnKind::Categorical,
                                ColumnKind::Categorical
                            )
                        )),
            "raw input columns differ from fitted inference schema"
        );
        let table = super::data::raw_table(features, rows)?;
        super::score::predict_bound(self.model, &table, &self.classes)
    }
}

pub fn export_fitted(artifact: &FittedRustArtifact, backend: Backend) -> Result<ExportOutcome> {
    export_inference(&FittedInferenceView::registered(artifact)?, backend)
}
pub fn export_synthesized(
    artifact: &FittedSynthesizedArtifact,
    backend: Backend,
) -> Result<ExportOutcome> {
    export_inference(&FittedInferenceView::synthesized(artifact)?, backend)
}

/// Only generated source, fitted input dictionaries, and the public boundary are selected.
/// The private envelope, recipe text, discovery identity, and emitter provenance manifest
/// are never serialized into this bundle.
pub fn export_inference(
    artifact: &FittedInferenceView<'_>,
    backend: Backend,
) -> Result<ExportOutcome> {
    artifact.validate()?;
    if let Some(reason) = fitted_portable_unsupported(artifact.model)? {
        return Ok(ExportOutcome::Unsupported {
            backend,
            reason: UnsupportedReason::FittedState(reason),
        });
    }
    if matches!(backend, Backend::Cpp17 | Backend::Cpp20) && artifact.task == Task::Multiclass {
        return Ok(ExportOutcome::Unsupported {
            backend,
            reason: UnsupportedReason::CppMatrix,
        });
    }
    let extra = match backend {
        Backend::JavaScript => fitted_javascript_unsupported(artifact.model)?,
        Backend::Cpp17 | Backend::Cpp20 => fitted_public_cpp_unsupported(artifact.model)?,
        _ => None,
    };
    if let Some(reason) = extra {
        return Ok(ExportOutcome::Unsupported {
            backend,
            reason: UnsupportedReason::FittedState(reason),
        });
    }
    let files = emit_files(artifact, backend)?;
    let manifest = PublicManifest {
        schema: "jope.fitted-public-bundle/v1".into(),
        backend,
        task: artifact.task,
        columns: artifact.model.schema.columns.clone(),
        classes: artifact.classes.clone(),
        raw_cell_encoding: RAW_CELL_ENCODING.into(),
        input_policy: artifact.input_policy.clone(),
        numerical_policy: backend.numerical_policy(),
        emitter: backend.emitter().into(),
        template_sha256: templates(backend),
        runtime: backend.runtime().into(),
        host_boundary: backend.host_boundary().into(),
        parity_policy: backend.parity_policy(),
        qualification: "emitted-unqualified".into(),
        entrypoint: match backend {
            Backend::Rust => "predict_raw(&[Vec<Option<String>>]) -> Result<String, String>",
            Backend::JavaScript => "predictRaw(rows) -> Predictions",
            _ => "predict_raw(rows) -> Predictions",
        }
        .into(),
        files: files
            .iter()
            .map(|(path, text)| (path.clone(), digest(text.as_bytes())))
            .collect(),
    };
    let bundle = PublicBundle { manifest, files };
    bundle.validate()?;
    let metadata = serde_json::to_string(&bundle.manifest)?;
    for text in bundle.files.values().chain(std::iter::once(&metadata)) {
        ensure!(
            lint_fitted_public_text(artifact.model, &artifact.classes, text).is_empty(),
            "public fitted artifact violates text policy"
        );
    }
    Ok(ExportOutcome::Emitted {
        bundle: Box::new(bundle),
    })
}

fn templates(backend: Backend) -> BTreeMap<String, String> {
    let mut result = fitted_portable_templates();
    for (name, text) in [
        ("boundary.py", include_str!("inference_boundary.py")),
        ("boundary.rs", include_str!("inference_boundary.rs.txt")),
        ("boundary.js", include_str!("inference_boundary.js.txt")),
        ("typed-host.py", include_str!("inference_cpp_boundary.py")),
    ] {
        result.insert(name.into(), digest(text.as_bytes()));
    }
    // Even unused templates are pinned: one emitter identity covers the complete matrix ABI.
    result.insert("backend".into(), digest(backend.emitter().as_bytes()));
    result
}

fn emit_files(
    artifact: &FittedInferenceView<'_>,
    backend: Backend,
) -> Result<BTreeMap<String, String>> {
    let mut files = BTreeMap::new();
    match backend {
        Backend::Python | Backend::Rust | Backend::JavaScript => {
            let emitted = emit_fitted_portable(artifact.model)?;
            if artifact.task != Task::Regression {
                ensure!(
                    emitted.class_codes == (0..artifact.classes.len() as i64).collect::<Vec<_>>(),
                    "fitted matrix class columns differ"
                );
            }
            match backend {
                Backend::Python => {
                    files.insert(
                        "model.py".into(),
                        emitted.python + "\n" + &python_boundary(artifact, None)?,
                    );
                }
                Backend::Rust => {
                    files.insert(
                        "model.rs".into(),
                        emitted.rust + "\n" + &rust_boundary(artifact)?,
                    );
                }
                Backend::JavaScript => {
                    let boundary = render(
                        include_str!("inference_boundary.js.txt"),
                        &[
                            ("__KINDS__", serde_json::to_string(&categorical(artifact))?),
                            ("__CLASSES__", serde_json::to_string(&artifact.classes)?),
                            (
                                "__REGRESSION__",
                                (artifact.task == Task::Regression).to_string(),
                            ),
                        ],
                    );
                    files.insert(
                        "model.js".into(),
                        emitted
                            .javascript
                            .context("JavaScript fitted state was not admitted")?
                            + "\n"
                            + &boundary,
                    );
                }
                _ => unreachable!(),
            }
        }
        Backend::Cpp17 | Backend::Cpp20 => {
            let emitted = emit_fitted_public_cpp17(artifact.model)?;
            files.insert("model.cpp".into(), emitted.source);
            files.insert("codec.json".into(), emitted.codec.clone());
            files.insert(
                "boundary.py".into(),
                python_boundary(artifact, Some(&emitted.codec))?,
            );
        }
    }
    Ok(files)
}
// Substitute only the template itself; user strings resembling markers remain literal.
fn render(mut template: &str, values: &[(&str, String)]) -> String {
    for (key, _) in values {
        assert!(template.contains(key), "missing template marker {key}");
    }
    let mut remaining = template.to_owned();
    for (key, _) in values {
        remaining = remaining.replace(key, "");
    }
    for fragment in remaining.split("__").skip(1).step_by(2) {
        assert!(
            !fragment
                .bytes()
                .all(|b| b.is_ascii_uppercase() || b == b'_'),
            "unknown template marker"
        );
    }
    let mut output = String::new();
    while let Some((offset, key, value)) = values
        .iter()
        .filter_map(|(key, value)| template.find(key).map(|offset| (offset, *key, value)))
        .min_by_key(|item| item.0)
    {
        output.push_str(&template[..offset]);
        output.push_str(value);
        template = &template[offset + key.len()..];
    }
    output.push_str(template);
    output
}

fn categorical(artifact: &FittedInferenceView<'_>) -> Vec<bool> {
    artifact
        .model
        .schema
        .columns
        .iter()
        .map(|column| column.kind == ColumnKind::Categorical)
        .collect()
}
fn python_boundary(artifact: &FittedInferenceView<'_>, cpp: Option<&str>) -> Result<String> {
    let implementation = if let Some(codec) = cpp {
        render(
            include_str!("inference_cpp_boundary.py"),
            &[("__CODEC_JSON__", serde_json::to_string(codec)?)],
        )
    } else {
        "def _matrix(rows):\n    return predict_encoded(rows)\n".into()
    };
    Ok(render(
        include_str!("inference_boundary.py"),
        &[
            ("__IMPLEMENTATION__", implementation),
            (
                "__KINDS_JSON__",
                serde_json::to_string(&serde_json::to_string(&categorical(artifact))?)?,
            ),
            (
                "__CLASSES_JSON__",
                serde_json::to_string(&serde_json::to_string(&artifact.classes)?)?,
            ),
            (
                "__REGRESSION__",
                (if artifact.task == Task::Regression {
                    "True"
                } else {
                    "False"
                })
                .into(),
            ),
        ],
    ))
}
fn rust_boundary(artifact: &FittedInferenceView<'_>) -> Result<String> {
    let compute = "let values=predict_encoded(&encoded)?;".to_owned();
    let prefix = if artifact.task == Task::Regression {
        "{\"kind\":\"values\",\"data\":[".to_owned()
    } else {
        format!(
            "{{\"kind\":\"probabilities\",\"data\":{{\"classes\":{},\"values\":[",
            serde_json::to_string(&artifact.classes)?
        )
    };
    Ok(render(
        include_str!("inference_boundary.rs.txt"),
        &[
            (
                "__KINDS__",
                categorical(artifact)
                    .iter()
                    .map(bool::to_string)
                    .collect::<Vec<_>>()
                    .join(","),
            ),
            ("__COMPUTE__", compute),
            (
                "__WIDTH__",
                if artifact.task == Task::Regression {
                    "1".into()
                } else {
                    artifact.classes.len().to_string()
                },
            ),
            ("__PREFIX__", format!("{prefix:?}")),
            (
                "__REGRESSION__",
                (if artifact.task == Task::Regression {
                    "true"
                } else {
                    "false"
                })
                .into(),
            ),
            (
                "__SUFFIX__",
                (if artifact.task == Task::Regression {
                    "\"]}\""
                } else {
                    "\"]}}\""
                })
                .into(),
            ),
        ],
    ))
}

/// A private replay plan. Its fixture and native expected values are never public files.
/// The controller executes argv under its existing deadline and resource limits.
/// Caller must supply only explicit synthetic or development rows, never sealed features,
/// and retain their provenance plus actual toolchain/execution receipts beside this private plan.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct QualificationPlan {
    pub schema: String,
    pub bundle_sha256: String,
    pub fitted_sha256: String,
    pub input_sha256: String,
    pub backend: Backend,
    pub parity_policy: ParityPolicy,
    pub files: BTreeMap<String, String>,
    pub commands: Vec<Vec<String>>,
    pub expected: Predictions,
}

pub fn qualification_plan(
    artifact: &FittedRustArtifact,
    bundle: &PublicBundle,
    rows: &[Vec<Option<String>>],
) -> Result<QualificationPlan> {
    qualification_plan_for_view(&FittedInferenceView::registered(artifact)?, bundle, rows)
}
pub fn qualification_plan_synthesized(
    artifact: &FittedSynthesizedArtifact,
    bundle: &PublicBundle,
    rows: &[Vec<Option<String>>],
) -> Result<QualificationPlan> {
    qualification_plan_for_view(&FittedInferenceView::synthesized(artifact)?, bundle, rows)
}
pub fn qualification_plan_for_view(
    artifact: &FittedInferenceView<'_>,
    bundle: &PublicBundle,
    rows: &[Vec<Option<String>>],
) -> Result<QualificationPlan> {
    bundle.validate()?;
    // Re-emission binds all exported coefficients to this exact fitted winner.
    let ExportOutcome::Emitted {
        bundle: expected_bundle,
    } = export_inference(artifact, bundle.manifest.backend)?
    else {
        anyhow::bail!("fitted backend no longer supported")
    };
    ensure!(
        bundle.identity()? == expected_bundle.identity()?,
        "public bundle differs from fitted artifact"
    );
    ensure!(
        !rows.is_empty(),
        "qualification requires nonempty prediction rows"
    );
    let columns = artifact
        .model
        .schema
        .columns
        .iter()
        .map(|column| jope_pipeline::experiment::Column {
            name: column.name.clone(),
            kind: match column.kind {
                ColumnKind::Numeric => jope_pipeline::experiment::ColumnKind::Numeric,
                ColumnKind::Categorical => jope_pipeline::experiment::ColumnKind::Categorical,
            },
        })
        .collect::<Vec<_>>();
    let expected = artifact.predict_raw(&columns, rows)?;
    let input = serde_json::to_string(rows)?;
    let mut files = bundle.files.clone();
    files.insert("qualification-input.json".into(), input.clone());
    let (harness_files, mut commands) = harness(bundle.manifest.backend, rows);
    files.extend(harness_files);
    if let Some(standard) = cpp_standard(bundle.manifest.backend) {
        let command = feat_math::cxx::command(Some(standard));
        let mut argv = vec![command.get_program().to_string_lossy().into_owned()];
        argv.extend(
            command
                .get_args()
                .map(|arg| arg.to_string_lossy().into_owned()),
        );
        argv.extend(CPP_BUILD_FLAGS.map(str::to_owned));
        commands.insert(0, argv);
    }
    let plan = QualificationPlan {
        schema: "jope.fitted-qualification-plan/v1".into(),
        bundle_sha256: bundle.identity()?,
        fitted_sha256: artifact.fitted_sha256.clone(),
        input_sha256: digest(input.as_bytes()),
        backend: bundle.manifest.backend,
        parity_policy: bundle.manifest.parity_policy,
        files,
        commands,
        expected,
    };
    verify_plan_binding(&plan, bundle)?;
    Ok(plan)
}

/// Public C++ library build flags after the host-resolved compiler, the `-std=` flag and
/// an optional `JOPE_CXX_MARCH` `-march=` flag. Floating-point contraction stays off.
const CPP_BUILD_FLAGS: [&str; 8] = [
    "-O2",
    "-ffp-contract=off",
    "-fno-fast-math",
    "-shared",
    "-fPIC",
    "model.cpp",
    "-o",
    "model.so",
];
fn cpp_standard(backend: Backend) -> Option<feat_math::cxx::CxxStd> {
    match backend {
        Backend::Cpp17 => Some(feat_math::cxx::CxxStd::Cxx17),
        Backend::Cpp20 => Some(feat_math::cxx::CxxStd::Cxx20),
        _ => None,
    }
}
/// The deterministic private replay harness: files beyond the public bundle and the
/// unlabeled input, plus the host-independent commands that run them. Row values reach
/// generated code only as JSON data or Rust debug-escaped literals, never as argv.
fn harness(
    backend: Backend,
    rows: &[Vec<Option<String>>],
) -> (BTreeMap<String, String>, Vec<Vec<String>>) {
    let mut files = BTreeMap::new();
    let mut commands = Vec::new();
    match backend {
        Backend::Python | Backend::Cpp17 | Backend::Cpp20 => {
            let module = if backend == Backend::Python {
                "model"
            } else {
                "boundary"
            };
            files.insert("qualification.py".into(),format!("import json\nimport {module}\nwith open('qualification-input.json',encoding='utf-8') as stream:\n    rows=json.load(stream)\nimport sys\nsys.stdout.buffer.write((json.dumps({module}.predict_raw(rows),ensure_ascii=False,allow_nan=False)+'\\n').encode('utf-8'))\n"));
            commands.push(vec!["python3".into(), "qualification.py".into()]);
        }
        Backend::JavaScript => {
            files.insert("qualification.js".into(),"const model=require('./model.js'); const fs=require('fs'); const rows=JSON.parse(fs.readFileSync('qualification-input.json','utf8')); process.stdout.write(JSON.stringify(model.predictRaw(rows))+'\\n');\n".into());
            commands.push(vec!["node".into(), "qualification.js".into()]);
        }
        Backend::Rust => {
            let literal = format!(
                "vec![{}]",
                rows.iter()
                    .map(|row| format!(
                        "vec![{}]",
                        row.iter()
                            .map(|value| value.as_ref().map_or_else(
                                || "None".to_owned(),
                                |text| format!("Some({text:?}.to_owned())")
                            ))
                            .collect::<Vec<_>>()
                            .join(",")
                    ))
                    .collect::<Vec<_>>()
                    .join(",")
            );
            files.insert("qualification.rs".into(),format!("mod model; fn main() {{ let rows:Vec<Vec<Option<String>>>={literal}; println!(\"{{}}\",model::predict_raw(&rows).expect(\"prediction\")); }}\n"));
            commands.push(
                [
                    "rustc",
                    "--edition=2021",
                    "-O",
                    "qualification.rs",
                    "-o",
                    "qualification",
                ]
                .map(str::to_owned)
                .to_vec(),
            );
            commands.push(vec!["./qualification".into()]);
        }
    }
    (files, commands)
}

/// Bind a private qualification plan to the shipped public bundle: the executed files
/// are exactly the bundle files, the unlabeled input and the deterministic harness, and
/// the commands are the fixed harness commands. The C++ build argv is checked except
/// for its host-resolved compiler program, which the pinned tool inventory binds.
/// This proves the executed code is the shipped code; it does not re-run prediction.
pub fn verify_plan_binding(plan: &QualificationPlan, bundle: &PublicBundle) -> Result<()> {
    bundle.validate()?;
    let backend = bundle.manifest.backend;
    ensure!(
        plan.schema == "jope.fitted-qualification-plan/v1"
            && plan.backend == backend
            && plan.parity_policy == backend.parity_policy()
            && plan.bundle_sha256 == bundle.identity()?,
        "qualification plan is bound to a different public bundle"
    );
    let input = plan
        .files
        .get("qualification-input.json")
        .context("qualification input absent")?;
    ensure!(
        digest(input.as_bytes()) == plan.input_sha256,
        "qualification input identity changed"
    );
    let rows: Vec<Vec<Option<String>>> =
        serde_json::from_str(input).context("qualification input rows")?;
    ensure!(
        !rows.is_empty() && serde_json::to_string(&rows)? == *input,
        "qualification input is not canonical nonempty rows"
    );
    let (harness_files, commands) = harness(backend, &rows);
    let mut files = bundle.files.clone();
    files.insert("qualification-input.json".into(), input.clone());
    files.extend(harness_files);
    ensure!(
        plan.files == files,
        "executed qualification files differ from the shipped public bundle"
    );
    let predictor = if let Some(standard) = cpp_standard(backend) {
        let build = plan.commands.first().context("C++ build absent")?;
        let prefix = build.len().saturating_sub(CPP_BUILD_FLAGS.len());
        ensure!(
            matches!(prefix, 2 | 3)
                && !build[0].is_empty()
                && build[1] == standard.flag()
                && (prefix == 2 || build[2].starts_with("-march="))
                && build[prefix..]
                    .iter()
                    .map(String::as_str)
                    .eq(CPP_BUILD_FLAGS),
            "C++ qualification build changed"
        );
        &plan.commands[1..]
    } else {
        &plan.commands[..]
    };
    ensure!(
        predictor == commands.as_slice(),
        "qualification commands changed"
    );
    Ok(())
}

impl QualificationPlan {
    pub fn identity(&self) -> Result<String> {
        Ok(digest(&serde_json::to_vec(self)?))
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct QualificationResult {
    pub schema: String,
    pub plan_sha256: String,
    pub bundle_sha256: String,
    pub fitted_sha256: String,
    pub input_sha256: String,
    pub prediction_sha256: String,
    pub backend: Backend,
    pub values: usize,
    pub maximum_absolute_error: f64,
    pub parity_policy: ParityPolicy,
    pub execution_receipt_required: bool,
}

/// Verify successful execution output, preserving original class order and full matrices.
/// Execution failures/unavailable compilers are controller failures, never passing receipts.
pub fn verify_qualification(
    plan: &QualificationPlan,
    expected_plan_sha256: &str,
    output: &[u8],
) -> Result<QualificationResult> {
    ensure!(
        plan.schema == "jope.fitted-qualification-plan/v1"
            && plan.parity_policy == plan.backend.parity_policy(),
        "qualification contract changed"
    );
    ensure!(
        plan.identity()? == expected_plan_sha256,
        "qualification plan identity changed"
    );
    ensure!(
        plan.files
            .get("qualification-input.json")
            .is_some_and(|input| digest(input.as_bytes()) == plan.input_sha256),
        "qualification input identity changed"
    );
    let actual: Predictions =
        serde_json::from_slice(output).context("decode emitted predictions")?;
    let (actual, expected) = match (&actual, &plan.expected) {
        (Predictions::Values(a), Predictions::Values(e)) => {
            ensure!(a.len() == e.len(), "prediction row count differs");
            (a.clone(), e.clone())
        }
        (
            Predictions::Probabilities {
                classes: a,
                values: av,
            },
            Predictions::Probabilities {
                classes: e,
                values: ev,
            },
        ) => {
            ensure!(
                a == e
                    && av.len() == ev.len()
                    && av.iter().all(|row| row.len() == a.len())
                    && ev.iter().all(|row| row.len() == e.len()),
                "probability matrix dimensions or class order differ"
            );
            ensure!(
                av.iter().all(|row| row
                    .iter()
                    .all(|value| value.is_finite() && *value >= 0.0 && *value <= 1.0)
                    && (row.iter().sum::<f64>() - 1.0).abs() <= 1e-6),
                "invalid fitted probability matrix"
            );
            (
                av.iter().flatten().copied().collect(),
                ev.iter().flatten().copied().collect(),
            )
        }
        _ => anyhow::bail!("prediction task representation differs"),
    };
    ensure!(!actual.is_empty(), "qualification produced zero values");
    let mut maximum = 0.0f64;
    for (a, e) in actual.iter().zip(&expected) {
        ensure!(
            a.is_finite() && e.is_finite(),
            "nonfinite qualification prediction"
        );
        let error = (a - e).abs();
        maximum = maximum.max(error);
        ensure!(
            match plan.parity_policy {
                ParityPolicy::ExactBits => a.to_bits() == e.to_bits(),
                ParityPolicy::AbsoluteRelative { absolute, relative } =>
                    error <= absolute + relative * e.abs(),
            },
            "emitted prediction differs under declared numerical policy"
        );
    }
    Ok(QualificationResult {
        schema: "jope.fitted-qualification-result/v1".into(),
        plan_sha256: expected_plan_sha256.to_owned(),
        bundle_sha256: plan.bundle_sha256.clone(),
        fitted_sha256: plan.fitted_sha256.clone(),
        input_sha256: plan.input_sha256.clone(),
        prediction_sha256: digest(output),
        backend: plan.backend,
        values: actual.len(),
        maximum_absolute_error: maximum,
        parity_policy: plan.parity_policy,
        execution_receipt_required: true,
    })
}

/// Pre-fit inventory: this reports implementation coverage, never numerical qualification.
/// Every registered winner retains complete private native state even if export is unavailable.
pub fn registered_inference_inventory() -> Result<Vec<serde_json::Value>> {
    let mut result = Vec::new();
    for task in [Task::Regression, Task::Binary, Task::Multiclass] {
        for method in super::methods::rust_methods(task)? {
            let python_rust = matches!(
                (task, method.family.as_str()),
                (
                    Task::Regression,
                    "ridge" | "tab_boost" | "extra_trees" | "gam_boost" | "pls"
                ) | (
                    Task::Binary,
                    "logit_ridge" | "tab_boost" | "extra_trees" | "gam_boost" | "portfolio"
                ) | (
                    Task::Multiclass,
                    "multiclass_ridge"
                        | "multiclass_ovr_logistic"
                        | "multiclass_diagonal_lda"
                        | "multiclass_nearest_centroid"
                )
            );
            let javascript = matches!(
                (task, method.family.as_str()),
                (Task::Regression, "ridge")
                    | (Task::Binary, "logit_ridge")
                    | (
                        Task::Multiclass,
                        "multiclass_ridge"
                            | "multiclass_ovr_logistic"
                            | "multiclass_diagonal_lda"
                            | "multiclass_nearest_centroid"
                    )
            );
            let cpp = task != Task::Multiclass
                && matches!(method.family.as_str(), "ridge" | "logit_ridge");
            result.push(serde_json::json!({"task":task,"method":method.id,"native_private_state":"complete-bound-model","native_reload":"supported",
                "standalone_python":{"supported":python_rust,"reason":if python_rust{None}else{Some(concat!("legacy fitted head lowering not ", "implemented"))}},
                "standalone_rust":{"supported":python_rust,"reason":if python_rust{None}else{Some(concat!("legacy fitted head lowering not ", "implemented"))}},
                "javascript":{"supported":javascript,"reason":if javascript{None}else{Some("head outside separately qualified JavaScript subset")}},
                "cpp17_cpp20":{"supported":cpp,"reason":if cpp{None}else if task==Task::Multiclass{Some("scalar ABI cannot return probability matrix")}else{Some("head outside public C++ subset")}},
                "qualification":"requires exact fitted artifact and input replay","unsupported_export_replaces_winner":false}));
        }
    }
    Ok(result)
}

#[path = "inference_package.rs"]
pub mod package;

#[cfg(test)]
#[path = "inference_contract_tests.rs"]
mod contract_tests;

#[cfg(test)]
#[path = "inference/tests.rs"]
pub(crate) mod tests;
