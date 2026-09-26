//! Allowlisted fitted inference exports and private qualification plans.
//! Exporting does not qualify a backend, refit a model, or replace the selected method.
use super::{
    data::RAW_CELL_ENCODING,
    methods::{digest, FittedRustArtifact, INPUT_POLICY},
    score::{Predictions, NUMERICAL_POLICY},
    synthesized::FittedSynthesizedArtifact,
};
use anyhow::{ensure, Context, Result};
use feat_math::advanced_v3::{
    emit_fitted_portable, emit_fitted_public_cpp17, fitted_javascript_unsupported,
    fitted_portable_templates, fitted_portable_unsupported, fitted_public_cpp_unsupported,
    lint_fitted_public_text, ColumnKind, ColumnSpec, PortableUnsupported, FITTED_PORTABLE_EMITTER,
    FITTED_PUBLIC_CPP_EMITTER,
};
use jope_datasets::catalog::Task;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Backend {
    Python,
    Rust,
    Cpp17,
    Cpp20,
    #[serde(rename = "javascript")]
    JavaScript,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum ParityPolicy {
    ExactBits,
    AbsoluteRelative { absolute: f64, relative: f64 },
}

impl Backend {
    pub fn numerical_policy(self) -> String {
        if self == Self::JavaScript {
            NUMERICAL_POLICY.replace("platform-libm-v1", "ecmascript-math-es2019-v1")
        } else {
            NUMERICAL_POLICY.into()
        }
    }
    fn emitter(self) -> &'static str {
        if matches!(self, Self::Cpp17 | Self::Cpp20) {
            FITTED_PUBLIC_CPP_EMITTER
        } else {
            FITTED_PORTABLE_EMITTER
        }
    }
    fn runtime(self) -> &'static str {
        match self {
            Self::JavaScript => "ES2019 or later; Node.js 12 or later; ECMAScript Math",
            Self::Python => "Python 3.10 or later; host libm",
            Self::Rust => "Rust edition 2021; host libm",
            Self::Cpp17 => "C++17 plus Python 3.10 ctypes typed-table host; host libm",
            Self::Cpp20 => "C++20 plus Python 3.10 ctypes typed-table host; host libm",
        }
    }
    fn host_boundary(self) -> &'static str {
        if matches!(self, Self::Cpp17 | Self::Cpp20) {
            "python-ctypes-typed-table"
        } else {
            "raw-rows"
        }
    }
    pub fn parity_policy(self) -> ParityPolicy {
        match self {
            Self::JavaScript => ParityPolicy::AbsoluteRelative {
                absolute: 1e-12,
                relative: 1e-12,
            },
            _ => ParityPolicy::ExactBits,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PublicManifest {
    pub schema: String,
    pub backend: Backend,
    pub task: Task,
    pub columns: Vec<ColumnSpec>,
    pub classes: Vec<String>,
    pub raw_cell_encoding: String,
    pub input_policy: String,
    pub numerical_policy: String,
    pub emitter: String,
    pub template_sha256: BTreeMap<String, String>,
    pub runtime: String,
    pub host_boundary: String,
    pub parity_policy: ParityPolicy,
    pub qualification: String,
    pub entrypoint: String,
    pub files: BTreeMap<String, String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PublicBundle {
    pub manifest: PublicManifest,
    pub files: BTreeMap<String, String>,
}
impl PublicBundle {
    pub fn identity(&self) -> Result<String> {
        Ok(digest(&serde_json::to_vec(self)?))
    }
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.manifest.schema == "jope.fitted-public-bundle/v1"
                && self.manifest.qualification == "emitted-unqualified"
                && self.manifest.raw_cell_encoding == RAW_CELL_ENCODING
                && matches!(
                    self.manifest.input_policy.as_str(),
                    INPUT_POLICY | FITTED_INPUT_POLICY
                )
                && self.manifest.numerical_policy == self.manifest.backend.numerical_policy()
                && self.manifest.emitter == self.manifest.backend.emitter()
                && self.manifest.runtime == self.manifest.backend.runtime()
                && self.manifest.host_boundary == self.manifest.backend.host_boundary()
                && self.manifest.template_sha256 == templates(self.manifest.backend)
                && self.manifest.parity_policy == self.manifest.backend.parity_policy(),
            "unsupported public bundle contract"
        );
        let expected: &[&str] = match self.manifest.backend {
            Backend::Python => &["model.py"],
            Backend::Rust => &["model.rs"],
            Backend::JavaScript => &["model.js"],
            Backend::Cpp17 | Backend::Cpp20 => &["model.cpp", "codec.json", "boundary.py"],
        };
        ensure!(
            self.files.len() == expected.len()
                && expected.iter().all(|name| self.files.contains_key(*name)),
            "public bundle file allowlist differs"
        );
        ensure!(
            self.manifest.files
                == self
                    .files
                    .iter()
                    .map(|(path, text)| (path.clone(), digest(text.as_bytes())))
                    .collect(),
            "public file checksum differs"
        );
        Ok(())
    }
}

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
    if let Some(reason) = fitted_portable_unsupported(&artifact.model)? {
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
        Backend::JavaScript => fitted_javascript_unsupported(&artifact.model)?,
        Backend::Cpp17 | Backend::Cpp20 => fitted_public_cpp_unsupported(&artifact.model)?,
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
            lint_fitted_public_text(&artifact.model, &artifact.classes, text).is_empty(),
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
        ("boundary.js", include_str!("inference_boundary.js")),
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
            let emitted = emit_fitted_portable(&artifact.model)?;
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
                        include_str!("inference_boundary.js"),
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
            let emitted = emit_fitted_public_cpp17(&artifact.model)?;
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
                "standalone_python":{"supported":python_rust,"reason":if python_rust{None}else{Some("legacy fitted head lowering not implemented")}},
                "standalone_rust":{"supported":python_rust,"reason":if python_rust{None}else{Some("legacy fitted head lowering not implemented")}},
                "javascript":{"supported":javascript,"reason":if javascript{None}else{Some("head outside separately qualified JavaScript subset")}},
                "cpp17_cpp20":{"supported":cpp,"reason":if cpp{None}else if task==Task::Multiclass{Some("scalar ABI cannot return probability matrix")}else{Some("head outside public C++ subset")}},
                "qualification":"requires exact fitted artifact and input replay","unsupported_export_replaces_winner":false}));
        }
    }
    Ok(result)
}

#[cfg(test)]
#[path = "inference_contract_tests.rs"]
mod contract_tests;

#[cfg(test)]
pub(crate) mod tests {
    use super::super::data::raw_table;
    use super::super::methods::{fit_rust, rust_methods, RustFitRequest};
    use super::*;
    use feat_math::advanced_v3::{DataSchema, RowSemantics};
    use jope_pipeline::experiment::{Column, ColumnKind as Kind};
    use std::process::Command;

    pub(crate) fn fixture(
        task: Task,
        family: &str,
    ) -> (FittedRustArtifact, Vec<Vec<Option<String>>>) {
        let (request, rows) = fixture_request(task, family);
        (fit_rust(&request).unwrap().artifact, rows)
    }
    pub(crate) fn fixture_request(
        task: Task,
        family: &str,
    ) -> (RustFitRequest, Vec<Vec<Option<String>>>) {
        let columns = vec![
            Column {
                name: "amount".into(),
                kind: Kind::Numeric,
            },
            Column {
                name: "country".into(),
                kind: Kind::Categorical,
            },
        ];
        let categories = [
            None,
            Some("NA"),
            Some("null"),
            Some(""),
            Some("line\n\"quoted\"\u{2028}"),
            Some("__PREDICT____REGRESSION____CLASSES_JSON__"),
        ];
        let rows = (0..18)
            .map(|i| {
                vec![
                    if i % 7 == 0 {
                        None
                    } else {
                        Some(i.to_string())
                    },
                    categories[i % categories.len()].map(str::to_owned),
                ]
            })
            .collect::<Vec<_>>();
        let prediction = vec![
            vec![None, Some("NA".into())],
            vec![Some("1000000".into()), Some("unseen".into())],
            vec![Some("-0.0".into()), None],
            vec![Some("3.25".into()), Some("line\n\"quoted\"\u{2028}".into())],
        ];
        let classes = match task {
            Task::Regression => vec![],
            Task::Binary => vec!["tree\nquoted\"".into(), "search".into()],
            _ => vec!["zebra".into(), "alpha".into(), "middle".into()],
        };
        let targets = (0..18)
            .map(|i| {
                if task == Task::Regression {
                    (i as f64) * 1.25 - 3.0
                } else {
                    (i % classes.len()) as f64
                }
            })
            .collect();
        let request = RustFitRequest {
            task,
            schema: DataSchema {
                columns: vec![
                    ColumnSpec::numeric("amount"),
                    ColumnSpec::categorical("country"),
                ],
                row_semantics: RowSemantics::Iid,
            },
            classes,
            train: raw_table(&columns, &rows).unwrap(),
            targets,
            predict: raw_table(&columns, &prediction).unwrap(),
            method: rust_methods(task)
                .unwrap()
                .into_iter()
                .find(|method| method.family == family)
                .unwrap(),
        };
        (request, prediction)
    }
    pub(crate) fn emitted(artifact: &FittedRustArtifact, backend: Backend) -> PublicBundle {
        match export_fitted(artifact, backend).unwrap() {
            ExportOutcome::Emitted { bundle } => *bundle,
            ExportOutcome::Unsupported { reason, .. } => panic!("{backend:?}: {reason:?}"),
        }
    }

    #[test]
    fn public_files_are_allowlisted_and_never_contain_private_recipe_or_envelope() {
        for (task, head) in [
            (Task::Regression, "ridge"),
            (Task::Binary, "logit_ridge"),
            (Task::Multiclass, "multiclass_ridge"),
        ] {
            let (artifact, _) = fixture(task, head);
            for backend in [
                Backend::Python,
                Backend::Rust,
                Backend::Cpp17,
                Backend::Cpp20,
                Backend::JavaScript,
            ] {
                if task == Task::Multiclass && matches!(backend, Backend::Cpp17 | Backend::Cpp20) {
                    continue;
                }
                let bundle = emitted(&artifact, backend);
                for text in bundle.files.values().chain(std::iter::once(
                    &serde_json::to_string(&bundle.manifest).unwrap(),
                )) {
                    assert!(
                        lint_fitted_public_text(&artifact.model, &artifact.classes, text)
                            .is_empty()
                    );
                }
                let public = serde_json::to_string(&bundle).unwrap();
                for canary in [
                    &artifact.method.dsl,
                    &artifact.model.genome_sha256,
                    "genome_sha256",
                    "jope.panel-fitted-rust/v1",
                    "discovery_provenance",
                ] {
                    assert!(
                        !public.contains(canary),
                        "private canary in {backend:?}: {canary}"
                    );
                }
                assert_eq!(bundle.manifest.qualification, "emitted-unqualified");
                assert_eq!(bundle.manifest.parity_policy, backend.parity_policy());
                let mut tampered = bundle.clone();
                tampered.files.insert(
                    "private-recipe.json".into(),
                    serde_json::to_string(&artifact).unwrap(),
                );
                assert!(tampered.validate().is_err());
                let mut tampered = bundle;
                tampered.files.values_mut().next().unwrap().push(' ');
                assert!(tampered.validate().is_err());
            }
        }
        let (artifact, _) = fixture(Task::Regression, "ridge");
        let cpp17 = emitted(&artifact, Backend::Cpp17);
        let cpp20 = emitted(&artifact, Backend::Cpp20);
        assert_eq!(
            cpp17.files, cpp20.files,
            "one C++ source and codec for both standards"
        );
    }

    #[test]
    fn generated_winners_preserve_raw_boundary_and_private_provenance_on_export() {
        use super::super::synthesized::{fit_synthesized, SynthesizedFitRequest};
        use jope_pipeline::synthesized_recipe::SynthesizedRecipe;
        let scratch = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../target/fitted-inference-tests");
        std::fs::create_dir_all(&scratch).unwrap();
        let root = tempfile::tempdir_in(scratch).unwrap();
        let mut replays = 0;
        for (task, head) in [
            (Task::Regression, "extra_trees"),
            (Task::Binary, "portfolio"),
            (Task::Multiclass, "multiclass_diagonal_lda"),
        ] {
            let (request, rows) = fixture_request(task, head);
            // Explicit generated input need not equal the roster recipe's median.
            let proposal = format!(
                "  {}\n",
                request.method.dsl.replace("numeric=median", "numeric=mean")
            );
            let request = SynthesizedFitRequest {
                task,
                schema: request.schema,
                classes: request.classes,
                train: request.train,
                targets: request.targets,
                predict: request.predict,
                recipe: SynthesizedRecipe::new(proposal).unwrap(),
                scheduling_seed: 17,
            };
            let result = fit_synthesized(&request).unwrap();
            let artifact = result.artifact;
            let private = artifact.bytes().unwrap();
            for backend in [Backend::Python, Backend::Rust] {
                let ExportOutcome::Emitted { bundle } =
                    export_synthesized(&artifact, backend).unwrap()
                else {
                    panic!("generated fitted head unsupported")
                };
                assert_eq!(bundle.manifest.input_policy, FITTED_INPUT_POLICY);
                let public = serde_json::to_string(&bundle).unwrap();
                for canary in [
                    &artifact.recipe.original_proposal,
                    &artifact.recipe.executable_canonical,
                    &artifact.recipe.executable_sha256,
                    "original_proposal",
                    "jope.panel-fitted-synthesized/v1",
                    "discovery_provenance",
                ] {
                    assert!(
                        !public.contains(canary),
                        "private generated provenance leaked"
                    );
                }
                let plan = qualification_plan_synthesized(&artifact, &bundle, &rows).unwrap();
                assert_eq!(plan.fitted_sha256, digest(&private));
                let directory = root.path().join(format!("{task:?}-{backend:?}"));
                std::fs::create_dir(&directory).unwrap();
                for (name, text) in &plan.files {
                    std::fs::write(directory.join(name), text).unwrap();
                }
                let mut output = Vec::new();
                for args in &plan.commands {
                    let run = Command::new(&args[0])
                        .args(&args[1..])
                        .current_dir(&directory)
                        .env("OMP_NUM_THREADS", "1")
                        .env("OPENBLAS_NUM_THREADS", "1")
                        .output()
                        .unwrap();
                    assert!(
                        run.status.success(),
                        "{head} {backend:?}: {}",
                        String::from_utf8_lossy(&run.stderr)
                    );
                    output = run.stdout;
                }
                verify_qualification(&plan, &plan.identity().unwrap(), &output).unwrap();
                replays += 1;
            }
            assert_eq!(artifact.bytes().unwrap(), private);
            let mut broken = artifact.clone();
            broken.recipe.original_proposal.push(' ');
            assert!(export_synthesized(&broken, Backend::Python).is_err());
        }
        assert_eq!(replays, 6);
        let (request, _) = fixture_request(Task::Regression, "ridge");
        let recipe =
            SynthesizedRecipe::new(request.method.dsl.replace("(ridge", "(standardize) (ridge"))
                .unwrap();
        let artifact = fit_synthesized(&SynthesizedFitRequest {
            task: request.task,
            schema: request.schema,
            classes: request.classes,
            train: request.train,
            targets: request.targets,
            predict: request.predict,
            recipe,
            scheduling_seed: 17,
        })
        .unwrap()
        .artifact;
        let original = artifact.bytes().unwrap();
        assert!(matches!(
            export_synthesized(&artifact, Backend::Python).unwrap(),
            ExportOutcome::Unsupported {
                reason: UnsupportedReason::FittedState(PortableUnsupported::Operations),
                ..
            }
        ));
        assert_eq!(artifact.bytes().unwrap(), original);
    }

    #[test]
    fn complete_registry_has_explicit_pre_fit_inference_coverage() {
        let rows = registered_inference_inventory().unwrap();
        assert_eq!(rows.len(), 26);
        assert_eq!(
            rows.iter()
                .filter(|r| r["standalone_python"]["supported"] == true)
                .count(),
            14
        );
        assert_eq!(
            rows.iter()
                .filter(|r| r["standalone_rust"]["supported"] == true)
                .count(),
            14
        );
        assert_eq!(
            rows.iter()
                .filter(|r| r["javascript"]["supported"] == true)
                .count(),
            6
        );
        assert!(rows
            .iter()
            .all(|r| r["unsupported_export_replaces_winner"] == false));
    }
    #[test]
    fn extended_scalar_export_keeps_auxiliary_backends_explicitly_unsupported() {
        let (artifact, _) = fixture(Task::Binary, "portfolio");
        let original = artifact.bytes().unwrap();
        for backend in [Backend::Cpp17, Backend::Cpp20, Backend::JavaScript] {
            assert!(matches!(
                export_fitted(&artifact, backend).unwrap(),
                ExportOutcome::Unsupported {
                    reason: UnsupportedReason::FittedState(PortableUnsupported::Head),
                    ..
                }
            ));
        }
        for backend in [Backend::Python, Backend::Rust] {
            let bundle = emitted(&artifact, backend);
            let text = serde_json::to_string(&bundle).unwrap();
            for canary in [
                &artifact.method.dsl,
                &artifact.model.genome_sha256,
                "genome_sha256",
                "\"sums\"",
                "\"counts\"",
                "discovery_provenance",
            ] {
                assert!(!text.contains(canary));
            }
        }
        assert_eq!(artifact.bytes().unwrap(), original);
    }

    #[test]
    fn unsupported_backend_keeps_the_fitted_winner_unchanged() {
        let (artifact, _) = fixture(Task::Regression, "histgbm");
        let before = artifact.bytes().unwrap();
        assert!(matches!(
            export_fitted(&artifact, Backend::JavaScript).unwrap(),
            ExportOutcome::Unsupported {
                reason: UnsupportedReason::FittedState(PortableUnsupported::Head),
                ..
            }
        ));
        assert_eq!(artifact.bytes().unwrap(), before);
        let (artifact, _) = fixture(Task::Multiclass, "multiclass_ridge");
        assert!(matches!(
            export_fitted(&artifact, Backend::Cpp17).unwrap(),
            ExportOutcome::Unsupported {
                reason: UnsupportedReason::CppMatrix,
                ..
            }
        ));
    }

    #[test]
    fn all_matrix_heads_and_scalar_boundaries_compile_and_replay_synthetic_fitted_state() {
        let scratch = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../target/fitted-inference-tests");
        std::fs::create_dir_all(&scratch).unwrap();
        let root = tempfile::tempdir_in(scratch).unwrap();
        let cases = [
            (Task::Regression, "tab_boost"),
            (Task::Regression, "extra_trees"),
            (Task::Regression, "gam_boost"),
            (Task::Regression, "pls"),
            (Task::Binary, "tab_boost"),
            (Task::Binary, "extra_trees"),
            (Task::Binary, "gam_boost"),
            (Task::Binary, "portfolio"),
            (Task::Regression, "ridge"),
            (Task::Binary, "logit_ridge"),
            (Task::Multiclass, "multiclass_ridge"),
            (Task::Multiclass, "multiclass_ovr_logistic"),
            (Task::Multiclass, "multiclass_diagonal_lda"),
            (Task::Multiclass, "multiclass_nearest_centroid"),
        ];
        let mut executed = 0;
        for (task, head) in cases {
            let (artifact, rows) = fixture(task, head);
            let mut backends = vec![Backend::Python, Backend::Rust];
            if fitted_javascript_unsupported(&artifact.model)
                .unwrap()
                .is_none()
            {
                backends.push(Backend::JavaScript);
            }
            if fitted_public_cpp_unsupported(&artifact.model)
                .unwrap()
                .is_none()
            {
                backends.extend([Backend::Cpp17, Backend::Cpp20]);
            }
            for backend in backends {
                let bundle = emitted(&artifact, backend);
                let plan = qualification_plan(&artifact, &bundle, &rows).unwrap();
                let plan_identity = plan.identity().unwrap();
                let directory = root.path().join(format!("{task:?}-{head}-{backend:?}"));
                std::fs::create_dir(&directory).unwrap();
                for (path, text) in &plan.files {
                    std::fs::write(directory.join(path), text).unwrap();
                }
                let mut output = Vec::new();
                for argv in &plan.commands {
                    let result = Command::new(&argv[0])
                        .args(&argv[1..])
                        .current_dir(&directory)
                        .env("OMP_NUM_THREADS", "1")
                        .env("OPENBLAS_NUM_THREADS", "1")
                        .output()
                        .unwrap_or_else(|error| {
                            panic!("required tool {} unavailable: {error}", argv[0])
                        });
                    assert!(
                        result.status.success(),
                        "{head} {backend:?} {argv:?}: {}",
                        String::from_utf8_lossy(&result.stderr)
                    );
                    output = result.stdout;
                }
                let receipt =
                    verify_qualification(&plan, &plan_identity, &output).unwrap_or_else(|error| {
                        panic!(
                            "{head} {backend:?}: {error:#}; {}",
                            String::from_utf8_lossy(&output)
                        )
                    });
                assert_eq!(
                    receipt.values,
                    rows.len()
                        * if task == Task::Regression {
                            1
                        } else {
                            artifact.classes.len()
                        }
                );
                assert_eq!(receipt.plan_sha256, plan_identity);
                assert!(verify_qualification(&plan, &"0".repeat(64), &output).is_err());
                let mut stale = plan.clone();
                stale
                    .files
                    .get_mut("qualification-input.json")
                    .unwrap()
                    .push(' ');
                assert!(verify_qualification(&stale, &plan_identity, &output).is_err());
                if let Predictions::Probabilities {
                    mut classes,
                    values,
                } = serde_json::from_slice::<Predictions>(&output).unwrap()
                {
                    classes.swap(0, 1);
                    let changed =
                        serde_json::to_vec(&Predictions::Probabilities { classes, values })
                            .unwrap();
                    assert!(verify_qualification(&plan, &plan_identity, &changed).is_err());
                }
                executed += 1;
            }
        }
        assert_eq!(executed, 38);
    }
    #[test]
    fn fitted_boundary_template_bytes_are_pinned_for_version_one() {
        let actual = templates(Backend::Python);
        for (name, sha) in [
            (
                "boundary.py",
                "88692ca12accbc84f741b0b632e299e2e5a8527a889de50aa10692431182cbdf",
            ),
            (
                "boundary.rs",
                "e3aec495101ca9e21aafe823e52ca05660029127eb084556a0a8ce32a2ce46fb",
            ),
            (
                "boundary.js",
                "5a13d2e63c13903691b641734692aa914e5e96d6fddcfad9e3894b0ac86a96be",
            ),
            (
                "typed-host.py",
                "1a05ba9e745daec635a3b20d9a8dba7426840be139e0cdf00b0e48f8578b6b1b",
            ),
        ] {
            assert_eq!(actual[name], sha);
        }
    }
    #[test]
    fn invalid_fitted_coefficients_fail_instead_of_becoming_unsupported() {
        let (artifact, _) = fixture(Task::Regression, "ridge");
        let mut encoded = serde_json::to_value(&artifact).unwrap();
        fn corrupt(value: &mut serde_json::Value) -> bool {
            match value {
                serde_json::Value::Object(fields) => {
                    if let Some(serde_json::Value::Array(scales)) = fields.get_mut("scales") {
                        if let Some(first) = scales.first_mut() {
                            *first = serde_json::json!(0.0);
                            return true;
                        }
                    }
                    fields.values_mut().any(corrupt)
                }
                serde_json::Value::Array(items) => items.iter_mut().any(corrupt),
                _ => false,
            }
        }
        assert!(corrupt(&mut encoded));
        let altered: FittedRustArtifact = serde_json::from_value(encoded).unwrap();
        for backend in [
            Backend::Python,
            Backend::Rust,
            Backend::Cpp17,
            Backend::Cpp20,
            Backend::JavaScript,
        ] {
            assert!(export_fitted(&altered, backend).is_err(), "{backend:?}");
        }
    }
}
