//! Opt-in target binding. Hashes prove retained bytes, not execution or authority.
use crate::transpile::EmitOptions;
use anyhow::{ensure, Context, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Component, Path};

pub const SCHEMA: &str = "jope.battle-artifact/v3";
pub const ENVELOPE: &str = "jope.canonical-json/v1";
const FORMULA: &str = "sha256(schema_utf8||0x00||u64_be(len)||canonical_payload_bytes)";

pub fn sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Target {
    Python,
    Rust,
    Cpp17,
    Cpp20,
    #[serde(rename = "javascript")]
    JavaScript,
    #[serde(rename = "typescript")]
    TypeScript,
    Cpp17RecipeV2,
    Cpp17Fitted,
}
impl Target {
    pub const RECIPES: [Self; 5] = [
        Self::Python,
        Self::Rust,
        Self::Cpp17,
        Self::Cpp20,
        Self::JavaScript,
    ];
    pub fn parse(value: &str) -> Result<Self> {
        Ok(match value.trim().to_ascii_lowercase().as_str() {
            "python" | "py" => Self::Python,
            "rust" | "rs" => Self::Rust,
            "cpp17" | "c++17" => Self::Cpp17,
            "cpp20" | "c++20" => Self::Cpp20,
            "javascript" | "js" => Self::JavaScript,
            "typescript" | "ts" => Self::TypeScript,
            "cpp17-recipe-v2" => Self::Cpp17RecipeV2,
            "cpp17-fitted" => Self::Cpp17Fitted,
            _ => anyhow::bail!("unknown requested target {value:?}"),
        })
    }
    /// New requests refuse duplicates, including aliases. Legacy parsers are unchanged.
    pub fn parse_list(value: &str) -> Result<Vec<Self>> {
        let mut out = Vec::new();
        for word in value.split(',') {
            if word.trim().eq_ignore_ascii_case("all") {
                out.extend(Self::RECIPES);
            } else {
                out.push(Self::parse(word)?);
            }
        }
        validate_requests(&out)?;
        out.sort();
        Ok(out)
    }
    pub fn path(self) -> &'static str {
        match self {
            Self::Python => "model.py",
            Self::Rust => "model.rs",
            Self::Cpp17 | Self::Cpp20 => "model.cpp",
            Self::JavaScript => "model.js",
            Self::TypeScript => "model.ts",
            Self::Cpp17RecipeV2 => "cpp17-recipe-v2/model.cpp",
            Self::Cpp17Fitted => "cpp17-fitted/model.cpp",
        }
    }
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Python => "python",
            Self::Rust => "rust",
            Self::Cpp17 => "cpp17",
            Self::Cpp20 => "cpp20",
            Self::JavaScript => "javascript",
            Self::TypeScript => "typescript",
            Self::Cpp17RecipeV2 => "cpp17-recipe-v2",
            Self::Cpp17Fitted => "cpp17-fitted",
        }
    }
    pub fn registry_target(self) -> &'static str {
        match self {
            Self::Python => "python_recipe",
            Self::Rust | Self::TypeScript => "rust_recipe",
            Self::Cpp17 | Self::Cpp20 => "cpp17_recipe",
            Self::JavaScript => "javascript",
            Self::Cpp17RecipeV2 => "cpp17_recipe_v2",
            Self::Cpp17Fitted => "cpp17_fitted",
        }
    }
    pub fn product(self, dialect: &str) -> &'static str {
        match self {
            Self::Cpp17RecipeV2 => "expression-recipe-v2",
            Self::Cpp17Fitted => "fitted-cpp17",
            _ if dialect == "advanced_v3" => "bound-predictor-v2",
            _ => "training-recipe-v1",
        }
    }
}
fn validate_requests(requested: &[Target]) -> Result<()> {
    ensure!(!requested.is_empty(), "empty requested target set");
    ensure!(
        requested.iter().collect::<BTreeSet<_>>().len() == requested.len(),
        "duplicate requested target"
    );
    Ok(())
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OptionsIdentity {
    pub title: Option<String>,
    pub feature_names: Vec<String>,
    pub context: Option<String>,
    pub fast_runtime: bool,
    pub math_policy: String,
    pub guard_zero_width_roll: bool,
    pub partition_point_quantile_bounds: bool,
}
impl From<&EmitOptions> for OptionsIdentity {
    fn from(o: &EmitOptions) -> Self {
        Self {
            title: o.title.clone(),
            feature_names: o.feature_names.clone(),
            context: o.context.clone(),
            fast_runtime: o.fast_runtime,
            math_policy: o.math_kernel.policy_id().into(),
            guard_zero_width_roll: o.guard_zero_width_roll,
            partition_point_quantile_bounds: o.partition_point_quantile_bounds,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FileBinding {
    pub path: String,
    pub bytes: u64,
    pub sha256: String,
}
impl FileBinding {
    pub fn new(path: &str, bytes: &[u8]) -> Self {
        Self {
            path: path.into(),
            bytes: bytes.len() as u64,
            sha256: sha256(bytes),
        }
    }
}

/// A recipe only. No compiler version or executable parity is asserted here.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BuildIdentity {
    pub toolchain: String,
    pub standard: String,
    pub flags: Vec<String>,
    pub backend: String,
    pub abi: String,
    pub execution: BuildExecution,
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BuildExecution {
    Unknown,
}
impl BuildIdentity {
    pub fn declared(target: Target) -> Self {
        let (toolchain, standard, flags, abi) = match target {
            Target::Python => ("python3".into(), "python3".into(), vec![], "script"),
            Target::Rust => (
                "rustc".into(),
                "rust2021".into(),
                vec!["-O".into()],
                "script",
            ),
            Target::JavaScript => ("node".into(), "es2020-commonjs".into(), vec![], "commonjs"),
            Target::TypeScript => (
                "rustc+tsc".into(),
                "typescript-es2020+wasm32".into(),
                vec!["--strict".into(), "--target=wasm32-unknown-unknown".into()],
                "typescript-wasm-v1",
            ),
            _ => {
                let standard = if target == Target::Cpp20 {
                    "c++20"
                } else {
                    "c++17"
                };
                let mut flags = vec![format!("-std={standard}"), "-O3".into()];
                flags.extend(crate::cxx::march_flag(None));
                if target == Target::Cpp17Fitted {
                    flags.push("-DJOPE_STANDALONE".into());
                }
                flags.extend(["-ffp-contract=off".into(), "-fno-fast-math".into()]);
                (crate::cxx::compiler(), standard.into(), flags, "c")
            }
        };
        Self {
            toolchain,
            standard,
            flags,
            backend: if target == Target::TypeScript { "wasm32-scalar" } else { "cpu-scalar" }.into(),
            abi: abi.into(),
            execution: BuildExecution::Unknown,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FittedIdentity {
    pub state_sha256: String,
    pub input_sha256: String,
    pub task_sha256: String,
    pub schema_sha256: String,
    pub vocabulary_sha256: String,
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "snake_case", deny_unknown_fields)]
pub enum TargetState {
    Supported {
        product: String,
        emitter: String,
        files: Vec<FileBinding>,
        build: BuildIdentity,
        fitted: Option<Box<FittedIdentity>>,
    },
    Unsupported {
        reason: String,
        registry_sha256: String,
    },
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TargetRecord {
    pub target: Target,
    pub state: TargetState,
}
impl TargetRecord {
    pub fn unsupported(target: Target, reason: String) -> Self {
        Self {
            target,
            state: TargetState::Unsupported {
                reason,
                registry_sha256: crate::kernel::registry_v2::source_sha256(),
            },
        }
    }
    pub fn supported(
        target: Target,
        dialect: &str,
        emitter: &str,
        sources: &[(&str, &str)],
        fitted: Option<FittedIdentity>,
    ) -> Self {
        Self {
            target,
            state: TargetState::Supported {
                product: target.product(dialect).into(),
                emitter: emitter.into(),
                files: sources
                    .iter()
                    .map(|(p, s)| FileBinding::new(p, s.as_bytes()))
                    .collect(),
                build: BuildIdentity::declared(target),
                fitted: fitted.map(Box::new),
            },
        }
    }
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ArtifactManifestV3 {
    pub schema: String,
    pub dsl: String,
    pub genome_sha256: String,
    pub dialect: String,
    pub registry_sha256: String,
    pub options: OptionsIdentity,
    pub requested: Vec<Target>,
    pub targets: Vec<TargetRecord>,
}
impl ArtifactManifestV3 {
    pub fn new(
        program: &str,
        dialect: &str,
        options: &EmitOptions,
        requested: &[Target],
        targets: Vec<TargetRecord>,
    ) -> Result<Self> {
        let m = Self {
            schema: SCHEMA.into(),
            dsl: program.into(),
            genome_sha256: sha256(program.as_bytes()),
            dialect: dialect.into(),
            registry_sha256: crate::kernel::registry_v2::source_sha256(),
            options: options.into(),
            requested: requested.to_vec(),
            targets,
        };
        m.validate()?;
        Ok(m)
    }
    pub fn validate(&self) -> Result<()> {
        ensure!(self.schema == SCHEMA, "unsupported artifact schema");
        ensure!(
            ["legacy_recipe", "advanced_v3"].contains(&self.dialect.as_str()),
            "unknown program dialect"
        );
        ensure!(
            self.genome_sha256 == sha256(self.dsl.as_bytes()),
            "selected program identity mismatch"
        );
        ensure!(
            crate::GenomeIr::parse(&self.dsl)?.canonical() == self.dsl,
            "noncanonical selected program"
        );
        check_digest(&self.registry_sha256)?;
        ensure!(
            self.options.math_policy == "platform-libm-v1",
            "unsupported math policy"
        );
        validate_requests(&self.requested)?;
        ensure!(
            self.targets.len() == self.requested.len(),
            "requested targets are incomplete"
        );
        let mut seen = BTreeSet::new();
        let mut paths: BTreeMap<&str, (Target, &FileBinding)> = BTreeMap::new();
        for record in &self.targets {
            ensure!(
                self.requested.contains(&record.target) && seen.insert(record.target),
                "unexpected or duplicate target record"
            );
            match &record.state {
                TargetState::Unsupported {
                    reason,
                    registry_sha256,
                } => {
                    ensure!(
                        !reason.trim().is_empty() && registry_sha256 == &self.registry_sha256,
                        "unsupported target lacks registry evidence"
                    );
                }
                TargetState::Supported {
                    product,
                    emitter,
                    files,
                    build,
                    fitted,
                } => {
                    ensure!(
                        product == record.target.product(&self.dialect)
                            && !emitter.trim().is_empty(),
                        "target product/emitter mismatch"
                    );
                    let expected: Vec<_> = if record.target == Target::Cpp17Fitted {
                        vec![
                            record.target.path(),
                            "cpp17-fitted/model.codec.json",
                            "cpp17-fitted/standalone.cpp",
                        ]
                    } else if record.target == Target::TypeScript {
                        vec!["model.ts", "model.kernel.rs"]
                    } else {
                        vec![record.target.path()]
                    };
                    ensure!(
                        files.iter().map(|f| f.path.as_str()).collect::<Vec<_>>() == expected,
                        "incomplete or ambiguous artifact closure"
                    );
                    let declared = BuildIdentity::declared(record.target);
                    // Compiler selection/flags belong to the pinned recipe, not the consumer's environment.
                    ensure!(
                        build.standard == declared.standard
                            && build.backend == declared.backend
                            && build.abi == declared.abi
                            && !build.toolchain.is_empty(),
                        "target build identity mismatch"
                    );
                    if matches!(
                        record.target,
                        Target::Cpp17 | Target::Cpp20 | Target::Cpp17RecipeV2 | Target::Cpp17Fitted
                    ) {
                        ensure!(
                            build.flags.contains(&format!("-std={}", build.standard))
                                && build.flags.contains(&"-ffp-contract=off".into())
                                && build.flags.contains(&"-fno-fast-math".into()),
                            "unsafe or mismatched build flags"
                        );
                    }
                    ensure!(
                        fitted.is_some()
                            == (record.target == Target::Cpp17Fitted
                                || self.dialect == "advanced_v3"),
                        "fitted state identity missing or unexpected"
                    );
                    if let Some(f) = fitted {
                        for d in [
                            &f.state_sha256,
                            &f.input_sha256,
                            &f.task_sha256,
                            &f.schema_sha256,
                            &f.vocabulary_sha256,
                        ] {
                            check_digest(d)?;
                        }
                    }
                    for file in files {
                        safe_relative(&file.path)?;
                        check_digest(&file.sha256)?;
                        ensure!(
                            file.bytes > 0 && file.bytes <= 9_007_199_254_740_991,
                            "invalid artifact length"
                        );
                        if let Some((other, previous)) =
                            paths.insert(&file.path, (record.target, file))
                        {
                            ensure!(
                                matches!(
                                    (other, record.target),
                                    (Target::Cpp17, Target::Cpp20) | (Target::Cpp20, Target::Cpp17)
                                ) && previous == file,
                                "ambiguous shared artifact path"
                            );
                        }
                    }
                }
            }
        }
        Ok(())
    }
    pub fn all_supported(&self) -> bool {
        self.validate().is_ok()
            && self
                .targets
                .iter()
                .all(|t| matches!(t.state, TargetState::Supported { .. }))
    }
    pub fn envelope_bytes(&self, domain: Domain) -> Result<Vec<u8>> {
        self.validate()?;
        let payload = canonical(self)?;
        canonical(&ArtifactEnvelope {
            envelope_schema: ENVELOPE.into(),
            domain,
            schema_id: SCHEMA.into(),
            own_id: OwnId {
                formula: FORMULA.into(),
                schema_id: SCHEMA.into(),
                payload_byte_len: payload.len() as u64,
                sha256: domain_hash(&payload),
            },
            payload: self.clone(),
        })
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Domain {
    Native,
    Inference,
    Storage,
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct OwnId {
    formula: String,
    schema_id: String,
    payload_byte_len: u64,
    sha256: String,
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ArtifactEnvelope {
    envelope_schema: String,
    domain: Domain,
    schema_id: String,
    own_id: OwnId,
    payload: ArtifactManifestV3,
}

// This closed schema has only ASCII field names, safe positive integers, strings,
// booleans and arrays. Sorted serde_json Value keys therefore have JCS UTF-16 order.
fn canonical(value: &impl Serialize) -> Result<Vec<u8>> {
    Ok(serde_json::to_vec(&serde_json::to_value(value)?)?)
}
fn domain_hash(payload: &[u8]) -> String {
    let mut bytes = SCHEMA.as_bytes().to_vec();
    bytes.push(0);
    bytes.extend_from_slice(&(payload.len() as u64).to_be_bytes());
    bytes.extend_from_slice(payload);
    sha256(&bytes)
}
fn check_digest(d: &str) -> Result<()> {
    ensure!(
        d.len() == 64
            && d.bytes()
                .all(|c| c.is_ascii_digit() || (b'a'..=b'f').contains(&c)),
        "invalid SHA-256"
    );
    Ok(())
}
fn safe_relative(path: &str) -> Result<()> {
    ensure!(
        !path.is_empty()
            && !path.contains('\\')
            && path
                .split('/')
                .all(|s| !s.is_empty() && s != "." && s != "..")
            && Path::new(path)
                .components()
                .all(|c| matches!(c, Component::Normal(_))),
        "unsafe artifact path"
    );
    Ok(())
}
/// Read one regular file, refusing symlinks at every relative component.
pub fn read_artifact(root: &Path, relative: &str) -> Result<Vec<u8>> {
    safe_relative(relative)?;
    ensure!(
        !std::fs::symlink_metadata(root)?.file_type().is_symlink(),
        "symlink artifact root"
    );
    let mut path = root.to_path_buf();
    for c in Path::new(relative).components() {
        path.push(c.as_os_str());
        ensure!(
            !std::fs::symlink_metadata(&path)?.file_type().is_symlink(),
            "symlink artifact path"
        );
    }
    ensure!(
        std::fs::metadata(&path)?.is_file(),
        "artifact is not a regular file"
    );
    std::fs::read(&path).with_context(|| format!("read artifact {}", path.display()))
}

/// An external trust anchor. Never derive this from loose files during verification.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExpectedArtifactV3 {
    pub manifest_sha256: String,
    pub program_sha256: String,
    pub requested: Vec<Target>,
}
impl ExpectedArtifactV3 {
    pub fn from_emission(bytes: &[u8], manifest: &ArtifactManifestV3) -> Self {
        Self {
            manifest_sha256: sha256(bytes),
            program_sha256: manifest.genome_sha256.clone(),
            requested: manifest.requested.clone(),
        }
    }
}
pub fn read_manifest(bytes: &[u8], expected: &ExpectedArtifactV3) -> Result<ArtifactManifestV3> {
    check_digest(&expected.manifest_sha256)?;
    check_digest(&expected.program_sha256)?;
    validate_requests(&expected.requested)?;
    ensure!(
        sha256(bytes) == expected.manifest_sha256,
        "pinned manifest digest mismatch"
    );
    let envelope: ArtifactEnvelope = serde_json::from_slice(bytes)?;
    ensure!(
        envelope.envelope_schema == ENVELOPE && envelope.schema_id == SCHEMA,
        "legacy or unknown manifest cannot claim v3 binding"
    );
    ensure!(
        canonical(&envelope)? == bytes,
        "noncanonical artifact envelope"
    );
    let payload = canonical(&envelope.payload)?;
    ensure!(
        envelope.own_id
            == OwnId {
                formula: FORMULA.into(),
                schema_id: SCHEMA.into(),
                payload_byte_len: payload.len() as u64,
                sha256: domain_hash(&payload)
            },
        "artifact payload identity mismatch"
    );
    let manifest = envelope.payload;
    manifest.validate()?;
    ensure!(
        manifest.genome_sha256 == expected.program_sha256,
        "selected program mismatch"
    );
    let mut requested = expected.requested.clone();
    requested.sort();
    let mut actual = manifest.requested.clone();
    actual.sort();
    ensure!(actual == requested, "expected request set mismatch");
    Ok(manifest)
}
pub fn verify_directory(
    root: &Path,
    manifest_path: &str,
    expected: &ExpectedArtifactV3,
) -> Result<ArtifactManifestV3> {
    let manifest = read_manifest(&read_artifact(root, manifest_path)?, expected)?;
    for target in &manifest.targets {
        if let TargetState::Supported { files, .. } = &target.state {
            for file in files {
                let bytes = read_artifact(root, &file.path)?;
                ensure!(
                    bytes.len() as u64 == file.bytes && sha256(&bytes) == file.sha256,
                    "{}: artifact bytes disagree with selected manifest",
                    file.path
                );
            }
        }
    }
    Ok(manifest)
}

/// Sources are assembled before publication; the manifest binds intended bytes,
/// and the writer must verify actual readback against it before returning success.
pub struct TargetEmission {
    pub manifest: ArtifactManifestV3,
    pub sources: BTreeMap<String, String>,
}
