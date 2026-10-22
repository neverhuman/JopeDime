//! `model.jope`: complete fitted state for prediction without refitting (SPECIFICATION.md §13.2).
//!
//! A bounded, versioned data container, not an executable archive:
//!
//! ```text
//! "JOPEFIT1" | u32 format_version | u32 minimum_reader_version | u64 header_len | header JSON
//!            | u64 body_len | body | sha256 of every preceding byte (32 B)
//! ```
//!
//! All integers are little-endian. The header (`jope.appliance-fitted/v1`) records the method,
//! task, class order, input schema, preprocessing policy, recipe and its hash, each fitted
//! section's length and sha256, the training procedure (selection record, budget, fitting-input
//! digest) and the package that fitted it. The body is the native `feat_math` bound model
//! (`battle-artifact-v2`) in its existing serialized form, which already carries every learned
//! parameter, category vocabulary and calibration. Loading checks the trailer, the header, the
//! section hash, the model's own binding and the recipe against this build's registry.
use crate::automl::Selection;
use crate::budget::BudgetRecord;
use crate::methods::native::{predict_bound, Predictions};
use crate::methods::registry::{
    rust_methods, task_spec, validate_classes, RustMethod, INPUT_POLICY,
};
use crate::table::{ColumnRecord, RAW_CELL_ENCODING};
use crate::task::{self, TaskKind};
use crate::{sha256_hex, Failure, RUNTIME_ABI};
use anyhow::{Context, Result};
use feat_math::advanced_v3::{BoundModel, DataSchema, RawTable};
use feat_math::GenomeIr;
use serde::{Deserialize, Serialize};

pub const MAGIC: [u8; 8] = *b"JOPEFIT1";
pub const FORMAT_VERSION: u32 = 1;
pub const HEADER_SCHEMA: &str = "jope.appliance-fitted/v1";
pub const MAX_HEADER_BYTES: u64 = 16 << 20;
/// The one section kind this version writes.
pub const BOUND_MODEL_SECTION: &str = "bound_model";
pub const BOUND_MODEL_FORMAT: &str = "json/battle-artifact-v2";

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Header {
    pub schema: String,
    pub minimum_reader_version: u32,
    pub product: String,
    pub product_version: String,
    pub runtime_abi: String,
    pub method: MethodIdentity,
    pub task: String,
    pub metric: String,
    pub class_order: Vec<String>,
    pub target: String,
    pub input_schema: Vec<ColumnRecord>,
    pub preprocessing: Preprocessing,
    pub recipe_sha256: String,
    /// Pretrained assets a reload needs; empty for native methods.
    pub required_asset_hashes: Vec<String>,
    pub fitted_sections: Vec<Section>,
    pub training_procedure: TrainingProcedure,
    pub retained_training_context: RetainedContext,
    pub created_by: CreatedBy,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MethodIdentity {
    pub id: String,
    pub family: String,
    pub configuration: String,
    pub recipe: String,
    pub implementation: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Preprocessing {
    pub raw_cell_encoding: String,
    pub input_policy: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Section {
    pub name: String,
    pub format: String,
    pub bytes: u64,
    pub sha256: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TrainingProcedure {
    pub rows: usize,
    pub fitting_input_sha256: String,
    pub training_file_sha256: String,
    /// `requested` or the selection policy id.
    pub method_choice: String,
    pub selection: Option<serde_json::Value>,
    pub budget: serde_json::Value,
    pub final_refit: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RetainedContext {
    pub neural_support_rows: usize,
    pub note: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, Default)]
#[serde(deny_unknown_fields)]
pub struct CreatedBy {
    pub source_sha40: String,
    /// sha256 of the signed package manifest of the executable that fitted, when packaged. The
    /// manifest binds that executable's code and asset hashes.
    pub package_manifest_sha256: Option<String>,
    pub package_profile: Option<String>,
}

/// A fitted model: header plus the bound native state.
#[derive(Clone, Debug)]
pub struct FittedModel {
    pub header: Header,
    pub model: BoundModel,
}

/// What a fit produced, before it is written.
pub struct FitRecord<'a> {
    pub method: &'a RustMethod,
    pub kind: TaskKind,
    pub classes: &'a [String],
    pub target: &'a str,
    pub schema: &'a DataSchema,
    pub model: BoundModel,
    pub rows: usize,
    pub fitting_input_sha256: String,
    pub training_file_sha256: String,
    pub selection: Option<&'a Selection>,
    pub budget: &'a BudgetRecord,
    pub created_by: CreatedBy,
}

const RETAINED_NOTE: &str = "native fitted parameters can embed training rows (kernel, \
     smoother and Gaussian-process heads keep support rows); treat this file as containing \
     training data";

impl FittedModel {
    pub fn new(record: FitRecord<'_>) -> Result<Self> {
        let body = serde_json::to_vec(&record.model)?;
        let header = Header {
            schema: HEADER_SCHEMA.to_owned(),
            minimum_reader_version: FORMAT_VERSION,
            product: "jopedime".to_owned(),
            product_version: env!("CARGO_PKG_VERSION").to_owned(),
            runtime_abi: RUNTIME_ABI.to_owned(),
            method: MethodIdentity {
                id: record.method.id.clone(),
                family: record.method.family.clone(),
                configuration: record.method.configuration.clone(),
                recipe: record.method.dsl.clone(),
                implementation: format!("feat_math::advanced_v3 {}", record.model.artifact_schema),
            },
            task: task::name(record.kind).to_owned(),
            metric: task::metric(record.kind).to_owned(),
            class_order: record.classes.to_vec(),
            target: record.target.to_owned(),
            input_schema: record
                .schema
                .columns
                .iter()
                .map(|c| ColumnRecord {
                    name: c.name.clone(),
                    kind: c.kind,
                })
                .collect(),
            preprocessing: Preprocessing {
                raw_cell_encoding: RAW_CELL_ENCODING.to_owned(),
                input_policy: INPUT_POLICY.to_owned(),
            },
            recipe_sha256: sha256_hex(record.method.dsl.as_bytes()),
            required_asset_hashes: Vec::new(),
            fitted_sections: vec![Section {
                name: BOUND_MODEL_SECTION.to_owned(),
                format: BOUND_MODEL_FORMAT.to_owned(),
                bytes: body.len() as u64,
                sha256: sha256_hex(&body),
            }],
            training_procedure: TrainingProcedure {
                rows: record.rows,
                fitting_input_sha256: record.fitting_input_sha256,
                training_file_sha256: record.training_file_sha256,
                method_choice: match record.selection {
                    Some(s) => s.policy.to_owned(),
                    None => "requested".to_owned(),
                },
                selection: record.selection.map(serde_json::to_value).transpose()?,
                budget: serde_json::to_value(record.budget)?,
                final_refit: true,
            },
            retained_training_context: RetainedContext {
                neural_support_rows: 0,
                note: RETAINED_NOTE.to_owned(),
            },
            created_by: record.created_by,
        };
        let fitted = Self {
            header,
            model: record.model,
        };
        fitted.validate()?;
        Ok(fitted)
    }

    pub fn kind(&self) -> Result<TaskKind> {
        task::SUPPORTED
            .into_iter()
            .find(|k| task::name(*k) == self.header.task)
            .ok_or_else(|| Failure::integrity(format!("unsupported task {:?}", self.header.task)))
    }

    /// The registered recipe this file names.
    pub fn method(&self) -> Result<RustMethod> {
        let kind = self.kind()?;
        rust_methods(kind)?
            .into_iter()
            .find(|m| m.id == self.header.method.id)
            .ok_or_else(|| {
                Failure::integrity(format!(
                    "fitted method {} is not registered in this build",
                    self.header.method.id
                ))
            })
    }

    /// Every cross-check between header, registry and bound model.
    pub fn validate(&self) -> Result<()> {
        let h = &self.header;
        let fail = |what: &str| Failure::integrity(format!("fitted model: {what}"));
        if h.schema != HEADER_SCHEMA || h.runtime_abi != RUNTIME_ABI {
            return Err(fail("unsupported header schema or runtime ABI"));
        }
        if h.preprocessing.raw_cell_encoding != RAW_CELL_ENCODING
            || h.preprocessing.input_policy != INPUT_POLICY
        {
            return Err(fail("preprocessing policy differs from this build"));
        }
        let kind = self.kind()?;
        validate_classes(kind, &h.class_order).map_err(|e| fail(&format!("{e:#}")))?;
        let method = self.method()?;
        if method.dsl != h.method.recipe
            || method.family != h.method.family
            || method.configuration != h.method.configuration
            || sha256_hex(method.dsl.as_bytes()) != h.recipe_sha256
        {
            return Err(fail("recipe differs from the registered method"));
        }
        let canonical = GenomeIr::parse(&method.dsl)?.canonical();
        if self.model.genome_sha256 != sha256_hex(canonical.as_bytes()) {
            return Err(fail("bound model was fitted from another recipe"));
        }
        if self.model.task != task_spec(kind)? {
            return Err(fail("bound model task differs from the header"));
        }
        let columns: Vec<ColumnRecord> = self
            .model
            .schema
            .columns
            .iter()
            .map(|c| ColumnRecord {
                name: c.name.clone(),
                kind: c.kind,
            })
            .collect();
        if columns != h.input_schema {
            return Err(fail("bound model schema differs from the header"));
        }
        if !h.required_asset_hashes.is_empty() {
            return Err(fail("native models require no pretrained assets"));
        }
        self.model
            .validate_binding()
            .map_err(|e| fail(&format!("{e:#}")))
    }

    pub fn schema(&self) -> &DataSchema {
        &self.model.schema
    }

    pub fn predict(&self, table: &RawTable) -> Result<Predictions> {
        predict_bound(&self.model, table, &self.header.class_order)
    }

    pub fn to_bytes(&self) -> Result<Vec<u8>> {
        let header = serde_json::to_vec(&self.header)?;
        let body = serde_json::to_vec(&self.model)?;
        let mut out = Vec::with_capacity(header.len() + body.len() + 64);
        out.extend_from_slice(&MAGIC);
        out.extend_from_slice(&FORMAT_VERSION.to_le_bytes());
        out.extend_from_slice(&self.header.minimum_reader_version.to_le_bytes());
        out.extend_from_slice(&(header.len() as u64).to_le_bytes());
        out.extend_from_slice(&header);
        out.extend_from_slice(&(body.len() as u64).to_le_bytes());
        out.extend_from_slice(&body);
        let digest = {
            use sha2::{Digest, Sha256};
            Sha256::digest(&out)
        };
        out.extend_from_slice(&digest);
        Ok(out)
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        let fail = |what: &str| Failure::integrity(format!("model file: {what}"));
        if bytes.len() < 8 + 4 + 4 + 8 + 8 + 32 || bytes[..8] != MAGIC {
            return Err(fail("not a jopedime fitted model"));
        }
        let (content, trailer) = bytes.split_at(bytes.len() - 32);
        let digest: [u8; 32] = {
            use sha2::{Digest, Sha256};
            Sha256::digest(content).into()
        };
        if digest != trailer {
            return Err(fail("checksum mismatch (truncated or modified)"));
        }
        let u32_at = |at: usize| u32::from_le_bytes(content[at..at + 4].try_into().expect("4"));
        let u64_at = |at: usize| -> Result<u64> {
            Ok(u64::from_le_bytes(
                content
                    .get(at..at + 8)
                    .ok_or_else(|| fail("truncated length"))?
                    .try_into()
                    .expect("8"),
            ))
        };
        let version = u32_at(8);
        let minimum = u32_at(12);
        if minimum > FORMAT_VERSION || version < minimum {
            return Err(fail(&format!(
                "format {version} needs reader version {minimum}; this build reads {FORMAT_VERSION}"
            )));
        }
        let header_len = u64_at(16)?;
        if header_len > MAX_HEADER_BYTES {
            return Err(fail("header exceeds its bound"));
        }
        let header_end = 24usize
            .checked_add(header_len as usize)
            .filter(|end| *end + 8 <= content.len())
            .ok_or_else(|| fail("header length out of range"))?;
        let body_len = u64_at(header_end)?;
        let body_start = header_end + 8;
        if (body_start as u64).checked_add(body_len) != Some(content.len() as u64) {
            return Err(fail("body length does not match the file"));
        }
        let header_bytes = &content[24..header_end];
        let body = &content[body_start..];
        let header: Header =
            serde_json::from_slice(header_bytes).map_err(|e| fail(&format!("header: {e}")))?;
        if serde_json::to_vec(&header)? != header_bytes {
            return Err(fail("header is not in canonical form"));
        }
        let section = match header.fitted_sections.as_slice() {
            [section]
                if section.name == BOUND_MODEL_SECTION && section.format == BOUND_MODEL_FORMAT =>
            {
                section
            }
            _ => return Err(fail("unsupported fitted sections")),
        };
        if section.bytes != body.len() as u64 || section.sha256 != sha256_hex(body) {
            return Err(fail("fitted section does not match its record"));
        }
        let model: BoundModel = crate::strict_json::decode(body)
            .context("decode bound model")
            .map_err(|e| fail(&format!("{e:#}")))?;
        let fitted = Self { header, model };
        fitted.validate()?;
        Ok(fitted)
    }

    pub fn read(path: &std::path::Path, max_bytes: u64) -> Result<Self> {
        let len = std::fs::metadata(path)
            .map_err(|e| Failure::input(format!("{}: {e}", path.display())))?
            .len();
        if len > max_bytes {
            return Err(Failure::input(format!(
                "{} is {len} bytes; the model limit is {max_bytes}",
                path.display()
            )));
        }
        let bytes =
            std::fs::read(path).map_err(|e| Failure::input(format!("{}: {e}", path.display())))?;
        Self::from_bytes(&bytes).with_context(|| path.display().to_string())
    }
}
