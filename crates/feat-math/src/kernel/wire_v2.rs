// jankurai:allow repo-rot.path.fake-versioned-source reason=active-B04-wire-v2-reservations owner=kernel2 expires=2027-04-01
//! Version-2 identity reservations and reader-first envelopes.
//!
//! These records name future operators. A decoded v2 record cannot be turned
//! into an executable legacy program or model, and no generator uses this table.
//! The historical v1 readers and their bytes remain in `program` and `exact`.

use super::codec::{Reader, Writer, MAX_BYTES};
use super::export::{self, ExportReport, FittedBasis, Route};
use super::linear_scaling::LinearScaledWinner;
use super::scalar_program::{ScalarCarrier, ScalarExpressionProgram, ScalarModel};
// B08 ordered-state routing.
use super::finite_difference::{
    DifferenceFittedCpp, DifferenceLanguage, DifferenceModel, DifferenceProgram, DifferenceRecipe,
    OrderedRow,
};
use super::ordered_axis::{FoldBoundary, TaskMode};
use super::Program;
use crate::{GenomeIr, Model};
use serde::{Deserialize, Serialize};
use serde_json::Value;

// B05–B07: explicit scalar recipe/fitted routes.
mod scalar_export;

const MAGIC_PREFIX: &[u8] = b"FMKPRG\0";
pub(super) const MAGIC_V2: &[u8] = b"FMKPRG\0\x02";
pub(super) const READER_VERSION: u8 = 2;
const RESERVATION_FEATURE: u64 = 1;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct Reservation {
    pub id: u16,
    pub name: Option<&'static str>,
    pub generatable: bool,
}

const fn reserved(id: u16, name: Option<&'static str>) -> Reservation {
    Reservation {
        id,
        name,
        generatable: false,
    }
}

/// The first 53 wire IDs remain owned by the frozen v1 registry.
pub const WIRE: &[Reservation] = &[
    reserved(53, Some("symlog")),
    reserved(54, Some("select_corr")),
    reserved(55, Some("impute")),
    reserved(56, Some("pair_ratio")),
    reserved(57, Some("spline")),
    reserved(58, Some("savgol")),
    reserved(59, Some("finite_diff")),
    reserved(60, Some("cumint")),
    reserved(61, Some("goertzel_bank")),
    reserved(62, Some("acf_bank")),
    reserved(63, Some("hampel")),
    reserved(64, Some("realized_path_stats")),
    reserved(65, None),
    reserved(66, None),
    reserved(67, None),
    reserved(68, None),
    reserved(69, None),
    reserved(70, None),
    reserved(71, None),
    reserved(72, None),
    reserved(73, None),
    reserved(74, None),
    reserved(75, Some("timescale_nabla")),
];

/// Func 0–18 remain in the v1 exact table. Only 19–22 have assigned names.
pub const FUNCTIONS: &[Reservation] = &[
    reserved(19, Some("probit")),
    reserved(20, Some("logit")),
    reserved(21, Some("expm1")),
    reserved(22, Some("log1mexp")),
    reserved(23, None),
    reserved(24, None),
    reserved(25, None),
    reserved(26, None),
    reserved(27, None),
    reserved(28, None),
    reserved(29, None),
    reserved(30, None),
    reserved(31, None),
    reserved(32, None),
    reserved(33, None),
    reserved(34, None),
];

/// Var 0–7 remain in the v1 exact table.
pub const VARIABLES: &[Reservation] = &[
    reserved(8, Some("row_median")),
    reserved(9, Some("row_iqr")),
];

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Domain {
    Wire,
    Function,
    Variable,
}

impl Domain {
    fn tag(self) -> u8 {
        match self {
            Self::Wire => 0,
            Self::Function => 1,
            Self::Variable => 2,
        }
    }

    fn from_tag(tag: u8) -> VersionResult<Self> {
        match tag {
            0 => Ok(Self::Wire),
            1 => Ok(Self::Function),
            2 => Ok(Self::Variable),
            _ => Err(VersionError::UnsupportedVersion(format!(
                "identity domain {tag}"
            ))),
        }
    }

    fn table(self) -> &'static [Reservation] {
        match self {
            Self::Wire => WIRE,
            Self::Function => FUNCTIONS,
            Self::Variable => VARIABLES,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct ReservationRef {
    pub domain: Domain,
    pub id: u16,
}

impl ReservationRef {
    pub fn new(domain: Domain, id: u16) -> VersionResult<Self> {
        let row = domain.table().iter().find(|row| row.id == id);
        if !row.is_some_and(|row| row.name.is_some()) {
            return Err(VersionError::UnsupportedVersion(format!(
                "unassigned {domain:?} id {id}"
            )));
        }
        Ok(Self { domain, id })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct ReservationRow {
    pub domain: Domain,
    pub id: u16,
    pub name: Option<&'static str>,
    pub lifecycle: &'static str,
    pub generatable: bool,
}

pub fn manifest_rows() -> Vec<ReservationRow> {
    [Domain::Wire, Domain::Function, Domain::Variable]
        .into_iter()
        .flat_map(|domain| {
            domain.table().iter().map(move |row| ReservationRow {
                domain,
                id: row.id,
                name: row.name,
                lifecycle: "reserved",
                generatable: false,
            })
        })
        .collect()
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum VersionError {
    UnsupportedVersion(String),
    Malformed(String),
    StageRejected { kind: super::Error, faults: u32 },
}

impl std::fmt::Display for VersionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}

impl std::error::Error for VersionError {}

impl From<super::Error> for VersionError {
    fn from(error: super::Error) -> Self {
        match error {
            super::Error::Unsupported(message) => Self::UnsupportedVersion(message),
            other => Self::Malformed(other.to_string()),
        }
    }
}

pub type VersionResult<T> = std::result::Result<T, VersionError>;

fn check_reservations(rows: &[ReservationRef]) -> VersionResult<()> {
    if rows.len() > WIRE.len() + FUNCTIONS.len() + VARIABLES.len() {
        return Err(VersionError::Malformed("reservation count".into()));
    }
    let mut previous = None;
    for &row in rows {
        ReservationRef::new(row.domain, row.id)?;
        if previous.is_some_and(|prev| prev >= row) {
            return Err(VersionError::Malformed(
                "duplicate or unordered reservation".into(),
            ));
        }
        previous = Some(row);
    }
    Ok(())
}

/// Reservation-only v2 program. It carries no executable AST or fit state.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ReservedProgram {
    reservations: Vec<ReservationRef>,
}

impl ReservedProgram {
    pub fn new(reservations: Vec<ReservationRef>) -> VersionResult<Self> {
        check_reservations(&reservations)?;
        Ok(Self { reservations })
    }

    pub fn reservations(&self) -> &[ReservationRef] {
        &self.reservations
    }

    pub fn to_bytes(&self) -> VersionResult<Vec<u8>> {
        check_reservations(&self.reservations)?;
        let mut writer = Writer::default();
        writer.raw(MAGIC_V2)?;
        writer.byte(0)?; // Reservation-only variant; later executable variants get new tags.
        writer.byte(READER_VERSION)?;
        writer.word(RESERVATION_FEATURE)?;
        writer.word(self.reservations.len() as u64)?;
        for row in &self.reservations {
            writer.byte(row.domain.tag())?;
            writer.word(u64::from(row.id))?;
        }
        Ok(writer.finish())
    }

    fn from_bytes(bytes: &[u8]) -> VersionResult<Self> {
        let mut reader = Reader::new(bytes)?;
        if reader.take(MAGIC_V2.len())? != MAGIC_V2 {
            return Err(VersionError::UnsupportedVersion(
                "program envelope version".into(),
            ));
        }
        if reader.byte()? != 0 {
            return Err(VersionError::UnsupportedVersion(
                "program v2 variant".into(),
            ));
        }
        if reader.byte()? != READER_VERSION {
            return Err(VersionError::UnsupportedVersion(
                "minimum reader version".into(),
            ));
        }
        if reader.word()? != RESERVATION_FEATURE {
            return Err(VersionError::UnsupportedVersion(
                "required feature set".into(),
            ));
        }
        let count = reader.length()?;
        if count > WIRE.len() + FUNCTIONS.len() + VARIABLES.len() {
            return Err(VersionError::Malformed("reservation count".into()));
        }
        let mut rows = Vec::with_capacity(count);
        for _ in 0..count {
            let domain = Domain::from_tag(reader.byte()?)?;
            let id = reader.word()?;
            let id = u16::try_from(id)
                .map_err(|_| VersionError::UnsupportedVersion(format!("{domain:?} id {id}")))?;
            rows.push(ReservationRef::new(domain, id)?);
        }
        reader.finish()?;
        Self::new(rows)
    }
}

#[derive(Clone, Debug)]
// B14: explicit wire-60 ordered program routes.
pub enum VersionedProgram {
    LegacyV1(Program),
    ReservedV2(ReservedProgram),
    ScalarV2(ScalarExpressionProgram),
    LinearScaledV2(LinearScaledWinner),
    // B12: FeatureWithinRow execution is a separate v2 subtype.
    RowVectorV2(super::row_vector::Program),
    DifferenceV2(DifferenceProgram),
    IntegrationV2(super::cumulative_integration::IntegrationProgram),
    TimeGradientV2(super::timescale_nabla::TimeGradientProgram),
    PathStatsV2(super::realized_path_stats::PathStatsProgram),
}

impl VersionedProgram {
    /// Explicit numeric-input scalar fitting, with no v1 coercion.
    pub fn fit_scalar(
        &self,
        training: &crate::Rows,
        labels: &[f64],
        carrier: ScalarCarrier,
        lambda: f64,
    ) -> VersionResult<VersionedModel> {
        match self {
            Self::ScalarV2(program) => Ok(VersionedModel::ScalarV2(Box::new(
                program.fit_training(training, labels, carrier, lambda)?,
            ))),
            _ => Err(VersionError::UnsupportedVersion(
                "program has no scalar fit route".into(),
            )),
        }
    }

    /// Ordered recipes carry their explicit input identity and continuation API.
    pub fn emit_ordered_recipe(
        &self,
        language: DifferenceLanguage,
    ) -> VersionResult<DifferenceRecipe> {
        match self {
            Self::DifferenceV2(program) => program.emit_recipe(language),
            _ => Err(VersionError::UnsupportedVersion(
                "program has no ordered recipe route".into(),
            )),
        }
    }

    /// B14 integration artifacts carry their own typed codec and continuation API.
    pub fn emit_integral_recipe(
        &self,
        language: DifferenceLanguage,
    ) -> VersionResult<super::cumulative_integration::IntegrationRecipe> {
        match self {
            Self::IntegrationV2(program) => program.emit_recipe(language),
            _ => Err(VersionError::UnsupportedVersion(
                "program has no integral recipe route".into(),
            )),
        }
    }

    pub fn emit_time_gradient_recipe(
        &self,
        language: DifferenceLanguage,
    ) -> VersionResult<super::timescale_nabla::TimeGradientRecipe> {
        match self {
            Self::TimeGradientV2(program) => program.emit_recipe(language),
            _ => Err(VersionError::UnsupportedVersion(
                "program has no time_gradient recipe route".into(),
            )),
        }
    }

    pub fn emit_path_stats_recipe(
        &self,
        language: DifferenceLanguage,
    ) -> VersionResult<super::realized_path_stats::PathStatsRecipe> {
        match self {
            Self::PathStatsV2(program) => program.emit_recipe(language),
            _ => Err(VersionError::UnsupportedVersion(
                "program has no path_stats recipe route".into(),
            )),
        }
    }

    pub fn from_bytes(bytes: &[u8]) -> VersionResult<Self> {
        if bytes.len() < MAGIC_V2.len() || &bytes[..MAGIC_PREFIX.len()] != MAGIC_PREFIX {
            return Err(VersionError::Malformed("program magic".into()));
        }
        match bytes[MAGIC_PREFIX.len()] {
            1 => Ok(Self::LegacyV1(Program::from_bytes(bytes)?)),
            2 => match bytes.get(MAGIC_V2.len()) {
                Some(0) => Ok(Self::ReservedV2(ReservedProgram::from_bytes(bytes)?)),
                Some(1) => Ok(Self::ScalarV2(ScalarExpressionProgram::from_bytes(bytes)?)),
                Some(2) => Ok(Self::LinearScaledV2(LinearScaledWinner::from_bytes(bytes)?)),
                Some(3) => Ok(Self::RowVectorV2(super::row_vector::Program::from_bytes(
                    bytes,
                )?)),
                Some(4) => Ok(Self::DifferenceV2(DifferenceProgram::from_bytes(bytes)?)),
                Some(5) => Ok(Self::IntegrationV2(
                    super::cumulative_integration::IntegrationProgram::from_bytes(bytes)?,
                )),
                Some(6) => Ok(Self::TimeGradientV2(
                    super::timescale_nabla::TimeGradientProgram::from_bytes(bytes)?,
                )),
                Some(7) => Ok(Self::PathStatsV2(
                    super::realized_path_stats::PathStatsProgram::from_bytes(bytes)?,
                )),
                Some(_) => Err(VersionError::UnsupportedVersion(
                    "program v2 variant".into(),
                )),
                None => Err(VersionError::Malformed("truncated program variant".into())),
            },
            version => Err(VersionError::UnsupportedVersion(format!(
                "program envelope {version}"
            ))),
        }
    }

    pub fn to_bytes(&self) -> VersionResult<Vec<u8>> {
        match self {
            Self::LegacyV1(program) => Ok(program.to_bytes()?),
            Self::ReservedV2(program) => program.to_bytes(),
            Self::ScalarV2(program) => program.to_bytes(),
            Self::LinearScaledV2(program) => program.to_bytes(),
            Self::RowVectorV2(program) => program.to_bytes(),
            Self::DifferenceV2(program) => program.to_bytes(),
            Self::IntegrationV2(program) => program.to_bytes(),
            Self::TimeGradientV2(program) => program.to_bytes(),
            Self::PathStatsV2(program) => program.to_bytes(),
        }
    }

    /// Explicit ordered-input route; legacy and reservation bodies cannot fall through.
    pub fn fit_ordered(
        &self,
        rows: &[OrderedRow],
        labels: &[f64],
        task: TaskMode,
        fold: FoldBoundary,
    ) -> VersionResult<VersionedModel> {
        match self {
            Self::DifferenceV2(program) => Ok(VersionedModel::DifferenceV2(Box::new(
                program.fit_training(rows, labels, task, fold)?,
            ))),
            Self::IntegrationV2(program) => Ok(VersionedModel::IntegrationV2(Box::new(
                program.fit_training(rows, labels, task, fold)?,
            ))),
            Self::TimeGradientV2(program) => Ok(VersionedModel::TimeGradientV2(Box::new(
                program.fit_training(rows, labels, task, fold)?,
            ))),
            Self::PathStatsV2(program) => Ok(VersionedModel::PathStatsV2(Box::new(
                program.fit_training(rows, labels, task, fold)?,
            ))),
            _ => Err(VersionError::UnsupportedVersion(
                "program has no ordered fit route".into(),
            )),
        }
    }

    pub fn legacy_genome(&self) -> VersionResult<&GenomeIr> {
        match self {
            Self::LegacyV1(Program::LegacyV1(genome)) => Ok(genome),
            Self::ReservedV2(_) => Err(VersionError::UnsupportedVersion(
                "version 2 reservations are not executable".into(),
            )),
            Self::RowVectorV2(_) => Err(VersionError::UnsupportedVersion(
                "v2 row-vector program cannot decode as a v1 genome".into(),
            )),
            Self::DifferenceV2(_) => Err(VersionError::UnsupportedVersion(
                "ordered input required for difference execution".into(),
            )),
            Self::IntegrationV2(_) => Err(VersionError::UnsupportedVersion(
                "ordered input required for integral execution".into(),
            )),
            Self::TimeGradientV2(_) => Err(VersionError::UnsupportedVersion(
                "ordered input required for time_gradient execution".into(),
            )),
            Self::PathStatsV2(_) => Err(VersionError::UnsupportedVersion(
                "ordered input required for path_stats execution".into(),
            )),
            Self::ScalarV2(_) => Err(VersionError::UnsupportedVersion(
                "version 2 scalar execution is not admitted".into(),
            )),
            Self::LinearScaledV2(_) => Err(VersionError::UnsupportedVersion(
                "linear-scaled winner requires its explicit execution profile".into(),
            )),
        }
    }

    pub fn to_lossless_text(&self) -> VersionResult<String> {
        match self {
            Self::LegacyV1(program) => Ok(program.to_lossless_text()?),
            Self::ReservedV2(_)
            | Self::ScalarV2(_)
            | Self::RowVectorV2(_)
            | Self::DifferenceV2(_)
            | Self::IntegrationV2(_)
            | Self::LinearScaledV2(_)
            | Self::TimeGradientV2(_)
            | Self::PathStatsV2(_) => Ok(format!(
                "(program codec=2 hex=\"{}\")",
                self.to_bytes()?
                    .iter()
                    .map(|byte| format!("{byte:02x}"))
                    .collect::<String>()
            )),
        }
    }

    pub fn parse_lossless_text(text: &str) -> VersionResult<Self> {
        if text.starts_with("(program codec=1 ") {
            return Ok(Self::LegacyV1(Program::parse_lossless_text(text)?));
        }
        let Some(hex) = text
            .strip_prefix("(program codec=2 hex=\"")
            .and_then(|value| value.strip_suffix("\")"))
        else {
            return if text.starts_with("(program codec=2 ") || text == "(program codec=2)" {
                Err(VersionError::Malformed("program text envelope".into()))
            } else if text.starts_with("(program codec=") {
                Err(VersionError::UnsupportedVersion(
                    "program text codec".into(),
                ))
            } else {
                Err(VersionError::Malformed("program text envelope".into()))
            };
        };
        if hex.len() > 2 * MAX_BYTES || hex.len() % 2 != 0 {
            return Err(VersionError::Malformed("program text length".into()));
        }
        let bytes = hex
            .as_bytes()
            .chunks_exact(2)
            .map(|pair| {
                (pair[0] as char)
                    .to_digit(16)
                    .zip((pair[1] as char).to_digit(16))
                    .map(|(high, low)| ((high << 4) | low) as u8)
                    .ok_or_else(|| VersionError::Malformed("hex digit".into()))
            })
            .collect::<VersionResult<Vec<_>>>()?;
        match Self::from_bytes(&bytes)? {
            value @ (Self::ReservedV2(_)
            | Self::ScalarV2(_)
            | Self::RowVectorV2(_)
            | Self::DifferenceV2(_)
            | Self::IntegrationV2(_)
            | Self::LinearScaledV2(_)
            | Self::TimeGradientV2(_)
            | Self::PathStatsV2(_)) => Ok(value),
            Self::LegacyV1(_) => Err(VersionError::UnsupportedVersion(
                "program text codec/body mismatch".into(),
            )),
        }
    }
}

/// Existing export routes remain available only for an executable v1 genome.
pub fn export_report(program: &VersionedProgram) -> ExportReport {
    match program {
        VersionedProgram::LegacyV1(Program::LegacyV1(genome)) => export::export_report(genome),
        VersionedProgram::ReservedV2(_) => {
            let refusal = Route::Unsupported(vec!["version 2 reservation-only".into()]);
            ExportReport {
                rust_recipe: refusal.clone(),
                python_recipe: refusal.clone(),
                cpp17_recipe: refusal.clone(),
                cpp17_fitted: refusal.clone(),
                cpp17_fitted_basis: FittedBasis::Declared,
                javascript: refusal,
            }
        }
        VersionedProgram::RowVectorV2(program) => {
            let support = match program.plan() {
                Ok(_) => Route::Ok,
                Err(e) => Route::Unsupported(vec![e.to_string()]),
            };
            ExportReport {
                rust_recipe: support.clone(),
                python_recipe: support.clone(),
                cpp17_recipe: support.clone(),
                cpp17_fitted: support,
                cpp17_fitted_basis: FittedBasis::Declared,
                javascript: Route::Unsupported(vec![
                    "v2 row-vector JavaScript route unavailable".into()
                ]),
            }
        }
        VersionedProgram::DifferenceV2(program) => {
            let support = match program.validate_recipe_closure() {
                Ok(()) => Route::Ok,
                Err(error) => Route::Unsupported(vec![error.to_string()]),
            };
            ExportReport {
                rust_recipe: support.clone(),
                python_recipe: support.clone(),
                cpp17_recipe: support.clone(),
                cpp17_fitted: support,
                cpp17_fitted_basis: FittedBasis::Declared,
                javascript: Route::Unsupported(vec![
                    "difference ordered JavaScript route unavailable".into(),
                ]),
            }
        }
        VersionedProgram::IntegrationV2(program) => {
            let support = match program.validate_recipe_closure() {
                Ok(()) => Route::Ok,
                Err(error) => Route::Unsupported(vec![error.to_string()]),
            };
            ExportReport {
                rust_recipe: support.clone(),
                python_recipe: support.clone(),
                cpp17_recipe: support.clone(),
                cpp17_fitted: support,
                cpp17_fitted_basis: FittedBasis::Declared,
                javascript: Route::Unsupported(vec![
                    "integral ordered JavaScript route unavailable".into(),
                ]),
            }
        }
        VersionedProgram::TimeGradientV2(program) => {
            let support = match program.validate_recipe_closure() {
                Ok(()) => Route::Ok,
                Err(error) => Route::Unsupported(vec![error.to_string()]),
            };
            ExportReport {
                rust_recipe: support.clone(),
                python_recipe: support.clone(),
                cpp17_recipe: support.clone(),
                cpp17_fitted: support,
                cpp17_fitted_basis: FittedBasis::Declared,
                javascript: Route::Unsupported(vec![
                    "time_gradient ordered JavaScript route unavailable".into(),
                ]),
            }
        }
        VersionedProgram::PathStatsV2(program) => {
            let support = match program.validate_recipe_closure() {
                Ok(()) => Route::Ok,
                Err(error) => Route::Unsupported(vec![error.to_string()]),
            };
            ExportReport {
                rust_recipe: support.clone(),
                python_recipe: support.clone(),
                cpp17_recipe: support.clone(),
                cpp17_fitted: support,
                cpp17_fitted_basis: FittedBasis::Declared,
                javascript: Route::Unsupported(vec![
                    "path_stats ordered JavaScript route unavailable".into(),
                ]),
            }
        }
        VersionedProgram::ScalarV2(_) => {
            let refusal = Route::Unsupported(vec![
                "scalar export requires carrier, width and lambda; use scalar_export_report".into(),
            ]);
            ExportReport {
                rust_recipe: refusal.clone(),
                python_recipe: refusal.clone(),
                cpp17_recipe: refusal.clone(),
                cpp17_fitted: refusal.clone(),
                cpp17_fitted_basis: FittedBasis::Declared,
                javascript: refusal,
            }
        }
        VersionedProgram::LinearScaledV2(_) => {
            let refusal = Route::Unsupported(vec![
                "explicit linear-scaling library export required".into(),
            ]);
            ExportReport {
                rust_recipe: refusal.clone(),
                python_recipe: refusal.clone(),
                cpp17_recipe: refusal.clone(),
                cpp17_fitted: refusal.clone(),
                cpp17_fitted_basis: FittedBasis::Declared,
                javascript: refusal,
            }
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ModelWireV2 {
    schema: String,
    format_version: u8,
    min_reader_version: u8,
    required_features: u64,
    numerical_policy: String,
    grammar_sha256: String,
    emitter_feature_identity: String,
    output_schema: Vec<String>,
    program: Vec<u8>,
    model: Value,
}

#[derive(Clone, Debug)]
pub struct ReservedModel {
    wire: ModelWireV2,
}

impl ReservedModel {
    pub fn new(
        model: &Model,
        program: &ReservedProgram,
        grammar_sha256: String,
        emitter_feature_identity: String,
        output_schema: Vec<String>,
    ) -> VersionResult<Self> {
        let wire = ModelWireV2 {
            schema: "feat-math.model-envelope".into(),
            format_version: 2,
            min_reader_version: READER_VERSION,
            required_features: RESERVATION_FEATURE,
            numerical_policy: "v2".into(),
            grammar_sha256,
            emitter_feature_identity,
            output_schema,
            program: program.to_bytes()?,
            model: serde_json::to_value(model)
                .map_err(|error| VersionError::Malformed(error.to_string()))?,
        };
        validate_model_wire(&wire)?;
        Ok(Self { wire })
    }

    pub fn to_json(&self) -> VersionResult<String> {
        validate_model_wire(&self.wire)?;
        let json = serde_json::to_string(&self.wire)
            .map_err(|error| VersionError::Malformed(error.to_string()))?;
        if json.len() > MAX_BYTES {
            return Err(VersionError::Malformed("model envelope bytes".into()));
        }
        Ok(json)
    }
}

fn validate_model_wire(wire: &ModelWireV2) -> VersionResult<()> {
    if wire.schema != "feat-math.model-envelope" || wire.format_version != 2 {
        return Err(VersionError::UnsupportedVersion("model envelope".into()));
    }
    if wire.min_reader_version != READER_VERSION {
        return Err(VersionError::UnsupportedVersion(
            "model minimum reader version".into(),
        ));
    }
    if wire.required_features != RESERVATION_FEATURE || wire.numerical_policy != "v2" {
        return Err(VersionError::UnsupportedVersion(
            "model required feature set or numerical policy".into(),
        ));
    }
    if wire.grammar_sha256.len() != 64
        || !wire
            .grammar_sha256
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit())
        || wire.emitter_feature_identity.is_empty()
        || wire.output_schema.is_empty()
    {
        return Err(VersionError::Malformed("model identity metadata".into()));
    }
    if !matches!(
        VersionedProgram::from_bytes(&wire.program)?,
        VersionedProgram::ReservedV2(_)
    ) {
        return Err(VersionError::UnsupportedVersion(
            "model requires a v2 program".into(),
        ));
    }
    serde_json::from_value::<Model>(wire.model.clone())
        .map_err(|error| VersionError::Malformed(format!("model payload: {error}")))?;
    Ok(())
}

#[derive(Clone, Debug)]
pub enum VersionedModel {
    LegacyV1(Model),
    ReservedV2(ReservedModel),
    RowVectorV2(super::row_vector::Model),
    DifferenceV2(Box<DifferenceModel>),
    IntegrationV2(Box<super::cumulative_integration::IntegrationModel>),
    ScalarV2(Box<ScalarModel>),
    TimeGradientV2(Box<super::timescale_nabla::TimeGradientModel>),
    PathStatsV2(Box<super::realized_path_stats::PathStatsModel>),
}

impl VersionedModel {
    /// Scalar predictions retain per-row faults and do not update fit state.
    pub fn predict_scalar(&self, rows: &crate::Rows) -> VersionResult<(Vec<f64>, Vec<u32>)> {
        match self {
            Self::ScalarV2(model) => Ok(model.predict(rows)?),
            _ => Err(VersionError::UnsupportedVersion(
                "model has no scalar prediction route".into(),
            )),
        }
    }

    /// Fitted ordered export starts each caller from the captured continuation.
    pub fn emit_ordered_fitted_cpp17(&self) -> VersionResult<DifferenceFittedCpp> {
        match self {
            Self::DifferenceV2(model) => model.emit_fitted_cpp17(),
            _ => Err(VersionError::UnsupportedVersion(
                "model has no ordered fitted export route".into(),
            )),
        }
    }

    /// B14 fitted integration export preserves its typed checkpoint codec.
    pub fn emit_integral_fitted_cpp17(
        &self,
    ) -> VersionResult<super::cumulative_integration::IntegrationFittedCpp> {
        match self {
            Self::IntegrationV2(model) => model.emit_fitted_cpp17(),
            _ => Err(VersionError::UnsupportedVersion(
                "model has no integral fitted export route".into(),
            )),
        }
    }

    pub fn emit_time_gradient_fitted_cpp17(
        &self,
    ) -> VersionResult<super::timescale_nabla::TimeGradientFittedCpp> {
        match self {
            Self::TimeGradientV2(model) => model.emit_fitted_cpp17(),
            _ => Err(VersionError::UnsupportedVersion(
                "model has no time_gradient fitted export route".into(),
            )),
        }
    }

    pub fn emit_path_stats_fitted_cpp17(
        &self,
    ) -> VersionResult<super::realized_path_stats::PathStatsFittedCpp> {
        match self {
            Self::PathStatsV2(model) => model.emit_fitted_cpp17(),
            _ => Err(VersionError::UnsupportedVersion(
                "model has no path_stats fitted export route".into(),
            )),
        }
    }

    /// This is the ingress for tagged model/bundle JSON. The old direct serde
    /// Model loader remains byte-for-byte unchanged for legacy callers.
    pub fn from_json(json: &str) -> VersionResult<Self> {
        if json.len() > MAX_BYTES {
            return Err(VersionError::Malformed("model envelope bytes".into()));
        }
        let value: Value = serde_json::from_str(json)
            .map_err(|error| VersionError::Malformed(error.to_string()))?;
        // B05–B07: scalar envelope reader runs before any legacy dispatch.
        if value.get("schema").and_then(Value::as_str) == Some(super::scalar_program::MODEL_SCHEMA)
        {
            return ScalarModel::from_json(json)
                .map(Box::new)
                .map(Self::ScalarV2);
        }
        // B12: tagged model ingress, with no v1 deserialization fallback.
        if value.get("schema").and_then(Value::as_str) == Some("feat-math.feature-row-model") {
            return super::row_vector::Model::from_json(json).map(Self::RowVectorV2);
        }
        if value.get("schema").is_none() && value.get("format_version").is_none() {
            return serde_json::from_value(value)
                .map(Self::LegacyV1)
                .map_err(|error| VersionError::Malformed(error.to_string()));
        }
        if value.get("format_version").and_then(Value::as_u64) != Some(2) {
            return Err(VersionError::UnsupportedVersion("model envelope".into()));
        }
        if value.get("schema").and_then(Value::as_str) == Some("feat-math.causal-difference-model")
        {
            return DifferenceModel::from_json(json)
                .map(Box::new)
                .map(Self::DifferenceV2);
        }
        if value.get("schema").and_then(Value::as_str) == Some("feat-math.causal-integral-model") {
            return super::cumulative_integration::IntegrationModel::from_json(json)
                .map(Box::new)
                .map(Self::IntegrationV2);
        }
        if value.get("schema").and_then(Value::as_str)
            == Some("feat-math.causal-time-gradient-model")
        {
            return super::timescale_nabla::TimeGradientModel::from_json(json)
                .map(Box::new)
                .map(Self::TimeGradientV2);
        }
        if value.get("schema").and_then(Value::as_str) == Some("feat-math.causal-path-stats-model")
        {
            return super::realized_path_stats::PathStatsModel::from_json(json)
                .map(Box::new)
                .map(Self::PathStatsV2);
        }
        let wire: ModelWireV2 = serde_json::from_value(value)
            .map_err(|error| VersionError::Malformed(error.to_string()))?;
        validate_model_wire(&wire)?;
        Ok(Self::ReservedV2(ReservedModel { wire }))
    }

    pub fn predict_ordered(&mut self, rows: &[OrderedRow]) -> VersionResult<Vec<f64>> {
        match self {
            Self::DifferenceV2(model) => Ok(model.predict(rows)?),
            Self::IntegrationV2(model) => Ok(model.predict(rows)?),
            Self::TimeGradientV2(model) => Ok(model.predict(rows)?),
            Self::PathStatsV2(model) => Ok(model.predict(rows)?),
            _ => Err(VersionError::UnsupportedVersion(
                "model has no ordered prediction route".into(),
            )),
        }
    }

    pub fn legacy_model(&self) -> VersionResult<&Model> {
        match self {
            Self::LegacyV1(model) => Ok(model),
            Self::ScalarV2(_) => Err(VersionError::UnsupportedVersion(
                "v2 scalar model cannot decode as a v1 model".into(),
            )),
            Self::RowVectorV2(_) => Err(VersionError::UnsupportedVersion(
                "v2 row-vector model cannot decode as a v1 model".into(),
            )),
            Self::DifferenceV2(_) => Err(VersionError::UnsupportedVersion(
                "ordered input required for difference prediction".into(),
            )),
            Self::IntegrationV2(_) => Err(VersionError::UnsupportedVersion(
                "ordered input required for integral prediction".into(),
            )),
            Self::TimeGradientV2(_) => Err(VersionError::UnsupportedVersion(
                "ordered input required for time_gradient prediction".into(),
            )),
            Self::PathStatsV2(_) => Err(VersionError::UnsupportedVersion(
                "ordered input required for path_stats prediction".into(),
            )),
            Self::ReservedV2(_) => Err(VersionError::UnsupportedVersion(
                "version 2 models are not executable".into(),
            )),
        }
    }
}
