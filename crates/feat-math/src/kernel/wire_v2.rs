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
use super::{cumulative_integration, realized_path_stats, row_vector, timescale_nabla};
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
pub enum KernelWireDomain {
    Wire,
    Function,
    Variable,
}

/// Backward-compatible name for the v2 kernel wire identity domain.
pub type Domain = KernelWireDomain;

impl KernelWireDomain {
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
    pub domain: KernelWireDomain,
    pub id: u16,
}

impl ReservationRef {
    pub fn new(domain: KernelWireDomain, id: u16) -> VersionResult<Self> {
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
    pub domain: KernelWireDomain,
    pub id: u16,
    pub name: Option<&'static str>,
    pub lifecycle: &'static str,
    pub generatable: bool,
}

pub fn manifest_rows() -> Vec<ReservationRow> {
    [
        KernelWireDomain::Wire,
        KernelWireDomain::Function,
        KernelWireDomain::Variable,
    ]
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
mod model;
mod program;

pub use model::{ReservedModel, VersionedModel};
pub use program::{export_report, ReservedProgram, VersionedProgram};
