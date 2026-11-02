//! Test-only fixture: synthetic qualified programs with planted sentinels.
//!
//! Every program is canonicalized at test time and admitted through the same path the state
//! reader uses, so no line can be non-canonical by hand. Every provenance field carries a
//! distinctive digest, the adaptive line carries a burn digest, and the family labels are
//! `canaryfamily01` and `canaryfamily02`. [`FakeState`] holds admitted lines as a learning
//! State would: one measured entry and one exact-identity pointer per line.

use crate::fragment::INPUT_STAGE;
use crate::mine::mine;
use crate::source::{
    Admission, FamilyMap, LabelOrigin, QualifiedProgram, QualifiedSet, QualifiedSource, TaskLabel,
};
use crate::table::{MineParameters, MotifTable};
use crate::teaching::{
    BurnReceipt, ContextFields, EligibleState, GuardFields, PointerKeyFields, PointerRecord,
    QualifiedEntry, QualifiedRead, ENTRY_SCHEMA, SET_SCHEMA,
};
use crate::validate::{CheckResult, MotifResult, Validation, VALIDATION_SCHEMA};
use crate::{sha256_hex, NAME_PREFIX};
use anyhow::{ensure, Result};
use feat_math::GenomeIr;
use std::cell::Cell;
use std::collections::BTreeMap;

mod programs;
mod sentinels;
mod state;

pub(crate) use programs::*;
pub(crate) use sentinels::*;
pub(crate) use state::*;
