//! Declared kernel capabilities. The frozen grammars and registry are checked against these rows.
//! Source names and probe construction stay in code; rendered data uses neutral element names.

use crate::heads::FittedHead;
use crate::kernel::ordered_axis::StageAxis;
use crate::kernel::wire_v2;
use crate::math_expr::{BinOp, FaultSet, Func, UnOp, Var};
use crate::ops::FittedOp;
use crate::registry::{self, OperatorRole, ShapeRule};
use crate::GenomeIr;
use serde::{Serialize, Serializer};
use std::collections::{BTreeMap, BTreeSet};
use std::sync::OnceLock;

include!("capabilities/expression_rows.rs");
include!("capabilities/manifest.rs");
include!("capabilities/grammar_validation.rs");
include!("capabilities/operation_rows.rs");
include!("capabilities/rendering.rs");
include!("capabilities/compatibility.rs");
