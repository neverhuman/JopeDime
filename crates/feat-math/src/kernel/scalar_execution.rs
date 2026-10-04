//! Standalone execution of a bounded scalar program, separate from legacy
//! model/export/search admission. Unsupported functions refuse the whole tree.
use super::numerical_policy::NumericalPolicy;
use super::scalar_program::ScalarFunction;
use super::Result;
use crate::math_expr::{BinOp, FaultSet, Var};

/// A hard refusal remains an error; faults raised before it are retained.
/// Successful values have the selected contract's protected final bound.
#[derive(Clone, Debug)]
pub struct ScalarEvaluation {
    pub value: Result<f64>,
    pub faults: FaultSet,
}

/// Derived postorder code, bounded by the program's 64-node admission limit.
/// It has no independent codec, model state or generation admission.
#[derive(Clone, Debug)]
pub struct CompiledScalarProgram {
    pub(crate) code: Vec<Instruction>,
    pub(crate) numerical_policy: NumericalPolicy,
}

impl CompiledScalarProgram {
    pub fn numerical_policy(&self) -> NumericalPolicy {
        self.numerical_policy
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) enum Instruction {
    Constant(f64),
    Variable(Var),
    Neg(u8),
    Binary(BinOp, u8, u8),
    Call(ScalarFunction, [u8; 3], u8),
}
