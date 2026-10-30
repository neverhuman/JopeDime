//! Reuse the raw protected operations without changing the frozen evaluator.
//! New policy helpers have their already-certified local bounds; legacy
//! intermediate arithmetic is sanitized only at the complete tree's output.
use super::super::fault::{Collect, FaultSink};
use super::super::{
    compute_binary_with, compute_func_with, eval_var_with, safe_with, EvalContext, Func, UnOp, Var,
    MAX_NODES,
};
use crate::kernel::numerical_policy::{asinh_v2, reduce_v2, NumericalPolicy, Reduction};
use crate::kernel::scalar_execution::{CompiledScalarProgram, Instruction, ScalarEvaluation};
use crate::kernel::scalar_program::{ScalarExpressionProgram, ScalarFunction, ScalarNode};
use crate::kernel::stable_exponential::StableExponential;
use crate::kernel::{Error, Result};

impl ScalarExpressionProgram {
    /// Evaluate explicitly in row scope. Missing policy keeps v1 arithmetic.
    /// A hard error retains preceding faults instead of becoming a successful zero.
    pub fn evaluate_row(&self, row: &[f64]) -> ScalarEvaluation {
        self.evaluate_context(&EvalContext { x: None, row })
    }

    /// Evaluate explicitly in element scope, with `x` and the original row.
    pub fn evaluate_element(&self, x: f64, row: &[f64]) -> ScalarEvaluation {
        self.evaluate_context(&EvalContext { x: Some(x), row })
    }

    /// Compile the admitted AST in left-to-right postorder. Construction and
    /// decoding already enforce the closed function, arity and resource budgets.
    pub fn compile(&self) -> Result<CompiledScalarProgram> {
        let mut code = Vec::new();
        lower(self.ast(), &mut code)?;
        Ok(CompiledScalarProgram {
            code,
            numerical_policy: self.numerical_policy(),
        })
    }

    fn evaluate_context(&self, context: &EvalContext<'_>) -> ScalarEvaluation {
        let mut sink = Collect::default();
        let value = interpret(self.ast(), context, self.numerical_policy(), &mut sink);
        finish(value, &mut sink)
    }
}

impl CompiledScalarProgram {
    pub fn evaluate_row(&self, row: &[f64]) -> ScalarEvaluation {
        self.evaluate_context(&EvalContext { x: None, row })
    }

    pub fn evaluate_element(&self, x: f64, row: &[f64]) -> ScalarEvaluation {
        self.evaluate_context(&EvalContext { x: Some(x), row })
    }

    fn evaluate_context(&self, context: &EvalContext<'_>) -> ScalarEvaluation {
        let mut registers = [0.0; MAX_NODES];
        let mut sink = Collect::default();
        let mut result = Ok(0.0);
        for (index, instruction) in self.code.iter().enumerate() {
            result = instruction_value(
                *instruction,
                &registers,
                context,
                self.numerical_policy,
                &mut sink,
            );
            match &result {
                Ok(value) => registers[index] = *value,
                Err(_) => break,
            }
        }
        finish(result, &mut sink)
    }
}

fn finish(value: Result<f64>, sink: &mut Collect) -> ScalarEvaluation {
    ScalarEvaluation {
        value: value.map(|v| safe_with(v, sink)),
        faults: sink.0,
    }
}

fn interpret(
    node: &ScalarNode,
    context: &EvalContext<'_>,
    policy: NumericalPolicy,
    sink: &mut Collect,
) -> Result<f64> {
    match node {
        ScalarNode::Const(value) => Ok(*value),
        ScalarNode::Var(var) => variable(*var, context, policy, sink),
        ScalarNode::Unary(UnOp::Neg, child) => Ok(-interpret(child, context, policy, sink)?),
        ScalarNode::Binary(op, left, right) => {
            let a = interpret(left, context, policy, sink)?;
            let b = interpret(right, context, policy, sink)?;
            Ok(compute_binary_with(*op, a, b, sink))
        }
        ScalarNode::Call(func, args) => {
            let mut values = [0.0; 3];
            for (value, child) in values.iter_mut().zip(args) {
                *value = interpret(child, context, policy, sink)?;
            }
            function(*func, values, policy, sink)
        }
    }
}

fn variable(
    var: Var,
    context: &EvalContext<'_>,
    policy: NumericalPolicy,
    sink: &mut Collect,
) -> Result<f64> {
    let reduction = match var {
        Var::RowSum => Some(Reduction::Sum),
        Var::RowMean => Some(Reduction::Mean),
        Var::RowL1 => Some(Reduction::L1),
        Var::RowL2 => Some(Reduction::L2),
        _ => None,
    };
    if let (NumericalPolicy::StableV2, Some(reduction)) = (policy, reduction) {
        let (value, faults) = reduce_v2(reduction, context.row);
        sink.raise(faults);
        return Ok(value);
    }
    eval_var_with(var, context, sink).map_err(|error| {
        if matches!(var, Var::Col(_)) {
            Error::Bounds(error.to_string())
        } else {
            Error::Domain(error.to_string())
        }
    })
}

fn function(
    function: ScalarFunction,
    values: [f64; 3],
    policy: NumericalPolicy,
    sink: &mut Collect,
) -> Result<f64> {
    let guarded = match function {
        ScalarFunction::V1(Func::Asinh) if policy == NumericalPolicy::StableV2 => {
            asinh_v2(values[0])
        }
        ScalarFunction::InverseLink(link) => link.evaluate(values[0]),
        ScalarFunction::V1(func) => {
            return compute_func_with(func, values, sink)
                .map_err(|error| Error::Domain(error.to_string()));
        }
        ScalarFunction::Expm1 => StableExponential::Expm1.evaluate(values[0]),
        ScalarFunction::Log1mexp => StableExponential::Log1mexp.evaluate(values[0]),
    };
    sink.raise(guarded.1);
    Ok(guarded.0)
}

fn lower(node: &ScalarNode, code: &mut Vec<Instruction>) -> Result<u8> {
    let instruction = match node {
        ScalarNode::Const(value) => Instruction::Constant(*value),
        ScalarNode::Var(var) => Instruction::Variable(*var),
        ScalarNode::Unary(UnOp::Neg, child) => Instruction::Neg(lower(child, code)?),
        ScalarNode::Binary(op, left, right) => {
            let a = lower(left, code)?;
            let b = lower(right, code)?;
            Instruction::Binary(*op, a, b)
        }
        ScalarNode::Call(func, args) => {
            let mut registers = [0; 3];
            for (register, child) in registers.iter_mut().zip(args) {
                *register = lower(child, code)?;
            }
            Instruction::Call(*func, registers, args.len() as u8)
        }
    };
    if code.len() >= MAX_NODES {
        return Err(Error::Resource("compiled scalar node limit".into()));
    }
    let index = code.len() as u8;
    code.push(instruction);
    Ok(index)
}

fn instruction_value(
    instruction: Instruction,
    registers: &[f64; MAX_NODES],
    context: &EvalContext<'_>,
    policy: NumericalPolicy,
    sink: &mut Collect,
) -> Result<f64> {
    match instruction {
        Instruction::Constant(value) => Ok(value),
        Instruction::Variable(var) => variable(var, context, policy, sink),
        Instruction::Neg(a) => Ok(-registers[a as usize]),
        Instruction::Binary(op, a, b) => Ok(compute_binary_with(
            op,
            registers[a as usize],
            registers[b as usize],
            sink,
        )),
        Instruction::Call(func, args, count) => {
            let mut values = [0.0; 3];
            for index in 0..count as usize {
                values[index] = registers[args[index] as usize];
            }
            function(func, values, policy, sink)
        }
    }
}
