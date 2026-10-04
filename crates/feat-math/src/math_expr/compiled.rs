use super::fault::Collect;
use super::*;

mod scalar;

#[derive(Clone, Copy, Debug)]
enum Instruction {
    Constant(f64),
    Variable(Var),
    Neg(u8),
    Binary(BinOp, u8, u8),
    Call(Func, [u8; 3], u8),
}

/// Fixed-arity postorder instructions. The AST remains the sole evolutionary
/// representation; this derived form preserves operand and reduction order.
#[derive(Clone, Debug)]
pub struct CompiledMathExpr {
    code: Vec<Instruction>,
    element_dependent: Vec<bool>,
}

impl MathExpr {
    pub fn compile(&self) -> Result<CompiledMathExpr> {
        self.check_structure(MAX_NODES, MAX_DEPTH)?;
        let mut compiled = CompiledMathExpr {
            code: Vec::new(),
            element_dependent: Vec::new(),
        };
        lower(&self.root, &mut compiled)?;
        Ok(compiled)
    }
}

fn lower(expr: &Expr, out: &mut CompiledMathExpr) -> Result<u8> {
    let (instruction, dependent) = match expr {
        Expr::Const(value) => (Instruction::Constant(*value), false),
        Expr::Var(var) => (Instruction::Variable(*var), *var == Var::X),
        Expr::Unary(UnOp::Neg, child) => {
            let a = lower(child, out)?;
            (Instruction::Neg(a), out.element_dependent[a as usize])
        }
        Expr::Binary(op, a, b) => {
            let a = lower(a, out)?;
            let b = lower(b, out)?;
            (
                Instruction::Binary(*op, a, b),
                out.element_dependent[a as usize] || out.element_dependent[b as usize],
            )
        }
        Expr::Call(func, args) => {
            validate_arity(*func, args.len())?;
            let mut registers = [0; 3];
            let mut dependent = false;
            for (slot, arg) in registers.iter_mut().zip(args) {
                *slot = lower(arg, out)?;
                dependent |= out.element_dependent[*slot as usize];
            }
            (
                Instruction::Call(*func, registers, args.len() as u8),
                dependent,
            )
        }
    };
    if out.code.len() >= MAX_NODES {
        bail!("compiled expression node limit");
    }
    let index = out.code.len() as u8;
    out.code.push(instruction);
    out.element_dependent.push(dependent);
    Ok(index)
}

impl CompiledMathExpr {
    pub(crate) fn uses_row_summaries(&self) -> bool {
        self.code.iter().any(|instruction| {
            matches!(instruction, Instruction::Variable(var) if summary_index(*var).is_some())
        })
    }

    pub fn eval_row(&self, row: &[f64]) -> Result<f64> {
        let mut registers = [0.0; MAX_NODES];
        let mut summaries = [None; 7];
        for (i, instruction) in self.code.iter().enumerate() {
            registers[i] = evaluate(*instruction, &registers, None, row, &mut summaries)?;
        }
        Ok(safe(registers[self.code.len() - 1]))
    }

    pub fn eval_element(&self, x: f64, row: &[f64]) -> Result<f64> {
        let mut registers = [0.0; MAX_NODES];
        let mut summaries = [None; 7];
        for (i, instruction) in self.code.iter().enumerate() {
            registers[i] = evaluate(*instruction, &registers, Some(x), row, &mut summaries)?;
        }
        Ok(safe(registers[self.code.len() - 1]))
    }

    /// One scratch register bank and one set of row summaries per row. An
    /// element failure only zeros that element, as in the legacy map caller.
    pub fn map_row(&self, row: &[f64]) -> Vec<f64> {
        let mut registers = [0.0; MAX_NODES];
        let mut summaries = [None; 7];
        for (i, instruction) in self.code.iter().enumerate() {
            if !self.element_dependent[i] {
                match evaluate(*instruction, &registers, None, row, &mut summaries) {
                    Ok(v) => registers[i] = v,
                    Err(_) => return vec![0.0; row.len()],
                }
            }
        }
        row.iter()
            .map(|x| {
                for (i, instruction) in self.code.iter().enumerate() {
                    if self.element_dependent[i] {
                        match evaluate(*instruction, &registers, Some(*x), row, &mut summaries) {
                            Ok(v) => registers[i] = v,
                            Err(_) => return 0.0,
                        }
                    }
                }
                safe(registers[self.code.len() - 1])
            })
            .collect()
    }
}

fn summary_index(var: Var) -> Option<usize> {
    match var {
        Var::RowMean => Some(0),
        Var::RowSum => Some(1),
        Var::RowL1 => Some(2),
        Var::RowL2 => Some(3),
        Var::RowMin => Some(4),
        Var::RowMax => Some(5),
        Var::RowStd => Some(6),
        _ => None,
    }
}
fn evaluate(
    instruction: Instruction,
    registers: &[f64; MAX_NODES],
    x: Option<f64>,
    row: &[f64],
    summaries: &mut [Option<f64>; 7],
) -> Result<f64> {
    Ok(match instruction {
        Instruction::Constant(v) => v,
        Instruction::Variable(var) => {
            if let Some(i) = summary_index(var) {
                match summaries[i] {
                    Some(value) => value,
                    None => {
                        let value = eval_var(var, &EvalContext { x, row })?;
                        summaries[i] = Some(value);
                        value
                    }
                }
            } else {
                eval_var(var, &EvalContext { x, row })?
            }
        }
        Instruction::Neg(a) => -registers[a as usize],
        Instruction::Binary(op, a, b) => {
            compute_binary(op, registers[a as usize], registers[b as usize])
        }
        Instruction::Call(func, args, n) => {
            let mut values = [0.0; 3];
            for i in 0..n as usize {
                values[i] = registers[args[i] as usize];
            }
            compute_func(func, values)?
        }
    })
}

/// The traced forms (kernel2 WP-1.3). Each equals its interpreter twin bit for bit and fault for
/// fault: the postorder instructions run in the interpreter's evaluation order and stop at the
/// same first error. The legacy entry points above keep `evaluate` and are unchanged.
impl CompiledMathExpr {
    /// [`CompiledMathExpr::eval_row`] with the faults its evaluation raised, in the compat shape
    /// of [`MathExpr::eval_row_traced`], whose result it equals.
    pub fn eval_row_traced(&self, row: &[f64]) -> (f64, FaultSet) {
        self.run_traced(None, row)
    }

    /// [`CompiledMathExpr::eval_element`] with the faults its evaluation raised, in the compat
    /// shape of [`MathExpr::eval_element_traced`], whose result it equals.
    pub fn eval_element_traced(&self, x: f64, row: &[f64]) -> (f64, FaultSet) {
        self.run_traced(Some(x), row)
    }

    /// [`CompiledMathExpr::map_row`] with the faults of each cell. The values are `map_row`'s,
    /// bit for bit, and the faults of cell `j` are those of
    /// [`MathExpr::eval_element_traced`] on `row[j]`. A stage's faults over the row are its
    /// cells': an empty row has no cell, so it has no fault.
    ///
    /// As in `map_row`, the element-independent instructions run once per row. Each keeps the
    /// faults its own evaluation raised, and each cell re-raises those of every such instruction
    /// its own evaluation reaches, in instruction order, stopping where that evaluation stops. A
    /// failed instruction therefore never reports the faults of a later one, and a failure in a
    /// later instruction never hides the faults of an earlier element-dependent one.
    pub fn map_row_traced(&self, row: &[f64]) -> (Vec<f64>, Vec<FaultSet>) {
        let mut registers = [0.0; MAX_NODES];
        let mut summaries = [None; 7];
        let mut independent = [FaultSet::EMPTY; MAX_NODES];
        let mut failed_at = None;
        for (i, instruction) in self.code.iter().enumerate() {
            if !self.element_dependent[i] {
                let mut sink = Collect::default();
                let result = evaluate_with(
                    *instruction,
                    &registers,
                    None,
                    row,
                    &mut summaries,
                    &mut sink,
                );
                independent[i] = sink.0;
                match result {
                    Ok(v) => registers[i] = v,
                    Err(_) => {
                        failed_at = Some(i);
                        break;
                    }
                }
            }
        }
        let mut values = Vec::with_capacity(row.len());
        let mut faults = Vec::with_capacity(row.len());
        for x in row {
            let mut sink = Collect::default();
            let mut failed = false;
            for (i, instruction) in self.code.iter().enumerate() {
                if self.element_dependent[i] {
                    let result = evaluate_with(
                        *instruction,
                        &registers,
                        Some(*x),
                        row,
                        &mut summaries,
                        &mut sink,
                    );
                    match result {
                        Ok(v) => registers[i] = v,
                        Err(_) => {
                            failed = true;
                            break;
                        }
                    }
                } else {
                    sink.raise(independent[i]);
                    if failed_at == Some(i) {
                        failed = true;
                        break;
                    }
                }
            }
            // An element failure is 0.0, as in `map_row`.
            values.push(if failed {
                0.0
            } else {
                safe_with(registers[self.code.len() - 1], &mut sink)
            });
            faults.push(sink.0);
        }
        (values, faults)
    }

    fn run_traced(&self, x: Option<f64>, row: &[f64]) -> (f64, FaultSet) {
        let mut registers = [0.0; MAX_NODES];
        let mut summaries = [None; 7];
        let mut sink = Collect::default();
        for (i, instruction) in self.code.iter().enumerate() {
            match evaluate_with(*instruction, &registers, x, row, &mut summaries, &mut sink) {
                Ok(v) => registers[i] = v,
                // The compat shape: an evaluation error is 0.0, with its faults.
                Err(_) => return (0.0, sink.0),
            }
        }
        (safe_with(registers[self.code.len() - 1], &mut sink), sink.0)
    }
}

/// [`evaluate`] with a fault sink. A row summary is cached with the faults its computation
/// raised, and every read of the cached summary raises them again, as the interpreter does by
/// computing the summary at each read.
fn evaluate_with<S: FaultSink>(
    instruction: Instruction,
    registers: &[f64; MAX_NODES],
    x: Option<f64>,
    row: &[f64],
    summaries: &mut [Option<(f64, FaultSet)>; 7],
    sink: &mut S,
) -> Result<f64> {
    Ok(match instruction {
        Instruction::Constant(v) => v,
        Instruction::Variable(var) => {
            if let Some(i) = summary_index(var) {
                match summaries[i] {
                    Some((value, faults)) => {
                        sink.raise(faults);
                        value
                    }
                    None => {
                        let mut own = Collect::default();
                        let value = eval_var_with(var, &EvalContext { x, row }, &mut own);
                        sink.raise(own.0);
                        let value = value?;
                        summaries[i] = Some((value, own.0));
                        value
                    }
                }
            } else {
                eval_var_with(var, &EvalContext { x, row }, sink)?
            }
        }
        Instruction::Neg(a) => -registers[a as usize],
        Instruction::Binary(op, a, b) => {
            compute_binary_with(op, registers[a as usize], registers[b as usize], sink)
        }
        Instruction::Call(func, args, n) => {
            let mut values = [0.0; 3];
            for i in 0..n as usize {
                values[i] = registers[args[i] as usize];
            }
            compute_func_with(func, values, sink)?
        }
    })
}
