use super::*;
use crate::math_expr::fault::Collect;

#[derive(Clone, Copy, Debug)]
pub(super) enum Op {
    Constant(f64),
    Param(u8),
    Variable(Var),
    Neg(u8),
    Binary(BinOp, u8, u8),
    Call(Func, [u8; 3], u8),
}

#[derive(Clone, Copy, Debug)]
pub(super) struct Dual {
    v: f64,
    d: Lanes,
}

pub(super) const DUAL_ZERO: Dual = Dual { v: 0.0, d: ZERO };

/// One row's evaluation: `eval_row_traced`'s value and faults, and the tangent of the value.
#[derive(Clone, Copy, Debug)]
pub(super) struct RowDual {
    pub value: f64,
    pub faults: FaultSet,
    pub tangent: Lanes,
    pub nonsmooth: bool,
}

/// A row expression in postorder with parameter sites.
#[derive(Clone, Debug)]
pub(super) struct Program {
    code: Vec<Op>,
    params: usize,
}

impl Program {
    /// `root` with the literal node at `sites[k]` as parameter `k`. Every site must be a literal
    /// (`Const` or `Neg(Const)`), and none may lie inside another.
    pub(in crate::math_expr::evolution::v2::refinement::lm) fn lower(
        root: &Expr,
        sites: &[Vec<usize>],
    ) -> Result<Self> {
        if sites.len() > MAX_PARAMS {
            bail!("{} parameter sites, at most {MAX_PARAMS}", sites.len());
        }
        let mut program = Self {
            code: Vec::new(),
            params: sites.len(),
        };
        let mut path = Vec::new();
        program.lower_node(root, sites, &mut path)?;
        let used = program
            .code
            .iter()
            .filter(|op| matches!(op, Op::Param(_)))
            .count();
        if used != sites.len() {
            bail!("a parameter site is not a node of the expression");
        }
        Ok(program)
    }

    pub(in crate::math_expr::evolution::v2::refinement::lm) fn params(&self) -> usize {
        self.params
    }

    fn lower_node(
        &mut self,
        expr: &Expr,
        sites: &[Vec<usize>],
        path: &mut Vec<usize>,
    ) -> Result<u8> {
        let op = if let Some(k) = sites.iter().position(|site| site == path) {
            if literal_value(expr).is_none() {
                bail!("parameter site is not a literal");
            }
            Op::Param(k as u8)
        } else {
            match expr {
                Expr::Const(value) => Op::Constant(*value),
                Expr::Var(var) => Op::Variable(*var),
                Expr::Unary(UnOp::Neg, child) => Op::Neg(self.lower_child(child, 0, sites, path)?),
                Expr::Binary(op, a, b) => {
                    let a = self.lower_child(a, 0, sites, path)?;
                    let b = self.lower_child(b, 1, sites, path)?;
                    Op::Binary(*op, a, b)
                }
                Expr::Call(func, args) => {
                    validate_arity(*func, args.len())?;
                    let mut registers = [0; 3];
                    for (i, (slot, arg)) in registers.iter_mut().zip(args).enumerate() {
                        *slot = self.lower_child(arg, i, sites, path)?;
                    }
                    Op::Call(*func, registers, args.len() as u8)
                }
            }
        };
        if self.code.len() >= MAX_NODES {
            bail!("dual expression node limit");
        }
        self.code.push(op);
        Ok((self.code.len() - 1) as u8)
    }

    fn lower_child(
        &mut self,
        child: &Expr,
        index: usize,
        sites: &[Vec<usize>],
        path: &mut Vec<usize>,
    ) -> Result<u8> {
        path.push(index);
        let register = self.lower_node(child, sites, path);
        path.pop();
        register
    }

    /// The row's value, faults and tangent at parameters `theta` (one value per site). As in
    /// `eval_row_traced`, an evaluation error gives the value 0.0 with the faults raised so far,
    /// and then the tangent is zero.
    pub(in crate::math_expr::evolution::v2::refinement::lm) fn row(
        &self,
        theta: &[f64],
        row: &[f64],
    ) -> RowDual {
        debug_assert_eq!(theta.len(), self.params);
        let mut registers = [DUAL_ZERO; MAX_NODES];
        let mut sink = Collect::default();
        let mut nonsmooth = false;
        for (i, op) in self.code.iter().enumerate() {
            match self.step(*op, &registers, theta, row, &mut sink, &mut nonsmooth) {
                Ok(dual) => registers[i] = dual,
                Err(_) => {
                    return RowDual {
                        value: 0.0,
                        faults: sink.0,
                        tangent: ZERO,
                        nonsmooth,
                    }
                }
            }
        }
        let top = registers[self.code.len() - 1];
        let value = safe_with(top.v, &mut sink);
        let tangent = if top.v.is_finite() {
            clamped(
                self.params,
                top.v,
                -MAX_ABS,
                MAX_ABS,
                &top.d,
                &mut nonsmooth,
            )
        } else {
            nonsmooth = true;
            ZERO
        };
        RowDual {
            value,
            faults: sink.0,
            tangent,
            nonsmooth,
        }
    }

    fn step(
        &self,
        op: Op,
        registers: &[Dual; MAX_NODES],
        theta: &[f64],
        row: &[f64],
        sink: &mut Collect,
        nonsmooth: &mut bool,
    ) -> Result<Dual> {
        let p = self.params;
        Ok(match op {
            Op::Constant(v) => Dual { v, d: ZERO },
            Op::Param(k) => {
                let mut d = ZERO;
                d[k as usize] = 1.0;
                Dual {
                    v: theta[k as usize],
                    d,
                }
            }
            Op::Variable(var) => Dual {
                v: eval_var_with(var, &EvalContext { x: None, row }, sink)?,
                d: ZERO,
            },
            Op::Neg(a) => {
                let a = &registers[a as usize];
                let d = if a.v.is_finite() {
                    scale(p, -1.0, &a.d)
                } else {
                    *nonsmooth = true;
                    ZERO
                };
                Dual { v: -a.v, d }
            }
            Op::Binary(op, a, b) => {
                let (a, b) = (&registers[a as usize], &registers[b as usize]);
                let v = compute_binary_with(op, a.v, b.v, sink);
                Dual {
                    v,
                    d: binary_tangent(p, op, a, b, v, nonsmooth),
                }
            }
            Op::Call(func, args, n) => {
                let n = n as usize;
                let mut values = [0.0; 3];
                let mut operands = [&DUAL_ZERO; 3];
                for i in 0..n {
                    operands[i] = &registers[args[i] as usize];
                    values[i] = operands[i].v;
                }
                let v = compute_func_with(func, values, sink)?;
                Dual {
                    v,
                    d: call_tangent(p, func, &operands[..n], v, nonsmooth),
                }
            }
        })
    }
}
