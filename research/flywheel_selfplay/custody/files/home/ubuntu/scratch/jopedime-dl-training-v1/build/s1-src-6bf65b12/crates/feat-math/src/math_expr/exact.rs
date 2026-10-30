//! Codec adapter for the historical serde AST; constants retain IEEE bits.
use super::*;
use crate::kernel::{
    codec::{Reader, Writer},
    Error, Result as KernelResult,
};

// Append-only IDs. These do not depend on Rust enum discriminants.
const VARS: &[Var] = &[
    Var::X,
    Var::RowMean,
    Var::RowSum,
    Var::RowL1,
    Var::RowL2,
    Var::RowMin,
    Var::RowMax,
    Var::RowStd,
];
const BINARY: &[BinOp] = &[BinOp::Add, BinOp::Sub, BinOp::Mul, BinOp::Div, BinOp::Pow];
const FUNCTIONS: &[Func] = &[
    Func::Abs,
    Func::Sqrt,
    Func::Log1p,
    Func::Sin,
    Func::Cos,
    Func::Tanh,
    Func::Sigmoid,
    Func::Min,
    Func::Max,
    Func::Clip,
    Func::Step,
    Func::Exp,
    Func::Log,
    Func::Asinh,
    Func::Atan,
    Func::Erf,
    Func::Sign,
    Func::Floor,
    Func::Round,
];

impl MathExpr {
    pub(crate) fn write_exact(&self, w: &mut Writer) -> KernelResult<()> {
        let mut remaining = MAX_NODES;
        write_expr(w, &self.root, 1, &mut remaining)
    }
    pub(crate) fn read_exact(r: &mut Reader<'_>) -> KernelResult<Self> {
        let mut remaining = MAX_NODES;
        Ok(Self {
            root: read_expr(r, 1, &mut remaining)?,
        })
    }
}

fn write_expr(
    w: &mut Writer,
    expr: &Expr,
    depth: usize,
    remaining: &mut usize,
) -> KernelResult<()> {
    if depth > MAX_DEPTH || *remaining == 0 {
        return Err(Error::Resource("math AST limit".into()));
    }
    *remaining -= 1;
    match expr {
        Expr::Const(v) => {
            w.byte(0)?;
            w.word(v.to_bits())?;
        }
        Expr::Var(Var::Col(i)) => {
            w.byte(1)?;
            w.word(*i as u64)?;
        }
        Expr::Var(var) => {
            w.byte(2)?;
            w.byte(VARS.iter().position(|v| v == var).unwrap() as u8)?;
        }
        Expr::Unary(UnOp::Neg, child) => {
            w.byte(3)?;
            write_expr(w, child, depth + 1, remaining)?;
        }
        Expr::Binary(op, a, b) => {
            w.byte(4)?;
            w.byte(BINARY.iter().position(|v| v == op).unwrap() as u8)?;
            write_expr(w, a, depth + 1, remaining)?;
            write_expr(w, b, depth + 1, remaining)?;
        }
        Expr::Call(func, args) => {
            validate_arity(*func, args.len()).map_err(|e| Error::Syntax(e.to_string()))?;
            w.byte(5)?;
            w.byte(FUNCTIONS.iter().position(|v| v == func).unwrap() as u8)?;
            w.word(args.len() as u64)?;
            for arg in args {
                write_expr(w, arg, depth + 1, remaining)?;
            }
        }
    }
    Ok(())
}

fn tagged<T: Copy>(r: &mut Reader<'_>, table: &[T]) -> KernelResult<T> {
    table
        .get(r.byte()? as usize)
        .copied()
        .ok_or_else(|| Error::Syntax("math operation tag".into()))
}
fn read_expr(r: &mut Reader<'_>, depth: usize, remaining: &mut usize) -> KernelResult<Expr> {
    if depth > MAX_DEPTH || *remaining == 0 {
        return Err(Error::Resource("math AST limit".into()));
    }
    *remaining -= 1;
    Ok(match r.byte()? {
        0 => Expr::Const(f64::from_bits(r.word()?)),
        1 => Expr::Var(Var::Col(
            usize::try_from(r.word()?).map_err(|_| Error::Bounds("column index".into()))?,
        )),
        2 => Expr::Var(tagged(r, VARS)?),
        3 => Expr::Unary(UnOp::Neg, Box::new(read_expr(r, depth + 1, remaining)?)),
        4 => Expr::Binary(
            tagged(r, BINARY)?,
            Box::new(read_expr(r, depth + 1, remaining)?),
            Box::new(read_expr(r, depth + 1, remaining)?),
        ),
        5 => {
            let func = tagged(r, FUNCTIONS)?;
            let n = r.length()?;
            validate_arity(func, n).map_err(|e| Error::Syntax(e.to_string()))?;
            Expr::Call(
                func,
                (0..n)
                    .map(|_| read_expr(r, depth + 1, remaining))
                    .collect::<KernelResult<_>>()?,
            )
        }
        _ => return Err(Error::Syntax("math AST tag".into())),
    })
}
