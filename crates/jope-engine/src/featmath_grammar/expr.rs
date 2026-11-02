use super::{strings, ExprSpec};

pub(super) fn definition() -> ExprSpec {
    ExprSpec {
        vars_map: strings(&["x"]),
        vars_common: strings(&["row_mean", "row_std", "row_min", "row_max", "row_l2"]),
        funcs1: strings(&["abs", "sqrt", "log1p", "sin", "cos", "tanh", "sigmoid"]),
        funcs2: strings(&["min", "max", "step"]),
        funcs3: strings(&["clip"]),
        binops: strings(&["+", "-", "*", "/"]),
        consts: strings(&[
            "-2", "-1", "-0.5", "-0.25", "-0.1", "0", "0.05", "0.1", "0.25", "0.5", "1", "2", "3",
            "5", "10", "20",
        ]),
        max_col: 8,
        max_depth: 4,
        max_nodes: 15,
    }
}
