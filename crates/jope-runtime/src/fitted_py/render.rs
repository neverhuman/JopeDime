use crate::error::Failure;
use anyhow::Result;

/// A Python literal.
#[derive(Clone, Debug)]
pub(crate) enum Py {
    None,
    B(bool),
    I(usize),
    /// A tree node's leaf marker.
    Leaf,
    F(f64),
    Str(String),
    List(Vec<Py>),
    Tuple(Vec<Py>),
    Dict(Vec<(Py, Py)>),
}

impl Py {
    pub(crate) fn floats(values: &[f64]) -> Self {
        Self::List(values.iter().copied().map(Self::F).collect())
    }

    pub(crate) fn rows(rows: &[Vec<f64>]) -> Self {
        Self::List(rows.iter().map(|row| Self::floats(row)).collect())
    }

    pub(crate) fn ints(values: &[usize]) -> Self {
        Self::List(values.iter().copied().map(Self::I).collect())
    }

    pub(crate) fn pairs(pairs: &[(usize, usize)]) -> Self {
        Self::List(
            pairs
                .iter()
                .map(|&(a, b)| Self::Tuple(vec![Self::I(a), Self::I(b)]))
                .collect(),
        )
    }
}

pub(crate) fn render(value: &Py) -> Result<String> {
    let mut out = String::new();
    write_py(value, &mut out)?;
    Ok(out)
}

fn write_py(value: &Py, out: &mut String) -> Result<()> {
    match value {
        Py::None => out.push_str("None"),
        Py::B(b) => out.push_str(if *b { "True" } else { "False" }),
        Py::I(i) => out.push_str(&i.to_string()),
        Py::Leaf => out.push_str("-1"),
        Py::F(v) => {
            if !v.is_finite() {
                return Err(Failure::integrity("non-finite fitted state"));
            }
            // `{:?}` is the shortest form that round-trips, always with a `.` or an exponent.
            out.push_str(&format!("{v:?}"));
        }
        Py::Str(s) => {
            out.push('"');
            for c in s.chars() {
                match c {
                    '"' => out.push_str("\\\""),
                    '\\' => out.push_str("\\\\"),
                    ' '..='~' => out.push(c),
                    _ => out.push_str(&format!("\\U{:08x}", u32::from(c))),
                }
            }
            out.push('"');
        }
        Py::Dict(items) => {
            out.push('{');
            for (i, (key, value)) in items.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                write_py(key, out)?;
                out.push_str(": ");
                write_py(value, out)?;
            }
            out.push('}');
        }
        Py::List(items) | Py::Tuple(items) => {
            let tuple = matches!(value, Py::Tuple(_));
            out.push(if tuple { '(' } else { '[' });
            for (i, item) in items.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                write_py(item, out)?;
            }
            if tuple && items.len() == 1 {
                out.push(',');
            }
            out.push(if tuple { ')' } else { ']' });
        }
    }
    Ok(())
}
