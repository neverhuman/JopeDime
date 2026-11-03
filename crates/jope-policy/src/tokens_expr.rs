//! Expression parsing and rendering for the Python `jope.policy.tokens_expr` facade.
//!
//! The tokenizer, precedence, column conversion and constant snapping preserve the
//! reference module's behavior. Python remains in place while its token facade imports it.

use anyhow::{bail, ensure, Result};
use std::collections::BTreeMap;

use crate::grammar::Expr;

#[derive(Clone, Debug, PartialEq)]
enum Node {
    Const(String),
    Var(String),
    Func(String, Vec<Node>),
    Binop(String, Box<Node>, Box<Node>),
}

fn number_end(text: &str, start: usize) -> Option<usize> {
    let bytes = text.as_bytes();
    let mut i = start;
    while bytes.get(i).is_some_and(u8::is_ascii_digit) {
        i += 1;
    }
    if i == start {
        return None;
    }
    if bytes.get(i) == Some(&b'.') {
        i += 1;
        while bytes.get(i).is_some_and(u8::is_ascii_digit) {
            i += 1;
        }
    }
    if matches!(bytes.get(i), Some(b'e' | b'E')) {
        let exponent = i;
        i += 1;
        if matches!(bytes.get(i), Some(b'+' | b'-')) {
            i += 1;
        }
        let digits = i;
        while bytes.get(i).is_some_and(u8::is_ascii_digit) {
            i += 1;
        }
        if digits == i {
            i = exponent;
        }
    }
    Some(i)
}

fn is_ident_start(byte: u8) -> bool {
    byte.is_ascii_alphabetic() || byte == b'_'
}

fn is_ident_continue(byte: u8) -> bool {
    is_ident_start(byte) || byte.is_ascii_digit()
}

fn is_numeric_token(token: &str) -> bool {
    let unsigned = token.strip_prefix('-').unwrap_or(token);
    number_end(unsigned, 0) == Some(unsigned.len())
}

fn tokenize(text: &str) -> Result<Vec<String>> {
    let bytes = text.as_bytes();
    let mut out: Vec<String> = Vec::new();
    let mut i = 0;
    while i < bytes.len() {
        let ch = text[i..].chars().next().expect("valid UTF-8 boundary");
        if ch.is_whitespace() {
            i += ch.len_utf8();
            continue;
        }
        if b"(),+*/".contains(&bytes[i]) {
            out.push((bytes[i] as char).to_string());
            i += 1;
            continue;
        }
        if bytes[i] == b'-' {
            let previous = out.last().map(String::as_str);
            let unary = previous.is_none_or(|token| "(,+-*/".contains(token));
            if unary
                && i + 1 < bytes.len()
                && (bytes[i + 1].is_ascii_digit() || bytes[i + 1] == b'.')
            {
                if let Some(end) = number_end(text, i + 1) {
                    out.push(text[i..end].to_owned());
                    i = end;
                    continue;
                }
            }
            out.push("-".to_owned());
            i += 1;
            continue;
        }
        if let Some(end) = number_end(text, i) {
            out.push(text[i..end].to_owned());
            i = end;
            continue;
        }
        if is_ident_start(bytes[i]) {
            let mut end = i + 1;
            while bytes.get(end).is_some_and(|byte| is_ident_continue(*byte)) {
                end += 1;
            }
            out.push(text[i..end].to_owned());
            i = end;
            continue;
        }
        let end = text[i..]
            .char_indices()
            .nth(8)
            .map_or(text.len(), |(offset, _)| i + offset);
        bail!("cannot tokenize expr at {:?}", &text[i..end]);
    }
    Ok(out)
}

struct Parser {
    tokens: Vec<String>,
    pos: usize,
}

impl Parser {
    fn peek(&self) -> Option<&str> {
        self.tokens.get(self.pos).map(String::as_str)
    }

    fn take(&mut self, expected: Option<&str>) -> Result<String> {
        let token = self
            .peek()
            .map(str::to_owned)
            .ok_or_else(|| anyhow::anyhow!("unexpected end of expr"))?;
        if let Some(expected) = expected {
            ensure!(
                token == expected,
                "expected {:?}, got {:?}",
                expected,
                token
            );
        }
        self.pos += 1;
        Ok(token)
    }

    fn primary(&mut self) -> Result<Node> {
        let token = self.take(None)?;
        if token == "(" {
            let node = self.binary(1)?;
            self.take(Some(")"))?;
            return Ok(node);
        }
        if is_numeric_token(&token) {
            return Ok(Node::Const(token));
        }
        if token.bytes().next().is_some_and(is_ident_start) && token.bytes().all(is_ident_continue)
        {
            if self.peek() != Some("(") {
                return Ok(Node::Var(token));
            }
            self.take(Some("("))?;
            let mut args = vec![self.binary(1)?];
            while self.peek() == Some(",") {
                self.take(Some(","))?;
                args.push(self.binary(1)?);
            }
            self.take(Some(")"))?;
            if token == "col" {
                ensure!(args.len() == 1, "col() expects one integer literal");
                let Node::Const(value) = &args[0] else {
                    bail!("col() expects one integer literal");
                };
                let column: f64 = value
                    .parse()
                    .map_err(|_| anyhow::anyhow!("col() expects one integer literal"))?;
                return Ok(Node::Var(format!("col({})", column.trunc() as i64)));
            }
            return Ok(Node::Func(token, args));
        }
        bail!("unexpected expr token {:?}", token)
    }

    fn binary(&mut self, min_precedence: u8) -> Result<Node> {
        let mut left = self.primary()?;
        loop {
            let Some(operator) = self.peek() else {
                return Ok(left);
            };
            let precedence = match operator {
                "+" | "-" => 1,
                "*" | "/" => 2,
                _ => return Ok(left),
            };
            if precedence < min_precedence {
                return Ok(left);
            }
            let operator = self.take(None)?;
            let right = self.binary(precedence + 1)?;
            left = Node::Binop(operator, Box::new(left), Box::new(right));
        }
    }
}

fn parse_expr(text: &str) -> Result<Node> {
    let mut parser = Parser {
        tokens: tokenize(text)?,
        pos: 0,
    };
    let node = parser.binary(1)?;
    if parser.pos != parser.tokens.len() {
        bail!("trailing expr tokens: {:?}", &parser.tokens[parser.pos..]);
    }
    Ok(node)
}

fn snap_const(value: f64, table: &Expr) -> usize {
    let mut best = 0;
    let mut best_cost = f64::INFINITY;
    for (index, bin_value) in table.const_values.iter().copied().enumerate() {
        if value == bin_value {
            return table.const_ids[index];
        }
        let cost = if bin_value != 0.0
            && value != 0.0
            && bin_value.is_sign_positive() == value.is_sign_positive()
        {
            (value.abs() / bin_value.abs()).ln().abs()
        } else {
            (value - bin_value).abs() + 1.0
        };
        if cost < best_cost {
            best_cost = cost;
            best = index;
        }
    }
    table.const_ids[best]
}

fn encode_node(
    node: &Node,
    table: &Expr,
    output: &mut Vec<usize>,
    counters: &mut BTreeMap<String, usize>,
) -> Result<()> {
    match node {
        Node::Const(token) => {
            let value: f64 = token.parse()?;
            let id = snap_const(value, table);
            let index = table
                .const_ids
                .iter()
                .position(|candidate| *candidate == id)
                .expect("snapped id is in const_ids");
            if (table.const_values[index] - value).abs() > 1e-12 {
                *counters.entry("const_snapped".to_owned()).or_default() += 1;
            }
            output.push(id);
        }
        Node::Var(name) => {
            let found = table.symbol.iter().find_map(|(id, symbol)| {
                (symbol == name
                    && (table.var_map_ids.contains(id)
                        || table.var_common_ids.contains(id)
                        || table.var_col_ids.contains(id)))
                .then_some(*id)
            });
            output.push(found.ok_or_else(|| anyhow::anyhow!("unknown expr var {:?}", name))?);
        }
        Node::Func(name, args) => {
            let found = table.func_arity.iter().find_map(|(id, arity)| {
                (!table.is_binop.get(id).copied().unwrap_or(false)
                    && table.symbol.get(id).is_some_and(|symbol| symbol == name))
                .then_some((*id, *arity))
            });
            let Some((id, arity)) = found else {
                bail!("unknown expr func {:?}", name)
            };
            ensure!(
                arity == args.len(),
                "{} arity {} != {}",
                name,
                args.len(),
                arity
            );
            output.push(id);
            for arg in args {
                encode_node(arg, table, output, counters)?;
            }
        }
        Node::Binop(operator, left, right) => {
            let found = table.func_arity.iter().find_map(|(id, _)| {
                (table.is_binop.get(id).copied().unwrap_or(false)
                    && table
                        .symbol
                        .get(id)
                        .is_some_and(|symbol| symbol == operator))
                .then_some(*id)
            });
            let Some(id) = found else {
                bail!("unknown expr binop {:?}", operator)
            };
            output.push(id);
            encode_node(left, table, output, counters)?;
            encode_node(right, table, output, counters)?;
        }
    }
    Ok(())
}

/// Encode an infix expression into prefix token IDs, excluding `EXPR_END`.
pub fn encode_expr(
    text: &str,
    table: &Expr,
    counters: Option<&mut BTreeMap<String, usize>>,
) -> Result<Vec<usize>> {
    let mut local = BTreeMap::new();
    let mut output = Vec::new();
    encode_node(
        &parse_expr(text)?,
        table,
        &mut output,
        counters.unwrap_or(&mut local),
    )?;
    Ok(output)
}

fn render_read(ids: &[usize], pos: &mut usize, table: &Expr) -> Result<String> {
    let id = *ids
        .get(*pos)
        .ok_or_else(|| anyhow::anyhow!("unexpected end of expr render"))?;
    *pos += 1;
    let symbol = table
        .symbol
        .get(&id)
        .ok_or_else(|| anyhow::anyhow!("unknown expr token {}", id))?;
    let Some(arity) = table.func_arity.get(&id).copied() else {
        return Ok(symbol.clone());
    };
    let mut args = Vec::with_capacity(arity);
    for _ in 0..arity {
        args.push(render_read(ids, pos, table)?);
    }
    if table.is_binop.get(&id).copied().unwrap_or(false) {
        ensure!(
            args.len() == 2,
            "binop token {} arity {} != 2",
            id,
            args.len()
        );
        Ok(format!("({} {} {})", args[0], symbol, args[1]))
    } else {
        Ok(format!("{}({})", symbol, args.join(", ")))
    }
}

/// Render one prefix expression into deterministic infix text.
pub fn render_expr(ids: &[usize], table: &Expr) -> Result<String> {
    let mut pos = 0;
    let text = render_read(ids, &mut pos, table)?;
    ensure!(pos == ids.len(), "trailing expr tokens in render");
    Ok(text)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn table() -> Expr {
        Expr {
            var_map_ids: vec![10],
            var_col_ids: vec![11],
            var_common_ids: vec![],
            const_ids: vec![1, 2, 3],
            const_values: vec![-1.0, 0.0, 2.0],
            func_arity: [(20, 2), (21, 2), (22, 1)].into_iter().collect(),
            end_id: 99,
            symbol: [
                (1, "-1"),
                (2, "0"),
                (3, "2"),
                (10, "x"),
                (11, "col(2)"),
                (20, "+"),
                (21, "*"),
                (22, "sin"),
            ]
            .into_iter()
            .map(|(id, symbol)| (id, symbol.to_owned()))
            .collect(),
            is_binop: [(20, true), (21, true), (22, false)].into_iter().collect(),
            max_depth: 32,
            max_nodes: 128,
        }
    }

    #[test]
    fn parses_precedence_and_roundtrips_prefix() {
        let table = table();
        let tokens = encode_expr("x + 2 * sin(-1)", &table, None).unwrap();
        assert_eq!(tokens, vec![20, 10, 21, 3, 22, 1]);
        assert_eq!(render_expr(&tokens, &table).unwrap(), "(x + (2 * sin(-1)))");
    }

    #[test]
    fn column_literal_truncates_toward_zero() {
        assert_eq!(parse_expr("col(3.9)").unwrap(), Node::Var("col(3)".into()));
        assert_eq!(
            parse_expr("col(-2.9)").unwrap(),
            Node::Var("col(-2)".into())
        );
    }

    #[test]
    fn snaps_constant_and_counts_it() {
        let table = table();
        let mut counters = BTreeMap::new();
        assert_eq!(
            encode_expr("1", &table, Some(&mut counters)).unwrap(),
            vec![3]
        );
        assert_eq!(counters.get("const_snapped"), Some(&1));
        assert_eq!(encode_expr("2", &table, None).unwrap(), vec![3]);
    }

    #[test]
    fn rejects_invalid_columns_and_trailing_prefix_ids() {
        assert!(parse_expr("col(x)").is_err());
        assert!(render_expr(&[2, 3], &table()).is_err());
    }
}
