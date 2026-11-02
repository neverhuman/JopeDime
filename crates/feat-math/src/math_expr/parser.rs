use super::*;

fn lex(src: &str) -> Result<Vec<Tok>> {
    let mut toks = Vec::new();
    let mut chars = src.char_indices().peekable();
    while let Some((idx, ch)) = chars.peek().copied() {
        match ch {
            c if c.is_whitespace() => {
                chars.next();
            }
            '+' => {
                toks.push(Tok::Plus);
                chars.next();
            }
            '-' => {
                toks.push(Tok::Minus);
                chars.next();
            }
            '*' => {
                toks.push(Tok::Star);
                chars.next();
            }
            '/' => {
                toks.push(Tok::Slash);
                chars.next();
            }
            '^' => {
                toks.push(Tok::Caret);
                chars.next();
            }
            '(' => {
                toks.push(Tok::LParen);
                chars.next();
            }
            ')' => {
                toks.push(Tok::RParen);
                chars.next();
            }
            ',' => {
                toks.push(Tok::Comma);
                chars.next();
            }
            c if c.is_ascii_digit() || c == '.' => {
                let start = idx;
                let mut end = idx + ch.len_utf8();
                chars.next();
                while let Some((j, c)) = chars.peek().copied() {
                    if c.is_ascii_digit()
                        || c == '.'
                        || c == 'e'
                        || c == 'E'
                        || ((c == '+' || c == '-') && j > start && exponent_sign_allowed(src, j))
                    {
                        end = j + c.len_utf8();
                        chars.next();
                    } else {
                        break;
                    }
                }
                let raw = &src[start..end];
                let value = raw
                    .parse::<f64>()
                    .with_context(|| format!("invalid numeric literal '{raw}'"))?;
                if !value.is_finite() {
                    bail!("non-finite numeric literal '{raw}'");
                }
                toks.push(Tok::Num(value));
            }
            c if c.is_ascii_alphabetic() || c == '_' => {
                let start = idx;
                let mut end = idx + ch.len_utf8();
                chars.next();
                while let Some((j, c)) = chars.peek().copied() {
                    if c.is_ascii_alphanumeric() || c == '_' {
                        end = j + c.len_utf8();
                        chars.next();
                    } else {
                        break;
                    }
                }
                toks.push(Tok::Ident(src[start..end].to_ascii_lowercase()));
            }
            _ => bail!("unexpected character '{ch}' in math expression"),
        }
    }
    toks.push(Tok::End);
    Ok(toks)
}

fn exponent_sign_allowed(src: &str, idx: usize) -> bool {
    src[..idx]
        .chars()
        .last()
        .map(|c| c == 'e' || c == 'E')
        .unwrap_or(false)
}

struct Parser<'a> {
    toks: &'a [Tok],
    pos: usize,
}

impl<'a> Parser<'a> {
    fn parse_expr(&mut self) -> Result<Expr> {
        self.parse_add()
    }

    fn parse_add(&mut self) -> Result<Expr> {
        let mut expr = self.parse_mul()?;
        loop {
            match self.peek() {
                Tok::Plus => {
                    self.pos += 1;
                    expr = Expr::Binary(BinOp::Add, Box::new(expr), Box::new(self.parse_mul()?));
                }
                Tok::Minus => {
                    self.pos += 1;
                    expr = Expr::Binary(BinOp::Sub, Box::new(expr), Box::new(self.parse_mul()?));
                }
                _ => return Ok(expr),
            }
        }
    }

    fn parse_mul(&mut self) -> Result<Expr> {
        let mut expr = self.parse_pow()?;
        loop {
            match self.peek() {
                Tok::Star => {
                    self.pos += 1;
                    expr = Expr::Binary(BinOp::Mul, Box::new(expr), Box::new(self.parse_pow()?));
                }
                Tok::Slash => {
                    self.pos += 1;
                    expr = Expr::Binary(BinOp::Div, Box::new(expr), Box::new(self.parse_pow()?));
                }
                _ => return Ok(expr),
            }
        }
    }

    fn parse_pow(&mut self) -> Result<Expr> {
        let expr = self.parse_unary()?;
        if matches!(self.peek(), Tok::Caret) {
            self.pos += 1;
            Ok(Expr::Binary(
                BinOp::Pow,
                Box::new(expr),
                Box::new(self.parse_pow()?),
            ))
        } else {
            Ok(expr)
        }
    }

    fn parse_unary(&mut self) -> Result<Expr> {
        match self.peek() {
            Tok::Plus => {
                self.pos += 1;
                self.parse_unary()
            }
            Tok::Minus => {
                self.pos += 1;
                Ok(Expr::Unary(UnOp::Neg, Box::new(self.parse_unary()?)))
            }
            _ => self.parse_primary(),
        }
    }

    fn parse_primary(&mut self) -> Result<Expr> {
        match self.peek().clone() {
            Tok::Num(v) => {
                self.pos += 1;
                Ok(Expr::Const(v))
            }
            Tok::Ident(name) => {
                self.pos += 1;
                if matches!(self.peek(), Tok::LParen) {
                    self.pos += 1;
                    let args = self.parse_args()?;
                    if name == "col" {
                        return parse_col_call(&args);
                    }
                    let func = parse_func(&name)?;
                    validate_arity(func, args.len())?;
                    Ok(Expr::Call(func, args))
                } else {
                    Ok(Expr::Var(parse_var(&name)?))
                }
            }
            Tok::LParen => {
                self.pos += 1;
                let expr = self.parse_expr()?;
                self.expect_rparen()?;
                Ok(expr)
            }
            other => bail!("expected expression, got {:?}", other),
        }
    }

    fn parse_args(&mut self) -> Result<Vec<Expr>> {
        let mut args = Vec::new();
        if matches!(self.peek(), Tok::RParen) {
            self.pos += 1;
            return Ok(args);
        }
        loop {
            args.push(self.parse_expr()?);
            match self.peek() {
                Tok::Comma => self.pos += 1,
                Tok::RParen => {
                    self.pos += 1;
                    return Ok(args);
                }
                other => bail!("expected ',' or ')', got {:?}", other),
            }
        }
    }

    fn expect_rparen(&mut self) -> Result<()> {
        match self.peek() {
            Tok::RParen => {
                self.pos += 1;
                Ok(())
            }
            other => bail!("expected ')', got {:?}", other),
        }
    }

    fn expect_end(&self) -> Result<()> {
        if matches!(self.peek(), Tok::End) {
            Ok(())
        } else {
            bail!("trailing tokens in math expression")
        }
    }

    fn peek(&self) -> &Tok {
        self.toks.get(self.pos).unwrap_or(&Tok::End)
    }
}

fn parse_col_call(args: &[Expr]) -> Result<Expr> {
    if args.len() != 1 {
        bail!("col() expects exactly one argument");
    }
    match args[0] {
        Expr::Const(v) if v >= 0.0 && (v - v.round()).abs() < 1e-12 => {
            Ok(Expr::Var(Var::Col(v as usize)))
        }
        _ => bail!("col() expects a non-negative integer literal"),
    }
}

fn parse_var(name: &str) -> Result<Var> {
    match name {
        "x" => Ok(Var::X),
        "row_mean" | "mean" => Ok(Var::RowMean),
        "row_sum" => Ok(Var::RowSum),
        "row_l1" => Ok(Var::RowL1),
        "row_l2" | "row_norm" => Ok(Var::RowL2),
        "row_min" => Ok(Var::RowMin),
        "row_max" => Ok(Var::RowMax),
        "row_std" => Ok(Var::RowStd),
        _ if name.starts_with('c') && name[1..].chars().all(|c| c.is_ascii_digit()) => {
            Ok(Var::Col(name[1..].parse()?))
        }
        _ if name.starts_with("col") && name[3..].chars().all(|c| c.is_ascii_digit()) => {
            Ok(Var::Col(name[3..].parse()?))
        }
        _ => bail!("unknown math variable '{name}'"),
    }
}

fn parse_func(name: &str) -> Result<Func> {
    match name {
        "abs" => Ok(Func::Abs),
        "sqrt" => Ok(Func::Sqrt),
        "log1p" => Ok(Func::Log1p),
        "sin" => Ok(Func::Sin),
        "cos" => Ok(Func::Cos),
        "tanh" => Ok(Func::Tanh),
        "sigmoid" => Ok(Func::Sigmoid),
        "min" => Ok(Func::Min),
        "max" => Ok(Func::Max),
        "clip" => Ok(Func::Clip),
        // Hard indicator: step(a, c) = 1 when a > c else 0 (an honest,
        // finite-valued threshold, replacing the clip(...*1e15,...) idiom).
        "step" => Ok(Func::Step),
        "exp" => Ok(Func::Exp),
        "log" => Ok(Func::Log),
        "asinh" => Ok(Func::Asinh),
        "atan" => Ok(Func::Atan),
        "erf" => Ok(Func::Erf),
        "sign" => Ok(Func::Sign),
        "floor" => Ok(Func::Floor),
        "round" => Ok(Func::Round),
        _ => bail!("unsupported math function '{name}'"),
    }
}

pub(crate) fn validate_arity(func: Func, len: usize) -> Result<()> {
    let ok = match func {
        Func::Abs
        | Func::Sqrt
        | Func::Log1p
        | Func::Sin
        | Func::Cos
        | Func::Tanh
        | Func::Sigmoid
        | Func::Exp
        | Func::Log
        | Func::Asinh
        | Func::Atan
        | Func::Erf
        | Func::Sign
        | Func::Floor
        | Func::Round => len == 1,
        Func::Min | Func::Max | Func::Step => len == 2,
        Func::Clip => len == 3,
    };
    if ok {
        Ok(())
    } else {
        bail!("{func:?} received {len} arguments")
    }
}

impl MathExpr {
    pub fn parse(src: &str) -> Result<Self> {
        let toks = lex(src)?;
        let mut parser = Parser {
            toks: &toks,
            pos: 0,
        };
        let root = parser.parse_expr()?;
        parser.expect_end()?;
        let nodes = count_nodes(&root);
        let depth = depth(&root);
        if nodes > MAX_NODES {
            bail!("math expression has {nodes} nodes, max is {MAX_NODES}");
        }
        if depth > MAX_DEPTH {
            bail!("math expression depth {depth} exceeds max {MAX_DEPTH}");
        }
        Ok(MathExpr { root })
    }
}
