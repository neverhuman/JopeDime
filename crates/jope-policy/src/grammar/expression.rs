use super::*;

pub(super) fn render_expr(tokens: &[usize], table: &Expr, pos: &mut usize) -> Result<String> {
    let id = *tokens.get(*pos).context("expression token absent")?;
    *pos += 1;
    let symbol = table.symbol.get(&id).context("unknown expression symbol")?;
    if let Some(&arity) = table.func_arity.get(&id) {
        let args = (0..arity)
            .map(|_| render_expr(tokens, table, pos))
            .collect::<Result<Vec<_>>>()?;
        Ok(if table.is_binop.get(&id) == Some(&true) {
            format!("({} {symbol} {})", args[0], args[1])
        } else {
            format!("{symbol}({})", args.join(", "))
        })
    } else {
        Ok(symbol.clone())
    }
}

#[derive(Clone)]
enum Lexeme {
    Name(String),
    Number(f64),
    Symbol(char),
}

fn lex_expr(text: &str) -> Result<Vec<Lexeme>> {
    let chars = text.chars().collect::<Vec<_>>();
    let mut lexical = Vec::new();
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        if c.is_whitespace() {
            i += 1;
            continue;
        }
        if c.is_ascii_digit() || c == '.' {
            let start = i;
            i += 1;
            while i < chars.len()
                && (chars[i].is_ascii_digit()
                    || matches!(chars[i], '.' | 'e' | 'E')
                    || matches!(chars[i], '+' | '-') && matches!(chars[i - 1], 'e' | 'E'))
            {
                i += 1;
            }
            lexical.push(Lexeme::Number(
                chars[start..i].iter().collect::<String>().parse()?,
            ));
        } else if c.is_ascii_alphabetic() || c == '_' {
            let start = i;
            i += 1;
            while i < chars.len() && (chars[i].is_ascii_alphanumeric() || chars[i] == '_') {
                i += 1;
            }
            lexical.push(Lexeme::Name(chars[start..i].iter().collect()));
        } else {
            ensure!("()+-*/,".contains(c), "unsupported expression character");
            lexical.push(Lexeme::Symbol(c));
            i += 1;
        }
    }
    Ok(lexical)
}

struct ExprParser<'a> {
    tokens: &'a [Lexeme],
    at: usize,
    table: &'a Expr,
}

impl ExprParser<'_> {
    fn symbol(&self, s: char) -> bool {
        matches!(self.tokens.get(self.at), Some(Lexeme::Symbol(c)) if *c == s)
    }

    fn need(&mut self, s: char) -> Result<()> {
        ensure!(self.symbol(s), "expected {s} in expression");
        self.at += 1;
        Ok(())
    }

    fn number(&self, n: f64) -> Result<usize> {
        self.table
            .const_values
            .iter()
            .position(|v| (*v - n).abs() < 1e-12)
            .map(|i| self.table.const_ids[i])
            .context("expression constant outside frozen vocabulary")
    }

    fn id(&self, s: &str, arity: Option<usize>) -> Result<usize> {
        self.table
            .symbol
            .iter()
            .find(|(id, sym)| sym.as_str() == s && self.table.func_arity.get(id).copied() == arity)
            .map(|(&id, _)| id)
            .context("expression symbol outside frozen vocabulary")
    }

    fn read(&mut self, min: u8) -> Result<Vec<usize>> {
        let token = self
            .tokens
            .get(self.at)
            .cloned()
            .context("expression ended early")?;
        self.at += 1;
        let mut left = match token {
            Lexeme::Number(n) => vec![self.number(n)?],
            Lexeme::Symbol('(') => {
                let x = self.read(0)?;
                self.need(')')?;
                x
            }
            Lexeme::Symbol('-') => {
                if let Some(Lexeme::Number(n)) = self.tokens.get(self.at) {
                    let n = *n;
                    self.at += 1;
                    vec![self.number(-n)?]
                } else {
                    let mut out = vec![self.id("-", Some(2))?, self.number(0.)?];
                    out.extend(self.read(3)?);
                    out
                }
            }
            Lexeme::Name(name) => {
                if self.symbol('(') {
                    self.at += 1;
                    if name == "col" {
                        let n = if let Some(Lexeme::Number(n)) = self.tokens.get(self.at) {
                            *n
                        } else {
                            bail!("column index missing")
                        };
                        self.at += 1;
                        self.need(')')?;
                        vec![self.id(&format!("col({n})"), None)?]
                    } else {
                        let mut args = vec![self.read(0)?];
                        while self.symbol(',') {
                            self.at += 1;
                            args.push(self.read(0)?);
                        }
                        self.need(')')?;
                        let mut out = vec![self.id(&name, Some(args.len()))?];
                        for arg in args {
                            out.extend(arg);
                        }
                        out
                    }
                } else {
                    vec![self.id(&name, None)?]
                }
            }
            _ => bail!("invalid expression prefix"),
        };
        while let Some(Lexeme::Symbol(op @ ('+' | '-' | '*' | '/'))) = self.tokens.get(self.at) {
            let op = *op;
            let prec = if matches!(op, '+' | '-') { 1 } else { 2 };
            if prec < min {
                break;
            }
            self.at += 1;
            let right = self.read(prec + 1)?;
            let mut out = vec![self.id(&op.to_string(), Some(2))?];
            out.extend(left);
            out.extend(right);
            left = out;
        }
        Ok(left)
    }
}

pub(super) fn parse_expr_tokens(text: &str, table: &Expr) -> Result<Vec<usize>> {
    let lexical = lex_expr(text)?;
    let mut parser = ExprParser {
        tokens: &lexical,
        at: 0,
        table,
    };
    let ids = parser.read(0)?;
    ensure!(parser.at == lexical.len(), "trailing expression text");
    Ok(ids)
}
