//! Genome flavors accepted by the battle runtime, the transpiler, and the judge.

use crate::dsl::{parse_dsl, parse_portfolio, DslExpr, ParamValue, Stage};
use crate::math_expr::MathExpr;
use anyhow::{bail, Result};

/// Blend parameters of the explicit nested portfolio form. v1 pins the whole
/// tuple (kind=nnls, folds=5, guard=clip); only `repeats` carries a choice.
#[derive(Clone, Debug)]
pub struct PortfolioIr {
    /// Member pipelines, order-preserving (canonicalization keeps the order).
    pub members: Vec<DslExpr>,
    pub folds: usize,
    /// true = `repeats=auto` (n-adaptive), false = `repeats=1`.
    pub repeats_auto: bool,
}

#[derive(Clone, Debug)]
pub enum GenomeIr {
    /// A staged feature pipeline ending in a model head:
    /// `(pipe X (robust_scale) (rff_cos n=32 seed=9) (ridge))`
    Pipeline(DslExpr),
    /// A single symbolic expression; with `affine`, predictions are `a*f(x)+b`
    /// with a,b fit by least squares on train (Keijzer linear scaling).
    Symbolic {
        source: String,
        expr: MathExpr,
        affine: bool,
    },
    /// The explicit nested portfolio form (spec §2):
    /// `(portfolio (member (pipe X ...))+ (blend kind=nnls folds=5 repeats=auto guard=clip))`.
    /// Compact portfolios (`(pipe X ... (portfolio preset=core ...))`) stay
    /// ordinary pipelines — the two forms canonicalize separately by design.
    Portfolio(PortfolioIr),
}

/// Validate the nested-form blend stage; only `repeats` is a real choice.
fn parse_blend(stage: &Stage) -> Result<(usize, bool)> {
    for key in stage.params.keys() {
        if !matches!(key.as_str(), "kind" | "folds" | "repeats" | "guard") {
            bail!("blend: unknown parameter '{key}'");
        }
    }
    match stage.params.get("kind") {
        None => {}
        Some(ParamValue::Str(s)) if s == "nnls" => {}
        Some(ParamValue::Str(s)) => {
            bail!("blend: kind must be nnls (got '{s}'; caruana is not implemented in v1)")
        }
        Some(_) => bail!("blend: kind must be nnls"),
    }
    let folds = match stage.params.get("folds") {
        None => 5,
        Some(v) if v.as_usize() == Some(5) => 5,
        Some(_) => bail!("blend: folds must be 5"),
    };
    let repeats_auto = match stage.params.get("repeats") {
        None => true,
        Some(ParamValue::Str(s)) if s == "auto" => true,
        Some(v) if v.as_usize() == Some(1) => false,
        Some(_) => bail!("blend: repeats must be auto or 1"),
    };
    match stage.params.get("guard") {
        None => {}
        Some(ParamValue::Str(s)) if s == "clip" => {}
        Some(_) => bail!("blend: guard must be clip"),
    }
    Ok((folds, repeats_auto))
}

impl GenomeIr {
    /// Parse from the claim wire form. Pipelines are the `(pipe X ...)` DSL;
    /// symbolic genomes use the prefix `(sym <affine|raw> "<expr>")`; the
    /// explicit portfolio form is `(portfolio (member ...)+ (blend ...))`.
    pub fn parse(src: &str) -> Result<GenomeIr> {
        let trimmed = src.trim();
        if trimmed.starts_with("(pipe") {
            return Ok(GenomeIr::Pipeline(parse_dsl(trimmed)?));
        }
        if trimmed.starts_with("(portfolio") {
            let (members, blend) = parse_portfolio(trimmed)?;
            let (folds, repeats_auto) = parse_blend(&blend)?;
            return Ok(GenomeIr::Portfolio(PortfolioIr {
                members,
                folds,
                repeats_auto,
            }));
        }
        if let Some(rest) = trimmed.strip_prefix("(sym ") {
            let Some(rest) = rest.strip_suffix(')') else {
                bail!("symbolic genome requires a closing delimiter");
            };
            let rest = rest.trim();
            let (mode, expr_src) = match rest.split_once(' ') {
                Some((m, e)) => (m, e.trim()),
                None => bail!("symbolic genome must be (sym <affine|raw> \"expr\")"),
            };
            if !matches!(mode, "affine" | "raw") {
                bail!("symbolic mode must be affine or raw");
            }
            let Some(expr_src) = expr_src.strip_prefix('"').and_then(|s| s.strip_suffix('"'))
            else {
                bail!("symbolic expression must have exactly one pair of quotes");
            };
            if expr_src.contains('"') {
                bail!("unexpected quote in symbolic expression");
            }
            let expr = MathExpr::parse(expr_src)?;
            return Ok(GenomeIr::Symbolic {
                source: expr_src.to_string(),
                expr,
                affine: mode == "affine",
            });
        }
        bail!("unrecognized genome form: {trimmed}");
    }

    pub fn canonical(&self) -> String {
        match self {
            GenomeIr::Pipeline(e) => e.to_string(),
            GenomeIr::Symbolic { source, affine, .. } => {
                format!(
                    "(sym {} \"{}\")",
                    if *affine { "affine" } else { "raw" },
                    source
                )
            }
            GenomeIr::Portfolio(p) => {
                let mut out = String::from("(portfolio");
                for m in &p.members {
                    out.push_str(" (member ");
                    out.push_str(&m.to_string());
                    out.push(')');
                }
                out.push_str(&format!(
                    " (blend folds={} guard=clip kind=nnls repeats={}))",
                    p.folds,
                    if p.repeats_auto { "auto" } else { "1" }
                ));
                out
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn symbolic_wire_form_rejects_unknown_modes_and_incomplete_delimiters() {
        for source in [
            "(sym bogus \"col(0)\")",
            "(sym affine \"col(0)\"",
            "(sym raw col(0))",
            "(sym raw \"col(0))",
            "(sym raw col(0)\")",
            "(sym raw \"\"col(0)\"\")",
            "(sym raw \"col(0)\") trailing",
        ] {
            assert!(GenomeIr::parse(source).is_err(), "accepted {source}");
        }
        for mode in ["raw", "affine"] {
            let source = format!("(sym {mode} \"col(0)\")");
            assert_eq!(GenomeIr::parse(&source).unwrap().canonical(), source);
        }
    }

    #[test]
    fn portfolio_form_parses_and_canonicalizes_order_preserving() {
        let src = "(portfolio (member (pipe X (ridge_gcv))) \
                   (member (pipe X (blom) (gpr kernel=rbf+linear steps=40))) \
                   (blend kind=nnls folds=5 repeats=auto guard=clip))";
        let g = GenomeIr::parse(src).unwrap();
        let GenomeIr::Portfolio(p) = &g else {
            panic!("expected portfolio")
        };
        assert_eq!(p.members.len(), 2);
        assert_eq!(p.folds, 5);
        assert!(p.repeats_auto);
        let canon = g.canonical();
        assert_eq!(
            canon,
            "(portfolio (member (pipe X (ridge_gcv))) \
             (member (pipe X (blom) (gpr kernel=rbf+linear steps=40))) \
             (blend folds=5 guard=clip kind=nnls repeats=auto))"
        );
        // Canonical form re-parses to the same canonical string.
        assert_eq!(GenomeIr::parse(&canon).unwrap().canonical(), canon);
        // Member order is preserved, not sorted.
        let flipped = "(portfolio (member (pipe X (blom) (gpr kernel=rbf+linear steps=40))) \
                       (member (pipe X (ridge_gcv))) (blend))";
        assert_ne!(GenomeIr::parse(flipped).unwrap().canonical(), canon);
    }

    #[test]
    fn portfolio_form_rejects_bad_shapes() {
        // Zero members.
        assert!(GenomeIr::parse("(portfolio (blend))").is_err());
        // Thirteen members (cap is 12).
        let mut src = String::from("(portfolio");
        for _ in 0..13 {
            src.push_str(" (member (pipe X (ridge)))");
        }
        src.push_str(" (blend))");
        assert!(GenomeIr::parse(&src).is_err());
        // Nested portfolio member.
        assert!(
            GenomeIr::parse("(portfolio (member (pipe X (portfolio preset=core))) (blend))")
                .is_err()
        );
        // Bad blend parameters.
        for bad in [
            "(portfolio (member (pipe X (ridge))) (blend kind=caruana))",
            "(portfolio (member (pipe X (ridge))) (blend folds=3))",
            "(portfolio (member (pipe X (ridge))) (blend repeats=2))",
            "(portfolio (member (pipe X (ridge))) (blend guard=none))",
            "(portfolio (member (pipe X (ridge))) (blend bogus=1))",
        ] {
            assert!(GenomeIr::parse(bad).is_err(), "{bad}");
        }
        // Members after the blend stage.
        assert!(GenomeIr::parse("(portfolio (blend) (member (pipe X (ridge))))").is_err());
        // repeats=1 accepted.
        let g = GenomeIr::parse("(portfolio (member (pipe X (ridge))) (blend repeats=1))").unwrap();
        let GenomeIr::Portfolio(p) = g else { panic!() };
        assert!(!p.repeats_auto);
    }
}
