use super::codec::{identity, Reader, Writer, MAX_BYTES};
use super::{Error, Result};
use crate::dsl::{DslExpr, ParamValue, Stage};
use crate::genome::{GenomeIr, PortfolioIr};
use crate::math_expr::MathExpr;

const MAGIC: &[u8] = b"FMKPRG\0\x01";

/// Legacy import copies typed values and the executable AST directly. It
/// never reparses the historical, deliberately frozen lossy printer.
#[derive(Clone, Debug)]
pub enum Program {
    LegacyV1(GenomeIr),
}

impl Program {
    pub fn import_legacy(genome: &GenomeIr) -> Self {
        Self::LegacyV1(genome.clone())
    }

    pub fn parse_legacy(source: &str) -> Result<Self> {
        GenomeIr::parse(source)
            .map(Self::LegacyV1)
            .map_err(|e| Error::Syntax(e.to_string()))
    }

    pub fn legacy_identity_text(&self) -> String {
        match self {
            Self::LegacyV1(g) => g.canonical(),
        }
    }

    pub fn to_bytes(&self) -> Result<Vec<u8>> {
        let mut w = Writer::default();
        w.raw(MAGIC)?;
        match self {
            Self::LegacyV1(g) => {
                w.byte(0)?;
                write_genome(&mut w, g)?;
            }
        }
        Ok(w.finish())
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        let mut r = Reader::new(bytes)?;
        if r.take(MAGIC.len())? != MAGIC {
            return Err(Error::Syntax("program codec version".into()));
        }
        let result = match r.byte()? {
            0 => Self::LegacyV1(read_genome(&mut r)?),
            _ => return Err(Error::Unsupported("program version".into())),
        };
        r.finish()?;
        Ok(result)
    }

    pub fn identity(&self) -> Result<String> {
        Ok(identity("program", &self.to_bytes()?))
    }

    /// Lossless wire text. The historical DSL printer remains available via
    /// legacy_identity_text; this tagged form also preserves NaN payloads.
    pub fn to_lossless_text(&self) -> Result<String> {
        Ok(format!(
            "(program codec=1 hex=\"{}\")",
            self.to_bytes()?
                .iter()
                .map(|b| format!("{b:02x}"))
                .collect::<String>()
        ))
    }

    pub fn parse_lossless_text(text: &str) -> Result<Self> {
        let hex = text
            .strip_prefix("(program codec=1 hex=\"")
            .and_then(|v| v.strip_suffix("\")"))
            .ok_or_else(|| Error::Syntax("lossless program envelope".into()))?;
        if hex.len() > 2 * MAX_BYTES || hex.len() % 2 != 0 {
            return Err(Error::Resource("program text length".into()));
        }
        let bytes = hex
            .as_bytes()
            .chunks_exact(2)
            .map(|p| {
                let h = (p[0] as char).to_digit(16);
                let l = (p[1] as char).to_digit(16);
                h.zip(l)
                    .map(|(h, l)| ((h << 4) | l) as u8)
                    .ok_or_else(|| Error::Syntax("hex digit".into()))
            })
            .collect::<Result<Vec<_>>>()?;
        Self::from_bytes(&bytes)
    }
}

fn write_stage(w: &mut Writer, s: &Stage) -> Result<()> {
    w.text(&s.op)?;
    w.word(s.params.len() as u64)?;
    for (key, value) in &s.params {
        w.text(key)?;
        match value {
            ParamValue::Int(v) => {
                w.byte(0)?;
                w.word(*v as u64)?;
            }
            ParamValue::Float(v) => {
                w.byte(1)?;
                w.word(v.to_bits())?;
            }
            ParamValue::Bool(v) => {
                w.byte(2)?;
                w.byte(u8::from(*v))?;
            }
            ParamValue::Str(v) => {
                w.byte(3)?;
                w.text(v)?;
            }
        }
    }
    Ok(())
}
fn read_stage(r: &mut Reader<'_>) -> Result<Stage> {
    let mut s = Stage::new(r.text()?);
    let n = r.length()?;
    let mut previous = None;
    for _ in 0..n {
        let key = r.text()?;
        if previous.as_ref().is_some_and(|prev: &String| prev >= &key) {
            return Err(Error::Syntax("duplicate or unordered parameter".into()));
        }
        previous = Some(key.clone());
        let value = match r.byte()? {
            0 => ParamValue::Int(r.word()? as i64),
            1 => ParamValue::Float(f64::from_bits(r.word()?)),
            2 => ParamValue::Bool(read_bool(r)?),
            3 => ParamValue::Str(r.text()?),
            _ => return Err(Error::Syntax("parameter tag".into())),
        };
        s.params.insert(key, value);
    }
    Ok(s)
}
fn write_pipe(w: &mut Writer, pipe: &DslExpr) -> Result<()> {
    w.word(pipe.stages.len() as u64)?;
    for s in &pipe.stages {
        write_stage(w, s)?;
    }
    Ok(())
}
fn read_pipe(r: &mut Reader<'_>) -> Result<DslExpr> {
    let n = r.length()?;
    Ok(DslExpr {
        stages: (0..n).map(|_| read_stage(r)).collect::<Result<_>>()?,
    })
}
fn read_bool(r: &mut Reader<'_>) -> Result<bool> {
    match r.byte()? {
        0 => Ok(false),
        1 => Ok(true),
        _ => Err(Error::Syntax("boolean tag".into())),
    }
}
fn write_genome(w: &mut Writer, g: &GenomeIr) -> Result<()> {
    match g {
        GenomeIr::Pipeline(pipe) => {
            w.byte(0)?;
            write_pipe(w, pipe)?;
        }
        GenomeIr::Symbolic {
            source,
            expr,
            affine,
        } => {
            w.byte(1)?;
            w.byte(u8::from(*affine))?;
            w.text(source)?;
            expr.write_exact(w)?;
        }
        GenomeIr::Portfolio(p) => {
            w.byte(2)?;
            w.word(p.folds as u64)?;
            w.byte(u8::from(p.repeats_auto))?;
            w.word(p.members.len() as u64)?;
            for m in &p.members {
                write_pipe(w, m)?;
            }
        }
    }
    Ok(())
}
fn read_genome(r: &mut Reader<'_>) -> Result<GenomeIr> {
    match r.byte()? {
        0 => Ok(GenomeIr::Pipeline(read_pipe(r)?)),
        1 => {
            let affine = read_bool(r)?;
            let source = r.text()?;
            let expr = MathExpr::read_exact(r)?;
            Ok(GenomeIr::Symbolic {
                source,
                expr,
                affine,
            })
        }
        2 => {
            let folds =
                usize::try_from(r.word()?).map_err(|_| Error::Bounds("portfolio folds".into()))?;
            let repeats_auto = read_bool(r)?;
            let n = r.length()?;
            if n == 0 || n > 12 || folds != 5 {
                return Err(Error::Bounds("legacy portfolio contract".into()));
            }
            Ok(GenomeIr::Portfolio(PortfolioIr {
                folds,
                repeats_auto,
                members: (0..n).map(|_| read_pipe(r)).collect::<Result<_>>()?,
            }))
        }
        _ => Err(Error::Syntax("legacy program tag".into())),
    }
}
