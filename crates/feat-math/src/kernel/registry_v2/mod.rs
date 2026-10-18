//! Opt-in registry projections. Frozen dialect metadata is inherited through its
//! existing APIs; only the new profile policy comes from operators.toml.

use crate::dsl::{DslExpr, ParamValue, Stage};
use crate::registry::{self, ParameterKind, ParameterValue};
use crate::{GenomeIr, Pcg32};
use serde::Serialize;
use sha2::{Digest, Sha256};

mod generated;
pub use generated::*;

pub const SOURCE: &str = include_str!("../../../registry/operators.toml");

#[derive(Clone, Copy, Debug, Serialize)]
pub struct Entry {
    pub dialect: &'static str,
    pub name: &'static str,
    pub inherit: &'static str,
    /// Historical wire ID, or a namespaced source identity. Never a new codec ID.
    pub identity: &'static str,
    pub axis: &'static str,
    pub admission: &'static str,
    pub weight: u32,
    pub targets: &'static [&'static str],
    pub reject: &'static [&'static str],
    /// parameter:selector:active-value; only new-profile validation uses this.
    pub inactive: &'static [&'static str],
    /// Supplemental parameters: name:kind:default:min:max.
    pub parameters: &'static [&'static str],
    pub effects: &'static [&'static str],
}

#[derive(Clone, Debug, Serialize)]
pub struct Facts {
    pub entry: &'static Entry,
    pub canonical: &'static str,
    pub wire_id: Option<u16>,
    pub semantic_version: String,
    pub role: &'static str,
    pub parameters: serde_json::Value,
    /// None means the inherited descriptor does not declare this fact.
    pub fitted: Option<bool>,
    pub requires_labels: bool,
    pub width: String,
    pub provenance: String,
}

pub fn source_sha256() -> String {
    format!("{:x}", Sha256::digest(SOURCE.as_bytes()))
}

pub fn operator(dialect: &str, name: &str) -> Option<&'static Entry> {
    OPERATORS
        .iter()
        .find(|e| e.dialect == dialect && e.name == name)
}

/// Check the generated policy against the actual inherited authorities. A data
/// edit cannot promote a quarantined/compatibility spelling or invent a wire ID.
pub fn validate_inheritance() -> Result<(), String> {
    if OPERATORS
        .iter()
        .filter(|e| e.inherit == "registry_v1")
        .count()
        != registry::operator_registry().len()
    {
        return Err("incomplete inherited wire registry".into());
    }
    for entry in OPERATORS {
        entry.check_inheritance()?;
    }
    Ok(())
}

impl Entry {
    pub fn check_inheritance(&self) -> Result<(), String> {
        if self.inherit == "advanced_v3" {
            return self.check_advanced_inheritance();
        }
        if self.inherit == "registry_v1" {
            let row = registry::operator(self.name)
                .ok_or_else(|| format!("{}: unknown inherited spelling", self.name))?;
            let admission = if row.lifecycle.quarantined {
                "quarantined"
            } else if row.lifecycle.generatable {
                "enabled"
            } else {
                "compatibility"
            };
            if self.identity != format!("wire:{}", row.id.0) || self.admission != admission {
                return Err(format!(
                    "{}: inherited identity/lifecycle cannot change",
                    self.name
                ));
            }
            if self.weight > 0
                && !registry::generatable_operators(true).any(|r| r.name == self.name)
            {
                return Err(format!(
                    "{}: inherited generation/license/provenance refusal",
                    self.name
                ));
            }
        } else if (self.inherit == "hurdle2_v1" && self.name != "hurdle2")
            || (self.inherit != "hurdle2_v1"
                && !crate::full_rust::operation_schemas()
                    .iter()
                    .any(|r| r.op == self.name))
        {
            return Err(format!("{}: unknown inherited recipe", self.name));
        }
        Ok(())
    }

    pub fn facts(&'static self) -> Facts {
        if self.inherit == "advanced_v3" {
            return self.advanced_facts();
        }
        if self.inherit == "registry_v1" {
            let row = registry::operator(self.name).expect("checked registry inheritance");
            return Facts {
                entry: self,
                canonical: row.semantic_key,
                wire_id: Some(row.id.0),
                semantic_version: row.semantic_version.to_string(),
                role: match row.role {
                    registry::OperatorRole::Head => "head",
                    registry::OperatorRole::Transform => "transform",
                    registry::OperatorRole::Target => "target",
                    registry::OperatorRole::Directive => "directive",
                },
                parameters: serde_json::to_value(row.parameters).expect("static parameters"),
                fitted: Some(row.fitted),
                requires_labels: row.requires_labels,
                width: format!("{:?}", row.shape),
                provenance: format!(
                    "{}@{}:{}; license={}",
                    row.source.uri, row.source.revision, row.source.tree_sha256, row.license.spdx
                ),
            };
        }
        let schema = crate::full_rust::operation_schemas()
            .into_iter()
            .find(|r| r.op == self.name);
        Facts {
            entry: self,
            canonical: self.name,
            wire_id: None,
            semantic_version: "1".into(),
            role: if crate::heads::is_head_op(self.name) {
                "head"
            } else {
                "transform"
            },
            parameters: schema.map_or_else(
                || serde_json::json!(self.parameters),
                |s| {
                    let source = if crate::heads::is_head_op(self.name) {
                        format!("(pipe X ({}))", self.name)
                    } else {
                        format!("(pipe X ({}) (ridge))", self.name)
                    };
                    let defaults = crate::full_rust::canonicalize(&source)
                        .map(|g| g.canonical)
                        .expect("inherited recipe defaults execute");
                    serde_json::json!({"domains": s
                        .params
                        .iter()
                        .map(|p| (p.name, format!("{:?}", p.domain)))
                        .collect::<Vec<_>>(), "defaults_pipeline": defaults})
                },
            ),
            fitted: Some(self.effects[0] != "fit=none"),
            requires_labels: self.effects[0] == "fit=xy",
            width: self.effects[1]
                .strip_prefix("width=")
                .expect("checked width effect")
                .into(),
            provenance: "feat-math recipe/head v1; Apache-2.0; no wire allocation".into(),
        }
    }

    pub fn supports(&self, target: &str) -> Result<(), String> {
        if self.targets.contains(&target) && self.admission != "quarantined" {
            Ok(())
        } else {
            Err(format!(
                "{}/{}: unsupported target {target}",
                self.dialect, self.name
            ))
        }
    }

    fn available_for_draw(&self, target: &str) -> bool {
        if self.admission != "enabled" || self.weight == 0 || self.supports(target).is_err() {
            return false;
        }
        if self.inherit == "registry_v1" {
            // Compatibility entries never become new search draws by metadata alone.
            return registry::generatable_operators(true).any(|row| row.name == self.name);
        }
        true
    }

    fn validate_legacy(&self, source: &Stage) -> Result<Stage, String> {
        if matches!(self.admission, "quarantined" | "disabled") {
            return Err(format!("{}: {}", self.name, self.admission));
        }
        for key in source.params.keys() {
            if self.reject.contains(&key.as_str()) {
                return Err(format!("{}: no-op parameter '{key}'", self.name));
            }
        }
        for rule in self.inactive {
            let parts: Vec<_> = rule.split(':').collect();
            if parts.len() != 3 {
                return Err("invalid inactive rule".into());
            }
            if source.params.contains_key(parts[0])
                && !matches!(source.params.get(parts[1]), Some(ParamValue::Str(v)) if v == parts[2])
            {
                return Err(format!("{}: inactive parameter '{}'", self.name, parts[0]));
            }
        }
        let mut stage = source.clone();
        if self.inherit == "registry_v1" {
            let row = registry::operator(self.name).ok_or("missing inherited operator")?;
            if !row.lifecycle.executable || row.lifecycle.quarantined {
                return Err(format!(
                    "{}: inherited quarantine/execution refusal",
                    self.name
                ));
            }
            for (key, value) in &source.params {
                if key == "append" && row.allow_append {
                    if !matches!(value, ParamValue::Bool(_)) {
                        return Err("append must be boolean".into());
                    }
                    continue;
                }
                let p = row
                    .parameters
                    .iter()
                    .find(|p| p.name == key)
                    .ok_or_else(|| format!("{}: unknown parameter '{key}'", self.name))?;
                validate_value(key, value, p.kind, p.minimum, p.maximum)?;
            }
            for p in row.parameters {
                if !stage.params.contains_key(p.name) {
                    if let Some(default) = p.default {
                        stage.params.insert(
                            p.name.to_owned(),
                            match default {
                                ParameterValue::Usize(v) => ParamValue::Int(v as i64),
                                ParameterValue::U64(v) => ParamValue::Int(v as i64),
                                ParameterValue::F64(v) => ParamValue::Float(v),
                                ParameterValue::Bool(v) => ParamValue::Bool(v),
                                ParameterValue::String(v) => ParamValue::Str(v.to_owned()),
                            },
                        );
                    } else if p.required {
                        return Err(format!("{}: missing parameter {}", self.name, p.name));
                    }
                }
            }
            stage.op = row.semantic_key.to_owned();
        } else if self.inherit == "hurdle2_v1" {
            for key in source.params.keys() {
                if !self
                    .parameters
                    .iter()
                    .any(|p| p.split(':').next() == Some(key))
                {
                    return Err(format!("{}: unknown parameter '{key}'", self.name));
                }
            }
            for spec in self.parameters {
                let p: Vec<_> = spec.split(':').collect();
                if p.len() != 5 {
                    return Err("invalid supplemental parameter".into());
                }
                let default: f64 = p[2].parse().map_err(|_| "invalid default")?;
                let value = stage
                    .params
                    .entry(p[0].into())
                    .or_insert(ParamValue::Float(default));
                validate_value(
                    p[0],
                    value,
                    ParameterKind::F64,
                    p[3].parse().ok(),
                    p[4].parse().ok(),
                )?;
            }
        } else {
            let schema = crate::full_rust::operation_schemas()
                .into_iter()
                .find(|r| r.op == self.name)
                .ok_or("missing inherited recipe schema")?;
            for (key, value) in &source.params {
                let param = schema
                    .params
                    .iter()
                    .find(|p| p.name == key)
                    .ok_or_else(|| format!("{}: unknown parameter '{key}'", self.name))?;
                use crate::full_rust::ParamDomain;
                let valid = match (&param.domain, value) {
                    (ParamDomain::Boolean, ParamValue::Bool(_)) => true,
                    (ParamDomain::Categorical(values), ParamValue::Str(v)) => {
                        values.contains(&v.as_str())
                    }
                    (ParamDomain::Categorical(values), ParamValue::Int(v)) => {
                        values.contains(&v.to_string().as_str())
                    }
                    (ParamDomain::Numeric(values), _) => {
                        let numeric: Vec<_> = values
                            .iter()
                            .filter_map(|v| match v {
                                ParamValue::Int(v) => Some(*v as f64),
                                ParamValue::Float(v) => Some(*v),
                                _ => None,
                            })
                            .collect();
                        let min = numeric.iter().copied().reduce(f64::min);
                        let max = numeric.iter().copied().reduce(f64::max);
                        let kind = if values.iter().all(|v| matches!(v, ParamValue::Int(_))) {
                            ParameterKind::Usize
                        } else {
                            ParameterKind::F64
                        };
                        validate_value(key, value, kind, min, max).is_ok()
                    }
                    (ParamDomain::Expression, ParamValue::Str(v)) => {
                        crate::math_expr::MathExpr::parse(v).is_ok()
                    }
                    _ => false,
                };
                if !valid {
                    return Err(format!("{}: invalid parameter '{key}'", self.name));
                }
            }
        }
        if stage.op == "target" {
            let Some(ParamValue::Str(kind)) = stage.params.get("kind") else {
                return Err("target: missing kind".into());
            };
            if !TARGET_VALID.contains(&kind.as_str()) {
                return Err(format!("target: unknown kind {kind}"));
            }
        }
        if matches!(stage.op.as_str(), "map" | "derive") {
            let Some(ParamValue::Str(expression)) = stage.params.get("expr") else {
                return Err(format!("{}: missing expression", stage.op));
            };
            crate::math_expr::MathExpr::parse(expression).map_err(|e| e.to_string())?;
        }
        for rule in self.inactive {
            let parts: Vec<_> = rule.split(':').collect();
            if !matches!(stage.params.get(parts[1]), Some(ParamValue::Str(v)) if v == parts[2]) {
                stage.params.remove(parts[0]);
            }
        }
        Ok(stage)
    }
}

fn validate_value(
    key: &str,
    value: &ParamValue,
    kind: ParameterKind,
    min: Option<f64>,
    max: Option<f64>,
) -> Result<(), String> {
    let numeric = match (kind, value) {
        (ParameterKind::Usize | ParameterKind::U64, ParamValue::Int(v)) if *v >= 0 => {
            Some(*v as f64)
        }
        (ParameterKind::F64, ParamValue::Int(v)) => Some(*v as f64),
        (ParameterKind::F64, ParamValue::Float(v)) => Some(*v),
        (ParameterKind::Bool, ParamValue::Bool(_))
        | (ParameterKind::String, ParamValue::Str(_)) => None,
        _ => return Err(format!("{key}: wrong parameter type")),
    };
    if numeric
        .is_some_and(|v| !v.is_finite() || min.is_some_and(|m| v < m) || max.is_some_and(|m| v > m))
    {
        return Err(format!("{key}: outside parameter domain"));
    }
    Ok(())
}

#[derive(Clone, Copy, Debug)]
pub struct RegistryProfile {
    pub dialect: &'static str,
}

impl RegistryProfile {
    pub fn entries(self) -> impl Iterator<Item = &'static Entry> {
        OPERATORS.iter().filter(move |e| e.dialect == self.dialect)
    }

    /// Explicit opt-in seed validation. Advanced-v3 reuses its existing strict
    /// materializer; callers still need schema/compiler and receipt admission.
    pub fn validate_stage(self, stage: &Stage, task: &crate::TaskSpec) -> Result<Stage, String> {
        let entry = operator(self.dialect, &stage.op)
            .ok_or_else(|| format!("{}/{}: unknown operation", self.dialect, stage.op))?;
        if self.dialect == "advanced_v3" {
            entry.validate_advanced(stage, task)
        } else {
            entry.validate_legacy(stage)
        }
    }

    pub fn parse(self, source: &str, target: &str) -> Result<GenomeIr, String> {
        if self.dialect != "legacy_recipe" {
            return Err("advanced-v3 requires its schema-bound compiler".into());
        }
        let GenomeIr::Pipeline(pipe) = GenomeIr::parse(source).map_err(|e| e.to_string())? else {
            return Err(
                "registry-v2: explicit nested portfolio/symbolic form requires a separate profile"
                    .into(),
            );
        };
        let mut stages = Vec::new();
        for (index, stage) in pipe.stages.iter().enumerate() {
            let entry = operator(self.dialect, &stage.op)
                .ok_or_else(|| format!("unknown operation {}", stage.op))?;
            entry.supports(target)?;
            let role = entry.facts().role;
            if (role == "head" && index + 1 != pipe.stages.len())
                || (role == "target" && index != 0)
                || role == "directive"
            {
                return Err(format!("{}: invalid stage position", stage.op));
            }
            stages.push(entry.validate_legacy(stage)?);
        }
        if !stages
            .last()
            .is_some_and(|s| crate::heads::is_head_op(&s.op))
        {
            return Err("registry-v2: explicit terminal head required".into());
        }
        let genome = GenomeIr::Pipeline(DslExpr { stages });
        match target {
            "runtime" => {}
            "rust_recipe" | "python_recipe" => {
                crate::transpile::supports(&genome).map_err(|e| format!("{e:?}"))?
            }
            "cpp17_recipe" => {
                crate::transpile::supports_cpp17(&genome).map_err(|e| format!("{e:?}"))?
            }
            "javascript" => {
                crate::transpile::supports_javascript(&genome).map_err(|e| format!("{e:?}"))?
            }
            _ => return Err(format!("unsupported target {target}")),
        }
        Ok(genome)
    }

    /// Ordered weighted draw in the new profile only. No historical sampler calls this.
    pub fn draw_head(self, rng: &mut Pcg32, target: &str) -> Result<Stage, String> {
        if self.dialect != "legacy_recipe" {
            return Err("advanced-v3 generation requires receipt/schema admission".into());
        }
        let rows: Vec<_> = self
            .entries()
            .filter(|e| crate::heads::is_head_op(e.name) && e.available_for_draw(target))
            .collect();
        let total: u32 = rows.iter().map(|e| e.weight).sum();
        if total == 0 {
            return Err(format!("no heads for target {target}"));
        }
        let mut draw = rng.next_u32() % total;
        for entry in rows {
            if draw < entry.weight {
                return entry.validate_legacy(&Stage::new(entry.name));
            }
            draw -= entry.weight;
        }
        unreachable!("bounded weighted draw")
    }
}
