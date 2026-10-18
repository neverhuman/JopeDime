use super::{canon, Constraints, InputSchema, OperatorRef, ProgramGraph, ProgramIdV2, Result};
use crate::{Model, Rows};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Effects {
    pub fit_x: bool,
    pub fit_y: bool,
    pub target_derived: bool,
    pub ordered_rows: bool,
    pub seeded: bool,
}
impl Effects {
    pub fn union(self, other: Self) -> Self {
        Self {
            fit_x: self.fit_x || other.fit_x,
            fit_y: self.fit_y || other.fit_y,
            target_derived: self.target_derived || other.target_derived,
            ordered_rows: self.ordered_rows || other.ordered_rows,
            seeded: self.seeded || other.seeded,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct InputIdentity {
    pub source_sha256: String,
    pub input_sha256: String,
    pub target_sha256: String,
    pub split_sha256: String,
}
impl InputIdentity {
    fn validate(&self) -> Result<()> {
        for value in [
            &self.source_sha256,
            &self.input_sha256,
            &self.target_sha256,
            &self.split_sha256,
        ] {
            canon::digest_ref(value)?;
        }
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub struct Dataset {
    pub schema_id: String,
    pub identity: InputIdentity,
    pub x: Rows,
    pub y: Vec<f64>,
}
impl Dataset {
    fn validate_envelope(&self, schema: &InputSchema) -> Result<()> {
        self.identity.validate()?;
        if self.schema_id != schema.identity()?
            || self.x.is_empty()
            || self.x.len() > 4096
            || self.x.len() != self.y.len()
        {
            return Err("dataset schema/row-count envelope mismatch".into());
        }
        Ok(())
    }
    fn features(&self, rows: &[usize], schema: &InputSchema) -> Result<Rows> {
        rows.iter()
            .map(|i| {
                let row = self.x.get(*i).ok_or("row outside dataset")?;
                if row.len() != schema.columns().len() || row.iter().any(|v| !v.is_finite()) {
                    return Err("accessed feature row shape/nonfinite value".into());
                }
                Ok(row.clone())
            })
            .collect()
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RowClosure {
    pub fit_rows: Vec<usize>,
    pub predict_rows: Vec<usize>,
}
impl RowClosure {
    pub fn validate(&self, n: usize) -> Result<()> {
        for rows in [&self.fit_rows, &self.predict_rows] {
            if rows.is_empty()
                || rows.len() > 4096
                || rows.iter().any(|r| *r >= n)
                || rows.windows(2).any(|w| w[0] >= w[1])
            {
                return Err("row closure must be sorted, unique, nonempty and in bounds".into());
            }
        }
        if self
            .fit_rows
            .iter()
            .any(|r| self.predict_rows.binary_search(r).is_ok())
        {
            return Err("fitting rows overlap held prediction rows".into());
        }
        Ok(())
    }
    fn subset_of(&self, parent: &Self) -> bool {
        self.fit_rows
            .iter()
            .all(|r| parent.fit_rows.binary_search(r).is_ok())
            && self
                .predict_rows
                .iter()
                .all(|r| parent.predict_rows.binary_search(r).is_ok())
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct FitPlan {
    schema: String,
    graph_id: ProgramIdV2,
    input_schema_id: String,
    identity: InputIdentity,
    closure: RowClosure,
    seed_u64_hex: String,
    operators: Vec<OperatorRef>,
    effects: Effects,
    fit_features_sha256: String,
    fit_targets_sha256: String,
    parent_plan: Option<ProgramIdV2>,
}
fn exact_rows(rows: &Rows) -> Vec<Vec<String>> {
    rows.iter()
        .map(|r| r.iter().map(|v| format!("{:016x}", v.to_bits())).collect())
        .collect()
}
fn hashes(
    graph: &ProgramGraph,
    data: &Dataset,
    closure: &RowClosure,
    effects: Effects,
) -> Result<(String, String)> {
    data.validate_envelope(graph.input_schema())?;
    closure.validate(data.x.len())?;
    if closure
        .fit_rows
        .len()
        .checked_mul(graph.input_schema().columns().len())
        .and_then(|n| n.checked_mul(20))
        .is_none_or(|n| n > canon::MAX_BYTES)
    {
        return Err("training identity byte bound before allocation".into());
    }
    let x = data.features(&closure.fit_rows, graph.input_schema())?;
    let mut targets = Vec::new();
    if effects.fit_y {
        for i in &closure.fit_rows {
            let v = data.y[*i];
            if !v.is_finite() {
                return Err("accessed target row is nonfinite".into());
            }
            targets.push(format!("{:016x}", v.to_bits()));
        }
    }
    Ok((
        canon::sha256(&canon::encode(&exact_rows(&x))?),
        canon::sha256(&canon::encode(&targets)?),
    ))
}
impl FitPlan {
    pub fn bind(
        graph: &ProgramGraph,
        data: &Dataset,
        closure: RowClosure,
        seed: u64,
        limits: &Constraints,
    ) -> Result<Self> {
        let effects = graph.validate(limits)?;
        let (x, y) = hashes(graph, data, &closure, effects)?;
        let operators = graph
            .nodes()
            .iter()
            .filter_map(|n| {
                if let super::Node::Operator { operator, .. } = n {
                    Some(operator.clone())
                } else {
                    None
                }
            })
            .collect();
        Ok(Self {
            schema: "jope.selfplay.graph-fit-plan/v1".into(),
            graph_id: graph.identity(limits)?,
            input_schema_id: graph.input_schema().identity()?,
            identity: data.identity.clone(),
            closure,
            seed_u64_hex: format!("{seed:016x}"),
            operators,
            effects,
            fit_features_sha256: x,
            fit_targets_sha256: y,
            parent_plan: None,
        })
    }
    pub fn narrow(
        &self,
        graph: &ProgramGraph,
        data: &Dataset,
        closure: RowClosure,
        limits: &Constraints,
    ) -> Result<Self> {
        self.validate_for_original_data(data)?;
        if !closure.subset_of(&self.closure) {
            return Err("child plan broadens parent row closure".into());
        }
        // Refuse additional label access before constructing any child data digest.
        let effects = graph.validate(limits)?;
        if graph.input_schema().identity()? != self.input_schema_id
            || (effects.fit_y && !self.effects.fit_y)
        {
            return Err("child plan broadens schema/target access".into());
        }
        let mut child = Self::bind(
            graph,
            data,
            closure,
            canon::hex_u64(&self.seed_u64_hex)?,
            limits,
        )?;
        if child.input_schema_id != self.input_schema_id
            || child.identity != self.identity
            || (child.effects.fit_y && !self.effects.fit_y)
        {
            return Err("child plan broadens schema/input/target access".into());
        }
        child.parent_plan = Some(self.identity()?);
        Ok(child)
    }
    pub fn closure(&self) -> &RowClosure {
        &self.closure
    }
    pub fn effects(&self) -> Effects {
        self.effects
    }
    pub fn identity(&self) -> Result<ProgramIdV2> {
        ProgramIdV2::compute(&self.schema, &canon::encode(self)?)
    }
    fn validate_for_original_data(&self, data: &Dataset) -> Result<()> {
        if self.schema != "jope.selfplay.graph-fit-plan/v1"
            || self.identity != data.identity
            || self.input_schema_id != data.schema_id
        {
            return Err("fit-plan source/input/target/split identity mismatch".into());
        }
        canon::hex_u64(&self.seed_u64_hex)?;
        self.closure.validate(data.x.len())?;
        Ok(())
    }
    fn validate(&self, graph: &ProgramGraph, data: &Dataset, limits: &Constraints) -> Result<()> {
        self.validate_for_original_data(data)?;
        if self.graph_id != graph.identity(limits)?
            || self.effects != graph.validate(limits)?
            || self.input_schema_id != graph.input_schema().identity()?
        {
            return Err("fit-plan graph/effect/schema binding mismatch before data access".into());
        }
        let rebound = Self::bind(
            graph,
            data,
            self.closure.clone(),
            canon::hex_u64(&self.seed_u64_hex)?,
            limits,
        )?;
        let mut expected = self.clone();
        expected.parent_plan = None;
        if expected != rebound {
            return Err("fit-plan graph/operator/effect/data binding mismatch".into());
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ExecutionStats {
    pub fit_calls: usize,
    pub feature_fit_rows: BTreeSet<usize>,
    pub target_fit_rows: BTreeSet<usize>,
    pub prediction_rows: BTreeSet<usize>,
}

pub struct FittedGraph {
    model: Model,
    input_schema: InputSchema,
    identity: InputIdentity,
    closure: RowClosure,
    columns: Vec<usize>,
    graph_id: ProgramIdV2,
    plan_id: ProgramIdV2,
}
impl ProgramGraph {
    pub fn fit(
        &self,
        plan: &FitPlan,
        data: &Dataset,
        limits: &Constraints,
        stats: &mut ExecutionStats,
    ) -> Result<FittedGraph> {
        let lowered = self.lower(limits)?;
        plan.validate(self, data, limits)?;
        let input = data.features(&plan.closure.fit_rows, self.input_schema())?;
        let x: Rows = input
            .iter()
            .map(|row| lowered.columns.iter().map(|i| row[*i]).collect())
            .collect();
        let y: Vec<_> = if lowered.effects.fit_y {
            plan.closure.fit_rows.iter().map(|i| data.y[*i]).collect()
        } else {
            vec![0.0; x.len()]
        };
        // Every structural, closure, version, domain and content check precedes this boundary.
        stats.fit_calls += 1;
        stats
            .feature_fit_rows
            .extend(plan.closure.fit_rows.iter().copied());
        if lowered.effects.fit_y {
            stats
                .target_fit_rows
                .extend(plan.closure.fit_rows.iter().copied());
        }
        let model = crate::model::fit(&lowered.genome, &x, &y).map_err(|e| e.to_string())?;
        Ok(FittedGraph {
            model,
            input_schema: self.input_schema().clone(),
            identity: data.identity.clone(),
            closure: plan.closure.clone(),
            columns: lowered.columns,
            graph_id: self.identity(limits)?,
            plan_id: plan.identity()?,
        })
    }
}
impl FittedGraph {
    pub fn predict(
        &self,
        data: &Dataset,
        rows: &[usize],
        stats: &mut ExecutionStats,
    ) -> Result<Vec<f64>> {
        data.validate_envelope(&self.input_schema)?;
        if data.identity != self.identity
            || rows.is_empty()
            || rows.len() > 4096
            || rows.windows(2).any(|w| w[0] >= w[1])
            || rows
                .iter()
                .any(|r| self.closure.predict_rows.binary_search(r).is_err())
        {
            return Err("prediction rows/source outside plan closure".into());
        }
        let input = data.features(rows, &self.input_schema)?;
        let x: Rows = input
            .iter()
            .map(|row| self.columns.iter().map(|i| row[*i]).collect())
            .collect();
        stats.prediction_rows.extend(rows.iter().copied());
        let prediction = self.model.predict(&x).map_err(|e| e.to_string())?;
        if prediction.len() != rows.len() || prediction.iter().any(|v| !v.is_finite()) {
            return Err("graph interpreter produced invalid predictions".into());
        }
        Ok(prediction)
    }
    /// Exact tagged model-state snapshot, before any scoring. This is fixture
    /// evidence, not a learned outcome or an independent leakage certificate.
    pub fn state_bytes(&self) -> Result<Vec<u8>> {
        fn exact(value: &mut serde_json::Value) {
            match value {
                serde_json::Value::Number(n) if n.is_f64() => {
                    if let Some(v) = n.as_f64() {
                        *value = serde_json::json!({"f64_bits":format!("{:016x}",v.to_bits())});
                    }
                }
                serde_json::Value::Array(v) => v.iter_mut().for_each(exact),
                serde_json::Value::Object(v) => v.values_mut().for_each(exact),
                _ => {}
            }
        }
        let mut model = serde_json::to_value(&self.model).map_err(|e| e.to_string())?;
        exact(&mut model);
        canon::encode(
            &serde_json::json!({"schema":"jope.selfplay.graph-fit-state/v1","graph":self.graph_id,"plan":self.plan_id,"model":model}),
        )
    }
}
