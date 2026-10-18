//! Deterministic legal actions. Enumeration and application never fit or draw RNG.
use super::{
    canon, Constraints, InputSchema, Node, OperatorRef, ProgramGraph, ProgramIdV2, Result, Value,
};
use crate::kernel::registry_v2::RegistryProfile;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "action", rename_all = "snake_case", deny_unknown_fields)]
pub enum EditKind {
    NoOp,
    ReplaceHead {
        site: usize,
        operator: OperatorRef,
        parameters: BTreeMap<String, Value>,
    },
    Insert {
        before: usize,
        operator: OperatorRef,
        parameters: BTreeMap<String, Value>,
    },
    Delete {
        site: usize,
    },
    Parameter {
        site: usize,
        key: String,
        value: Value,
    },
    Column {
        site: usize,
        position: usize,
        column: super::ColumnRef,
    },
    CrossoverHead {
        donor: Box<ProgramGraph>,
    },
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Edit {
    pub schema: String,
    pub parents: Vec<ProgramIdV2>,
    pub kind: EditKind,
}
impl Edit {
    pub fn new(parent: &ProgramGraph, kind: EditKind, limits: &Constraints) -> Result<Self> {
        Ok(Self {
            schema: "jope.selfplay.attempt-edit/v1".into(),
            parents: vec![parent.identity(limits)?],
            kind,
        })
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EditOutcome {
    Changed,
    Unchanged,
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EditRecord {
    pub schema: String,
    pub edit: Edit,
    pub child: ProgramIdV2,
    pub outcome: EditOutcome,
}
#[derive(Debug)]
pub struct AppliedEdit {
    pub graph: ProgramGraph,
    pub record: EditRecord,
}

fn binding(
    graph: &ProgramGraph,
    schema: &InputSchema,
    profile: RegistryProfile,
    limits: &Constraints,
) -> Result<()> {
    graph.validate(limits)?;
    if graph.input_schema() != schema || profile.dialect != "legacy_recipe" {
        return Err("edit schema/profile mismatch".into());
    }
    Ok(())
}

pub fn apply(
    graph: &ProgramGraph,
    edit: &Edit,
    schema: &InputSchema,
    profile: RegistryProfile,
    limits: &Constraints,
) -> Result<AppliedEdit> {
    binding(graph, schema, profile, limits)?;
    if edit.schema != "jope.selfplay.attempt-edit/v1"
        || edit.parents.first() != Some(&graph.identity(limits)?)
        || canon::encode(edit)?.len() > limits.max_bytes
    {
        return Err("edit version/parent/byte binding mismatch".into());
    }
    let cross = matches!(edit.kind, EditKind::CrossoverHead { .. });
    if edit.parents.len() != if cross { 2 } else { 1 } {
        return Err("edit requires exact parent inventory".into());
    }
    let mut child = graph.clone();
    match &edit.kind {
        EditKind::NoOp => {}
        EditKind::ReplaceHead {
            site,
            operator,
            parameters,
        } => {
            if *site != child.output {
                return Err("head replacement site mismatch".into());
            }
            let Some(Node::Operator { input, .. }) = child.nodes.get(*site) else {
                return Err("head operator required".into());
            };
            child.nodes[*site] = Node::Operator {
                input: *input,
                operator: operator.clone(),
                parameters: parameters.clone(),
            };
        }
        EditKind::Insert {
            before,
            operator,
            parameters,
        } => {
            if *before == 0
                || *before >= child.nodes.len()
                || !matches!(child.nodes.get(*before), Some(Node::Operator { .. }))
            {
                return Err("insert site must be in native stage chain".into());
            }
            let input = child.nodes[*before].dependencies()[0];
            child.nodes.insert(
                *before,
                Node::Operator {
                    input,
                    operator: operator.clone(),
                    parameters: parameters.clone(),
                },
            );
            for (index, node) in child.nodes.iter_mut().enumerate().skip(*before + 1) {
                let Node::Operator { input, .. } = node else {
                    return Err("insert requires native stage chain".into());
                };
                *input = index - 1;
            }
            child.output += 1;
        }
        EditKind::Delete { site } => {
            if *site == 0
                || *site >= child.output
                || !matches!(child.nodes.get(*site), Some(Node::Operator { .. }))
            {
                return Err("delete requires a nonterminal stage".into());
            }
            child.nodes.remove(*site);
            for (index, node) in child.nodes.iter_mut().enumerate().skip(*site) {
                let Node::Operator { input, .. } = node else {
                    return Err("delete requires native stage chain".into());
                };
                *input = index - 1;
            }
            child.output -= 1;
        }
        EditKind::Parameter { site, key, value } => {
            let Some(Node::Operator { parameters, .. }) = child.nodes.get_mut(*site) else {
                return Err("parameter edit site".into());
            };
            parameters.insert(key.clone(), value.clone());
        }
        EditKind::Column {
            site,
            position,
            column,
        } => {
            let Some(Node::Columns { columns, .. }) = child.nodes.get_mut(*site) else {
                return Err("column edit site".into());
            };
            let entry = columns.get_mut(*position).ok_or("column edit position")?;
            *entry = column.clone();
        }
        EditKind::CrossoverHead { donor } => {
            binding(donor, schema, profile, limits)?;
            if edit.parents.get(1) != Some(&donor.identity(limits)?) {
                return Err("second-parent identity mismatch".into());
            }
            let Some(Node::Operator {
                operator,
                parameters,
                ..
            }) = donor.nodes.get(donor.output)
            else {
                return Err("donor terminal head required".into());
            };
            let Some(Node::Operator { input, .. }) = child.nodes.get(child.output) else {
                return Err("recipient terminal head required".into());
            };
            child.nodes[child.output] = Node::Operator {
                input: *input,
                operator: operator.clone(),
                parameters: parameters.clone(),
            };
        }
    }
    child.validate(limits)?;
    let id = child.identity(limits)?;
    let outcome = if edit.parents[0] == id {
        EditOutcome::Unchanged
    } else {
        EditOutcome::Changed
    };
    Ok(AppliedEdit {
        graph: child,
        record: EditRecord {
            schema: "jope.selfplay.graph-edit-record/v1".into(),
            edit: edit.clone(),
            child: id,
            outcome,
        },
    })
}

pub fn crossover_edit(
    left: &ProgramGraph,
    right: &ProgramGraph,
    schema: &InputSchema,
    profile: RegistryProfile,
    limits: &Constraints,
) -> Result<Edit> {
    let edit = Edit {
        schema: "jope.selfplay.attempt-edit/v1".into(),
        parents: vec![left.identity(limits)?, right.identity(limits)?],
        kind: EditKind::CrossoverHead {
            donor: Box::new(right.clone()),
        },
    };
    apply(left, &edit, schema, profile, limits)?;
    Ok(edit)
}

pub fn legal_edits(
    graph: &ProgramGraph,
    schema: &InputSchema,
    profile: RegistryProfile,
    limits: &Constraints,
) -> Result<Vec<Edit>> {
    binding(graph, schema, profile, limits)?;
    let mut candidates = vec![EditKind::NoOp];
    if matches!(graph.nodes.get(graph.output), Some(Node::Operator { .. })) {
        for entry in profile
            .entries()
            .filter(|e| e.admission == "enabled" && e.weight > 0)
        {
            let facts = entry.facts();
            if facts.role == "head" && super::graph::supported_head(facts.canonical) {
                candidates.push(EditKind::ReplaceHead {
                    site: graph.output,
                    operator: OperatorRef::new(entry.dialect, entry.name)?,
                    parameters: BTreeMap::new(),
                });
            }
            // This enumerator's insertion subset is inherited width-preserving stages.
            if facts.role == "transform"
                && crate::registry::operator(entry.name)
                    .is_some_and(|r| r.shape == crate::registry::ShapeRule::Preserve)
            {
                for (before, node) in graph.nodes.iter().enumerate().skip(1) {
                    if matches!(node, Node::Operator { .. }) {
                        candidates.push(EditKind::Insert {
                            before,
                            operator: OperatorRef::new(entry.dialect, entry.name)?,
                            parameters: BTreeMap::new(),
                        });
                    }
                }
            }
        }
    }
    for (site, node) in graph.nodes.iter().enumerate() {
        if let Node::Operator { operator, .. } = node {
            if site < graph.output {
                candidates.push(EditKind::Delete { site });
            }
            if let Some(row) = crate::registry::operator(&operator.name) {
                for parameter in row.parameters {
                    for value in parameter.variants {
                        let raw = match value {
                            crate::registry::ParameterValue::Usize(v) => {
                                crate::dsl::ParamValue::Int(*v as i64)
                            }
                            crate::registry::ParameterValue::U64(v) => {
                                crate::dsl::ParamValue::Int(*v as i64)
                            }
                            crate::registry::ParameterValue::F64(v) => {
                                crate::dsl::ParamValue::Float(*v)
                            }
                            crate::registry::ParameterValue::Bool(v) => {
                                crate::dsl::ParamValue::Bool(*v)
                            }
                            crate::registry::ParameterValue::String(v) => {
                                crate::dsl::ParamValue::Str((*v).into())
                            }
                        };
                        candidates.push(EditKind::Parameter {
                            site,
                            key: parameter.name.into(),
                            value: Value::from_legacy(&raw)?,
                        });
                    }
                }
            }
        }
        if let Node::Columns { columns, .. } = node {
            for position in 0..columns.len() {
                for index in 0..schema.columns().len() {
                    candidates.push(EditKind::Column {
                        site,
                        position,
                        column: schema.column(index)?,
                    });
                    if candidates.len() > 4096 {
                        return Err("legal edit enumeration bound".into());
                    }
                }
            }
        }
        if candidates.len() > 4096 {
            return Err("legal edit enumeration bound".into());
        }
    }
    let mut result = Vec::new();
    for kind in candidates {
        let edit = Edit::new(graph, kind, limits)?;
        if apply(graph, &edit, schema, profile, limits).is_ok() {
            result.push(edit);
        }
    }
    Ok(result)
}
