//! Historical bytes and IDs remain authoritative even when typing is unavailable.
use super::{
    canon, Constraints, Expression, InputSchema, Node, OperatorRef, ProgramGraph, ProgramIdV2,
    Result, Value,
};
use crate::kernel::Program;
use crate::GenomeIr;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "snake_case", deny_unknown_fields)]
pub enum BridgeStatus {
    Typed {
        graph: ProgramGraph,
        graph_id: ProgramIdV2,
    },
    Opaque {
        reason: String,
    },
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LegacyBridge {
    pub schema: String,
    pub original_hex: String,
    pub legacy_program_id: String,
    pub legacy_identity_text: String,
    pub bridge: BridgeStatus,
}

impl LegacyBridge {
    pub fn import(program: &Program, schema: &InputSchema, limits: &Constraints) -> Result<Self> {
        let bytes = program.to_bytes().map_err(|e| e.to_string())?;
        let graph = (|| {
            let Program::LegacyV1(genome) = program;
            let mut nodes = vec![Node::Input];
            match genome {
                GenomeIr::Pipeline(pipe) => {
                    for stage in &pipe.stages {
                        let operator = OperatorRef::new("legacy_recipe", &stage.op)?;
                        let parameters = stage
                            .params
                            .iter()
                            .map(|(k, v)| Ok((k.clone(), Value::from_legacy(v)?)))
                            .collect::<Result<_>>()?;
                        nodes.push(Node::Operator {
                            input: nodes.len() - 1,
                            operator,
                            parameters,
                        });
                    }
                    if !pipe
                        .stages
                        .last()
                        .is_some_and(|s| crate::heads::is_head_op(&s.op))
                    {
                        nodes.push(Node::Operator {
                            input: nodes.len() - 1,
                            operator: OperatorRef::new("legacy_recipe", "ridge")?,
                            parameters: Default::default(),
                        });
                    }
                }
                GenomeIr::Symbolic { expr, affine, .. } => nodes.push(Node::Scalar {
                    input: 0,
                    expression: Expression::from_ast(expr.ast(), schema)?,
                    affine: *affine,
                }),
                GenomeIr::Portfolio(_) => {
                    return Err(
                        "explicit portfolio: nested cross-fitting adapter unavailable".into(),
                    )
                }
            }
            let output = nodes.len() - 1;
            ProgramGraph::new(schema.clone(), nodes, output, limits)
        })();
        let bridge = match graph {
            Ok(graph) => BridgeStatus::Typed {
                graph_id: graph.identity(limits)?,
                graph,
            },
            Err(reason) => BridgeStatus::Opaque { reason },
        };
        Ok(Self {
            schema: "jope.selfplay.legacy-graph-bridge/v1".into(),
            original_hex: bytes.iter().map(|b| format!("{b:02x}")).collect(),
            legacy_program_id: program.identity().map_err(|e| e.to_string())?,
            legacy_identity_text: program.legacy_identity_text(),
            bridge,
        })
    }
    pub fn original(&self) -> Result<Program> {
        if self.schema != "jope.selfplay.legacy-graph-bridge/v1"
            || self.original_hex.len() > 32 * 1024 * 1024
            || !self.original_hex.len().is_multiple_of(2)
        {
            return Err("legacy bridge envelope/byte bound".into());
        }
        let bytes = self
            .original_hex
            .as_bytes()
            .chunks_exact(2)
            .map(|s| {
                let text = std::str::from_utf8(s).map_err(|e| e.to_string())?;
                u8::from_str_radix(text, 16).map_err(|e| e.to_string())
            })
            .collect::<Result<Vec<_>>>()?;
        let program = Program::from_bytes(&bytes).map_err(|e| e.to_string())?;
        if program.identity().map_err(|e| e.to_string())? != self.legacy_program_id
            || program.legacy_identity_text() != self.legacy_identity_text
        {
            return Err("legacy bridge identity mismatch".into());
        }
        Ok(program)
    }
    fn validate_relation(&self, limits: &Constraints) -> Result<()> {
        let original = self.original()?;
        if let BridgeStatus::Typed { graph, graph_id } = &self.bridge {
            if graph.identity(limits)? != *graph_id {
                return Err("bridge graph identity mismatch".into());
            }
            let expected = Self::import(&original, graph.input_schema(), limits)?;
            if expected.bridge != self.bridge {
                return Err("forged legacy-to-typed bridge relation".into());
            }
        }
        Ok(())
    }
    pub fn to_bytes(&self) -> Result<Vec<u8>> {
        self.validate_relation(&Constraints::default())?;
        canon::encode(self)
    }
    pub fn from_bytes(bytes: &[u8], limits: &Constraints) -> Result<Self> {
        if bytes.len() > canon::MAX_BYTES {
            return Err("bridge bytes exceed transport bound".into());
        }
        let bridge: Self = serde_json::from_slice(bytes).map_err(|e| e.to_string())?;
        bridge.validate_relation(limits)?;
        if bridge.to_bytes()? != bytes {
            return Err("noncanonical bridge bytes".into());
        }
        Ok(bridge)
    }
}
