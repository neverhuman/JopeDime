//! Lossless snapshots and replayable endpoint differences. A diff is not an operator witness.
use anyhow::{bail, ensure, Result};
use feat_math::{
    dsl::{DslExpr, ParamValue, Stage},
    kernel::Program,
    GenomeIr,
};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

pub const MAX_PROGRAM_BYTES: usize = 64 * 1024;
const MAX_ALIGNED_STAGES: usize = 128;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProgramSnapshot {
    /// Versioned codec used for `lossless`.
    pub codec: String,
    /// Stable identity of the parsed program.
    pub identity: String,
    /// Reversible serialized program representation.
    pub lossless: String,
    /// `typed_capture` preserves the producer's values; `source_capture` preserves only
    /// values present in the supplied text, never precision already lost upstream.
    pub precision: String,
}

impl ProgramSnapshot {
    pub fn from_typed(genome: &GenomeIr) -> Result<Self> {
        Self::from_program(&Program::import_legacy(genome), "typed_capture")
    }
    pub fn from_legacy(source: &str) -> Result<Self> {
        ensure!(
            source.len() <= MAX_PROGRAM_BYTES,
            "program exceeds capture limit"
        );
        Self::from_program(&Program::parse_legacy(source)?, "source_capture")
    }
    fn from_program(program: &Program, precision: &str) -> Result<Self> {
        ensure!(
            program.to_bytes()?.len() <= MAX_PROGRAM_BYTES,
            "program exceeds capture limit"
        );
        Ok(Self {
            codec: "feat-math/program/1".into(),
            identity: program.identity()?,
            lossless: program.to_lossless_text()?,
            precision: precision.into(),
        })
    }
    pub fn validate(&self) -> Result<Program> {
        ensure!(
            self.codec == "feat-math/program/1",
            "unsupported program codec"
        );
        ensure!(
            matches!(self.precision.as_str(), "source_capture" | "typed_capture"),
            "unsupported precision scope"
        );
        ensure!(
            self.lossless.len() <= MAX_PROGRAM_BYTES * 2 + 64,
            "program exceeds capture limit"
        );
        let program = Program::parse_lossless_text(&self.lossless)?;
        ensure!(
            program.to_bytes()?.len() <= MAX_PROGRAM_BYTES,
            "program exceeds capture limit"
        );
        ensure!(
            program.identity()? == self.identity,
            "program identity mismatch"
        );
        Ok(program)
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(
    tag = "type",
    content = "value",
    rename_all = "snake_case",
    deny_unknown_fields
)]
pub enum ExactValue {
    Int(String),
    FloatBits(String),
    Bool(bool),
    Str(String),
}
impl ExactValue {
    pub fn from_param(p: &ParamValue) -> Self {
        match p {
            ParamValue::Int(v) => Self::Int(v.to_string()),
            ParamValue::Float(v) => Self::FloatBits(format!("{:016x}", v.to_bits())),
            ParamValue::Bool(v) => Self::Bool(*v),
            ParamValue::Str(v) => Self::Str(v.clone()),
        }
    }
    pub fn to_param(&self) -> Result<ParamValue> {
        Ok(match self {
            Self::Int(v) => {
                let n: i64 = v.parse()?;
                ensure!(n.to_string() == *v, "noncanonical integer");
                ParamValue::Int(n)
            }
            Self::FloatBits(v) => {
                ensure!(
                    v.len() == 16
                        && v.bytes()
                            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b)),
                    "invalid float bits"
                );
                ParamValue::Float(f64::from_bits(u64::from_str_radix(v, 16)?))
            }
            Self::Bool(v) => ParamValue::Bool(*v),
            Self::Str(v) => ParamValue::Str(v.clone()),
        })
    }
    pub fn display(&self) -> String {
        match self {
            Self::FloatBits(v) => match self.to_param() {
                Ok(ParamValue::Float(f)) if f.is_finite() => format!("{f} [bits {v}]"),
                _ => format!("float bits {v}"),
            },
            Self::Int(v) => v.clone(),
            Self::Bool(v) => v.to_string(),
            Self::Str(v) => serde_json::to_string(v).unwrap_or_default(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StageView {
    /// Registered stage operation name.
    pub op: String,
    /// Exact parameter values keyed by parameter name.
    pub params: BTreeMap<String, ExactValue>,
}
impl StageView {
    fn from_stage(stage: &Stage) -> Self {
        Self {
            op: stage.op.clone(),
            params: stage
                .params
                .iter()
                .map(|(k, v)| (k.clone(), ExactValue::from_param(v)))
                .collect(),
        }
    }
    fn to_stage(&self) -> Result<Stage> {
        Ok(Stage {
            op: self.op.clone(),
            params: self
                .params
                .iter()
                .map(|(k, v)| Ok((k.clone(), v.to_param()?)))
                .collect::<Result<_>>()?,
        })
    }
}

/// Sites are zero-based in the evolving stage sequence. Preconditions prevent applying a
/// valid edit to the wrong parent. ReplaceProgram remains exact for unsupported dialects.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum AtomicEdit {
    Insert {
        at: usize,
        stage: StageView,
    },
    Delete {
        at: usize,
        before: StageView,
    },
    Replace {
        at: usize,
        before: StageView,
        after: StageView,
    },
    Parameter {
        at: usize,
        op: String,
        key: String,
        before: Option<ExactValue>,
        after: Option<ExactValue>,
    },
    Swap {
        at: usize,
        left: StageView,
        right: StageView,
    },
    ReplaceProgram {
        before: String,
        after: ProgramSnapshot,
    },
}

/// Original-parent and final-child stage coordinates, independent of the sequential
/// patch cursor (`AtomicEdit::at`). A whole-program replacement has neither coordinate.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EditSite {
    /// Stage index in the original parent, if this edit refers to one.
    pub before_index: Option<usize>,
    /// Stage index in the final child, if this edit refers to one.
    pub after_index: Option<usize>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProgramDelta {
    /// Program snapshot from which replay starts.
    pub parent: ProgramSnapshot,
    /// Expected program snapshot after replay.
    pub child: ProgramSnapshot,
    /// Ordered reversible edits from `parent` to `child`.
    pub edits: Vec<AtomicEdit>,
    /// Original-parent/final-child coordinates corresponding to `edits`.
    pub sites: Vec<EditSite>,
    /// Coverage level of the structural comparison.
    pub coverage: String,
    /// Source representation from which the difference was derived.
    pub provenance: String,
}

impl ProgramDelta {
    pub fn replay(&self) -> Result<ProgramSnapshot> {
        ensure!(
            self.sites.len() == self.edits.len(),
            "patch coordinate count mismatch"
        );
        let parent = self.parent.validate()?;
        self.child.validate()?;
        if let [AtomicEdit::ReplaceProgram { before, after }] = self.edits.as_slice() {
            ensure!(
                *before == self.parent.identity && after == &self.child,
                "replacement binding mismatch"
            );
            after.validate()?;
            return Ok(after.clone());
        }
        let Program::LegacyV1(GenomeIr::Pipeline(pipe)) = parent else {
            ensure!(
                self.edits.is_empty() && self.parent.identity == self.child.identity,
                "unsupported structural replay"
            );
            return Ok(self.child.clone());
        };
        let mut stages: Vec<_> = pipe.stages.iter().map(StageView::from_stage).collect();
        for edit in &self.edits {
            match edit {
                AtomicEdit::Insert { at, stage } => {
                    ensure!(*at <= stages.len(), "insert site out of range");
                    stages.insert(*at, stage.clone());
                }
                AtomicEdit::Delete { at, before } => {
                    ensure!(stages.get(*at) == Some(before), "delete precondition");
                    stages.remove(*at);
                }
                AtomicEdit::Replace { at, before, after } => {
                    ensure!(stages.get(*at) == Some(before), "replace precondition");
                    stages[*at] = after.clone();
                }
                AtomicEdit::Parameter {
                    at,
                    op,
                    key,
                    before,
                    after,
                } => {
                    let stage = stages
                        .get_mut(*at)
                        .ok_or_else(|| anyhow::anyhow!("parameter site out of range"))?;
                    ensure!(
                        stage.op == *op && stage.params.get(key) == before.as_ref(),
                        "parameter precondition"
                    );
                    if let Some(v) = after {
                        stage.params.insert(key.clone(), v.clone());
                    } else {
                        stage.params.remove(key);
                    }
                }
                AtomicEdit::Swap { at, left, right } => {
                    ensure!(
                        stages.get(*at) == Some(left) && stages.get(at + 1) == Some(right),
                        "swap precondition"
                    );
                    stages.swap(*at, at + 1);
                }
                AtomicEdit::ReplaceProgram { .. } => bail!("mixed whole-program replacement"),
            }
        }
        let genome = GenomeIr::Pipeline(DslExpr {
            stages: stages
                .iter()
                .map(StageView::to_stage)
                .collect::<Result<_>>()?,
        });
        let result =
            ProgramSnapshot::from_program(&Program::import_legacy(&genome), &self.child.precision)?;
        ensure!(result == self.child, "patch does not reconstruct child");
        Ok(result)
    }
}

pub fn diff_programs(parent: &ProgramSnapshot, child: &ProgramSnapshot) -> Result<ProgramDelta> {
    let (p, c) = (parent.validate()?, child.validate()?);
    let mut delta = ProgramDelta {
        parent: parent.clone(),
        child: child.clone(),
        edits: Vec::new(),
        sites: Vec::new(),
        coverage: "pipeline_structure".into(),
        provenance: "verified_endpoint_difference".into(),
    };
    if parent.identity == child.identity {
        delta.replay()?;
        return Ok(delta);
    }
    match (p, c) {
        (Program::LegacyV1(GenomeIr::Pipeline(p)), Program::LegacyV1(GenomeIr::Pipeline(c)))
            if p.stages.len().max(c.stages.len()) <= MAX_ALIGNED_STAGES =>
        {
            let a: Vec<_> = p.stages.iter().map(StageView::from_stage).collect();
            let b: Vec<_> = c.stages.iter().map(StageView::from_stage).collect();
            let dif: Vec<_> = (0..a.len().min(b.len()))
                .filter(|i| a[*i] != b[*i])
                .collect();
            if a.len() == b.len()
                && dif.len() == 2
                && dif[1] == dif[0] + 1
                && a[dif[0]] == b[dif[1]]
                && a[dif[1]] == b[dif[0]]
            {
                delta.sites.push(EditSite {
                    before_index: Some(dif[0]),
                    after_index: Some(dif[0]),
                });
                delta.edits.push(AtomicEdit::Swap {
                    at: dif[0],
                    left: a[dif[0]].clone(),
                    right: a[dif[1]].clone(),
                });
            } else {
                let (n, m) = (a.len(), b.len());
                let mut cost = vec![vec![0usize; m + 1]; n + 1];
                for i in (0..=n).rev() {
                    for j in (0..=m).rev() {
                        cost[i][j] = if i == n {
                            2 * (m - j)
                        } else if j == m {
                            2 * (n - i)
                        } else {
                            (cost[i + 1][j + 1]
                                + if a[i] == b[j] {
                                    0
                                } else if a[i].op == b[j].op {
                                    1
                                } else {
                                    3
                                })
                            .min(2 + cost[i + 1][j])
                            .min(2 + cost[i][j + 1])
                        };
                    }
                }
                let (mut i, mut j, mut at) = (0, 0, 0);
                while i < n || j < m {
                    let sub = if i < n && j < m {
                        Some(
                            cost[i + 1][j + 1]
                                + if a[i] == b[j] {
                                    0
                                } else if a[i].op == b[j].op {
                                    1
                                } else {
                                    3
                                },
                        )
                    } else {
                        None
                    };
                    if sub == Some(cost[i][j]) {
                        if a[i].op == b[j].op {
                            let keys: std::collections::BTreeSet<_> = a[i]
                                .params
                                .keys()
                                .chain(b[j].params.keys())
                                .cloned()
                                .collect();
                            for key in keys {
                                if a[i].params.get(&key) != b[j].params.get(&key) {
                                    delta.sites.push(EditSite {
                                        before_index: Some(i),
                                        after_index: Some(j),
                                    });
                                    delta.edits.push(AtomicEdit::Parameter {
                                        at,
                                        op: a[i].op.clone(),
                                        before: a[i].params.get(&key).cloned(),
                                        after: b[j].params.get(&key).cloned(),
                                        key,
                                    });
                                }
                            }
                        } else {
                            delta.sites.push(EditSite {
                                before_index: Some(i),
                                after_index: Some(j),
                            });
                            delta.edits.push(AtomicEdit::Replace {
                                at,
                                before: a[i].clone(),
                                after: b[j].clone(),
                            });
                        }
                        i += 1;
                        j += 1;
                        at += 1;
                    } else if i < n && cost[i][j] == 2 + cost[i + 1][j] {
                        delta.sites.push(EditSite {
                            before_index: Some(i),
                            after_index: None,
                        });
                        delta.edits.push(AtomicEdit::Delete {
                            at,
                            before: a[i].clone(),
                        });
                        i += 1;
                    } else {
                        delta.sites.push(EditSite {
                            before_index: None,
                            after_index: Some(j),
                        });
                        delta.edits.push(AtomicEdit::Insert {
                            at,
                            stage: b[j].clone(),
                        });
                        j += 1;
                        at += 1;
                    }
                }
            }
        }
        _ => {
            delta.coverage = "whole_program_only".into();
            delta.sites.push(EditSite {
                before_index: None,
                after_index: None,
            });
            delta.edits.push(AtomicEdit::ReplaceProgram {
                before: parent.identity.clone(),
                after: child.clone(),
            });
        }
    }
    delta.replay()?;
    Ok(delta)
}
