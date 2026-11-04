use serde::Serialize;
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ElementKind {
    Stage,
    Head,
    Carrier,
    Function,
    Binary,
    Variable,
    Policy,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize)]
pub struct ProfileElement {
    pub kind: ElementKind,
    pub name: &'static str,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum TextChoice {
    Value(&'static str),
    Omit,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SearchParamDomain {
    IntRange {
        lo: i64,
        hi: i64,
    },
    FloatRange {
        lo: f64,
        hi: f64,
    },
    IntChoice(Vec<i64>),
    FloatChoice(Vec<f64>),
    TextChoice(Vec<TextChoice>),
    Flag {
        true_numerator: u32,
        denominator: u32,
    },
}

pub type ParamDomain = SearchParamDomain;

pub type DomainRow = BTreeMap<&'static str, SearchParamDomain>;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Limit {
    Caller,
    Fixed(usize),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CountRule {
    EveryStageBeforeHead,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct MaxTransforms {
    pub limit: Limit,
    pub counts: CountRule,
}

#[derive(Clone, Debug, Serialize)]
pub struct ExprStages {
    pub carriers: Vec<&'static str>,
    pub max_expr_stages: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Negation {
    ConstantsOnly,
}

#[derive(Clone, Debug, Serialize)]
pub struct ExprPool {
    pub functions: Vec<&'static str>,
    pub binops: Vec<&'static str>,
    pub variables: Vec<&'static str>,
    pub weight_each: u32,
    pub column_upper_exclusive: usize,
    pub columns_bounded_by_lower_width: bool,
    pub birth_constants: Vec<f64>,
    pub continuous_after_birth: bool,
    pub negation: Negation,
    pub max_nodes: usize,
    pub max_depth: usize,
}

/// Version-two expression edits; draw an arm before choosing its site.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ExpressionArm {
    Point,
    Subtree,
    Hoist,
    Shrink,
    ConstantJitter,
    Wrap,
    InsertBinary,
}

impl ExpressionArm {
    pub const ALL: [Self; 7] = [
        Self::Point,
        Self::Subtree,
        Self::Hoist,
        Self::Shrink,
        Self::ConstantJitter,
        Self::Wrap,
        Self::InsertBinary,
    ];
}

#[derive(Clone, Copy, Debug, Serialize)]
pub struct ExpressionArmWeight {
    pub arm: ExpressionArm,
    pub weight: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum KernelMutationArm {
    ParameterJitter,
    Insert,
    Delete,
    AdjacentSwap,
    HeadReplacement,
    Simplify,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct ArmWeight {
    pub arm: KernelMutationArm,
    pub weight: u32,
}

/// Compatibility alias for the kernel mutation operator arm.
pub type Arm = KernelMutationArm;

/// Revision-three pipeline operators. ExprCross always takes a second parent.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum PipelineArm {
    ParameterJitter,
    Insert,
    Delete,
    AdjacentSwap,
    HeadReplacement,
    Simplify,
    ExprGrow,
    ExprEdit,
    ExprDrop,
    StageCross,
    ExprCross,
}

impl From<KernelMutationArm> for PipelineArm {
    fn from(arm: KernelMutationArm) -> Self {
        match arm {
            KernelMutationArm::ParameterJitter => Self::ParameterJitter,
            KernelMutationArm::Insert => Self::Insert,
            KernelMutationArm::Delete => Self::Delete,
            KernelMutationArm::AdjacentSwap => Self::AdjacentSwap,
            KernelMutationArm::HeadReplacement => Self::HeadReplacement,
            KernelMutationArm::Simplify => Self::Simplify,
        }
    }
}

#[derive(Clone, Copy, Debug, Serialize)]
pub struct PipelineArmWeight {
    pub arm: PipelineArm,
    pub weight: u32,
}

#[derive(Clone, Debug, Serialize)]
pub struct PipelineOperators {
    /// Combined with the six stage weights in edits.drawn, before selecting a site.
    #[serde(rename = "expression_edits")]
    pub expression_mutations: Vec<PipelineArmWeight>,
    #[serde(rename = "splices")]
    pub crossovers: Vec<PipelineArmWeight>,
    pub birth_expression_numerator: u32,
    pub birth_expression_denominator: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum JitterVersion {
    V1,
    V2,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SimplifyFallback {
    ParameterJitter,
    DeleteRandomStage,
}

#[derive(Clone, Debug, Serialize)]
pub struct EditSpec {
    pub drawn: Vec<ArmWeight>,
    pub explicit_arm: bool,
    pub jitter: JitterVersion,
    pub transform_share_numerator: u32,
    pub transform_share_denominator: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub drawn_simplify_fallback: Option<SimplifyFallback>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub explicit_simplify_fallback: Option<SimplifyFallback>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SpliceKind {
    OnePointStages,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct SpliceSpec {
    pub kind: SpliceKind,
    pub truncate_to_limit: bool,
    pub head_ratio_numerator: u32,
    pub head_ratio_denominator: u32,
}

#[derive(Clone, Debug, Serialize)]
pub struct SearchProfile {
    pub id: &'static str,
    pub revision: u32,
    #[serde(skip)]
    pub description: &'static str,
    pub stages: Vec<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub expansion: Option<Vec<&'static str>>,
    pub heads: Vec<&'static str>,
    pub domains: BTreeMap<&'static str, DomainRow>,
    pub redraw_domains: BTreeMap<&'static str, DomainRow>,
    pub max_transforms: MaxTransforms,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub expr_stages: Option<ExprStages>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub expr: Option<ExprPool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub expression_edits: Option<Vec<ExpressionArmWeight>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pipeline_operators: Option<PipelineOperators>,
    pub edits: EditSpec,
    pub splice: SpliceSpec,
}

impl SearchProfile {
    pub fn canonical_json(&self) -> String {
        serde_json::to_string(self).expect("profile serializes")
    }
    pub fn sha256(&self) -> String {
        let digest = Sha256::digest(self.canonical_json().as_bytes());
        let mut out = String::with_capacity(64);
        for byte in digest {
            use std::fmt::Write;
            write!(&mut out, "{byte:02x}").expect("string write");
        }
        out
    }
    /// Each element appears once under its kind and name, irrespective of draw weight.
    pub fn elements(&self) -> Vec<ProfileElement> {
        let mut seen = BTreeSet::new();
        let mut out = Vec::new();
        let mut add = |kind, name| {
            if seen.insert((kind, name)) {
                out.push(ProfileElement { kind, name });
            }
        };
        for &name in &self.stages {
            add(ElementKind::Stage, name);
        }
        if let Some(expansion) = &self.expansion {
            for &name in expansion {
                add(ElementKind::Stage, name);
            }
        }
        for &name in &self.heads {
            add(ElementKind::Head, name);
        }
        if let Some(stages) = &self.expr_stages {
            for &name in &stages.carriers {
                add(ElementKind::Carrier, name);
            }
        }
        if let Some(expr) = &self.expr {
            for &name in &expr.functions {
                add(ElementKind::Function, name);
            }
            for &name in &expr.binops {
                add(ElementKind::Binary, name);
            }
            for &name in &expr.variables {
                add(ElementKind::Variable, name);
            }
            for name in ["columns", "constants", "negation"] {
                add(ElementKind::Policy, name);
            }
        }
        out
    }
}
