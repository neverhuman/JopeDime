//! Draft, versioned typed-edit policy ABI.
//!
//! This is an additive challenger. It does not alter the 401-token grammar or accepted model
//! identities. I08's graph/effect types and I25's fit-closure adapters are still draft inputs;
//! `ResidualClosure` records the corresponding binding without claiming those adapters exist.

use anyhow::{ensure, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};

pub const EDIT_POLICY_SCHEMA: &str = "jope.typed-edit-policy.v1";
pub const CONTEXT_ABI_SCHEMA: &str = "jope.context-abi.v2";
pub const EDIT_ARCHITECTURE: &str = "N1_factorized_typed_mixture_v1";
pub const CONTEXT_FEATURES: usize = 2048;
pub const MAX_PROGRAM_TOKENS: usize = 128;
pub const MAX_SCHEMA_COLUMNS: usize = 4096;
pub const REFERENCE_BACKEND: &str = "cpu_f64_reference_v1";
const DIGEST_LEN: usize = 64;

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ColumnDescriptor {
    pub key: String,
    pub kind: String,
}

/// I25 closure identity carried as draft metadata until the owning residual adapter is accepted.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ResidualClosure {
    pub producer_sha256: String,
    pub fold_sha256: String,
    pub row_order_sha256: String,
    pub model_state_sha256: String,
    pub feature_selection_sha256: String,
    pub role_ancestry_sha256: String,
    pub diagnostics_charged_micros: u64,
}

/// Complete candidate context identity. Schema order is significant in its digest; pointer
/// remapping uses stable column keys and checks equivalent schemas before changing the index.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ContextAbiV2 {
    pub schema: String,
    pub backend: String,
    pub schema_sha256: String,
    pub columns: Vec<ColumnDescriptor>,
    pub selected_column_keys: Vec<String>,
    pub feature_selection_sha256: String,
    pub fold_sha256: String,
    pub row_order_sha256: String,
    /// Required by time-lag edits; absent means row order has no declared temporal meaning.
    pub ordered_time_semantics: Option<String>,
    pub model_state_sha256: String,
    pub program_tokens: Vec<u32>,
    pub vocabulary_size: u32,
    pub max_tokens: usize,
    pub context_features: Vec<f32>,
    pub residual: ResidualClosure,
}

impl ContextAbiV2 {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        columns: Vec<ColumnDescriptor>,
        selected_column_keys: Vec<String>,
        fold_sha256: String,
        row_order_sha256: String,
        ordered_time_semantics: Option<String>,
        model_state_sha256: String,
        program_tokens: Vec<u32>,
        vocabulary_size: u32,
        residual_producer_sha256: String,
        role_ancestry_sha256: String,
        diagnostics_charged_micros: u64,
    ) -> Result<Self> {
        let selected_column_keys = sorted_unique(selected_column_keys)?;
        let schema_sha256 = schema_digest(&columns)?;
        let feature_selection_sha256 = digest_json(&selected_column_keys)?;
        let residual = ResidualClosure {
            producer_sha256: residual_producer_sha256,
            fold_sha256: fold_sha256.clone(),
            row_order_sha256: row_order_sha256.clone(),
            model_state_sha256: model_state_sha256.clone(),
            feature_selection_sha256: feature_selection_sha256.clone(),
            role_ancestry_sha256,
            diagnostics_charged_micros,
        };
        let value = Self {
            schema: CONTEXT_ABI_SCHEMA.into(),
            backend: REFERENCE_BACKEND.into(),
            schema_sha256,
            columns,
            selected_column_keys,
            feature_selection_sha256,
            fold_sha256,
            row_order_sha256,
            ordered_time_semantics,
            model_state_sha256,
            program_tokens,
            vocabulary_size,
            max_tokens: MAX_PROGRAM_TOKENS,
            context_features: vec![0.0; CONTEXT_FEATURES],
            residual,
        };
        value.validate()?;
        Ok(value)
    }

    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.schema == CONTEXT_ABI_SCHEMA,
            "unsupported context ABI schema"
        );
        ensure!(
            self.backend == REFERENCE_BACKEND,
            "unsupported edit policy backend"
        );
        ensure!(
            self.columns.len() <= MAX_SCHEMA_COLUMNS && !self.columns.is_empty(),
            "unsupported schema column count"
        );
        ensure!(
            schema_digest(&self.columns)? == self.schema_sha256,
            "context schema digest mismatch"
        );
        let mut columns = BTreeMap::new();
        for column in &self.columns {
            ensure!(
                !column.key.is_empty() && !column.kind.is_empty(),
                "empty schema column identity"
            );
            ensure!(
                columns.insert(&column.key, &column.kind).is_none(),
                "duplicate schema column key"
            );
        }
        ensure!(
            self.selected_column_keys
                .windows(2)
                .all(|pair| pair[0] < pair[1]),
            "selected columns must be unique and sorted"
        );
        ensure!(
            self.selected_column_keys
                .iter()
                .all(|key| columns.contains_key(key)),
            "feature selection names a column outside the schema"
        );
        ensure!(
            digest_json(&self.selected_column_keys)? == self.feature_selection_sha256,
            "feature selection digest mismatch"
        );
        for digest in [
            &self.fold_sha256,
            &self.row_order_sha256,
            &self.model_state_sha256,
            &self.residual.producer_sha256,
            &self.residual.role_ancestry_sha256,
        ] {
            ensure!(is_digest(digest), "invalid context identity digest");
        }
        ensure!(
            self.max_tokens > 0 && self.max_tokens <= MAX_PROGRAM_TOKENS,
            "unsupported token limit"
        );
        ensure!(
            !self.program_tokens.is_empty()
                && self.program_tokens.len() <= self.max_tokens
                && self
                    .program_tokens
                    .iter()
                    .all(|token| *token < self.vocabulary_size),
            "unsupported program length or token"
        );
        ensure!(self.vocabulary_size > 0, "empty edit vocabulary");
        ensure!(
            self.context_features.len() == CONTEXT_FEATURES
                && self.context_features.iter().all(|value| value.is_finite()),
            "unsupported context feature length or value"
        );
        ensure!(
            self.residual.fold_sha256 == self.fold_sha256
                && self.residual.row_order_sha256 == self.row_order_sha256
                && self.residual.model_state_sha256 == self.model_state_sha256
                && self.residual.feature_selection_sha256 == self.feature_selection_sha256,
            "residual closure differs from the current fold/context"
        );
        if let Some(semantics) = &self.ordered_time_semantics {
            ensure!(!semantics.trim().is_empty(), "empty ordered-time semantics");
        }
        Ok(())
    }

    pub fn column_pointer(&self, index: usize) -> Result<ColumnPointer> {
        self.validate()?;
        let column = self
            .columns
            .get(index)
            .ok_or_else(|| anyhow::anyhow!("column index outside schema"))?;
        Ok(ColumnPointer {
            schema_sha256: self.schema_sha256.clone(),
            index,
            key: column.key.clone(),
        })
    }

    pub fn remap_pointer(
        &self,
        pointer: &ColumnPointer,
        target: &ContextAbiV2,
    ) -> Result<ColumnPointer> {
        self.validate()?;
        target.validate()?;
        pointer.validate_against(self)?;
        let source: BTreeMap<_, _> = self.columns.iter().map(|c| (&c.key, &c.kind)).collect();
        let destination: BTreeMap<_, _> =
            target.columns.iter().map(|c| (&c.key, &c.kind)).collect();
        ensure!(
            source == destination,
            "column remap requires the same complete schema"
        );
        let index = target
            .columns
            .iter()
            .position(|column| column.key == pointer.key)
            .ok_or_else(|| anyhow::anyhow!("column absent from target schema"))?;
        ensure!(
            self.fold_sha256 == target.fold_sha256
                && self.row_order_sha256 == target.row_order_sha256
                && self.model_state_sha256 == target.model_state_sha256,
            "column remap changed fold, row order, or model state"
        );
        ensure!(
            self.feature_selection_sha256 == target.feature_selection_sha256
                && self.residual == target.residual
                && target
                    .selected_column_keys
                    .binary_search(&pointer.key)
                    .is_ok(),
            "column remap changed feature selection or residual closure"
        );
        Ok(ColumnPointer {
            schema_sha256: target.schema_sha256.clone(),
            index,
            key: pointer.key.clone(),
        })
    }
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ColumnPointer {
    pub schema_sha256: String,
    pub index: usize,
    pub key: String,
}

impl ColumnPointer {
    pub fn validate_against(&self, context: &ContextAbiV2) -> Result<()> {
        ensure!(
            self.schema_sha256 == context.schema_sha256,
            "column pointer schema mismatch"
        );
        let column = context
            .columns
            .get(self.index)
            .ok_or_else(|| anyhow::anyhow!("column pointer outside schema"))?;
        ensure!(column.key == self.key, "column pointer index/key mismatch");
        ensure!(
            context
                .selected_column_keys
                .binary_search(&self.key)
                .is_ok(),
            "column not selected by context"
        );
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EditKind {
    ColumnTransform,
    ProgramTransform,
    Eos,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(tag = "site", content = "value", rename_all = "snake_case")]
pub enum EditSite {
    Column(ColumnPointer),
    Node(usize),
    Program,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(tag = "operator", content = "value", rename_all = "snake_case")]
pub enum EditOperator {
    Replace,
    Insert,
    Remove,
    TimeLag,
    Versioned(u16),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Fidelity {
    Exact,
    Approximate,
}

/// Discrete factors of N1. EOS is represented by one legal atom with `stop = true` and no
/// site/operator/fidelity; edit atoms have all factors present and `stop = false`.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EditKey {
    pub kind: EditKind,
    pub site: Option<EditSite>,
    pub operator: Option<EditOperator>,
    pub fidelity: Option<Fidelity>,
    pub stop: bool,
}

impl EditKey {
    pub fn eos() -> Self {
        Self {
            kind: EditKind::Eos,
            site: None,
            operator: None,
            fidelity: None,
            stop: true,
        }
    }

    pub fn validate(&self, context: &ContextAbiV2) -> Result<()> {
        match self.kind {
            EditKind::Eos => ensure!(
                self.stop
                    && self.site.is_none()
                    && self.operator.is_none()
                    && self.fidelity.is_none(),
                "invalid EOS action factors"
            ),
            _ => {
                ensure!(!self.stop, "edit action cannot set stop");
                ensure!(
                    self.site.is_some() && self.operator.is_some() && self.fidelity.is_some(),
                    "incomplete edit factors"
                );
                if let Some(EditSite::Column(pointer)) = &self.site {
                    pointer.validate_against(context)?;
                }
                if let Some(EditSite::Node(node)) = &self.site {
                    ensure!(
                        *node < context.program_tokens.len(),
                        "edit node outside program"
                    );
                }
                if self.operator == Some(EditOperator::TimeLag) {
                    ensure!(
                        context.ordered_time_semantics.is_some(),
                        "time-lag edit requires declared row semantics"
                    );
                }
            }
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NormalDensity {
    pub mean: f64,
    pub standard_deviation: f64,
}

impl NormalDensity {
    fn validate(&self) -> Result<()> {
        ensure!(
            self.mean.is_finite()
                && self.standard_deviation.is_finite()
                && self.standard_deviation > 0.0,
            "invalid continuous parameter density"
        );
        Ok(())
    }

    fn pdf(&self, value: f64) -> f64 {
        let z = (value - self.mean) / self.standard_deviation;
        (-0.5 * z * z).exp() / (self.standard_deviation * (2.0 * std::f64::consts::PI).sqrt())
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PolicyComponent {
    pub weight: f64,
    pub key: EditKey,
    /// `None` is a discrete atom; `Some` is a continuous Lebesgue density.
    pub parameter: Option<NormalDensity>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EditPolicy {
    pub schema: String,
    pub architecture: String,
    pub context: ContextAbiV2,
    pub components: Vec<PolicyComponent>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ActionMask {
    pub allowed: BTreeSet<EditKey>,
}

impl ActionMask {
    pub fn new(keys: impl IntoIterator<Item = EditKey>) -> Result<Self> {
        let allowed: BTreeSet<_> = keys.into_iter().collect();
        ensure!(!allowed.is_empty(), "edit action mask is empty");
        Ok(Self { allowed })
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EditAction {
    pub key: EditKey,
    pub parameter: Option<f64>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ProbabilityMeasure {
    Mass(f64),
    Density(f64),
}

impl EditPolicy {
    pub fn new(context: ContextAbiV2, components: Vec<PolicyComponent>) -> Result<Self> {
        let value = Self {
            schema: EDIT_POLICY_SCHEMA.into(),
            architecture: EDIT_ARCHITECTURE.into(),
            context,
            components,
        };
        value.validate()?;
        Ok(value)
    }

    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.schema == EDIT_POLICY_SCHEMA,
            "unsupported edit policy schema"
        );
        ensure!(
            self.architecture == EDIT_ARCHITECTURE,
            "unsupported edit architecture"
        );
        self.context.validate()?;
        ensure!(!self.components.is_empty(), "edit policy has no components");
        let mut total = 0.0;
        for component in &self.components {
            ensure!(
                component.weight.is_finite() && component.weight > 0.0,
                "invalid mixture weight"
            );
            component.key.validate(&self.context)?;
            if let Some(density) = &component.parameter {
                density.validate()?;
            }
            ensure!(
                component.key.kind != EditKind::Eos || component.parameter.is_none(),
                "EOS cannot carry a continuous parameter"
            );
            total += component.weight;
        }
        ensure!(
            (total - 1.0).abs() <= 1e-9,
            "edit mixture weights are not normalized"
        );
        Ok(())
    }

    pub fn action_mask(&self) -> Result<ActionMask> {
        self.validate()?;
        ActionMask::new(
            self.components
                .iter()
                .map(|component| component.key.clone()),
        )
    }

    /// Deterministic draw interface: the caller supplies a unit mixture draw and a standard
    /// normal draw, making sampled actions exactly replayable from their logged inputs.
    pub fn sample(
        &self,
        mask: &ActionMask,
        mixture_draw: f64,
        normal_draw: f64,
    ) -> Result<EditAction> {
        let components = self.legal_components(mask)?;
        ensure!(
            mixture_draw.is_finite() && (0.0..1.0).contains(&mixture_draw),
            "invalid mixture draw"
        );
        ensure!(normal_draw.is_finite(), "invalid normal draw");
        let total: f64 = components.iter().map(|component| component.weight).sum();
        ensure!(
            total.is_finite() && total > 0.0,
            "legal edit mixture has zero mass"
        );
        let needle = mixture_draw * total;
        let mut cumulative = 0.0;
        let selected = components
            .iter()
            .find(|component| {
                cumulative += component.weight;
                needle < cumulative
            })
            .unwrap_or_else(|| components.last().expect("nonempty after validation"));
        let parameter = selected
            .parameter
            .as_ref()
            .map(|density| density.mean + density.standard_deviation * normal_draw);
        Ok(EditAction {
            key: selected.key.clone(),
            parameter,
        })
    }

    /// Score on the mixed counting/Lebesgue measure: EOS and discrete edits return mass;
    /// parameterized edits return a continuous density. Mixture weights are renormalized over
    /// the exact legal mask used by `sample`.
    pub fn score(&self, mask: &ActionMask, action: &EditAction) -> Result<ProbabilityMeasure> {
        action.key.validate(&self.context)?;
        ensure!(
            action.parameter.is_none_or(f64::is_finite),
            "non-finite edit parameter"
        );
        let components = self.legal_components(mask)?;
        ensure!(
            mask.allowed.contains(&action.key),
            "action is illegal under this mask"
        );
        let normalizer: f64 = components.iter().map(|component| component.weight).sum();
        ensure!(
            normalizer > 0.0 && normalizer.is_finite(),
            "legal edit mixture has zero mass"
        );
        let value = components
            .iter()
            .filter(|component| component.key == action.key)
            .filter_map(
                |component| match (&action.parameter, &component.parameter) {
                    (None, None) => Some(component.weight / normalizer),
                    (Some(value), Some(density)) => {
                        Some(component.weight / normalizer * density.pdf(*value))
                    }
                    _ => None,
                },
            )
            .sum::<f64>();
        ensure!(
            value.is_finite() && value > 0.0,
            "action has zero policy support"
        );
        Ok(if action.parameter.is_some() {
            ProbabilityMeasure::Density(value)
        } else {
            ProbabilityMeasure::Mass(value)
        })
    }

    fn legal_components<'a>(&'a self, mask: &ActionMask) -> Result<Vec<&'a PolicyComponent>> {
        self.validate()?;
        ensure!(!mask.allowed.is_empty(), "edit action mask is empty");
        for key in &mask.allowed {
            key.validate(&self.context)?;
        }
        let components: Vec<_> = self
            .components
            .iter()
            .filter(|component| mask.allowed.contains(&component.key))
            .collect();
        ensure!(
            !components.is_empty(),
            "edit mask removes all policy support"
        );
        Ok(components)
    }
}

fn sorted_unique(mut values: Vec<String>) -> Result<Vec<String>> {
    ensure!(
        values.iter().all(|value| !value.is_empty()),
        "empty selected column key"
    );
    values.sort();
    ensure!(
        values.windows(2).all(|pair| pair[0] != pair[1]),
        "duplicate selected column key"
    );
    Ok(values)
}

fn schema_digest(columns: &[ColumnDescriptor]) -> Result<String> {
    digest_json(&(CONTEXT_ABI_SCHEMA, columns))
}

fn digest_json(value: &impl Serialize) -> Result<String> {
    let bytes = serde_json::to_vec(value)?;
    Ok(format!("{:x}", Sha256::digest(bytes)))
}

fn is_digest(value: &str) -> bool {
    value.len() == DIGEST_LEN
        && value
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn digest(byte: u8) -> String {
        format!("{byte:02x}").repeat(32)
    }

    fn context(columns: Vec<ColumnDescriptor>) -> ContextAbiV2 {
        let keys = columns.iter().map(|column| column.key.clone()).collect();
        ContextAbiV2::new(
            columns,
            keys,
            digest(1),
            digest(2),
            None,
            digest(3),
            vec![1, 7, 2],
            64,
            digest(4),
            digest(5),
            1200,
        )
        .unwrap()
    }

    fn columns(count: usize) -> Vec<ColumnDescriptor> {
        (0..count)
            .map(|index| ColumnDescriptor {
                key: format!("feature-{index:03}"),
                kind: "continuous".into(),
            })
            .collect()
    }

    fn column_edit(pointer: ColumnPointer) -> EditKey {
        EditKey {
            kind: EditKind::ColumnTransform,
            site: Some(EditSite::Column(pointer)),
            operator: Some(EditOperator::Replace),
            fidelity: Some(Fidelity::Exact),
            stop: false,
        }
    }

    #[test]
    fn columns_17_and_63_remap_equivariantly_under_schema_permutation() {
        let source = context(columns(64));
        let mut permuted_columns = columns(64);
        permuted_columns.swap(17, 63);
        let target = context(permuted_columns);
        let pointer17 = source.column_pointer(17).unwrap();
        let pointer63 = source.column_pointer(63).unwrap();
        let mapped17 = source.remap_pointer(&pointer17, &target).unwrap();
        let mapped63 = source.remap_pointer(&pointer63, &target).unwrap();
        assert_eq!(pointer17.key, mapped17.key);
        assert_eq!(pointer63.key, mapped63.key);
        assert_eq!(mapped17.index, 63);
        assert_eq!(mapped63.index, 17);
        mapped17.validate_against(&target).unwrap();
        mapped63.validate_against(&target).unwrap();

        let source_policy = EditPolicy::new(
            source.clone(),
            vec![PolicyComponent {
                weight: 1.0,
                key: column_edit(pointer17.clone()),
                parameter: None,
            }],
        )
        .unwrap();
        let target_policy = EditPolicy::new(
            target.clone(),
            vec![PolicyComponent {
                weight: 1.0,
                key: column_edit(mapped17.clone()),
                parameter: None,
            }],
        )
        .unwrap();
        let source_action = EditAction {
            key: column_edit(pointer17),
            parameter: None,
        };
        let target_action = EditAction {
            key: column_edit(mapped17),
            parameter: None,
        };
        assert_eq!(
            source_policy
                .score(&source_policy.action_mask().unwrap(), &source_action)
                .unwrap(),
            target_policy
                .score(&target_policy.action_mask().unwrap(), &target_action)
                .unwrap()
        );
    }

    #[test]
    fn eos_mass_and_continuous_density_match_sampler_and_scorer() {
        let context = context(columns(64));
        let pointer = context.column_pointer(17).unwrap();
        let edit = column_edit(pointer);
        let policy = EditPolicy::new(
            context,
            vec![
                PolicyComponent {
                    weight: 0.2,
                    key: EditKey::eos(),
                    parameter: None,
                },
                PolicyComponent {
                    weight: 0.3,
                    key: EditKey {
                        kind: EditKind::ProgramTransform,
                        site: Some(EditSite::Program),
                        operator: Some(EditOperator::Insert),
                        fidelity: Some(Fidelity::Approximate),
                        stop: false,
                    },
                    parameter: None,
                },
                PolicyComponent {
                    weight: 0.5,
                    key: edit.clone(),
                    parameter: Some(NormalDensity {
                        mean: 0.4,
                        standard_deviation: 0.5,
                    }),
                },
            ],
        )
        .unwrap();
        let mask = policy.action_mask().unwrap();
        let eos = policy.sample(&mask, 0.1, 0.0).unwrap();
        assert_eq!(
            policy.score(&mask, &eos).unwrap(),
            ProbabilityMeasure::Mass(0.2)
        );
        let discrete = policy.sample(&mask, 0.3, 0.0).unwrap();
        assert!(discrete.parameter.is_none());
        assert_eq!(
            policy.score(&mask, &discrete).unwrap(),
            ProbabilityMeasure::Mass(0.3)
        );
        let continuous = policy.sample(&mask, 0.75, 0.5).unwrap();
        assert!((continuous.parameter.unwrap() - 0.65).abs() < 1e-12);
        let scored = policy.score(&mask, &continuous).unwrap();
        let expected = 0.5 * (-0.125_f64).exp() / (0.5 * (2.0 * std::f64::consts::PI).sqrt());
        assert!(
            matches!(scored, ProbabilityMeasure::Density(value) if (value - expected).abs() < 1e-12)
        );
    }

    #[test]
    fn unsupported_backend_length_and_schema_are_rejected_before_policy_use() {
        let mut bad_backend = context(columns(64));
        bad_backend.backend = "cuda_unqualified".into();
        assert!(bad_backend
            .validate()
            .unwrap_err()
            .to_string()
            .contains("backend"));
        let mut bad_length = context(columns(64));
        bad_length.program_tokens = vec![1; MAX_PROGRAM_TOKENS + 1];
        assert!(bad_length
            .validate()
            .unwrap_err()
            .to_string()
            .contains("length"));
        let mut bad_schema = context(columns(64));
        bad_schema.schema_sha256 = digest(9);
        assert!(bad_schema
            .validate()
            .unwrap_err()
            .to_string()
            .contains("schema digest"));
    }

    #[test]
    fn illegal_or_zero_support_actions_and_undeclared_time_edits_refuse() {
        let context = context(columns(64));
        let pointer = context.column_pointer(63).unwrap();
        let edit = column_edit(pointer);
        let policy = EditPolicy::new(
            context,
            vec![PolicyComponent {
                weight: 1.0,
                key: edit.clone(),
                parameter: None,
            }],
        )
        .unwrap();
        let wrong_mask = ActionMask::new([EditKey::eos()]).unwrap();
        assert!(policy.sample(&wrong_mask, 0.5, 0.0).is_err());
        assert!(policy
            .score(
                &policy.action_mask().unwrap(),
                &EditAction {
                    key: EditKey::eos(),
                    parameter: None
                }
            )
            .is_err());
        let time = EditKey {
            kind: EditKind::ProgramTransform,
            site: Some(EditSite::Program),
            operator: Some(EditOperator::TimeLag),
            fidelity: Some(Fidelity::Exact),
            stop: false,
        };
        assert!(time
            .validate(&policy.context)
            .unwrap_err()
            .to_string()
            .contains("row semantics"));
    }
}
