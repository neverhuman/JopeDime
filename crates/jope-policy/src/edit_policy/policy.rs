use anyhow::{ensure, Result};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

use super::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PolicyEditKind {
    ColumnTransform,
    ProgramTransform,
    Eos,
}

/// Backwards-compatible name for policy edit kinds.
pub type EditKind = PolicyEditKind;

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
    pub kind: PolicyEditKind,
    pub site: Option<EditSite>,
    pub operator: Option<EditOperator>,
    pub fidelity: Option<Fidelity>,
    pub stop: bool,
}

impl EditKey {
    pub fn eos() -> Self {
        Self {
            kind: PolicyEditKind::Eos,
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
