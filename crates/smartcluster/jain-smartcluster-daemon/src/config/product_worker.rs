//! Template names accepted by `workers.product_worker_templates`.
//!
//! The set is closed: every typed [`JainWorkload`] known to this build plus
//! `primary_inference`, the July fabric-v2 template that deployed nodes
//! declare but that has no `JainWorkload` variant on this build. Any other
//! name is rejected at parse time.
//!
//! Reserved: this build has no executor hook that routes a template to the
//! product worker. Every `Workload::Jain` still launches `workers.jain_worker`.
//! The product-worker keys are parsed and validated only.

use jain_smartcluster_core::JainWorkload;
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Deserialize, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ProductWorkerTemplate {
    DatasetProfile,
    FeatureExpansion,
    Training,
    ReportGeneration,
    Prediction,
    Export,
    Chat,
    Research,
    ManagedBrowser,
    JopeEncode,
    JopePropose,
    LimeRun,
    LimeBatch,
    /// July fabric-v2 template name; no typed workload exists in this build.
    PrimaryInference,
}

impl ProductWorkerTemplate {
    /// The typed workload this template names, when this build has one.
    pub const fn workload(self) -> Option<JainWorkload> {
        Some(match self {
            Self::DatasetProfile => JainWorkload::DatasetProfile,
            Self::FeatureExpansion => JainWorkload::FeatureExpansion,
            Self::Training => JainWorkload::Training,
            Self::ReportGeneration => JainWorkload::ReportGeneration,
            Self::Prediction => JainWorkload::Prediction,
            Self::Export => JainWorkload::Export,
            Self::Chat => JainWorkload::Chat,
            Self::Research => JainWorkload::Research,
            Self::ManagedBrowser => JainWorkload::ManagedBrowser,
            Self::JopeEncode => JainWorkload::JopeEncode,
            Self::JopePropose => JainWorkload::JopePropose,
            Self::LimeRun => JainWorkload::LimeRun,
            Self::LimeBatch => JainWorkload::LimeBatch,
            Self::PrimaryInference => return None,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every typed workload in this build is a valid product-worker template
    /// under the same snake_case name.
    #[test]
    fn every_typed_workload_has_a_template_name() {
        for template in [
            ProductWorkerTemplate::DatasetProfile,
            ProductWorkerTemplate::FeatureExpansion,
            ProductWorkerTemplate::Training,
            ProductWorkerTemplate::ReportGeneration,
            ProductWorkerTemplate::Prediction,
            ProductWorkerTemplate::Export,
            ProductWorkerTemplate::Chat,
            ProductWorkerTemplate::Research,
            ProductWorkerTemplate::ManagedBrowser,
            ProductWorkerTemplate::JopeEncode,
            ProductWorkerTemplate::JopePropose,
            ProductWorkerTemplate::LimeRun,
            ProductWorkerTemplate::LimeBatch,
        ] {
            let workload = template.workload().expect("typed workload");
            assert_eq!(
                serde_json::to_string(&template).unwrap(),
                serde_json::to_string(&workload).unwrap()
            );
            // Exhaustiveness guard: a new JainWorkload variant must be added here.
            match workload {
                JainWorkload::DatasetProfile
                | JainWorkload::FeatureExpansion
                | JainWorkload::Training
                | JainWorkload::ReportGeneration
                | JainWorkload::Prediction
                | JainWorkload::Export
                | JainWorkload::Chat
                | JainWorkload::Research
                | JainWorkload::ManagedBrowser
                | JainWorkload::JopeEncode
                | JainWorkload::JopePropose
                | JainWorkload::LimeRun
                | JainWorkload::LimeBatch => {}
            }
        }
        assert_eq!(ProductWorkerTemplate::PrimaryInference.workload(), None);
    }
}
