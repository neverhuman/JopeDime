use super::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Best5ArmV1 {
    EvolvabilityUcbLineages,
    CrossFittedResidualDlgp,
    SparseOofPortfolioGp,
    HeadFamilyMapElites,
    SurrogateUcbOffspringSearch,
    ConfidenceGpControl,
}

impl Best5ArmV1 {
    pub const EXPERIMENTS: [Self; 5] = [
        Self::EvolvabilityUcbLineages,
        Self::CrossFittedResidualDlgp,
        Self::SparseOofPortfolioGp,
        Self::HeadFamilyMapElites,
        Self::SurrogateUcbOffspringSearch,
    ];
    pub const ALL: [Self; 6] = [
        Self::EvolvabilityUcbLineages,
        Self::CrossFittedResidualDlgp,
        Self::SparseOofPortfolioGp,
        Self::HeadFamilyMapElites,
        Self::SurrogateUcbOffspringSearch,
        Self::ConfidenceGpControl,
    ];

    pub fn is_experiment(self) -> bool {
        self != Self::ConfidenceGpControl
    }

    pub fn simplicity_order(self) -> usize {
        match self {
            Self::EvolvabilityUcbLineages => 0,
            Self::HeadFamilyMapElites => 1,
            Self::SurrogateUcbOffspringSearch => 2,
            Self::CrossFittedResidualDlgp => 3,
            Self::SparseOofPortfolioGp => 4,
            Self::ConfidenceGpControl => 5,
        }
    }
}
