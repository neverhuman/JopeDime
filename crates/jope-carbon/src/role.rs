//! Cell roles, effort budgets, and Free / Pro / Enterprise admission limits for Carbon.

use std::str::FromStr;

use serde::{Deserialize, Serialize};

use crate::error::{CarbonError, CarbonResult};

/// `CarbonRole` is an enum for the execution role selected by `CARBON_ROLE` or CLI dispatch.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CarbonRole {
    /// Regional public ingress, auth, rate limiting, and S3 presigning (`128 MB`–`256 MB` zip).
    Edge,
    /// Sealed neural Wave-0 conditioning and multi-wave coordinator (`3,008 MB` container).
    EngineOrchestrate,
    /// Stateless batch candidate evaluation worker (`512 MB`–`7,076 MB` zip).
    EvalBatch,
    /// Self-hosted VPC / on-premises single-binary server (`carbon serve`).
    EnterpriseServe,
    /// Local multi-cell Lambda Runtime API + S3 CAS emulator (`carbon emu`).
    LocalEmulator,
}

impl CarbonRole {
    /// Canonical string label for environment and structured telemetry logs.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Edge => "edge",
            Self::EngineOrchestrate => "engine",
            Self::EvalBatch => "eval",
            Self::EnterpriseServe => "serve",
            Self::LocalEmulator => "emu",
        }
    }
}

impl FromStr for CarbonRole {
    type Err = CarbonError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.trim() {
            "edge" => Ok(Self::Edge),
            "engine" | "orchestrate" => Ok(Self::EngineOrchestrate),
            "eval" => Ok(Self::EvalBatch),
            "serve" => Ok(Self::EnterpriseServe),
            "emu" => Ok(Self::LocalEmulator),
            other => Err(CarbonError::InvalidRoleOrCommand {
                value: other.to_string(),
            }),
        }
    }
}

/// `CarbonEffortTier` is an enum for the four Carbon search effort budgets (`fast`..`ultra`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum CarbonEffortTier {
    /// Single Wave-0 pass (neural seed + fast baseline screen).
    Fast,
    /// Multi-wave search (`4` waves, `64` candidates/wave, `60 s` wall ceiling).
    Medium,
    /// Deep multi-wave search (`12` waves, `192` candidates/wave, `240 s` wall ceiling).
    High,
    /// Extended multi-wave search (`32` waves, `384` candidates/wave, resumable across `840 s` legs).
    Ultra,
}

impl CarbonEffortTier {
    /// Canonical lowercase tier identifier.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Fast => "fast",
            Self::Medium => "medium",
            Self::High => "high",
            Self::Ultra => "ultra",
        }
    }

    /// Planned wave count for this effort tier.
    pub fn planned_waves(self) -> u32 {
        match self {
            Self::Fast => 1,
            Self::Medium => 4,
            Self::High => 12,
            Self::Ultra => 32,
        }
    }

    /// Candidate count evaluated per wave for this effort tier.
    pub fn candidates_per_wave(self) -> u32 {
        match self {
            Self::Fast => 32,
            Self::Medium => 64,
            Self::High => 192,
            Self::Ultra => 384,
        }
    }

    /// Single-invocation Lambda lease ceiling in seconds (`840 s` = `14 min`, leaving `60 s` drain margin before AWS's `900 s` limit).
    pub fn lease_ceiling_secs(self) -> u32 {
        match self {
            Self::Fast => 25,
            Self::Medium => 90,
            Self::High => 300,
            Self::Ultra => 840,
        }
    }
}

impl FromStr for CarbonEffortTier {
    type Err = CarbonError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.trim() {
            "fast" => Ok(Self::Fast),
            "medium" => Ok(Self::Medium),
            "high" => Ok(Self::High),
            "ultra" => Ok(Self::Ultra),
            other => Err(CarbonError::InvalidRoleOrCommand {
                value: other.to_string(),
            }),
        }
    }
}

/// `CarbonProductTier` is an enum for the commercial admission tier (`Free`, `Pro`, `Enterprise`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum CarbonProductTier {
    /// Anonymous / free tier (`<= 5,000` rows, `<= 64` features, `fast`/`medium`).
    Free,
    /// Authenticated Pro cloud tier (`<= 100,000` rows, `<= 512` features, `fast`..`ultra`).
    Pro,
    /// VPC / on-premises Enterprise server (`no_limit()` on rows and features, `fast`..`ultra`).
    Enterprise,
}

impl CarbonProductTier {
    /// Construct the Enterprise tier with no artificial row or feature cap.
    pub const fn no_limit() -> Self {
        Self::Enterprise
    }

    /// Canonical tier name.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Free => "free",
            Self::Pro => "pro",
            Self::Enterprise => "enterprise",
        }
    }

    /// Optional row cap (`None` for `Enterprise`).
    pub const fn max_rows(self) -> Option<usize> {
        match self {
            Self::Free => Some(5_000),
            Self::Pro => Some(100_000),
            Self::Enterprise => None,
        }
    }

    /// Optional feature column cap (`None` for `Enterprise`).
    pub const fn max_features(self) -> Option<usize> {
        match self {
            Self::Free => Some(64),
            Self::Pro => Some(512),
            Self::Enterprise => None,
        }
    }

    /// Validate that `(total_rows, features, effort)` is admitted under this product tier.
    pub fn validate_admission(
        self,
        total_rows: usize,
        features: usize,
        effort: CarbonEffortTier,
    ) -> CarbonResult<()> {
        if total_rows < 2 || features == 0 {
            return Err(CarbonError::InvalidPlanInput {
                reason: "dataset requires total_rows >= 2 and features >= 1",
            });
        }
        if let Some(limit) = self.max_rows() {
            if total_rows > limit {
                return Err(CarbonError::RowLimitExceeded {
                    tier: self.as_str(),
                    rows: total_rows,
                    limit,
                });
            }
        }
        if let Some(limit) = self.max_features() {
            if features > limit {
                return Err(CarbonError::FeatureLimitExceeded {
                    tier: self.as_str(),
                    features,
                    limit,
                });
            }
        }
        if self == Self::Free && matches!(effort, CarbonEffortTier::High | CarbonEffortTier::Ultra)
        {
            return Err(CarbonError::EffortTierNotPermitted {
                product_tier: self.as_str(),
                effort: effort.as_str(),
            });
        }
        Ok(())
    }
}
