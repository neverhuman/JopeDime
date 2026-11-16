//! Typed failures and agent-friendly repair guidance for the Carbon execution plane.

use std::fmt;

/// Where the `jope-carbon` runtime, planner, sealing, and tier rules are documented.
pub const CARBON_DOCS_URL: &str = "crates/jope-carbon/README.md";

/// `CarbonError` is an enum for the Carbon runtime failure contract in `jope-carbon`: every tier
/// refusal, planner input error, checkpoint/user-envelope authentication failure, capability auth
/// error, or CLI argument refusal is reported as a typed value without panicking.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CarbonError {
    /// Dataset row count exceeds the active `CarbonProductTier` ceiling.
    RowLimitExceeded {
        tier: &'static str,
        rows: usize,
        limit: usize,
    },
    /// Dataset feature count exceeds the active `CarbonProductTier` ceiling.
    FeatureLimitExceeded {
        tier: &'static str,
        features: usize,
        limit: usize,
    },
    /// Requested effort tier is not permitted on the active `CarbonProductTier`.
    EffortTierNotPermitted {
        product_tier: &'static str,
        effort: &'static str,
    },
    /// Dataset or planner dimensions are zero or out of range.
    InvalidPlanInput { reason: &'static str },
    /// Dataset working-set footprint exceeds the largest stateless Lambda worker class (`7,076 MB`).
    WorkingSetExceedsLambdaClass {
        required_mb: usize,
        max_lambda_mb: usize,
    },
    /// `AES-256-GCM` checkpoint sealing or opening failed authentication, magic, domain, or context binding.
    SealAuthenticationFailed { reason: &'static str },
    /// Per-user zero-knowledge `CUEN\x01` or `CUDK\x01` envelope encryption/decryption failed.
    UserEnvelopeFailed { reason: &'static str },
    /// Capability token or tenant identity verification failed at the Carbon edge.
    CapabilityAuthFailed { reason: &'static str },
    /// Unknown or malformed `CARBON_ROLE` / CLI argument.
    InvalidRoleOrCommand { value: String },
}

impl fmt::Display for CarbonError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::RowLimitExceeded { tier, rows, limit } => {
                write!(
                    f,
                    "{tier} tier admits at most {limit} rows (received {rows})"
                )
            }
            Self::FeatureLimitExceeded {
                tier,
                features,
                limit,
            } => {
                write!(
                    f,
                    "{tier} tier admits at most {limit} feature columns (received {features})"
                )
            }
            Self::EffortTierNotPermitted {
                product_tier,
                effort,
            } => {
                write!(
                    f,
                    "effort tier `{effort}` is not available on the {product_tier} product tier"
                )
            }
            Self::InvalidPlanInput { reason } => {
                write!(f, "invalid Carbon execution plan input: {reason}")
            }
            Self::WorkingSetExceedsLambdaClass {
                required_mb,
                max_lambda_mb,
            } => {
                write!(
                    f,
                    "estimated working set ({required_mb} MB) exceeds maximum stateless Lambda class ({max_lambda_mb} MB); use Enterprise `carbon serve`"
                )
            }
            Self::SealAuthenticationFailed { reason } => {
                write!(f, "sealed checkpoint authentication failed: {reason}")
            }
            Self::UserEnvelopeFailed { reason } => {
                write!(f, "per-user CUEN/CUDK envelope error: {reason}")
            }
            Self::CapabilityAuthFailed { reason } => {
                write!(f, "Carbon capability authentication refused: {reason}")
            }
            Self::InvalidRoleOrCommand { value } => {
                write!(f, "unrecognized Carbon role or subcommand `{value}`")
            }
        }
    }
}

impl std::error::Error for CarbonError {}

/// `CarbonResult` is the result alias used across fallible operations in `jope-carbon`.
pub type CarbonResult<T> = Result<T, CarbonError>;

/// Convert any `Display` error by value into `String` without creating per-callsite closures.
pub(crate) fn err_to_string<E: fmt::Display>(err: E) -> String {
    err.to_string()
}

/// Extension trait mapping `Result<T, E: Display>` to `Result<T, String>` with a static context
/// prefix without generating an individual closure per call site.
pub(crate) trait CarbonErrCtx<T> {
    fn err_ctx(self, prefix: &str) -> Result<T, String>;
}

impl<T, E: fmt::Display> CarbonErrCtx<T> for Result<T, E> {
    #[cold]
    fn err_ctx(self, prefix: &str) -> Result<T, String> {
        match self {
            Ok(v) => Ok(v),
            Err(e) => Err(format!("{prefix}: {e}")),
        }
    }
}

/// Serialize `value` to compact JSON bytes, mapping errors via [`err_to_string`].
pub(crate) fn to_json_vec<T: serde::Serialize>(value: &T) -> Result<Vec<u8>, String> {
    serde_json::to_vec(value).map_err(err_to_string)
}

/// Serialize `value` to a compact JSON `String`, mapping errors via [`err_to_string`].
pub(crate) fn to_json_string<T: serde::Serialize>(value: &T) -> Result<String, String> {
    serde_json::to_string(value).map_err(err_to_string)
}

/// `CarbonGuidance` is a struct for the typed repair surface contract in `jope-carbon`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CarbonGuidance {
    /// Control boundary that refused the operation.
    pub purpose: &'static str,
    /// Exact error description.
    pub reason: String,
    /// Actionable remedies ordered by likelihood.
    pub common_fixes: Vec<&'static str>,
    /// Local documentation reference.
    pub docs_url: &'static str,
    /// Deterministic local verification command.
    pub repair_hint: &'static str,
}

impl CarbonError {
    /// Typed repair guidance for every `CarbonError` variant.
    pub fn guidance(&self) -> CarbonGuidance {
        let (purpose, common_fixes, repair_hint) = match self {
            Self::RowLimitExceeded { .. }
            | Self::FeatureLimitExceeded { .. }
            | Self::EffortTierNotPermitted { .. } => (
                "enforce per-tier row, feature, and effort budgets before compute dispatch",
                vec![
                    "subsample rows or select features within the Free/Pro tier ceiling",
                    "upgrade to Pro (<= 100,000 rows, <= 512 cols, high/ultra) or Enterprise (no_limit)",
                ],
                "inspect dataset shape with `carbon inspect <CSV>` or upgrade via `--tier pro` / `carbon serve`",
            ),
            Self::InvalidPlanInput { .. } | Self::WorkingSetExceedsLambdaClass { .. } => (
                "right-size stateless Lambda memory classes within physical container limits",
                vec![
                    "pass non-zero fit_rows, features, and candidate counts",
                    "route multi-gigabyte tables to self-hosted Enterprise `carbon serve`",
                ],
                "check dataset dimensions with `carbon inspect <CSV>` or preview routing via `carbon plan --help`",
            ),
            Self::SealAuthenticationFailed { .. } => (
                "prevent cleartext exposure, domain confusion, or cross-tenant replay of sealed state",
                vec![
                    "open the envelope with the exact master secret, seal domain, tenant_id, run_id, and U_DEK",
                    "verify transport or storage did not alter sealed_checkpoint.enc or sealed_champion.enc",
                ],
                "verify the saved `--model <FILE>` handle matches this run and check `carbon doctor`",
            ),
            Self::UserEnvelopeFailed { .. } => (
                "enforce per-user zero-knowledge ECDH-P256 + HKDF-SHA256 + AES-256-GCM encryption",
                vec![
                    "supply the exact per-run wrapped U_DEK (CUDK v1) matching carbon-engine's P-256 KEK",
                    "ensure CUEN v1 payload slots (FitJmat, HoldoutJmat, LandmarkJlmb, Predict) are not swapped or modified",
                ],
                "run `carbon self-verify` and `carbon doctor` to check regional KEK pins and envelope keys",
            ),
            Self::CapabilityAuthFailed { .. } => (
                "authenticate tenant capability tokens and isolate tenant namespaces at the edge",
                vec![
                    "acquire a Free token via POST /v1/carbon/auth/anonymous or pass a signed Pro Bearer token",
                    "ensure tenant_id matches ^[A-Za-z0-9_-]{3,64}$ and matches the token's bound tenant_id",
                ],
                "check active token status with `carbon whoami` or register a Free token with `carbon register`",
            ),
            Self::InvalidRoleOrCommand { .. } => (
                "dispatch only recognized Carbon cell roles and CLI subcommands",
                vec![
                    "set CARBON_ROLE to edge, engine, eval, serve, or emu",
                    "run `carbon --help` or `carbon-edge --healthcheck`",
                ],
                "run `carbon --help` to list valid commands (`fit`, `predict`, `inspect`, `explain`, `export`, `doctor`)",
            ),
        };
        CarbonGuidance {
            purpose,
            reason: self.to_string(),
            common_fixes,
            docs_url: CARBON_DOCS_URL,
            repair_hint,
        }
    }
}
