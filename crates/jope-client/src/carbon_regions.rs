//! Multi-region latency-probing smart router for Carbon Lambda cells (`ADR-0007` Phase C8).
//!
//! Tracks smoothed round-trip latency (`EWMA`) and consecutive-failure circuit-breaker
//! state across the 17 global Carbon cells (`/.well-known/carbon-regions.json`) so clients
//! automatically route to the fastest healthy regional endpoint and fail over cleanly.

use std::collections::BTreeMap;
use std::fmt;

use jope_api::carbon::{CarbonPingResponse, CarbonRegionCatalog, CarbonRegionDescriptor};

const EWMA_ALPHA: f64 = 0.30;
const CIRCUIT_BREAKER_THRESHOLD: u32 = 3;
const CIRCUIT_COOLDOWN_MS: u64 = 30_000;

/// `CarbonRouteError` is an enum describing smart-router configuration or availability errors.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CarbonRouteError {
    EmptyCatalog,
    AllRegionsUnavailable { total_regions: usize },
    UnknownRegion { region: String },
}

impl CarbonRouteError {
    /// Operator-facing remediation guidance for this routing error.
    pub fn guidance(&self) -> &'static str {
        match self {
            Self::EmptyCatalog => {
                "Supply at least one regional endpoint in CarbonRegionCatalog or an Enterprise serve URL."
            }
            Self::AllRegionsUnavailable { .. } => {
                "Wait for regional circuit-breaker cooldown or verify network reachability to /.well-known/carbon-regions.json."
            }
            Self::UnknownRegion { .. } => {
                "Record probe samples only for regions present in the active CarbonRegionCatalog."
            }
        }
    }
}

impl fmt::Display for CarbonRouteError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyCatalog => f.write_str("Carbon region catalog contains zero endpoints"),
            Self::AllRegionsUnavailable { total_regions } => write!(
                f,
                "all {total_regions} Carbon regional cells have open circuit breakers"
            ),
            Self::UnknownRegion { region } => {
                write!(f, "unknown Carbon region '{region}' in smart router")
            }
        }
    }
}

impl std::error::Error for CarbonRouteError {}

/// `CarbonRegionHealth` is a struct tracking EWMA RTT and circuit-breaker state for one region.
#[derive(Clone, Debug, PartialEq)]
pub struct CarbonRegionHealth {
    pub descriptor: CarbonRegionDescriptor,
    pub ewma_rtt_ms: Option<f64>,
    pub consecutive_failures: u32,
    pub circuit_open_until_ms: Option<u64>,
    pub samples: u64,
}

impl CarbonRegionHealth {
    /// Returns true if the circuit breaker is closed (or its cooldown has expired at `now_ms`).
    pub fn is_available_at(&self, now_ms: u64) -> bool {
        self.circuit_open_until_ms
            .is_none_or(|until| now_ms >= until)
    }

    fn effective_score_ms(&self, preferred_geography: Option<&str>) -> f64 {
        let base = self
            .ewma_rtt_ms
            .unwrap_or(500.0 + f64::from(self.descriptor.priority_rank) * 5.0);
        let geo_bonus = if preferred_geography.is_some_and(|g| g == self.descriptor.geography) {
            -2.5
        } else {
            0.0
        };
        let penalty = f64::from(self.consecutive_failures) * 40.0;
        (base + geo_bonus + penalty).max(0.1)
    }
}

/// `CarbonSmartRouter` is a struct selecting the fastest healthy regional Carbon cell across
/// up to 17 global AWS Graviton2 (`arm64`) regions (or an Enterprise VPC endpoint).
#[derive(Clone, Debug)]
pub struct CarbonSmartRouter {
    states: BTreeMap<String, CarbonRegionHealth>,
    preferred_geography: Option<String>,
}

impl CarbonSmartRouter {
    /// Construct a `CarbonSmartRouter` from a `CarbonRegionCatalog`.
    pub fn from_catalog(
        catalog: CarbonRegionCatalog,
        preferred_geography: Option<String>,
    ) -> std::result::Result<Self, CarbonRouteError> {
        if catalog.regions.is_empty() {
            return Err(CarbonRouteError::EmptyCatalog);
        }
        let mut states = BTreeMap::new();
        for descriptor in catalog.regions {
            states.insert(
                descriptor.region.clone(),
                CarbonRegionHealth {
                    descriptor,
                    ewma_rtt_ms: None,
                    consecutive_failures: 0,
                    circuit_open_until_ms: None,
                    samples: 0,
                },
            );
        }
        Ok(Self {
            states,
            preferred_geography,
        })
    }

    /// Construct a `CarbonSmartRouter` seeded with the canonical 17-region catalog.
    pub fn default_17_regions(
        domain_suffix: &str,
        preferred_geography: Option<String>,
    ) -> std::result::Result<Self, CarbonRouteError> {
        Self::from_catalog(
            CarbonRegionCatalog::default_17_region_catalog(domain_suffix),
            preferred_geography,
        )
    }

    /// Record a successful `/v1/carbon/ping` RTT measurement (`rtt_ms`) at `now_ms`.
    pub fn record_probe_success(
        &mut self,
        region: &str,
        rtt_ms: f64,
        _ping: Option<&CarbonPingResponse>,
        now_ms: u64,
    ) -> std::result::Result<(), CarbonRouteError> {
        let entry = self
            .states
            .get_mut(region)
            .ok_or_else(|| CarbonRouteError::UnknownRegion {
                region: region.to_owned(),
            })?;
        let clamped = rtt_ms.clamp(0.1, 60_000.0);
        entry.ewma_rtt_ms = Some(match entry.ewma_rtt_ms {
            Some(prev) => EWMA_ALPHA * clamped + (1.0 - EWMA_ALPHA) * prev,
            None => clamped,
        });
        let _ = now_ms;
        entry.consecutive_failures = 0;
        entry.circuit_open_until_ms = None;
        entry.samples = entry.samples.saturating_add(1);
        Ok(())
    }

    /// Record a probe or request failure for `region` at `now_ms`, tripping the circuit breaker
    /// after `3` consecutive failures.
    pub fn record_failure(
        &mut self,
        region: &str,
        now_ms: u64,
    ) -> std::result::Result<(), CarbonRouteError> {
        let entry = self
            .states
            .get_mut(region)
            .ok_or_else(|| CarbonRouteError::UnknownRegion {
                region: region.to_owned(),
            })?;
        entry.consecutive_failures = entry.consecutive_failures.saturating_add(1);
        if entry.consecutive_failures >= CIRCUIT_BREAKER_THRESHOLD {
            entry.circuit_open_until_ms = Some(now_ms.saturating_add(CIRCUIT_COOLDOWN_MS));
        }
        Ok(())
    }

    /// Return all currently available regional descriptors ordered fastest-first at `now_ms`.
    pub fn ranked_candidates(
        &self,
        now_ms: u64,
    ) -> std::result::Result<Vec<CarbonRegionDescriptor>, CarbonRouteError> {
        let pref = self.preferred_geography.as_deref();
        let mut available: Vec<&CarbonRegionHealth> = self
            .states
            .values()
            .filter(|state| state.is_available_at(now_ms))
            .collect();
        if available.is_empty() {
            return Err(CarbonRouteError::AllRegionsUnavailable {
                total_regions: self.states.len(),
            });
        }
        available.sort_by(|a, b| {
            a.effective_score_ms(pref)
                .partial_cmp(&b.effective_score_ms(pref))
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.descriptor.priority_rank.cmp(&b.descriptor.priority_rank))
                .then_with(|| a.descriptor.region.cmp(&b.descriptor.region))
        });
        Ok(available
            .into_iter()
            .map(|s| s.descriptor.clone())
            .collect())
    }

    /// Select the single fastest healthy regional endpoint at `now_ms`.
    pub fn select_best(
        &self,
        now_ms: u64,
    ) -> std::result::Result<CarbonRegionDescriptor, CarbonRouteError> {
        self.ranked_candidates(now_ms)?.into_iter().next().ok_or(
            CarbonRouteError::AllRegionsUnavailable {
                total_regions: self.states.len(),
            },
        )
    }

    /// Inspect tracked health state for `region`.
    pub fn region_health(&self, region: &str) -> Option<&CarbonRegionHealth> {
        self.states.get(region)
    }
}
