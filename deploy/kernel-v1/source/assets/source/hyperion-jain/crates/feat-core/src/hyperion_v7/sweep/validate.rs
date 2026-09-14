use anyhow::{bail, Result};

use super::super::config::SweepPlan;

pub(super) fn validate_plan(plan: &SweepPlan, max_patch: usize) -> Result<()> {
    if plan.sweeps == 0 {
        bail!("invalid Hyperion sweep plan: sweeps must be greater than 0");
    }
    if plan.batch == 0 {
        bail!("invalid Hyperion sweep plan: batch must be greater than 0");
    }
    if plan.scales.is_empty() {
        bail!("invalid Hyperion sweep plan: at least one scale (area) is required");
    }
    for &p in &plan.scales {
        if p < 2 {
            bail!("invalid Hyperion sweep plan: scale {p} is below min_patch 2");
        }
        if p > max_patch {
            bail!("invalid Hyperion sweep plan: scale {p} exceeds max_patch {max_patch}");
        }
    }
    Ok(())
}

pub(super) fn validate_permutation(perm: &[usize], expected_len: usize, name: &str) -> Result<()> {
    if perm.len() != expected_len {
        bail!(
            "invalid Hyperion sweep fixed {name} permutation: expected length {}, got {}",
            expected_len,
            perm.len()
        );
    }
    let mut seen = vec![false; expected_len];
    for &idx in perm {
        if idx >= expected_len {
            bail!(
                "invalid Hyperion sweep fixed {name} permutation: index {} out of range {}",
                idx,
                expected_len
            );
        }
        if seen[idx] {
            bail!(
                "invalid Hyperion sweep fixed {name} permutation: duplicate index {}",
                idx
            );
        }
        seen[idx] = true;
    }
    Ok(())
}
