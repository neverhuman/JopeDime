//! Versioned descriptor axes and deterministic quality-diversity cell assignment.
//!
//! This module only defines a proposal-memory descriptor contract. It does not qualify
//! candidates or make archive scores comparable across evaluator/fidelity identities.

use anyhow::{ensure, Context, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};

pub const DESCRIPTOR_SCHEMA: &str = "jope.qd-descriptors/v1";

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DescriptorAxis {
    pub name: String,
    pub version: String,
    /// Increasing upper boundaries; bins are lower-inclusive and upper-exclusive.
    pub boundaries: Vec<f64>,
}

impl DescriptorAxis {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            !self.name.trim().is_empty(),
            "descriptor axis name is empty"
        );
        ensure!(
            !self.version.trim().is_empty(),
            "descriptor axis version is empty"
        );
        ensure!(
            self.boundaries.iter().all(|v| v.is_finite()),
            "descriptor boundaries must be finite"
        );
        ensure!(
            self.boundaries.windows(2).all(|pair| pair[0] < pair[1]),
            "descriptor boundaries must be strictly increasing"
        );
        Ok(())
    }

    pub fn bin(&self, value: f64) -> Result<u32> {
        self.validate()?;
        ensure!(value.is_finite(), "descriptor value must be finite");
        u32::try_from(
            self.boundaries
                .iter()
                .take_while(|boundary| value >= **boundary)
                .count(),
        )
        .context("descriptor bin index overflow")
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DescriptorSpace {
    pub schema: String,
    pub version: String,
    pub axes: Vec<DescriptorAxis>,
}

impl DescriptorSpace {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.schema == DESCRIPTOR_SCHEMA,
            "unknown descriptor schema"
        );
        ensure!(
            !self.version.trim().is_empty(),
            "descriptor version is empty"
        );
        ensure!(
            (2..=4).contains(&self.axes.len()),
            "descriptor space needs 2 to 4 axes"
        );
        let mut names = BTreeSet::new();
        for axis in &self.axes {
            axis.validate()?;
            ensure!(
                names.insert(axis.name.as_str()),
                "duplicate descriptor axis name"
            );
        }
        Ok(())
    }

    /// Stable identity includes schema, descriptor version, axis versions and all boundaries.
    pub fn identity_sha256(&self) -> Result<String> {
        self.validate()?;
        let bytes = serde_json::to_vec(self).context("serialize descriptor identity")?;
        Ok(hex_digest(Sha256::digest(bytes).as_slice()))
    }

    pub fn bins(&self, values: &BTreeMap<String, f64>) -> Result<Vec<u32>> {
        self.validate()?;
        ensure!(
            values.len() == self.axes.len(),
            "descriptor value axis count mismatch"
        );
        self.axes
            .iter()
            .map(|axis| {
                values
                    .get(&axis.name)
                    .copied()
                    .with_context(|| format!("missing descriptor {}", axis.name))
                    .and_then(|value| axis.bin(value))
            })
            .collect()
    }
}

fn hex_digest(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        out.push(HEX[(byte >> 4) as usize] as char);
        out.push(HEX[(byte & 0x0f) as usize] as char);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn axis(name: &str, bounds: &[f64]) -> DescriptorAxis {
        DescriptorAxis {
            name: name.into(),
            version: "axis-v1".into(),
            boundaries: bounds.to_vec(),
        }
    }

    fn space(version: &str) -> DescriptorSpace {
        DescriptorSpace {
            schema: DESCRIPTOR_SCHEMA.into(),
            version: version.into(),
            axes: vec![axis("movement", &[0.25, 0.75]), axis("energy", &[1.0])],
        }
    }

    #[test]
    fn descriptor_version_and_bounds_change_identity() {
        assert_ne!(
            space("v1").identity_sha256().unwrap(),
            space("v2").identity_sha256().unwrap()
        );
        let mut changed_bounds = space("v1");
        changed_bounds.axes[0].boundaries[0] = 0.2;
        assert_ne!(
            space("v1").identity_sha256().unwrap(),
            changed_bounds.identity_sha256().unwrap()
        );
    }

    #[test]
    fn bins_are_deterministic_and_upper_exclusive() {
        let axis = axis("movement", &[0.25, 0.75]);
        assert_eq!(axis.bin(0.249).unwrap(), 0);
        assert_eq!(axis.bin(0.25).unwrap(), 1);
        assert_eq!(axis.bin(0.75).unwrap(), 2);
        assert!(axis.bin(f64::NAN).is_err());
    }

    #[test]
    fn descriptor_space_requires_two_to_four_unique_axes() {
        let mut one = space("v1");
        one.axes.pop();
        assert!(one.validate().is_err());
        let mut repeated = space("v1");
        repeated.axes[1].name = repeated.axes[0].name.clone();
        assert!(repeated.validate().is_err());
    }
}
