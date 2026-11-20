//! `verify --manifests-only`: the committed `models/` directory is self-consistent without any
//! download — index ↔ bundle manifests ↔ schema ↔ `PINNED`.

use crate::index::Repo;
use anyhow::Result;
use jope_datasets::hash::sha256_hex;
use std::collections::BTreeSet;

/// Summary and validation errors from checking committed model manifests.
#[derive(Clone, Debug, Default)]
pub struct ManifestsReport {
    /// `(bundle_id, files, bytes)` per indexed bundle.
    pub bundles: Vec<(String, usize, u64)>,
    /// Bundle identifiers listed in `models/PINNED`.
    pub pinned: Vec<String>,
    /// Validation errors found across the index, manifests, and pinned list.
    pub errors: Vec<String>,
}

/// Validate the committed model index, bundle manifests, and pinned list without network access.
pub fn verify_manifests(repo: &Repo) -> Result<ManifestsReport> {
    let mut report = ManifestsReport::default();
    let index = repo.read_index()?;
    let mut known = BTreeSet::new();
    for row in &index.bundles {
        let p = repo.root.join(&row.manifest);
        let label = row.manifest.clone();
        let bytes = match std::fs::read(&p) {
            Ok(b) => b,
            Err(e) => {
                report.errors.push(format!("{label}: {e}"));
                continue;
            }
        };
        if sha256_hex(&bytes) != row.manifest_sha256 {
            report
                .errors
                .push(format!("{label}: sha256 differs from models/manifest.json"));
        }
        let m = match crate::manifest::BundleManifest::parse(&bytes) {
            Ok(m) => m,
            Err(errs) => {
                report.errors.push(format!("{label}: {}", errs.join("; ")));
                continue;
            }
        };
        known.insert(m.bundle_id.clone());
        if m.bundle_id != row.bundle_id {
            report.errors.push(format!(
                "{label}: bundle_id {} != index {}",
                m.bundle_id, row.bundle_id
            ));
        }
        if p != repo.manifest_path(&m.bundle_id) {
            report.errors.push(format!(
                "{label}: expected models/bundles/{}.json",
                m.bundle_id
            ));
        }
        if row.s3_key != crate::manifest_key(&m.bundle_id) {
            report
                .errors
                .push(format!("{label}: s3_key {}", row.s3_key));
        }
        if row.files != m.files.len() || row.bytes != m.total_bytes() {
            report
                .errors
                .push(format!("{label}: files/bytes totals differ from the index"));
        }
        let licences: BTreeSet<String> = row.licenses.iter().cloned().collect();
        if licences != m.license_set() {
            report.errors.push(format!(
                "{label}: index licences {licences:?} != manifest {:?}",
                m.license_set()
            ));
        }
        if let Some(bucket) = m.extra.get("bucket").and_then(|v| v.as_str()) {
            if bucket != index.bucket {
                report.errors.push(format!(
                    "{label}: bucket {bucket} != index {}",
                    index.bucket
                ));
            }
        }
        report
            .bundles
            .push((m.bundle_id.clone(), m.files.len(), m.total_bytes()));
    }
    match repo.read_pinned() {
        Ok(pinned) => {
            for id in &pinned {
                if !known.contains(id) {
                    report
                        .errors
                        .push(format!("models/PINNED names unknown bundle {id}"));
                }
            }
            report.pinned = pinned;
        }
        Err(e) => report.errors.push(format!("{e:#}")),
    }
    for p in repo.manifest_files() {
        let id = p
            .file_stem()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_default();
        if !known.contains(&id) {
            report.errors.push(format!(
                "{} is not listed in models/manifest.json",
                p.display()
            ));
        }
    }
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    #[test]
    fn committed_models_dir_is_consistent() {
        let repo = Repo::at(Path::new(env!("CARGO_MANIFEST_DIR")).join("../.."));
        let r = verify_manifests(&repo).unwrap();
        assert!(r.errors.is_empty(), "{:?}", r.errors);
        assert_eq!(r.bundles.len(), 6);
        assert!(r.bundles.iter().any(|(id, _, _)| id == "r882"));
        assert!(r.bundles.iter().any(|(id, files, bytes)| id
            == crate::UNIFIED_ICL_REGBIN_V2_BUNDLE_ID
            && *files == 6
            && *bytes == 1_074_424_101));
        assert_eq!(r.pinned.len(), 4);
        assert!(r
            .pinned
            .iter()
            .any(|id| id == crate::UNIFIED_ICL_REGBIN_V2_BUNDLE_ID));
    }

    #[test]
    fn unified_icl_regbin_v2_pins_match_committed_manifest() {
        let repo = Repo::at(Path::new(env!("CARGO_MANIFEST_DIR")).join("../.."));
        let manifest_path = repo.manifest_path(crate::UNIFIED_ICL_REGBIN_V2_BUNDLE_ID);
        let bytes = std::fs::read(&manifest_path).unwrap();
        assert_eq!(
            sha256_hex(&bytes),
            crate::UNIFIED_ICL_REGBIN_V2_MANIFEST_SHA256
        );
        let m = crate::manifest::BundleManifest::parse(&bytes).unwrap();
        assert_eq!(m.bundle_id, crate::UNIFIED_ICL_REGBIN_V2_BUNDLE_ID);
        assert_eq!(
            m.extra.get("bundle_version").and_then(|v| v.as_str()),
            Some(crate::UNIFIED_ICL_REGBIN_V2_VERSION)
        );
        let digest_of = |path: &str| {
            m.files
                .iter()
                .find(|f| f.path == path)
                .map(|f| f.sha256.as_str())
                .unwrap_or("")
        };
        assert_eq!(
            digest_of("unified_icl_regbin_M.safetensors"),
            crate::UNIFIED_ICL_REGBIN_M_V2_SHA256
        );
        assert_eq!(
            digest_of("unified_icl_regbin_L.safetensors"),
            crate::UNIFIED_ICL_REGBIN_L_V2_SHA256
        );
        assert_eq!(
            digest_of("unified_icl_regbin_M_stage2a_noarena.safetensors"),
            crate::UNIFIED_ICL_REGBIN_M_V2_STAGE2A_SHA256
        );
        assert_eq!(
            digest_of("unified_icl_regbin_L_stage2a_noarena.safetensors"),
            crate::UNIFIED_ICL_REGBIN_L_V2_STAGE2A_SHA256
        );
        assert_eq!(
            digest_of("dope.safetensors"),
            crate::UNIFIED_ICL_REGBIN_DOPE_SHA256
        );
    }
}
