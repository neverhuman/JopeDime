//! The internal r882 proposal path: signed embedded assets -> admitted ranker ->
//! the existing Engine discovery, native refit, persisted model and prediction.
//! This capability does not implement or alias the historical step-60000 policy.
use crate::{ranker::cuda::release::load_r882_embedded, run::learned::AdmittedRanker};
use anyhow::{Result, ensure};
use jope_assets::{Package, Scope};
use jope_runtime::methods::MethodRecord;
use serde::Deserialize;

pub const METHOD: &str = "proposal.r882.regression";
const DESCRIPTOR: &str = include_str!("../../../models/bundles/r882.json");

#[derive(Deserialize)]
struct Descriptor {
    files: Vec<File>,
}
#[derive(Deserialize)]
struct File {
    path: String,
    sha256: String,
    bytes: u64,
}

/// Complete, source-pinned decoded closure. Build-time profile locking uses this
/// to reject an incomplete or differently bound method before normal packing.
pub fn required_assets() -> Result<Vec<(String, String, u64)>> {
    let descriptor: Descriptor = serde_json::from_str(DESCRIPTOR)?;
    Ok(descriptor
        .files
        .into_iter()
        .map(|file| (format!("model/r882/{}", file.path), file.sha256, file.bytes))
        .collect())
}

/// Install only the explicit authenticated method promised by this executable.
/// An integrity/closure failure propagates; it never becomes heuristic fallback.
pub fn from_package(package: &Package) -> Result<Option<AdmittedRanker>> {
    if !package.manifest().methods.iter().any(|id| id == METHOD) {
        return Ok(None);
    }
    ensure!(
        matches!(
            package.manifest().scope,
            Scope::Internal | Scope::InternalRestricted
        ),
        "r882 proposal runtime is internal-only"
    );
    Ok(Some(AdmittedRanker::r882(load_r882_embedded(package)?)?))
}

pub(crate) fn from_self() -> Result<Option<AdmittedRanker>> {
    Package::open_self()?
        .as_ref()
        .map(from_package)
        .transpose()
        .map(Option::flatten)
}

/// Actual service capability. Without a complete authenticated package it is
/// compiled, but neither assets-present nor executable. Auto stays quality gated.
pub fn record(package: Option<&Package>) -> Result<MethodRecord> {
    let executable = package.map(from_package).transpose()?.flatten().is_some();
    Ok(MethodRecord {
        id: METHOD.into(),
        task: "regression",
        kind: "proposal",
        family: "jope-service Engine discovery/native refit".into(),
        configuration: "r882; run --guidance learned --landmarks full; numeric regression".into(),
        aliases: vec![],
        compiled: true,
        required_assets: required_assets()?
            .into_iter()
            .map(|(id, _, _)| id)
            .collect(),
        assets_present: executable,
        rights_authorized: executable,
        qualification: "internal opt-in only; authentication is not M2/M3/M5/S4/Q1 or final release qualification",
        executable,
        reason: (!executable).then(|| {
            "requires the internal signed full-profile r882 closure; use the v1 run command".into()
        }),
        recipe: None,
    })
}

/// Build-time availability, followed by exact closure validation in jope-dist.
/// Other proposed methods retain their existing refusal, including step60000.
pub fn uncompiled(promised: &[String]) -> Result<Vec<(String, String)>> {
    jope_runtime::methods::unexecutable(
        &promised
            .iter()
            .filter(|id| id.as_str() != METHOD)
            .cloned()
            .collect::<Vec<_>>(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compiled_adapter_needs_assets_and_does_not_impersonate_step60000() {
        let record = record(None).unwrap();
        assert!(record.compiled);
        assert!(!record.executable && !record.assets_present && !record.rights_authorized);
        assert!(
            record
                .required_assets
                .contains(&"model/r882/dope.safetensors".into())
        );
        assert!(
            record
                .required_assets
                .contains(&"model/r882/release-catalog.json".into())
        );
        let blocked = uncompiled(&[METHOD.into(), "proposal.step60000.regression".into()]).unwrap();
        assert_eq!(blocked.len(), 1);
        assert_eq!(blocked[0].0, "proposal.step60000.regression");
    }

    #[test]
    #[ignore = "requires the already release-signed internal r882 asset package; no new signing"]
    fn embedded_loader_consumes_the_existing_signed_r882_without_sidecars() {
        let path = std::env::var_os("JOPE_R882_SIGNED_PACKAGE").expect("signed package required");
        let package = Package::open(std::path::Path::new(&path)).unwrap().unwrap();
        let bundle = load_r882_embedded(&package).unwrap();
        assert_eq!(
            bundle.digest().to_string(),
            crate::ranker::cuda::release::R882
        );
        let ranker = AdmittedRanker::r882(bundle).unwrap();
        assert_eq!(ranker.digest(), crate::ranker::cuda::release::R882);
        // The existing internal asset-authentication package promises no runtime
        // methods: its signature cannot be reused to enable this new full profile.
        assert!(!package.manifest().methods.iter().any(|id| id == METHOD));
        assert!(from_package(&package).unwrap().is_none());
        assert!(!record(Some(&package)).unwrap().executable);
    }
}
