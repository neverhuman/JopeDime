//! Deployment profiles (`deploy/appliance/profiles/<name>.toml`, `jope.appliance-profile/v1`) and
//! the model lock derived from them (`deploy/appliance/models.lock.json`,
//! `jope.appliance-model-lock/v1`), SPECIFICATION.md §5.3–5.4.
//!
//! A profile names the methods a package promises, the distribution scope, the redistribution
//! classes that scope admits, and the components (files of committed `jope.model-bundle/v2`
//! manifests) it embeds. The lock resolves every file to its sha256, size and licence, and
//! records each reason the profile cannot be released yet — a file whose rights the scope does not
//! admit, a missing owner acknowledgement, a weight with no licence record, a promised method this
//! build cannot execute. A blocked profile locks fine (the lock is a faithful derivation, checked
//! in CI without weights) but `jope-dist pack` refuses it.

use anyhow::{bail, ensure, Context, Result};
use jope_assets::Scope;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

pub const PROFILE_SCHEMA: &str = "jope.appliance-profile/v1";
pub const LOCK_SCHEMA: &str = "jope.appliance-model-lock/v1";
/// In `methods`: every native method of the registry.
pub const NATIVE: &str = "native";
const CLASSES: [&str; 4] = ["permitted", "internal", "prohibited", "conditional"];

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Profile {
    pub schema: String,
    pub name: String,
    pub description: String,
    pub scope: String,
    pub redistribution: Vec<String>,
    pub methods: Vec<String>,
    /// Repository-relative directory of the bundle manifests the components name.
    pub bundles: String,
    pub max_payload_mib: u64,
    /// Repository-relative path of the owner acknowledgement `prohibited` files need.
    pub owner_ack: Option<String>,
    pub codec: CodecConfig,
    #[serde(rename = "component", default)]
    pub components: Vec<Component>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CodecConfig {
    pub name: String,
    pub level: i32,
    pub frame_mib: usize,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Component {
    pub id: String,
    pub bundle: String,
    pub files: Vec<String>,
    /// `model`, `config` or `manifest`.
    pub role: String,
    #[serde(default = "cpu_only")]
    pub backends: Vec<String>,
    #[serde(default)]
    pub requires: Vec<String>,
}

fn cpu_only() -> Vec<String> {
    vec!["cpu".to_owned()]
}

/// The widest redistribution classes each scope may admit. Development packages run on
/// untrusted CI runners, so they carry only `permitted` (generated) files.
pub fn admitted(scope: Scope) -> &'static [&'static str] {
    match scope {
        Scope::Development | Scope::Public => &["permitted"],
        Scope::Internal => &["permitted", "internal", "conditional"],
        Scope::InternalRestricted => &["permitted", "internal", "conditional", "prohibited"],
    }
}

impl Profile {
    /// Parse and check a profile; returns it and the sha256 of its bytes.
    pub fn load(path: &Path) -> Result<(Self, String)> {
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("read profile {}", path.display()))?;
        let profile: Profile =
            toml::from_str(&text).with_context(|| format!("parse {}", path.display()))?;
        ensure!(
            profile.schema == PROFILE_SCHEMA,
            "{} is not {PROFILE_SCHEMA}",
            path.display()
        );
        ensure!(
            path.file_stem().and_then(|s| s.to_str()) == Some(profile.name.as_str()),
            "{}: name must match the file name",
            path.display()
        );
        ensure!(
            Scope::parse(&profile.scope).is_some(),
            "{}: unknown scope {}",
            path.display(),
            profile.scope
        );
        ensure!(
            ["shuffle+zstd", "zstd", "raw"].contains(&profile.codec.name.as_str()),
            "unknown codec {}",
            profile.codec.name
        );
        ensure!(
            (1..=22).contains(&profile.codec.level) && (1..=64).contains(&profile.codec.frame_mib),
            "codec level must be 1-22 and frame_mib 1-64"
        );
        for class in &profile.redistribution {
            ensure!(
                CLASSES.contains(&class.as_str()),
                "unknown redistribution class {class}"
            );
        }
        let ceiling = admitted(profile.scope());
        for class in &profile.redistribution {
            ensure!(
                ceiling.contains(&class.as_str()),
                "scope {} admits at most {ceiling:?}; `{class}` is wider",
                profile.scope
            );
        }
        if profile.redistribution.iter().any(|c| c == "prohibited") {
            ensure!(
                profile.owner_ack.is_some(),
                "a profile admitting `prohibited` files must name an owner_ack"
            );
        }
        let ids: BTreeSet<&str> = profile.components.iter().map(|c| c.id.as_str()).collect();
        ensure!(
            ids.len() == profile.components.len(),
            "component ids must be unique"
        );
        for component in &profile.components {
            ensure!(
                ["model", "config", "manifest"].contains(&component.role.as_str()),
                "component {}: unknown role {}",
                component.id,
                component.role
            );
            for required in &component.requires {
                ensure!(
                    ids.contains(required.as_str()),
                    "component {} requires {required}, which the profile does not carry",
                    component.id
                );
            }
            for backend in &component.backends {
                ensure!(
                    ["cpu", "cuda"].contains(&backend.as_str()),
                    "component {}: unknown backend {backend}",
                    component.id
                );
            }
        }
        Ok((profile, jope_assets::sha256_hex(text.as_bytes())))
    }

    pub fn scope(&self) -> Scope {
        Scope::parse(&self.scope).expect("checked at load")
    }

    /// The promised method ids: `native` expanded to the registry's native roster, sorted.
    pub fn expanded_methods(&self) -> Result<Vec<String>> {
        let mut out = BTreeSet::new();
        for method in &self.methods {
            if method == NATIVE {
                for kind in jope_runtime::task::SUPPORTED {
                    for m in jope_runtime::methods::rust_methods(kind)? {
                        out.insert(m.id);
                    }
                }
                out.insert(jope_runtime::methods::AUTO.to_owned());
            } else {
                out.insert(method.clone());
            }
        }
        Ok(out.into_iter().collect())
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct Lock {
    pub schema: String,
    pub profiles: BTreeMap<String, LockedProfile>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct LockedProfile {
    pub profile_sha256: String,
    pub scope: String,
    /// `ready`, or `blocked` with the reasons in `blockers`.
    pub status: String,
    pub blockers: Vec<String>,
    pub methods: Vec<String>,
    pub components: Vec<LockedComponent>,
    pub total_bytes: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct LockedComponent {
    pub id: String,
    pub role: String,
    pub bundle_id: String,
    pub bundle_manifest_sha256: String,
    pub backends: Vec<String>,
    pub files: Vec<LockedFile>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct LockedFile {
    pub path: String,
    pub sha256: String,
    pub bytes: u64,
    pub kind: String,
    /// The SPDX expression, or `none` for a metadata file with no licence block.
    pub spdx: String,
    /// `permitted`, `internal`, `prohibited`, `conditional`, or `none` (metadata).
    pub redistribution: String,
}

#[derive(Deserialize)]
struct BundleManifest {
    schema: String,
    bundle_id: String,
    files: Vec<BundleFile>,
}

#[derive(Deserialize)]
struct BundleFile {
    path: String,
    sha256: String,
    bytes: u64,
    kind: String,
    license: Option<BundleLicense>,
}

#[derive(Deserialize)]
struct BundleLicense {
    spdx_expression: String,
    redistribution: String,
}

/// Resolve one profile against its bundle manifests (`<repo>/<bundles>/<id>.json`).
pub fn lock_profile(profile: &Profile, profile_sha256: &str, repo: &Path) -> Result<LockedProfile> {
    let mut blockers = Vec::new();
    let mut components = Vec::new();
    let mut total = 0u64;
    let bundles = repo.join(&profile.bundles);
    for component in &profile.components {
        let path = bundles.join(format!("{}.json", component.bundle));
        let text = std::fs::read(&path)
            .with_context(|| format!("read bundle manifest {}", path.display()))?;
        let bundle: BundleManifest =
            serde_json::from_slice(&text).with_context(|| format!("parse {}", path.display()))?;
        ensure!(
            bundle.schema == "jope.model-bundle/v2" && bundle.bundle_id == component.bundle,
            "{} is not the jope.model-bundle/v2 manifest of {}",
            path.display(),
            component.bundle
        );
        let mut files = Vec::new();
        for wanted in &component.files {
            let file = bundle
                .files
                .iter()
                .find(|f| &f.path == wanted)
                .with_context(|| format!("bundle {} has no file {wanted}", component.bundle))?;
            ensure!(
                jope_assets::manifest::is_name(&file.path),
                "{}/{}: not a plain relative path",
                component.bundle,
                file.path
            );
            let (spdx, redistribution) = match &file.license {
                Some(l) => (l.spdx_expression.clone(), l.redistribution.clone()),
                None => ("none".to_owned(), "none".to_owned()),
            };
            let name = format!("{}/{}", component.bundle, file.path);
            if redistribution == "none" && file.path.ends_with(".safetensors") {
                blockers.push(format!(
                    "{name}: weights with no licence record (transitive provenance unresolved)"
                ));
            } else if redistribution != "none" && !profile.redistribution.contains(&redistribution)
            {
                blockers.push(format!(
                    "{name}: redistribution `{redistribution}` is not admitted by scope {}",
                    profile.scope
                ));
            }
            if redistribution == "prohibited" {
                match &profile.owner_ack {
                    Some(ack) if repo.join(ack).is_file() => {}
                    Some(ack) => blockers.push(format!(
                        "{name}: `prohibited` needs the owner acknowledgement {ack}, which does not exist"
                    )),
                    None => blockers.push(format!(
                        "{name}: `prohibited` needs an owner_ack in the profile"
                    )),
                }
            }
            total += file.bytes;
            files.push(LockedFile {
                path: file.path.clone(),
                sha256: file.sha256.clone(),
                bytes: file.bytes,
                kind: file.kind.clone(),
                spdx,
                redistribution,
            });
        }
        let mut backends = component.backends.clone();
        backends.sort();
        backends.dedup();
        components.push(LockedComponent {
            id: component.id.clone(),
            role: component.role.clone(),
            bundle_id: component.bundle.clone(),
            bundle_manifest_sha256: jope_assets::sha256_hex(&text),
            backends,
            files,
        });
    }
    if total > profile.max_payload_mib.saturating_mul(1 << 20) {
        blockers.push(format!(
            "{total} asset bytes exceed max_payload_mib = {}",
            profile.max_payload_mib
        ));
    }
    let methods = profile.expanded_methods()?;
    for (id, reason) in jope_service::proposal::uncompiled(&methods)? {
        blockers.push(format!("method {id}: {reason}"));
    }
    if methods
        .iter()
        .any(|id| id == jope_service::proposal::METHOD)
    {
        if !matches!(profile.scope(), Scope::Internal | Scope::InternalRestricted) {
            blockers.push("r882 proposal runtime is internal-only".into());
        }
        for (id, sha, bytes) in jope_service::proposal::required_assets()? {
            let actual = components
                .iter()
                .flat_map(|component| component.files.iter().map(move |file| (component, file)))
                .find(|(component, file)| {
                    format!("{}/{}/{}", component.role, component.id, file.path) == id
                });
            if !actual.is_some_and(|(_, file)| file.sha256 == sha && file.bytes == bytes) {
                blockers.push(format!(
                    "method {}: missing or different asset {id}",
                    jope_service::proposal::METHOD
                ));
            }
        }
    }
    Ok(LockedProfile {
        profile_sha256: profile_sha256.to_owned(),
        scope: profile.scope.clone(),
        status: if blockers.is_empty() {
            "ready"
        } else {
            "blocked"
        }
        .to_owned(),
        blockers,
        methods,
        components,
        total_bytes: total,
    })
}

/// Every `*.toml` in the profiles directory, sorted.
pub fn profile_paths(dir: &Path) -> Result<Vec<PathBuf>> {
    let mut out: Vec<PathBuf> = std::fs::read_dir(dir)
        .with_context(|| format!("read {}", dir.display()))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "toml"))
        .collect();
    out.sort();
    if out.is_empty() {
        bail!("no profiles in {}", dir.display());
    }
    Ok(out)
}

/// Derive the lock for every profile in `dir`.
pub fn derive_lock(dir: &Path, repo: &Path) -> Result<Lock> {
    let mut profiles = BTreeMap::new();
    for path in profile_paths(dir)? {
        let (profile, sha) = Profile::load(&path)?;
        profiles.insert(profile.name.clone(), lock_profile(&profile, &sha, repo)?);
    }
    Ok(Lock {
        schema: LOCK_SCHEMA.to_owned(),
        profiles,
    })
}

impl Lock {
    pub fn load(path: &Path) -> Result<(Self, String)> {
        let bytes = std::fs::read(path).with_context(|| format!("read lock {}", path.display()))?;
        let lock: Lock =
            serde_json::from_slice(&bytes).with_context(|| format!("parse {}", path.display()))?;
        ensure!(
            lock.schema == LOCK_SCHEMA,
            "{} is not {LOCK_SCHEMA}",
            path.display()
        );
        Ok((lock, jope_assets::sha256_hex(&bytes)))
    }

    /// The canonical text: two-space JSON with a trailing newline.
    pub fn to_text(&self) -> String {
        let mut text = serde_json::to_string_pretty(self).expect("lock serializes");
        text.push('\n');
        text
    }
}
