//! Audit the selected normal/build closure, including dependency versions and feature caps.
//! Workspace-wide metadata unifies unrelated research features, so use `cargo tree` per selection.
use anyhow::{ensure, Context, Result};
use serde::Deserialize;
use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;
use std::process::Command;

pub const POLICY_SCHEMA: &str = "jope.appliance-dependency-policy/v1";
type Packages = BTreeMap<(String, String), BTreeSet<String>>;

/// A package-deny exception; it never grants native or denied-feature approval.
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PackageException {
    pub name: String,
    /// Literal version, never a version requirement or wildcard.
    pub version: String,
    /// Exact root feature selections, compared as sets.
    pub feature_sets: Vec<Vec<String>>,
    /// A cap: empty means no dependency features are permitted.
    pub allowed_features: BTreeSet<String>,
    pub reason: String,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Policy {
    pub schema: String,
    pub package: String,
    pub target: String,
    pub feature_sets: Vec<Vec<String>>,
    pub deny: Vec<String>,
    pub native: Vec<String>,
    #[serde(default)]
    pub native_allowed: BTreeMap<String, String>,
    #[serde(default)]
    pub denied_features: Vec<String>,
    #[serde(default)]
    pub package_exceptions: Vec<PackageException>,
}

impl Policy {
    pub fn load(path: &Path) -> Result<Self> {
        let text =
            std::fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
        let policy: Self =
            toml::from_str(&text).with_context(|| format!("parse {}", path.display()))?;
        policy
            .validate()
            .with_context(|| format!("invalid {}", path.display()))?;
        Ok(policy)
    }

    fn validate(&self) -> Result<()> {
        ensure!(
            self.schema == POLICY_SCHEMA,
            "policy is not {POLICY_SCHEMA}"
        );
        ensure!(
            !self.feature_sets.is_empty(),
            "policy has no feature selections"
        );
        for exception in &self.package_exceptions {
            ensure!(
                !exception.name.is_empty()
                    && exception
                        .name
                        .bytes()
                        .all(|c| c.is_ascii_alphanumeric() || b"-_".contains(&c)),
                "exception package must be an exact name"
            );
            ensure!(
                exception
                    .version
                    .as_bytes()
                    .first()
                    .is_some_and(u8::is_ascii_digit)
                    && exception
                        .version
                        .bytes()
                        .all(|c| c.is_ascii_alphanumeric() || b".-+".contains(&c)),
                "exception version must be literal, not a requirement"
            );
            ensure!(
                !exception.reason.trim().is_empty(),
                "exception needs a reason"
            );
            ensure!(
                !exception.feature_sets.is_empty()
                    && exception.feature_sets.iter().all(|set| {
                        !set.is_empty()
                            && set.iter().all(|f| {
                                !f.is_empty()
                                    && !f.contains('*')
                                    && !f.chars().any(char::is_whitespace)
                            })
                    }),
                "exception needs explicit nonempty feature selections"
            );
        }
        Ok(())
    }
}

fn matches(pattern: &str, name: &str) -> bool {
    match pattern.strip_suffix('*') {
        Some(prefix) => name.starts_with(prefix),
        None => pattern == name,
    }
}

fn parse_tree(text: &str) -> Result<Packages> {
    let mut out = Packages::new();
    for line in text.lines() {
        let Some((package, enabled)) = line.rsplit_once('|') else {
            continue;
        };
        let mut fields = package.split_whitespace();
        let name = fields.next().context("cargo tree omitted package name")?;
        let version = fields
            .next()
            .and_then(|v| v.strip_prefix('v'))
            .filter(|v| !v.is_empty())
            .context("cargo tree omitted package version")?;
        out.entry((name.to_owned(), version.to_owned()))
            .or_default()
            .extend(
                enabled
                    .trim_end_matches(" (*)")
                    .trim()
                    .split(',')
                    .filter(|f| !f.is_empty())
                    .map(str::to_owned),
            );
    }
    Ok(out)
}

fn selected_packages(
    manifest: &Path,
    package: &str,
    target: &str,
    features: &[String],
) -> Result<Packages> {
    let cargo = std::env::var("CARGO").unwrap_or_else(|_| "cargo".to_owned());
    let mut command = Command::new(cargo);
    command.args([
        "tree",
        "--locked",
        "--offline",
        "--color",
        "never",
        "--manifest-path",
    ]);
    command.arg(manifest);
    command.args([
        "-p",
        package,
        "--target",
        target,
        "--edges",
        "normal,build",
        "--prefix",
        "none",
        "--format",
        "{p}|{f}",
        "--no-default-features",
    ]);
    if !features.is_empty() {
        command.args(["--features", &features.join(",")]);
    }
    let output = command.output().context("run cargo tree")?;
    ensure!(
        output.status.success(),
        "cargo tree failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    parse_tree(&String::from_utf8(output.stdout)?)
}

/// Preserve the existing name-to-features API, merging versions only for this inventory view.
pub fn closure_features(
    manifest: &Path,
    package: &str,
    target: &str,
    features: &[String],
) -> Result<BTreeMap<String, BTreeSet<String>>> {
    let mut out: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    for ((name, _), enabled) in selected_packages(manifest, package, target, features)? {
        out.entry(name).or_default().extend(enabled);
    }
    Ok(out)
}

pub fn closure(
    manifest: &Path,
    package: &str,
    target: &str,
    features: &[String],
) -> Result<BTreeSet<String>> {
    Ok(closure_features(manifest, package, target, features)?
        .into_keys()
        .collect())
}

fn selection_violations(
    policy: &Policy,
    features: &[String],
    packages: &Packages,
) -> Result<Vec<String>> {
    let mut violations = Vec::new();
    let selected: BTreeSet<_> = features.iter().collect();
    let label = if features.is_empty() {
        "no features".to_owned()
    } else {
        features.join(",")
    };
    for ((name, version), enabled) in packages {
        for denied in &policy.denied_features {
            let (krate, feature) = denied
                .split_once('/')
                .with_context(|| format!("denied feature {denied:?} is not crate/feature"))?;
            if name == krate && enabled.contains(feature) {
                violations.push(format!("[{label}] {denied} is enabled (v{version})"));
            }
        }
        if let Some(pattern) = policy.deny.iter().find(|p| matches(p, name)) {
            let excepted = policy.package_exceptions.iter().any(|e| {
                e.name == *name
                    && e.version == *version
                    && enabled.is_subset(&e.allowed_features)
                    && e.feature_sets
                        .iter()
                        .any(|set| set.iter().collect::<BTreeSet<_>>() == selected)
            });
            if !excepted {
                violations.push(format!("[{label}] {name} v{version} is denied ({pattern})"));
            }
        }
        if policy.native.iter().any(|p| matches(p, name))
            && !policy.native_allowed.contains_key(name)
        {
            violations.push(format!(
                "[{label}] {name} v{version} builds native code and is not an approved exception"
            ));
        }
    }
    Ok(violations)
}

/// Empty means every selected package/version/feature closure passed this policy.
pub fn audit(manifest: &Path, policy: &Policy) -> Result<Vec<String>> {
    policy.validate()?;
    let mut violations = Vec::new();
    for features in &policy.feature_sets {
        let resolved = selected_packages(manifest, &policy.package, &policy.target, features)?;
        ensure!(
            resolved.keys().any(|(name, _)| name == &policy.package),
            "cargo tree did not resolve {}",
            policy.package
        );
        violations.extend(selection_violations(policy, features, &resolved)?);
    }
    Ok(violations)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn full() -> Vec<String> {
        ["embedded", "engine", "mcp", "client"]
            .map(str::to_owned)
            .to_vec()
    }
    fn fixture() -> Policy {
        Policy {
            schema: POLICY_SCHEMA.into(),
            package: "jope-appliance".into(),
            target: "x86_64-unknown-linux-gnu".into(),
            feature_sets: vec![full()],
            deny: vec!["candle-*".into()],
            native: vec!["cc".into()],
            native_allowed: BTreeMap::new(),
            denied_features: vec!["candle-core/cuda".into()],
            package_exceptions: vec![PackageException {
                name: "candle-core".into(),
                version: "0.11.0".into(),
                feature_sets: vec![full()],
                allowed_features: BTreeSet::from(["default".into()]),
                reason: "Reviewed CPU ranker, no CUDA or native code".into(),
            }],
        }
    }
    fn cpu() -> Packages {
        parse_tree("candle-core v0.11.0 (/vendor)|default\n").unwrap()
    }

    #[test]
    fn exact_scope_is_order_independent() {
        let policy = fixture();
        policy.validate().unwrap();
        let mut reordered = full();
        reordered.reverse();
        assert!(selection_violations(&policy, &reordered, &cpu())
            .unwrap()
            .is_empty());
    }
    #[test]
    fn minimal_partial_and_expanded_scopes_do_not_inherit_exceptions() {
        for selected in [
            vec![],
            vec!["embedded".into()],
            vec!["engine".into(), "client".into()],
            {
                let mut selected = full();
                selected.push("research".into());
                selected
            },
        ] {
            assert!(!selection_violations(&fixture(), &selected, &cpu())
                .unwrap()
                .is_empty());
        }
    }
    #[test]
    fn versions_remain_separate_and_new_versions_are_denied() {
        let mut packages = cpu();
        packages.extend(parse_tree("candle-core v0.12.0|default\n").unwrap());
        assert_eq!(packages.len(), 2);
        let violations = selection_violations(&fixture(), &full(), &packages).unwrap();
        assert_eq!(violations.len(), 1);
        assert!(violations[0].contains("v0.12.0"));
    }
    #[test]
    fn dependency_features_cannot_silently_expand() {
        let packages = parse_tree("candle-core v0.11.0|default,extra\n").unwrap();
        assert!(!selection_violations(&fixture(), &full(), &packages)
            .unwrap()
            .is_empty());
    }
    #[test]
    fn package_exception_never_overrides_denied_features() {
        let mut policy = fixture();
        policy.package_exceptions[0]
            .allowed_features
            .insert("cuda".into());
        let packages = parse_tree("candle-core v0.11.0|default,cuda\n").unwrap();
        let violations = selection_violations(&policy, &full(), &packages).unwrap();
        assert_eq!(violations.len(), 1);
        assert!(violations[0].contains("candle-core/cuda is enabled"));
    }
    #[test]
    fn package_exception_never_grants_native_approval() {
        let mut policy = fixture();
        policy.native.push("candle-core".into());
        let violations = selection_violations(&policy, &full(), &cpu()).unwrap();
        assert_eq!(violations.len(), 1);
        assert!(violations[0].contains("builds native code"));
    }
    #[test]
    fn wildcard_family_members_stay_denied() {
        let packages = parse_tree("candle-nn v0.11.0|default\n").unwrap();
        assert!(!selection_violations(&fixture(), &full(), &packages)
            .unwrap()
            .is_empty());
    }
    #[test]
    fn malformed_or_unscoped_exceptions_are_rejected() {
        for index in 0..5 {
            let mut policy = fixture();
            let e = &mut policy.package_exceptions[0];
            match index {
                0 => e.name = "candle-*".into(),
                1 => e.version = "^0.11".into(),
                2 => e.reason = " ".into(),
                3 => e.feature_sets.clear(),
                _ => e.feature_sets = vec![vec![]],
            }
            assert!(policy.validate().is_err());
        }
    }
    #[test]
    fn tree_parser_retains_features_on_repeats_and_refuses_missing_versions() {
        let packages =
            parse_tree("candle-core v0.11.0|default\ncandle-core v0.11.0|extra (*)\n").unwrap();
        assert_eq!(
            packages.values().next().unwrap(),
            &BTreeSet::from(["default".into(), "extra".into()])
        );
        assert!(parse_tree("candle-core|default\n").is_err());
    }
}
