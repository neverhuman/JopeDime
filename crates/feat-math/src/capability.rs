//! Language admission for pipeline recipes, shared by proposal and plan consumers.
//!
//! This is recipe coverage, not fitted-state export or numerical parity qualification.
//! Python/Rust fitted glue, calibration and categorical input lowering have separate
//! export gates. TypeScript means ES2020 plus declarations, never the Rust/Wasm bridge.

use crate::kernel::capabilities::{Manifest, OpKind};
use crate::{transpile, GenomeIr};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

pub const SCHEMA: &str = "jope.capability-registry/v1";
pub const LANGUAGE_INTERSECTION_EMPTY: &str = "LANGUAGE_INTERSECTION_EMPTY";

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Language {
    Rust,
    Python,
    Javascript,
    Typescript,
}

impl Language {
    pub const ALL: [Self; 4] = [Self::Rust, Self::Python, Self::Javascript, Self::Typescript];
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CapabilityKind {
    Operation,
    Head,
    Calibration,
    Encoding,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct Capability {
    pub kind: CapabilityKind,
    pub name: &'static str,
    pub supported: BTreeMap<Language, bool>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct Registry {
    pub schema: &'static str,
    pub coverage: &'static str,
    pub entries: Vec<Capability>,
}

impl Registry {
    /// Project the existing emitter inventory without changing its frozen vocabulary.
    pub fn current() -> Self {
        let mut entries = Vec::new();
        for op in Manifest::current().ops {
            let kind = match op.kind {
                OpKind::Head => CapabilityKind::Head,
                OpKind::Transform | OpKind::Target | OpKind::Directive => CapabilityKind::Operation,
                OpKind::Input => continue,
            };
            entries.push(Capability {
                kind,
                name: op.name,
                supported: BTreeMap::from([
                    (Language::Rust, op.emitters.rust_recipe.accepts()),
                    (Language::Python, op.emitters.python_recipe.accepts()),
                    (Language::Javascript, op.emitters.javascript.accepts()),
                    // EXP-06 owns declarations; a different TypeScript bridge exists.
                    (Language::Typescript, false),
                ]),
            });
        }
        // Absence of calibration and dense numeric input are the recipe baseline.
        // These rows do not certify the separate fitted-portable input representation.
        for (kind, name, supported) in [
            (CapabilityKind::Calibration, "none", true),
            (CapabilityKind::Calibration, "affine_floor", false),
            (CapabilityKind::Calibration, "platt", false),
            (CapabilityKind::Calibration, "isotonic", false),
            (CapabilityKind::Encoding, "dense_numeric", true),
            (CapabilityKind::Encoding, "categorical_onehot", false),
            (CapabilityKind::Encoding, "categorical_target", false),
            (CapabilityKind::Encoding, "missing_aware", false),
        ] {
            entries.push(Capability {
                kind,
                name,
                supported: Language::ALL
                    .into_iter()
                    .map(|language| (language, supported && language != Language::Typescript))
                    .collect(),
            });
        }
        entries.sort_by_key(|row| (row.kind, row.name));
        Self {
            schema: SCHEMA,
            coverage: "pipeline_recipe",
            entries,
        }
    }

    /// Unknown names fail closed. Aliases have their own inventory rows.
    pub fn supports(&self, kind: CapabilityKind, name: &str, language: Language) -> bool {
        self.entries
            .iter()
            .find(|row| row.kind == kind && row.name == name)
            .and_then(|row| row.supported.get(&language))
            .copied()
            .unwrap_or(false)
    }

    /// Compute proposal admission and its plan-visible complement together.
    /// A head-only pipeline is valid; a transform-only intersection is not.
    pub fn allowed_ops(
        &self,
        languages: &[Language],
    ) -> Result<LanguageFilter, LanguageIntersectionEmpty> {
        let languages: Vec<_> = languages
            .iter()
            .copied()
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect();
        if languages.is_empty() {
            return Err(LanguageIntersectionEmpty);
        }
        let mut allowed_ops = Vec::new();
        let mut excluded_ops = Vec::new();
        let mut has_head = false;
        for row in &self.entries {
            if !matches!(row.kind, CapabilityKind::Operation | CapabilityKind::Head) {
                continue;
            }
            if languages
                .iter()
                .all(|language| row.supported.get(language) == Some(&true))
            {
                allowed_ops.push(row.name);
                has_head |= row.kind == CapabilityKind::Head;
            } else {
                excluded_ops.push(row.name);
            }
        }
        if !has_head {
            return Err(LanguageIntersectionEmpty);
        }
        allowed_ops.sort_unstable();
        excluded_ops.sort_unstable();
        Ok(LanguageFilter {
            languages,
            allowed_ops,
            excluded_ops,
        })
    }
}

/// Default recipe registry entry point for the proposer.
pub fn allowed_ops(languages: &[Language]) -> Result<LanguageFilter, LanguageIntersectionEmpty> {
    Registry::current().allowed_ops(languages)
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct LanguageFilter {
    languages: Vec<Language>,
    /// Sorted operation and head names; independent grammar/task constraints still apply.
    allowed_ops: Vec<&'static str>,
    /// Sorted complement for the plan's `languages.excluded_operations`.
    excluded_ops: Vec<&'static str>,
}

impl LanguageFilter {
    pub fn languages(&self) -> &[Language] {
        &self.languages
    }

    pub fn allowed_ops(&self) -> &[&'static str] {
        &self.allowed_ops
    }

    pub fn excluded_ops(&self) -> &[&'static str] {
        &self.excluded_ops
    }

    pub fn allows(&self, op: &str) -> bool {
        self.allowed_ops.binary_search(&op).is_ok()
    }

    /// Filter pipeline seeds using the same names AND the actual emitter predicates.
    /// This also refuses unsupported forms and parameter-dependent lowering failures.
    pub fn accepts_seed(&self, genome: &GenomeIr) -> bool {
        let GenomeIr::Pipeline(pipeline) = genome else {
            return false;
        };
        pipeline.stages.iter().all(|stage| self.allows(&stage.op))
            && self.languages.iter().all(|language| match language {
                Language::Rust | Language::Python => transpile::supports(genome).is_ok(),
                Language::Javascript => transpile::supports_javascript(genome).is_ok(),
                Language::Typescript => false,
            })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LanguageIntersectionEmpty;

impl fmt::Display for LanguageIntersectionEmpty {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(LANGUAGE_INTERSECTION_EMPTY)
    }
}

impl std::error::Error for LanguageIntersectionEmpty {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::full_rust;

    const THREE: [Language; 3] = [Language::Rust, Language::Python, Language::Javascript];

    #[test]
    fn three_language_intersection_and_exclusions_are_exact() {
        let registry = Registry::current();
        let filter = allowed_ops(&THREE).unwrap();
        let mut total = 0;
        for row in &registry.entries {
            if !matches!(row.kind, CapabilityKind::Operation | CapabilityKind::Head) {
                continue;
            }
            let supported = THREE
                .iter()
                .all(|&language| registry.supports(row.kind, row.name, language));
            assert_eq!(filter.allows(row.name), supported, "{}", row.name);
            assert_eq!(filter.excluded_ops().contains(&row.name), !supported);
            total += 1;
        }
        assert_eq!(
            filter.allowed_ops().len() + filter.excluded_ops().len(),
            total
        );
        assert!(filter.allows("ridge"));
        assert!(filter.allows("elasticnet"));
        assert!(filter.allows("standardize"));
        assert!(!filter.allows("gpr"));
        assert!(!filter.allows("roll"));
        assert!(!filter.allows("not_an_operation"));
        assert_eq!(
            filter,
            allowed_ops(&[
                Language::Javascript,
                Language::Rust,
                Language::Python,
                Language::Rust
            ])
            .unwrap()
        );
        let rust_python = allowed_ops(&[Language::Rust, Language::Python]).unwrap();
        assert!(rust_python.allows("gpr"));
        assert!(rust_python.allows("roll"));
    }

    #[test]
    fn impossible_or_empty_requests_have_stable_error() {
        for languages in [&[][..], &[Language::Typescript], &Language::ALL] {
            let error = allowed_ops(languages).unwrap_err();
            assert_eq!(error.to_string(), LANGUAGE_INTERSECTION_EMPTY);
        }
        let mut registry = Registry::current();
        registry
            .entries
            .retain(|row| row.kind != CapabilityKind::Head);
        assert_eq!(registry.allowed_ops(&THREE), Err(LanguageIntersectionEmpty));
        registry.entries = Registry::current()
            .entries
            .into_iter()
            .filter(|row| row.kind == CapabilityKind::Head)
            .collect();
        assert!(registry.allowed_ops(&THREE).unwrap().allows("ridge"));
    }

    #[test]
    fn coverage_agrees_with_actual_recipe_emitters() {
        let registry = Registry::current();
        let mut probes = BTreeSet::new();
        for row in &registry.entries {
            let source = match (row.kind, row.name) {
                (CapabilityKind::Head, name) => format!("(pipe X ({name}))"),
                (CapabilityKind::Operation, "target") => {
                    "(pipe X (target kind=log1p) (ridge))".to_string()
                }
                (CapabilityKind::Operation, name @ ("map" | "derive")) => {
                    format!("(pipe X ({name} expr=\"c0\") (ridge))")
                }
                (CapabilityKind::Operation, name) => format!("(pipe X ({name}) (ridge))"),
                _ => continue,
            };
            let genome = GenomeIr::parse(&source).unwrap();
            // Exercise every declared operation/head, including aliases and refusals,
            // through real emitters rather than comparing two declaration tables.
            let emitted = transpile::emit(&genome, &transpile::EmitOptions::default()).is_ok();
            for language in [Language::Rust, Language::Python] {
                assert_eq!(
                    registry.supports(row.kind, row.name, language),
                    emitted,
                    "{source}"
                );
            }
            assert_eq!(
                registry.supports(row.kind, row.name, Language::Javascript),
                transpile::supports_javascript(&genome).is_ok(),
                "{source}"
            );
            probes.insert(row.name);
        }
        for schema in full_rust::operation_schemas() {
            assert!(
                probes.contains(schema.op),
                "missing canonical recipe {}",
                schema.op
            );
        }
        assert!(probes.len() >= 48);
    }

    #[test]
    fn seed_filter_checks_names_forms_and_parameters() {
        let filter = allowed_ops(&THREE).unwrap();
        for source in [
            "(pipe X (ridge))",
            "(pipe X (standardize) (elasticnet))",
            "(pipe X (norm_features) (logistic))",
        ] {
            assert!(
                filter.accepts_seed(&GenomeIr::parse(source).unwrap()),
                "{source}"
            );
        }
        for source in [
            "(pipe X (gpr))",
            "(pipe X (roll) (ridge))",
            "(pipe X (target kind=log1p) (ridge))",
            "(pipe X (unknown) (ridge))",
        ] {
            assert!(
                !filter.accepts_seed(&GenomeIr::parse(source).unwrap()),
                "{source}"
            );
        }
        let rust = allowed_ops(&[Language::Rust]).unwrap();
        assert!(rust.accepts_seed(&GenomeIr::parse("(pipe X (map expr=\"c0\") (ridge))").unwrap()));
        // The name is allowed in Rust, but its required expression is absent.
        assert!(!rust.accepts_seed(&GenomeIr::parse("(pipe X (map) (ridge))").unwrap()));
        assert!(!rust.accepts_seed(&GenomeIr::parse("(sym affine \"c0\")").unwrap()));
    }

    #[test]
    fn schema_covers_all_axes_without_claiming_future_export_gates() {
        let registry = Registry::current();
        assert_eq!(registry.schema, "jope.capability-registry/v1");
        assert_eq!(registry.coverage, "pipeline_recipe");
        let keys: BTreeSet<_> = registry.entries.iter().map(|r| (r.kind, r.name)).collect();
        assert_eq!(keys.len(), registry.entries.len());
        for row in &registry.entries {
            assert_eq!(row.supported.len(), Language::ALL.len());
        }
        for language in THREE {
            assert!(registry.supports(CapabilityKind::Calibration, "none", language));
            assert!(registry.supports(CapabilityKind::Encoding, "dense_numeric", language));
            for name in ["affine_floor", "platt", "isotonic"] {
                assert!(!registry.supports(CapabilityKind::Calibration, name, language));
            }
            assert!(!registry.supports(CapabilityKind::Encoding, "categorical_onehot", language));
            assert!(!registry.supports(CapabilityKind::Encoding, "unknown", language));
        }
        let json = serde_json::to_value(&registry).unwrap();
        assert_eq!(json["schema"], SCHEMA);
        assert_eq!(json["coverage"], "pipeline_recipe");
        assert_eq!(json["entries"][0]["kind"], "operation");
        assert_eq!(
            serde_json::from_str::<Language>("\"javascript\"").unwrap(),
            Language::Javascript
        );
        assert!(serde_json::from_str::<Language>("\"cuda\"").is_err());
        assert!(serde_json::from_str::<Language>("\"wasm\"").is_err());
    }
}
