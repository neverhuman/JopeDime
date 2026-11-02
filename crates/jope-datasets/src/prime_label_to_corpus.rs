//! Grammar selection used when preparing Prime realization labels for training.
//!
//! The Python adapter asks the policy tokenizer whether a genome can be encoded under each
//! grammar. This crate deliberately does not depend on the policy crate; callers supply that
//! same acceptance check while this type preserves the adapter's priority, hint, and cache rules.

use std::collections::{HashMap, HashSet};

/// Grammar engines tried when no valid grammar hint is supplied.
pub const GRAMMAR_ENGINE_PRIORITY: [&str; 3] = ["featmath", "newmath", "prime"];

/// Resolve an encodable grammar engine, caching results per `(task_type, genome)`.
#[derive(Debug, Clone)]
pub struct GenomeEngineResolver {
    priority: Vec<String>,
    cache: HashMap<(String, String), Option<String>>,
}

impl GenomeEngineResolver {
    /// Keep only prioritized engines present in the supplied grammar vocabulary.
    pub fn new<'a>(vocabulary_engines: impl IntoIterator<Item = &'a str>) -> Self {
        let available: HashSet<&str> = vocabulary_engines.into_iter().collect();
        let priority = GRAMMAR_ENGINE_PRIORITY
            .iter()
            .copied()
            .filter(|engine| available.contains(engine))
            .map(str::to_owned)
            .collect();
        Self {
            priority,
            cache: HashMap::new(),
        }
    }

    /// Grammar engines retained from the supplied vocabulary, in probe order.
    pub fn priority(&self) -> &[String] {
        &self.priority
    }

    /// Return the first grammar accepted by `can_encode`.
    ///
    /// A valid `declared_engine` is tried first, followed by the remaining configured priority.
    /// Empty genomes and genomes rejected by every engine resolve to `None`; both outcomes are
    /// cached using the same task-and-genome identity as the Python adapter.
    pub fn resolve<F>(
        &mut self,
        genome: &str,
        declared_engine: Option<&str>,
        task_type: &str,
        mut can_encode: F,
    ) -> Option<String>
    where
        F: FnMut(&str, &str, &str) -> bool,
    {
        if genome.is_empty() {
            return None;
        }

        let key = (task_type.to_owned(), genome.to_owned());
        if let Some(cached) = self.cache.get(&key) {
            return cached.clone();
        }

        let mut order = Vec::with_capacity(self.priority.len());
        if let Some(hint) =
            declared_engine.filter(|engine| self.priority.iter().any(|e| e == *engine))
        {
            order.push(hint.to_owned());
        }
        order.extend(
            self.priority
                .iter()
                .filter(|engine| Some(engine.as_str()) != declared_engine)
                .cloned(),
        );

        let resolved = order
            .into_iter()
            .find(|engine| can_encode(engine, genome, task_type));
        self.cache.insert(key, resolved.clone());
        resolved
    }
}

#[cfg(test)]
mod tests {
    use super::GenomeEngineResolver;
    use std::cell::Cell;

    #[test]
    fn priority_filters_vocabulary_and_tries_in_declared_order() {
        let mut resolver = GenomeEngineResolver::new(["prime", "featmath", "other"]);
        assert_eq!(
            resolver.priority(),
            &["featmath".to_owned(), "prime".to_owned()]
        );
        let seen = Cell::new(0);
        let resolved = resolver.resolve("(pipe X)", Some("prime"), "regression", |engine, _, _| {
            seen.set(seen.get() + 1);
            engine == "prime"
        });
        assert_eq!(resolved.as_deref(), Some("prime"));
        assert_eq!(seen.get(), 1);
    }

    #[test]
    fn default_priority_and_task_scoped_cache_match_adapter_rules() {
        let mut resolver = GenomeEngineResolver::new(["featmath", "newmath", "prime"]);
        let seen = Cell::new(0);
        let accepts_featmath = |engine: &str, _: &str, _: &str| {
            seen.set(seen.get() + 1);
            engine == "newmath"
        };
        assert_eq!(
            resolver
                .resolve("genome", None, "regression", accepts_featmath)
                .as_deref(),
            Some("newmath")
        );
        assert_eq!(seen.get(), 2);
        assert_eq!(
            resolver
                .resolve("genome", None, "regression", |_, _, _| panic!(
                    "cached result"
                ))
                .as_deref(),
            Some("newmath")
        );
        assert_eq!(
            resolver
                .resolve("genome", None, "binary", |engine, _, _| engine == "prime")
                .as_deref(),
            Some("prime")
        );
    }

    #[test]
    fn empty_and_unresolvable_genomes_have_expected_results() {
        let mut resolver = GenomeEngineResolver::new(["featmath"]);
        assert_eq!(
            resolver.resolve("", None, "regression", |_, _, _| panic!("empty genome")),
            None
        );
        assert_eq!(
            resolver.resolve("bad", None, "regression", |_, _, _| false),
            None
        );
        assert_eq!(
            resolver.resolve("bad", None, "regression", |_, _, _| panic!("cached miss")),
            None
        );
    }
}
