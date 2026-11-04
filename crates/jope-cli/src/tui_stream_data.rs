//! Incremental gene-feed state and lazy source fetching for the gene-stream view.
//!
//! This is the Rust counterpart of `jope.tui_stream_data`. The Python TUI still imports its
//! original mixin; callers can adopt this state holder without changing that importer.

use serde_json::{Map, Value};
use std::path::PathBuf;

pub const MAX_GENES: usize = 2_000;
pub const MAX_CODE_CACHE: usize = 64;

/// Minimal interface implemented by the stream's incremental JSONL reader.
pub trait EventTailer {
    fn poll(&mut self) -> Vec<Value>;
}

/// Minimal interface implemented by a source emitter for one engine/canonical program pair.
pub trait SourceRouter {
    fn emit_one(&mut self, engine: &str, canonical: &str) -> Result<Value, String>;
}

#[derive(Clone, Debug, PartialEq)]
pub struct Gene {
    pub gene_id: String,
    pub engine: String,
    pub canonical: String,
    pub head: Option<String>,
    pub mu: Option<f64>,
    pub sigma: Option<f64>,
    pub p_bad: Option<f64>,
    pub oof: Option<f64>,
    pub test_r2: Option<f64>,
    pub state: String,
    pub total_ms: Option<f64>,
    pub seq: usize,
}

impl Gene {
    fn new(gene_id: String, seq: usize) -> Self {
        Self {
            gene_id,
            engine: String::new(),
            canonical: String::new(),
            head: None,
            mu: None,
            sigma: None,
            p_bad: None,
            oof: None,
            test_r2: None,
            state: "proposed".into(),
            total_ms: None,
            seq,
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct StreamState {
    pub cursor: usize,
    pub follow: bool,
    pub order: Vec<String>,
}

/// Mutable feed view state. Genes and the code cache retain arrival/LRU order.
pub struct StreamData<T, R> {
    pub run_dir: PathBuf,
    pub genes: Vec<Gene>,
    pub state: StreamState,
    pub tailer: T,
    pub router: R,
    pub meta_path: PathBuf,
    seq: usize,
    code_cache: Vec<((String, String, String), Value)>,
}

impl<T, R> StreamData<T, R>
where
    T: EventTailer,
    R: SourceRouter,
{
    pub fn new(run_dir: impl Into<PathBuf>, tailer: T, router: R) -> Self {
        let run_dir = run_dir.into();
        Self {
            meta_path: run_dir.join("gene_events.jsonl.meta.json"),
            run_dir,
            genes: Vec::new(),
            state: StreamState {
                follow: true,
                ..StreamState::default()
            },
            tailer,
            router,
            seq: 0,
            code_cache: Vec::new(),
        }
    }

    /// Find the newest per-generation training feed, then try the historical fixed names.
    pub fn auto_train_metrics(&self) -> std::io::Result<Option<PathBuf>> {
        let mut generations = Vec::new();
        for entry in std::fs::read_dir(&self.run_dir)? {
            let entry = entry?;
            let path = entry.path();
            let Some(name) = path.file_name().and_then(|name| name.to_str()) else {
                continue;
            };
            if name.starts_with("gen") && name.ends_with("_train_metrics.jsonl") {
                generations.push((entry.metadata()?.modified()?, path));
            }
        }
        generations.sort_by_key(|(modified, _)| *modified);
        if let Some((_, path)) = generations.pop() {
            return Ok(Some(path));
        }
        for name in [
            "train_metrics.jsonl",
            "policy_metrics.jsonl",
            "stream_metrics.jsonl",
            "metrics.jsonl",
        ] {
            let path = self.run_dir.join(name);
            if path.exists() {
                return Ok(Some(path));
            }
        }
        Ok(None)
    }

    /// Apply appended events, preserving the Python mixin's transitions and oldest-first cap.
    pub fn ingest(&mut self) {
        for event in self.tailer.poll() {
            let Some(gene_id) = event.get("gene_id").and_then(Value::as_str) else {
                continue;
            };
            if gene_id.is_empty() {
                continue;
            }
            let index = match self.genes.iter().position(|gene| gene.gene_id == gene_id) {
                Some(index) => index,
                None => {
                    self.seq += 1;
                    self.genes.push(Gene::new(gene_id.to_owned(), self.seq));
                    self.state.order.push(gene_id.to_owned());
                    self.genes.len() - 1
                }
            };
            let gene = &mut self.genes[index];
            set_truthy_string(&event, "engine", &mut gene.engine);
            set_truthy_string(&event, "canonical", &mut gene.canonical);
            set_optional_string(&event, "head", &mut gene.head);
            match event.get("event").and_then(Value::as_str) {
                Some("proposed") => {
                    set_optional_number(&event, "mu", &mut gene.mu);
                    set_optional_number(&event, "sigma", &mut gene.sigma);
                    set_optional_number(&event, "p_bad", &mut gene.p_bad);
                }
                Some("selected") if gene.state == "proposed" => gene.state = "selected".into(),
                Some("scored") => {
                    gene.state = "scored".into();
                    gene.oof = fallback_number(&event, "oof_r2", "oof_auc", gene.oof);
                    gene.test_r2 = fallback_number(&event, "test_r2", "test_auc", gene.test_r2);
                    set_optional_number(&event, "total_ms", &mut gene.total_ms);
                }
                _ => {}
            }
        }
        while self.genes.len() > MAX_GENES {
            let old = self.genes.remove(0).gene_id;
            if let Some(position) = self.state.order.iter().position(|gene_id| gene_id == &old) {
                self.state.order.remove(position);
            }
        }
        if self.state.follow && !self.state.order.is_empty() {
            self.state.cursor = self.state.order.len() - 1;
        }
        self.state.cursor = self
            .state
            .cursor
            .min(self.state.order.len().saturating_sub(1));
    }

    /// Read the small sidecar; missing, unreadable, or malformed JSON is an empty object.
    pub fn meta(&self) -> Value {
        std::fs::read_to_string(&self.meta_path)
            .ok()
            .and_then(|text| serde_json::from_str(&text).ok())
            .unwrap_or_else(|| Value::Object(Map::new()))
    }

    /// Return a cached emission, or fetch it lazily and retain the newest 64 requests.
    pub fn code_for(&mut self, gene_id: &str, lang: &str) -> Value {
        let Some(index) = self.genes.iter().position(|gene| gene.gene_id == gene_id) else {
            return unsupported("no gene selected");
        };
        let gene = &self.genes[index];
        let key = (gene.engine.clone(), gene.canonical.clone(), lang.to_owned());
        if let Some(position) = self
            .code_cache
            .iter()
            .position(|(cached, _)| cached == &key)
        {
            let entry = self.code_cache.remove(position);
            let value = entry.1.clone();
            self.code_cache.push(entry);
            return value;
        }
        if key.0.is_empty() || key.1.is_empty() {
            return unsupported("no gene selected");
        }
        let emitted = self
            .router
            .emit_one(&key.0, &key.1)
            .unwrap_or_else(|error| unsupported(&error));
        self.code_cache.push((key, emitted.clone()));
        if self.code_cache.len() > MAX_CODE_CACHE {
            self.code_cache.remove(0);
        }
        emitted
    }
}

fn unsupported(message: &str) -> Value {
    serde_json::json!({"supported": false, "errors": [message]})
}

fn set_truthy_string(event: &Value, key: &str, target: &mut String) {
    if let Some(value) = event
        .get(key)
        .and_then(Value::as_str)
        .filter(|value| !value.is_empty())
    {
        *target = value.to_owned();
    }
}

fn set_optional_string(event: &Value, key: &str, target: &mut Option<String>) {
    if let Some(value) = event.get(key) {
        *target = value.as_str().map(str::to_owned);
    }
}

fn set_optional_number(event: &Value, key: &str, target: &mut Option<f64>) {
    if let Some(value) = event.get(key) {
        *target = value.as_f64();
    }
}

fn fallback_number(
    event: &Value,
    primary: &str,
    fallback: &str,
    current: Option<f64>,
) -> Option<f64> {
    if let Some(value) = event.get(primary) {
        value.as_f64()
    } else if let Some(value) = event.get(fallback) {
        value.as_f64()
    } else {
        current
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;

    struct FixedTailer(Vec<Value>);
    impl EventTailer for FixedTailer {
        fn poll(&mut self) -> Vec<Value> {
            std::mem::take(&mut self.0)
        }
    }

    struct CountingRouter(Cell<usize>);
    impl SourceRouter for CountingRouter {
        fn emit_one(&mut self, engine: &str, canonical: &str) -> Result<Value, String> {
            self.0.set(self.0.get() + 1);
            Ok(serde_json::json!({"supported": true, "engine": engine, "canonical": canonical}))
        }
    }

    #[test]
    fn ingest_preserves_gene_order_transitions_and_follow_cursor() {
        let events = vec![
            serde_json::json!({"gene_id":"g1","event":"proposed","engine":"e","canonical":"c","mu":0.4}),
            serde_json::json!({"gene_id":"g1","event":"selected"}),
            serde_json::json!({"gene_id":"g1","event":"scored","oof_r2":0.7,"total_ms":12}),
            serde_json::json!({"gene_id":"g2","event":"proposed"}),
        ];
        let mut data = StreamData::new(".", FixedTailer(events), CountingRouter(Cell::new(0)));
        data.ingest();
        assert_eq!(data.state.order, ["g1", "g2"]);
        assert_eq!(data.state.cursor, 1);
        assert_eq!(data.genes[0].state, "scored");
        assert_eq!(data.genes[0].oof, Some(0.7));
        assert_eq!(data.genes[0].total_ms, Some(12.0));
    }

    #[test]
    fn code_fetch_is_cached_and_empty_gene_is_refused() {
        let events = vec![
            serde_json::json!({"gene_id":"g1","event":"proposed","engine":"e","canonical":"c"}),
        ];
        let mut data = StreamData::new(".", FixedTailer(events), CountingRouter(Cell::new(0)));
        data.ingest();
        assert_eq!(
            data.code_for("missing", "python")["errors"][0],
            "no gene selected"
        );
        let first = data.code_for("g1", "python");
        let second = data.code_for("g1", "python");
        assert_eq!(first, second);
        assert_eq!(data.router.0.get(), 1);
    }
}
