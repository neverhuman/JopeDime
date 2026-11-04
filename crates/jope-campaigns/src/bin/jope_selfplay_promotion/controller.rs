use super::*;

// ---------------------------------------------------------------------------
// Controller state machine
// ---------------------------------------------------------------------------

pub(super) struct Controller {
    root: PathBuf,
    production: bool,
    journal: ActionJournal,
    state_path: PathBuf,
}

impl Controller {
    pub(super) fn open(root: &Path) -> Result<Self> {
        Ok(Self {
            root: root.to_path_buf(),
            production: root == Path::new(root_str()),
            journal: ActionJournal::open(root)?,
            state_path: root.join("ops/flywheel-state.json"),
        })
    }

    pub(super) fn state(&self) -> Result<Value> {
        if self.state_path.is_file() {
            return read_json(&self.state_path);
        }
        Ok(json!({
            "schema": "JopeSelfplayControllerState.v1",
            "last_frozen_watermark": 0,
            "consecutive_rejections": 0,
            "tournament": null,
        }))
    }

    pub(super) fn cycle(&self, snapshot: &Value) -> Result<Value> {
        let entries = snapshot
            .get("catalog_entries")
            .and_then(Value::as_array)
            .ok_or_else(|| anyhow!("snapshot has no catalog entries"))?;
        let declared: BTreeMap<String, i64> = snapshot
            .get("catalog_counts")
            .and_then(Value::as_object)
            .map(|map| {
                map.iter()
                    .map(|(k, v)| (k.clone(), v.as_i64().unwrap_or(-1)))
                    .collect()
            })
            .ok_or_else(|| anyhow!("snapshot has no catalog counts"))?;
        let counts = validate_catalog_counts(entries, &declared)?;
        let mut state = self.state()?;
        let rejections = state
            .get("consecutive_rejections")
            .and_then(Value::as_i64)
            .unwrap_or(0);
        // Legacy unstarted tournaments used three competing recipes.
        let needs_normalization = {
            let tournament = state.get("tournament").filter(|t| !t.is_null());
            match tournament {
                Some(tournament) => {
                    tournament
                        .get("phase_index")
                        .and_then(Value::as_i64)
                        .unwrap_or(0)
                        == 0
                        && tournament
                            .get("artifacts")
                            .and_then(Value::as_object)
                            .map(|a| a.is_empty())
                            .unwrap_or(true)
                        && tournament
                            .get("recipes")
                            .and_then(Value::as_array)
                            .map(|r| r.len() != 1)
                            .unwrap_or(true)
                }
                None => false,
            }
        };
        if needs_normalization {
            let full_consolidation = rejections >= 5;
            let tournament = state.get_mut("tournament").unwrap();
            tournament["recipes"] = json!([if full_consolidation { "a" } else { "c" }]);
            tournament["corpus_mode"] = json!(if full_consolidation {
                "full_consolidation"
            } else {
                "recent_plus_replay"
            });
            state["consecutive_rejections"] = json!(rejections);
            atomic_json(&self.state_path, &state)?;
        }
        if state.get("tournament").map(Value::is_null).unwrap_or(true) {
            let labels = get_i64(snapshot, "valid_labels")?;
            let last_frozen = state
                .get("last_frozen_watermark")
                .and_then(Value::as_i64)
                .unwrap_or(0);
            if labels - last_frozen < LABEL_THRESHOLD {
                let mut catalog_counts = Map::new();
                for (task, count) in counts {
                    catalog_counts.insert(task, json!(count));
                }
                return Ok(json!({
                    "phase": "collecting_labels",
                    "valid_labels": labels,
                    "next_watermark": last_frozen + LABEL_THRESHOLD,
                    "catalog_counts": catalog_counts,
                }));
            }
            let full_consolidation = labels / 500_000 > last_frozen / 500_000 || rejections >= 5;
            let tournament = json!({
                "watermark": labels,
                "accepted_ancestor": snapshot["accepted_ancestor"],
                "catalog_sha256": snapshot["catalog_sha256"],
                "evaluator_sha256": snapshot["evaluator_sha256"],
                "phase_index": 0,
                "artifacts": {},
                "recipes": [if full_consolidation { "a" } else { "c" }],
                "corpus_mode": if full_consolidation {
                    "full_consolidation"
                } else {
                    "recent_plus_replay"
                },
                "seed": DEFAULT_SEED,
            });
            state["last_frozen_watermark"] = json!(labels);
            state["tournament"] = tournament;
            atomic_json(&self.state_path, &state)?;
        }
        let tournament = state["tournament"].clone();
        let phase_index = get_i64(&tournament, "phase_index")? as usize;
        let phase = PHASES[phase_index];
        let mut request = tournament.clone();
        request["action"] = json!(phase);
        let watermark = get_i64(&tournament, "watermark")?;
        let key = action_key(
            phase,
            watermark,
            get_str(&tournament, "accepted_ancestor")?,
            if phase == "launch_checkpoint_tournament" {
                "a+b+c"
            } else {
                "tournament"
            },
            get_i64(&tournament, "seed")?,
            get_str(&tournament, "catalog_sha256")?,
            get_str(&tournament, "evaluator_sha256")?,
        )?;
        let root = self.root.clone();
        let production = self.production;
        let outcome = self.journal.perform(&key, phase, &request, |req| {
            dispatch_phase(phase, req, &root, production)
        });
        let result = match outcome {
            Ok(result) => result,
            Err(error) => {
                if let Some(detail) = as_pending(&error) {
                    return Ok(json!({
                        "phase": phase,
                        "pending": true,
                        "detail": detail.to_string(),
                        "watermark": tournament["watermark"],
                    }));
                }
                return Err(error);
            }
        };
        {
            let tournament = state.get_mut("tournament").unwrap();
            tournament["artifacts"][phase] = result.clone();
            let gate_failed =
                phase == "run_checkpoint_gate" && result.get("pass") == Some(&Value::Bool(false));
            if gate_failed {
                tournament["artifacts"]["advance_accepted_pointer"] = json!({
                    "advanced": false,
                    "checkpoint_sha256": tournament["accepted_ancestor"],
                });
                tournament["phase_index"] = json!(phase_index as i64 + 2);
            } else {
                tournament["phase_index"] = json!(phase_index as i64 + 1);
            }
        }
        let final_index = state
            .pointer("/tournament/phase_index")
            .and_then(Value::as_i64)
            .unwrap_or(0);
        if final_index >= PHASES.len() as i64 {
            let tournament = state["tournament"].clone();
            let gate = tournament
                .pointer("/artifacts/run_checkpoint_gate")
                .cloned()
                .unwrap_or(json!({}));
            let terminal = if gate.get("pass") == Some(&Value::Bool(true)) {
                "accepted"
            } else {
                "rejected"
            };
            let mut completed = tournament.clone();
            completed["terminal"] = json!(terminal);
            completed["completed_at_ms"] = json!(now_ms());
            let completed_path = self
                .root
                .join("training/tournaments")
                .join(format!("watermark-{watermark:09}.json"));
            atomic_json(&completed_path, &completed)?;
            state["tournament"] = Value::Null;
            state["last_terminal"] = json!(terminal);
            state["last_tournament"] = json!(completed_path.to_string_lossy());
            state["consecutive_rejections"] = json!(if terminal == "accepted" {
                0
            } else {
                state
                    .get("consecutive_rejections")
                    .and_then(Value::as_i64)
                    .unwrap_or(0)
                    + 1
            });
        }
        atomic_json(&self.state_path, &state)?;
        Ok(json!({
            "phase": phase,
            "completed": true,
            "result": result,
            "watermark": tournament["watermark"],
        }))
    }
}

pub(super) fn live_snapshot(root: &Path) -> Result<Value> {
    let pointer_path = root.join("catalog/current.json");
    let authority = validate_catalog_manifest(&pointer_path)?;
    let pointer = read_json(&pointer_path)?;
    let catalog = read_json(Path::new(get_str(&pointer, "catalog")?))?;
    let corpus = Connection::open_with_flags(
        root.join("corpus/strict-test-exact.sqlite"),
        rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY,
    )?;
    corpus.busy_timeout(std::time::Duration::from_secs(60))?;
    let labels: i64 =
        corpus.query_row("SELECT COUNT(*) FROM observations", [], |row| row.get(0))?;
    drop(corpus);
    let (_, ancestor) = accepted_checkpoint()?;
    Ok(json!({
        "valid_labels": labels,
        "accepted_ancestor": ancestor,
        "catalog_sha256": pointer["catalog_sha256"],
        "catalog_entries": catalog["entries"],
        "catalog_counts": authority["task_counts"],
        "evaluator_sha256": sha256_file(Path::new(evaluator_str()))?,
    }))
}
