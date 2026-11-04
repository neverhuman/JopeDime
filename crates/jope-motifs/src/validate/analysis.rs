use super::*;

pub(super) struct Artifact {
    pub(super) kind: &'static str,
    pub(super) text: String,
}

/// Fitted state and probe predictions of one fit, as bytes and bit patterns.
#[derive(Clone, Debug)]
pub(super) struct Fitted {
    pub(super) model: BoundModel,
    pub(super) state: Vec<u8>,
    pub(super) predictions: Vec<u64>,
    pub(super) probabilities: Vec<Vec<u64>>,
}

impl Fitted {
    pub(super) fn same(&self, other: &Self) -> bool {
        self.state == other.state
            && self.predictions == other.predictions
            && self.probabilities == other.probabilities
    }
}

pub(super) enum Outcome {
    Refused,
    Fitted(Box<Fitted>),
    Nondeterministic,
}

/// Fit on one synthetic table and predict its probe. Probe predictions must be finite, and
/// multiclass probabilities must be valid (the engine refuses invalid ones).
pub(super) fn fit_once(ir: &GenomeIr, task: TaskLabel, data: &Synthetic) -> Result<Fitted> {
    let spec = task.task_spec()?;
    let model = fit_table(ir, &spec, &data.schema, &data.train, data.targets(task))?;
    let state = serde_json::to_vec(&model)?;
    let predictions = predict_table(&model, &data.probe)?;
    ensure!(
        predictions.iter().all(|value| value.is_finite()),
        "non-finite probe prediction"
    );
    let probabilities: Vec<Vec<u64>> = if task == TaskLabel::Multiclass {
        predict_multiclass_table(&model, &data.probe)?
            .probabilities
            .iter()
            .map(|row| row.iter().map(|value| value.to_bits()).collect())
            .collect()
    } else {
        Vec::new()
    };
    Ok(Fitted {
        predictions: predictions.iter().map(|value| value.to_bits()).collect(),
        model,
        state,
        probabilities,
    })
}

/// Fit twice and compare fitted state bytes and prediction bits.
pub(super) fn fit_twice(ir: &GenomeIr, task: TaskLabel, data: &Synthetic) -> Outcome {
    let Ok(first) = fit_once(ir, task, data) else {
        return Outcome::Refused;
    };
    match fit_once(ir, task, data) {
        Ok(second) if first.same(&second) => Outcome::Fitted(Box::new(first)),
        _ => Outcome::Nondeterministic,
    }
}

pub(super) fn data(tables: &[Synthetic], shape: Shape) -> Result<&Synthetic> {
    tables
        .iter()
        .find(|table| table.shape == shape)
        .ok_or(Code("io"))
        .map_err(Into::into)
}

#[derive(Clone, Copy, Debug, Default)]
pub(super) struct LegacyCoverage {
    pub(super) plan: bool,
    pub(super) cpp17: bool,
    pub(super) javascript: bool,
}

#[derive(Clone, Copy, Debug, Default)]
pub(super) struct BoundCoverage {
    pub(super) bound_emit: bool,
    pub(super) fitted_cpp17: bool,
}

/// Legacy training-program artifacts: Python, Rust and manifest with the default and the
/// fast-runtime profile, plus C++17 and JavaScript where covered.
pub(super) fn legacy_artifacts(ir: &GenomeIr, out: &mut Vec<Artifact>) -> LegacyCoverage {
    let plan = transpile::supports(ir).is_ok();
    if plan {
        for fast_runtime in [false, true] {
            let options = EmitOptions {
                fast_runtime,
                ..EmitOptions::default()
            };
            if let Ok(emitted) = transpile::emit(ir, &options) {
                out.push(Artifact {
                    kind: "legacy_python",
                    text: emitted.python,
                });
                out.push(Artifact {
                    kind: "legacy_rust",
                    text: emitted.rust,
                });
                out.push(Artifact {
                    kind: "legacy_manifest",
                    text: emitted.manifest,
                });
            }
        }
    }
    let cpp17 = transpile::supports_cpp17(ir).is_ok();
    if let Ok(source) = transpile::emit_cpp17(ir, &EmitOptions::default()) {
        out.push(Artifact {
            kind: "legacy_cpp17",
            text: source,
        });
    }
    let javascript = transpile::supports_javascript(ir).is_ok();
    if let Ok(source) = transpile::emit_javascript(ir, &EmitOptions::default()) {
        out.push(Artifact {
            kind: "legacy_javascript",
            text: source,
        });
    }
    LegacyCoverage {
        plan,
        cpp17,
        javascript,
    }
}

/// Artifacts of one fitted model: the bound prediction program and the fitted C++17 source
/// and codec, where supported.
pub(super) fn bound_artifacts(model: &BoundModel, out: &mut Vec<Artifact>) -> BoundCoverage {
    let bound_emit = match emit_bound(model, &EmitOptions::default()) {
        Ok(emitted) => {
            out.push(Artifact {
                kind: "bound_python",
                text: emitted.python,
            });
            out.push(Artifact {
                kind: "bound_rust",
                text: emitted.rust,
            });
            out.push(Artifact {
                kind: "bound_manifest",
                text: emitted.manifest,
            });
            true
        }
        Err(_) => false,
    };
    let fitted_cpp17 = supports_fitted_cpp17(model).is_ok();
    if let Ok(artifact) = emit_fitted_cpp17(model) {
        out.push(Artifact {
            kind: "fitted_cpp17_source",
            text: artifact.source,
        });
        out.push(Artifact {
            kind: "fitted_cpp17_codec",
            text: artifact.codec,
        });
    }
    BoundCoverage {
        bound_emit,
        fitted_cpp17,
    }
}

/// C1: for every recorded occurrence, rewrite at its offset and everywhere, expand, and
/// require the host bytes and recipe digest back. The unexpanded text must be refused by the
/// slot contract and the schema-bound grammar.
pub(super) fn round_trip(table: &MotifTable, motif: &MotifDef) -> Result<CheckResult> {
    let schema = classification_schema();
    let mut result = CheckResult::default();
    for occurrence in &motif.provenance {
        let host = table
            .body
            .hosts
            .get(&occurrence.recipe_sha256)
            .ok_or(Code("provenance_inconsistent"))?;
        let offset = usize::try_from(occurrence.stage_offset)?;
        let single = rewrite(host, motif, At::Offset(offset));
        let every = rewrite(host, motif, At::All);
        for (label, rewritten) in [("rewrite_offset", &single), ("rewrite_all", &every)] {
            let code = match rewritten {
                Err(_) => Some("canonical_mismatch"),
                Ok(text) => match expand(text, table) {
                    Err(_) => Some("canonical_mismatch"),
                    Ok(expanded) if expanded.canonical != *host => Some("canonical_mismatch"),
                    Ok(expanded) if expanded.recipe_sha256 != occurrence.recipe_sha256 => {
                        Some("recipe_hash_mismatch")
                    }
                    Ok(_) => None,
                },
            };
            if let Some(code) = code {
                result.failures.push(
                    Failure::new(code)
                        .recipe(&occurrence.recipe_sha256)
                        .context(label)
                        .task(occurrence.task),
                );
            }
        }
        let accepted = single
            .as_ref()
            .is_ok_and(|text| unexpanded_is_accepted(text, occurrence.task, &schema));
        if accepted {
            result.failures.push(
                Failure::new("unexpanded_accepted")
                    .recipe(&occurrence.recipe_sha256)
                    .task(occurrence.task),
            );
        }
        result.exercised += 1;
    }
    Ok(result.finish("round_trip_not_exercised"))
}

/// The slot contract or the schema-bound grammar accepts text that still holds a reference.
pub(super) fn unexpanded_is_accepted(text: &str, task: TaskLabel, schema: &DataSchema) -> bool {
    let Ok(ir) = GenomeIr::parse(text) else {
        return false;
    };
    let slots = matches!(
        &ir,
        GenomeIr::Pipeline(expr) if PipelineSlots::from_pipeline(expr).is_ok()
    );
    let grammar = task
        .task_spec()
        .is_ok_and(|spec| validate_v3_genome(&ir, &spec, schema).is_ok());
    slots || grammar
}

/// C4: no artifact may contain the name, the full id, its 12- or 16-hex prefix or the
/// reserved prefix. At least one artifact must have been emitted: a fragment for which no
/// emitter produced anything (for example one observed only with multiclass hosts, which the
/// scalar bound and fitted C++ emitters refuse and whose native text the legacy planner
/// refuses) fails with `name_absence_not_exercised`.
pub(super) fn name_absence(motif: &MotifDef, artifacts: &[Artifact]) -> CheckResult {
    let mut needles = vec![motif.name.clone(), motif.id.clone(), NAME_PREFIX.to_owned()];
    for length in [12, 16] {
        if let Some(prefix) = motif.id.get(..length) {
            needles.push(prefix.to_owned());
        }
    }
    let mut result = CheckResult::default();
    for artifact in artifacts {
        result.exercised += 1;
        if needles
            .iter()
            .any(|needle| artifact.text.contains(needle.as_str()))
        {
            result
                .failures
                .push(Failure::new("name_in_emission").context(artifact.kind));
        }
    }
    result.finish("name_absence_not_exercised")
}

/// The head the engine adds when a program names none (`advanced_v3/pipeline.rs:364-370`).
pub(super) fn default_head(task: TaskLabel) -> &'static str {
    match task {
        TaskLabel::Regression => "ridge",
        TaskLabel::Binary => "logit_ridge",
        TaskLabel::Multiclass => "multiclass_ridge",
    }
}
