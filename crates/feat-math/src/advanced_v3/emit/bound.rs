use super::*;

// ---------------------------- bound artifacts -----------------------------

#[derive(Clone, Debug)]
pub struct BoundEmitted {
    pub python: String,
    pub rust: String,
    pub manifest: String,
    pub py_tokens: crate::transpile::TokenCount,
    pub rs_tokens: crate::transpile::TokenCount,
    pub timing: crate::transpile::EmitTiming,
}

/// The bound artifact's sidecar. The JSON key is `program_sha256`; the
/// `BoundModel` field it comes from keeps its Rust name because the fitted
/// model codec persists it.
#[derive(Serialize)]
struct BoundManifest<'a> {
    schema: &'a str,
    task: &'a TaskSpec,
    task_sha256: &'a str,
    schema_sha256: &'a str,
    vocabulary_sha256: &'a str,
    program_sha256: &'a str,
    emitter_version: &'a str,
    python_sha256: String,
    rust_sha256: String,
}

/// The strings a bound or fitted artifact carries that come from the caller's
/// data rather than from the emitter: the column names and every categorical
/// vocabulary entry. The shipped-text lint runs with them masked
/// ([`lint_shipped_masked`]), so a column called `island` or a category named
/// `population` cannot fail emission, while everything the emitter itself
/// writes is still checked.
///
/// Only strings that would trip the lint on their own are masked (an innocent
/// name never touches emitter text), and only in the spellings the artifacts
/// embed them in: raw, JSON-escaped (the C++ codec), Rust `Debug`-escaped (the
/// Rust artifact's string literals) and the JSON quotation of the JSON form (the Python
/// artifact embeds its JSON literals through valid JSON string quotation). Entries shorter than three
/// bytes cannot contain a banned token and are skipped. Longest first, so a name
/// that contains another is masked whole. One edge fails closed: a name that
/// does not trip raw but whose escaped spelling completes a token (a leading
/// newline followed by `nls`, say) is not masked, and the artifact is refused
/// rather than shipped.
pub(crate) fn user_text(model: &BoundModel) -> Vec<String> {
    let mut raw: Vec<String> = model
        .schema
        .columns
        .iter()
        .map(|column| column.name.clone())
        .collect();
    for column in &model.input.columns {
        if let FittedColumn::Categorical { vocabulary, .. } = column {
            raw.extend(vocabulary.iter().cloned());
        }
    }
    let trips = |text: &str| {
        !crate::transpile::lint_vocabulary_portfolio(text).is_empty()
            || !crate::transpile::lint_shipped_phrases(text).is_empty()
    };
    let mut forms = Vec::new();
    for text in raw {
        if text.len() < 3 || !trips(&text) {
            continue;
        }
        let json = serde_json::to_string(&text).unwrap_or_default();
        let json_inner = json.get(1..json.len().saturating_sub(1)).unwrap_or("");
        let debug = format!("{text:?}");
        let debug_inner = &debug[1..debug.len() - 1];
        let debug_of_json = serde_json::to_string(json_inner).unwrap_or_default();
        let debug_of_json_inner = &debug_of_json[1..debug_of_json.len() - 1];
        for form in [json_inner, debug_inner, debug_of_json_inner] {
            if !form.is_empty() && form != text {
                forms.push(form.to_owned());
            }
        }
        forms.push(text);
    }
    forms.sort_by(|a, b| b.len().cmp(&a.len()).then_with(|| a.cmp(b)));
    forms.dedup();
    forms
}

/// The shipped-text lint over `text` with every user-supplied string
/// ([`user_text`]) replaced by underscores of the same length. Portfolio heads
/// get the frozen emitter's stricter portfolio rule (`oof` is banned too).
pub(crate) fn lint_shipped_masked(text: &str, user: &[String], portfolio: bool) -> Vec<String> {
    let mut masked = text.to_owned();
    for entry in user {
        if masked.contains(entry.as_str()) {
            masked = masked.replace(entry.as_str(), &"_".repeat(entry.len()));
        }
    }
    let mut hits = if portfolio {
        crate::transpile::lint_vocabulary_portfolio(&masked)
    } else {
        crate::transpile::lint_vocabulary(&masked)
    };
    hits.extend(crate::transpile::lint_shipped_phrases(&masked));
    hits
}

/// Emit a fitted, raw-CSV prediction program. This is deliberately separate
/// from `transpile::emit`: the latter remains the frozen artifact-v1 training
/// program (pipeline artifacts are byte-identical across emitter versions
/// 2.1 and 2.2; see `EMITTER_VERSION`). Every artifact is linted with the
/// caller's column names and categorical vocabularies masked before
/// `lint_shipped_masked` runs.
pub fn emit_bound(
    model: &BoundModel,
    options: &crate::transpile::EmitOptions,
) -> Result<BoundEmitted> {
    crate::transpile::validate_math_kernel(options)?;
    model.validate_binding()?;
    let BoundPredictor::Pipeline { ops, head, .. } = &model.predictor else {
        bail!("bound emission currently requires a v3 pipeline model");
    };
    for op in ops {
        op.validate_fitted_state(&model.schema)?;
    }
    let portfolio = matches!(head, V3Head::ClassificationPortfolio(_));
    let user = user_text(model);
    let start = std::time::Instant::now();
    let python = emit_bound_python(model, options.partition_point_quantile_bounds)?;
    let rust = emit_bound_rust(model)?;
    let elapsed = start.elapsed().as_nanos().max(1);
    for (lang, text) in [("Python", &python), ("Rust", &rust)] {
        let hits = lint_shipped_masked(text, &user, portfolio);
        if !hits.is_empty() {
            bail!(
                "banned vocabulary in bound {lang} artifact: {}",
                hits.join(", ")
            );
        }
    }
    let py_tokens = crate::tokens::count_tokens(&python, crate::tokens::Lang::Python);
    let rs_tokens = crate::tokens::count_tokens(&rust, crate::tokens::Lang::Rust);
    let py_total = py_tokens.code_tokens + py_tokens.doc_tokens;
    let rs_total = rs_tokens.code_tokens + rs_tokens.doc_tokens;
    let emitter_version = if options.partition_point_quantile_bounds {
        "feat-math-quantile-partition-point-v1"
    } else {
        crate::transpile::EMITTER_VERSION
    };
    let manifest = serde_json::to_string(&BoundManifest {
        schema: ARTIFACT_SCHEMA,
        task: &model.task,
        task_sha256: &model.task_sha256,
        schema_sha256: &model.schema_sha256,
        vocabulary_sha256: &model.vocabulary_sha256,
        program_sha256: &model.genome_sha256,
        emitter_version,
        python_sha256: text_hash(&python),
        rust_sha256: text_hash(&rust),
    })?;
    let hits = lint_shipped_masked(&manifest, &user, portfolio);
    if !hits.is_empty() {
        bail!(
            "banned vocabulary in bound artifact manifest: {}",
            hits.join(", ")
        );
    }
    Ok(BoundEmitted {
        python,
        rust,
        manifest,
        py_tokens,
        rs_tokens,
        timing: crate::transpile::EmitTiming {
            emit_ns: elapsed,
            py_tokens: py_total,
            rs_tokens: rs_total,
            tokens_per_sec: (py_total + rs_total) as f64 / (elapsed as f64 / 1e9),
        },
    })
}

/// Fast-path Rust-only bound emitter for `jope-micro`: emits and lints the
/// standalone Rust prediction program (`bound_model.rs`) without generating or
/// linting the unused Python twin or JSON manifest.
///
/// Contract note: because Python emission, Python vocabulary linting, and
/// manifest construction are intentionally skipped (`python` and `manifest` are
/// returned empty), this function validates only the bound Rust artifact and
/// can succeed on models where [`emit_bound`] would fail closed in the Python
/// emitter or Python linter.
pub fn emit_bound_rust_only(
    model: &BoundModel,
    options: &crate::transpile::EmitOptions,
) -> Result<BoundEmitted> {
    crate::transpile::validate_math_kernel(options)?;
    model.validate_binding()?;
    let BoundPredictor::Pipeline { ops, head, .. } = &model.predictor else {
        bail!("bound emission currently requires a v3 pipeline model");
    };
    for op in ops {
        op.validate_fitted_state(&model.schema)?;
    }
    let portfolio = matches!(head, V3Head::ClassificationPortfolio(_));
    let user = user_text(model);
    let start = std::time::Instant::now();
    let rust = emit_bound_rust(model)?;
    let elapsed = start.elapsed().as_nanos().max(1);
    let hits = lint_shipped_masked(&rust, &user, portfolio);
    if !hits.is_empty() {
        bail!(
            "banned vocabulary in bound Rust artifact: {}",
            hits.join(", ")
        );
    }
    let py_tokens = crate::transpile::TokenCount {
        code_tokens: 0,
        doc_tokens: 0,
        bytes: 0,
    };
    let rs_tokens = crate::tokens::count_tokens(&rust, crate::tokens::Lang::Rust);
    let rs_total = rs_tokens.code_tokens + rs_tokens.doc_tokens;
    Ok(BoundEmitted {
        python: String::new(),
        rust,
        manifest: String::new(),
        py_tokens,
        rs_tokens,
        timing: crate::transpile::EmitTiming {
            emit_ns: elapsed,
            py_tokens: 0,
            rs_tokens: rs_total,
            tokens_per_sec: rs_total as f64 / (elapsed as f64 / 1e9),
        },
    })
}
