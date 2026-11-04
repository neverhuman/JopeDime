use super::*;

// -------------------------------- output arena -----------------------------

/// Single-buffer section accumulator. Sections are appended with a fixed
/// separator, each trimmed of trailing whitespace, into one pre-reserved
/// `String`. This replaces the old `Vec<String>` + `join`, which allocated a
/// temporary owned+trimmed copy per section and then copied the whole program
/// a second time at join time; here every section is copied exactly once.
pub struct Arena {
    buf: String,
    sep: &'static str,
    first: bool,
}

impl Arena {
    pub fn new(sep: &'static str, capacity: usize) -> Arena {
        Arena {
            buf: String::with_capacity(capacity),
            sep,
            first: true,
        }
    }

    /// Append one section (its trailing whitespace trimmed, matching the prior
    /// per-section `trim_end()`), preceded by the separator except for the first.
    pub fn push(&mut self, section: &str) {
        if !self.first {
            self.buf.push_str(self.sep);
        }
        self.first = false;
        self.buf.push_str(section.trim_end());
    }

    /// Finalize with the trailing newline the emitters always appended.
    pub fn finish(mut self) -> String {
        self.buf.push('\n');
        self.buf
    }
}

// ------------------------------ documentation ------------------------------

/// Wrap words greedily to `width` columns; deterministic.
pub fn wrap(text: &str, width: usize) -> Vec<String> {
    let mut lines = Vec::new();
    let mut cur = String::new();
    for word in text.split_whitespace() {
        if cur.is_empty() {
            cur = word.to_string();
        } else if cur.len() + 1 + word.len() <= width {
            cur.push(' ');
            cur.push_str(word);
        } else {
            lines.push(std::mem::take(&mut cur));
            cur = word.to_string();
        }
    }
    if !cur.is_empty() {
        lines.push(cur);
    }
    lines
}

/// Shared module documentation as paragraphs of plain text (no comment
/// markers); the emitters add the language-specific framing.
pub fn doc_paragraphs(plan: &Plan, opts: &crate::transpile::EmitOptions) -> (String, Vec<String>) {
    let title = match &opts.title {
        Some(t) => t.clone(),
        None => default_title(plan),
    };
    let mut paragraphs: Vec<String> = Vec::new();
    paragraphs.push(
        "This program trains a regression model on train.csv, in which every row holds the \
         feature columns followed by the target as the last column, and then writes one \
         prediction per line for the rows of test_features.csv. All arithmetic is IEEE-754 \
         double precision in a fixed evaluation order, so repeated runs give identical bits."
            .to_string(),
    );
    match plan {
        Plan::Pipeline {
            target,
            stages,
            head,
        } => {
            if let Some(target) = target {
                paragraphs.push(format!(
                    "Before any feature stage, the training target is rewritten by a fitted \
                     target transform ({name}), which {doc} The feature stages and the head are \
                     trained on the transformed target, and every prediction is mapped back \
                     through the transform's inverse and then clamped to [-60, 60].",
                    name = target_fn_name(target.kind),
                    doc = target_doc(target)
                ));
            }
            let mut p = String::new();
            if stages.is_empty() {
                p.push_str("The model applies the regression head directly to the input columns.");
            } else {
                // State the append/replace and sanitize convention exactly once,
                // then list each stage compactly. Repeated identical stages refer
                // back to their first appearance instead of restating the same
                // sentence, so long pipelines carry no duplicated prose.
                p.push_str(&format!(
                    "The feature pipeline has {n} stage{plural}, applied in order; each stage is \
                     marked \"replaced\" when it overwrites the columns it receives or \"appended\" \
                     when it keeps them and adds its outputs alongside, and after every stage \
                     non-finite cells become 0.0 and each row is capped at 512 columns. ",
                    n = stages.len(),
                    plural = if stages.len() == 1 { "" } else { "s" }
                ));
                let mut seen_docs: Vec<(String, usize)> = Vec::new();
                for (i, stage) in stages.iter().enumerate() {
                    let n = i + 1;
                    let name = op_fn_name(&stage.op);
                    let mode = if stage.append { "appended" } else { "replaced" };
                    let doc = op_doc(&stage.op);
                    match seen_docs.iter().find(|(d, _)| *d == doc) {
                        Some((_, first)) => p.push_str(&format!(
                            "Stage {n} ({name}, {mode}) repeats the transform of stage {first}. "
                        )),
                        None => {
                            p.push_str(&format!("Stage {n} ({name}, {mode}) {doc} "));
                            seen_docs.push((doc, n));
                        }
                    }
                }
            }
            paragraphs.push(p.trim_end().to_string());
            paragraphs.push(format!(
                "The head ({name}) is {doc} Seeded stages draw weights from a permuted \
                 congruential generator (PCG XSH-RR 64/32; O'Neill, 2014) keyed by the stage seed \
                 and the input/output widths, so training is fully deterministic. Missing or \
                 malformed CSV cells parse as 0.0 and short rows are padded with zeros to the \
                 training width.",
                name = head_fn_name(head),
                doc = head_doc(head)
            ));
        }
        Plan::Symbolic { source, affine, .. } => {
            paragraphs.push(format!(
                "The model is the fixed closed-form expression f(row) = {source}, evaluated with \
                 guarded arithmetic: division by a magnitude under 1e-12 gives 0, exponents are \
                 clamped to [-8, 8], sqrt uses max(v, 0), log1p is 0 at or under -1, the logistic \
                 argument is clamped to [-30, 30], and the final value is clamped to [-1e12, 1e12] \
                 with non-finite results replaced by 0."
            ));
            if *affine {
                paragraphs.push(
                    "Predictions are a*f(row) + b where the slope a and intercept b are the \
                     ordinary least-squares line of the training target on f, a linear \
                     calibration of the closed form; outputs are clipped to \
                     [-60, 60]. Missing or malformed CSV cells parse as 0.0 and short rows are \
                     padded with zeros to the training width."
                        .to_string(),
                );
            } else {
                paragraphs.push(
                    "Predictions are f(row) directly (slope 1, intercept 0), clipped to \
                     [-60, 60]. Missing or malformed CSV cells parse as 0.0 and short rows are \
                     padded with zeros to the training width."
                        .to_string(),
                );
            }
        }
    }
    if !opts.feature_names.is_empty() {
        paragraphs.push(format!("Input columns: {}.", opts.feature_names.join(", ")));
    }
    if let Some(ctx) = &opts.context {
        paragraphs.push(ctx.clone());
    }
    (title, paragraphs)
}

fn default_title(plan: &Plan) -> String {
    match plan {
        Plan::Pipeline {
            target,
            stages,
            head,
        } => {
            let mut parts: Vec<&str> = Vec::new();
            if let Some(target) = target {
                parts.push(target_short(target.kind));
            }
            parts.extend(stages.iter().map(|s| op_short(&s.op)));
            parts.push(head_short(head));
            format!("Regression model: {}", parts.join(", then "))
        }
        Plan::Symbolic { affine, .. } => {
            if *affine {
                "Regression model: closed-form expression with least-squares calibration"
                    .to_string()
            } else {
                "Regression model: closed-form expression".to_string()
            }
        }
    }
}
