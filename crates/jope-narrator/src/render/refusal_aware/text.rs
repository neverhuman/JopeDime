use super::*;

pub(crate) fn short(value: &str) -> String {
    safe(&value.chars().take(12).collect::<String>())
}
pub fn safe(value: &str) -> String {
    let mut out = value
        .chars()
        .take(256)
        .flat_map(|c| {
            if c.is_control() || matches!(c,'\u{202a}'..='\u{202e}'|'\u{2066}'..='\u{2069}') {
                c.escape_unicode().collect::<Vec<_>>()
            } else {
                vec![c]
            }
        })
        .collect::<String>();
    if value.chars().nth(256).is_some() {
        out.push('…');
    }
    out
}
pub(crate) fn stage_text(stage: &crate::StageView) -> String {
    let params = stage
        .params
        .iter()
        .map(|(k, v)| format!("{}={}", safe(k), safe(&v.display())))
        .collect::<Vec<_>>()
        .join(", ");
    if params.is_empty() {
        safe(&stage.op)
    } else {
        format!("{}({params})", safe(&stage.op))
    }
}

pub(crate) fn value(value: &Option<crate::ExactValue>) -> String {
    value
        .as_ref()
        .map(|v| safe(&v.display()))
        .unwrap_or_else(|| "omitted".into())
}
pub(crate) fn edit_text(edit: &AtomicEdit) -> String {
    match edit {
        AtomicEdit::Parameter {
            at,
            op,
            key,
            before,
            after,
        } => format!(
            "stage[{at}] {}.{} changes {} → {}",
            safe(op),
            safe(key),
            value(before),
            value(after)
        ),
        AtomicEdit::Insert { at, stage } => format!("insert {} at stage[{at}]", stage_text(stage)),
        AtomicEdit::Delete { at, before } => {
            format!("remove {} at stage[{at}]", stage_text(before))
        }
        AtomicEdit::Replace { at, before, after } => format!(
            "stage[{at}] replaces {} with {}",
            stage_text(before),
            stage_text(after)
        ),
        AtomicEdit::Swap { at, left, right } => format!(
            "swap stage[{at}] {} and stage[{}] {}",
            stage_text(left),
            at + 1,
            stage_text(right)
        ),
        AtomicEdit::ReplaceProgram { .. } => {
            "replace the whole captured program; finer semantic coverage unavailable".into()
        }
    }
}
pub(crate) fn hypothesis(delta: &ProgramDelta) -> String {
    if let [AtomicEdit::Parameter { op, key, .. }] = delta.edits.as_slice() {
        if matches!(op.as_str(), "ridge" | "logistic") && key == "lambda_frac" {
            return "Observer question: would this different ridge regularization setting help validation? The measured result is pending.".into();
        }
        if key == "seed" {
            return "Observer question: how will this recorded seeded construction measure? A different seed is not evidence of improved quality. Validation is pending.".into();
        }
    }
    "Observer question: how will the changed recipe measure as a whole? Validation is pending, and any eventual gain cannot yet be attributed to an individual edit.".into()
}

pub(crate) fn evidence_refs(links: &Links, evidence: &str) -> Vec<String> {
    std::iter::once(evidence.to_string())
        .chain(
            [
                links.search_event_id.clone(),
                links.observation_sha256.clone(),
                links.draw_id.clone(),
                links.request_id.clone(),
                links.attempt_id.clone(),
            ]
            .into_iter()
            .flatten(),
        )
        .collect()
}

pub(crate) fn append_boundary_note(
    out: &mut Narration,
    fact: &LineageFact,
    links: &Links,
    claim: &impl Fn(&str, &str, Value) -> Claim,
) {
    if matches!(fact, LineageFact::Measured { .. })
        && links.observation_sha256.is_some()
        && links.draw_id.is_none()
        && links.request_id.is_none()
        && links.attempt_id.is_none()
    {
        out.text.push_str(" Native draw/request/attempt references are unavailable; the observation digest was computed from the retained boundary data.");
        out.claims.push(claim(
            "limitation",
            "derived_boundary_observation",
            json!({"observation_sha256":links.observation_sha256,"native_refs":"unavailable"}),
        ));
    }
}
