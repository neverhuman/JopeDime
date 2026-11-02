//! Engine names, task identifiers, evaluator routing, and head-cost metadata.

pub const ENGINE_FEATMATH: &str = "featmath";
pub const ENGINE_NEWMATH: &str = "newmath";
pub const ENGINES: [&str; 2] = [ENGINE_FEATMATH, ENGINE_NEWMATH];
pub const DEFAULT_ENGINE: &str = ENGINE_FEATMATH;

pub const TASK_REGRESSION: &str = "regression";
pub const TASK_BINARY: &str = "binary";
pub const TASKS: [&str; 2] = [TASK_REGRESSION, TASK_BINARY];
pub const DEFAULT_TASK: &str = TASK_REGRESSION;

/// Task names and bridge subcommands, matching Python's `EVAL_SUBCOMMAND`.
pub const EVAL_SUBCOMMAND: [(&str, &str); 2] = [
    (TASK_REGRESSION, "eval-regression"),
    (TASK_BINARY, "eval-binary"),
];

/// Evaluator versions keyed by `(engine, task)`; the `prime` row is intentional.
pub const EVALUATOR_VERSIONS_BY_TASK: [((&str, &str), &str); 5] = [
    (
        (ENGINE_FEATMATH, TASK_REGRESSION),
        "jope-validate.eval-regression.v2",
    ),
    (
        (ENGINE_NEWMATH, TASK_REGRESSION),
        "jope-newmath-validate.eval-regression.v1",
    ),
    (
        (ENGINE_FEATMATH, TASK_BINARY),
        "jope-validate.eval-binary.v1",
    ),
    (
        (ENGINE_NEWMATH, TASK_BINARY),
        "jope-newmath-validate.eval-binary.v1",
    ),
    (
        ("prime", TASK_REGRESSION),
        "jope-prime-evald.compiled-plan.v1",
    ),
];

/// Legacy engine-only aliases use each engine's regression evaluator version.
pub const EVALUATOR_VERSIONS: [(&str, &str); 2] = [
    (ENGINE_FEATMATH, "jope-validate.eval-regression.v2"),
    (ENGINE_NEWMATH, "jope-newmath-validate.eval-regression.v1"),
];

pub const FALLBACK_GENOMES: [(&str, &str); 2] = [
    (ENGINE_FEATMATH, "(pipe X (robust_scale) (ridge))"),
    (ENGINE_NEWMATH, "(pipe X (robust_scale) (ridge alpha=1.0))"),
];

pub const TIMEOUT_BY_CLASS: [(&str, f64); 5] = [
    ("cheap", 10.0),
    ("medium", 30.0),
    ("expensive", 90.0),
    ("extreme", 120.0),
    ("portfolio", 600.0),
];

const FEATMATH_HEAD_COSTS: [(&str, &str); 14] = [
    ("ridge", "cheap"),
    ("logistic", "cheap"),
    ("elasticnet", "cheap"),
    ("ridge_gcv", "medium"),
    ("histgbm", "medium"),
    ("gbm", "medium"),
    ("ksmooth", "medium"),
    ("ensemble", "medium"),
    ("ensemble_ridge_gbm", "medium"),
    ("histgbm2", "expensive"),
    ("krr", "expensive"),
    ("stack", "expensive"),
    ("gpr", "extreme"),
    ("portfolio", "portfolio"),
];
const NEWMATH_HEAD_COSTS: [(&str, &str); 8] = [
    ("ridge", "cheap"),
    ("logistic", "cheap"),
    ("elasticnet", "cheap"),
    ("huber_ridge", "cheap"),
    ("knn", "medium"),
    ("gbm", "medium"),
    ("random_forest", "medium"),
    ("ensemble_ridge_gbm", "medium"),
];

/// Engine-specific head operation classes, preserving the source table contents.
pub const HEAD_COST_CLASS: [(&str, &[(&str, &str)]); 2] = [
    (ENGINE_FEATMATH, &FEATMATH_HEAD_COSTS),
    (ENGINE_NEWMATH, &NEWMATH_HEAD_COSTS),
];

pub fn eval_subcommand(task: &str) -> Option<&'static str> {
    EVAL_SUBCOMMAND
        .iter()
        .find_map(|(key, value)| (*key == task).then_some(*value))
}

pub fn evaluator_version(engine: &str, task: &str) -> Option<&'static str> {
    EVALUATOR_VERSIONS_BY_TASK
        .iter()
        .find_map(|((known_engine, known_task), version)| {
            (*known_engine == engine && *known_task == task).then_some(*version)
        })
}

pub fn fallback_genome(engine: &str) -> Option<&'static str> {
    FALLBACK_GENOMES
        .iter()
        .find_map(|(known, genome)| (*known == engine).then_some(*genome))
}

pub fn timeout_for_class(class: &str) -> Option<f64> {
    TIMEOUT_BY_CLASS
        .iter()
        .find_map(|(known, seconds)| (*known == class).then_some(*seconds))
}

pub fn head_cost_class(engine: &str, head: &str) -> Option<&'static str> {
    HEAD_COST_CLASS
        .iter()
        .find(|(known_engine, _)| *known_engine == engine)
        .and_then(|(_, entries)| {
            entries
                .iter()
                .find_map(|(known_head, cost)| (*known_head == head).then_some(*cost))
        })
}

/// Best-effort final-stage operation extraction, matching `_HEAD_RE` and `head_op_of`.
pub fn head_op_of(genome: &str) -> Option<&str> {
    let chars: Vec<(usize, char)> = genome.char_indices().collect();
    for (position, (start, ch)) in chars.iter().enumerate() {
        if *ch != '(' {
            continue;
        }
        let mut cursor = position + 1;
        let capture_start = cursor;
        while cursor < chars.len() && is_head_char(chars[cursor].1) {
            cursor += 1;
        }
        if cursor == capture_start {
            continue;
        }
        while cursor < chars.len() && chars[cursor].1 != '(' && chars[cursor].1 != ')' {
            cursor += 1;
        }
        if cursor == chars.len() || chars[cursor].1 != ')' {
            continue;
        }
        cursor += 1;
        while cursor < chars.len() && chars[cursor].1.is_whitespace() {
            cursor += 1;
        }
        if cursor < chars.len() && chars[cursor].1 == ')' {
            cursor += 1;
        } else {
            continue;
        }
        // Python's `$` also accepts the position immediately before one final newline.
        if cursor == chars.len() || (cursor + 1 == chars.len() && chars[cursor].1 == '\n') {
            let capture_begin = chars[capture_start].0;
            let capture_end = chars[cursor - 2].0 + chars[cursor - 2].1.len_utf8();
            return genome.get(capture_begin..capture_end);
        }
        let _ = start;
    }
    None
}

fn is_head_char(ch: char) -> bool {
    ch.is_ascii_lowercase() || ch.is_ascii_digit() || ch == '_'
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn engine_task_and_evaluator_tables_match_python_configuration() {
        assert_eq!(ENGINES, ["featmath", "newmath"]);
        assert_eq!(TASKS, ["regression", "binary"]);
        assert_eq!(DEFAULT_ENGINE, ENGINE_FEATMATH);
        assert_eq!(DEFAULT_TASK, TASK_REGRESSION);
        assert_eq!(eval_subcommand("binary"), Some("eval-binary"));
        assert_eq!(eval_subcommand("unknown"), None);
        assert_eq!(
            evaluator_version("prime", "regression"),
            Some("jope-prime-evald.compiled-plan.v1")
        );
        assert_eq!(evaluator_version("prime", "binary"), None);
        assert_eq!(
            fallback_genome(ENGINE_NEWMATH),
            Some("(pipe X (robust_scale) (ridge alpha=1.0))")
        );
        assert_eq!(fallback_genome("prime"), None);
    }

    #[test]
    fn head_operation_and_cost_class_match_configuration() {
        assert_eq!(head_op_of("(pipe X (ridge))"), Some("ridge"));
        assert_eq!(
            head_op_of("(pipe X (robust_scale) (ridge alpha=1.0))"),
            Some("ridge")
        );
        assert_eq!(head_op_of("(pipe X (histgbm) )"), Some("histgbm"));
        assert_eq!(head_op_of("(pipe X (ridge)"), None);
        assert_eq!(head_op_of("(PIPE X (Ridge))"), None);
        assert_eq!(head_cost_class(ENGINE_FEATMATH, "gpr"), Some("extreme"));
        assert_eq!(head_cost_class(ENGINE_NEWMATH, "gpr"), None);
        assert_eq!(timeout_for_class("portfolio"), Some(600.0));
        assert_eq!(timeout_for_class("unknown"), None);
    }
}
