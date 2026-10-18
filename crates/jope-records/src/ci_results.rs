//! `ci-results`: reconcile runner execution summaries with JUnit XML without inventing omitted
//! skip counts, ported from `deploy/ci/results.py` (`PORT-RUST-03`).

use std::collections::BTreeMap;
use std::path::Path;

use regex::Regex;

use crate::py;
use crate::pyjson::{Dict, Value};
use crate::{require, Error, Result};

pub const FIELDS: [&str; 6] = [
    "collected",
    "executed",
    "passed",
    "failed",
    "skipped",
    "excluded",
];

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Counts {
    pub collected: Option<i64>,
    pub executed: Option<i64>,
    pub passed: Option<i64>,
    pub failed: Option<i64>,
    pub skipped: Option<i64>,
    pub excluded: Option<i64>,
}

impl Counts {
    pub fn none() -> Self {
        Self {
            collected: None,
            executed: None,
            passed: None,
            failed: None,
            skipped: None,
            excluded: None,
        }
    }

    pub fn from_numbers(passed: i64, failed: i64, skipped: i64, excluded: i64) -> Self {
        Self {
            collected: Some(passed + failed + skipped + excluded),
            executed: Some(passed + failed),
            passed: Some(passed),
            failed: Some(failed),
            skipped: Some(skipped),
            excluded: Some(excluded),
        }
    }

    pub fn to_json(&self) -> Value {
        let opt = |v: Option<i64>| v.map_or(Value::Null, Value::Int);
        let mut dict = Dict::new();
        dict.insert("collected", opt(self.collected));
        dict.insert("executed", opt(self.executed));
        dict.insert("passed", opt(self.passed));
        dict.insert("failed", opt(self.failed));
        dict.insert("skipped", opt(self.skipped));
        dict.insert("excluded", opt(self.excluded));
        Value::Dict(dict)
    }
}

/// Extract `<testsuite>` and `<testcase>` counts from a JUnit XML file without inventing omitted
/// `skipped` attributes.
pub fn junit_counts(path: &Path) -> Result<Counts> {
    let text = py::read_text(path)?;
    parse_junit_str(&text)
}

pub fn parse_junit_str(xml: &str) -> Result<Counts> {
    let trimmed = xml.trim();
    if trimmed.is_empty() || !trimmed.starts_with('<') {
        return Err(Error::Invalid("invalid JUnit: empty or non-XML".into()));
    }
    let suite_re = Regex::new(r"<testsuite\b([^>]*?)>").unwrap();
    let attr_skipped_re = Regex::new(r#"\bskipped="(\d+)""#).unwrap();
    let mut suite_count = 0usize;
    let mut suite_skipped_sum = 0i64;
    let mut all_suites_have_skipped = true;
    for cap in suite_re.captures_iter(trimmed) {
        let attrs = &cap[1];
        suite_count += 1;
        if let Some(m) = attr_skipped_re.captures(attrs) {
            suite_skipped_sum += m[1]
                .parse::<i64>()
                .map_err(|_| Error::Invalid("invalid JUnit skipped attribute".into()))?;
        } else {
            all_suites_have_skipped = false;
        }
    }

    // Match both self-closing `<testcase .../>` and paired `<testcase ...>...</testcase>`.
    let case_re = Regex::new(r"(?s)<testcase\b([^>]*?)(?:/>|>(.*?)</testcase>)").unwrap();
    let child_skip_re = Regex::new(r"<skipped\b").unwrap();
    let child_fail_re = Regex::new(r"<(?:failure|error)\b").unwrap();
    let mut total_cases = 0i64;
    let mut skipped_cases = 0i64;
    let mut failed_cases = 0i64;
    for cap in case_re.captures_iter(trimmed) {
        total_cases += 1;
        let body = cap.get(2).map_or("", |m| m.as_str());
        if child_skip_re.is_match(body) {
            skipped_cases += 1;
        }
        if child_fail_re.is_match(body) {
            failed_cases += 1;
        }
    }

    let skip_count = if suite_count > 0 && all_suites_have_skipped {
        Some(suite_skipped_sum)
    } else {
        None
    };
    if let Some(sc) = skip_count {
        require!(
            sc == skipped_cases,
            "JUnit skipped attributes disagree with serialized cases"
        );
    }

    let executed = total_cases - skipped_cases;
    let passed = executed - failed_cases;
    Ok(Counts {
        collected: None,
        executed: Some(executed),
        passed: Some(passed),
        failed: Some(failed_cases),
        skipped: skip_count,
        excluded: None,
    })
}

/// Summarize execution log for `backend` (`none`, `nextest`, `pytest`, `rust`, `unittest`) and
/// optionally reconcile against `junit`.
pub fn summarize(log: &str, backend: &str, junit: Option<&Path>) -> Result<Counts> {
    let ansi_re = Regex::new(r"\x1b\[[0-9;]*m").unwrap();
    let clean = ansi_re.replace_all(log, "");

    if backend == "none" {
        return Ok(Counts::none());
    }

    let result = match backend {
        "nextest" => {
            let summary_re =
                Regex::new(r"Summary\s+\[[^\]]+\]\s+(\d+) tests run:\s*([^\n]+)").unwrap();
            let caps: Vec<_> = summary_re.captures_iter(&clean).collect();
            require!(
                caps.len() == 1,
                "expected exactly one nextest execution summary"
            );
            let total: i64 = caps[0][1]
                .parse()
                .map_err(|_| Error::Invalid("invalid nextest total".into()))?;
            let detail = &caps[0][2];
            let item_re = Regex::new(r"(\d+) (passed|failed|skipped)").unwrap();
            let mut values: BTreeMap<String, i64> = BTreeMap::new();
            for cap in item_re.captures_iter(detail) {
                values.insert(cap[2].to_owned(), cap[1].parse().unwrap_or(0));
            }
            require!(
                values.contains_key("passed")
                    && !["timed out", "leaked", "aborted"]
                        .iter()
                        .any(|w| detail.contains(w)),
                "incomplete or abnormal nextest summary"
            );
            let c = Counts::from_numbers(
                *values.get("passed").unwrap_or(&0),
                *values.get("failed").unwrap_or(&0),
                *values.get("skipped").unwrap_or(&0),
                0,
            );
            require!(
                c.executed == Some(total),
                "nextest execution total mismatch"
            );
            c
        }
        "pytest" => {
            let line_re =
                Regex::new(r"\b\d+ (?:passed|failed|skipped|error|errors)\b.*\bin [0-9.]+s")
                    .unwrap();
            let summaries: Vec<&str> = clean.lines().filter(|l| line_re.is_match(l)).collect();
            require!(
                summaries.len() == 1,
                "expected exactly one pytest execution summary"
            );
            let item_re =
                Regex::new(r"(\d+) (passed|failed|skipped|deselected|errors?|xfailed|xpassed)")
                    .unwrap();
            let mut values: BTreeMap<String, i64> = BTreeMap::new();
            for cap in item_re.captures_iter(summaries[0]) {
                values.insert(cap[2].to_owned(), cap[1].parse().unwrap_or(0));
            }
            require!(
                !["error", "errors", "xfailed", "xpassed"]
                    .iter()
                    .any(|k| *values.get(*k).unwrap_or(&0) != 0),
                "pytest errors/xfail/xpass require an explicit disposition"
            );
            Counts::from_numbers(
                *values.get("passed").unwrap_or(&0),
                *values.get("failed").unwrap_or(&0),
                *values.get("skipped").unwrap_or(&0),
                *values.get("deselected").unwrap_or(&0),
            )
        }
        "rust" => {
            let rust_re = Regex::new(
                r"test result: (?:ok|FAILED)\. (\d+) passed; (\d+) failed; (\d+) ignored; \d+ measured; (\d+) filtered out",
            )
            .unwrap();
            let mut passed = 0i64;
            let mut failed = 0i64;
            let mut ignored = 0i64;
            let mut filtered = 0i64;
            let mut matched = false;
            for cap in rust_re.captures_iter(&clean) {
                matched = true;
                passed += cap[1].parse::<i64>().unwrap_or(0);
                failed += cap[2].parse::<i64>().unwrap_or(0);
                ignored += cap[3].parse::<i64>().unwrap_or(0);
                filtered += cap[4].parse::<i64>().unwrap_or(0);
            }
            require!(matched, "missing Rust harness summaries");
            Counts::from_numbers(passed, failed, ignored, filtered)
        }
        "unittest" => {
            let unit_re = Regex::new(
                r"Ran (\d+) tests? in [^\n]+\n\s*\n(OK(?: \(skipped=(\d+)\))?|FAILED[^\n]*)",
            )
            .unwrap();
            let caps: Vec<_> = unit_re.captures_iter(&clean).collect();
            require!(
                !caps.is_empty() && caps.iter().all(|c| c[2].starts_with("OK")),
                "missing or failed unittest summaries"
            );
            let total: i64 = caps.iter().map(|c| c[1].parse::<i64>().unwrap_or(0)).sum();
            let skipped: i64 = caps
                .iter()
                .map(|c| {
                    c.get(3)
                        .map_or(0, |m| m.as_str().parse::<i64>().unwrap_or(0))
                })
                .sum();
            Counts::from_numbers(total - skipped, 0, skipped, 0)
        }
        other => {
            return Err(Error::Invalid(format!(
                "unsupported result backend: {other}"
            )))
        }
    };

    if let Some(junit_path) = junit {
        let report = junit_counts(junit_path)?;
        if report.executed != result.executed {
            return Err(Error::Invalid(
                "JUnit and execution summary disagree: executed".into(),
            ));
        }
        if report.passed != result.passed {
            return Err(Error::Invalid(
                "JUnit and execution summary disagree: passed".into(),
            ));
        }
        if report.failed != result.failed {
            return Err(Error::Invalid(
                "JUnit and execution summary disagree: failed".into(),
            ));
        }
        if report.skipped.is_some() && report.skipped != result.skipped {
            return Err(Error::Invalid(
                "JUnit and execution summary disagree: skipped".into(),
            ));
        }
    }

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn summarizes_nextest_and_reconciles_with_junit_missing_skipped_attr() {
        let log = "────────────\n     Summary [   1.234s] 3 tests run: 3 passed, 2 skipped\n";
        let xml = r#"<testsuites><testsuite name="s" tests="3" failures="0" errors="0">
            <testcase classname="a" name="t1"/>
            <testcase classname="a" name="t2"/>
            <testcase classname="a" name="t3"/>
        </testsuite></testsuites>"#;
        let dir = tempfile::tempdir().unwrap();
        let junit_path = dir.path().join("junit.xml");
        std::fs::write(&junit_path, xml).unwrap();
        let counts = summarize(log, "nextest", Some(&junit_path)).unwrap();
        assert_eq!(
            counts,
            Counts {
                collected: Some(5),
                executed: Some(3),
                passed: Some(3),
                failed: Some(0),
                skipped: Some(2),
                excluded: Some(0),
            }
        );
    }

    #[test]
    fn summarizes_rust_and_pytest_logs() {
        let rust_log = "test result: ok. 4 passed; 0 failed; 1 ignored; 0 measured; 2 filtered out; finished in 0.01s\n";
        let c = summarize(rust_log, "rust", None).unwrap();
        assert_eq!(c, Counts::from_numbers(4, 0, 1, 2));

        let pytest_log = "=== 10 passed, 2 skipped, 3 deselected in 0.42s ===\n";
        let cp = summarize(pytest_log, "pytest", None).unwrap();
        assert_eq!(cp, Counts::from_numbers(10, 0, 2, 3));
    }

    #[test]
    fn rejects_pytest_errors_or_nextest_mismatch() {
        let bad_pytest = "=== 4 passed, 1 error in 0.12s ===\n";
        assert!(summarize(bad_pytest, "pytest", None)
            .unwrap_err()
            .to_string()
            .contains("pytest errors/xfail/xpass"));

        let bad_nextest = "Summary [ 1.0s] 5 tests run: 3 passed, 1 failed\n";
        assert!(summarize(bad_nextest, "nextest", None)
            .unwrap_err()
            .to_string()
            .contains("nextest execution total mismatch"));
    }
}
