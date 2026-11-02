use super::*;

#[test]
fn consistent_documents_pass() {
    assert_eq!(
        problems(WRITEUP_OK, LEDGER_OK, TRACE_OK),
        Vec::<String>::new()
    );
}

#[test]
fn a_drifted_printed_number_fails() {
    let w = WRITEUP_OK.replace("893,957 parameters", "893,958 parameters");
    let p = problems(&w, LEDGER_OK, TRACE_OK);
    assert!(
        p.iter().any(|m| m.contains("893958 is not in any claim")),
        "{p:?}"
    );
}

#[test]
fn a_drifted_ledger_value_fails() {
    // Leave the TeX at its old value: formatting must not mask the drift.
    let l = LEDGER_OK
        .replace("893,957 parameters", "894,000 parameters")
        .replace("\"893,957\"", "\"894,000\"");
    let p = problems(WRITEUP_OK, &l, TRACE_OK);
    assert!(
        p.iter().any(|m| m.contains("893957 is not in any claim")),
        "{p:?}"
    );
}

#[test]
fn a_tex_that_disagrees_with_its_record_fails() {
    // C01's statement and value move to 893,958 while its tex keeps 893{,}957: the stale tex
    // is reported, and it no longer accounts for the write-up's 893,957.
    let l = LEDGER_OK
        .replace("893,957 parameters", "893,958 parameters")
        .replace("\"893,957\"", "\"893,958\"");
    let p = problems(WRITEUP_OK, &l, TRACE_OK);
    assert!(
        p.iter().any(|m| m.contains("C01 tex prints 893957")),
        "{p:?}"
    );
    assert!(
        p.iter().any(|m| m.contains("893957 is not in any claim")),
        "{p:?}"
    );
}

#[test]
fn latex_times_and_plus_minus_are_normalized() {
    let n = |s: &str| {
        numbers(&normalize(s))
            .into_iter()
            .map(|p| p.text)
            .collect::<Vec<_>>()
    };
    assert_eq!(n("$3.3{\\times}10^{-8}$"), ["3.3e-8"]);
    assert_eq!(n("$1.24\\times10^5$/h"), ["1.24e5"]);
    assert_eq!(n("clip +/-10"), ["-10", "10"]);
    assert_eq!(n("$\\pm 10$"), ["-10", "10"]);
    assert_eq!(n("±10"), ["-10", "10"]);
}

#[test]
fn rounding_is_accepted_only_at_the_printed_precision() {
    // 0.0912345 prints as 0.09; 0.10 would not.
    assert_eq!(
        problems(WRITEUP_OK, LEDGER_OK, TRACE_OK),
        Vec::<String>::new()
    );
    let w = WRITEUP_OK.replace("| anchored | 0.09 |", "| anchored | 0.10 |");
    assert!(!problems(&w, LEDGER_OK, TRACE_OK).is_empty());
}

#[test]
fn a_range_dash_is_not_a_minus_sign() {
    let printed: Vec<String> = printed_numbers("0.004–0.011 and −0.0625 to 3.75\\times10^{-5}")
        .into_iter()
        .map(|p| p.text)
        .collect();
    assert_eq!(printed, ["0.004", "0.011", "-0.0625", "3.75e-5"]);
}

#[test]
fn identifiers_and_references_are_not_numbers() {
    let line = "R23 and C26 (Section 10.3.1; Loshchilov and Hutter, 2017) at 2026-10-13T00:06Z, `v2/r3.1` p50 float32 SHA-256 [C140–C143]. The 2026-10-12 direction";
    assert_eq!(printed_numbers(line), vec![]);
}

/// "At 2,000" (read as "At 2000" once grouping commas go) is a number to check, not an
/// author-year reference; the citation shapes around it stay excluded.
#[test]
fn a_capitalised_word_before_a_number_is_not_a_reference() {
    let line = "At 2,000 evaluations per hour (Loshchilov and Hutter, 2017; Chen et al., 2021), as Parisi et al. (2019) note.";
    let texts: Vec<String> = printed_numbers(line).into_iter().map(|p| p.text).collect();
    assert_eq!(texts, ["2000"]);
    let w = WRITEUP_OK.replace(
        "PolicyLM has 893,957 parameters [C01].",
        "PolicyLM has 893,957 parameters [C01]. At 2,048 evaluations per hour [C01].",
    );
    let p = problems(&w, LEDGER_OK, TRACE_OK);
    assert!(
        p.iter().any(|m| m.contains("2048 is not in any claim")),
        "{p:?}"
    );
}

#[test]
fn totals_and_pending_table_must_match_the_ledger() {
    let w = WRITEUP_OK.replace("1 non-qualifying", "2 non-qualifying");
    let p = problems(&w, LEDGER_OK, TRACE_OK);
    assert!(p.iter().any(|m| m.contains("Appendix G")), "{p:?}");
    let w = WRITEUP_OK.replace("| C03 | E1 primary | pending |\n", "");
    let p = problems(&w, LEDGER_OK, TRACE_OK);
    assert!(p.iter().any(|m| m.contains("Section 10.4")), "{p:?}");
}

#[test]
fn citations_must_resolve_and_cover_the_ledger() {
    let w = WRITEUP_OK.replace("[C01]", "[C09]");
    let p = problems(&w, LEDGER_OK, TRACE_OK);
    assert!(p.iter().any(|m| m.contains("cites C09")), "{p:?}");
    assert!(
        p.iter()
            .any(|m| m.contains("C01 is in the ledger but never cited")),
        "{p:?}"
    );
}

#[test]
fn the_trace_must_follow_the_ledger() {
    let t = TRACE_OK.replace("| C02 | non-qualifying |", "| C02 | established |");
    let p = problems(WRITEUP_OK, LEDGER_OK, &t);
    assert!(
        p.iter()
            .any(|m| m.contains("C02 is \"established\" in the trace")),
        "{p:?}"
    );
    let t = TRACE_OK.replace("| C03 | pending | — | LANE:c.json |\n", "");
    let p = problems(WRITEUP_OK, LEDGER_OK, &t);
    assert!(
        p.iter().any(|m| m.contains("no Claim trace row for C03")),
        "{p:?}"
    );
    let t = format!("{TRACE_OK}\n## Fresh-clone run\n\nFRESH_RUN_PLACEHOLDER\n");
    let p = problems(WRITEUP_OK, LEDGER_OK, &t);
    assert!(
        p.iter()
            .any(|m| m.contains("unfilled placeholder: FRESH_RUN_PLACEHOLDER")),
        "{p:?}"
    );
}

#[test]
fn a_stale_exemption_fails() {
    let w = WRITEUP_OK.replace("Illustrative: q = log 10 ≈ 2.303.\n", "");
    let p = problems(&w, LEDGER_OK, TRACE_OK);
    assert!(p.iter().any(|m| m.contains("no longer occurs")), "{p:?}");
}

#[test]
fn ledger_rules() {
    let l = LEDGER_OK
        .replace("\"source\": \"LANE:c.json\"", "\"source\": \"NOPE:c.json\"")
        .replace(
            "\"sha256\": \"2222222222222222222222222222222222222222222222222222222222222222\"",
            "\"sha256\": \"\"",
        );
    let p = problems(WRITEUP_OK, &l, TRACE_OK);
    assert!(p.iter().any(|m| m.contains("root NOPE")), "{p:?}");
    assert!(
        p.iter().any(|m| m.contains("C03 pins no 64-hex sha256")),
        "{p:?}"
    );
}
