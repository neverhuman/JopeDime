use super::super::paper::{accounted, field, ledger, normalize, numbers, printed_numbers};
use super::super::readme::{grouped, words};
use super::*;

#[test]
fn readme_glance_drift_fails() {
    let g = Glance {
        version: "0.3.0".into(),
        members: 26,
        default_members: 8,
        versions: [("0.3.0".to_owned(), 19), ("8.0.0".to_owned(), 7)].into(),
        vendored: 7,
        vendored_version: "8.0.0".into(),
        emit_targets: 6,
        rust_tests: 4189,
        rust_ignored: 66,
        rust_packages: 26,
        python_tests: 182,
        ci_jobs: 9,
        full_families: 11,
        coverage_items: 678,
        coverage_lanes: 24,
        matrix: vec![824, 353, 61, 1, 402, 7],
        synthetic: 40,
        test_datasets: 10,
        python_files: 1184,
        python_under_python: 159,
        adrs: 6,
    };
    let readme = "\
| What | 0.3.0 |
| `jope` with no arguments | prints `jope 0.3.0` <sup>1</sup> |
| Workspace | 26 members: 19 crates at 0.3.0 and 7 vendored SmartCluster crates at 8.0.0; 8 are `default-members` |
| Emit targets | 6: `python` |
| Rust tests | 4,189 listed in 26 packages, 66 of them ignored |
| Python tests | 182, found by a static scan |
| CI | 9 jobs per pull request in `ci.yml`; 11 `full-main-cpu` families |
| Coverage map | 678 items over 24 lanes |
| Capability matrix | 824 rows: 353 moved, 61 new, 1 pending, 402 dropped, 7 lost |
| Datasets in Git | 40 synthetic fixtures (10 each) and 10 test datasets |
| Python still tracked | 1,184 `.py` files, 159 of them under `python/` |
| Architecture decision records | 6 <sup>10</sup> |
# jope 0.3.0
source of v0.3.0; the release
Pull-request CI is `.github/workflows/ci.yml`: nine jobs on runners.
forty generated synthetic fixtures and ten tiny test datasets.
The 26 workspace members, by directory.
";
    assert_eq!(check_readme(readme, &g), Vec::<String>::new());
    let stale = readme.replace("4,189 listed", "4,100 listed");
    assert!(!check_readme(&stale, &g).is_empty());
    let stale = readme.replace("nine jobs", "eight jobs");
    assert!(!check_readme(&stale, &g).is_empty());
}

#[test]
fn number_words_and_grouping() {
    assert_eq!(words(9), "nine");
    assert_eq!(words(40), "forty");
    assert_eq!(words(26), "twenty-six");
    assert_eq!(grouped(4189), "4,189");
    assert_eq!(grouped(678), "678");
    assert_eq!(grouped(1234567), "1,234,567");
}

/// The committed paper, ledger, trace, Figure 1 and README agree with each other and with the
/// files the README's footnotes name. A number edited on one side only fails here.
#[test]
fn the_committed_documents_agree() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    if !root.join(".git").exists() {
        eprintln!(
            "skipping committed-document check: requires a Git checkout; .git metadata is absent"
        );
        return;
    }
    let (report, ok) = check(&root, false, false).expect("readable");
    assert!(ok, "{report}");

    // C140's old TeX bound widened the accepted set despite the corrected record.
    // Keep the recorded bound accepted, and reject the old double-rounded value.
    let ledger_text = std::fs::read_to_string(root.join(LEDGER)).expect("ledger readable");
    let mut issues = Vec::new();
    let claims = ledger(&ledger_text, &mut issues).expect("ledger parses");
    assert!(issues.is_empty(), "{issues:?}");
    let c140 = claims
        .iter()
        .find(|claim| claim.id == "C140")
        .expect("C140");
    assert!(accounted(&printed_numbers("0.8963")[0], &c140.numbers));
    assert!(!accounted(&printed_numbers("0.8964")[0], &c140.numbers));
    let raw = pyjson::parse_str(&ledger_text).expect("raw ledger parses");
    let raw_claims = raw.get("claims").and_then(Value::as_list).expect("claims");
    for id in ["C04", "C93", "C140", "C168"] {
        let presentation = raw_claims
            .iter()
            .find(|entry| field(entry, "id") == id)
            .expect("claim");
        let canonical = claims.iter().find(|entry| entry.id == id).expect("record");
        for printed in numbers(&normalize(field(presentation, "tex"))) {
            assert!(
                accounted(&printed, &canonical.numbers),
                "{id} TeX introduces {}",
                printed.text
            );
        }
    }

    let trace_text = std::fs::read_to_string(root.join(TRACE)).expect("trace readable");
    let writeup_text = std::fs::read_to_string(root.join(WRITEUP)).expect("writeup readable");
    let readme_text = std::fs::read_to_string(root.join(README)).expect("readme readable");
    let bad_svg_sha = trace_text.replace(
        "04590ecc014a7856f1a97751cb6d3e7115b4efc6886bcc45c7f886503f42669d",
        "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff",
    );
    let svg_probs = check_svg_figures(
        &root,
        &bad_svg_sha,
        &writeup_text,
        &readme_text,
        &ledger_text,
    )
    .expect("svg check runs");
    assert!(
        svg_probs
            .iter()
            .any(|p| p.contains("clover-why-it-matters-3panel.svg pins ffffffff")),
        "{svg_probs:?}"
    );
    let unlinked_readme = readme_text.replace(
        "](docs/paper/figures/readme-clover-overview.svg)",
        "](nope)",
    );
    let link_probs = check_svg_figures(
        &root,
        &trace_text,
        &writeup_text,
        &unlinked_readme,
        &ledger_text,
    )
    .expect("svg link check runs");
    assert!(
        link_probs.iter().any(|p| p
            .contains("README.md: does not link docs/paper/figures/readme-clover-overview.svg")),
        "{link_probs:?}"
    );

    let exp_jsonl =
        std::fs::read_to_string(root.join(EXPERIMENTS_JSONL)).expect("experiments readable");
    let exp_readme = std::fs::read_to_string(root.join(EXPERIMENTS_README))
        .expect("experiments readme readable");
    let exp_synthesis = std::fs::read_to_string(root.join(EXPERIMENTS_SYNTHESIS))
        .expect("experiments synthesis readable");
    let bad_exp_claim = exp_jsonl.replacen("\"C01\"", "\"C999\"", 1);
    let exp_probs = check_experiments(
        &root,
        &bad_exp_claim,
        &exp_readme,
        &exp_synthesis,
        &ledger_text,
    )
    .expect("exp check runs");
    assert!(
        exp_probs
            .iter()
            .any(|p| p.contains("cites unknown claim_id \"C999\"")),
        "{exp_probs:?}"
    );
    let bad_exp_commit = exp_readme.replace(
        "8fcf7c62f4a4de7973193dd2407803387c9e6851",
        "0000000000000000000000000000000000000000",
    );
    let commit_probs = check_experiments(
        &root,
        &exp_jsonl,
        &bad_exp_commit,
        &exp_synthesis,
        &ledger_text,
    )
    .expect("exp commit check runs");
    assert!(
        commit_probs
            .iter()
            .any(|p| p.contains("catalog_snapshot_commit")),
        "{commit_probs:?}"
    );
    let bad_exp_path = exp_jsonl.replacen("\"docs/paper/claims.json\"", "\"../outside.json\"", 1);
    let path_probs = check_experiments(
        &root,
        &bad_exp_path,
        &exp_readme,
        &exp_synthesis,
        &ledger_text,
    )
    .expect("exp path check runs");
    assert!(
        path_probs
            .iter()
            .any(|p| p.contains("unsafe data.sources path")),
        "{path_probs:?}"
    );
    let undeclared_prose_claim = exp_jsonl.replacen(
        "\"summary\": \"Trained the",
        "\"summary\": \"Trained the (C03)",
        1,
    );
    let undeclared_probs = check_experiments(
        &root,
        &undeclared_prose_claim,
        &exp_readme,
        &exp_synthesis,
        &ledger_text,
    )
    .expect("undeclared prose claim check runs");
    assert!(
        undeclared_probs.iter().any(|p| p.contains(
            "EXP-001-v1-flywheel-baseline cites C03 in prose but omits it from claim_ids"
        )),
        "{undeclared_probs:?}"
    );
    let bad_anchor = exp_synthesis.replacen(
        "../paper/NUMBERS_TRACE.md#fresh-clone-run",
        "../paper/NUMBERS_TRACE.md#no-such-anchor",
        1,
    );
    let anchor_probs = check_experiments(&root, &exp_jsonl, &exp_readme, &bad_anchor, &ledger_text)
        .expect("bad anchor check runs");
    assert!(
        anchor_probs
            .iter()
            .any(|p| p.contains("broken anchor #no-such-anchor")),
        "{anchor_probs:?}"
    );
    let bad_matrix = exp_readme.replacen("| `ESTABLISHED` |", "| `PENDING` |", 1);
    let matrix_probs =
        check_experiments(&root, &exp_jsonl, &bad_matrix, &exp_synthesis, &ledger_text)
            .expect("bad matrix check runs");
    assert!(
        matrix_probs.iter().any(|p| p.contains(
            "Master Experiment Matrix EXP-001-v1-flywheel-baseline has status \"`PENDING`\""
        )),
        "{matrix_probs:?}"
    );
    let unlinked_synthesis = exp_synthesis.replace(
        "../paper/figures/clover-why-it-matters-3panel.svg",
        "../paper/figures/why-it-matters.svg",
    );
    let synth_svg_probs = check_experiments(
        &root,
        &exp_jsonl,
        &exp_readme,
        &unlinked_synthesis,
        &ledger_text,
    )
    .expect("unlinked synthesis check runs");
    assert!(
            synth_svg_probs.iter().any(|p| p.contains(
                "SYNTHESIS_AND_NEXT_EXPERIMENTS.md: does not link ../paper/figures/clover-why-it-matters-3panel.svg"
            )),
            "{synth_svg_probs:?}"
        );
}
