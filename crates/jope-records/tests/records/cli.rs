//! The `jope-records` binary: same flags, defaults and printed summaries as the Python CLIs.

use std::path::Path;
use std::process::{Command, Output};

use jope_records::pyjson::{self, Value};
use jope_records::{
    claims, dispositions, findings, layout, progress_view, task_status, test_inventory,
};

use crate::{repo, scratch};

fn run(args: &[&str], cwd: &Path) -> Output {
    Command::new(env!("CARGO_BIN_EXE_jope-records"))
        .args(args)
        .current_dir(cwd)
        .output()
        .expect("run jope-records")
}

/// `print(json.dumps(summary, sort_keys=True))`.
fn printed(summary: &Value) -> String {
    format!("{}\n", pyjson::dumps(summary, true, None))
}

#[track_caller]
fn succeeds(output: &Output) -> String {
    assert!(
        output.status.success(),
        "exit {:?}: {}",
        output.status,
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8(output.stdout.clone()).expect("utf-8 summary")
}

#[test]
fn committed_records_print_the_python_summaries() {
    let repo = repo();
    let repo_arg = repo.display().to_string();
    let findings_root = layout::findings(&repo);
    let expected = [
        (
            vec!["verify-findings"],
            printed(
                &findings::verify(&findings_root)
                    .expect("findings")
                    .to_json(),
            ),
        ),
        (
            vec!["verify-dispositions"],
            printed(
                &dispositions::verify(&findings_root, false, None)
                    .expect("dispositions")
                    .to_json(),
            ),
        ),
        (
            vec!["verify-claims"],
            printed(
                &claims::verify(&layout::claims(&repo), None, false)
                    .expect("claims")
                    .to_json(),
            ),
        ),
        (
            vec!["verify-test-inventory"],
            printed(
                &test_inventory::verify(&layout::test_inventory(&repo), &repo)
                    .expect("inventory")
                    .to_json(),
            ),
        ),
        (
            vec!["task-status"],
            printed(&task_status::summary(
                &task_status::verify(&layout::work_orders(&repo)).expect("task status"),
            )),
        ),
        (
            vec!["progress-view"],
            printed(&progress_view::summary(
                &progress_view::collect(&layout::core_ip(&repo)).expect("progress"),
            )),
        ),
    ];
    for (args, summary) in expected {
        // Defaults resolve from the git toplevel of the working directory ...
        let from_git = run(&args, &repo.join("docs"));
        assert_eq!(succeeds(&from_git), summary, "{args:?}");
        // ... or from --repo.
        let mut explicit = args.clone();
        explicit.extend(["--repo", &repo_arg]);
        assert_eq!(succeeds(&run(&explicit, &repo)), summary, "{args:?} --repo");
    }
    let findings = succeeds(&run(&["verify-findings"], &repo));
    assert_eq!(
        findings,
        "{\"entries\": 195, \"gap\": 97, \"improvement\": 98, \"scope\": \"inventory integrity only; no product finding closed\"}\n"
    );
}

#[test]
fn explicit_paths_and_failures() {
    let repo = repo();
    let work_orders = layout::work_orders(&repo);
    let directory = scratch();
    let root = directory.path();
    for entry in std::fs::read_dir(&work_orders).expect("work orders") {
        let entry = entry.expect("entry");
        let name = entry.file_name().to_string_lossy().into_owned();
        if name.ends_with(".tasks.md") || name == "task-status.json" {
            std::fs::copy(entry.path(), root.join(&name)).expect("copy");
        }
    }
    std::fs::write(root.join("STATUS.md"), "hand-edited view\n").expect("stale view");
    let root_arg = root.display().to_string();
    let stale = run(&["task-status", "--root", &root_arg], &repo);
    assert_eq!(stale.status.code(), Some(1));
    assert!(String::from_utf8_lossy(&stale.stderr)
        .contains("STATUS.md is stale; regenerate with --write"));
    succeeds(&run(
        &["task-status", "--root", &root_arg, "--write"],
        &repo,
    ));
    assert_eq!(
        std::fs::read(root.join("STATUS.md")).expect("regenerated"),
        std::fs::read(work_orders.join("STATUS.md")).expect("committed")
    );
    succeeds(&run(&["task-status", "--root", &root_arg], &repo));

    // A directory outside any checkout needs --repo for its defaults.
    let outside = run(&["verify-findings"], Path::new("/"));
    assert_eq!(outside.status.code(), Some(1));
    assert!(String::from_utf8_lossy(&outside.stderr).contains("--repo"));

    let missing = run(
        &[
            "verify-claims",
            "--claims",
            &root.join("absent.json").display().to_string(),
        ],
        &repo,
    );
    assert_eq!(missing.status.code(), Some(1));
    let custody = run(&["verify-claims", "--require-custody"], &repo);
    let available = claims::verify(&layout::claims(&repo), None, true).is_ok();
    assert_eq!(custody.status.success(), available);
}
