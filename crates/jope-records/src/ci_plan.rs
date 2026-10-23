//! `ci-plan`: declare every fast lane's applicability before execution, ported from
//! `deploy/ci/ci-plan.py` (`PORT-RUST-03`).

use std::collections::BTreeMap;
use std::fs::OpenOptions;
use std::io::Write;
use std::path::Path;
use std::process::Command;

use regex::Regex;

use crate::pyjson::{self, Dict, Value};
use crate::{require, Error, Result};

pub const SCHEMA: &str = "jope.ci-plan/v1";
pub const LANES: [&str; 8] = [
    "changes",
    "hygiene",
    "rust-check",
    "rust-test",
    "emitters",
    "inputs",
    "python",
    "e2e-smoke",
];

const IMAGE: &str = r"ops/ci/|tools/security-lane\.sh$|deploy/runner/|deploy/docker/|\.github/actions/|deploy/ci/(cargo-artifact|receipt|results|ci-plan|fast-lane|pytest-evidence|aggregate|coverage|training-proof|test-training-proof)";

fn filters() -> Vec<(&'static str, Regex)> {
    let common =
        format!(r"Cargo\.(toml|lock)$|rust-toolchain\.toml$|{IMAGE}|\.github/workflows/ci\.yml$");
    vec![
        (
            "rust",
            Regex::new(&format!(
                r"^(crates/|benchmarks/|schemas/|{common}|\.github/workflows/typescript\.yml$|\.config/nextest\.toml$|deny\.toml$|deploy/ci/|deploy/systemd/|deploy/appliance/|models/bundles/|docs/release/signing\.pub$)"
            ))
            .expect("valid rust regex"),
        ),
        (
            "emitters",
            Regex::new(&format!(
                r"^(crates/feat-math/|datasets/test/|deploy/ci/emitters|{common})"
            ))
            .expect("valid emitters regex"),
        ),
        (
            "inputs",
            Regex::new(&format!(
                r"^(datasets/synthetic/|crates/(jope-cli|jope-campaigns|jope-arena)/|python/src/jope/research/|{common})"
            ))
            .expect("valid inputs regex"),
        ),
        (
            "python",
            Regex::new(&format!(
                r"^(python/|benchmarks/|crates/jope-bench/adapters/|deploy/systemd/|deploy/dl-flywheel/pipelines/rlfw/|deploy/dl-flywheel/pipelines/standalone_paper_monitor_v1/|deploy/dl-flywheel/dashboard/|{IMAGE}|\.github/workflows/ci\.yml$)"
            ))
            .expect("valid python regex"),
        ),
        (
            "e2e",
            Regex::new(&format!(
                r"^(crates/(jope-pipeline|jope-cli|jope-engine|jope-datasets|feat-math|jope-assets|jope-runtime|jope-appliance|jope-dist)/|datasets/test/|schemas/|deploy/ci/(e2e-smoke|emitters)|deploy/appliance/|docs/release/signing\.pub$|{common})"
            ))
            .expect("valid e2e regex"),
        ),
    ]
}

fn gate_for_lane(lane: &str) -> Option<&'static str> {
    match lane {
        "rust-check" | "rust-test" => Some("rust"),
        "emitters" => Some("emitters"),
        "inputs" => Some("inputs"),
        "python" => Some("python"),
        "e2e-smoke" => Some("e2e"),
        _ => None,
    }
}

/// Selected lanes by name, and the path gates in their declared order.
pub type Classification = (BTreeMap<String, bool>, Vec<(&'static str, bool)>);

/// The run that a plan is bound to: the CI event and the commits and attempt it describes.
pub struct PlanRun<'a> {
    pub event: &'a str,
    pub base_sha: &'a str,
    pub source_sha: &'a str,
    pub checkout_sha: &'a str,
    pub run_id: &'a str,
    pub attempt: i64,
}

/// Classify changed `paths` under `event` (`pull_request` or `workflow_dispatch`), returning
/// `(lanes, gates)` with the exact lane and gate order of `deploy/ci/ci-plan.py`.
pub fn classify(paths: &[String], event: &str) -> Result<Classification> {
    require!(
        event == "pull_request" || event == "workflow_dispatch",
        "unsupported fast CI event: {event}"
    );
    let mut gates = Vec::new();
    let mut gate_map = BTreeMap::new();
    for (name, pattern) in filters() {
        let active = event == "workflow_dispatch" || paths.iter().any(|p| pattern.is_match(p));
        gates.push((name, active));
        gate_map.insert(name, active);
    }
    let mut lanes = BTreeMap::new();
    for lane in LANES {
        let selected = match gate_for_lane(lane) {
            Some(gate) => *gate_map.get(gate).unwrap_or(&true),
            None => true,
        };
        lanes.insert(lane.to_owned(), selected);
    }
    Ok((lanes, gates))
}

/// Build the `jope.ci-plan/v1` JSON object.
pub fn build_plan_value(
    run: &PlanRun<'_>,
    paths: &[String],
    lanes: &BTreeMap<String, bool>,
) -> Value {
    let PlanRun {
        event,
        base_sha,
        source_sha,
        checkout_sha,
        run_id,
        attempt,
    } = *run;
    let mut lane_dict = Dict::new();
    for lane in LANES {
        if let Some(&v) = lanes.get(lane) {
            lane_dict.insert(lane, Value::Bool(v));
        }
    }
    let mut plan = Dict::new();
    plan.insert("schema", SCHEMA.into());
    plan.insert("event", event.into());
    plan.insert("base_sha", base_sha.into());
    plan.insert("source_sha", source_sha.into());
    plan.insert("checkout_sha", checkout_sha.into());
    plan.insert("run_id", run_id.into());
    plan.insert("attempt", attempt.into());
    plan.insert(
        "paths",
        Value::List(paths.iter().map(|p| Value::from(p.as_str())).collect()),
    );
    plan.insert("lanes", Value::Dict(lane_dict));
    Value::Dict(plan)
}

/// Run `ci-plan` from environment variables (`GITHUB_EVENT_NAME`, `BASE_SHA`, `JOPE_SOURCE_SHA`,
/// `GITHUB_SHA`, `GITHUB_RUN_ID`, `GITHUB_RUN_ATTEMPT`, `GITHUB_OUTPUT`) and write `--output`
/// create-only.
pub fn run_from_env(repo: &Path, output: &Path) -> Result<(String, bool)> {
    let event = std::env::var("GITHUB_EVENT_NAME")
        .map_err(|_| Error::Shape("missing GITHUB_EVENT_NAME".into()))?;
    let base = std::env::var("BASE_SHA").unwrap_or_default();
    let sha_re = Regex::new(r"^[a-f0-9]{40}$").unwrap();
    if event == "pull_request" && !sha_re.is_match(&base) {
        return Err(Error::Invalid(
            "PR path classification requires full base SHA".into(),
        ));
    }
    let mut paths: Vec<String> = if !base.is_empty() {
        let out = Command::new("git")
            .args(["diff", "--name-only", "-z", &base, "HEAD"])
            .current_dir(repo)
            .output()
            .map_err(|e| Error::Command(format!("git diff: {e}")))?;
        if !out.status.success() {
            return Err(Error::Command("git diff failed".into()));
        }
        String::from_utf8_lossy(&out.stdout)
            .split('\0')
            .filter(|s| !s.is_empty())
            .map(str::to_owned)
            .collect()
    } else {
        Vec::new()
    };
    paths.sort();
    paths.dedup();

    let (lanes, gates) = classify(&paths, &event)?;
    let source_sha = std::env::var("JOPE_SOURCE_SHA")
        .map_err(|_| Error::Shape("missing JOPE_SOURCE_SHA".into()))?;
    let checkout_sha =
        std::env::var("GITHUB_SHA").map_err(|_| Error::Shape("missing GITHUB_SHA".into()))?;
    let run_id =
        std::env::var("GITHUB_RUN_ID").map_err(|_| Error::Shape("missing GITHUB_RUN_ID".into()))?;
    let attempt: i64 = std::env::var("GITHUB_RUN_ATTEMPT")
        .map_err(|_| Error::Shape("missing GITHUB_RUN_ATTEMPT".into()))?
        .parse()
        .map_err(|_| Error::Invalid("invalid GITHUB_RUN_ATTEMPT".into()))?;

    let run = PlanRun {
        event: &event,
        base_sha: &base,
        source_sha: &source_sha,
        checkout_sha: &checkout_sha,
        run_id: &run_id,
        attempt,
    };
    let plan = build_plan_value(&run, &paths, &lanes);
    let rendered = format!("{}\n", pyjson::dumps(&plan, false, Some(2)));
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(output)
        .map_err(|e| Error::io(output, e))?;
    file.write_all(rendered.as_bytes())
        .map_err(|e| Error::io(output, e))?;

    if let Ok(gh_out) = std::env::var("GITHUB_OUTPUT") {
        let gh_path = Path::new(&gh_out);
        let mut gh_file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(gh_path)
            .map_err(|e| Error::io(gh_path, e))?;
        for (name, selected) in gates {
            writeln!(gh_file, "{name}={selected}").map_err(|e| Error::io(gh_path, e))?;
        }
    }

    Ok((rendered, true))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn docs_only_pull_request_runs_only_changes_and_hygiene() {
        let paths = vec!["docs/learning.md".to_owned(), "README.md".to_owned()];
        let (lanes, _) = classify(&paths, "pull_request").unwrap();
        assert!(lanes["changes"]);
        assert!(lanes["hygiene"]);
        assert!(!lanes["rust-check"]);
        assert!(!lanes["rust-test"]);
        assert!(!lanes["emitters"]);
        assert!(!lanes["inputs"]);
        assert!(!lanes["python"]);
        assert!(!lanes["e2e-smoke"]);
    }

    #[test]
    fn shared_ci_wrappers_and_security_lane_trigger_every_lane() {
        for path in [
            "ops/ci/fast-lane.sh",
            "ops/ci/lib.sh",
            "tools/security-lane.sh",
        ] {
            let (lanes, _) = classify(&[path.to_owned()], "pull_request").unwrap();
            for lane in LANES {
                assert!(lanes[lane], "{path} must activate {lane}");
            }
        }
        for path in ["ops/circuit/unrelated.sh", "tools/security-lane.sh.bak"] {
            let (lanes, _) = classify(&[path.to_owned()], "pull_request").unwrap();
            for lane in LANES {
                assert_eq!(
                    lanes[lane],
                    matches!(lane, "changes" | "hygiene"),
                    "{path}: {lane}"
                );
            }
        }
    }

    #[test]
    fn python_and_rust_classification_agree_for_shared_ci_and_path_boundaries() {
        let paths = [
            "ops/ci/fast-lane.sh",
            "ops/ci/lib.sh",
            "tools/security-lane.sh",
            "ops/circuit/unrelated.sh",
            "tools/security-lane.sh.bak",
            "docs/learning.md",
            "crates/feat-math/src/lib.rs",
            "deploy/ci/aggregate.py",
            ".github/workflows/ci.yml",
            ".github/workflows/typescript.yml",
        ];
        let script = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../deploy/ci/ci-plan.py");
        let output = Command::new("python3")
            .env("PYTHONDONTWRITEBYTECODE", "1")
            .args([
                "-c",
                "import importlib.util,json,sys; s=importlib.util.spec_from_file_location('ci_plan',sys.argv[1]); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); print(json.dumps([[event,path,*m.classify([path],event)] for event in ('pull_request','workflow_dispatch') for path in sys.argv[2:]]))",
            ])
            .arg(script)
            .args(paths)
            .output()
            .expect("python3 is required to check the retained CI classifier parity");
        assert!(
            output.status.success(),
            "Python classifier failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        let mut expected = Vec::new();
        for event in ["pull_request", "workflow_dispatch"] {
            for path in paths {
                let (lanes, gates) = classify(&[path.to_owned()], event).unwrap();
                expected.push(Value::List(vec![
                    event.into(),
                    path.into(),
                    Value::Dict(
                        lanes
                            .into_iter()
                            .map(|(key, value)| (key, value.into()))
                            .collect(),
                    ),
                    Value::Dict(
                        gates
                            .into_iter()
                            .map(|(key, value)| (key, value.into()))
                            .collect(),
                    ),
                ]));
            }
        }
        assert_eq!(
            pyjson::parse_bytes(&output.stdout).unwrap(),
            Value::List(expected)
        );
    }

    #[test]
    fn workflow_dispatch_enables_all_lanes() {
        let (lanes, _) = classify(&[], "workflow_dispatch").unwrap();
        for lane in LANES {
            assert!(lanes[lane], "lane {lane} should be true");
        }
    }

    #[test]
    fn feat_math_change_triggers_rust_emitters_and_e2e_only() {
        let paths = vec!["crates/feat-math/src/lib.rs".to_owned()];
        let (lanes, _) = classify(&paths, "pull_request").unwrap();
        assert!(lanes["rust-check"]);
        assert!(lanes["rust-test"]);
        assert!(lanes["emitters"]);
        assert!(lanes["e2e-smoke"]);
        assert!(!lanes["inputs"]);
        assert!(!lanes["python"]);
    }

    #[test]
    fn appliance_changes_trigger_rust_and_e2e_only() {
        for path in [
            "crates/jope-assets/src/envelope.rs",
            "crates/jope-runtime/src/automl.rs",
            "deploy/appliance/profiles/fixture.toml",
            "docs/release/signing.pub",
        ] {
            let (lanes, _) = classify(&[path.to_owned()], "pull_request").unwrap();
            assert!(lanes["rust-check"], "{path}");
            assert!(lanes["rust-test"], "{path}");
            assert!(lanes["e2e-smoke"], "{path}");
            assert!(!lanes["emitters"], "{path}");
            assert!(!lanes["inputs"], "{path}");
            assert!(!lanes["python"], "{path}");
        }
        let (lanes, _) = classify(
            &["models/bundles/hyperion-v7.json".to_owned()],
            "pull_request",
        )
        .unwrap();
        assert!(lanes["rust-test"]);
        assert!(!lanes["e2e-smoke"]);
    }

    #[test]
    fn rejects_unsupported_event() {
        let err = classify(&[], "push").unwrap_err().to_string();
        assert_eq!(err, "unsupported fast CI event: push");
    }
}
