use super::*;

fn sites() -> Vec<RenderSite> {
    scan_repo(&repo_root()).expect("scan the repo")
}

/// The scanner must not regress into finding nothing: a silent scan failure
/// would make every other check in this file vacuous.
#[test]
fn scanner_finds_the_known_render_sites() {
    let s = sites();
    assert!(
        s.len() >= 8,
        "expected >=8 spec render sites, found {}: {:#?}",
        s.len(),
        s.iter().map(|x| x.location()).collect::<Vec<_>>()
    );
    for (file, function, _) in REGISTRY {
        assert!(
            s.iter().any(|x| x.file == *file && x.function == *function),
            "registry entry {file}::{function} matched no render site -- the scanner \
             or the file moved"
        );
    }
}

#[test]
fn workspace_scan_detects_new_renderers_in_both_languages() {
    let root = tempfile::tempdir().unwrap();
    for (directory, name, source) in [
        (
            "crates/jope-campaigns/src/bin",
            "new.rs",
            "fn render_spec() {\n let s = \"[resource]\\nram_mb = 768\\n\";\n}\n",
        ),
        (
            "python/test-support/jope/research",
            "new.py",
            "def render_spec():\n return '''[resource]\nram_mb = 768\n'''\n",
        ),
    ] {
        let path = root.path().join(directory);
        std::fs::create_dir_all(&path).unwrap();
        std::fs::write(path.join(name), source).unwrap();
    }
    let sites = scan_repo(root.path()).unwrap();
    assert_eq!(sites.len(), 2);
    let findings = audit(&sites);
    for site in sites {
        assert!(findings.iter().any(|f| matches!(
            f, Finding::UnregisteredSite { site: found } if *found == site.location()
        )));
    }
    // Passing a crate directory instead of the workspace cannot pass silently.
    assert!(!audit(&scan_repo(&root.path().join("crates/jope-campaigns")).unwrap()).is_empty());
}

/// A commented-out reservation must never be read as a live one. This
/// regression is real: `python/test-support/jope/research/test_exact_ops.py` documents the old 49152
/// directly above the live 12288.
#[test]
fn commented_out_values_are_not_parsed() {
    let py = "[resource]\n# ram_mb = 49152 was the old value\nram_mb = 12288\n";
    let lines = strip_comments(py, true);
    let joined = lines.join("\n");
    assert_eq!(find_value(&joined, "ram_mb").as_deref(), Some("12288"));

    let rs = "[resource]\n// ram_mb = 49152 stalled the lane\nram_mb = 16384\n";
    let lines = strip_comments(rs, false);
    let joined = lines.join("\n");
    assert_eq!(find_value(&joined, "ram_mb").as_deref(), Some("16384"));
}

/// All three literal styles used in this tree must parse identically.
#[test]
fn value_extraction_handles_every_literal_style() {
    assert_eq!(
        find_value("        \"ram_mb = 12288\",", "ram_mb").as_deref(),
        Some("12288")
    );
    assert_eq!(
        find_value("        \"ram_mb = 49152\".to_owned(),", "ram_mb").as_deref(),
        Some("49152")
    );
    assert_eq!(
        find_value("         ram_mb = 16384\\n\\", "ram_mb").as_deref(),
        Some("16384")
    );
    assert_eq!(
        find_value("         placement = \\\"single_node\\\"\\n", "placement").as_deref(),
        Some("single_node")
    );
    // must not match a longer identifier that merely ends in the key
    assert_eq!(find_value("packed_ram_mb = 99", "ram_mb"), None);
}

/// THE DELIVERABLE. Two renderers of one family must never disagree on a
/// shared parameter, and no renderer may use a form scq drops or rejects.
///
/// This is the test that would have caught all four of 2026-07-28's stalls
/// before they shipped.
#[test]
fn no_two_renderers_disagree_and_none_use_a_dropped_form() {
    let findings = audit(&sites());
    if !findings.is_empty() {
        let mut msg = format!(
            "\n{} spec-drift finding(s) -- a job family with two reservations is an \
             outage waiting to happen:\n\n",
            findings.len()
        );
        for f in &findings {
            msg.push_str(&format!("{f}\n\n"));
        }
        panic!("{msg}");
    }
}

/// The scanner itself must be able to SEE drift. Without this, a scanner bug
/// that returns nothing would make the deliverable silently pass forever.
#[test]
fn audit_detects_injected_disagreement() {
    let mk = |file: &str, ram: &str| RenderSite {
        file: file.to_owned(),
        line: 1,
        function: "render_spec".to_owned(),
        family: Some(Family::Trainer),
        params: BTreeMap::from([("ram_mb".to_owned(), ram.to_owned())]),
        dep_form: DepForm::Smoke,
        emits_resource_node: false,
        emits_ram_gb: false,
        probes_etc_hostname: false,
    };
    let injected = vec![mk("a.rs", "16384"), mk("b.py", "8192")];
    let findings = audit(&injected);
    assert!(
        findings.iter().any(|f| matches!(
            f,
            Finding::Disagreement { key, .. } if key == "ram_mb"
        )),
        "audit failed to notice 16384 vs 8192 -- the drift test is broken: {findings:#?}"
    );
}

/// The checked-in JSON the python/test-support/jope/research/ lane reads must match this crate exactly.
/// Without this the Python renderers would silently keep a stale copy of the
/// numbers -- which is the very failure mode this workstream exists to end.
///
/// Regenerate with:
///   cargo run --release -p jope-dime-regressor --bin jope_spec_drift -- --emit-json \
///     > python/test-support/jope/research/jope-spec-policy.json
#[test]
fn checked_in_policy_json_matches_this_module() {
    let path = repo_root().join("python/test-support/jope/research/jope-spec-policy.json");
    let on_disk = std::fs::read_to_string(&path).unwrap_or_else(|e| {
        panic!(
            "{} is missing ({e}). Regenerate: cargo run --release -p jope-dime-regressor \
             --bin jope_spec_drift -- --emit-json > python/test-support/jope/research/jope-spec-policy.json",
            path.display()
        )
    });
    assert_eq!(
        on_disk,
        spec_policy::policy_json(),
        "python/test-support/jope/research/jope-spec-policy.json is STALE -- the Python lane is reading \
         different numbers from the Rust policy. Regenerate: cargo run --release \
         -p jope-dime-regressor --bin jope_spec_drift -- --emit-json \
         > python/test-support/jope/research/jope-spec-policy.json"
    );
}

#[test]
fn python_projection_matches_rust_rendering() {
    let output = std::process::Command::new("python3")
        .arg("-c")
        .arg(
            r#"
import os, sys
from unittest.mock import patch
sys.path.insert(0, sys.argv[1])
import jope_spec_policy as p
for settings, expected in [
({'JOPE_STATE': '/state//./explicit/', 'XDG_STATE_HOME': '/xdg'}, '/state//./explicit/'),
({'JOPE_STATE': '', 'XDG_STATE_HOME': '/xdg/state'}, '/xdg/state/jope'),
({'JOPE_STATE': '', 'XDG_STATE_HOME': 'relative/ignored', 'JOPE_HOME': '/source'},
 os.path.join(os.environ['HOME'], '.local/state/jope')
 if os.environ.get('HOME', '').startswith('/') else '/source/state'),
]:
with patch.dict(os.environ, settings):
    assert p._state_root() == expected, (settings, p._state_root(), expected)
for invalid in ({'unknown': 'x'}, {'literal': 3}, {'under_flywheel_root': '../x'},
            {'under_flywheel_root': '/x'}, {'literal': 'x', 'extra': 'y'}):
try:
    p._render_arg(invalid)
except p.SpecPolicyError:
    pass
else:
    raise AssertionError(f'accepted invalid argument: {invalid}')
for family in p.FAMILIES:
print(p.render_resource_block(family) + p.render_placement_block(family), end='')
"#,
        )
        .arg(repo_root().join("python/test-support/jope/research"))
        .output()
        .expect("python3 is required for cross-language policy verification");
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let expected: String = Family::all()
        .iter()
        .map(|family| {
            let p = spec_policy::policy(*family);
            p.render_resource_block() + &p.render_placement_block()
        })
        .collect();
    assert_eq!(String::from_utf8(output.stdout).unwrap(), expected);
}

/// Both spellings of a smoke dependency are enforced by scq and must both be
/// recognised. The live canary now probes an immutable bundle capability;
/// a synthetic legacy renderer keeps coverage for the inline spelling and
/// the forbidden hostname form.
#[test]
fn both_smoke_spellings_are_recognised_and_hostname_pins_are_flagged() {
    let s = sites();
    let canary = s
        .iter()
        .find(|x| x.file == "python/test-support/jope/research/test_exact_canary.py")
        .expect("canary render site");
    assert_eq!(
        canary.dep_form,
        DepForm::Smoke,
        "the canary capability must use an enforced smoke dependency"
    );
    assert!(!canary.probes_etc_hostname);
    assert!(
        !audit(std::slice::from_ref(canary))
            .iter()
            .any(|f| matches!(f, Finding::HostnamePinnedProbe { .. })),
        "the immutable-bundle capability must not be reported as a hostname pin"
    );

    let root = tempfile::tempdir().expect("temporary synthetic renderer root");
    std::fs::create_dir_all(root.path().join("python/test-support/jope/research")).unwrap();
    std::fs::write(
        root.path().join("python/test-support/jope/research/test_exact_canary.py"),
        "def render():\n    return '''[resource]\nram_mb = 768\n[dependencies]\nsmoke = [{ command = \"/usr/bin/grep\", args = [\"-qx\", \"xbabe2\", \"/etc/hostname\"] }]\n'''\n",
    )
    .unwrap();
    let synthetic = scan_file(
        root.path(),
        "python/test-support/jope/research/test_exact_canary.py",
    )
    .unwrap();
    let legacy = synthetic.first().expect("synthetic inline smoke site");
    assert_eq!(legacy.dep_form, DepForm::SmokeInline);
    assert!(legacy.probes_etc_hostname);
    assert!(
        audit(std::slice::from_ref(legacy))
            .iter()
            .any(|f| matches!(f, Finding::HostnamePinnedProbe { .. })),
        "the audit must raise HostnamePinnedProbe for an /etc/hostname probe"
    );
}

/// A whole spec rendered as ONE `\n`-escaped line must parse exactly like a
/// multi-line one. Five real render sites in `crates/jope-campaigns/src/test_adaptive` are written
/// that way; before the window was unescaped they reported `params {}`,
/// so every reservation in them was invisible to the drift check.
#[test]
fn single_line_escaped_specs_are_parsed() {
    let s = sites();
    let site = s
        .iter()
        .find(|x| {
            x.file == "crates/jope-campaigns/src/test_adaptive/scq_batch.rs"
                && x.function == "submit_true"
        })
        .expect("scq_batch::submit_true render site");
    assert_eq!(
        site.params.get("gpus_per_task").map(String::as_str),
        Some("0"),
        "a one-line `\\n`-escaped spec must yield its [resource] values: {:?}",
        site.params
    );
    assert_eq!(site.dep_form, DepForm::Smoke);
    assert!(!site.emits_resource_node);
}

/// `[resource] node` must be detected from the SECTION, not from a stray
/// `format!` named argument, and a placeholder value must not hide it.
#[test]
fn resource_node_is_detected_from_the_section_only() {
    let spec = "[resource]\ngpus_per_task = 0\nnode = {node:?}\ngang_size = 1\n\n\
                [limits]\nmax_restarts = 0\n";
    assert!(section_declares(resource_section(spec), "node"));
    // A `node` key in a LATER section is not a [resource] pin.
    let other = "[resource]\ngpus_per_task = 0\n\n[meta]\nnode = \"xbabe3\"\n";
    assert!(!section_declares(resource_section(other), "node"));
    // `placement = "single_node"` must never read as a node pin.
    let placement = "[resource]\ngang_size = 1\nplacement = \"single_node\"\n";
    assert!(!section_declares(resource_section(placement), "node"));
}

/// Setting both RAM units is a hard submit rejection and must be a finding.
#[test]
fn both_ram_units_is_a_finding() {
    let site = RenderSite {
        file: "crates/jope-campaigns/src/bin/whatever.rs".to_owned(),
        line: 1,
        function: "render_spec".to_owned(),
        family: Some(Family::Eval),
        params: BTreeMap::from([("ram_mb".to_owned(), "512".to_owned())]),
        dep_form: DepForm::None,
        emits_resource_node: false,
        emits_ram_gb: true,
        probes_etc_hostname: false,
    };
    assert!(
        audit(&[site])
            .iter()
            .any(|f| matches!(f, Finding::RamUnitConflict { .. })),
        "a spec setting ram_mb AND ram_gb must be reported -- scq refuses it"
    );
}

/// An unregistered renderer must fail, so a sixth spec path cannot appear
/// unreviewed.
#[test]
fn audit_rejects_an_unregistered_render_site() {
    let orphan = RenderSite {
        file: "crates/jope-campaigns/src/bin/brand_new_thing.rs".to_owned(),
        line: 42,
        function: "render_spec".to_owned(),
        family: Some(Family::Eval),
        params: BTreeMap::new(),
        dep_form: DepForm::None,
        emits_resource_node: false,
        emits_ram_gb: false,
        probes_etc_hostname: false,
    };
    assert!(
        audit(&[orphan])
            .iter()
            .any(|f| matches!(f, Finding::UnregisteredSite { .. })),
        "an unregistered render site must fail the drift test"
    );
}
