//! WS-2: spec drift checker.
//!
//! Reads every scq spec renderer in the tree and fails when two renderers of the
//! same job family disagree, when any renderer disagrees with `spec_policy.rs`,
//! or when a renderer uses a form scq silently drops (`[dependencies] hostname`)
//! or rejects outright (`[resource] node`).
//!
//! The same checks run as `cargo test --lib spec_drift`; this binary exists so
//! ops can run them against a deployed tree, and so `--verify-probes` can be run
//! ON each node.
//!
//! ```text
//! jope_spec_drift                 # audit; exit 1 on any finding
//! jope_spec_drift --list          # every render site the scanner sees
//! jope_spec_drift --emit-json     # the machine-readable policy (scripts/ lane)
//! jope_spec_drift --verify-probes # run this family's smoke probes HERE
//! ```
//!
//! `--verify-probes` is the check that catches a capability probe which is
//! really a hostname pin. Run it on every node:
//! ```text
//! for h in xbabe1 xbabe3; do ssh $h jope_spec_drift --verify-probes; done
//! ```
//! A probe that passes on only one node makes a job UNSUBMITTABLE whenever that
//! node is busy -- scq probes only nodes that already have the requested
//! resources free, and rejects the submit if none of them pass.

use std::path::PathBuf;
use std::process::{Command, ExitCode};

use jope_dime_regressor::spec_drift;
use jope_dime_regressor::spec_policy::{self, Family, ProbeArg};

fn usage() -> &'static str {
    "jope_spec_drift [--list|--emit-json|--verify-probes|--root <dir>]"
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut root: PathBuf = spec_drift::repo_root();
    let mut mode = "audit";

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--list" => mode = "list",
            "--emit-json" => mode = "emit-json",
            "--verify-probes" => mode = "verify-probes",
            "--emit-blocks" => mode = "emit-blocks",
            "-h" | "--help" => {
                println!("{}", usage());
                return ExitCode::SUCCESS;
            }
            "--root" => {
                i += 1;
                match args.get(i) {
                    Some(v) => root = PathBuf::from(v),
                    None => {
                        eprintln!("--root needs a directory\n{}", usage());
                        return ExitCode::FAILURE;
                    }
                }
            }
            other => {
                eprintln!("unknown argument {other}\n{}", usage());
                return ExitCode::FAILURE;
            }
        }
        i += 1;
    }

    if mode == "emit-json" {
        print!("{}", spec_policy::policy_json());
        return ExitCode::SUCCESS;
    }
    if mode == "verify-probes" {
        return verify_probes();
    }
    if mode == "emit-blocks" {
        // Byte-for-byte comparison target for scripts/jope_spec_policy.py, so the
        // two language bindings can be proven identical rather than assumed so.
        for family in Family::all() {
            let p = spec_policy::policy(family);
            println!("===== {} =====", family.as_str());
            print!(
                "{}{}",
                p.render_resource_block(),
                p.render_placement_block()
            );
        }
        return ExitCode::SUCCESS;
    }

    let sites = match spec_drift::scan_repo(&root) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("cannot scan {}: {e}", root.display());
            return ExitCode::FAILURE;
        }
    };

    if mode == "list" {
        println!(
            "{} spec render site(s) under {}\n",
            sites.len(),
            root.display()
        );
        for s in &sites {
            let fam = match s.family {
                Some(f) => f.as_str(),
                None => "(out of scope)",
            };
            println!("{:<12} {}", fam, s.location());
            println!("             params {:?}", s.params);
            println!("             placement {}", s.dep_form);
        }
        return ExitCode::SUCCESS;
    }

    let findings = spec_drift::audit(&sites);
    if findings.is_empty() {
        println!(
            "spec-drift: OK -- {} render site(s), every family agrees with itself and \
             with spec_policy.rs",
            sites.len()
        );
        return ExitCode::SUCCESS;
    }
    eprintln!(
        "spec-drift: {} finding(s) across {} render site(s)\n",
        findings.len(),
        sites.len()
    );
    for f in &findings {
        eprintln!("{f}\n");
    }
    ExitCode::FAILURE
}

/// Execute every distinct smoke probe on THIS host.
///
/// A probe is only a capability probe if it passes wherever the capability is
/// installed. One that passes on a single node is a hostname pin, and it turns a
/// busy cluster into a submit failure rather than a queue.
fn verify_probes() -> ExitCode {
    let host = std::fs::read_to_string("/etc/hostname")
        .map(|s| s.trim().to_owned())
        .unwrap_or_else(|_| "<unknown>".to_owned());
    println!("smoke probe verification on {host}");

    let mut seen: Vec<&str> = vec![];
    let mut failed = 0usize;
    let mut checked = 0usize;
    for family in Family::all() {
        let policy = spec_policy::policy(family);
        for probe in policy.smoke {
            if seen.contains(&probe.name) {
                continue;
            }
            seen.push(probe.name);
            checked += 1;
            let status = Command::new(probe.command)
                .args(probe.args.iter().map(ProbeArg::render))
                .status();
            let ok = matches!(&status, Ok(s) if s.success());
            if !ok {
                failed += 1;
            }
            println!(
                "  [{}] {:<34} {} {}",
                if ok { "PASS" } else { "FAIL" },
                probe.name,
                probe.command,
                probe
                    .args
                    .iter()
                    .map(ProbeArg::render)
                    .collect::<Vec<_>>()
                    .join(" ")
            );
            if let Err(e) = &status {
                println!("         could not execute: {e}");
            }
        }
    }
    if checked == 0 {
        println!("  (no smoke probes declared)");
        return ExitCode::SUCCESS;
    }
    if failed > 0 {
        eprintln!(
            "\n{failed} of {checked} probe(s) FAIL on {host}. Any GPU job declaring them \
             cannot be submitted when {host} is the only node with free capacity."
        );
        return ExitCode::FAILURE;
    }
    println!("\nall {checked} probe(s) pass on {host}");
    ExitCode::SUCCESS
}

/// Compile-time proof that the checker still understands the two forms scq
/// treats specially. If these ever stop being distinguishable the audit is
/// meaningless.
#[cfg(test)]
mod tests {
    use jope_dime_regressor::spec_drift::DepForm;

    #[test]
    fn dep_forms_are_distinguishable() {
        assert_ne!(DepForm::Smoke, DepForm::HostnameNoOp);
        assert_ne!(DepForm::Smoke, DepForm::None);
        assert_eq!(format!("{}", DepForm::Smoke), "[[dependencies.smoke]]");
    }
}
