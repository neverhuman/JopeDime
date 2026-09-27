//! `rust-scope-check`: assert `deploy/ci/rust-test-scope.sh` partitions the workspace exactly.
//!
//! A crate added later must not land in neither list: that fails OPEN — it would be gated by no
//! lane and flagged by nothing. jope-arena did exactly that once, between two pushes of the lane
//! itself, which is why this check exists rather than a comment asking people to remember.
//!
//! Rust port of `deploy/ci/rust-test-scope-check.py` (archived at `8a55c4fef`; see
//! `docs/archive/python-archive-manifest.json`) with the same two independent readings of the
//! script and the same report.

use std::collections::BTreeSet;
use std::path::Path;
use std::process::Command;

use regex::Regex;

use crate::py;
use crate::pyjson::{self, Value};
use crate::{Error, Result};

pub const SCRIPT: &str = "deploy/ci/rust-test-scope.sh";

/// The package names the script ACTUALLY EMITS for this invocation.
///
/// Reading the `product=( ... )` / `deferred=( ... )` arrays out of the source instead would check
/// something the lane never runs: a second assignment later in the file wins in bash but loses to
/// a first-match regex, and the `case` block that chooses between the two arrays would not be
/// checked at all. Running the script is the only reading of it that cannot drift from the lane.
///
/// Repeats are kept: the script emits `-p X -p X` for a name written twice, `cargo` accepts it,
/// and turning that into a set here would make the counts printed below disagree with the file.
fn emitted(root: &Path, args: &[&str]) -> Result<String> {
    let output = Command::new("bash")
        .arg(root.join(SCRIPT))
        .args(args)
        .current_dir(root)
        .output()
        .map_err(|e| Error::io(Path::new("bash"), e))?;
    if !output.status.success() {
        return Err(Error::Command(format!(
            "rust-test-scope.sh {}: exit {}\n{}",
            args.join(" "),
            output.status.code().unwrap_or(-1),
            py::strip(&String::from_utf8_lossy(&output.stderr))
        )));
    }
    Ok(String::from_utf8_lossy(&output.stdout).into_owned())
}

/// The package names in the script's stdout for the invocation `shown` (`-p <name>` pairs).
fn parse_emitted(stdout: &str, shown: &str) -> Result<Vec<String>> {
    let tokens: Vec<&str> = stdout
        .split(py::is_space)
        .filter(|t| !t.is_empty())
        .collect();
    if tokens.is_empty()
        || !tokens.len().is_multiple_of(2)
        || tokens.iter().step_by(2).any(|t| *t != "-p")
    {
        return Err(Error::Command(format!(
            "rust-test-scope.sh {shown}: expected `-p <name>` pairs, got: {}",
            pyjson::str_repr(stdout)
        )));
    }
    Ok(tokens
        .iter()
        .skip(1)
        .step_by(2)
        .map(|t| (*t).to_owned())
        .collect())
}

/// The `<var>=( ... )` array as a reader of the file sees it: first match, repeats kept.
///
/// This is deliberately a SECOND, independent reading. Where it disagrees with what the script
/// emits, the file and the lane disagree, and one of the two is lying to whoever reads it.
fn declared(text: &str, var: &str) -> Result<Vec<String>> {
    let pattern =
        Regex::new(&format!(r"(?m)^{}=\(([^)]*)\)", regex::escape(var))).expect("escaped pattern");
    let captures = pattern.captures(text).ok_or_else(|| {
        Error::Command(format!("rust-test-scope.sh: no {var}=( ... ) array found"))
    })?;
    Ok(captures[1]
        .split(py::is_space)
        .filter(|t| !t.is_empty())
        .map(str::to_owned)
        .collect())
}

fn sorted(items: &[String]) -> Vec<String> {
    let mut items = items.to_vec();
    items.sort();
    items
}

/// The report and whether the scope partitions the workspace.
pub fn check(root: &Path) -> Result<(String, bool)> {
    let output = Command::new("cargo")
        .args(["metadata", "--no-deps", "--locked", "--format-version", "1"])
        .current_dir(root)
        .output()
        .map_err(|e| Error::io(Path::new("cargo"), e))?;
    if !output.status.success() {
        return Err(Error::Command(format!(
            "cargo metadata exited {}: {}",
            output.status.code().unwrap_or(-1),
            String::from_utf8_lossy(&output.stderr).trim()
        )));
    }
    let meta = pyjson::parse_bytes(&output.stdout)
        .map_err(|m| Error::parse(Path::new("cargo metadata"), m))?;
    let workspace: BTreeSet<String> = meta
        .get("packages")
        .and_then(Value::as_list)
        .map(|packages| {
            packages
                .iter()
                .filter_map(|p| p.get("name").and_then(Value::as_str).map(str::to_owned))
                .collect()
        })
        .unwrap_or_default();
    let product = emitted(root, &[])?;
    let deferred = emitted(root, &["--deferred"])?;
    let text = py::read_text(&root.join(SCRIPT))?;
    evaluate(&workspace, &product, &deferred, &text)
}

/// The report for the workspace package names, the script's stdout for both scopes and the
/// script text, and whether the scopes partition the workspace. `check` gathers the inputs; the
/// Python-parity fixtures replay recorded ones.
pub fn evaluate(
    workspace: &BTreeSet<String>,
    product_stdout: &str,
    deferred_stdout: &str,
    text: &str,
) -> Result<(String, bool)> {
    let product_list = parse_emitted(product_stdout, "")?;
    let deferred_list = parse_emitted(deferred_stdout, "--deferred")?;
    let mut problems = vec![];
    for (var, listed) in [("product", &product_list), ("deferred", &deferred_list)] {
        let written = declared(text, var)?;
        if sorted(listed) != sorted(&written) {
            problems.push(format!(
                "the {var} scope emits {} but the {var}=( ... ) array reads {} — the file and the lane disagree",
                py::str_list_repr(&sorted(listed)),
                py::str_list_repr(&sorted(&written))
            ));
        }
        let repeats: BTreeSet<&String> = listed
            .iter()
            .filter(|n| listed.iter().filter(|m| m == n).count() > 1)
            .collect();
        if !repeats.is_empty() {
            let repeats: Vec<&str> = repeats.into_iter().map(String::as_str).collect();
            problems.push(format!(
                "emitted more than once for the {var} scope: {}",
                repeats.join(", ")
            ));
        }
    }
    let product: BTreeSet<String> = product_list.iter().cloned().collect();
    let deferred: BTreeSet<String> = deferred_list.iter().cloned().collect();
    let either: BTreeSet<String> = product.union(&deferred).cloned().collect();
    let join = |set: BTreeSet<&String>| {
        set.into_iter()
            .map(String::as_str)
            .collect::<Vec<_>>()
            .join(", ")
    };
    let unplaced: BTreeSet<&String> = workspace.difference(&either).collect();
    if !unplaced.is_empty() {
        problems.push(format!(
            "in the workspace but emitted by neither scope (they would be gated by nothing): {}",
            join(unplaced)
        ));
    }
    let ghosts: BTreeSet<&String> = either.difference(workspace).collect();
    if !ghosts.is_empty() {
        problems.push(format!(
            "emitted but not in the workspace: {}",
            join(ghosts)
        ));
    }
    let both: BTreeSet<&String> = product.intersection(&deferred).collect();
    if !both.is_empty() {
        problems.push(format!("emitted by both scopes: {}", join(both)));
    }
    if !problems.is_empty() {
        let mut out = String::from("rust-test scope does not partition the workspace:\n");
        for problem in &problems {
            out.push_str(&format!("  - {problem}\n"));
        }
        out.push_str(
            "  fix deploy/ci/rust-test-scope.sh (product = gated on pull requests, deferred = self-hosted)\n",
        );
        return Ok((out, false));
    }
    Ok((
        format!(
            "ok: {} workspace packages = {} product + {} deferred (no repeats, no overlap, none unplaced)\n",
            workspace.len(),
            product_list.len(),
            deferred_list.len()
        ),
        true,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The reader's view takes the first assignment, repeats kept; a missing array refuses.
    #[test]
    fn declared_reads_the_first_assignment() {
        let text = "product=(a b a)\nproduct=(c)\n";
        assert_eq!(declared(text, "product").unwrap(), ["a", "b", "a"]);
        assert!(declared(text, "deferred").is_err());
    }
}
