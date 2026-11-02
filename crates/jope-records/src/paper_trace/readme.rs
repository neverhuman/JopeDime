use super::*;

/// `1234567` as `1,234,567`.
pub(super) fn grouped(n: usize) -> String {
    let digits = n.to_string();
    let mut out = String::new();
    for (i, d) in digits.chars().enumerate() {
        if i > 0 && (digits.len() - i).is_multiple_of(3) {
            out.push(',');
        }
        out.push(d);
    }
    out
}

/// Small counts the README spells out ("nine jobs", "forty ... fixtures").
pub(super) fn words(n: usize) -> String {
    const ONES: [&str; 20] = [
        "zero",
        "one",
        "two",
        "three",
        "four",
        "five",
        "six",
        "seven",
        "eight",
        "nine",
        "ten",
        "eleven",
        "twelve",
        "thirteen",
        "fourteen",
        "fifteen",
        "sixteen",
        "seventeen",
        "eighteen",
        "nineteen",
    ];
    const TENS: [&str; 10] = [
        "", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety",
    ];
    match n {
        0..=19 => ONES[n].to_owned(),
        20..=99 if n.is_multiple_of(10) => TENS[n / 10].to_owned(),
        20..=99 => format!("{}-{}", TENS[n / 10], ONES[n % 10]),
        _ => n.to_string(),
    }
}

/// The values the README's "At a glance" footnotes name, recomputed from committed files.
#[derive(Debug, Default, PartialEq)]
pub struct Glance {
    pub version: String,
    pub members: usize,
    pub default_members: usize,
    pub versions: BTreeMap<String, usize>,
    pub vendored: usize,
    pub vendored_version: String,
    pub emit_targets: usize,
    pub rust_tests: usize,
    pub rust_ignored: usize,
    pub rust_packages: usize,
    pub python_tests: usize,
    pub ci_jobs: usize,
    pub full_families: usize,
    pub coverage_items: usize,
    pub coverage_lanes: usize,
    pub matrix: Vec<usize>,
    pub synthetic: usize,
    pub test_datasets: usize,
    pub python_files: usize,
    pub python_under_python: usize,
    pub adrs: usize,
}

fn json(root: &Path, rel: &str) -> Result<Value> {
    let path = root.join(rel);
    pyjson::parse_str_big(&py::read_text(&path)?).map_err(|m| Error::parse(&path, m))
}

fn len_at(doc: &Value, path: &[&str]) -> usize {
    let mut v = Some(doc);
    for key in path {
        v = v.and_then(|x| x.get(key));
    }
    match v {
        Some(Value::List(l)) => l.len(),
        Some(Value::Dict(d)) => d.len(),
        _ => 0,
    }
}

fn int_at(doc: &Value, path: &[&str]) -> usize {
    let mut v = Some(doc);
    for key in path {
        v = v.and_then(|x| x.get(key));
    }
    v.and_then(Value::as_int)
        .and_then(|i| usize::try_from(i).ok())
        .unwrap_or(0)
}

fn git_ls(root: &Path, pathspec: &str) -> Result<usize> {
    let output = Command::new("git")
        .args(["ls-files", "--", pathspec])
        .current_dir(root)
        .output()
        .map_err(|e| Error::io(Path::new("git"), e))?;
    if !output.status.success() {
        return Err(Error::Command(format!("git ls-files {pathspec} failed")));
    }
    Ok(String::from_utf8_lossy(&output.stdout).lines().count())
}

fn toml_doc(path: &Path) -> Result<toml::Table> {
    py::read_text(path)?
        .parse::<toml::Table>()
        .map_err(|e| Error::parse(path, e.to_string()))
}

/// Recompute every "At a glance" value whose footnote reads a committed file.
pub fn glance(root: &Path) -> Result<Glance> {
    let mut g = Glance::default();
    let manifest = toml_doc(&root.join("Cargo.toml"))?;
    let workspace = manifest
        .get("workspace")
        .and_then(toml::Value::as_table)
        .ok_or_else(|| Error::Shape("Cargo.toml: no [workspace]".into()))?;
    g.version = workspace
        .get("package")
        .and_then(|p| p.get("version"))
        .and_then(toml::Value::as_str)
        .unwrap_or_default()
        .to_owned();
    let list = |key: &str| -> Vec<String> {
        workspace
            .get(key)
            .and_then(toml::Value::as_array)
            .map(|a| {
                a.iter()
                    .filter_map(|m| m.as_str().map(str::to_owned))
                    .collect()
            })
            .unwrap_or_default()
    };
    let members = list("members");
    g.members = members.len();
    g.default_members = list("default-members").len();
    for member in &members {
        let package = toml_doc(&root.join(member).join("Cargo.toml"))?;
        let version = match package.get("package").and_then(|p| p.get("version")) {
            Some(toml::Value::String(v)) => v.clone(),
            _ => g.version.clone(),
        };
        if member.starts_with("crates/smartcluster/") {
            g.vendored += 1;
            g.vendored_version = version.clone();
        }
        *g.versions.entry(version).or_default() += 1;
    }

    let kernel = py::read_text(&root.join("crates/jope-cli/src/kernel.rs"))?;
    let arm = re(r#"^\s+"([a-z0-9+-]+)"(?: \| "[a-z0-9+-]+")* => vec!\["#);
    g.emit_targets = kernel
        .split("fn languages")
        .nth(1)
        .unwrap_or("")
        .lines()
        .take_while(|l| !l.starts_with('}'))
        .filter_map(|l| arm.captures(l))
        .filter(|c| &c[1] != "all")
        .count();

    let inventory = json(root, "docs/parity/test-inventory.json")?;
    g.rust_tests = int_at(&inventory, &["rust", "count"]);
    g.rust_ignored = int_at(&inventory, &["rust", "ignored"]);
    g.rust_packages = len_at(&inventory, &["rust", "packages"]);
    g.python_tests = int_at(&inventory, &["python", "count"]);

    let ci = py::read_text(&root.join(".github/workflows/ci.yml"))?;
    let job = re(r"^  [A-Za-z0-9_-]+:\s*$");
    let mut in_jobs = false;
    for line in ci.lines() {
        if !line.starts_with(' ') && !line.is_empty() {
            in_jobs = line.starts_with("jobs:");
            continue;
        }
        if in_jobs && job.is_match(line) {
            g.ci_jobs += 1;
        }
    }
    let coverage = json(root, "deploy/ci/coverage.json")?;
    g.coverage_items = len_at(&coverage, &["items"]);
    g.coverage_lanes = len_at(&coverage, &["lanes"]);
    g.full_families = coverage
        .get("lanes")
        .and_then(Value::as_dict)
        .map_or(0, |d| d.keys().filter(|k| k.starts_with("full-")).count());

    let matrix = py::read_text(&root.join("docs/parity/capability-matrix.md"))?;
    let total = re(r"^\| \*\*total\*\* \|((?: \*\*\d+\*\* \|)+)\s*$");
    let cell = re(r"\d+");
    g.matrix = matrix
        .lines()
        .find_map(|l| total.captures(l))
        .map(|c| {
            cell.find_iter(&c[1])
                .filter_map(|m| m.as_str().parse().ok())
                .collect()
        })
        .unwrap_or_default();

    g.synthetic = len_at(
        &json(root, "datasets/synthetic/catalog.json")?,
        &["datasets"],
    );
    g.test_datasets = len_at(&json(root, "datasets/manifest.json")?, &["members"]);
    g.python_files = git_ls(root, "*.py")?;
    g.python_under_python = git_ls(root, "python/*.py")?;
    g.adrs = git_ls(root, "docs/adr/ADR-*.md")?;
    Ok(g)
}

/// Rule 6: each README statement that prints one of the recomputed values.
pub fn check_readme(readme: &str, g: &Glance) -> Vec<String> {
    let mut problems = vec![];
    let main_crates = g.versions.get(&g.version).copied().unwrap_or(0);
    let expect_row = |row: &str, parts: Vec<String>, problems: &mut Vec<String>| {
        let Some(line) = readme.lines().find(|l| l.starts_with(row)) else {
            problems.push(format!("{README}: no row starting {row:?}"));
            return;
        };
        for part in parts {
            if !line.contains(&part) {
                problems.push(format!("{README}: {row:?} must say {part:?}"));
            }
        }
    };
    expect_row(
        "| What |",
        vec![format!("| {} |", g.version)],
        &mut problems,
    );
    expect_row(
        "| `jope` with no arguments |",
        vec![format!("prints `jope {}`", g.version)],
        &mut problems,
    );
    expect_row(
        "| Workspace |",
        vec![
            format!("{} members", g.members),
            format!("{main_crates} crates at {}", g.version),
            format!(
                "{} vendored SmartCluster crates at {}",
                g.vendored, g.vendored_version
            ),
            format!("{} are `default-members`", g.default_members),
        ],
        &mut problems,
    );
    expect_row(
        "| Emit targets |",
        vec![format!("| {}: ", g.emit_targets)],
        &mut problems,
    );
    expect_row(
        "| Rust tests |",
        vec![
            format!(
                "{} listed in {} packages",
                grouped(g.rust_tests),
                g.rust_packages
            ),
            format!("{} of them ignored", g.rust_ignored),
        ],
        &mut problems,
    );
    expect_row(
        "| Python tests |",
        vec![format!("| {}, ", grouped(g.python_tests))],
        &mut problems,
    );
    expect_row(
        "| CI |",
        vec![
            format!("{} jobs per pull request", g.ci_jobs),
            format!("{} `full-main-cpu` families", g.full_families),
        ],
        &mut problems,
    );
    expect_row(
        "| Coverage map |",
        vec![format!(
            "{} items over {} lanes",
            grouped(g.coverage_items),
            g.coverage_lanes
        )],
        &mut problems,
    );
    let m = |i: usize| g.matrix.get(i).copied().unwrap_or(0);
    expect_row(
        "| Capability matrix |",
        vec![format!(
            "{} rows: {} moved, {} new, {} pending, {} dropped, {} lost",
            m(0),
            m(1),
            m(2),
            m(3),
            m(4),
            m(5)
        )],
        &mut problems,
    );
    expect_row(
        "| Datasets in Git |",
        vec![
            format!("{} synthetic fixtures", g.synthetic),
            format!("{} test datasets", g.test_datasets),
        ],
        &mut problems,
    );
    expect_row(
        "| Python still tracked |",
        vec![format!(
            "{} `.py` files, {} of them under `python/`",
            grouped(g.python_files),
            g.python_under_python
        )],
        &mut problems,
    );
    expect_row(
        "| Architecture decision records |",
        vec![format!("| {} ", g.adrs)],
        &mut problems,
    );
    for prose in [
        format!("# jope {}", g.version),
        format!("source of v{};", g.version),
        format!(
            "CI is `.github/workflows/ci.yml`: {} jobs",
            words(g.ci_jobs)
        ),
        format!(
            "{} generated synthetic fixtures and {} tiny test datasets",
            words(g.synthetic),
            words(g.test_datasets)
        ),
        format!("The {} workspace members", g.members),
    ] {
        if !readme.contains(&prose) {
            problems.push(format!("{README}: must say {prose:?}"));
        }
    }
    problems
}
