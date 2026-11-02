use super::matrix::counts_row_kind;
use super::*;

/// What the check reads from the tree besides the matrix.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Tree {
    /// `jope-parallel` subcommands: the `Cmd` variants of `crates/jope-engine/src/main.rs`.
    pub clap: Vec<String>,
    /// `(package, kind, name, src path)` of every bin/example target.
    pub bins: Vec<(String, String, String, String)>,
    /// Python tools under `python/src/jope/{tools,research}`.
    pub tools: Vec<String>,
    /// `jope-py` commands, when `python/src/jope/cli.py` exists.
    pub jope_py: Option<Vec<String>>,
    /// systemd units under `deploy/systemd`.
    pub units: Vec<String>,
}

/// Read the tree's entry points (`root` canonical).
pub fn scan_tree(root: &Path) -> Result<Tree> {
    let clap = clap_names(root)?;
    scan_rest(root, clap)
}

/// The `jope-parallel` subcommands of `crates/jope-engine/src/main.rs`, when it exists
/// (`main_rs.exists()`, which raised on an error other than a missing path: that refuses).
fn clap_names(root: &Path) -> Result<Vec<String>> {
    let main_rs = root.join("crates/jope-engine/src/main.rs");
    Ok(if exists(&main_rs, "clap subcommands")? {
        parse_clap_enum(&main_rs, "Cmd")?
            .into_iter()
            .map(|c| c.name)
            .collect()
    } else {
        vec![]
    })
}

/// The entry points read after the clap subcommands (and, in `check`, after the `--help` text):
/// binaries and examples, Python tools, `jope-py` commands, systemd units, in the Python's order.
fn scan_rest(root: &Path, clap: Vec<String>) -> Result<Tree> {
    let bins = cargo_bins(root)?;
    let tools = python_tools(root)?;
    let cli = root.join("python/src/jope/cli.py");
    let jope_py = if exists(&cli, "jope-py commands")? {
        Some(parse_argparse_commands(&cli)?)
    } else {
        None
    };
    let units = systemd_units(root)?;
    Ok(Tree {
        clap,
        bins,
        tools,
        jope_py,
        units,
    })
}

/// The row targets whose existence `evaluate` asks about, in the order the Python's row loop
/// asked (`moved`/`new` rows whose head names a path).
pub fn checked_heads(matrix_text: &str) -> Vec<String> {
    let mut heads = vec![];
    for r in parse_matrix(matrix_text) {
        if !matches!(status_class(&r.status), "moved" | "new") {
            continue;
        }
        let head = head_of(&r.new);
        if !head.is_empty()
            && !matches!(head, "dropped" | "lost" | "pending" | "-" | "none")
            && head.contains('/')
        {
            heads.push(head.to_owned());
        }
    }
    heads
}

/// Every row target that names a path (`evaluate` asks whether those exist).
pub fn target_heads(matrix_text: &str) -> BTreeSet<String> {
    parse_matrix(matrix_text)
        .iter()
        .map(|r| head_of(&r.new).to_owned())
        .filter(|h| h.contains('/'))
        .collect()
}

/// The check report and whether the matrix passes.
///
/// The inputs are read in `cmd_check`'s order, so the first failure is the one the Python
/// stopped at: the matrix, the row targets' existence (`Path.exists`, which raised on an error
/// other than a missing path), the clap enum, the `--help` text, then the rest of the tree.
pub fn check(matrix: &Path, root: &Path, help_text: Option<&Path>) -> Result<(String, bool)> {
    let root = root.canonicalize().map_err(|e| Error::io(root, e))?;
    // Every entry point is named relative to the resolved root and compared with the matrix's
    // text. A root that is not UTF-8 refuses here, before anything is read (stricter: the Python
    // named entry points relative to its surrogate-escaped root and ran the check), rather than be
    // compared lossily.
    if root.to_str().is_none() {
        return Err(Error::Shape(format!(
            "parity-check root {} is not UTF-8; its entry points cannot be named as text",
            py::os_repr(root.as_os_str())
        )));
    }
    let text = py::read_text(matrix)?;
    let mut exists = BTreeMap::new();
    for head in checked_heads(&text) {
        let present = py::path_exists(&root.join(&head)).map_err(|e| {
            Error::Shape(format!(
                "row target `{head}`: {}",
                py::os_error_path(&e, &root.join(&head))
            ))
        })?;
        exists.insert(head, present);
    }
    let clap = clap_names(&root)?;
    let help = help_text.map(py::read_text).transpose()?;
    let tree = scan_rest(&root, clap)?;
    evaluate(&text, &tree, help.as_deref(), &|head: &str| exists[head])
}

/// The check report for a matrix text against a tree, and whether the matrix passes. `exists`
/// answers whether a row target path exists. `check` gathers the inputs; the Python-parity
/// fixtures replay recorded ones. Refuses where the Python's `int()` raised (a `pending-WP<n>`
/// status or a covering-test tally of more than 4300 digits) and where the strict number grammar
/// does not read what it read (such a number in other Unicode digits).
pub fn evaluate(
    matrix_text: &str,
    tree: &Tree,
    help_text: Option<&str>,
    exists: &dyn Fn(&str) -> bool,
) -> Result<(String, bool)> {
    let rows = parse_matrix(matrix_text);
    let mut problems: Vec<String> = vec![];
    let mut targets: BTreeSet<String> = BTreeSet::new();
    let mut parens: BTreeSet<String> = BTreeSet::new();
    let paren = Regex::new(r"\(([^)]*)\)").expect("static pattern");
    let archive_location = ARCHIVE.split("/git/").next().unwrap_or(ARCHIVE);
    for (i, r) in rows.iter().enumerate() {
        let place = format!("row {} [{} | {}]", i + 1, r.kind, r.old);
        let class = status_class(&r.status);
        if class == "invalid" {
            problems.push(format!("{place}: invalid status `{}`", r.status));
            continue;
        }
        let head = head_of(&r.new);
        for m in paren.captures_iter(&r.new) {
            parens.insert(py::strip(&m[1]).to_owned());
        }
        if !head.is_empty() {
            targets.insert(head.to_owned());
        }
        match class {
            "pending" => {
                let n = pending_int(&r.status, &place)?;
                if !n.parse::<u64>().is_ok_and(|n| (4..=12).contains(&n)) {
                    problems.push(format!("{place}: pending-WP{n} is outside WP4..WP12"));
                }
            }
            "dropped" => {
                if !r.status.contains(archive_location) {
                    problems.push(format!("{place}: dropped without the archive location"));
                }
                if r.status.chars().count() < "dropped: x (archive: )".len() + 20 {
                    problems.push(format!("{place}: dropped without a reason"));
                }
                if !matches!(r.new.as_str(), "dropped" | "-" | "") {
                    problems.push(format!(
                        "{place}: dropped row still names a target `{}`",
                        r.new
                    ));
                }
            }
            "lost" => {
                if r.new != "lost" {
                    problems.push(format!("{place}: lost row must have new home `lost`"));
                }
            }
            _ => {
                if head.is_empty() || matches!(head, "dropped" | "lost" | "pending" | "-" | "none")
                {
                    problems.push(format!("{place}: status `{}` but no target", r.status));
                } else if head.contains('/') && !exists(head) {
                    problems.push(format!("{place}: target `{head}` does not exist"));
                }
            }
        }
    }
    let mut missing: Vec<String> = vec![];
    for name in &tree.clap {
        if !parens.contains(&format!("jope-parallel {name}")) {
            missing.push(format!("clap subcommand: jope-parallel {name}"));
        }
    }
    if let Some(help) = help_text {
        for name in parse_help_commands(help) {
            if !parens.contains(&format!("jope-parallel {name}")) {
                missing.push(format!(
                    "clap subcommand (from --help): jope-parallel {name}"
                ));
            }
        }
    }
    for (pkg, kind, name, src) in &tree.bins {
        if !targets.contains(src) {
            missing.push(format!("{kind} {pkg}/{name}: {src}"));
        }
    }
    for tool in &tree.tools {
        if !targets.contains(tool) {
            missing.push(format!("python tool: {tool}"));
        }
    }
    for command in tree.jope_py.iter().flatten() {
        if !parens.contains(&format!("jope-py {command}")) {
            missing.push(format!("jope-py command: {command}"));
        }
    }
    for unit in &tree.units {
        if !targets.contains(unit) {
            missing.push(format!("systemd unit: {unit}"));
        }
    }
    for m in missing {
        problems.push(format!("no row for {m}"));
    }
    let (want_table, want_no_test) = counts_lines(&rows);
    let (got_table, got_no_test, got_prose) = committed_counts(matrix_text)?;
    for line in COUNTS_PREAMBLE {
        if !got_prose.iter().any(|p| p == line) {
            problems.push(format!(
                "## Counts is missing the line documenting why it is gated: `{line}` \
                 (restore it from COUNTS_PREAMBLE in crates/jope-records/src/parity.rs)"
            ));
        }
    }
    if got_table.is_empty() {
        problems.push("the matrix has no `## Counts` table".to_owned());
    } else if got_table != want_table {
        let wanted: BTreeMap<String, &String> =
            want_table.iter().map(|l| (counts_row_kind(l), l)).collect();
        let gotten: BTreeMap<String, &String> =
            got_table.iter().map(|l| (counts_row_kind(l), l)).collect();
        let kinds: BTreeSet<&String> = wanted.keys().chain(gotten.keys()).collect();
        for kind in kinds {
            if wanted.get(kind) != gotten.get(kind) {
                problems.push(format!(
                    "## Counts row `{kind}` disagrees with the rows below it: committed `{}`, recomputed `{}`",
                    gotten.get(kind).map_or("(no such row)", |l| l.as_str()),
                    wanted.get(kind).map_or("(no such kind)", |l| l.as_str()),
                ));
            }
        }
    }
    match got_no_test {
        None => problems
            .push("## Counts has no `moved`/`new` rows without a covering test: <n>. line".into()),
        Some(n) if n != want_no_test.to_string() => problems.push(format!(
            "## Counts says {n} `moved`/`new` rows without a covering test; the rows say {want_no_test}"
        )),
        Some(_) => {}
    }
    let mut by_status: BTreeMap<&str, usize> = BTreeMap::new();
    let mut by_kind: BTreeMap<&str, usize> = BTreeMap::new();
    for r in &rows {
        *by_status.entry(status_class(&r.status)).or_default() += 1;
        *by_kind.entry(&r.kind).or_default() += 1;
    }
    let listing = |m: &BTreeMap<&str, usize>| {
        m.iter()
            .map(|(k, v)| format!("{k}={v}"))
            .collect::<Vec<_>>()
            .join(", ")
    };
    let mut out = format!("capability matrix: {} rows\n", rows.len());
    out.push_str(&format!("by status: {}\n", listing(&by_status)));
    out.push_str(&format!("by kind: {}\n", listing(&by_kind)));
    out.push_str(&format!(
        "this tree: {} binaries, {} examples, {} clap subcommands, {} python tools, {} systemd units\n",
        tree.bins.iter().filter(|b| b.1 == "bin").count(),
        tree.bins.iter().filter(|b| b.1 == "example").count(),
        tree.clap.len(),
        tree.tools.len(),
        tree.units.len()
    ));
    if !problems.is_empty() {
        out.push_str(&format!("\n{} problem(s):\n", problems.len()));
        for p in &problems {
            out.push_str(&format!("  - {p}\n"));
        }
        return Ok((out, false));
    }
    out.push_str("ok: every row resolved and every entry point of this tree has a row\n");
    Ok((out, true))
}
