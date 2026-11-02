use super::*;

#[test]
fn coverage_validate_matches_the_captured_python_outputs() {
    let cases = captured();
    let discovered = json("coverage/discovered.json");
    let discovered = discovered
        .get("items")
        .and_then(Value::as_dict)
        .unwrap()
        .clone();
    let base = text("coverage/mapping.json");
    let mut replayed = vec![("validate".to_owned(), base.clone())];
    replayed.extend(variants("coverage/variants.json", &base));
    assert_eq!(replayed.len(), 10);
    for (name, mapping) in replayed {
        let case = format!("cov-{name}");
        let (stdout, ok) =
            coverage::validate_text(&pyjson::parse_str(&mapping).unwrap(), &discovered).unwrap();
        check(&case, &cases[&case], ok, &stdout);
    }
}

#[test]
fn coverage_discover_matches_the_captured_python_output() {
    let cases = captured();
    let metadata = json("coverage/metadata.json");
    let files = text("coverage/files.txt");
    let workflows = json("coverage/workflows.json");
    let read = |name: &str| -> jope_records::Result<String> {
        Ok(s(workflows
            .get(name)
            .unwrap_or_else(|| panic!("no workflow text for {name}")))
        .to_owned())
    };
    let discovered =
        coverage::discover_from(&metadata, files.lines(), Path::new("@ROOT@"), &read).unwrap();
    let stdout = coverage::discover_document(discovered);
    check("cov-discover", &cases["cov-discover"], true, &stdout);
}

#[test]
fn rust_scope_check_matches_the_captured_python_output() {
    let cases = captured();
    let scope = json("scope/scope.json");
    let packages: BTreeSet<String> = scope
        .get("packages")
        .and_then(Value::as_list)
        .unwrap()
        .iter()
        .map(|p| s(p).to_owned())
        .collect();
    let (stdout, ok) = scope_check::evaluate(
        &packages,
        s(scope.get("product_stdout").unwrap()),
        s(scope.get("deferred_stdout").unwrap()),
        s(scope.get("script").unwrap()),
    )
    .unwrap();
    check("scope", &cases["scope"], ok, &stdout);
}

/// What `parity-check` read from the tree, as `capture_parity_tree` recorded it.
fn recorded_tree() -> (parity::Tree, BTreeMap<String, bool>) {
    let doc = json("parity/tree.json");
    let strings = |key: &str| -> Vec<String> {
        doc.get(key)
            .and_then(Value::as_list)
            .unwrap()
            .iter()
            .map(|v| s(v).to_owned())
            .collect()
    };
    let bins = doc
        .get("bins")
        .and_then(Value::as_list)
        .unwrap()
        .iter()
        .map(|b| {
            let b: Vec<String> = b
                .as_list()
                .unwrap()
                .iter()
                .map(|v| s(v).to_owned())
                .collect();
            (b[0].clone(), b[1].clone(), b[2].clone(), b[3].clone())
        })
        .collect();
    let jope_py = match doc.get("jope_py") {
        Some(Value::Null) | None => None,
        Some(_) => Some(strings("jope_py")),
    };
    let tree = parity::Tree {
        clap: strings("clap"),
        bins,
        tools: strings("tools"),
        jope_py,
        units: strings("units"),
    };
    let exists = doc
        .get("targets_exist")
        .and_then(Value::as_dict)
        .unwrap()
        .iter()
        .map(|(k, v)| (k.to_owned(), *v == Value::Bool(true)))
        .collect();
    (tree, exists)
}

#[test]
fn parity_check_matches_the_captured_python_outputs() {
    let cases = captured();
    let (tree, exists) = recorded_tree();
    let exists = |head: &str| -> bool {
        *exists
            .get(head)
            .unwrap_or_else(|| panic!("tree.json does not record whether {head} existed"))
    };
    let base = text("parity/capability-matrix.md");
    let help_doc = json("parity/help.json");
    let help = s(help_doc.get("text").unwrap()).to_owned();
    assert_eq!(
        py::sha256_hex(help.as_bytes()),
        s(help_doc.get("sha256").unwrap()),
        "parity/help.json: not the help text the Python read"
    );
    let mut replayed = vec![
        ("parity".to_owned(), base.clone(), None),
        ("parity-help".to_owned(), base.clone(), Some(help.as_str())),
    ];
    for (name, matrix) in variants("parity/variants.json", &base) {
        replayed.push((format!("parity-{name}"), matrix, None));
    }
    assert_eq!(replayed.len(), 10);
    for (name, matrix, help) in replayed {
        let (stdout, ok) = parity::evaluate(&matrix, &tree, help, &exists).unwrap();
        let mut want = cases[&name].clone();
        if name == "parity-preamble" {
            // Allowed difference: the remediation hint names the Rust source, not the census.
            let python = "(re-run `deploy/ci/parity-census.py census`, which emits it)";
            let rust = "(restore it from COUNTS_PREAMBLE in crates/jope-records/src/parity.rs)";
            assert_eq!(want.1.matches(python).count(), 1, "{name}: captured hint");
            want.1 = want.1.replace(python, rust);
        }
        check(&name, &want, ok, &stdout);
    }
}

pub(super) fn copy_tree(from: &Path, to: &Path) {
    fs::create_dir_all(to).unwrap();
    for entry in fs::read_dir(from).unwrap() {
        let entry = entry.unwrap();
        let target = to.join(entry.file_name());
        if entry.file_type().unwrap().is_dir() {
            copy_tree(&entry.path(), &target);
        } else {
            fs::copy(entry.path(), &target).unwrap();
        }
    }
}

#[test]
fn readme_benchmarks_matches_the_captured_python_outputs() {
    let cases = captured();
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path().join("root");
    copy_tree(&fixture().join("readme/root"), &root);
    let readme_path = root.join("README.md");
    let stale = text("readme/README.stale.md");
    let written = text("readme/README.written.md");
    // --check on the stale README: exit 1, the out-of-date notice on stderr.
    fs::write(&readme_path, &stale).unwrap();
    let (message, ok) = readme::run(&root, false).unwrap();
    let want = &cases["readme-stale"];
    assert_eq!((want.0, ok), (1, false), "readme-stale: exit status");
    assert_eq!(want.1, "", "readme-stale: stdout");
    // Allowed difference: the remediation hint names the Rust command, not the retired script.
    let python = "Run: python3 deploy/ci/readme-benchmarks.py --write\n";
    let rust = "Run: cargo run -q --locked -p jope-records -- readme-benchmarks --write\n";
    assert_eq!(
        want.2.matches(python).count(),
        1,
        "readme-stale: captured hint"
    );
    assert_eq!(
        message,
        want.2.replace(python, rust),
        "readme-stale: notice"
    );
    // --write from the stale README reproduces what the Python wrote, byte for byte.
    let (message, ok) = readme::run(&root, true).unwrap();
    check("readme-write", &cases["readme-write"], ok, &message);
    assert_eq!(fs::read_to_string(&readme_path).unwrap(), written);
    // --check on the result.
    let (message, ok) = readme::run(&root, false).unwrap();
    check("readme", &cases["readme"], ok, &message);
}

/// Record what `parity-check` reads from a tree, for the replay above. Run it on the export of the
/// differential run's tree:
///
/// ```sh
/// JOPE_PARITY_PIN_ROOT=<export> cargo test -p jope-records --test python_parity -- --ignored capture_parity_tree
/// ```
#[test]
#[ignore = "fixture capture: needs JOPE_PARITY_PIN_ROOT"]
fn capture_parity_tree() {
    let root = PathBuf::from(std::env::var("JOPE_PARITY_PIN_ROOT").unwrap())
        .canonicalize()
        .unwrap();
    let tree = parity::scan_tree(&root).unwrap();
    let base = text("parity/capability-matrix.md");
    let mut heads = BTreeSet::new();
    let mut matrices = vec![base.clone()];
    matrices.extend(
        variants("parity/variants.json", &base)
            .into_iter()
            .map(|(_, m)| m),
    );
    for matrix in &matrices {
        heads.extend(parity::target_heads(matrix));
    }
    let strings =
        |items: &[String]| Value::List(items.iter().map(|i| Value::from(i.as_str())).collect());
    let mut exists = Dict::new();
    for head in heads {
        let present = root.join(&head).exists();
        exists.insert(head, Value::Bool(present));
    }
    let doc: Dict = [
        ("clap", strings(&tree.clap)),
        (
            "bins",
            Value::List(
                tree.bins
                    .iter()
                    .map(|(a, b, c, d)| strings(&[a.clone(), b.clone(), c.clone(), d.clone()]))
                    .collect(),
            ),
        ),
        ("tools", strings(&tree.tools)),
        (
            "jope_py",
            tree.jope_py.as_deref().map_or(Value::Null, strings),
        ),
        ("units", strings(&tree.units)),
        ("targets_exist", Value::Dict(exists)),
    ]
    .into_iter()
    .collect();
    fs::write(
        fixture().join("parity/tree.json"),
        pyjson::dumps(&Value::Dict(doc), false, Some(1)) + "\n",
    )
    .unwrap();
}

// ------------------------------------------------------------------------------------------------
// Malformed inputs (see the module documentation).

/// Malformed inputs the Python accepted (exit status 0) and the port refuses, by fixture and case.
/// Being stricter than the Python is allowed; being more permissive never is.
const RUST_STRICTER: &[(&str, &str, &str)] = &[
    (
        "coverage/negative-validate.json",
        "scope-bigint",
        "the port's JSON reader refuses integers beyond 64 bits",
    ),
    (
        "coverage/negative-discover.json",
        "feature-deps-int",
        "a feature dependency that is not a string is refused rather than sorted",
    ),
    (
        "coverage/negative-discover.json",
        "required-int-member",
        "a required feature that is not a string is refused rather than sorted",
    ),
    (
        "parity/negative.json",
        "package-name-int",
        "a package name that is not a string is refused rather than sorted and printed",
    ),
    (
        "parity/negative.json",
        "target-name-int",
        "a target name that is not a string is refused rather than sorted and printed",
    ),
    (
        "readme/negative.json",
        "valid-rankable-float",
        "`rankable_rows` must be an integer; the Python printed `format(8718.0, ',')`",
    ),
    // Round four: the strict number grammar (`py::decimal`, `py::float`).
    (
        "readme/negative-r2.json",
        "csv-average-rank-nan",
        "`float()` of `nan` is outside the strict number grammar (no committed ladder writes it)",
    ),
    (
        "readme/negative-r3.json",
        "csv-int-underscores-4300-digits",
        "`int()` of digits with `_` separators is outside the strict number grammar",
    ),
    (
        "readme/negative-r4.json",
        "published-scored-nbsp",
        "U+00A0 around an integer is outside the strict number grammar (ASCII whitespace only)",
    ),
    (
        "readme/negative-r4.json",
        "published-scored-em-space",
        "U+2003 around an integer is outside the strict number grammar (ASCII whitespace only)",
    ),
    (
        "readme/negative-r4.json",
        "published-scored-ideographic-space",
        "U+3000 around an integer is outside the strict number grammar (ASCII whitespace only)",
    ),
    (
        "readme/negative-r4.json",
        "published-scored-underscore",
        "`int()` of digits with a `_` separator is outside the strict number grammar",
    ),
    (
        "readme/negative-r4.json",
        "published-scored-arabic-indic",
        "`int()` of Arabic-Indic digits is outside the strict number grammar (ASCII digits only)",
    ),
    (
        "readme/negative-r4.json",
        "published-scored-fullwidth",
        "`int()` of fullwidth digits is outside the strict number grammar (ASCII digits only)",
    ),
    (
        "readme/negative-r4.json",
        "samecut-rank-nbsp",
        "U+00A0 around an integer is outside the strict number grammar (ASCII whitespace only)",
    ),
    (
        "readme/negative-r4.json",
        "valid-average-nbsp",
        "U+00A0 around a float is outside the strict number grammar (ASCII whitespace only)",
    ),
    (
        "readme/negative-r4.json",
        "valid-average-em-space",
        "U+2003 around a float is outside the strict number grammar (ASCII whitespace only)",
    ),
    (
        "readme/negative-r4.json",
        "valid-average-underscore",
        "`float()` of digits with a `_` separator is outside the strict number grammar",
    ),
    (
        "readme/negative-r4.json",
        "valid-average-arabic-indic",
        "`float()` of Arabic-Indic digits is outside the strict number grammar (ASCII digits only)",
    ),
    (
        "readme/negative-r4.json",
        "valid-average-nan",
        "`float()` of `nan` is outside the strict number grammar (no committed ladder writes it)",
    ),
    (
        "readme/negative-r4.json",
        "valid-average-inf",
        "`float()` of `inf` is outside the strict number grammar (no committed ladder writes it)",
    ),
    (
        "readme/negative-r4.json",
        "valid-average-minus-infinity",
        "`float()` of `-Infinity` is outside the strict number grammar (no committed ladder writes it)",
    ),
];

/// README cases where both refuse with an explicit message and the port's words differ (round
/// five): the Python counted the reports it globbed or found none, the port refuses the entry
/// itself.
const README_MESSAGE_DIFFERS: &[(&str, &str, &str)] = &[
    (
        "readme/negative-r5.json",
        "readme-r5-report-name-not-utf8",
        "the Python read the report and refused the count; the port refuses the undecodable name",
    ),
    (
        "readme/negative-r5.json",
        "readme-r5-reports-unlistable",
        "the Python's glob listed nothing (\"no reports\"); the port refuses to list the directory",
    ),
];

/// Cases where the Python refused an integer of more than 4300 digits and the port refuses the
/// text earlier, because it is outside the strict number grammar: the refusal is not required to
/// carry CPython's message about the limit.
const REFUSED_BEFORE_THE_LIMIT: &[(&str, &str, &str)] = &[(
    "readme/negative-r3.json",
    "csv-int-underscores-4301-digits",
    "`_` separators are outside the strict number grammar, which refuses the text before it counts digits",
)];
