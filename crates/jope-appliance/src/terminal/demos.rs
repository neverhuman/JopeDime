//! Demo datasets bundled into the executable, opened as virtual `example/<name>.csv` files.
//!
//! The CSVs are derived deterministically from `test-support/datasets/v1` (see [`SOURCES`]):
//! every k-th row of a fixture's `train.csv` under its catalog column headers. The tests
//! re-derive every byte from the fixtures; `assets/demos/manifest.json` records the SHA-256,
//! row selection and licence of each. Together they stay below [`MAX_TOTAL_BYTES`].

/// The virtual directory that bundled demos appear under.
pub const PREFIX: &str = "example/";
/// Budget for every embedded demo together.
pub const MAX_TOTAL_BYTES: usize = 1_000_000;

#[derive(Debug)]
pub struct Demo {
    /// Short name used by `/demo NAME` and the file name stem.
    pub name: &'static str,
    pub title: &'static str,
    /// `regression`, `binary` or `multiclass`: the `--task` preset.
    pub task: &'static str,
    /// The target column preset.
    pub target: &'static str,
    /// An objective preset, or `None` for the engine's default for the task.
    pub objective: Option<&'static str>,
    /// Licence of the data, as recorded in assets/demos/manifest.json: the SPDX id, where the
    /// licence was found, and the attribution it requires. Printed when the demo is selected
    /// ([`Demo::license_notice`]).
    pub license: &'static str,
    pub license_url: &'static str,
    pub credit: &'static str,
    pub bytes: &'static [u8],
}

impl Demo {
    /// The virtual path, `example/<name>.csv`.
    pub fn path(&self) -> String {
        format!("{PREFIX}{}.csv", self.name)
    }

    /// Printed when a user selects this demo (`/demo NAME` or `/open example/NAME.csv`).
    pub fn license_notice(&self) -> Vec<String> {
        vec![
            format!("Data licence: {} ({})", self.license, self.license_url),
            format!(
                "Credit: {}. Values are a min-max scaled subset of the original.",
                self.credit
            ),
        ]
    }

    pub fn rows(&self) -> usize {
        self.bytes
            .iter()
            .filter(|b| **b == b'\n')
            .count()
            .saturating_sub(1)
    }
}

pub static DEMOS: &[Demo] = &[
    Demo {
        name: "regression_parkinsons",
        title: "Parkinsons telemonitoring: predict the UPDRS score",
        task: "regression",
        target: "target",
        objective: Some("rmse"),
        license: "CC-BY-4.0",
        license_url: "https://archive.ics.uci.edu/dataset/189/parkinsons+telemonitoring",
        credit: "Tsanas & Little (2009), UCI ML Repository, doi:10.24432/C5ZS3N",
        bytes: include_bytes!("../../assets/demos/regression_parkinsons.csv"),
    },
    Demo {
        name: "regression_apartment_rent",
        title: "US apartment listings: predict monthly rent",
        task: "regression",
        target: "price",
        objective: Some("rmse"),
        license: "CC-BY-4.0",
        license_url: "https://archive.ics.uci.edu/dataset/555/apartment+for+rent+classified",
        credit: "Apartment for Rent Classified (2019), UCI ML Repository, doi:10.24432/C5X623",
        bytes: include_bytes!("../../assets/demos/regression_apartment_rent.csv"),
    },
];

/// How each demo is derived: (name, fixture under `test-support/datasets/v1`, keep every k-th
/// training row).
pub const SOURCES: [(&str, &str, usize); 2] = [
    ("regression_parkinsons", "d2-regression", 1),
    ("regression_apartment_rent", "grouped-apartment-rent", 5),
];

/// Find a demo by exact name, by task (`regression`, `binary`: the first of that task), or
/// by a unique name prefix.
pub fn find(name: &str) -> Option<&'static Demo> {
    let name = name.trim().trim_end_matches(".csv");
    let name = name.strip_prefix(PREFIX).unwrap_or(name);
    if let Some(demo) = DEMOS.iter().find(|demo| demo.name == name) {
        return Some(demo);
    }
    if let Some(demo) = DEMOS.iter().find(|demo| demo.task == name) {
        return Some(demo);
    }
    let mut matches = DEMOS
        .iter()
        .filter(|demo| !name.is_empty() && demo.name.starts_with(name));
    match (matches.next(), matches.next()) {
        (Some(demo), None) => Some(demo),
        _ => None,
    }
}

/// The bundled demo for a path such as `example/regression_parkinsons.csv` (also accepting
/// `./example/...` and Windows separators). Only exact file names match.
pub fn from_path(path: &str) -> Option<&'static Demo> {
    let normalized = path.replace('\\', "/");
    let mut rest = normalized.as_str();
    while let Some(stripped) = rest.strip_prefix("./") {
        rest = stripped;
    }
    let file = rest.strip_prefix(PREFIX)?;
    DEMOS
        .iter()
        .find(|demo| file == format!("{}.csv", demo.name))
}

/// A command-line path to read: the file itself when it exists, otherwise a bundled demo's
/// `example/<name>.csv` written once into `<JOPEDIME_HOME>/examples/`, so that `jain run`,
/// `jain fit` and agents reach the same demos as the TUI without a terminal (#2142). Any other
/// path is returned unchanged (and fails later as a missing file).
pub fn resolve(path: &std::path::Path) -> std::path::PathBuf {
    resolve_in(path, &jope_paths::jopedime_home())
}

pub fn resolve_in(path: &std::path::Path, home: &std::path::Path) -> std::path::PathBuf {
    if path.exists() {
        return path.to_owned();
    }
    let Some(demo) = path.to_str().and_then(from_path) else {
        return path.to_owned();
    };
    materialize(demo, home).unwrap_or_else(|_| path.to_owned())
}

/// Write `demo` to `<home>/examples/<name>.csv` unless an identical copy is already there.
pub fn materialize(demo: &Demo, home: &std::path::Path) -> std::io::Result<std::path::PathBuf> {
    let directory = home.join("examples");
    std::fs::create_dir_all(&directory)?;
    let file = directory.join(format!("{}.csv", demo.name));
    if std::fs::read(&file).ok().as_deref() != Some(demo.bytes) {
        let temporary = directory.join(format!(".{}.csv.{}.tmp", demo.name, std::process::id()));
        std::fs::write(&temporary, demo.bytes)?;
        std::fs::rename(&temporary, &file)?;
    }
    Ok(file)
}

/// One line per demo, for `/demo` without a name.
pub fn listing() -> Vec<String> {
    let mut lines = vec!["Bundled demos (no account needed). Run one with /demo NAME:".to_owned()];
    for demo in DEMOS {
        lines.push(format!(
            "  {:<26} {:<10} {:>5} rows · {}",
            demo.name,
            demo.task,
            demo.rows(),
            demo.title
        ));
    }
    lines.push("Or open one to set the target yourself: /open example/<name>.csv".into());
    lines.push("Each demo's data licence and credit are shown when you select it.".into());
    lines
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn example_paths_resolve_to_a_written_copy_and_real_files_win() {
        let home = tempfile::tempdir().unwrap();
        let resolved = resolve_in(
            std::path::Path::new("example/regression_parkinsons.csv"),
            home.path(),
        );
        assert_eq!(
            resolved,
            home.path().join("examples/regression_parkinsons.csv")
        );
        assert_eq!(std::fs::read(&resolved).unwrap(), DEMOS[0].bytes);
        // Resolving again reuses the identical copy.
        assert_eq!(
            resolve_in(
                std::path::Path::new("./example/regression_parkinsons.csv"),
                home.path()
            ),
            resolved
        );
        // An existing file and an unknown example are returned unchanged.
        let real = home.path().join("examples");
        assert_eq!(resolve_in(&real, home.path()), real);
        let unknown = std::path::Path::new("example/nope.csv");
        assert_eq!(resolve_in(unknown, home.path()), unknown);
    }

    #[test]
    fn demos_fit_the_budget_and_presets_name_real_header_columns() {
        let total: usize = DEMOS.iter().map(|demo| demo.bytes.len()).sum();
        assert!(total < MAX_TOTAL_BYTES, "{total} bytes");
        let manifest: serde_json::Value =
            serde_json::from_str(include_str!("../../assets/demos/manifest.json")).unwrap();
        assert_eq!(manifest["total_bytes"].as_u64(), Some(total as u64));
        for demo in DEMOS {
            let text = std::str::from_utf8(demo.bytes).unwrap();
            let header: Vec<&str> = text.lines().next().unwrap().split(',').collect();
            assert!(header.contains(&demo.target), "{}", demo.name);
            assert!(demo.rows() >= 25, "{}", demo.name);
            assert!(text
                .lines()
                .all(|line| line.split(',').count() == header.len()));
            let entry = manifest["demos"]
                .as_array()
                .unwrap()
                .iter()
                .find(|entry| entry["file"] == demo.path())
                .unwrap_or_else(|| panic!("{} missing from manifest", demo.name));
            assert_eq!(entry["task"], demo.task);
            assert_eq!(entry["license"]["spdx"], demo.license);
            assert_eq!(entry["license"]["evidence_url"], demo.license_url);
            assert_eq!(entry["target"], demo.target);
            assert_eq!(entry["rows"].as_u64(), Some(demo.rows() as u64));
            assert_eq!(
                entry["sha256"].as_str().unwrap(),
                jope_runtime::sha256_hex(demo.bytes)
            );
            assert!(matches!(demo.task, "regression" | "binary" | "multiclass"));
            assert!(demo.name.starts_with(demo.task));
        }
    }

    /// Regenerate every bundled CSV from its committed fixture and require identical bytes.
    #[test]
    fn bundled_csvs_are_exactly_derived_from_the_reference_fixtures() {
        let root =
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../test-support/datasets/v1");
        assert_eq!(SOURCES.len(), DEMOS.len());
        for (name, fixture, step) in SOURCES {
            let demo = DEMOS.iter().find(|demo| demo.name == name).unwrap();
            let meta: serde_json::Value = serde_json::from_slice(
                &std::fs::read(root.join(fixture).join("meta.json")).unwrap(),
            )
            .unwrap();
            let header: Vec<&str> = meta["output_headers"]
                .as_array()
                .unwrap()
                .iter()
                .map(|value| value.as_str().unwrap())
                .collect();
            let train = std::fs::read_to_string(root.join(fixture).join("train.csv")).unwrap();
            let mut expected = header.join(",");
            for row in train.lines().step_by(step) {
                expected.push('\n');
                expected.push_str(row);
            }
            expected.push('\n');
            assert_eq!(meta["target_column"], demo.target, "{name}");
            assert!(
                expected.as_bytes() == demo.bytes,
                "{name} differs from {fixture}"
            );
        }
    }

    #[test]
    fn find_accepts_names_tasks_prefixes_and_paths() {
        assert_eq!(find("regression").unwrap().name, "regression_parkinsons");
        assert_eq!(
            find("regression_apartment_rent").unwrap().name,
            "regression_apartment_rent"
        );
        assert_eq!(
            find("regression_apart").unwrap().name,
            "regression_apartment_rent"
        );
        assert_eq!(
            find("example/regression_apartment_rent.csv").unwrap().name,
            "regression_apartment_rent"
        );
        assert!(find("regression_").is_none(), "ambiguous prefix");
        assert!(find("").is_none());
        // Only CC-BY-4.0 demos ship; there is no binary or multiclass demo.
        assert!(find("binary").is_none());
        assert!(find("multiclass").is_none());
    }

    /// Every bundled demo carries a licence accepted for redistribution inside the public
    /// binary (docs/public-readiness-checklist.md); copyleft and share-alike data stay out.
    #[test]
    fn only_attribution_licensed_demos_are_bundled() {
        for demo in DEMOS {
            assert_eq!(demo.license, "CC-BY-4.0", "{}", demo.name);
        }
    }

    #[test]
    fn only_exact_example_paths_are_virtual_files() {
        assert_eq!(
            from_path("example/regression_parkinsons.csv").unwrap().name,
            "regression_parkinsons"
        );
        assert!(from_path("./example/regression_parkinsons.csv").is_some());
        assert!(from_path("example\\regression_parkinsons.csv").is_some());
        assert!(from_path("example/regression_parkinsons").is_none());
        assert!(from_path("other/regression_parkinsons.csv").is_none());
        assert!(from_path("example/../secret.csv").is_none());
        assert!(from_path("/example/regression_parkinsons.csv").is_none());
    }

    #[test]
    fn selecting_a_demo_prints_its_licence_and_credit() {
        for demo in DEMOS {
            let notice = demo.license_notice().join("\n");
            assert!(notice.contains(demo.license), "{notice}");
            assert!(notice.contains(demo.license_url), "{notice}");
            assert!(notice.contains(demo.credit), "{notice}");
            assert!(demo.license_url.starts_with("https://"), "{}", demo.name);
        }
    }

    #[test]
    fn listing_names_every_demo() {
        let lines = listing();
        for demo in DEMOS {
            assert!(lines.iter().any(|line| line.contains(demo.name)));
        }
    }
}
