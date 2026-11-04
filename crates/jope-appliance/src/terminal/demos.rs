//! Demo datasets bundled into the executable, opened as virtual `example/<name>.csv` files.
//!
//! The CSVs are generated deterministically from `test-support/datasets/v1` by
//! `assets/demos/generate.py`, which records their row selection and SHA-256 receipts in
//! `assets/demos/manifest.json`. Together they stay below [`MAX_TOTAL_BYTES`].

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
    /// SPDX licence and short credit, shown by `/demo` (see assets/demos/manifest.json).
    pub license: &'static str,
    pub credit: &'static str,
    pub bytes: &'static [u8],
}

impl Demo {
    /// The virtual path, `example/<name>.csv`.
    pub fn path(&self) -> String {
        format!("{PREFIX}{}.csv", self.name)
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
        credit: "Tsanas & Little (2009), UCI ML Repository, doi:10.24432/C5ZS3N",
        bytes: include_bytes!("../../assets/demos/regression_parkinsons.csv"),
    },
    Demo {
        name: "binary_loan_approval",
        title: "Mortgage applications: predict approval",
        task: "binary",
        target: "approve",
        objective: Some("logloss"),
        license: "GPL-3.0-only",
        credit: "Hunter & Walker (1996) via the R package wooldridge (GPL-3)",
        bytes: include_bytes!("../../assets/demos/binary_loan_approval.csv"),
    },
    Demo {
        name: "regression_apartment_rent",
        title: "US apartment listings: predict monthly rent",
        task: "regression",
        target: "price",
        objective: Some("rmse"),
        license: "CC-BY-4.0",
        credit: "Apartment for Rent Classified (2019), UCI ML Repository, doi:10.24432/C5X623",
        bytes: include_bytes!("../../assets/demos/regression_apartment_rent.csv"),
    },
    Demo {
        name: "regression_tiny_soil",
        title: "30-row soil survey table: a few-second smoke test",
        task: "regression",
        target: "target",
        objective: Some("rmse"),
        license: "CC-BY-SA-4.0",
        credit: "LimeSoDa BB.30_2, Schmidinger et al. (2025)",
        bytes: include_bytes!("../../assets/demos/regression_tiny_soil.csv"),
    },
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

/// The bundled demo for a path such as `example/binary_loan_approval.csv` (also accepting
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
    lines.push("Sources (values are min-max scaled subsets):".into());
    for demo in DEMOS {
        lines.push(format!(
            "  {}: {} · {}",
            demo.name, demo.credit, demo.license
        ));
    }
    lines
}

#[cfg(test)]
mod tests {
    use super::*;

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

    #[test]
    fn find_accepts_names_tasks_prefixes_and_paths() {
        assert_eq!(find("regression").unwrap().name, "regression_parkinsons");
        assert_eq!(find("binary").unwrap().name, "binary_loan_approval");
        assert_eq!(
            find("regression_tiny_soil").unwrap().name,
            "regression_tiny_soil"
        );
        assert_eq!(
            find("regression_apart").unwrap().name,
            "regression_apartment_rent"
        );
        assert_eq!(
            find("example/binary_loan_approval.csv").unwrap().name,
            "binary_loan_approval"
        );
        assert!(find("regression_").is_none(), "ambiguous prefix");
        assert!(find("").is_none());
        assert!(find("multiclass").is_none());
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
    fn listing_names_every_demo() {
        let lines = listing();
        for demo in DEMOS {
            assert!(lines.iter().any(|line| line.contains(demo.name)));
        }
    }
}
