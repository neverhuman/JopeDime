use super::*;

/// The shared transform body of the fixture hosts, in canonical text.
pub(crate) const E1: &str = "(cheb append=true degree=3) (pair_prod append=true topk=24)";
/// A cheap body used where engine time matters.
pub(crate) const SMALL: &str = "(standardize) (sigmoid)";
/// A body every fit refuses: the table-v3 fitter accepts only `oof=5`, while the planner and
/// the grammar ignore the parameter.
pub(crate) const REFUSED: &str = "(standardize) (target_bin_encode oof=3)";
/// A closed-form line, copied from an existing engine test.
pub(crate) const SYMBOLIC: &str = "(sym affine \"col(0)\")";
/// The explicit portfolio form.
pub(crate) const PORTFOLIO: &str =
    "(portfolio (member (pipe X (ridge_gcv))) (blend folds=5 guard=clip kind=nnls repeats=auto))";
/// The emitter lane's four-language control program.
pub(crate) const PORTABLE_4LANG: &str =
    include_str!("../../../feat-math/tests/fixtures/emitted/portable-4lang.dsl");
pub(crate) const FAMILY_A: &str = "canaryfamily01";
pub(crate) const FAMILY_B: &str = "canaryfamily02";
/// A distinctive, fixed 64-hex digest.
pub(crate) fn digest(tag: &str) -> String {
    sha256_hex(format!("jope-motifs-canary/{tag}").as_bytes())
}

/// The dataset digest of a fixture dataset name.
pub(crate) fn dataset(name: &str) -> String {
    digest(&format!("dataset/{name}"))
}

pub(crate) fn canonical(text: &str) -> String {
    GenomeIr::parse(text)
        .expect("fixture text parses")
        .canonical()
}

/// A native-shaped program: the panel input stage, a transform body and a head.
pub(crate) fn v3(body: &str, head: &str) -> String {
    canonical(&format!("(pipe X {INPUT_STAGE} {body} ({head}))"))
}

/// A roster-shaped program: the panel input stage and a head only.
pub(crate) fn roster(head: &str) -> String {
    canonical(&format!("(pipe X {INPUT_STAGE} ({head}))"))
}

/// The frozen panel's heads per task (`crates/jope-bench/src/panel/methods.rs:64-109`).
pub(crate) fn roster_heads() -> Vec<(TaskLabel, Vec<&'static str>)> {
    vec![
        (
            TaskLabel::Regression,
            vec![
                "ridge",
                "ridge_gcv",
                "elasticnet",
                "histgbm",
                "histgbm2",
                "ksmooth",
                "ensemble",
                "krr",
                "gpr",
                "hurdle2",
                "stack",
                "tab_boost",
                "extra_trees",
                "gam_boost",
                "pls",
                "portfolio preset=core",
                "gpr kernel=rbf+linear",
            ],
        ),
        (
            TaskLabel::Binary,
            vec![
                "logit_ridge",
                "tab_boost",
                "extra_trees",
                "gam_boost",
                "portfolio",
            ],
        ),
        (
            TaskLabel::Multiclass,
            vec![
                "multiclass_ridge",
                "multiclass_ovr_logistic",
                "multiclass_diagonal_lda",
                "multiclass_nearest_centroid",
            ],
        ),
    ]
}

pub(crate) fn validation_label() -> LabelOrigin {
    LabelOrigin {
        original_role: String::from("validation"),
        burn_sha256: None,
    }
}

/// The label origin of the real teaching configuration.
pub(crate) fn development_label() -> LabelOrigin {
    LabelOrigin {
        original_role: String::from("development"),
        burn_sha256: None,
    }
}

pub(crate) fn adaptive_label() -> LabelOrigin {
    LabelOrigin {
        original_role: String::from("test"),
        burn_sha256: Some(digest("burn/adaptive")),
    }
}

/// One admitted line with a non-adaptive label.
pub(crate) fn qualified(
    text: &str,
    task: TaskLabel,
    dataset_name: &str,
    record: &str,
) -> QualifiedProgram {
    qualified_with(text, task, dataset_name, record, validation_label())
}

/// One admitted line. The text is canonicalized first.
pub(crate) fn qualified_with(
    text: &str,
    task: TaskLabel,
    dataset_name: &str,
    record: &str,
    label: LabelOrigin,
) -> QualifiedProgram {
    let program = canonical(text);
    let recipe = sha256_hex(program.as_bytes());
    let dataset_sha256 = dataset(dataset_name);
    let cut_sha256 = digest(&format!("cut/{dataset_name}"));
    let comparison_sha256 = digest(&format!("comparison/{record}"));
    let record_sha256 = digest(&format!("record/{record}"));
    QualifiedProgram::admit(Admission {
        program: &program,
        recipe_sha256: &recipe,
        task: task.as_str(),
        dataset_sha256: &dataset_sha256,
        cut_sha256: &cut_sha256,
        comparison_sha256: &comparison_sha256,
        record_sha256: &record_sha256,
        label,
    })
    .expect("fixture line admits")
}

/// The fourteen fixture lines of the design (section 9).
pub(crate) fn lines() -> Vec<QualifiedProgram> {
    use TaskLabel::{Binary, Multiclass, Regression};
    let champion = feat_math::CHAMPION_GENOME.trim();
    let portable = PORTABLE_4LANG.trim();
    vec![
        qualified(&v3(E1, "ridge"), Regression, "d1", "r01"),
        qualified(&v3(E1, "elasticnet"), Regression, "d2", "r02"),
        // The same program again on d1 under a second comparison identity.
        qualified(&v3(E1, "ridge"), Regression, "d1", "r03"),
        qualified(&v3(E1, "logit_ridge"), Binary, "d3", "r04"),
        qualified(
            &v3(&format!("(robust_scale) {E1}"), "multiclass_ridge"),
            Multiclass,
            "d4",
            "r05",
        ),
        // Roster-shaped: a host with an empty body.
        qualified(&roster("tab_boost"), Regression, "d1", "r06"),
        // A body broken by an expression stage.
        qualified(
            &v3(
                "(cheb append=true degree=3) (map expr=\"sin(x)\") (pair_prod append=true topk=24)",
                "ridge",
            ),
            Regression,
            "d2",
            "r07",
        ),
        // Order-dependent: unsupported under independent rows.
        qualified(
            &v3(&format!("(lag k=1) {E1}"), "ridge"),
            Regression,
            "d3",
            "r08",
        ),
        // A quarantined operation: refused by the slot contract.
        qualified(
            &v3(&format!("(rff_orthogonal n=8) {E1}"), "ridge"),
            Regression,
            "d4",
            "r09",
        ),
        // The reserved prefix inside a parameter.
        qualified(
            &v3(
                "(cheb append=true degree=3 tag=xmotif_) (pair_prod append=true topk=24)",
                "ridge",
            ),
            Regression,
            "d5",
            "r10",
        ),
        qualified(SYMBOLIC, Regression, "d6", "r11"),
        // Optimization-visible test labels: excluded by default.
        qualified_with(&v3(E1, "ridge"), Regression, "d7", "r12", adaptive_label()),
        // The two legacy controls.
        qualified(champion, Regression, "d5", "r13"),
        qualified(portable, Regression, "d6", "r14"),
    ]
}
