//! Test-only fixture: synthetic qualified programs with planted sentinels.
//!
//! Every program is canonicalized at test time and admitted through the same path the state
//! reader uses, so no line can be non-canonical by hand. Every provenance field carries a
//! distinctive digest, the adaptive line carries a burn digest, and the family labels are
//! `canaryfamily01` and `canaryfamily02`. [`FakeState`] holds admitted lines as a learning
//! State would: one measured entry and one exact-identity pointer per line.

use crate::fragment::INPUT_STAGE;
use crate::mine::mine;
use crate::source::{
    Admission, FamilyMap, LabelOrigin, QualifiedProgram, QualifiedSet, QualifiedSource, TaskLabel,
};
use crate::table::{MineParameters, MotifTable};
use crate::teaching::{
    BurnReceipt, ContextFields, EligibleState, GuardFields, PointerKeyFields, PointerRecord,
    QualifiedEntry, QualifiedRead, ENTRY_SCHEMA, SET_SCHEMA,
};
use crate::validate::{CheckResult, MotifResult, Validation, VALIDATION_SCHEMA};
use crate::{sha256_hex, NAME_PREFIX};
use anyhow::{ensure, Result};
use feat_math::GenomeIr;
use std::cell::Cell;
use std::collections::BTreeMap;

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
    include_str!("../../feat-math/tests/fixtures/emitted/portable-4lang.dsl");
pub(crate) const FAMILY_A: &str = "canaryfamily01";
pub(crate) const FAMILY_B: &str = "canaryfamily02";
const SOURCE_SCHEMA: &str = "jope.qualified-source-fixture/v1";

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

/// A fixed in-memory reader of qualified programs.
struct FixtureSource(Vec<QualifiedProgram>);

impl QualifiedSource for FixtureSource {
    fn qualified(&self) -> Result<QualifiedSet> {
        QualifiedSet::new(self.0.clone(), SOURCE_SCHEMA, &digest("source"), 0)
    }
}

pub(crate) fn set_of(programs: Vec<QualifiedProgram>) -> QualifiedSet {
    FixtureSource(programs).qualified().expect("fixture set")
}

pub(crate) fn set() -> QualifiedSet {
    set_of(lines())
}

/// Mine fixture lines with the default parameters and no family map.
pub(crate) fn mined(programs: Vec<QualifiedProgram>) -> MotifTable {
    mine(&set_of(programs), None, MineParameters::default()).expect("fixture mines")
}

/// Mine fixture lines with [`family_map`], recording the map's own digest.
pub(crate) fn mined_with_families(programs: Vec<QualifiedProgram>) -> MotifTable {
    let map = family_map();
    let parameters = MineParameters {
        family_map_sha256: Some(map.sha256().to_owned()),
        ..MineParameters::default()
    };
    mine(&set_of(programs), Some(&map), parameters).expect("fixture mines with families")
}

/// The table mined from every fixture line with the default parameters.
pub(crate) fn table() -> MotifTable {
    mined(lines())
}

/// The table mined from every fixture line with the family map, so the family-label
/// canaries are planted in its provenance.
pub(crate) fn labelled_table() -> MotifTable {
    mined_with_families(lines())
}

/// The two lines of [`small_table`].
pub(crate) fn small_lines() -> Vec<QualifiedProgram> {
    vec![
        qualified(&v3(SMALL, "elasticnet"), TaskLabel::Regression, "d1", "s01"),
        qualified(&v3(SMALL, "elasticnet"), TaskLabel::Regression, "d2", "s02"),
    ]
}

/// The two lines of [`labelled_small_table`], on datasets of both canary families.
pub(crate) fn labelled_small_lines() -> Vec<QualifiedProgram> {
    vec![
        qualified(&v3(SMALL, "elasticnet"), TaskLabel::Regression, "d1", "s01"),
        qualified(&v3(SMALL, "elasticnet"), TaskLabel::Regression, "d3", "s03"),
    ]
}

/// One fragment, `(standardize) (sigmoid)`, on two datasets with one recipe. The host's head
/// is not the task default, so no context repeats it.
pub(crate) fn small_table() -> MotifTable {
    mined(small_lines())
}

/// The small fragment on datasets of both canary families, mined with the family map.
pub(crate) fn labelled_small_table() -> MotifTable {
    mined_with_families(labelled_small_lines())
}

/// One fragment that every fit refuses, on two datasets with two distinct recipes.
pub(crate) fn refused_table() -> MotifTable {
    mined(vec![
        qualified(&v3(REFUSED, "ridge"), TaskLabel::Regression, "d1", "f01"),
        qualified(
            &v3(REFUSED, "elasticnet"),
            TaskLabel::Regression,
            "d2",
            "f02",
        ),
    ])
}

/// The small fragment observed only with multiclass hosts, for which no emitter produces
/// an artifact.
pub(crate) fn multiclass_table() -> MotifTable {
    mined(vec![
        qualified(
            &v3(SMALL, "multiclass_ridge"),
            TaskLabel::Multiclass,
            "d1",
            "m01",
        ),
        qualified(
            &v3(SMALL, "multiclass_ridge"),
            TaskLabel::Multiclass,
            "d2",
            "m02",
        ),
    ])
}

/// The small fragment with one host that carries the default head, so Q1 repeats it.
pub(crate) fn default_head_table() -> MotifTable {
    mined(vec![
        qualified(&v3(SMALL, "ridge"), TaskLabel::Regression, "d1", "h01"),
        qualified(&v3(SMALL, "elasticnet"), TaskLabel::Regression, "d2", "h02"),
    ])
}

/// The small fragment with hosts equal to both Q1 and Q2 once the default head is written
/// out, so neither native context is independent.
pub(crate) fn host_context_table() -> MotifTable {
    mined(vec![
        qualified(&v3(SMALL, "ridge"), TaskLabel::Regression, "d1", "c01"),
        qualified(
            &v3(&format!("(standardize) {SMALL}"), "ridge"),
            TaskLabel::Regression,
            "d2",
            "c02",
        ),
    ])
}

/// Family labels of every fixture dataset that yields windows.
fn family_labels() -> BTreeMap<String, String> {
    let mut map = BTreeMap::new();
    for (name, family) in [
        ("d1", FAMILY_A),
        ("d2", FAMILY_A),
        ("d3", FAMILY_B),
        ("d4", FAMILY_B),
        ("d5", FAMILY_B),
        ("d6", FAMILY_B),
    ] {
        map.insert(dataset(name), family.to_owned());
    }
    map
}

/// A map labelling every fixture dataset that yields windows.
pub(crate) fn family_map() -> FamilyMap {
    FamilyMap::from_map(family_labels()).expect("valid family map")
}

/// The private family-map file of [`family_map`]: its compact JSON, so a map parsed from these
/// bytes carries the same digest.
pub(crate) fn family_map_bytes() -> Vec<u8> {
    serde_json::to_vec(&family_labels()).expect("json")
}

/// The burn receipt a learning State holds for a line whose test labels were burned: its
/// dataset and cut, relabelled development.
pub(crate) fn burn_receipt(line: &QualifiedProgram) -> BurnReceipt {
    BurnReceipt {
        schema: String::from("jope.label-burn/v1"),
        dataset_sha256: line.dataset_sha256().to_owned(),
        cut_sha256: line.cut_sha256().to_owned(),
        original_role: String::from("test"),
        effective_role: String::from("development"),
    }
}

/// The qualified-set entry and exact-identity pointer a learning State holds for one admitted
/// line: a measured admission, under the known guard, that improves in every fold. A line
/// with a burn digest also has its receipt in the State.
pub(crate) fn state_entry(line: &QualifiedProgram) -> (QualifiedEntry, PointerRecord) {
    let task = line.task().as_str().to_owned();
    let entry = QualifiedEntry {
        schema: ENTRY_SCHEMA.to_owned(),
        freeze_comparison_sha256: line.comparison_sha256().to_owned(),
        comparison_sha256: line.comparison_sha256().to_owned(),
        task: task.clone(),
        dataset_sha256: line.dataset_sha256().to_owned(),
        cut_sha256: line.cut_sha256().to_owned(),
        evaluator_sha256: digest("evaluator"),
        vocabulary_sha256: digest("vocabulary"),
        protocol: String::from("train-oof-fixture-v1"),
        context: Some(ContextFields {
            numerical_policy: String::from(
                "asm-panel-f64-v1:platform-libm-v1:probability-epsilon=1e-15:row-sum-tolerance=1e-6",
            ),
            stage: String::from("train_oof"),
            aggregation: String::from("mean_fold_loss"),
            label_role: String::from("development"),
            recipe_encoding: String::from("native-canonical-dsl-sha256/v1"),
            sealed: false,
            benchmark_valid: false,
            checkpoint_sha256: None,
        }),
        record_sha256: line.record_sha256().to_owned(),
        program_sha256: line.recipe_sha256().to_owned(),
        guard: GuardFields::current(),
        label_role: String::from("development"),
        original_label_role: line.label().original_role.clone(),
        label_burn_sha256: line.label().burn_sha256.clone(),
        label_burn: line
            .label()
            .burn_sha256
            .as_ref()
            .map(|_| burn_receipt(line)),
        mean_loss: 0.5,
        baseline_loss: 1.0,
        improved_folds: 5,
        fold_losses: vec![(0.5, 1.0); 5],
        context_outer_fold: None,
        context_fitting_rows_sha256: None,
    };
    let pointer = PointerRecord {
        key: PointerKeyFields {
            task: task.clone(),
            dataset_sha256: line.dataset_sha256().to_owned(),
            cut_sha256: line.cut_sha256().to_owned(),
            identity_sha256: Some(line.comparison_sha256().to_owned()),
        },
        task,
        dataset_sha256: line.dataset_sha256().to_owned(),
        cut_sha256: line.cut_sha256().to_owned(),
        identity_sha256: line.comparison_sha256().to_owned(),
        has_comparison: true,
        record_sha256: line.record_sha256().to_owned(),
        program: line.program().to_owned(),
        program_sha256: line.recipe_sha256().to_owned(),
        sealed: false,
        benchmark_valid: false,
        replayable: true,
        valid: true,
    };
    (entry, pointer)
}

/// A learning State in memory. Every qualified-set read returns `read`, except that reads
/// after the first return `later` when it is set; `broken` fails every read.
#[derive(Clone, Debug)]
pub(crate) struct FakeState {
    pub(crate) read: QualifiedRead,
    pub(crate) later: Option<QualifiedRead>,
    pub(crate) pointers: Vec<PointerRecord>,
    pub(crate) broken: bool,
    reads: Cell<u32>,
}

impl FakeState {
    /// A State whose qualified set holds exactly `lines`, each with its exact pointer.
    pub(crate) fn of(lines: &[QualifiedProgram]) -> Self {
        let (entries, pointers): (Vec<QualifiedEntry>, Vec<PointerRecord>) =
            lines.iter().map(state_entry).unzip();
        Self {
            read: QualifiedRead {
                schema: SET_SCHEMA.to_owned(),
                sha256: digest("state-read"),
                excluded: 0,
                entries,
            },
            later: None,
            pointers,
            broken: false,
            reads: Cell::new(0),
        }
    }

    /// Qualified-set reads so far.
    pub(crate) fn reads(&self) -> u32 {
        self.reads.get()
    }
}

impl EligibleState for FakeState {
    fn qualified_read(&self) -> Result<QualifiedRead> {
        ensure!(!self.broken, "fixture ledger chain broken");
        let done = self.reads.get();
        self.reads.set(done + 1);
        Ok(match &self.later {
            Some(later) if done > 0 => later.clone(),
            _ => self.read.clone(),
        })
    }

    fn pointer_records(&self) -> Result<Vec<PointerRecord>> {
        ensure!(!self.broken, "fixture pointer directory unreadable");
        Ok(self.pointers.clone())
    }
}

fn push_digest(out: &mut Vec<String>, digest: &str) {
    out.push(digest.to_owned());
    if let Some(prefix) = digest.get(..16) {
        out.push(prefix.to_owned());
    }
}

/// The identifiers one table carries: names, full ids and their 12- and 16-hex prefixes,
/// every provenance digest (burn included) with its 16-hex prefix, the family labels in its
/// provenance, the source label and digest, and the family map digest. Host texts,
/// expansions, stage texts and recipe digests are listed separately, because a bound
/// manifest legitimately carries the digest of its own program. The table digest is not
/// listed: operator-private stdout may carry it, and the shareable summary is checked for it
/// separately.
fn carried_identifiers(table: &MotifTable) -> Vec<String> {
    let mut out = Vec::new();
    for motif in &table.body.motifs {
        out.push(motif.name.clone());
        out.push(motif.id.clone());
        out.push(motif.id[..12].to_owned());
        out.push(motif.id[..16].to_owned());
        for occurrence in &motif.provenance {
            for digest in [
                &occurrence.dataset_sha256,
                &occurrence.cut_sha256,
                &occurrence.comparison_sha256,
                &occurrence.record_sha256,
            ] {
                push_digest(&mut out, digest);
            }
            if let Some(burn) = &occurrence.label.burn_sha256 {
                push_digest(&mut out, burn);
            }
            if let Some(family) = &occurrence.family {
                out.push(family.clone());
            }
        }
    }
    out.push(table.body.source.schema.clone());
    push_digest(&mut out, &table.body.source.sha256);
    if let Some(map) = &table.body.parameters.family_map_sha256 {
        push_digest(&mut out, map);
    }
    out
}

/// The program texts one table carries: expansions, stage texts, host programs and recipe
/// digests with their 16-hex prefixes.
fn carried_texts(table: &MotifTable) -> Vec<String> {
    let mut out = Vec::new();
    for motif in &table.body.motifs {
        out.push(motif.expansion.clone());
        if let Ok(stages) = crate::fragment::expansion_stages(&motif.expansion) {
            out.extend(crate::fragment::stage_texts(&stages));
        }
    }
    for (recipe, host) in &table.body.hosts {
        push_digest(&mut out, recipe);
        out.push(host.clone());
    }
    out
}

/// Positive control for every absence assertion: each private value the sentinels derive
/// from this table really is in the table's private bytes, and a labelled table carries
/// both family canaries. Without it an absence assertion could pass because the canary was
/// never planted.
pub(crate) fn assert_carried(table: &MotifTable) {
    let bytes = String::from_utf8(table.to_json().expect("json")).expect("utf-8");
    let mut values = carried_identifiers(table);
    values.extend(carried_texts(table));
    assert!(!values.is_empty());
    for value in values {
        let encoded = serde_json::to_string(&value).expect("json string");
        let inner = &encoded[1..encoded.len() - 1];
        assert!(bytes.contains(inner), "{value} is not in the private table");
    }
    if table.body.parameters.family_map_sha256.is_some() {
        for family in [FAMILY_A, FAMILY_B] {
            assert!(bytes.contains(family), "{family} is not planted");
        }
    }
}

/// What no emitted artifact may contain: the reserved prefix, both family labels, the
/// identifiers the table carries, and every digest of every fixture line.
pub(crate) fn emitted_sentinels(table: &MotifTable) -> Vec<String> {
    let mut out = vec![
        NAME_PREFIX.to_owned(),
        FAMILY_A.to_owned(),
        FAMILY_B.to_owned(),
    ];
    out.extend(carried_identifiers(table));
    for line in lines() {
        for digest in [
            line.dataset_sha256(),
            line.cut_sha256(),
            line.comparison_sha256(),
            line.record_sha256(),
        ] {
            push_digest(&mut out, digest);
        }
        if let Some(burn) = &line.label().burn_sha256 {
            push_digest(&mut out, burn);
        }
    }
    out
}

/// Everything planted: the emitted sentinels plus expansions, stage texts, host programs and
/// recipe digests.
pub(crate) fn sentinels(table: &MotifTable) -> Vec<String> {
    let mut out = emitted_sentinels(table);
    out.extend(carried_texts(table));
    out
}

/// A report in which every check of every fragment passed once.
pub(crate) fn passing_validation(table: &MotifTable) -> Validation {
    let pass = CheckResult {
        passed: true,
        exercised: 1,
        failures: Vec::new(),
    };
    Validation {
        schema: VALIDATION_SCHEMA.to_owned(),
        table_sha256: table.table_sha256.clone(),
        source_verified: false,
        source_digest_equal: None,
        motifs: (0..table.body.motifs.len())
            .map(|index| MotifResult {
                index: u32::try_from(index).expect("index"),
                round_trip: pass.clone(),
                determinism: pass.clone(),
                context_compatibility: pass.clone(),
                name_absence: pass.clone(),
                not_exercised: 0,
                backend: Vec::new(),
            })
            .collect(),
    }
}
