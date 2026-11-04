use super::*;

const SOURCE_SCHEMA: &str = "jope.qualified-source-fixture/v1";

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
