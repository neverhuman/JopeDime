use super::super::schema::ColumnSpec;
use super::super::types::Objective;
use super::*;

fn regression() -> TaskSpec {
    TaskSpec::regression(Objective::SquaredError).unwrap()
}

fn binary() -> TaskSpec {
    TaskSpec::binary(Objective::LogLoss).unwrap()
}

fn numeric_schema(width: usize) -> DataSchema {
    DataSchema {
        columns: (0..width)
            .map(|column| ColumnSpec::numeric(format!("x{column}")))
            .collect(),
        row_semantics: RowSemantics::Iid,
    }
}

fn ordered_schema() -> DataSchema {
    DataSchema {
        columns: vec![ColumnSpec::numeric("time"), ColumnSpec::numeric("value")],
        row_semantics: RowSemantics::Ordered {
            group_by: vec![],
            order_by: vec!["time".to_string()],
        },
    }
}

fn parse(source: &str) -> GenomeIr {
    GenomeIr::parse(source).unwrap()
}

#[test]
fn registry_names_are_unique_and_aliases_are_task_unambiguous() {
    let mut names = BTreeSet::new();
    for descriptor in v3_operation_registry() {
        assert!(names.insert(descriptor.name));
        for task in [TaskKind::Regression, TaskKind::Binary] {
            if descriptor.tasks.contains(task) {
                for alias in descriptor.aliases {
                    let matches = REGISTRY
                        .iter()
                        .filter(|candidate| {
                            candidate.tasks.contains(task)
                                && (candidate.name == *alias || candidate.aliases.contains(alias))
                        })
                        .count();
                    assert_eq!(matches, 1, "ambiguous alias {alias} for {task:?}");
                }
            }
        }
    }
}

#[test]
fn omitted_and_explicit_compatibility_defaults_share_a_semantic_hash() {
    let task = regression();
    let schema = numeric_schema(3);
    let a = compile_v3_genome(
        &parse("(pipe X (ridge))"),
        &task,
        &schema,
        &CompileV3Options::prototype(None),
    )
    .unwrap();
    let b = compile_v3_genome(
            &parse(
                "(pipe X (input numeric=zero missing=none categorical=native max_onehot=16 min_count=2 prior=10 folds=5) (ridge lambda_frac=0.02))",
            ),
            &task,
            &schema,
            &CompileV3Options::prototype(None),
        )
        .unwrap();
    assert_eq!(a.canonical(), b.canonical());
    assert_eq!(a.semantic_hash(), b.semantic_hash());
    assert_eq!(a.stages().len(), 2);
}

#[test]
fn explicit_input_uses_its_distinct_defaults() {
    let compiled = compile_v3_genome(
        &parse("(pipe X (input) (ridge))"),
        &regression(),
        &numeric_schema(2),
        &CompileV3Options::prototype(None),
    )
    .unwrap();
    assert!(compiled.canonical().contains("missing=indicator"));
    assert_eq!(compiled.feature_widths().raw_derived, 2);
    assert_eq!(compiled.feature_widths().generated_sparse, 2);
}

#[test]
fn strict_types_unknown_keys_and_nonfinite_values_are_rejected() {
    let task = regression();
    let schema = numeric_schema(2);
    for source in [
        "(pipe X (standardize mystery=1) (ridge))",
        "(pipe X (piecewise_linear bins=8.0) (ridge))",
        "(pipe X (piecewise_linear bins=8.5) (ridge))",
        "(pipe X (standardize append=1) (ridge))",
        "(pipe X (step t=NaN) (ridge))",
    ] {
        assert!(
            compile_v3_genome(
                &parse(source),
                &task,
                &schema,
                &CompileV3Options::prototype(None)
            )
            .is_err(),
            "accepted {source}"
        );
    }
}

#[test]
fn aliases_canonicalize_before_hashing() {
    let task = regression();
    let schema = numeric_schema(2);
    let a = compile_v3_genome(
        &parse("(pipe X (norm_features) (logistic))"),
        &task,
        &schema,
        &CompileV3Options::prototype(None),
    )
    .unwrap();
    let b = compile_v3_genome(
        &parse("(pipe X (standardize) (ridge))"),
        &task,
        &schema,
        &CompileV3Options::prototype(None),
    )
    .unwrap();
    assert_eq!(a.canonical(), b.canonical());
    assert_eq!(a.semantic_hash(), b.semantic_hash());
}

#[test]
fn misplaced_and_task_incompatible_stages_are_rejected() {
    let schema = numeric_schema(2);
    assert!(compile_v3_genome(
        &parse("(pipe X (standardize) (input) (ridge))"),
        &regression(),
        &schema,
        &CompileV3Options::prototype(None),
    )
    .is_err());
    assert!(compile_v3_genome(
        &parse("(pipe X (ridge) (standardize))"),
        &regression(),
        &schema,
        &CompileV3Options::prototype(None),
    )
    .is_err());
    assert!(compile_v3_genome(
        &parse("(pipe X (target kind=zscore) (logit_ridge))"),
        &binary(),
        &schema,
        &CompileV3Options::prototype(None),
    )
    .is_err());
}

#[test]
fn row_operations_require_ordered_semantics() {
    let genome = parse("(pipe X (lag k=1) (ridge))");
    assert!(compile_v3_genome(
        &genome,
        &regression(),
        &numeric_schema(2),
        &CompileV3Options::prototype(None),
    )
    .is_err());
    let compiled = compile_v3_genome(
        &genome,
        &regression(),
        &ordered_schema(),
        &CompileV3Options::prototype(None),
    )
    .unwrap();
    assert!(compiled.requires_ordered_rows());
}

#[test]
fn raw_wide_one_to_one_is_preserved_but_generated_dense_is_bounded() {
    let task = regression();
    let wide = numeric_schema(1_024);
    let accepted = compile_v3_genome(
        &parse("(pipe X (standardize) (ridge))"),
        &task,
        &wide,
        &CompileV3Options::prototype(None),
    )
    .unwrap();
    assert_eq!(accepted.feature_widths().raw_derived, 1_024);
    assert_eq!(accepted.feature_widths().generated_dense, 0);
    assert!(compile_v3_genome(
        &parse("(pipe X (standardize append=true) (ridge))"),
        &task,
        &wide,
        &CompileV3Options::prototype(None),
    )
    .is_err());

    let boundary = numeric_schema(MAX_GENERATED_DENSE_WIDTH / 8);
    let accepted = compile_v3_genome(
        &parse("(pipe X (piecewise_linear bins=8) (ridge))"),
        &task,
        &boundary,
        &CompileV3Options::prototype(None),
    )
    .unwrap();
    assert_eq!(
        accepted.feature_widths().generated_dense,
        MAX_GENERATED_DENSE_WIDTH
    );
    assert!(compile_v3_genome(
        &parse("(pipe X (piecewise_linear bins=8) (ridge))"),
        &task,
        &numeric_schema(MAX_GENERATED_DENSE_WIDTH / 8 + 1),
        &CompileV3Options::prototype(None),
    )
    .is_err());
}

#[test]
fn production_grids_and_family_admission_are_enforced() {
    let task = regression();
    let schema = numeric_schema(4);
    let emitters = [EmitterTarget::Runtime];
    let admitted = ["tab_boost"];
    assert!(compile_v3_genome(
            &parse("(pipe X (tab_boost rounds=128 depth=3 bins=32 lr=0.05 min_leaf=5 lambda=1 row_fraction=1 feature_fraction=1))"),
            &task,
            &schema,
            &CompileV3Options::production(&emitters, &[], None),
        )
        .is_err());
    assert!(compile_v3_genome(
        &parse("(pipe X (tab_boost rounds=100))"),
        &task,
        &schema,
        &CompileV3Options::production(&emitters, &admitted, None),
    )
    .is_err());
    assert!(compile_v3_genome(
            &parse("(pipe X (tab_boost rounds=128 depth=3 bins=32 lr=0.05 min_leaf=5 lambda=1 row_fraction=1 feature_fraction=1))"),
            &task,
            &schema,
            &CompileV3Options::production(&emitters, &admitted, None),
        )
        .is_ok());
}

#[test]
fn unsupported_production_emitter_is_rejected() {
    let emitters = [EmitterTarget::Python];
    let admitted = ["row_operations"];
    assert!(compile_v3_genome(
        &parse("(pipe X (lag k=1) (ridge))"),
        &regression(),
        &ordered_schema(),
        &CompileV3Options::production(&emitters, &admitted, None),
    )
    .is_err());
}

#[test]
fn native_roles_are_prototype_compatible_but_block_numeric_production_heads() {
    let schema = DataSchema {
        columns: vec![ColumnSpec::categorical("category")],
        row_semantics: RowSemantics::Iid,
    };
    let genome = parse("(pipe X (ridge))");
    assert!(compile_v3_genome(
        &genome,
        &regression(),
        &schema,
        &CompileV3Options::prototype(None),
    )
    .is_ok());
    assert!(compile_v3_genome(
        &genome,
        &regression(),
        &schema,
        &CompileV3Options::production(&[EmitterTarget::Runtime], &[], None),
    )
    .is_err());
}

#[test]
fn onehot_and_hybrid_never_guess_an_exact_layout() {
    let schema = DataSchema {
        columns: vec![ColumnSpec::categorical("category")],
        row_semantics: RowSemantics::Iid,
    };
    let genome = parse("(pipe X (input categorical=onehot missing=none) (ridge))");
    assert!(compile_v3_genome(
        &genome,
        &regression(),
        &schema,
        &CompileV3Options::prototype(None),
    )
    .is_err());
    let layout = CompileInputLayout::new(vec![CompileFeature {
        role: FeatureRole::CategoricalOneHot,
        origin: FeatureOrigin::GeneratedSparse,
        missing_mask: false,
    }])
    .unwrap();
    let compiled = compile_v3_genome(
        &genome,
        &regression(),
        &schema,
        &CompileV3Options::prototype(Some(&layout)),
    )
    .unwrap();
    assert_eq!(compiled.feature_widths().generated_sparse, 1);
}

#[test]
fn target_bin_absence_is_hash_distinct_from_explicit_oof_five() {
    let task = regression();
    let schema = numeric_schema(2);
    let legacy = compile_v3_genome(
        &parse("(pipe X (target_bin_encode) (ridge))"),
        &task,
        &schema,
        &CompileV3Options::prototype(None),
    )
    .unwrap();
    let honest = compile_v3_genome(
        &parse("(pipe X (target_bin_encode oof=5) (ridge))"),
        &task,
        &schema,
        &CompileV3Options::prototype(None),
    )
    .unwrap();
    assert_ne!(legacy.semantic_hash(), honest.semantic_hash());
    assert!(legacy.canonical().contains("oof=legacy"));
    assert!(honest.canonical().contains("oof=5"));
}
