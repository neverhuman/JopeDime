use super::*;

#[test]
fn legacy_transform_wide_bound_model_predicts_identically_after_reload() {
    let width = 300;
    let schema = DataSchema {
        columns: (0..width)
            .map(|j| ColumnSpec::numeric(format!("x{j}")))
            .collect(),
        row_semantics: RowSemantics::Iid,
    };
    let headers: Vec<_> = schema.columns.iter().map(|c| c.name.clone()).collect();
    let rows: Vec<Vec<String>> = (0..12)
        .map(|i| (0..width).map(|j| ((i + j) % 7).to_string()).collect())
        .collect();
    let table = RawTable::new(headers, rows).unwrap();
    let target: Vec<_> = (0..12)
        .map(|i| if i % 7 >= 3 { 1.0 } else { 0.0 })
        .collect();
    let genome = GenomeIr::parse("(pipe X (input numeric=median missing=none) (robust_scale append=true) (histgbm depth=1 rounds=1 bins=8 min_leaf=1 lr=1 lambda=0))").unwrap();
    let model = fit_table(
        &genome,
        &TaskSpec::regression(Objective::R2).unwrap(),
        &schema,
        &table,
        &target,
    )
    .unwrap();
    let expected = model.predict_table(&table).unwrap();
    assert!(expected.iter().all(|v| v.is_finite()));
    assert!(expected.windows(2).any(|pair| pair[0] != pair[1]));
    let reloaded: BoundModel =
        serde_json::from_slice(&serde_json::to_vec(&model).unwrap()).unwrap();
    reloaded.validate_binding().unwrap();
    let actual = reloaded.predict_table(&table).unwrap();
    assert_eq!(
        expected.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        actual.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
    );
}

#[test]
fn synthesis_width_overflow_is_explicit_not_a_mask_dimension_failure() {
    let schema = DataSchema {
        columns: (0..6)
            .map(|j| ColumnSpec::numeric(format!("x{j}")))
            .collect(),
        row_semantics: RowSemantics::Iid,
    };
    let table = RawTable::new(
        schema.columns.iter().map(|c| c.name.clone()).collect(),
        (0..20)
            .map(|i| (0..6).map(|j| ((i * (j + 1)) % 13).to_string()).collect())
            .collect(),
    )
    .unwrap();
    let target: Vec<_> = (0..20).map(|i| i as f64).collect();
    // Same transform programs as the two retained failed native worker recipes;
    // the numerical input here is entirely synthetic and carries no panel rows.
    for (program, width) in [
        ("(pipe X (input categorical=onehot max_onehot=256 min_count=1 missing=indicator numeric=median) (polar append=true) (target_bin_encode bins=8) (robust_scale append=true) (poly_interact) (standardize append=true) (histgbm depth=2 rounds=180))", 1024),
        ("(pipe X (input categorical=onehot max_onehot=256 min_count=1 missing=indicator numeric=median) (svd_project append=false iters=16 k=16) (bin_onehot bins=12) (norm_features append=true) (robust_scale append=true) (ridge))", 576),
    ] {
        let genome = GenomeIr::parse(program).unwrap();
        let error = fit_table(&genome, &TaskSpec::regression(Objective::R2).unwrap(), &schema, &table, &target).unwrap_err().to_string();
        assert!(error.contains(&format!("generated dense width {width} exceeds 512")), "{error}");
        assert!(!error.contains("row/mask"));
    }
}

#[test]
fn v3_admission_matches_manifest() {
    use crate::kernel::capabilities::{Manifest, OpKind, V3Admission};
    use grammar::{ProductionAdmission, RowRequirement, StageClass};
    let manifest = Manifest::current();
    let mut found = std::collections::BTreeSet::new();
    let mut descriptors = 0;
    for descriptor in grammar::v3_operation_registry() {
        descriptors += 1;
        for &name in std::iter::once(&descriptor.name).chain(descriptor.aliases.iter()) {
            let row = manifest
                .op(name)
                .unwrap_or_else(|| panic!("missing {name}"));
            found.insert(name);
            let admission = match descriptor.admission {
                ProductionAdmission::Baseline => V3Admission::Baseline,
                ProductionAdmission::RequiresReceipt => V3Admission::RequiresReceipt,
                ProductionAdmission::Disabled => V3Admission::Disabled,
            };
            assert_eq!(row.v3_admission, admission, "{name}");
            let kind = match descriptor.stage_class {
                StageClass::Target => OpKind::Target,
                StageClass::Input => OpKind::Input,
                StageClass::Transform => OpKind::Transform,
                StageClass::Learner => OpKind::Head,
            };
            assert_eq!(row.kind, kind, "{name}");
            let role = match descriptor.stage_class {
                StageClass::Target => "target",
                StageClass::Input => "input",
                StageClass::Transform => "transform",
                StageClass::Learner => "head",
            };
            let entry = crate::kernel::registry_v2::operator("advanced_v3", name)
                .unwrap_or_else(|| panic!("missing advanced-v3 registry entry for {name}"));
            assert_eq!(entry.facts().role, role, "{name}");
            assert_eq!(row.requires_labels, descriptor.target_aware, "{name}");
            assert_eq!(
                row.requires_row_order,
                descriptor.row_requirement == RowRequirement::Ordered,
                "{name}"
            );
            if crate::registry::operator(name).is_none() {
                assert_eq!(row.width_effect, descriptor.width_effect, "{name}");
            }
        }
    }
    assert_eq!(descriptors, 26);
    for row in &manifest.ops {
        if !found.contains(row.name) {
            assert_eq!(row.v3_admission, V3Admission::Absent, "{}", row.name);
        }
    }
}
