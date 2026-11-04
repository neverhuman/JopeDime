fn legacy_generators() -> Vec<GeneratorCap> {
    vec![
        GeneratorCap {
            id: "legacy-expr-edit-v1",
            stage_ops: vec![],
            functions: FUNCTIONS.iter().map(|row| row.1).collect(),
            binops: BINOPS.iter().map(|row| row.1).collect(),
            variables: vec!["x", "row_mean", "col_under_8"],
            unary: vec![],
        },
        GeneratorCap {
            id: "legacy-full-grower",
            stage_ops: crate::full_rust::TRANSFORMS
                .into_iter()
                .chain(crate::full_rust::EXPANSION_TRANSFORMS)
                .chain(crate::full_rust::HEADS)
                .collect(),
            functions: FUNCTIONS.iter().map(|row| row.1).collect(),
            binops: BINOPS.iter().map(|row| row.1).collect(),
            variables: vec![
                "x",
                "col_under_8",
                "col_8_plus",
                "row_mean",
                "row_sum",
                "row_l1",
                "row_l2",
                "row_min",
                "row_max",
                "row_std",
            ],
            unary: vec!["neg"],
        },
        GeneratorCap {
            id: "legacy-slot-insert-v1",
            stage_ops: vec![
                "standardize",
                "robust_scale",
                "sigmoid",
                "norm_row",
                "symlog",
                "step",
                "row_stats",
            ],
            functions: vec![],
            binops: vec![],
            variables: vec![],
            unary: vec![],
        },
        GeneratorCap {
            id: "legacy-v3-typed-edit-v1",
            stage_ops: vec![
                "target",
                "quantile_bins",
                "blom",
                "pair_prod",
                "elasticnet",
                "winsor_clip",
                "cheb",
                "ridge",
                "rff_cos",
                "ridge_gcv",
                "hinge",
                "standardize",
            ],
            functions: vec![],
            binops: vec![],
            variables: vec![],
            unary: vec![],
        },
    ]
}

/// Stable JSON assembled from typed rows and ordered maps.
pub fn manifest_json() -> String {
    serde_json::to_string(&Manifest::current()).expect("manifest serializes")
}

/// Human-readable view generated from the same rows as the JSON.
pub fn render_markdown() -> String {
    let manifest = Manifest::current();
    let mut out = String::from("# Kernel capability matrix\n\nThis declaration is checked against the frozen grammars and available routes.\n\n");
    out.push_str("## Authority\n\nThe two frozen token samplers use registry authority; every later grammar uses the manifest. Each grammar's file hash selects the rule.\n\n| Consumer | Authority |\n| --- | --- |\n| External product generators | Registry |\n| Frozen token samplers | Each file's flags, checked under registry authority |\n| Kernel profiles and later grammars | Manifest |\n\n");
    out.push_str("## Expression elements\n\n| Kind | Element | Class | Rust recipe | Python recipe | C++17 recipe | C++17 fitted | JavaScript |\n| --- | --- | --- | --- | --- | --- | --- | --- |\n");
    for row in &manifest.expression {
        out.push_str(&format!(
            "| {} | {} | {} | {} | {} | {} | {} | {} |\n",
            row.kind,
            row.name,
            class_label(row.capability.class),
            support_label(row.capability.emitters.rust_recipe),
            support_label(row.capability.emitters.python_recipe),
            support_label(row.capability.emitters.cpp17_recipe),
            support_label(row.capability.emitters.cpp17_fitted),
            support_label(row.capability.emitters.javascript)
        ));
    }
    out.push_str("\n## Policies\n\n| Policy | Class | Rust recipe | Python recipe | C++17 recipe | C++17 fitted | JavaScript | Note |\n| --- | --- | --- | --- | --- | --- | --- | --- |\n");
    for row in &manifest.policies {
        out.push_str(&format!(
            "| {} | {} | {} | {} | {} | {} | {} | {} |\n",
            row.name,
            class_label(row.class),
            support_label(row.emitters.rust_recipe),
            support_label(row.emitters.python_recipe),
            support_label(row.emitters.cpp17_recipe),
            support_label(row.emitters.cpp17_fitted),
            support_label(row.emitters.javascript),
            row.notes
        ));
    }
    out.push_str("\n## Operations\n\n| Element | Kind | Kernel tier | V3 admission | Rust recipe | Python recipe | C++17 recipe | C++17 fitted | JavaScript |\n| --- | --- | --- | --- | --- | --- | --- | --- | --- |\n");
    for row in &manifest.ops {
        out.push_str(&format!(
            "| {} | {:?} | {} | {:?} | {} | {} | {} | {} | {} |\n",
            row.name,
            row.kind,
            kernel_label(row.kernel2),
            row.v3_admission,
            support_label(row.emitters.rust_recipe),
            support_label(row.emitters.python_recipe),
            support_label(row.emitters.cpp17_recipe),
            support_label(row.emitters.cpp17_fitted),
            support_label(row.emitters.javascript)
        ));
    }
    out.push_str("\n## Program forms\n\n| Form | Kernel tier | Rust recipe | Python recipe | C++17 recipe | C++17 fitted | JavaScript |\n| --- | --- | --- | --- | --- | --- | --- | --- |\n");
    for row in &manifest.forms {
        out.push_str(&format!(
            "| {} | {} | {} | {} | {} | {} | {} |\n",
            row.name,
            kernel_label(row.kernel2),
            support_label(row.emitters.rust_recipe),
            support_label(row.emitters.python_recipe),
            support_label(row.emitters.cpp17_recipe),
            support_label(row.emitters.cpp17_fitted),
            support_label(row.emitters.javascript)
        ));
    }
    out.push_str("\n## Version 2 reservations\n\nThese IDs have no generator or executable route. Unnamed slots are held for later assignment.\n\n| Domain | ID | Assigned name | Lifecycle | Generatable |\n| --- | ---: | --- | --- | --- |\n");
    for row in &manifest.v2_reservations {
        out.push_str(&format!(
            "| {:?} | {} | {} | {} | {} |\n",
            row.domain,
            row.id,
            row.name.unwrap_or("—"),
            row.lifecycle,
            row.generatable
        ));
    }
    out
}

fn class_label(value: Class) -> &'static str {
    match value {
        Class::Searchable {
            tier: Tier::Baseline,
            ..
        } => "baseline",
        Class::Searchable {
            tier: Tier::Experimental,
            ..
        } => "experimental",
        Class::RuntimeOnlyIntentional { .. } => "runtime only",
        Class::Compatibility => "compatibility",
        Class::Quarantined => "quarantined",
        Class::Unsupported => "unsupported",
    }
}
fn kernel_label(value: Kernel2Use) -> &'static str {
    match value {
        Kernel2Use::Baseline => "baseline",
        Kernel2Use::Experimental => "experimental",
        Kernel2Use::NotGenerated { .. } => "not generated",
    }
}
fn support_label(value: Support) -> &'static str {
    match value {
        Support::Supported => "supported",
        Support::Planned { .. } => "planned",
        Support::Deferred { .. } => "deferred",
        Support::Unsupported { .. } => "unsupported",
        Support::SubsetOnly { .. } => "subset only",
    }
}

/// B12: admission belongs exclusively to the v2 FeatureWithinRow profile.
/// Baseline/legacy grammar admission and their manifest hashes remain frozen.
pub fn row_vector_v2_capability(wire: u16) -> Option<ElementCap> {
    // B13: wire 61 shares the profile, emitters and fault classes; Experimental only.
    if wire == 61 {
        return Some(ElementCap { class: Class::Searchable { since: Grammar::RowVectorV2, tier: Tier::Experimental }, fault_classes: READ.union(FaultSet::MAGNITUDE_CLAMPED).union(FaultSet::NONFINITE_SANITIZED), emitters: EmitterSupport { rust_recipe: YES, python_recipe: YES, cpp17_recipe: YES, cpp17_fitted: YES, javascript: NO_JS }, portable_twin: true, notes: "wire61 goertzel_bank, FeatureWithinRow only, prior<=0.25, 1..=32 frozen frequencies in [0,0.5], output width = bank length, input width<=512; Baseline requires equal-budget promotion" });
    }
    if wire != 58 {
        return None;
    }
    Some(ElementCap { class: Class::Searchable { since: Grammar::RowVectorV2, tier: Tier::Experimental }, fault_classes: READ.union(FaultSet::MAGNITUDE_CLAMPED).union(FaultSet::NONFINITE_SANITIZED), emitters: EmitterSupport { rust_recipe: YES, python_recipe: YES, cpp17_recipe: YES, cpp17_fitted: YES, javascript: NO_JS }, portable_twin: true, notes: "wire58, FeatureWithinRow only, prior<=0.25, deterministic preserve width, shrink boundary, stage-local BOUNDARY_DEGREE bit18; Baseline requires equal-budget promotion" })
}
