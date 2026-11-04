use super::*;

#[test]
fn legacy_birth_and_expansion_domains_match_draws() {
    use feat_math::{evolve, GenomeIr};
    let v1 = profile::profile("legacy-wrappers-v1", 1).unwrap();
    let v2 = profile::profile("legacy-wrappers-v2", 1).unwrap();
    let mut birth_seen = BTreeMap::new();
    for seed in 0..20_000 {
        let text = evolve::random_canonical_pipeline(seed, 6);
        let GenomeIr::Pipeline(pipe) = GenomeIr::parse(&text).unwrap() else {
            panic!()
        };
        for stage in &pipe.stages {
            check_birth_stage(stage, &v1.domains, &mut birth_seen);
        }
    }
    assert_domain_witnesses(&v1.domains, &birth_seen);
    let mut expansion_seen = BTreeMap::new();
    for seed in 0..50_000 {
        let text =
            evolve::mutate_canonical_pipeline_v2("(pipe X (standardize) (ridge))", seed, 6, true)
                .unwrap();
        let GenomeIr::Pipeline(pipe) = GenomeIr::parse(&text).unwrap() else {
            panic!()
        };
        for stage in &pipe.stages {
            if feat_math::evolve::EXPANSION_TRANSFORMS.contains(&stage.op.as_str()) {
                check_birth_stage(stage, &v2.domains, &mut expansion_seen);
            }
        }
    }
    let expansion: BTreeMap<_, _> = v2
        .domains
        .iter()
        .filter(|(name, _)| feat_math::evolve::EXPANSION_TRANSFORMS.contains(name))
        .map(|(name, row)| (*name, row.clone()))
        .collect();
    assert_domain_witnesses(&expansion, &expansion_seen);
}

#[test]
fn legacy_redraw_domains_match_empty_parameter_edits() {
    use feat_math::{evolve, GenomeIr};
    let v1 = profile::profile("legacy-wrappers-v1", 1).unwrap();
    let targets = [
        "landmark_dist",
        "nystrom",
        "svd_project",
        "gbm",
        "ensemble_ridge_gbm",
        "impute",
        "symlog",
        "select_corr",
        "pair_ratio",
        "spline",
    ];
    for name in targets {
        let parent = if matches!(name, "gbm" | "ensemble_ridge_gbm") {
            format!("(pipe X (standardize) ({name}))")
        } else {
            format!("(pipe X ({name}) (ridge))")
        };
        let mut seen = BTreeMap::new();
        let row = v1.redraw_domains.get(name).unwrap();
        let only: BTreeMap<_, _> = [(name, row.clone())].into();
        for seed in 0..50_000 {
            let text = evolve::mutate_canonical_pipeline(&parent, seed, 6).unwrap();
            let GenomeIr::Pipeline(pipe) = GenomeIr::parse(&text).unwrap() else {
                panic!()
            };
            for stage in &pipe.stages {
                if stage.op == name && !stage.params.is_empty() {
                    check_birth_stage(stage, &only, &mut seen);
                }
            }
        }
        assert_domain_witnesses(&only, &seen);
    }
    let mut lag = BTreeSet::new();
    let mut krr = BTreeSet::new();
    for seed in 0..10_000 {
        let text = evolve::mutate_canonical_pipeline("(pipe X (lag) (ridge))", seed, 6).unwrap();
        let GenomeIr::Pipeline(pipe) = GenomeIr::parse(&text).unwrap() else {
            panic!()
        };
        for stage in pipe.stages.iter().filter(|stage| stage.op == "lag") {
            lag.insert(stage.params.get("append").map_or("absent", |_| "true"));
        }
        let text =
            evolve::mutate_canonical_pipeline("(pipe X (standardize) (krr))", seed, 6).unwrap();
        let GenomeIr::Pipeline(pipe) = GenomeIr::parse(&text).unwrap() else {
            panic!()
        };
        for stage in pipe.stages.iter().filter(|stage| stage.op == "krr") {
            krr.insert(stage.params.is_empty());
        }
    }
    assert_eq!(lag, BTreeSet::from(["absent", "true"]));
    assert!(krr.contains(&true));
}

#[test]
fn legacy_expression_edit_and_growers_match_declared_elements() {
    use feat_math::math_expr::ExpressionScope;
    use feat_math::rng::Pcg32;
    let manifest = Manifest::current();
    let edit = manifest
        .legacy_generators
        .iter()
        .find(|row| row.id == "legacy-expr-edit-v1")
        .unwrap();
    let mut introduced = BTreeSet::new();
    for parent in ["c0", "row_mean", "x", "0.5"] {
        let parsed = MathExpr::parse(parent).unwrap();
        let original = expression_elements(&parsed);
        for seed in 0..10_000 {
            let child = parsed
                .mutate_v1(
                    seed,
                    ExpressionScope {
                        columns: 3,
                        elementwise: true,
                        max_nodes: 64,
                        max_depth: 16,
                    },
                )
                .unwrap();
            introduced.extend(
                expression_elements(&child.expression)
                    .difference(&original)
                    .copied(),
            );
        }
    }
    let declared: BTreeSet<_> = edit
        .functions
        .iter()
        .map(|&name| ("function", name))
        .chain(edit.binops.iter().map(|&name| ("binary", name)))
        .chain(edit.variables.iter().map(|&name| ("variable", name)))
        .collect();
    assert_eq!(introduced, declared);
    assert!(!introduced.contains(&("unary", "neg")));

    let grower = manifest
        .legacy_generators
        .iter()
        .find(|row| row.id == "legacy-full-grower")
        .unwrap();
    let mut runs = Vec::new();
    for mode in 0..3 {
        let mut found = BTreeSet::new();
        for seed in 0..10_000 {
            let mut rng = Pcg32::new(seed, 0);
            let source = match mode {
                0 => full_rust::random_expression(&mut rng, 5),
                1 => full_rust::random_expression_expanded(&mut rng, 5, false),
                _ => full_rust::random_expression_expanded(&mut rng, 5, true),
            };
            found.extend(expression_elements(&MathExpr::parse(&source).unwrap()));
        }
        runs.push(found);
    }
    let expected_base_funcs: BTreeSet<_> = [
        "abs", "sqrt", "log1p", "sin", "cos", "tanh", "sigmoid", "min", "max", "step", "clip",
    ]
    .into_iter()
    .map(|name| ("function", name))
    .collect();
    let expected_all_funcs: BTreeSet<_> = grower
        .functions
        .iter()
        .map(|&name| ("function", name))
        .collect();
    let binops: BTreeSet<_> = grower.binops.iter().map(|&name| ("binary", name)).collect();
    let base_vars: BTreeSet<_> = [
        "col_under_8",
        "row_mean",
        "row_sum",
        "row_l1",
        "row_l2",
        "row_min",
        "row_max",
        "row_std",
    ]
    .into_iter()
    .map(|name| ("variable", name))
    .collect();
    let mut expected_base = expected_base_funcs;
    expected_base.extend(binops.iter().copied());
    expected_base.extend(base_vars.iter().copied());
    expected_base.insert(("unary", "neg"));
    assert_eq!(runs[0], expected_base);
    let mut expected_expanded = expected_all_funcs;
    expected_expanded.extend(binops);
    expected_expanded.extend(base_vars);
    expected_expanded.insert(("unary", "neg"));
    assert_eq!(runs[1], expected_expanded);
    expected_expanded.insert(("variable", "x"));
    assert_eq!(runs[2], expected_expanded);
}

#[test]
fn legacy_slot_and_v3_edits_match_declared_stage_names() {
    use feat_math::kernel::pipeline::PipelineSlots;
    use feat_math::{initial_population, mutate_candidate, GenomeIr, V3EvolutionConfig};
    let manifest = Manifest::current();
    let slot = manifest
        .legacy_generators
        .iter()
        .find(|row| row.id == "legacy-slot-insert-v1")
        .unwrap();
    let GenomeIr::Pipeline(parent) = GenomeIr::parse("(pipe X (ridge))").unwrap() else {
        panic!()
    };
    let parent = PipelineSlots::from_pipeline(&parent).unwrap();
    let mut inserted = BTreeSet::new();
    for seed in 0..1_000 {
        let child = parent.mutate_v1(seed, 32, 3, 6).unwrap();
        assert_eq!(child.transforms.len(), 1);
        inserted.insert(child.transforms[0].op.clone());
    }
    assert_eq!(
        inserted,
        slot.stage_ops.iter().map(|name| name.to_string()).collect()
    );

    let baselines = initial_population(&V3EvolutionConfig::default(), &[]);
    assert_eq!(baselines.len(), 80);
    let mut all = BTreeSet::new();
    let mut newly_inserted = BTreeSet::new();
    for seed in 0..1_000 {
        let parent = &baselines[seed % baselines.len()];
        let child = mutate_candidate(parent, seed as u64, 0, usize::from(parent.island)).unwrap();
        let ops = |candidate: &feat_math::V3Candidate| {
            let GenomeIr::Pipeline(pipe) = GenomeIr::parse(&candidate.syntax).unwrap() else {
                panic!()
            };
            pipe.stages
                .into_iter()
                .map(|stage| stage.op)
                .collect::<BTreeSet<_>>()
        };
        let original = ops(parent);
        let added = ops(&child);
        all.extend(original.iter().cloned());
        all.extend(added.iter().cloned());
        newly_inserted.extend(added.difference(&original).cloned());
    }
    let v3 = manifest
        .legacy_generators
        .iter()
        .find(|row| row.id == "legacy-v3-typed-edit-v1")
        .unwrap();
    assert_eq!(
        all,
        v3.stage_ops.iter().map(|name| name.to_string()).collect()
    );
    assert_eq!(
        newly_inserted,
        BTreeSet::from(["winsor_clip".to_owned(), "standardize".to_owned()])
    );
}

#[test]
fn kernel_profiles_equal_the_manifest_tier_in_both_directions() {
    let manifest = Manifest::current();
    for id in ["kernel2-baseline", "kernel2-experimental"] {
        let profile = profile::profile(id, 1).unwrap();
        let tier = if id == "kernel2-baseline" {
            Tier::Baseline
        } else {
            Tier::Experimental
        };
        let declared: BTreeSet<_> = manifest
            .ops
            .iter()
            .filter(|row| {
                matches!(row.kernel2, Kernel2Use::Baseline)
                    || (tier == Tier::Experimental
                        && matches!(row.kernel2, Kernel2Use::Experimental))
            })
            .map(|row| ("op", row.name))
            .chain(
                manifest
                    .expression
                    .iter()
                    .filter(|row| row.capability.class.admits(tier))
                    .map(|row| ("expr", row.name)),
            )
            .chain([("policy", "constants")])
            .collect();
        let from_profile: BTreeSet<_> = profile
            .elements()
            .iter()
            .map(|element| match element.kind {
                ElementKind::Stage | ElementKind::Head | ElementKind::Carrier => {
                    ("op", element.name)
                }
                ElementKind::Function | ElementKind::Binary | ElementKind::Variable => {
                    ("expr", element.name)
                }
                ElementKind::Policy if element.name == "columns" => ("expr", "col_under_8"),
                ElementKind::Policy if element.name == "negation" => ("expr", "neg"),
                ElementKind::Policy => ("policy", element.name),
            })
            .collect();
        assert_eq!(from_profile, declared, "{id}");
        assert_eq!(from_profile.len(), 51);
    }
}

#[test]
fn legacy_wrapper_body_hashes_are_pinned() {
    use sha2::{Digest, Sha256};
    let names = [
        "rand_params",
        "mutate",
        "mutate_v2",
        "jitter_params",
        "jitter_params_v2",
        "crossover",
    ];
    let hashes: Vec<_> = names
        .into_iter()
        .map(|name| {
            let header = format!("fn {name}(");
            let body = EVOLVE_SOURCE
                .split_once(&header)
                .unwrap()
                .1
                .split_once("\n}\n")
                .unwrap()
                .0;
            let text = format!("{header}{body}\n}}\n");
            let digest = Sha256::digest(text.as_bytes());
            digest
                .iter()
                .map(|byte| format!("{byte:02x}"))
                .collect::<String>()
        })
        .collect();
    assert_eq!(
        hashes,
        [
            "8eff39f344bb80adf94837c38075afcccd0d58689cfb64a760b0f8abb25c4243",
            "7bba18c491c2fa690a9d8a9cb38615b1882311b3b3a700c88c8047ac2bdb1c2e",
            "f3635380f2a0e37c646c3a1056d6b8f407488611ac00a5d7de4cd4f0f54ba0ec",
            "c6d876d79dd9a0749081b363b161c4347a000c9de6ac8cc4618da06d31ba7319",
            "dd149702ef64f1bd77826dc90d8212217e7121ae15af132f144eb2ed32cf5e10",
            "27224666daef770db3186ca0ad37d516f88dea79d0ed95c9b50912e7ab6f9a2a",
        ]
    );
}

#[test]
fn every_expression_element_has_a_codec_id() {
    use feat_math::kernel::Program;
    use feat_math::GenomeIr;
    for (_, name, _) in capabilities::FUNCTIONS {
        let arity = match *name {
            "clip" => 3,
            "min" | "max" | "step" => 2,
            _ => 1,
        };
        let args = ["c0", "c1", "c2"][..arity].join(",");
        let source = format!("(sym raw \"{name}({args})\")");
        let genome = GenomeIr::parse(&source).unwrap();
        assert!(Program::import_legacy(&genome).to_bytes().is_ok(), "{name}");
    }
    for (_, name, _) in capabilities::BINOPS {
        let source = format!("(sym raw \"(c0{name}c1)\")");
        let genome = GenomeIr::parse(&source).unwrap();
        assert!(Program::import_legacy(&genome).to_bytes().is_ok(), "{name}");
    }
    for (_, name, _) in capabilities::VARIABLES {
        let source = if *name == "col_under_8" {
            "c7"
        } else if *name == "col_8_plus" {
            "c8"
        } else {
            name
        };
        let genome = GenomeIr::parse(&format!("(sym raw \"{source}\")")).unwrap();
        assert!(Program::import_legacy(&genome).to_bytes().is_ok(), "{name}");
    }
}

#[test]
fn legacy_edit_arm_names_match_the_declared_draw_table() {
    use feat_math::evolve::CanonicalMutationArm;
    use profile::Arm;
    let variants = [
        CanonicalMutationArm::ParameterJitter,
        CanonicalMutationArm::Insert,
        CanonicalMutationArm::Delete,
        CanonicalMutationArm::AdjacentSwap,
        CanonicalMutationArm::HeadReplacement,
        CanonicalMutationArm::Simplify,
    ];
    let mapped: Vec<_> = variants
        .into_iter()
        .map(|variant| match variant {
            CanonicalMutationArm::ParameterJitter => Arm::ParameterJitter,
            CanonicalMutationArm::Insert => Arm::Insert,
            CanonicalMutationArm::Delete => Arm::Delete,
            CanonicalMutationArm::AdjacentSwap => Arm::AdjacentSwap,
            CanonicalMutationArm::HeadReplacement => Arm::HeadReplacement,
            CanonicalMutationArm::Simplify => Arm::Simplify,
        })
        .collect();
    for id in [
        "legacy-wrappers-v2",
        "kernel2-baseline",
        "kernel2-experimental",
    ] {
        let profile = profile::profile(id, 1).unwrap();
        assert_eq!(
            profile
                .edits
                .drawn
                .iter()
                .map(|row| row.arm)
                .collect::<Vec<_>>(),
            mapped
        );
        assert_eq!(
            profile
                .edits
                .drawn
                .iter()
                .map(|row| row.weight)
                .collect::<Vec<_>>(),
            [2, 1, 1, 1, 1, 1]
        );
    }
    let v1 = profile::profile("legacy-wrappers-v1", 1).unwrap();
    assert_eq!(
        v1.edits.drawn.iter().map(|row| row.arm).collect::<Vec<_>>(),
        mapped[..5]
    );
    assert_eq!(
        v1.edits
            .drawn
            .iter()
            .map(|row| row.weight)
            .collect::<Vec<_>>(),
        [2, 1, 1, 1, 1]
    );
}
