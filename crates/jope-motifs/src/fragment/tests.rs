use super::*;
use crate::fixture;
use crate::mine::mine;
use crate::table::MineParameters;

fn stages(text: &str) -> Vec<Stage> {
    expansion_stages(text).expect("stages parse")
}

fn classify_text(text: &str, task: TaskLabel) -> Classified {
    classify(&fixture::qualified(text, task, "d1", "r1")).expect("classifies")
}

#[test]
fn roster_recipe_is_a_host_with_an_empty_body() {
    for (task, heads) in fixture::roster_heads() {
        for head in heads {
            let classified = classify_text(&fixture::roster(head), task);
            assert_eq!(classified.class, HostClass::Used, "{head} for {task:?}");
            assert!(classified.body.is_empty());
            assert_eq!(classified.body_start, 1);
        }
    }
    let input = input_stage().expect("input stage");
    assert_eq!(
        input,
        "(input categorical=onehot max_onehot=256 min_count=1 missing=indicator numeric=median)"
    );
}

#[test]
fn roster_recipe_with_a_transform_body_is_mined_not_skipped() {
    let text = fixture::v3("(standardize) (cheb degree=2)", "tab_boost");
    let classified = classify_text(&text, TaskLabel::Regression);
    assert_eq!(classified.class, HostClass::Used);
    assert_eq!(
        stage_texts(&classified.body),
        ["(standardize)", "(cheb degree=2)"]
    );
    let set = fixture::set_of(vec![
        fixture::qualified(&text, TaskLabel::Regression, "d1", "r1"),
        fixture::qualified(&text, TaskLabel::Regression, "d2", "r2"),
    ]);
    let table = mine(&set, None, MineParameters::default()).expect("mines");
    assert_eq!(table.body.motifs.len(), 1);
    assert_eq!(
        table.body.motifs[0].expansion,
        "(standardize) (cheb degree=2)"
    );
}

#[test]
fn transform_body_excludes_target_input_and_learner() {
    let text = fixture::canonical(&format!(
        "(pipe X (target kind=log1p) {INPUT_STAGE} (standardize) (cheb degree=2) (ridge))"
    ));
    let classified = classify_text(&text, TaskLabel::Regression);
    assert_eq!(classified.class, HostClass::Used);
    assert_eq!(classified.body_start, 2);
    assert_eq!(
        stage_texts(&classified.body),
        ["(standardize)", "(cheb degree=2)"]
    );
    let legacy = fixture::canonical("(pipe X (standardize) (cheb degree=2) (ridge))");
    let classified = classify_text(&legacy, TaskLabel::Regression);
    assert_eq!(classified.class, HostClass::Used);
    assert_eq!(classified.body_start, 0);
    assert_eq!(classified.body.len(), 2);
}

#[test]
fn runs_break_at_expression_order_dependent_and_unregistered_stages() {
    let body = stages(
        "(standardize) (cheb degree=2) (map expr=\"sin(x)\") (rank) (sigmoid) \
             (symlog) (blom) (rank) (maxabs_scale) (standardize) (lag k=1) (cheb degree=2) \
             (rank) (derive expr=\"c0\") (roll k=2) (robust_scale)",
    );
    assert_eq!(runs(&body), [0..2, 3..5, 6..8, 9..10, 11..13, 15..16]);
    for op in ["map", "derive", "lag", "roll", "symlog", "maxabs_scale"] {
        assert!(!stage_eligible(&Stage::new(op)), "{op}");
    }
    for op in [
        "standardize",
        "cheb",
        "pair_prod",
        "target_bin_encode",
        "sigmoid",
    ] {
        assert!(stage_eligible(&Stage::new(op)), "{op}");
    }
    // Quarantined rows are never eligible.
    assert!(!stage_eligible(&Stage::new("rff_orthogonal")));
    assert_eq!(windows(&(0..2), 6), [(0, 2)]);
    assert!(windows(&(9..10), 6).is_empty());
    assert_eq!(
        windows(&(0..4), 3),
        [(0, 2), (1, 2), (2, 2), (0, 3), (1, 3)]
    );

    // Under ordered rows the grammar admits a lag host; its run still breaks at the lag.
    let ordered = DataSchema {
        row_semantics: RowSemantics::Ordered {
            group_by: Vec::new(),
            order_by: vec![String::from("c0")],
        },
        ..classification_schema()
    };
    let lagged = fixture::v3(
        "(standardize) (cheb degree=2) (lag k=1) (rank) (sigmoid)",
        "ridge",
    );
    let ir = GenomeIr::parse(&lagged).expect("parses");
    let spec = TaskLabel::Regression.task_spec().expect("spec");
    assert!(validate_v3_genome(&ir, &spec, &ordered).is_ok());
    assert!(validate_v3_genome(&ir, &spec, &classification_schema()).is_err());
    let GenomeIr::Pipeline(expr) = &ir else {
        panic!("a pipeline")
    };
    let slots = PipelineSlots::from_pipeline(expr).expect("slots");
    assert_eq!(runs(&slots.transforms), [0..2, 3..5]);
}

#[test]
fn unslotted_programs_are_skipped_not_mined() {
    for body in ["(rff_orthogonal n=8) (standardize)", "(bag) (standardize)"] {
        let classified = classify_text(&fixture::v3(body, "ridge"), TaskLabel::Regression);
        assert_eq!(classified.class, HostClass::SkipSlots, "{body}");
    }
    let middle = fixture::canonical(&format!(
        "(pipe X {INPUT_STAGE} (ridge) (standardize) (ridge))"
    ));
    let classified = classify_text(&middle, TaskLabel::Regression);
    assert_eq!(classified.class, HostClass::SkipSlots);
}

#[test]
fn non_pipeline_programs_are_skipped() {
    for text in [fixture::SYMBOLIC, fixture::PORTFOLIO] {
        let classified = classify_text(text, TaskLabel::Regression);
        assert_eq!(classified.class, HostClass::SkipForm, "{text}");
    }
}

#[test]
fn reserved_prefix_anywhere_skips_the_host() {
    let parameter = fixture::v3(
        "(cheb append=true degree=3 tag=xmotif_) (pair_prod append=true topk=24)",
        "ridge",
    );
    let reference = fixture::v3("(motif_0123456789abcdef) (standardize)", "ridge");
    for text in [parameter, reference] {
        let classified = classify_text(&text, TaskLabel::Regression);
        assert_eq!(classified.class, HostClass::SkipReserved, "{text}");
    }
}

#[test]
fn window_text_is_an_exact_substring_of_the_canonical_host() {
    let mut checked = 0;
    for program in fixture::lines() {
        let classified = classify(&program).expect("classifies");
        if classified.class != HostClass::Used {
            continue;
        }
        let host_stages = match GenomeIr::parse(program.program()).expect("parses") {
            GenomeIr::Pipeline(expr) => expr.stages,
            _ => panic!("a used host is a pipeline"),
        };
        for run in runs(&classified.body) {
            for (start, length) in windows(&run, 6) {
                let window = &classified.body[start..start + length];
                let key = join(window);
                assert!(program.program().contains(&key), "{key}");
                let absolute = classified.body_start + start;
                assert_eq!(
                    stage_texts(&host_stages[absolute..absolute + length]),
                    stage_texts(window)
                );
                checked += 1;
            }
        }
    }
    assert!(checked > 0);
}

#[test]
fn signature_is_registry_derived() {
    let body = stages(fixture::E1);
    let table = fixture::table();
    assert_eq!(table.body.motifs[0].expansion, fixture::E1);
    let occurrences = &table.body.motifs[0].provenance;
    let signature = signature(&body, occurrences).expect("signature");
    assert_eq!(signature.slot, "transform_run");
    assert_eq!(signature.stages, 2);
    assert_eq!(signature.ops, ["cheb", "pair_prod"]);
    assert_eq!(signature.semantic_keys, ["cheb", "pair_prod"]);
    assert!(signature.fitted);
    assert!(signature.requires_labels);
    assert!(signature.tags.iter().any(|tag| tag == "target_aware"));
    assert_eq!(
        signature.tags,
        ["basis", "expansion", "interaction", "target_aware"]
    );
    assert_eq!(signature.appends, [true, true]);
    assert_eq!(
        signature.declared_tasks,
        TaskMask {
            regression: true,
            binary: true,
            multiclass: false,
        }
    );
    assert_eq!(
        signature.observed_tasks,
        [
            TaskLabel::Regression,
            TaskLabel::Binary,
            TaskLabel::Multiclass
        ]
    );
    assert!(signature.all_registry_generatable);
    let unregistered = stages("(symlog) (standardize)");
    assert!(super::signature(&unregistered, occurrences).is_err());
}

#[test]
fn signature_round_trips_through_json_without_registry_types() {
    let body = stages(fixture::E1);
    let signature = signature(&body, &[]).expect("signature");
    assert_eq!(
        signature.shapes,
        [r#"{"multiply":"degree"}"#, r#"{"top_pairs":"topk"}"#]
    );
    assert!(signature.observed_tasks.is_empty());
    let text = serde_json::to_string(&signature).expect("serialize");
    let back: SignatureV1 = serde_json::from_str(&text).expect("deserialize");
    assert_eq!(back, signature);
    let preserve = super::signature(&stages("(standardize) (sigmoid)"), &[]).expect("sig");
    assert_eq!(preserve.shapes, [r#""preserve""#, r#""preserve""#]);
    assert!(!preserve.requires_labels);
    assert_eq!(preserve.appends, [false, false]);
}
