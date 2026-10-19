use jope_pipeline::pathway::{
    plan, validate, Mode, PathwayRequest, PlanError, ThinkingLevel, REQUEST_SCHEMA,
};

fn request(mode: Mode, budget: u64) -> PathwayRequest {
    PathwayRequest {
        schema: REQUEST_SCHEMA.into(),
        release_id: "native-policy-lm-282-r1".into(),
        dataset_identity: "dataset-000c7ae796cb2672".into(),
        fit_scope: "legal_rows".into(),
        mode,
        targets: vec!["y".into()],
        seed: 7,
        budget,
    }
}

#[test]
fn thinking_caps_are_the_exact_whole_request_budgets() {
    assert_eq!(ThinkingLevel::Low.cap(), 64);
    assert_eq!(ThinkingLevel::Medium.cap(), 256);
    assert_eq!(ThinkingLevel::High.cap(), 1024);
    assert_eq!(ThinkingLevel::Ultra.cap(), 4096);
}

#[test]
fn thinking_plan_accepts_a_budget_within_its_cap_and_keeps_the_request() {
    let r = request(Mode::Thinking(ThinkingLevel::Medium), 256);
    let p = plan(&r).unwrap();
    assert_eq!(p.level, ThinkingLevel::Medium);
    assert_eq!(p.evaluation_budget, 256);
    assert_eq!(p.seed, 7);
    assert_eq!(p.targets, vec!["y".to_string()]);
}

#[test]
fn thinking_plan_refuses_a_budget_above_its_cap() {
    let err = plan(&request(Mode::Thinking(ThinkingLevel::Low), 65)).unwrap_err();
    assert_eq!(
        err,
        PlanError::BudgetExceeded {
            requested: 65,
            cap: 64
        }
    );
}

#[test]
fn direct_is_refused_without_inference_and_names_no_fallback() {
    let err = plan(&request(Mode::Direct, 1)).unwrap_err();
    assert!(matches!(err, PlanError::InferenceUnavailable { .. }));
    let text = err.to_string();
    assert!(text.contains("no fallback is substituted"), "{text}");
    assert!(!text.to_lowercase().contains("ridge"), "{text}");
}

#[test]
fn request_round_trips_through_json_and_refuses_unknown_fields() {
    let r = request(Mode::Thinking(ThinkingLevel::High), 1024);
    let text = serde_json::to_string(&r).unwrap();
    assert!(text.contains(r#""mode":{"thinking":"high"}"#), "{text}");
    let back: PathwayRequest = serde_json::from_str(&text).unwrap();
    assert_eq!(back, r);
    let extra = text.replacen('{', r#"{"unexpected":1,"#, 1);
    assert!(serde_json::from_str::<PathwayRequest>(&extra).is_err());
}

#[test]
fn malformed_requests_are_refused_before_any_plan() {
    let mut wrong_schema = request(Mode::Direct, 1);
    wrong_schema.schema = "jope.other/v1".into();
    assert!(validate(&wrong_schema).is_err());

    let mut no_release = request(Mode::Thinking(ThinkingLevel::Low), 1);
    no_release.release_id = "  ".into();
    assert!(validate(&no_release).is_err());

    let mut no_targets = request(Mode::Thinking(ThinkingLevel::Low), 1);
    no_targets.targets.clear();
    assert!(validate(&no_targets).is_err());

    let mut repeated = request(Mode::Thinking(ThinkingLevel::Low), 1);
    repeated.targets = vec!["y".into(), "y".into()];
    assert!(validate(&repeated).is_err());

    let zero = request(Mode::Thinking(ThinkingLevel::Low), 0);
    assert!(validate(&zero).is_err());
}
