use super::*;

#[test]
fn prediction_checkpoint_budget_refuses_atomically_in_all_routes() {
    use feat_math::kernel::wire_v2::{VersionError, VersionedModel};
    let p = program(0.125);
    let rows = rows();
    let native = p
        .fit_training(
            &rows[..4],
            &[1., 2., 3., 4.],
            TaskMode::Forecasting,
            FoldBoundary::new("recipe-cut", 4, 8).unwrap(),
        )
        .unwrap();
    let mut public = VersionedModel::from_json(&native.to_json().unwrap()).unwrap();
    let before = native.to_json().unwrap();
    let fold = FoldBoundary::new("recipe-cut", 4, 8).unwrap();
    let mut oversized = TimeGradientState::from_bytes(
        &state(&native),
        TaskMode::Forecasting,
        p.input_identity(),
        fold.clone(),
    )
    .unwrap();
    oversized
        .push(
            GroupKey::categorical("q".repeat(3 * 1024 * 1024)).unwrap(),
            0.5,
            &[1., 2.],
        )
        .unwrap();
    let payload = oversized.to_bytes().unwrap();
    assert!(payload.len() < 16 * 1024 * 1024);
    assert_eq!(
        TimeGradientState::from_bytes(&payload, TaskMode::Forecasting, p.input_identity(), fold,)
            .unwrap()
            .to_bytes()
            .unwrap(),
        payload
    );
    let mut envelope: serde_json::Value = serde_json::from_str(&before).unwrap();
    envelope["state"] = serde_json::to_value(&payload).unwrap();
    let oversized_json = envelope.to_string();
    assert!(oversized_json.len() > 16 * 1024 * 1024);
    assert!(matches!(
        TimeGradientModel::from_json(&oversized_json),
        Err(feat_math::kernel::wire_v2::VersionError::Malformed(message))
            if message == "time-gradient model bytes"
    ));
    for bytes in [9 * 1024 * 1024, 3 * 1024 * 1024] {
        let oversized = OrderedRow {
            group: GroupKey::categorical("q".repeat(bytes)).unwrap(),
            timestamp: 0.5,
            values: vec![1., 2.],
        };
        assert!(matches!(
            public.predict_ordered(&[oversized]),
            Err(VersionError::StageRejected {
                kind: feat_math::kernel::Error::Resource(_),
                faults: 0
            })
        ));
        let VersionedModel::TimeGradientV2(checkpoint) = &public else {
            panic!("wrong public variant")
        };
        assert_eq!(checkpoint.to_json().unwrap(), before);
    }
    let expected = public.predict_ordered(&rows[4..]).unwrap();
    let mut replay = VersionedModel::from_json(&before).unwrap();
    assert_eq!(
        expected.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        replay
            .predict_ordered(&rows[4..])
            .unwrap()
            .iter()
            .map(|v| v.to_bits())
            .collect::<Vec<_>>()
    );
    let dir = tempfile::tempdir().unwrap().keep();
    for lang in [
        TimeGradientLanguage::Rust,
        TimeGradientLanguage::Python,
        TimeGradientLanguage::Cpp17,
    ] {
        let source = p.emit_recipe(lang).unwrap().source;
        let mut driver = match lang {
            TimeGradientLanguage::Rust => rust_driver(&rows, &[1., 2., 3., 4.], &[vec![]]),
            TimeGradientLanguage::Python => python_driver(&rows, &[1., 2., 3., 4.], &[vec![]]),
            TimeGradientLanguage::Cpp17 => cpp_driver(&rows, &[1., 2., 3., 4.], &[vec![]], false),
        };
        let (marker, guard) = match lang {
            TimeGradientLanguage::Rust => (
                "let initial=model.state_bytes().unwrap();",
                r#"for bytes in [9*1024*1024,3*1024*1024] { let bad=OrderedRow{group:OrderedGroup::Text("q".repeat(bytes)),timestamp:0.5,values:vec![1.,2.]};let e=model.predict(&[bad]).unwrap_err();assert_eq!(e.kind,"resource");assert_eq!(e.faults,0);assert_eq!(model.state_bytes().unwrap(),initial); }
let mut oversized=model.state.clone();oversized.push(OrderedGroup::Text("q".repeat(3*1024*1024)),0.5,&[1.,2.]).unwrap();let payload=oversized.to_bytes().unwrap();assert!(payload.len()<16*1024*1024);assert_eq!(model.replace_state(&payload).unwrap_err(),"resource");assert_eq!(model.state_bytes().unwrap(),initial);"#,
            ),
            TimeGradientLanguage::Python => (
                "initial=model.state_bytes()",
                r#"
for size in [9*1024*1024,3*1024*1024]:
 try: model.predict([OrderedRow((1,'q'*size),0.5,[1.,2.])])
 except OrderedError as error: assert str(error)=='resource' and error.faults==0
 else: raise AssertionError('oversized continuation accepted')
 assert model.state_bytes()==initial
oversized=copy.deepcopy(model.state)
oversized.push((1,'q'*(3*1024*1024)),0.5,[1.,2.])
payload=oversized.to_bytes()
assert len(payload)<16*1024*1024
try: model.replace_state(payload)
except OrderedError as error: assert str(error)=='resource' and error.faults==0
else: raise AssertionError('oversized model checkpoint restored')
assert model.state_bytes()==initial
"#,
            ),
            TimeGradientLanguage::Cpp17 => (
                "bool refused=false;",
                r#"for(auto size:{9u*1024u*1024u,3u*1024u*1024u}) { bool stopped=false;try {model.predict({OrderedRow{OrderedGroup::categorical(std::string(size,'q')),0.5,{1.,2.}}});}catch(const OrderedError& e){stopped=true;if(std::string(e.what())!="resource"||e.faults)return 20;}if(!stopped||model.state_bytes()!=initial)return 21;}
auto oversized=model.state;oversized.push(OrderedGroup::categorical(std::string(3u*1024u*1024u,'q')),0.5,{1.,2.});const auto payload=oversized.to_bytes();if(payload.size()>=16u*1024u*1024u)return 24;bool stopped=false;try{model.replace_state(payload);}catch(const OrderedError& e){stopped=true;if(std::string(e.what())!="resource"||e.faults)return 25;}if(!stopped||model.state_bytes()!=initial)return 26;"#,
            ),
        };
        driver = driver.replacen(marker, &format!("{marker}\n{guard}"), 1);
        execute(&dir, &format!("budget-{lang:?}"), &(source + &driver), lang);
    }
    let fitted = native.emit_fitted_cpp17().unwrap();
    let driver = cpp_driver(&rows,&[1.,2.,3.,4.],&[vec![]],true).replacen("bool refused=false;",r#"bool refused=false;for(auto size:{9u*1024u*1024u,3u*1024u*1024u}) {bool stopped=false;try {model.predict({OrderedRow{OrderedGroup::categorical(std::string(size,'q')),0.5,{1.,2.}}});}catch(const OrderedError& e){stopped=true;if(std::string(e.what())!="resource"||e.faults)return 22;}if(!stopped||model.state_bytes()!=initial)return 23;}
auto oversized=model.state;oversized.push(OrderedGroup::categorical(std::string(3u*1024u*1024u,'q')),0.5,{1.,2.});const auto payload=oversized.to_bytes();if(payload.size()>=16u*1024u*1024u)return 27;bool stopped=false;try{model.replace_state(payload);}catch(const OrderedError& e){stopped=true;if(std::string(e.what())!="resource"||e.faults)return 28;}if(!stopped||model.state_bytes()!=initial)return 29;"#,1);
    execute(
        &dir,
        "budget-fitted",
        &(fitted.source + &driver),
        TimeGradientLanguage::Cpp17,
    );
}
