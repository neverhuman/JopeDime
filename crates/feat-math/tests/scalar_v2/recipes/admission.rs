#[test]
fn scalar_recipe_admission_refuses_bad_scope_shape_and_labels() {
    let variable = ScalarExpressionProgram::new(ScalarNode::Var(Var::X)).unwrap();
    assert!(variable.plan_training(ScalarCarrier::Row, 2, 0.02).is_err());
    let p = ScalarExpressionProgram::new(ScalarNode::Call(
        ScalarFunction::from_versioned_id(2, 21).unwrap(),
        vec![ScalarNode::Var(Var::Col(0))],
    ))
    .unwrap();
    for width in [0, 513] {
        assert!(p.plan_training(ScalarCarrier::Row, width, 0.02).is_err());
    }
    for lambda in [f64::NAN, f64::INFINITY, 0.0, 2.001] {
        assert!(p.plan_training(ScalarCarrier::Row, 1, lambda).is_err());
    }
    let source = p
        .emit_recipe(ScalarLanguage::Python, ScalarCarrier::Row, 1, 0.02)
        .unwrap();
    let source = format!("__name__='scalar_admission'\n{source}\nfor action in [lambda:Model().fit([],[]),lambda:Model().fit([[0.0]],[1e13]),lambda:Model().fit([[0.0]],[]),lambda:Model().fit([[0.0,1.0]],[0.0])]:\n    try: action()\n    except ValueError: pass\n    else: raise AssertionError('missing fit refusal')\n");
    run(Command::new("python3").arg("-c").arg(source));
}
#[test]
fn nested_clip_nan_bounds_refuse_before_parent_can_hide_the_failure() {
    let clip = ScalarNode::Call(
        ScalarFunction::from_versioned_id(2, 9).unwrap(),
        vec![
            ScalarNode::Var(Var::Col(0)),
            ScalarNode::Var(Var::Col(1)),
            ScalarNode::Var(Var::Col(2)),
        ],
    );
    let p = ScalarExpressionProgram::new(ScalarNode::Call(
        ScalarFunction::from_versioned_id(2, 21).unwrap(),
        vec![clip],
    ))
    .unwrap();
    let model = p
        .fit_training(
            &vec![vec![0.0, -1.0, 1.0], vec![0.5, -1.0, 1.0]],
            &[0.0, 1.0],
            ScalarCarrier::Row,
            0.02,
        )
        .unwrap();
    assert!(model.predict(&vec![vec![0.0, f64::NAN, f64::NAN]]).is_err());
    let directory = Path::new(env!("CARGO_TARGET_TMPDIR")).join("scalar-v2-clip-refusal");
    fs::create_dir_all(&directory).unwrap();
    for lang in [
        ScalarLanguage::Rust,
        ScalarLanguage::Python,
        ScalarLanguage::Cpp17,
    ] {
        let source = p.emit_recipe(lang, ScalarCarrier::Row, 3, 0.02).unwrap();
        let (source,file) = match lang {
            ScalarLanguage::Rust => (format!("{}\nfn main(){{assert!(std::panic::catch_unwind(||scalar_stage1_values(&vec![vec![0.0,f64::NAN,f64::NAN]])).is_err());}}",source.replace("\nfn main(","\nfn scalar_cli_main(")), "refusal.rs"),
            ScalarLanguage::Python => (format!("__name__='scalar_refusal'\n{source}\ntry: scalar_stage1_values([[0.0,float('nan'),float('nan')]])\nexcept ArithmeticError: pass\nelse: raise AssertionError('hidden clip failure')\n"),"refusal.py"),
            ScalarLanguage::Cpp17 => (format!("{source}\nint main(){{bool refused=false;try{{scalar_stage1_values({{{{0.0,std::numeric_limits<double>::quiet_NaN(),std::numeric_limits<double>::quiet_NaN()}}}});}}catch(const std::domain_error&){{refused=true;}}return refused?0:1;}}"),"refusal.cpp"),
        };
        let path = directory.join(file);
        fs::write(&path, source).unwrap();
        match lang {
            ScalarLanguage::Rust => {
                let binary = directory.join("rust");
                run(Command::new("rustc")
                    .args(["--edition=2021", "-C", "opt-level=2"])
                    .arg(&path)
                    .arg("-o")
                    .arg(&binary));
                run(&mut Command::new(binary));
            }
            ScalarLanguage::Python => {
                run(Command::new("python3").arg(&path));
            }
            ScalarLanguage::Cpp17 => {
                let binary = directory.join("cpp");
                run(Command::new("g++")
                    .args(["-std=c++17", "-O2", "-ffp-contract=off", "-fno-fast-math"])
                    .arg(&path)
                    .arg("-o")
                    .arg(&binary));
                run(&mut Command::new(binary));
            }
        }
    }
}
