use super::*;

#[test]
fn the_validation_text_and_new_records_pass_the_lint() {
    let lint = |text: &str| {
        let hits = feat_math::transpile::lint_shipped_text(text);
        assert!(hits.is_empty(), "{text}: {hits:?}");
    };
    let refusal = lm_selection()
        .validate(search::Refinement {
            top_k: 1,
            trials_per_program: 2,
        })
        .unwrap_err();
    lint(&format!("{refusal:#}"));
    let records = vec![
        StageTrace::Failed { stage: 0 },
        StageTrace::Stopped { stage: 1 },
    ];
    lint(&serde_json::to_string(&records).unwrap());
    for (program, c, trials) in [(TIE, cos_data(), 60), (START_FAULT, sine_data(), 60)] {
        let (result, _, _) = refine(&lm(), program, &c, trials, trials);
        lint(&serde_json::to_string(&result.refinement).unwrap());
    }
}

#[test]
fn the_default_route_keeps_the_pre_wiring_results() {
    let e = evaluator();
    let train_loss = |program: &str, c: &Cut| -> Option<u64> {
        let model = feat_math::fit(&GenomeIr::parse(program).ok()?, &c.train_x, &c.train_y).ok()?;
        let p = model.predict(&c.train_x).ok()?;
        let mse = p
            .iter()
            .zip(&c.train_y)
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            / p.len() as f64;
        Some(mse.to_bits())
    };
    let mut lines = String::new();
    let mut rows = Vec::new();
    for (name, c) in [("sine", sine_data()), ("line", data())] {
        for program in [SINE, MAP, MIXED, FAULTING, TWO] {
            for (trials, limit) in [(3usize, 10usize), (7, 7), (20, 20), (60, 60)] {
                let (result, used, asked) = refine(&e, program, &c, trials, limit);
                let loss = train_loss(&result.refinement.program, &c);
                lines.push_str(&format!(
                    "UNIT\t{name}\t{program}\t{trials}\t{limit}\t{}\t{used}\t{asked}\t{loss:?}\n",
                    serde_json::to_string(&result).unwrap()
                ));
                rows.push(((name, program, trials), (result.refinement, loss)));
            }
        }
    }
    assert_eq!(
        crate::sha256_hex(lines.as_bytes()),
        "027b30765712af8811e5ffe237d837970e8e698ba97f6734cd38065f84b6aae3"
    );
    // Four of those rows in full: program text, evaluations, rejected trials, failure and the
    // training loss bits of the result.
    let row = |name: &str, program: &str, trials: usize| {
        rows.iter()
            .find(|((n, p, t), _)| *n == name && *p == program && *t == trials)
            .map(|(_, r)| r.clone())
            .unwrap()
    };
    for ((name, program, trials), text, counts, failure, bits) in [
        (
            ("sine", SINE, 60),
            "(pipe X (derive append=false expr=\"sin((1.7000007629394531*c0))\") (ridge lambda_frac=0.000000001))",
            (60, 0),
            None,
            4482624987248756777u64,
        ),
        (
            ("sine", MIXED, 20),
            "(pipe X (map expr=\"sin((1.71875*x))\") (derive append=false expr=\"sin(0.7*c0)\") (ridge))",
            (20, 0),
            None,
            4571347593955826666,
        ),
        (
            ("sine", FAULTING, 60),
            "(pipe X (derive append=false expr=\"sqrt(c0-0.5)\") (ridge))",
            (1, 1),
            Some("faulting constant trial"),
            4612781733306906755,
        ),
        (
            ("line", TWO, 60),
            "(pipe X (derive append=false expr=\"(sin((0.5*c0))+(14.3*c0))\") (ridge))",
            (60, 15),
            None,
            4568963289968063070,
        ),
    ] {
        let (refinement, loss) = row(name, program, trials);
        assert_eq!(refinement.program, text);
        assert_eq!(
            (refinement.evaluations, refinement.rejected_trials),
            counts,
            "{text}"
        );
        assert_eq!(refinement.failure.as_deref(), failure, "{text}");
        assert!(refinement.trace.is_none());
        assert_eq!(loss, Some(bits), "{text}");
    }
    // The generation hook, children and their measurements included.
    let c = sine_data();
    let parents = vec![
        measured(&e, SINE, &c),
        measured(&e, MIXED, &c),
        measured(&e, TWO, &c),
    ];
    for (limit, digest, used) in [
        (
            12,
            "cfff0c3a86da51ced4e20eb771c70c68c07a778a9292d1ee1981c71f74159b01",
            11,
        ),
        (
            40,
            "15ace454ac4038af8cfe954e2c719cfa12fa1fcf65e554e7317f9a26ee956700",
            33,
        ),
        (
            200,
            "15ace454ac4038af8cfe954e2c719cfa12fa1fcf65e554e7317f9a26ee956700",
            33,
        ),
    ] {
        let mut budget = EvaluationBudget::new(limit);
        let result = e
            .finish_generation_guarded(
                &parents,
                &c,
                4,
                3,
                10,
                Clock::Frozen,
                &mut budget,
                &mut || true,
            )
            .unwrap();
        let bytes = serde_json::to_vec(&result).unwrap();
        assert_eq!(
            (crate::sha256_hex(&bytes), budget.used()),
            (digest.to_string(), used),
            "{limit}"
        );
    }
}

#[test]
fn extreme_surrogate_counts_are_refused_without_overflow() {
    fn surrogate(t: &mut RefinementTrace) -> &mut SurrogateTrace {
        match &mut t.stages[1] {
            StageTrace::Surrogate { surrogate, .. } => surrogate,
            _ => unreachable!(),
        }
    }
    for (accepted, jacobians) in [
        (usize::MAX, usize::MAX),
        (usize::MAX - 1, usize::MAX),
        (usize::MAX, usize::MAX - 1),
        (0, usize::MAX),
        (usize::MAX, 0),
    ] {
        let mut trace = synthetic_trace();
        let s = surrogate(&mut trace);
        s.accepted = accepted;
        s.jacobians = jacobians;
        let charged = trace.charged("sites");
        assert!(
            charged.is_err(),
            "accepted {accepted}, jacobians {jacobians}: {charged:?}"
        );
    }
}

#[test]
fn an_untrusted_parent_expression_is_bounded_before_it_is_parsed() {
    use super::refinement::{MAX_PARENT_EXPRESSION_NESTING, SMALL_ALLOWANCE};
    let trace = RefinementTrace {
        method: DAMPED_LS.into(),
        stages: vec![StageTrace::Coordinate {
            stage: 0,
            reason: SMALL_ALLOWANCE.into(),
            evaluations: 5,
            rejected_trials: 0,
        }],
    };
    let parent = |source: &str| {
        format!("(pipe X (derive expr=\"{source}\" append=false) (ridge lambda_frac=0.000000001))")
    };
    let body = "sin(0.5*c0)";
    assert_eq!(parent(body), SINE);
    assert_eq!(trace.check(SINE, "sites", 60).unwrap(), (5, 0));
    // `(` and `*`.
    let room = MAX_PARENT_EXPRESSION_NESTING - 2;
    let plus = |n: usize| format!("{}{body}", "+".repeat(n));
    let grouped = |n: usize| format!("{}{body}{}", "(".repeat(n), ")".repeat(n));
    let product = |n: usize| format!("{body}{}", "*1".repeat(n));
    for source in [plus(room), grouped(room)] {
        assert_eq!(
            trace.check(&parent(&source), "sites", 60).unwrap(),
            (5, 0),
            "{source}"
        );
    }
    for source in [plus(room + 1), grouped(room + 1), product(room + 1)] {
        let error = trace.check(&parent(&source), "sites", 60).unwrap_err();
        assert!(
            format!("{error:#}").contains("nests too deeply"),
            "{source}: {error:#}"
        );
    }
    let closed_form = format!("(sym raw \"{}\")", plus(room + 1));
    let error = trace.check(&closed_form, "sites", 60).unwrap_err();
    assert!(
        format!("{error:#}").contains("not in the pipeline form"),
        "{error:#}"
    );
}
