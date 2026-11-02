use super::*;

#[test]
/// The rehashed-evidence refusals on a fixture that has every shape they edit: a surrogate stage
/// with two literal sites and a pipeline decision. Each edit is checked to have changed the
/// evidence before its refusal is checked, and the two records the review found accepted (a
/// fallback with equal counts, Jacobians traded for trials) are among them.
fn rehashed_multi_site_lm_traces_are_refused_with_every_edit_applied() {
    use super::refinement::{ELEMENTWISE, SMALL_ALLOWANCE, UNSUPPORTED};
    let temp = tempfile::tempdir().unwrap();
    let opt = flag_run(&temp.path().join("multi"), MULTI_SITE_SEED, 40, 12, "lm");
    let receipt = run::run(&opt).unwrap();
    let output = &opt.run.output;
    let path = output.join("receipt.json");
    let originals: Vec<(PathBuf, Vec<u8>)> = [
        "receipt.json",
        "search/cut-0/search.json",
        "search/cut-0/refinements.json",
    ]
    .iter()
    .map(|r| (output.join(r), std::fs::read(output.join(r)).unwrap()))
    .collect();
    let restore = || {
        for (file, bytes) in &originals {
            std::fs::write(file, bytes).unwrap();
        }
    };
    let engine = receipt.search.engine_selection.clone();
    let refinements: Value = serde_json::from_slice(&originals[2].1).unwrap();
    rewrite_lm_evidence(output, &receipt, &refinements, engine.clone());
    publication::verify_run(&path).unwrap();
    restore();
    // The fixture's guarantee, checked rather than assumed.
    let pointer = refinements
        .as_array()
        .unwrap()
        .iter()
        .enumerate()
        .find_map(|(r, record)| {
            let stages = record["refinement"]["trace"]["stages"].as_array()?;
            let s = stages.iter().position(|s| {
                s["kind"] == "surrogate"
                    && s["surrogate"]["sites"].as_u64() >= Some(2)
                    && s["pipeline"]["evaluations"] == 2
                    && s["pipeline"]["winner"].is_u64()
            })?;
            Some(format!("/{r}/refinement/trace/stages/{s}"))
        })
        .expect("a multi-site surrogate stage with a pipeline decision");
    let stage = refinements.pointer(&pointer).unwrap().clone();
    let count = |value: &Value| value.as_u64().unwrap();
    let charge = |s: &Value| {
        let surrogate = &s["surrogate"];
        count(&surrogate["initial"])
            + count(&surrogate["trials"])
            + count(&surrogate["jacobian_cost"]) * count(&surrogate["jacobians"])
            + count(&surrogate["rescore"])
            + count(&s["pipeline"]["evaluations"])
    };
    let (evaluations, rejected) = (
        charge(&stage),
        count(&stage["surrogate"]["rejected_trials"])
            + count(&stage["pipeline"]["rejected_trials"]),
    );
    let mut edits: Vec<(String, Value)> = Vec::new();
    let mut edit = |label: String, change: &dyn Fn(&mut Value)| {
        let mut edited = refinements.clone();
        change(edited.pointer_mut(&pointer).unwrap());
        assert_ne!(edited, refinements, "{label}: the edit was not applied");
        edits.push((label, edited));
    };
    for field in [
        "trials",
        "jacobians",
        "initial",
        "rescore",
        "rejected_trials",
        "sites",
        "accepted",
    ] {
        edit(field.into(), &|s| {
            let value = &mut s["surrogate"][field];
            *value = (value.as_u64().unwrap() + 1).into();
        });
    }
    edit("pipeline decision".into(), &|s| {
        let accepted = &mut s["pipeline"]["accepted"];
        *accepted = (!accepted.as_bool().unwrap()).into();
    });
    edit("winner's pass dropped".into(), &|s| {
        let pipeline = s["pipeline"].as_object_mut().unwrap();
        pipeline.remove("winner");
        pipeline.insert("evaluations".into(), 1.into());
        pipeline.insert("accepted".into(), false.into());
    });
    edit("start's loss dropped".into(), &|s| {
        s["pipeline"].as_object_mut().unwrap().remove("start");
    });
    edit("pipeline passes dropped".into(), &|s| {
        s.as_object_mut().unwrap().remove("pipeline");
    });
    // The review's first trigger: the stage replaced by a fallback with equal counts.
    for reason in [ELEMENTWISE, SMALL_ALLOWANCE, UNSUPPORTED] {
        edit(format!("fallback {reason}"), &|s| {
            *s = json!({
                "kind": "coordinate",
                "stage": s["stage"],
                "reason": reason,
                "evaluations": evaluations,
                "rejected_trials": rejected,
            });
        });
    }
    // The review's second trigger at this stage's charge: one more Jacobian, and a Jacobian's
    // worth fewer trials with the steps they took, keep the stage's charge and every refusal.
    edit("jacobians for trials".into(), &|s| {
        let before = charge(s);
        let surrogate = &mut s["surrogate"];
        let cost = count(&surrogate["jacobian_cost"]);
        let iterations = surrogate["iterations"].as_array_mut().unwrap();
        let removed: Vec<Value> = iterations
            .drain(iterations.len() - cost as usize..)
            .collect();
        assert!(removed
            .iter()
            .all(|i| i["solved"] == true && i["trial"].is_u64()));
        let steps = removed.iter().filter(|i| i["accepted"] == true).count() as u64;
        surrogate["accepted"] = (count(&surrogate["accepted"]) - steps).into();
        surrogate["trials"] = (count(&surrogate["trials"]) - cost).into();
        surrogate["jacobians"] = (count(&surrogate["jacobians"]) + 1).into();
        assert_eq!(charge(s), before);
    });
    for (label, edited) in &edits {
        let path = rewrite_lm_evidence(output, &receipt, edited, engine.clone());
        assert!(publication::verify_run(&path).is_err(), "{label}");
        restore();
    }
    // A stage with two sites re-derives a different charge under `one`.
    let mut record = engine.clone().unwrap();
    record.refine.as_mut().unwrap().jacobian = "one".into();
    let path = rewrite_lm_evidence(output, &receipt, &refinements, Some(record));
    assert!(publication::verify_run(&path).is_err(), "jacobian charge");
    restore();
    publication::verify_run(&output.join("receipt.json")).unwrap();
}

#[test]
/// The default runs against the stack before the method was wired: these digests were measured
/// at a8bd39855, the commit before the wiring (fix-01 probe, the same two runs in this lane on
/// that commit's sources). The search sidecars are pinned; emission is not this change's subject.
fn default_runs_keep_the_pre_wiring_search_sidecars() {
    let temp = tempfile::tempdir().unwrap();
    for (opt, pins) in [
        (
            seeded(&temp.path().join("seeded")),
            [
                "8b8dccfe93aff8fe789ebc37dd125145a28b9d9d702b42dea2fa737575008059",
                "b2444c5c34dee80c91aa52d19a4a34aeaa358bf68dd3919723f52463fb2c5420",
                "ad1e6ab3e218f237120668e39321daf2a60cc43b5a503011f1a05ecbbb907767",
                "a6e3af0c1a0b0e85bfe64e3d8bd5c869e6a58b890403ba3438efba144e12a4d7",
            ],
        ),
        (
            options(&temp.path().join("plain"), 16),
            [
                "8b8dccfe93aff8fe789ebc37dd125145a28b9d9d702b42dea2fa737575008059",
                "fb3f1637f853a232ca9d13d7cc5aa9b8d4644704ce46ad121377a3837c7948fb",
                "fa6fd28f1aa37a516a39719ebd49d476624f3e220af92ec2a67b4104dc48bb96",
                "75b7234a70a005c8ba7533ad9ec200dcbe70bb3537e3d97ffe5f75bc449bd3fa",
            ],
        ),
    ] {
        run::run(&opt).unwrap();
        for (relative, pin) in [
            "prepared/prepared.json",
            "search/cut-0/search.json",
            "search/cut-0/observations.json",
            "search/cut-0/refinements.json",
        ]
        .into_iter()
        .zip(pins)
        {
            let bytes = std::fs::read(opt.run.output.join(relative)).unwrap();
            assert_eq!(crate::sha256_hex(&bytes), pin, "{relative}");
        }
    }
}

#[test]
fn v4_schema_matches_its_generated_type_and_legacy_schemas_are_unchanged() {
    assert_eq!(receipt::schema(), receipt::derived_schema());
    assert_eq!(
        receipt::schema()["properties"]["schema"]["const"],
        receipt::SCHEMA_ID
    );
    let legacy = crate::receipt::schema();
    assert_eq!(legacy, crate::receipt::derived_schema());
    assert_eq!(
        legacy["properties"]["schema"]["const"],
        crate::receipt::RECEIPT_SCHEMA_ID
    );
}
