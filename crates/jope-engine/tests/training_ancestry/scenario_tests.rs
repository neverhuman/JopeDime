use super::*;

pub(super) fn retained_numeric_steps_do_not_supply_versioned_committed_credit() {
    use jope_parallel::native_loop::lineage::{LineageBundle, LINEAGE_SCHEMA};
    let s = scenario();
    let g = s.genealogy();
    let anchor = &g.nodes[&digest(&s.a2)];
    let bundle: LineageBundle = serde_json::from_value(json!({"lineage":{
        "schema":LINEAGE_SCHEMA,"anchor":{"checkpoint_sha256":anchor.digest,
        "weights_sha256":anchor.state["model_sha256"],"optimizer_step":anchor.step,
        "objective_sha256":null,"committed_updates":null,
        "architecture_sha256":"a".repeat(64),"vocabulary_sha256":"b".repeat(64),"support_sha256":"c".repeat(64)},
        "protocol":{"objective_sha256":"d".repeat(64),"recipe_sha256":"e".repeat(64),
        "architecture_sha256":"a".repeat(64),"vocabulary_sha256":"b".repeat(64),"support_sha256":"c".repeat(64)},
        "edges":["f".repeat(64)]},"evidence":{}})).unwrap();
    let error = ancestry::verified_lineage_exposure(&g, &bundle).unwrap_err();
    assert!(error.to_string().contains("missing lineage evidence"));
    // The retained genealogy still reports its original step independently.
    assert_eq!(g.nodes[&anchor.digest].step, anchor.step);
}

pub(super) fn the_genealogy_resolves_retained_edges_and_checks_independently() {
    let s = scenario();
    let g = s.genealogy();
    let edge = |dir: &Path| &g.edges[&digest(dir)];
    let resolved = |dir: &Path| Link::Resolved {
        digest: digest(dir),
    };
    assert_eq!(g.nodes.len(), 6);
    let a2 = edge(&s.a2);
    assert_eq!((a2.kind, a2.from_step), ("lineage_start", Some(0)));
    assert_eq!(
        (a2.weights.clone(), a2.optimizer_state),
        (Link::None, "none")
    );
    assert_eq!(edge(&s.a4).weights, resolved(&s.a2));
    assert_eq!(edge(&s.a4).kind, "same_lineage");
    // The exact resume continues lineage A: same lineage record.
    assert_eq!(edge(&s.a6).weights, resolved(&s.a4));
    assert_eq!(edge(&s.a6).from_step, Some(4));
    let b6 = edge(&s.b6);
    assert_eq!(
        (b6.kind, b6.mode.as_str(), b6.from_step),
        ("lineage_start", "new_corpus_transition", Some(4))
    );
    assert_eq!(
        (b6.weights.clone(), b6.optimizer.clone(), b6.optimizer_state),
        (resolved(&s.a4), resolved(&s.a4), "retained")
    );
    assert_eq!(edge(&s.b8).weights, resolved(&s.b6));
    let c2 = edge(&s.c2);
    assert_eq!(
        (c2.weights.clone(), c2.optimizer.clone(), c2.optimizer_state),
        (resolved(&s.a6), Link::None, "reset")
    );
    assert_eq!(c2.from_step, Some(0));
    // Order-independent and idempotent: shuffled, duplicated ingestion
    // describes the same genealogy.
    let mut nodes = ancestry::load_runs(&s.runs()).unwrap();
    nodes.reverse();
    nodes.extend(ancestry::load_runs(&s.runs()).unwrap());
    assert_eq!(
        ancestry::describe(&ancestry::build(nodes).unwrap()),
        ancestry::describe(&g)
    );
    // Every node digest is the independent SHA-256 of its manifest bytes.
    for dir in [&s.a2, &s.a4, &s.a6, &s.b6, &s.b8, &s.c2] {
        let d = hash(&fs::read(dir.join("state.json")).unwrap());
        assert!(g.nodes[&d].weights_retained && g.nodes[&d].optimizer_retained);
    }
}

pub(super) fn exposure_counts_each_committed_segment_once_and_excludes_siblings() {
    let s = scenario();
    let g = s.genealogy();
    let b8 = digest(&s.b8);
    let c2 = digest(&s.c2);
    let branch = ancestry::exposure(&s.root, &g, &[&b8]).unwrap();
    // B8 ← B6 ← A4 ← A2: four segments of two committed updates each. The
    // lost update after A4 is not committed exposure.
    assert_eq!(branch.segments.len(), 4);
    assert_eq!(branch.possible_presentations, 4 * 2 * GLOBAL as u64);
    assert_eq!(branch.exact.presentations, branch.possible_presentations);
    assert!(branch.possible_complete && branch.exact_complete);
    assert!(branch.unresolved_ancestors.is_empty());
    let expected = journal_records(&s.a_first, 4, CORPUS_A)
        .union(&journal_records(&s.b_journal, 8, CORPUS_B))
        .cloned()
        .collect::<BTreeSet<_>>();
    assert_eq!(branch.unique_committed_records, expected.len() as u64);
    // A union of branches counts the shared ancestors A2 and A4 once.
    let union = ancestry::exposure(&s.root, &g, &[&b8, &c2]).unwrap();
    assert_eq!(union.segments.len(), 6);
    assert_eq!(union.possible_presentations, 6 * 2 * GLOBAL as u64);
    assert_eq!(union.exact.presentations, union.possible_presentations);
    let expected_union = expected
        .union(&journal_records(&s.a_second, 6, CORPUS_A))
        .cloned()
        .collect::<BTreeSet<_>>()
        .union(&journal_records(&s.c_journal, 2, CORPUS_A))
        .cloned()
        .collect::<BTreeSet<_>>();
    assert_eq!(union.unique_committed_records, expected_union.len() as u64);
    // The sibling branch's work never enters B8's ancestry.
    assert!(branch
        .segments
        .iter()
        .all(|seg| seg.child != digest(&s.a6) && seg.child != c2));
    // Availability comes from each segment's own frozen inputs manifest.
    let b6 = branch
        .segments
        .iter()
        .find(|seg| seg.child == digest(&s.b6))
        .unwrap();
    assert_eq!(b6.availability, json!({"run":"b"}));
    assert_eq!(b6.corpus_sha256, CORPUS_B);
}

pub(super) fn unretained_or_journal_less_history_stays_unknown() {
    // A removed middle checkpoint: B's claimed parent is unresolved, and A6
    // now spans a window its journal does not cover.
    let s = scenario();
    let a4 = digest(&s.a4);
    fs::remove_dir_all(&s.a4).unwrap();
    let g = s.genealogy();
    assert_eq!(
        g.edges[&digest(&s.b6)].weights,
        Link::Unresolved {
            digest: Some(a4.clone())
        }
    );
    let b = ancestry::exposure(&s.root, &g, &[&digest(&s.b8)]).unwrap();
    assert_eq!(b.unresolved_ancestors, vec![Some(a4)]);
    assert!(!b.possible_complete && !b.exact_complete);
    // The retained new-corpus segments stay exact; nothing older is assumed.
    assert_eq!(b.segments.len(), 2);
    assert_eq!(b.exact.presentations, 2 * 2 * GLOBAL as u64);
    let a6 = ancestry::exposure(&s.root, &g, &[&digest(&s.a6)]).unwrap();
    let segment = a6
        .segments
        .iter()
        .find(|seg| seg.child == digest(&s.a6))
        .unwrap();
    assert_eq!(segment.from_step, Some(2));
    assert_eq!(segment.possible_presentations, Some(4 * GLOBAL as u64));
    assert_eq!(segment.exact_status, "journal_starts_after_segment");
    assert!(segment.exact.is_none() && !a6.exact_complete && a6.possible_complete);

    // A bound journal that is no longer retained.
    let s = scenario();
    fs::remove_file(&s.a_second).unwrap();
    let g = s.genealogy();
    let a6 = ancestry::exposure(&s.root, &g, &[&digest(&s.a6)]).unwrap();
    let segment = a6
        .segments
        .iter()
        .find(|seg| seg.child == digest(&s.a6))
        .unwrap();
    assert_eq!(segment.exact_status, "journal_not_retained");
    assert!(!a6.exact_complete);

    // A historical checkpoint with no consumption journal: possible only.
    let s = scenario();
    let legacy = manual(
        &s.root,
        "legacy",
        json!({"schema":"parallel.checkpoint.v1","step":3000,"world":8,"global_batch":2048,
            "corpus_sha256":CORPUS_A,"initialization":{"kind":"fresh"}}),
    );
    let g = ancestry::build([ancestry::load_node(&legacy).unwrap()]).unwrap();
    let old = ancestry::exposure(&s.root, &g, &[&digest(&legacy)]).unwrap();
    assert_eq!(old.possible_presentations, 3000 * 2048);
    assert!(old.possible_complete && !old.exact_complete);
    assert_eq!(old.segments[0].exact_status, "no_consumption_journal");
    assert_eq!(old.unique_committed_records, 0);
    assert!(!g.nodes[&digest(&legacy)].weights_retained);

    // A successful new run from a parent that is not retained does not
    // manufacture the missing history.
    let s = scenario();
    let missing = "e".repeat(64);
    let parent = manual(
        &s.root,
        "parent",
        json!({"schema":"parallel.checkpoint.v1","step":60000,"world":1,"global_batch":GLOBAL,
            "corpus_sha256":CORPUS_A,"initialization":{"kind":"fresh"}}),
    );
    let mut child = open(
        &s.root,
        "d",
        1,
        CORPUS_A,
        json!({"kind":"warm_start","state_sha256":missing,"parent_step":60000,
            "exact_same_experiment_resume":false,"continuation_mode":"weight_warm_start"}),
        None,
        None,
        None,
    );
    child.train_to(2);
    let d2 = child.commit();
    let g = ancestry::build(vec![
        ancestry::load_node(&parent).unwrap(),
        ancestry::load_node(&d2).unwrap(),
    ])
    .unwrap();
    let d = ancestry::exposure(&s.root, &g, &[&digest(&d2)]).unwrap();
    assert_eq!(d.unresolved_ancestors, vec![Some(missing)]);
    assert!(!d.exact_complete && !d.possible_complete);
    assert_eq!(d.exact.presentations, 2 * GLOBAL as u64);
    // A manifest with no lineage record is its own orphan of unknown origin.
    let orphan = manual(
        &s.root,
        "orphan",
        json!({"schema":"parallel.checkpoint.v1","step":7,"world":1,"global_batch":GLOBAL,
            "corpus_sha256":CORPUS_A}),
    );
    let g = ancestry::build([ancestry::load_node(&orphan).unwrap()]).unwrap();
    let o = ancestry::exposure(&s.root, &g, &[&digest(&orphan)]).unwrap();
    assert_eq!(o.unresolved_ancestors, vec![None]);
    assert_eq!(o.segments[0].exact_status, "unknown_origin");
    assert!(o.segments[0].possible_presentations.is_none());
}

pub(super) fn unsupported_continuity_claims_and_altered_evidence_are_refused() {
    let s = scenario();
    let nodes = || ancestry::load_runs(&s.runs()).unwrap();
    let a2: Value = read_json(&s.a2.join("state.json")).unwrap();
    let a4 = digest(&s.a4);
    let with = |name: &str, init: Value| {
        let mut state = a2.clone();
        state["step"] = json!(5);
        state["initialization"] = init;
        ancestry::load_node(&manual(&s.root, name, state)).unwrap()
    };
    let cases = vec![
        ("contradictory step", {
            let mut state = a2.clone();
            state["validation"] = json!({"sequence_nll":1.0});
            ancestry::load_node(&manual(&s.root, "twin", state)).unwrap()
        }),
        (
            "wrong parent step",
            with(
                "wrong-step",
                json!({"kind":"warm_start","state_sha256":a4,"parent_step":3}),
            ),
        ),
        (
            "wrong optimizer step",
            with(
                "wrong-optimizer",
                json!({"kind":"warm_start","state_sha256":a4,"parent_step":4,
                    "optimizer_parent_step":5}),
            ),
        ),
        (
            "continuity contradicts parent",
            with(
                "wrong-continuity",
                json!({"kind":"warm_start","state_sha256":a4,"parent_step":4,
                    "continuity":{"parent_step":4,"objective":{"parent_sha256":"f".repeat(64)},
                        "corpus":{"parent":CORPUS_A}}}),
            ),
        ),
        (
            "lineage claims exact resume",
            with(
                "resume-claim",
                json!({"kind":"fresh","exact_same_experiment_resume":true}),
            ),
        ),
    ];
    for (case, node) in cases {
        let mut all = nodes();
        all.push(node);
        assert!(ancestry::build(all).is_err(), "{case}: accepted");
    }
    // Positive control: a consistent claim on the same retained parent.
    let mut all = nodes();
    all.push(with(
        "consistent",
        json!({"kind":"warm_start","state_sha256":a4,"parent_step":4,
            "continuity":{"parent_step":4,"objective":{"parent_sha256":objective::identity_sha256()},
                "corpus":{"parent":CORPUS_A}}}),
    ));
    ancestry::build(all).unwrap();

    // Altered evidence is an error, never an unknown.
    let s = scenario();
    let mut bytes = fs::read(&s.a_first).unwrap();
    bytes[60] ^= 1;
    fs::write(&s.a_first, bytes).unwrap();
    let g = s.genealogy();
    assert!(ancestry::exposure(&s.root, &g, &[&digest(&s.a4)]).is_err());
    // Journal exposure disagreeing with committed steps × batch.
    let s = scenario();
    let mut state: Value = read_json(&s.a2.join("state.json")).unwrap();
    state["global_batch"] = json!(16);
    let altered = manual(&s.root, "batch", state);
    let g = ancestry::build([ancestry::load_node(&altered).unwrap()]).unwrap();
    let error = ancestry::exposure(&s.root, &g, &[&digest(&altered)]).unwrap_err();
    assert!(error.to_string().contains("disagrees"), "{error}");
    // Unknown targets are refused.
    let g = s.genealogy();
    assert!(ancestry::exposure(&s.root, &g, &[&"0".repeat(64)]).is_err());
}
