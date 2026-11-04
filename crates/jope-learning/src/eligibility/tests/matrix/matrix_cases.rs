use super::*;

#[test]
fn authorized_submission_passes_every_route_for_every_task() {
    for task in [Task::Regression, Task::Binary, Task::Multiclass] {
        let f = fixture(task, 5);
        let q = qualify(&f.state, &f.record, &f.evidence).unwrap();
        assert_eq!(revalidate_qualification(&f.state, &f.record).unwrap(), q);
        let report = deliver(&f, &f.record, "");
        assert_eq!((report.accepted, report.rejected), (1, 0));
        assert_eq!(freeze_teaching(&f.state).unwrap().entries, vec![q.clone()]);
        assert_eq!(
            revalidate_replay(&f.state, std::slice::from_ref(&q))
                .unwrap()
                .entries,
            vec![q.clone()]
        );
        let expected = BTreeMap::from([(q.comparison.sha256().unwrap(), q.record_sha256)]);
        assert_eq!(independent_replay(&f.state), expected);
        assert_eq!(manifest(&f.state), expected);
        eprintln!(
            "W3-T07 authorized {}: all five routes admitted",
            task.as_str()
        );
    }
}

#[test]
fn independent_replay_reproduces_the_winner_in_either_arrival_order() {
    for strong_first in [false, true] {
        let f = fixture(Task::Regression, 4);
        let (strong, strong_evidence) = f.candidate("strong", &exact_on(5));
        let (unqualified, _) = f.candidate("unqualified", &exact_on(5));
        qualify(&f.state, &f.record, &f.evidence).unwrap();
        let stronger = qualify(&f.state, &strong, &strong_evidence).unwrap();
        let mut order = vec![(&f.record, ""), (&strong, "strong")];
        if strong_first {
            order.reverse();
        }
        order.insert(1, (&unqualified, "unqualified"));
        for (record, tag) in order {
            deliver(&f, record, tag);
        }
        let expected = BTreeMap::from([(
            stronger.comparison.sha256().unwrap(),
            stronger.record_sha256.clone(),
        )]);
        assert_eq!(independent_replay(&f.state), expected);
        assert_eq!(manifest(&f.state), expected);
        let verified = hwm::verify(&f.state, &[]).unwrap();
        assert!(verified.ok(), "{:?}", verified.breaks);
        assert_eq!(verified.initialized, 1);
    }
}
