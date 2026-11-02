use super::*;

#[test]
fn dependencies_are_recorded_and_delivered_first() {
    let e = env();
    let r = run("run-w8-deps", 2, 3);
    let items = producer(&r);
    let mut o = Outbox::open(&e.state, BIG).unwrap();
    let dir = Outbox::dir(&e.state);

    let before = snapshot(&dir);
    for early in [1usize, 4, 7] {
        let err = items[early].append(&mut o).unwrap_err();
        assert!(
            format!("{err:#}").contains("has not recorded"),
            "item {early}: {err:#}"
        );
    }
    assert_eq!(snapshot(&dir), before, "a refused append wrote something");
    assert_eq!(o.records(), 0);

    for item in &items {
        item.append(&mut o).unwrap();
    }
    for (i, item) in items.iter().enumerate() {
        let record = o.record(i as u64 + 1).unwrap();
        assert_eq!(
            record
                .dependencies
                .iter()
                .map(Dependency::key)
                .collect::<BTreeSet<_>>(),
            item.deps().into_iter().collect::<BTreeSet<_>>()
        );
    }
    let faulty = Faulty::new(&e.remote);
    o.deliver(&faulty, None).unwrap();
    let order = faulty.keys("put");
    assert_eq!(order.len(), items.len());
    for item in &items {
        let at = order.iter().position(|k| *k == item.key()).unwrap();
        for dep in item.deps() {
            let dep_at = order.iter().position(|k| *k == dep).unwrap();
            assert!(dep_at < at, "{dep} delivered after {}", item.key());
        }
    }
}
