use super::*;

/// Re-derive digest and metadata after a semantic mutation, so only the invariant is wrong.
fn rehash(pools: &[Pool], manifest: &mut Manifest) {
    for (pool, entry) in pools.iter().zip(&mut manifest.pools) {
        entry.parent_id = pool.parent_id.clone();
        entry.members = pool.members.len();
        entry.measured = pool
            .members
            .iter()
            .filter(|m| m.outcome().is_some())
            .count();
        entry.pool_sha256 = pool.sha256().unwrap();
    }
}

fn assert_refused(pools: &[Pool], manifest: &Manifest, needle: &str) {
    let inputs = Inputs {
        context_features: None,
        program_features: None,
        external: vec![],
        unavailable: vec![],
    };
    let checks = [
        ("verify", verify(pools, manifest).err()),
        ("views", export_views(pools, manifest).err()),
        ("evaluate", evaluate(pools, manifest, &inputs, &cfg()).err()),
    ];
    for (path, err) in checks {
        let err = err.unwrap_or_else(|| panic!("{path} accepted an artifact violating {needle:?}"));
        assert!(
            format!("{err:#}").contains(needle),
            "{path}: {err:#} lacks {needle:?}"
        );
    }
}

pub(super) fn reloaded_valid_pools_pass_and_rehashed_invalid_pools_are_refused_body() {
    let (pools, manifest) = freeze(&synthetic_rows(6, 18, 12), &rule(), &h("input")).unwrap();
    // Valid freeze -> JSON -> reload path.
    let reload: Vec<Pool> = pools
        .iter()
        .map(|p| serde_json::from_str(&serde_json::to_string(p).unwrap()).unwrap())
        .collect();
    let remanifest: Manifest =
        serde_json::from_str(&serde_json::to_string(&manifest).unwrap()).unwrap();
    verify(&reload, &remanifest).unwrap();
    export_views(&reload, &remanifest).unwrap();
    let ok = Inputs {
        context_features: None,
        program_features: None,
        external: vec![],
        unavailable: vec![],
    };
    evaluate(&reload, &remanifest, &ok, &cfg()).unwrap();

    type Mutation = (&'static str, fn(&mut Vec<Pool>, &mut Manifest));
    let cases: [Mutation; 12] = [
        ("may not enter a shadow pool", |p, _| {
            p[1].role = "sealed".into()
        }),
        ("may not enter a shadow pool", |p, _| {
            p[2].role = "validation".into()
        }),
        ("unknown outcome", |p, _| {
            p[0].members[0].status = Status::Unknown;
            p[0].members[0].q = None;
        }),
        ("q present", |p, _| {
            p[0].members[1].status = Status::ProgramFailed;
            p[0].members[1].q = Some(0.4);
        }),
        ("complete row without q", |p, _| {
            p[0].members[2].status = Status::Complete;
            p[0].members[2].q = None;
        }),
        ("min_pool", |p, _| p[3].members.truncate(4)),
        ("parent_id is required", |p, _| p[0].parent_id = " ".into()),
        ("lowercase sha256", |p, m| {
            p[0].context_id = "0".into();
            m.pools[0].context_id = "0".into();
        }),
        ("incumbent_q must be finite", |p, _| {
            p[0].incumbent_q = Some(f64::NAN)
        }),
        ("distinct and sorted", |p, _| {
            let dup = p[0].members[0].clone();
            p[0].members.insert(1, dup);
        }),
        ("both pooled and excluded", |_, m| {
            m.excluded.push(Exclusion {
                context_id: m.pools[0].context_id.clone(),
                reason: "fixture".into(),
            });
        }),
        ("shadow-only", |_, m| m.qualified_claim = true),
    ];
    for (needle, mutate) in cases {
        let (mut bad, mut bad_manifest) = (pools.clone(), manifest.clone());
        mutate(&mut bad, &mut bad_manifest);
        rehash(&bad, &mut bad_manifest);
        assert_refused(&bad, &bad_manifest, needle);
    }
    // Metadata that disagrees with a correctly hashed pool is refused too.
    let mut stale = manifest.clone();
    stale.pools[0].measured += 1;
    assert_refused(&pools, &stale, "metadata differs");
}
