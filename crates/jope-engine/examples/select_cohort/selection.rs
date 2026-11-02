use super::*;

fn tie(scope: &str, value: &str) -> String {
    hash(format!("{SEED}:{scope}:{value}").as_bytes())
}
fn stratum(c: &Candidate) -> (u8, u8) {
    let rows = if c.rows < 256 {
        0
    } else if c.rows < 1024 {
        1
    } else if c.rows < 4096 {
        2
    } else if c.rows < 16384 {
        3
    } else {
        4
    };
    let features = if c.features <= 4 {
        0
    } else if c.features <= 16 {
        1
    } else if c.features <= 64 {
        2
    } else {
        3
    };
    (rows, features)
}
pub(super) fn quotas() -> BTreeMap<String, usize> {
    BTreeMap::from([
        ("meta_train".into(), 48),
        ("meta_validation".into(), 8),
        ("meta_test".into(), 8),
    ])
}

pub(super) fn select(candidates: &[Candidate]) -> Result<Vec<Candidate>> {
    let mut families: BTreeMap<&str, Vec<&Candidate>> = BTreeMap::new();
    let mut ids = BTreeSet::new();
    for c in candidates {
        ensure!(
            c.eligible && c.holds.is_empty(),
            "held or unadmitted candidate: {}",
            c.dataset
        );
        ensure!(ids.insert(&c.dataset), "duplicate candidate dataset");
        ensure!(
            c.rows > 0 && c.rows <= ROW_LIMIT && c.features > 0,
            "invalid candidate dimensions"
        );
        ensure!(
            quotas().contains_key(&c.partition),
            "unknown candidate partition"
        );
        families.entry(&c.family).or_default().push(c);
    }
    let mut representatives = Vec::new();
    for (family, mut versions) in families {
        ensure!(
            versions
                .iter()
                .all(|c| c.partition == versions[0].partition),
            "family crosses partitions"
        );
        versions.sort_by_key(|c| tie(&format!("representative:{family}"), &c.dataset));
        representatives.push(versions[0].clone());
    }
    let mut selected = Vec::new();
    for (partition, quota) in quotas() {
        let eligible = representatives
            .iter()
            .filter(|c| c.partition == partition)
            .count();
        ensure!(
            eligible >= quota,
            "insufficient {partition} families: {eligible}, required {quota}"
        );
        let mut buckets: BTreeMap<(u8, u8), Vec<Candidate>> = BTreeMap::new();
        for c in representatives.iter().filter(|c| c.partition == partition) {
            buckets.entry(stratum(c)).or_default().push((*c).clone());
        }
        let mut buckets = buckets
            .into_iter()
            .map(|(key, mut values)| {
                values.sort_by_key(|c| tie(&format!("family:{partition}"), &c.family));
                (key, VecDeque::from(values))
            })
            .collect::<Vec<_>>();
        buckets.sort_by_key(|(key, _)| {
            tie(
                &format!("stratum:{partition}"),
                &format!("{}:{}", key.0, key.1),
            )
        });
        let mut count = 0;
        while count < quota {
            for (_, bucket) in &mut buckets {
                if let Some(c) = bucket.pop_front() {
                    selected.push(c);
                    count += 1;
                }
                if count == quota {
                    break;
                }
            }
        }
    }
    selected.sort_by(|a, b| (&a.partition, &a.family).cmp(&(&b.partition, &b.family)));
    ensure!(
        selected.len() == 64
            && selected
                .iter()
                .map(|c| &c.family)
                .collect::<BTreeSet<_>>()
                .len()
                == 64,
        "cohort family collision"
    );
    Ok(selected)
}
