use super::*;

pub fn derive_binary_source_family(meta_path: &Path) -> Result<String> {
    let metadata: Value = if meta_path.is_file() {
        serde_json::from_slice(&std::fs::read(meta_path)?)?
    } else {
        Value::Object(Map::new())
    };

    if let Some(family) = metadata
        .get("source_family")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|family| !family.is_empty())
    {
        return Ok(family.to_owned());
    }

    let source_real_data = metadata
        .get("source_real_data")
        .filter(|value| json_truthy(value));
    let material = if let Some(source) = source_real_data {
        serde_json::json!({"origin": "real", "source": source})
    } else {
        let recipe = metadata.get("generator_recipe").and_then(Value::as_object);
        let feature_family_counts = match recipe
            .and_then(|recipe| recipe.get("feature_family_counts"))
            .filter(|value| json_truthy(value))
            .cloned()
        {
            Some(value) => value,
            None => serde_json::json!({}),
        };
        let origin = match metadata
            .get("origin")
            .filter(|value| json_truthy(value))
            .cloned()
        {
            Some(value) => value,
            None => Value::String("unknown".to_owned()),
        };
        let generator = match metadata
            .get("generator_name")
            .filter(|value| json_truthy(value))
            .cloned()
        {
            Some(value) => value,
            None => Value::String("unknown".to_owned()),
        };
        serde_json::json!({
            "origin": origin,
            "generator": generator,
            "feature_family_counts": feature_family_counts,
        })
    };
    let canonical = python_ascii_json(&crate::meta::canonical_json(&material));
    let digest = sha256_hex(canonical.as_bytes());
    Ok(format!("family-{}", &digest[..16]))
}

fn python_ascii_json(canonical: &str) -> String {
    let mut escaped = String::with_capacity(canonical.len());
    for character in canonical.chars() {
        let codepoint = character as u32;
        if (0x20..=0x7e).contains(&codepoint) {
            escaped.push(character);
        } else if codepoint <= 0xffff {
            escaped.push_str(&format!("\\u{codepoint:04x}"));
        } else {
            let supplementary = codepoint - 0x1_0000;
            let high = 0xd800 + (supplementary >> 10);
            let low = 0xdc00 + (supplementary & 0x3ff);
            escaped.push_str(&format!("\\u{high:04x}\\u{low:04x}"));
        }
    }
    escaped
}

fn json_truthy(value: &Value) -> bool {
    match value {
        Value::Null => false,
        Value::Bool(value) => *value,
        Value::Number(value) => value.as_f64().is_some_and(|value| value != 0.0),
        Value::String(value) => !value.is_empty(),
        Value::Array(value) => !value.is_empty(),
        Value::Object(value) => !value.is_empty(),
    }
}

/// Assign a source family to the stable campaign train/validation partition.
/// This preserves the historical SHA-256 prefix threshold used by self-play
/// expansion so every consumer makes the same label-independent decision.
pub fn source_family_campaign_split(family: &str) -> &'static str {
    let digest = sha2::Sha256::digest(family.as_bytes());
    if u16::from_be_bytes([digest[0], digest[1]]) < (u16::MAX / 10) {
        "validation"
    } else {
        "train"
    }
}

/// Assign source families to train/development while covering every requested
/// task on both sides. Selection is seeded by family identity only; labels,
/// scores, dataset hashes, and row counts do not influence the assignment.
pub fn build_training_family_partition(
    families_by_task: &BTreeMap<String, BTreeSet<String>>,
    seed: u64,
) -> Result<BTreeMap<String, String>> {
    ensure!(
        !families_by_task.is_empty() && families_by_task.len() <= 3,
        "unsupported requested task inventory"
    );
    ensure!(
        families_by_task
            .values()
            .all(|families| families.len() >= 2),
        "real training requires at least two independently evidenced families per category"
    );
    let mut all = families_by_task
        .values()
        .flat_map(|families| families.iter().cloned())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect::<Vec<_>>();
    ensure!(
        all.len() <= 30,
        "family registry exceeds bounded panel scope"
    );
    all.sort_by_key(|name| {
        sha256_hex(format!("jope.family-partition/v1\0{seed}\0{name}").as_bytes())
    });
    let acceptable = |selected: &BTreeSet<String>| {
        families_by_task
            .values()
            .all(|set| !set.is_disjoint(selected) && !set.is_subset(selected))
    };
    let mut selected = None;
    'outer: for a in 0..all.len() {
        for b in a..all.len() {
            for c in b..all.len() {
                let set = [all[a].clone(), all[b].clone(), all[c].clone()]
                    .into_iter()
                    .collect();
                if acceptable(&set) {
                    selected = Some(set);
                    break 'outer;
                }
            }
        }
    }
    let mut selected = selected.ok_or_else(|| {
        anyhow::anyhow!("no family-disjoint partition covers every requested category")
    })?;
    let target = all.len().div_ceil(5);
    for name in &all {
        if selected.len() >= target {
            break;
        }
        let mut trial = selected.clone();
        trial.insert(name.clone());
        if acceptable(&trial) {
            selected = trial;
        }
    }
    Ok(all
        .into_iter()
        .map(|family| {
            let side = if selected.contains(&family) {
                "development"
            } else {
                "train"
            };
            (family, side.to_owned())
        })
        .collect())
}

/// A deterministic family-disjoint holdout split with exact requested counts.
pub fn freeze_source_family_split(
    dataset_families: &BTreeMap<String, String>,
    train_count: usize,
    holdout_count: usize,
    seed: u64,
) -> Result<(Vec<String>, Vec<String>)> {
    ensure!(
        train_count.checked_add(holdout_count) == Some(dataset_families.len()),
        "split counts must cover every dataset exactly once"
    );

    let mut families: BTreeMap<&str, Vec<&str>> = BTreeMap::new();
    for (dataset, family) in dataset_families {
        ensure!(
            !dataset.is_empty() && !family.is_empty(),
            "dataset and source-family identities must be non-empty"
        );
        families.entry(family).or_default().push(dataset);
    }
    let mut ordered: Vec<_> = families.keys().copied().collect();
    ordered.sort_by_key(|family| sha256_hex(format!("{seed}|{family}").as_bytes()));

    let mut reachable: BTreeMap<usize, Vec<&str>> = BTreeMap::from([(0, Vec::new())]);
    for family in ordered {
        let size = families[family].len();
        let prior: Vec<_> = reachable
            .iter()
            .rev()
            .map(|(total, chosen)| (*total, chosen.clone()))
            .collect();
        for (total, chosen) in prior {
            let Some(candidate) = total.checked_add(size) else {
                continue;
            };
            if candidate <= holdout_count && !reachable.contains_key(&candidate) {
                let mut next = chosen;
                next.push(family);
                reachable.insert(candidate, next);
            }
        }
    }
    let held = reachable
        .get(&holdout_count)
        .ok_or_else(|| {
            anyhow::anyhow!("source families cannot realize the requested exact holdout count")
        })?
        .iter()
        .copied()
        .collect::<BTreeSet<_>>();
    let mut holdout = families
        .iter()
        .filter(|(family, _)| held.contains(*family))
        .flat_map(|(_, datasets)| datasets.iter().copied())
        .map(str::to_owned)
        .collect::<Vec<_>>();
    holdout.sort();
    let mut train = dataset_families
        .keys()
        .filter(|dataset| holdout.binary_search(*dataset).is_err())
        .cloned()
        .collect::<Vec<_>>();
    train.sort();
    ensure!(
        train.len() == train_count,
        "source-family split count invariant failed"
    );
    Ok((train, holdout))
}
