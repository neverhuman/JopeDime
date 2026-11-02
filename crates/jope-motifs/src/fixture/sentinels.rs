use super::*;

fn push_digest(out: &mut Vec<String>, digest: &str) {
    out.push(digest.to_owned());
    if let Some(prefix) = digest.get(..16) {
        out.push(prefix.to_owned());
    }
}

/// The identifiers one table carries: names, full ids and their 12- and 16-hex prefixes,
/// every provenance digest (burn included) with its 16-hex prefix, the family labels in its
/// provenance, the source label and digest, and the family map digest. Host texts,
/// expansions, stage texts and recipe digests are listed separately, because a bound
/// manifest legitimately carries the digest of its own program. The table digest is not
/// listed: operator-private stdout may carry it, and the shareable summary is checked for it
/// separately.
fn carried_identifiers(table: &MotifTable) -> Vec<String> {
    let mut out = Vec::new();
    for motif in &table.body.motifs {
        out.push(motif.name.clone());
        out.push(motif.id.clone());
        out.push(motif.id[..12].to_owned());
        out.push(motif.id[..16].to_owned());
        for occurrence in &motif.provenance {
            for digest in [
                &occurrence.dataset_sha256,
                &occurrence.cut_sha256,
                &occurrence.comparison_sha256,
                &occurrence.record_sha256,
            ] {
                push_digest(&mut out, digest);
            }
            if let Some(burn) = &occurrence.label.burn_sha256 {
                push_digest(&mut out, burn);
            }
            if let Some(family) = &occurrence.family {
                out.push(family.clone());
            }
        }
    }
    out.push(table.body.source.schema.clone());
    push_digest(&mut out, &table.body.source.sha256);
    if let Some(map) = &table.body.parameters.family_map_sha256 {
        push_digest(&mut out, map);
    }
    out
}

/// The program texts one table carries: expansions, stage texts, host programs and recipe
/// digests with their 16-hex prefixes.
fn carried_texts(table: &MotifTable) -> Vec<String> {
    let mut out = Vec::new();
    for motif in &table.body.motifs {
        out.push(motif.expansion.clone());
        if let Ok(stages) = crate::fragment::expansion_stages(&motif.expansion) {
            out.extend(crate::fragment::stage_texts(&stages));
        }
    }
    for (recipe, host) in &table.body.hosts {
        push_digest(&mut out, recipe);
        out.push(host.clone());
    }
    out
}

/// Positive control for every absence assertion: each private value the sentinels derive
/// from this table really is in the table's private bytes, and a labelled table carries
/// both family canaries. Without it an absence assertion could pass because the canary was
/// never planted.
pub(crate) fn assert_carried(table: &MotifTable) {
    let bytes = String::from_utf8(table.to_json().expect("json")).expect("utf-8");
    let mut values = carried_identifiers(table);
    values.extend(carried_texts(table));
    assert!(!values.is_empty());
    for value in values {
        let encoded = serde_json::to_string(&value).expect("json string");
        let inner = &encoded[1..encoded.len() - 1];
        assert!(bytes.contains(inner), "{value} is not in the private table");
    }
    if table.body.parameters.family_map_sha256.is_some() {
        for family in [FAMILY_A, FAMILY_B] {
            assert!(bytes.contains(family), "{family} is not planted");
        }
    }
}

/// What no emitted artifact may contain: the reserved prefix, both family labels, the
/// identifiers the table carries, and every digest of every fixture line.
pub(crate) fn emitted_sentinels(table: &MotifTable) -> Vec<String> {
    let mut out = vec![
        NAME_PREFIX.to_owned(),
        FAMILY_A.to_owned(),
        FAMILY_B.to_owned(),
    ];
    out.extend(carried_identifiers(table));
    for line in lines() {
        for digest in [
            line.dataset_sha256(),
            line.cut_sha256(),
            line.comparison_sha256(),
            line.record_sha256(),
        ] {
            push_digest(&mut out, digest);
        }
        if let Some(burn) = &line.label().burn_sha256 {
            push_digest(&mut out, burn);
        }
    }
    out
}

/// Everything planted: the emitted sentinels plus expansions, stage texts, host programs and
/// recipe digests.
pub(crate) fn sentinels(table: &MotifTable) -> Vec<String> {
    let mut out = emitted_sentinels(table);
    out.extend(carried_texts(table));
    out
}

/// A report in which every check of every fragment passed once.
pub(crate) fn passing_validation(table: &MotifTable) -> Validation {
    let pass = CheckResult {
        passed: true,
        exercised: 1,
        failures: Vec::new(),
    };
    Validation {
        schema: VALIDATION_SCHEMA.to_owned(),
        table_sha256: table.table_sha256.clone(),
        source_verified: false,
        source_digest_equal: None,
        motifs: (0..table.body.motifs.len())
            .map(|index| MotifResult {
                index: u32::try_from(index).expect("index"),
                round_trip: pass.clone(),
                determinism: pass.clone(),
                context_compatibility: pass.clone(),
                name_absence: pass.clone(),
                not_exercised: 0,
                backend: Vec::new(),
            })
            .collect(),
    }
}
