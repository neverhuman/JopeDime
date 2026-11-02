use super::*;

pub(super) fn validate_digest(value: &str, field: &str) -> Result<()> {
    ensure!(
        value.len() == 64
            && value
                .bytes()
                .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b)),
        "{field} must be a SHA-256 hex digest"
    );
    Ok(())
}

pub(super) fn sum_bytes(objects: &[PhysicalObject]) -> Result<u64> {
    objects.iter().try_fold(0u64, |sum, object| {
        sum.checked_add(object.bytes)
            .ok_or_else(|| anyhow::anyhow!("byte total overflow"))
    })
}

pub(super) fn replacement_for(
    candidate: &Candidate,
    policy: &RetentionPolicyV2,
    current_objects: &BTreeSet<PhysicalObject>,
    keeper_objects: &BTreeSet<PhysicalObject>,
) -> Result<Option<(Recommendation, PhysicalObject)>> {
    let Some(proof) = &candidate.replacement else {
        return Ok(None);
    };
    match proof {
        ReplacementProof::ExactDuplicate {
            replacement,
            content_hash_verified,
            restore_tested,
        } => {
            replacement.validate()?;
            if candidate.sources.len() != 1 {
                return Ok(None);
            }
            let source = &candidate.sources[0];
            if *content_hash_verified
                && *restore_tested
                && source.sha256 == replacement.sha256
                && source.bytes == replacement.bytes
                && source != replacement
                && current_objects.contains(replacement)
                && keeper_objects.contains(replacement)
            {
                Ok(Some((Recommendation::ReviewDuplicate, replacement.clone())))
            } else {
                Ok(None)
            }
        }
        ReplacementProof::Pack {
            replacement,
            members,
            manifest_verified,
            restore_tested,
        } => {
            replacement.validate()?;
            let source_set: BTreeSet<_> = candidate.sources.iter().cloned().collect();
            let member_set: BTreeSet<_> = members.iter().cloned().collect();
            if *manifest_verified
                && *restore_tested
                && source_set.len() == candidate.sources.len()
                && source_set == member_set
                && member_set.len() == members.len()
                && !current_objects.contains(replacement)
            {
                Ok(Some((Recommendation::ReviewPack, replacement.clone())))
            } else {
                Ok(None)
            }
        }
        ReplacementProof::Supersession {
            replacement,
            qualified_champion,
            relative_margin,
            days_since_dominance,
            score_rows_retained,
            recovery_closure_verified,
            restore_tested,
        } => {
            replacement.validate()?;
            ensure!(
                relative_margin.is_finite() && *relative_margin >= 0.0,
                "supersession margin must be finite and nonnegative"
            );
            if candidate.sources.len() == 1
                && *qualified_champion
                && *relative_margin >= policy.margin_rel
                && *days_since_dominance >= policy.grace_days
                && *score_rows_retained
                && *recovery_closure_verified
                && *restore_tested
                && current_objects.contains(replacement)
                && keeper_objects.contains(replacement)
            {
                Ok(Some((
                    Recommendation::ReviewSupersession,
                    replacement.clone(),
                )))
            } else {
                Ok(None)
            }
        }
    }
}

pub(super) fn replacement_gap(
    candidate: &Candidate,
    policy: &RetentionPolicyV2,
    current_objects: &BTreeSet<PhysicalObject>,
    keeper_objects: &BTreeSet<PhysicalObject>,
) -> String {
    match candidate.replacement.as_ref() {
        None => "replacement or recovery proof is missing".into(),
        Some(ReplacementProof::ExactDuplicate {
            replacement,
            content_hash_verified,
            restore_tested,
        }) => {
            if !content_hash_verified {
                "exact duplicate content hash is not independently verified".into()
            } else if !restore_tested {
                "duplicate recovery restore has not been tested".into()
            } else if !current_objects.contains(replacement) {
                "exact primary version is absent from the current inventory".into()
            } else if !keeper_objects.contains(replacement) {
                "exact primary version is not protected as a keeper".into()
            } else {
                "duplicate physical identity, bytes, or key comparison is incomplete".into()
            }
        }
        Some(ReplacementProof::Pack {
            replacement,
            members,
            manifest_verified,
            restore_tested,
        }) => {
            if !manifest_verified {
                "pack manifest does not verify the exact source closure".into()
            } else if !restore_tested {
                "pack recovery restore has not been tested".into()
            } else if current_objects.contains(replacement) {
                "pack output already exists in the current inventory; use an exact-version proof"
                    .into()
            } else if members.iter().cloned().collect::<BTreeSet<_>>()
                != candidate.sources.iter().cloned().collect()
                || members.iter().cloned().collect::<BTreeSet<_>>().len() != members.len()
            {
                "pack member versions do not exactly match the proposed source closure".into()
            } else {
                "pack proof is incomplete".into()
            }
        }
        Some(ReplacementProof::Supersession {
            replacement,
            qualified_champion,
            relative_margin,
            days_since_dominance,
            score_rows_retained,
            recovery_closure_verified,
            restore_tested,
            ..
        }) => {
            if !qualified_champion {
                "superseding champion is not qualified".into()
            } else if !relative_margin.is_finite() || *relative_margin < policy.margin_rel {
                "qualified champion does not meet the configured dominance margin".into()
            } else if *days_since_dominance < policy.grace_days {
                "qualified dominance has not completed the configured grace period".into()
            } else if !score_rows_retained {
                "supersession proof does not retain the score rows".into()
            } else if !recovery_closure_verified {
                "supersession recovery closure is not verified".into()
            } else if !restore_tested {
                "supersession recovery restore has not been tested".into()
            } else if !current_objects.contains(replacement) {
                "qualified replacement version is absent from the current inventory".into()
            } else if !keeper_objects.contains(replacement) {
                "qualified replacement version is not protected as a keeper".into()
            } else {
                "supersession candidate does not have one exact source object".into()
            }
        }
    }
}
