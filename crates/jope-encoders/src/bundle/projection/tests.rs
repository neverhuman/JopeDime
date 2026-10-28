use super::*;

#[test]
fn reviewed_projection_names_only_the_permitted_policy_closure() {
    let layout = VerifiedJopePolicyAssets::reviewed_layout().unwrap();
    assert_eq!(layout.files.len(), 4);
    assert_eq!(
        layout.files.iter().map(|f| f.bytes).sum::<u64>(),
        117_501_806
    );
    assert_eq!(layout.notices.len(), 1);
    assert!(layout
        .notices
        .contains_key("docs/notices/jain-governed-models.txt"));
    assert!(!layout.files.iter().any(|f| f.path == FOUNDATION_ARTIFACT));
    assert!(layout.distribution.contains("owner-review-required"));
}

#[test]
fn changed_parent_is_rejected_before_any_asset_is_opened() {
    let mut changed = REVIEWED_MANIFEST.to_vec();
    changed.push(b' ');
    let error = VerifiedJopePolicyAssets::verify(&changed, |_, _| {
        panic!("changed parent must not grant any asset read")
    })
    .err()
    .unwrap();
    assert!(error.to_string().contains("parent manifest identity"));
}

#[test]
fn entry_projection_rejects_missing_extra_and_duplicate_inputs() {
    assert!(VerifiedJopePolicyAssets::from_entries(REVIEWED_MANIFEST, &[]).is_err());
    assert!(VerifiedJopePolicyAssets::from_entries(
        REVIEWED_MANIFEST,
        &[(FOUNDATION_ARTIFACT, b"prohibited")]
    )
    .is_err());
    assert!(VerifiedJopePolicyAssets::from_entries(
        REVIEWED_MANIFEST,
        &[(JOPE_POLICY_ARTIFACT, b"x"), (JOPE_POLICY_ARTIFACT, b"x")]
    )
    .is_err());
}
