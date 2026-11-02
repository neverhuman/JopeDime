use super::{active_sampler_id, REPLAY_SAMPLER_ID, SAMPLER_ID, SEARCH_BEATING_SAMPLER_ID};
use serde_json::json;

#[test]
fn search_beating_identity_is_opt_in() {
    assert_eq!(active_sampler_id(None), SAMPLER_ID);
    assert_eq!(
        active_sampler_id(Some(&json!({"replay":"fresh_balanced"}))),
        REPLAY_SAMPLER_ID
    );
    assert_eq!(
        active_sampler_id(Some(&json!({"discoveries":"teacher_improved"}))),
        SEARCH_BEATING_SAMPLER_ID
    );
    assert_eq!(
        active_sampler_id(Some(&json!({"search_beating":true}))),
        SEARCH_BEATING_SAMPLER_ID
    );
}
