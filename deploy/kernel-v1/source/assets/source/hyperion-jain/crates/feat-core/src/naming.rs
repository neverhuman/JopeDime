//! User-facing model display names.
//!
//! Internally the gradient-boosting member is called `catboost` (and `catboost_s2`,
//! `catboost_residual_starforge`) — this string is a real dispatch/routing key
//! (`router::routing::route_algorithm` matches the `catboost` prefix, `backend` selects the FFI
//! library) and MUST NOT change. But the product brands it **jboost**, so every place the name reaches
//! a user — the cockpit leaderboard/reasoning, the CLI, and the downloaded model manifest/files — maps
//! the internal id through [`display_model_name`] at the boundary. Non-catboost names pass through
//! unchanged.

/// Map an internal model/member id to its user-facing display name (`catboost*` → `jboost*`).
/// Everything else (xgboost, lightgbm, ensemble, starforge, …) is returned unchanged.
pub fn display_model_name(name: &str) -> String {
    if let Some(rest) = name.strip_prefix("catboost") {
        format!("jboost{rest}")
    } else {
        name.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn maps_catboost_family_only() {
        assert_eq!(display_model_name("catboost"), "jboost");
        assert_eq!(display_model_name("catboost_s2"), "jboost_s2");
        assert_eq!(
            display_model_name("catboost_residual_starforge"),
            "jboost_residual_starforge"
        );
        // Other backends are untouched.
        assert_eq!(display_model_name("xgboost"), "xgboost");
        assert_eq!(display_model_name("lightgbm"), "lightgbm");
        assert_eq!(display_model_name("ensemble"), "ensemble");
        assert_eq!(display_model_name("starforge"), "starforge");
    }
}
