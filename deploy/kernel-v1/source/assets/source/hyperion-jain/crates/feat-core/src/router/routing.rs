use super::{AlgorithmRoute, RouteFamily, RouterDecision};
use crate::hashing::mix64;

pub fn route_algorithm(name: &str, anchor: Option<&str>) -> AlgorithmRoute {
    let l = name.to_ascii_lowercase();
    let mut audit = Vec::new();
    let (selected, family) = if l.starts_with("catboost") {
        ("catboost".into(), RouteFamily::CatBoost)
    } else if l.starts_with("xgboost") {
        ("xgboost".into(), RouteFamily::XGBoost)
    } else if l.starts_with("lightgbm") {
        ("lightgbm".into(), RouteFamily::LightGbm)
    } else if l.contains("starlight_core") || l.contains("tabicl") {
        ("starlight_core".into(), RouteFamily::StarlightCore)
    } else {
        let default_route = anchor.unwrap_or("catboost");
        let routed = route_algorithm(default_route, None);
        audit.push(format!(
            "unsupported router algorithm '{name}' mapped to default route '{}'",
            routed.selected
        ));
        (routed.selected, RouteFamily::Unsupported)
    };
    AlgorithmRoute {
        requested: name.into(),
        selected,
        family,
        audit,
    }
}

pub fn algorithm_family_name(name: &str) -> String {
    let l = name.to_ascii_lowercase();
    if l.starts_with("catboost") {
        "catboost".into()
    } else if l.starts_with("xgboost") {
        "xgboost".into()
    } else if l.starts_with("lightgbm") {
        "lightgbm".into()
    } else if l.contains("starlight_core") || l.contains("tabicl") {
        "starlight_core".into()
    } else {
        l.split(['_', '-', ':']).next().unwrap_or(&l).to_string()
    }
}

pub fn stable_router_hash(decision: &RouterDecision) -> u64 {
    let mut h = 0x3010_0000_0000_0001u64;
    for b in decision.selected_algorithm.bytes() {
        h = mix64(h ^ b as u64);
    }
    h ^ (decision.selected_index as u64)
}
