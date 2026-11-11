use jope_api::{
    carbon::{CarbonPingResponse, CarbonRegionCatalog},
    decimal::DecimalU64,
};
use jope_client::carbon_regions::{CarbonRouteError, CarbonSmartRouter};

#[test]
fn smart_router_ranks_by_ewma_rtt_and_trips_circuit_breaker_on_failures() {
    let mut router =
        CarbonSmartRouter::default_17_regions("carbon.jopedime.com", Some("americas".to_owned()))
            .expect("17-region router");

    assert_eq!(router.ranked_candidates(1_000).expect("ranked").len(), 17);

    let ping_eu = CarbonPingResponse {
        region: "eu-west-1".to_owned(),
        healthy: true,
        server_epoch_ms: DecimalU64(1_000),
        enterprise_self_hosted: false,
    };
    router
        .record_probe_success("us-east-2", 42.0, None, 1_000)
        .expect("probe us-east-2");
    router
        .record_probe_success("eu-west-1", 11.5, Some(&ping_eu), 1_000)
        .expect("probe eu-west-1");
    router
        .record_probe_success("ap-northeast-1", 130.0, None, 1_000)
        .expect("probe ap-northeast-1");

    let best = router.select_best(1_050).expect("best");
    assert_eq!(best.region, "eu-west-1");

    // Trip eu-west-1 circuit breaker with 3 consecutive failures.
    for tick in [1_100, 1_200, 1_300] {
        router
            .record_failure("eu-west-1", tick)
            .expect("record failure");
    }
    assert!(!router
        .region_health("eu-west-1")
        .expect("exists")
        .is_available_at(1_500));
    let failover = router.select_best(1_500).expect("failover");
    assert_eq!(failover.region, "us-east-2");

    // After 30s cooldown expires, eu-west-1 re-enters candidate pool.
    assert!(router
        .region_health("eu-west-1")
        .expect("exists")
        .is_available_at(32_000));
}

#[test]
fn smart_router_refuses_empty_catalog_with_guidance() {
    let empty = CarbonRegionCatalog {
        catalog_version: 1,
        regions: vec![],
    };
    let err = CarbonSmartRouter::from_catalog(empty, None).expect_err("empty must fail");
    assert_eq!(err, CarbonRouteError::EmptyCatalog);
    assert!(!err.guidance().is_empty());
}
