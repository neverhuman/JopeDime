use feat_math::genome::GenomeIr;
use feat_math::metrics::rmse;
use feat_math::{fit, Rows};
use jope_pipeline::search_cache::{fit_cached, FitIdentity, LaneStageCache};

fn identity<'a>(ids: &'a [u64]) -> FitIdentity<'a> {
    FitIdentity {
        registry_identity: b"registry-v1:fixture",
        evaluator_identity: b"evaluator-v1:fixture",
        train_row_ids: ids,
        fold_plan: b"fold-plan-v1:fold-0-of-2",
        fit_seed: 17,
        effect_closure: b"effects:none",
        numeric_profile: b"f64:strict",
        folds: 2,
        fold: 0,
    }
}

fn training() -> (Rows, Vec<f64>) {
    (
        vec![vec![0.0], vec![1.0], vec![2.0], vec![3.0], vec![4.0]],
        vec![0.0, 1.2, 1.9, 3.1, 3.9],
    )
}

#[test]
fn full_context_separates_order_rows_seed_fold_and_profile() {
    let (x, y) = training();
    let ids = [10, 11, 12, 13, 14];
    let base = identity(&ids).stage_context(&x, &y).context_tag;

    let reversed_ids = [14, 13, 12, 11, 10];
    assert_ne!(
        base,
        identity(&reversed_ids).stage_context(&x, &y).context_tag
    );
    let mut changed_x = x.clone();
    changed_x[0][0] = -0.0;
    assert_ne!(
        base,
        identity(&ids).stage_context(&changed_x, &y).context_tag
    );
    let mut reordered_x = x.clone();
    reordered_x.swap(0, 1);
    assert_ne!(
        base,
        identity(&ids).stage_context(&reordered_x, &y).context_tag
    );
    let mut changed_y = y.clone();
    changed_y[0] = -0.0;
    assert_ne!(
        base,
        identity(&ids).stage_context(&x, &changed_y).context_tag
    );
    let mut changed = identity(&ids);
    changed.fit_seed += 1;
    assert_ne!(base, changed.stage_context(&x, &y).context_tag);
    let mut changed = identity(&ids);
    changed.fold = 1;
    assert_ne!(base, changed.stage_context(&x, &y).context_tag);
    let mut changed = identity(&ids);
    changed.numeric_profile = b"f64:alternate";
    assert_ne!(base, changed.stage_context(&x, &y).context_tag);
    let mut changed = identity(&ids);
    changed.registry_identity = b"registry-v1:other";
    assert_ne!(base, changed.stage_context(&x, &y).context_tag);
    let mut changed = identity(&ids);
    changed.evaluator_identity = b"evaluator-v1:other";
    assert_ne!(base, changed.stage_context(&x, &y).context_tag);
    let mut changed = identity(&ids);
    changed.effect_closure = b"effects:other";
    assert_ne!(base, changed.stage_context(&x, &y).context_tag);
}

#[test]
fn cached_uncached_and_evicted_fits_preserve_predictions_and_champion() {
    let (x, y) = training();
    let hold_x = vec![vec![0.5], vec![2.5], vec![3.5]];
    let hold_y = vec![0.7, 2.4, 3.6];
    let ids = [10, 11, 12, 13, 14];
    let identity = identity(&ids);
    let genomes = [
        GenomeIr::parse("(pipe X (standardize) (ridge))").unwrap(),
        GenomeIr::parse("(pipe X (robust_scale) (ridge))").unwrap(),
    ];

    let cache = LaneStageCache::default_lane();
    cache.begin_generation(0);
    let cached_predictions: Vec<_> = genomes
        .iter()
        .map(|genome| {
            let model = fit_cached(genome, &x, &y, &identity, &cache).unwrap();
            model.predict(&hold_x).unwrap()
        })
        .collect();
    let uncached_predictions: Vec<_> = genomes
        .iter()
        .map(|genome| fit(genome, &x, &y).unwrap().predict(&hold_x).unwrap())
        .collect();
    assert_eq!(cached_predictions, uncached_predictions);

    let cached_champion = cached_predictions
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| rmse(&hold_y, a).total_cmp(&rmse(&hold_y, b)))
        .map(|(index, _)| genomes[index].canonical())
        .unwrap();
    let uncached_champion = uncached_predictions
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| rmse(&hold_y, a).total_cmp(&rmse(&hold_y, b)))
        .map(|(index, _)| genomes[index].canonical())
        .unwrap();
    assert_eq!(cached_champion, uncached_champion);

    let warm = fit_cached(&genomes[0], &x, &y, &identity, &cache)
        .unwrap()
        .predict(&hold_x)
        .unwrap();
    assert_eq!(warm, uncached_predictions[0]);
    assert!(
        cache.profile().hits > 0,
        "warm fit should reuse a cached prefix"
    );

    let capped = LaneStageCache::new(1024);
    capped.begin_generation(0);
    let before = fit_cached(&genomes[0], &x, &y, &identity, &capped)
        .unwrap()
        .predict(&hold_x)
        .unwrap();
    assert!(
        capped.profile().resident_bytes > 0,
        "fixture should cache a stage"
    );
    capped.begin_generation(3);
    capped.evict_stale();
    let profile = capped.profile();
    assert!(profile.resident_bytes <= profile.byte_cap);
    assert_eq!(profile.resident_bytes, 0, "stale entry should be evicted");
    let resumed = fit_cached(&genomes[0], &x, &y, &identity, &capped)
        .unwrap()
        .predict(&hold_x)
        .unwrap();
    assert_eq!(before, uncached_predictions[0]);
    assert_eq!(resumed, uncached_predictions[0]);
}
