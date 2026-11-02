use super::*;
use crate::dsl::parse_dsl;

fn stage(src: &str) -> Stage {
    parse_dsl(&format!("(pipe X {src})")).unwrap().stages[0].clone()
}

fn ctx() -> StageCacheCtx {
    StageCacheCtx {
        context_tag: "fixture|rows=10|features=3".to_string(),
        folds: 5,
        fold: 0,
    }
}

#[test]
fn chain_keys_separate_context_target_and_stages() {
    let cache = StageCache::new(DEFAULT_BYTE_CAP);
    let base = cache.chain_root(&ctx(), None);
    let with_target = cache.chain_root(&ctx(), Some(&stage("(target kind=log1p)")));
    assert!(base != with_target, "target stage must alter the root");
    let other_fold = cache.chain_root(&StageCacheCtx { fold: 1, ..ctx() }, None);
    assert!(base != other_fold, "fold must alter the root");
    let a = cache.chain_extend(&base, &stage("(standardize)"));
    let b = cache.chain_extend(&base, &stage("(standardize append=true)"));
    assert!(a != b, "stage params must alter the chain");
    // The exact encoding walks parameters in key order, so source order does not matter.
    let c1 = cache.chain_extend(&base, &stage("(rff_cos gamma=1.5 n=8 seed=3)"));
    let c2 = cache.chain_extend(&base, &stage("(rff_cos seed=3 n=8 gamma=1.5)"));
    assert!(c1 == c2, "param order must not alter the chain");
}

#[test]
fn stages_that_print_alike_get_distinct_keys_and_entries() {
    // Canonical text prints a float within 1e-12 of an integer as that integer, so `near`
    // prints exactly as `exact` does although it is a different computation.
    let exact = stage("(rff_cos gamma=1.0 n=8 seed=3)");
    let mut near = exact.clone();
    near.params
        .insert("gamma".to_string(), ParamValue::Float(1.0 + 5e-13));
    assert_ne!(1.0f64.to_bits(), (1.0f64 + 5e-13).to_bits());
    assert_eq!(exact.to_string(), near.to_string());
    let cache = StageCache::new(DEFAULT_BYTE_CAP);
    let base = cache.chain_root(&ctx(), None);
    let exact_key = cache.chain_extend(&base, &exact);
    let near_key = cache.chain_extend(&base, &near);
    assert!(
        exact_key != near_key,
        "print-alike stages must not share a key"
    );
    assert!(
        cache.chain_root(&ctx(), Some(&exact)) != cache.chain_root(&ctx(), Some(&near)),
        "print-alike target stages must not share a root"
    );
    // An entry for one is never a hit for the other, and an equal stage still hits.
    cache.begin_generation(0);
    let block = Arc::new(vec![vec![0.0; 2]; 3]);
    cache.insert(exact_key, FittedOp::Sigmoid { append: false }, block);
    assert!(cache.get(&near_key).is_none());
    assert!(cache
        .get(&cache.chain_extend(&base, &exact.clone()))
        .is_some());
    // Signed zero, an integer against its float, and a string against a number differ too.
    let with = |name: &str, value: ParamValue| {
        let mut s = Stage::new("winsor_clip");
        s.params.insert(name.to_string(), value);
        cache.chain_extend(&base, &s)
    };
    assert!(with("q", ParamValue::Float(0.0)) != with("q", ParamValue::Float(-0.0)));
    assert!(with("k", ParamValue::Int(1)) != with("k", ParamValue::Float(1.0)));
    assert!(with("k", ParamValue::Str("1".into())) != with("k", ParamValue::Int(1)));
    assert!(with("k", ParamValue::Bool(true)) != with("k", ParamValue::Int(1)));
}

#[test]
fn every_value_class_that_prints_alike_keys_apart() {
    let cache = StageCache::new(DEFAULT_BYTE_CAP);
    let base = cache.chain_root(&ctx(), None);
    let symlog = |value: ParamValue| {
        let mut stage = Stage::new("symlog");
        stage.params.insert("c".to_string(), value);
        stage
    };
    let float = ParamValue::Float;
    let text = |value: &str| ParamValue::Str(value.to_string());
    assert_ne!(f64::NAN.to_bits(), (-f64::NAN).to_bits());
    // Each pair prints one canonical text for two different values: a float near an
    // integer, a float near zero on either side, two NaNs, and a string that reads as a
    // number or a bool.
    for (left, right) in [
        (float(1.0), float(1.0 + 5e-13)),
        (float(0.0), float(1e-13)),
        (float(-0.0), float(-1e-13)),
        (float(f64::NAN), float(-f64::NAN)),
        (text("1.5"), float(1.5)),
        (text("7"), ParamValue::Int(7)),
        (text("true"), ParamValue::Bool(true)),
    ] {
        let (a, b) = (symlog(left.clone()), symlog(right.clone()));
        assert_eq!(
            a.to_string(),
            b.to_string(),
            "{left:?} and {right:?} print alike"
        );
        assert!(
            cache.chain_extend(&base, &a) != cache.chain_extend(&base, &b),
            "{left:?} and {right:?} share a stage key"
        );
        assert!(
            cache.chain_root(&ctx(), Some(&a)) != cache.chain_root(&ctx(), Some(&b)),
            "{left:?} and {right:?} share a target root"
        );
    }
    // The type tag separates an integer from the float with the same eight bytes.
    assert!(
        cache.chain_extend(&base, &symlog(ParamValue::Int(1)))
            != cache.chain_extend(&base, &symlog(float(f64::from_bits(1))))
    );
    // The display text is unchanged.
    assert_eq!(symlog(float(1.0 + 5e-13)).to_string(), "(symlog c=1.0)");
    assert_eq!(symlog(float(-1e-13)).to_string(), "(symlog c=-0.0)");
    assert_eq!(symlog(text("1.5")).to_string(), "(symlog c=1.5)");
}

#[test]
fn a_print_alike_constant_gets_its_own_fit_through_the_cache() {
    use crate::dsl::DslExpr;
    use crate::model::{fit, fit_with_stage_cache, Model};
    use crate::GenomeIr;
    let mut rng = crate::Pcg32::new(11, 3);
    let x: Rows = (0..60)
        .map(|_| (0..3).map(|_| rng.next_f64() * 4.0 - 2.0).collect())
        .collect();
    let y: Vec<f64> = x.iter().map(|row| row[0] - 0.5 * row[1] * row[2]).collect();
    let genome = |c: f64| {
        let mut symlog = Stage::new("symlog");
        symlog.params.insert("c".to_string(), ParamValue::Float(c));
        GenomeIr::Pipeline(DslExpr {
            stages: vec![symlog, Stage::new("ridge")],
        })
    };
    let bits = |model: &Model| -> Vec<u64> {
        model
            .predict(&x)
            .unwrap()
            .iter()
            .map(|v| v.to_bits())
            .collect()
    };
    let cache = StageCache::new(DEFAULT_BYTE_CAP);
    cache.begin_generation(0);
    let mut predictions = Vec::new();
    // 1.0 and 1.0 + 5e-13 both print as (symlog c=1.0).
    for c in [1.0, 1.0 + 5e-13] {
        let plain = fit(&genome(c), &x, &y).unwrap();
        let cached = fit_with_stage_cache(&genome(c), &x, &y, &cache, &ctx()).unwrap();
        assert_eq!(
            serde_json::to_string(&plain).unwrap(),
            serde_json::to_string(&cached).unwrap(),
            "c = {c:e}"
        );
        assert_eq!(bits(&plain), bits(&cached), "c = {c:e}");
        predictions.push(bits(&cached));
    }
    // The constant reaches the predictions, so a shared entry would have been wrong.
    assert_ne!(predictions[0], predictions[1]);
    // Each symlog stage missed and was fitted: the second never took the first's entry.
    let (hits, misses, _) = cache.stats();
    assert_eq!((hits, misses), (0, 2));
}

#[test]
fn insert_get_sweep_and_byte_accounting() {
    let cache = StageCache::new(1_000_000);
    cache.begin_generation(0);
    let key = cache.chain_extend(&cache.chain_root(&ctx(), None), &stage("(standardize)"));
    let block = Arc::new(vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
    let fitted = FittedOp::Sigmoid { append: false };
    assert!(cache.get(&key).is_none());
    cache.insert(key, fitted, Arc::clone(&block));
    let (_, cached) = cache.get(&key).expect("hit after insert");
    assert!(Arc::ptr_eq(&cached, &block));
    let (hits, misses, bytes) = cache.stats();
    assert_eq!((hits, misses), (1, 1));
    assert!(bytes > 32);
    // Entry used in generation 0 survives the generation-1 sweep but not
    // the generation-3 sweep.
    cache.begin_generation(1);
    cache.sweep();
    assert!(cache.get(&key).is_some());
    cache.begin_generation(3);
    cache.sweep();
    assert!(cache.get(&key).is_none());
    assert_eq!(cache.stats().2, 0, "sweep must return all bytes");
}

#[test]
fn admission_under_pressure_is_deterministic_and_a_sweep_frees_budget() {
    // Each entry holds 64 block bytes and a 28-byte fit, so a 400-byte cap admits four.
    let cache = StageCache::new(400);
    cache.begin_generation(0);
    let root = cache.chain_root(&ctx(), None);
    let keys: Vec<ChainKey> = (0..5)
        .map(|index| cache.chain_extend(&root, &stage(&format!("(lag k={})", index + 1))))
        .collect();
    for (index, key) in keys.iter().enumerate() {
        cache.insert(
            *key,
            FittedOp::Sigmoid { append: false },
            Arc::new(vec![vec![index as f64; 8]]),
        );
    }
    let resident: Vec<bool> = keys.iter().map(|key| cache.get(key).is_some()).collect();
    assert_eq!(resident, vec![true, true, true, true, false]);
    assert_eq!(cache.stats().2, 4 * 92);
    // Two generations on, the unused entries are stale: the sweep returns their bytes and
    // the refused key now fits.
    cache.begin_generation(2);
    cache.sweep();
    assert_eq!(cache.stats().2, 0);
    cache.insert(
        keys[4],
        FittedOp::Sigmoid { append: false },
        Arc::new(vec![vec![4.0; 8]]),
    );
    assert!(cache.get(&keys[4]).is_some());
    assert_eq!(cache.stats().2, 92);
}

#[test]
fn insertions_never_pass_the_byte_cap_between_sweeps() {
    let cache = StageCache::new(400);
    cache.begin_generation(0);
    let root = cache.chain_root(&ctx(), None);
    let mut kept = 0;
    for index in 0..20 {
        let key = cache.chain_extend(&root, &stage(&format!("(lag k={})", index + 1)));
        cache.insert(
            key,
            FittedOp::Sigmoid { append: false },
            Arc::new(vec![vec![index as f64; 8]]),
        );
        assert!(
            cache.stats().2 <= 400,
            "resident bytes exceed the cap at insert {index}"
        );
        if cache.get(&key).is_some() {
            kept += 1;
        }
    }
    // The first entries fit and stay retrievable; the rest are dropped, never half-kept.
    assert!((1..20).contains(&kept), "{kept}");
    // Replacing an entry with an equal one keeps the accounting exact. The cache has room
    // for the second reservation, so the replacement really runs.
    let roomy = StageCache::new(1_000);
    roomy.begin_generation(0);
    let key = roomy.chain_extend(&roomy.chain_root(&ctx(), None), &stage("(lag k=1)"));
    roomy.insert(
        key,
        FittedOp::Sigmoid { append: false },
        Arc::new(vec![vec![0.0; 8]]),
    );
    let one = roomy.stats().2;
    assert_eq!(one, 92);
    let replacement = Arc::new(vec![vec![0.0; 8]]);
    roomy.insert(
        key,
        FittedOp::Sigmoid { append: false },
        Arc::clone(&replacement),
    );
    let (_, cached) = roomy.get(&key).expect("replacement resident");
    assert!(
        Arc::ptr_eq(&cached, &replacement),
        "the replacement is stored"
    );
    assert_eq!(roomy.stats().2, one, "a replacement does not double-count");
}

#[test]
fn streaming_fit_size_matches_json_and_refuses_oversized_entries() {
    let fitted = FittedOp::Rank {
        sorted_cols: vec![vec![-0.0, 1.25, f64::MAX]; 32],
        append: true,
    };
    let expected = serde_json::to_vec(&fitted).unwrap().len();
    let mut counter = ByteCounter {
        bytes: 0,
        limit: expected,
    };
    serde_json::to_writer(&mut counter, &fitted).unwrap();
    assert_eq!(counter.bytes, expected);
    let block = Arc::new(vec![vec![1.0; 8]]);
    for cap in [0, 63, 64, expected + 63, expected + 64] {
        let cache = StageCache::new(cap);
        let key = cache.chain_root(&ctx(), None);
        cache.insert(key, fitted.clone(), block.clone());
        assert_eq!(cache.get(&key).is_some(), cap == expected + 64);
        assert_eq!(cache.stats().2, if cap == expected + 64 { cap } else { 0 });
    }
    let mut overflow = ByteCounter {
        bytes: usize::MAX,
        limit: usize::MAX,
    };
    assert!(std::io::Write::write(&mut overflow, b"x").is_err());
}

#[test]
fn concurrent_reads_replacements_and_sweeps_preserve_cap_and_values() {
    let cache = StageCache::new(4096);
    let key = cache.chain_root(&ctx(), None);
    let block = Arc::new(vec![vec![1.0; 8]]);
    cache.insert(key, FittedOp::Sigmoid { append: false }, block.clone());
    std::thread::scope(|scope| {
        for _ in 0..8 {
            let block = &block;
            let cache = &cache;
            scope.spawn(move || {
                for _ in 0..200 {
                    cache.insert(key, FittedOp::Sigmoid { append: false }, block.clone());
                    let (fitted, found) = cache.get(&key).unwrap();
                    assert!(matches!(fitted, FittedOp::Sigmoid { append: false }));
                    assert_eq!(found.as_ref(), block.as_ref());
                    assert!(cache.stats().2 <= 4096);
                    cache.sweep();
                }
            });
        }
    });
    assert_eq!(cache.stats().2, 92);
    cache.begin_generation(2);
    cache.sweep();
    assert_eq!(cache.stats().2, 0);
    assert_eq!(
        block.as_ref(),
        &vec![vec![1.0; 8]],
        "readers retain their block"
    );
}
