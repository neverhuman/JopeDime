use feat_math::{evolve::{self, EvolveConfig, Observer, GenStats, Scored, StageReuseOptions}, transpile::Emitted, stage_cache_v2::{Mode,BYTE_CAP}};
struct Quiet;
impl Observer for Quiet {
    fn on_generation(&self,_:&GenStats){}
    fn on_improvement(&self,_:&Scored,_:&Emitted,_:&GenStats){}
}
#[test]
fn existing_disabled_miss_and_hit_searches_have_identical_results() {
    let x:Vec<Vec<f64>>=(0..64).map(|i|vec![(i as f64*0.3).sin(),(i%7)as f64]).collect();
    let y:Vec<_>=x.iter().map(|r|r[0]*r[1]+0.1).collect();
    let cfg=EvolveConfig{pop:24,gens:4,seed:410300,time_budget_secs:0.0,..EvolveConfig::default()};
    let plain=evolve::evolve_lite_race(&x,&y,&feat_math::metrics::r2,&cfg,&Quiet);
    let expected=plain.candidates.iter().map(|c|(c.dsl.clone(),c.rung_fold_scores.clone().map(|r|r.map(|v|v.iter().map(|s|s.to_bits()).collect::<Vec<_>>())))).collect::<Vec<_>>();
    for options in [None,Some(StageReuseOptions{mode:Mode::Overhead,byte_cap:0,evaluator_source:[2;32]}),Some(StageReuseOptions{mode:Mode::Reuse,byte_cap:BYTE_CAP,evaluator_source:[2;32]})] {
        let (actual,receipt)=evolve::evolve_lite_race_with_stage_reuse(&x,&y,&feat_math::metrics::r2,&cfg,&Quiet,None,options);
        assert_eq!(plain.rung_evals,actual.rung_evals); assert_eq!(plain.target_hit,actual.target_hit);
        assert_eq!(plain.ranked[0].dsl,actual.ranked[0].dsl);
        let values=actual.candidates.iter().map(|c|(c.dsl.clone(),c.rung_fold_scores.clone().map(|r|r.map(|v|v.iter().map(|s|s.to_bits()).collect::<Vec<_>>())))).collect::<Vec<_>>();
        assert_eq!(expected,values);
        if let Some(r)=receipt { assert!(r.cache.peak_accounted_payload<=BYTE_CAP);if options.unwrap().mode==Mode::Reuse { assert!(r.cache.hits>0); } }
    }
}
