
use feat_math::{GenomeIr, V3Candidate, V3EvolutionConfig, initial_population, mutate_candidate, crossover_candidates};
use feat_math::evolve::{mutate_canonical_pipeline_v2, crossover_canonical_pipelines, mutate_canonical_pipeline_with_arm, CanonicalMutationArm};
fn check(source: &str)->Result<(),String>{let ir=GenomeIr::parse(source).map_err(|e|e.to_string())?;feat_math::transpile::supports(&ir).map(|_|()).map_err(|e|e.join("; "))}
fn main(){
let source="(pipe X (target kind=log1p) (standardize) (ridge))";
for v2 in [false,true]{let mut bad=0;for seed in 0..10000{let c=if v2{mutate_canonical_pipeline_v2(source,seed,6,true).unwrap()}else{feat_math::evolve::mutate_canonical_pipeline(source,seed,6).unwrap()};if let Err(e)=check(&c){bad+=1;if bad==1{println!("legacy-v{} first invalid seed={seed}: {c}; {e}",if v2{2}else{1});}}}println!("legacy-v{} invalid {bad}/10000",if v2{2}else{1});}
let mut bad=0;for seed in 0..10000{let c=crossover_canonical_pipelines(source,source,seed,6).unwrap();if let Err(e)=check(&c){bad+=1;if bad==1{println!("legacy crossover first invalid seed={seed}: {c}; {e}");}}}println!("legacy crossover invalid {bad}/10000");
let pop=initial_population(&V3EvolutionConfig::default(),&[]);let mut bad=0;for seed in 0..10000{let p=&pop[seed as usize%pop.len()];let c=mutate_candidate(p,seed,1,p.island as usize).unwrap();if let Err(e)=check(&c.syntax){bad+=1;if bad==1{println!("V3 first invalid seed={seed}: {}; {e}",c.syntax);}}}println!("V3 mutation invalid {bad}/10000");
let mut bad=0;for seed in 0..10000{let a=&pop[seed as usize%pop.len()];let b=&pop[(seed as usize*37+11)%pop.len()];let c=crossover_candidates(a,b,seed,1,a.island as usize).unwrap();if let Err(e)=check(&c.syntax){bad+=1;if bad==1{println!("V3 crossover first invalid seed={seed}: {}; {e}",c.syntax);}}}println!("V3 crossover invalid {bad}/10000");
let mut p=pop[20].clone();for seed in 0..5000 {let c=mutate_candidate(&p,seed,seed as usize+1,1).unwrap();if c.complexity()>p.complexity(){p=c;}if p.complexity()>100{break;}}println!("V3 accepted child max stages={} (no bound)",p.complexity());
let parent="(pipe X (map expr=\"sin(x)+0.1\") (ridge))";let mut changed=0;for seed in 0..1000{let c=mutate_canonical_pipeline_with_arm(parent,seed,6,true,CanonicalMutationArm::ParameterJitter).unwrap();if c!=parent{changed+=1}}println!("map-only parent parameter-jitter changed {changed}/1000");
let p=V3Candidate{syntax:"(pipe X (target kind=log1p) (standardize) (ridge))".into(),semantic_hash:String::new(),lineage:vec![],island:0,operator:String::new(),step_micros:100000,generation:0};
let x=vec![vec![1.],vec![2.],vec![3.]];let y=vec![1.,2.,3.];for seed in 0..1000{let c=mutate_candidate(&p,seed,1,0).unwrap();if check(&c.syntax).is_err(){println!("V3 invalid fit seed={seed}: {:?}",feat_math::fit(&GenomeIr::parse(&c.syntax).unwrap(),&x,&y).err());break;}}
}
