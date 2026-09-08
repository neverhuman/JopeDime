use feat_math::{GenomeIr,math_expr::MathExpr};
use feat_math::dsl::{parse_dsl,ParamValue};
fn main(){
std::panic::set_hook(Box::new(|_| {}));
for src in [r#"(sym gibberish "c0")"#,r#"(sym raw "c0""#] {println!("sym {:?} => {:?}",src,GenomeIr::parse(src).map(|g|g.canonical()));}
for src in [r#"(pipe X (ridge lambda=1.0 lambda=2.0))"#,r#"(pipe X (ridge key="true"))"#,r#"(pipe X (ridge key="0.5"))"#,r#"(pipe X (ridge lambda=0.0000000000001))"#] {
let d=parse_dsl(src).unwrap();let c=d.to_string();let back=parse_dsl(&c).unwrap();println!("roundtrip {:?} => {:?}; equal={}",src,c,d==back);}
for src in ["clip(1,(-1)^0.5,(-1)^0.5)","-2^2","2^100","1/0","log(-1)","c99"] {
let m=MathExpr::parse(src).unwrap();println!("math {:?} => {:?}",src,std::panic::catch_unwind(||m.eval_row(&[2.0])));}
for src in [r#"(pipe X (map expr="0.5") (ridge))"#,r#"(pipe X (step t=0.0000000000001) (ridge))"#] {
let g=GenomeIr::parse(src).unwrap();let c=g.canonical();let back=GenomeIr::parse(&c).unwrap();
let (GenomeIr::Pipeline(a),GenomeIr::Pipeline(b))=(&g,&back) else {panic!()};
let x=vec![vec![5e-14]];let y=vec![1.0];
println!("effective {:?} => {:?}; support-before={:?}, support-after={:?}; op-before={:?}; op-after={:?}",src,c,feat_math::transpile::supports(&g),feat_math::transpile::supports(&back),feat_math::ops::fit_op(&a.stages[0],&x,&y).map(|o|feat_math::ops::apply_op(&o,&x)),feat_math::ops::fit_op(&b.stages[0],&x,&y).map(|o|feat_math::ops::apply_op(&o,&x)));
}
println!("1.9 to usize {:?}; inf to usize {:?}",ParamValue::Float(1.9).as_usize(),ParamValue::Float(f64::INFINITY).as_usize());
}
