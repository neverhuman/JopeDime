use feat_math::{GenomeIr, TaskSpec,Objective,DataSchema,ColumnSpec,RowSemantics,RawTable};
fn main(){
let task=TaskSpec::regression(Objective::Rmse).unwrap();let schema=DataSchema{columns:vec![ColumnSpec::numeric("a")],row_semantics:RowSemantics::Iid};let table=RawTable{headers:vec!["a".to_string()],rows:vec![vec!["1".to_string()],vec!["2".to_string()],vec!["3".to_string()],vec!["4".to_string()]]};let y=vec![2.,4.,6.,8.];
for src in [r#"(pipe X (map expr="c0*c0") (ridge))"#,r#"(pipe X (ridge misspelled_parameter=7))"#,r#"(pipe X (ridge lambda_frac="junk"))"#,r#"(sym raw "c0")"#,r#"(pipe X (hurdle2))"#] {
let g=GenomeIr::parse(src).unwrap();println!("source={src}; validator={:?}; fit_table={:?}; legacy_fit={:?}; legacy_emit_support={:?}",feat_math::advanced_v3::validate_v3_genome(&g,&task,&schema),feat_math::fit_table(&g,&task,&schema,&table,&y).map(|_|()),feat_math::fit(&g,&vec![vec![1.],vec![2.],vec![3.],vec![4.]],&y).map(|_|()),feat_math::transpile::supports(&g));
}
}
