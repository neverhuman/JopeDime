use feat_math::{
    transpile::{emit, EmitOptions},
    GenomeIr,
};
use std::{fs, process::Command};

#[test]
fn invalid_intermediates_abort_complete_python_and_rust_expressions() {
    for (i, source) in [
        "(sym raw \"min(clip(1,(-1)^0.5,(-1)^0.5),7)+1\")",
        "(pipe X (map expr=\"min(clip(x,(-1)^0.5,(-1)^0.5),7)+1\") (ridge))",
        "(pipe X (derive expr=\"min(clip(c0,(-1)^0.5,(-1)^0.5),7)+1\") (ridge))",
    ]
    .iter()
    .enumerate()
    {
        let genome = GenomeIr::parse(source).unwrap();
        let artifacts = emit(&genome, &EmitOptions::default()).unwrap();
        assert!(artifacts.manifest.contains("feat-math-clip-safety-v2"));
        let scratch = tempfile::Builder::new()
            .prefix(&format!("kernel-export-safety-{i}-"))
            .tempdir()
            .unwrap();
        let dir = scratch.path();
        fs::write(dir.join("train.csv"), "1,2,0\n2,3,1\n3,4,2\n4,5,3\n").unwrap();
        fs::write(dir.join("test.csv"), "2,4\n-1,3\n").unwrap();
        fs::write(dir.join("model.py"), artifacts.python).unwrap();
        fs::write(dir.join("model.rs"), artifacts.rust).unwrap();
        let compiled = Command::new("rustc")
            .args(["-O", "--edition=2021", "model.rs", "-o", "predictor"])
            .current_dir(dir)
            .output()
            .unwrap();
        assert!(
            compiled.status.success(),
            "{}",
            String::from_utf8_lossy(&compiled.stderr)
        );
        let expected = feat_math::fit_predict(
            &genome,
            &vec![vec![1., 2.], vec![2., 3.], vec![3., 4.], vec![4., 5.]],
            &[0., 1., 2., 3.],
            &vec![vec![2., 4.], vec![-1., 3.]],
        )
        .unwrap();
        for (executable, prefix) in [("python3", vec!["model.py"]), ("./predictor", vec![])] {
            let result = Command::new(executable)
                .args(prefix)
                .args(["train.csv", "test.csv", "prediction.txt"])
                .current_dir(dir)
                .output()
                .unwrap();
            assert!(
                result.status.success(),
                "{source} {executable}: {}",
                String::from_utf8_lossy(&result.stderr)
            );
            let actual: Vec<_> = fs::read_to_string(dir.join("prediction.txt"))
                .unwrap()
                .lines()
                .map(|s| s.parse::<f64>().unwrap().to_bits())
                .collect();
            assert_eq!(
                actual,
                expected.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
            );
        }
        scratch.close().unwrap();
    }
}
