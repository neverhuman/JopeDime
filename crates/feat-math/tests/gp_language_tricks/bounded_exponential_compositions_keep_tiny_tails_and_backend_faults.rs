use super::*;

#[test]
fn bounded_exponential_compositions_keep_tiny_tails_and_backend_faults() {
    let scratch = tempfile::tempdir().unwrap();
    let root = std::env::var_os("JOPE_GP_REDTEAM_EVIDENCE")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| scratch.path().to_owned())
        .join("bounded-tails");
    fs::create_dir_all(&root).unwrap();
    // Each guarded exponent remains <=8, and exp sees [-21.875,0]. The
    // resulting trees stay inside the existing node/depth budget; no guard changes.
    let cases = [
        ("exp", "(exp(c0/32)^8)^4"),
        ("softmax", "((exp((c0-max(c0,c1))/32)^8)^4)/(((exp((c0-max(c0,c1))/32)^8)^4)+((exp((c1-max(c0,c1))/32)^8)^4))"),
        ("log_sum_exp", "max(c0,c1)+log1p((exp((min(c0,c1)-max(c0,c1))/32)^8)^4)"),
        ("rbf", "(exp(-0.5*(c0*c0+c1*c1)/32)^8)^4"),
    ];
    let oracle = r#"import decimal,struct,sys
with decimal.localcontext() as c:
 c.prec=400
 words=[int(v) for v in open(sys.argv[2]).read().split()]
 vals=[decimal.Decimal.from_float(struct.unpack('<d',struct.pack('<Q',v))[0]) for v in words]
 for i in range(0,len(vals),4):
  a,b=vals[i:i+2]
  if sys.argv[1]=='exp': value=a.exp()
  elif sys.argv[1]=='rbf': value=(-(a*a+b*b)/2).exp()
  else:
   m=max(a,b); e=(min(a,b)-m).exp()
   value=(a-m).exp()/(1+e) if sys.argv[1]=='softmax' else m+(1+e).ln()
  print('%016x'%struct.unpack('<Q',struct.pack('<d',float(value)))[0])
"#;
    let mut matrix = Vec::new();
    for (name, tree) in cases {
        let dir = root.join(name);
        fs::create_dir_all(&dir).unwrap();
        let expr = MathExpr::parse(tree).unwrap();
        let canonical = expr.to_source().unwrap();
        assert_eq!(
            canonical,
            MathExpr::parse(&canonical).unwrap().to_source().unwrap()
        );
        let program = ScalarExpressionProgram::from_v1(&expr).unwrap();
        let bytes = program.to_bytes().unwrap();
        assert_eq!(
            bytes,
            ScalarExpressionProgram::from_bytes(&bytes)
                .unwrap()
                .to_bytes()
                .unwrap()
        );
        let mut rows = if name == "rbf" {
            (0..75)
                .map(|i| vec![i as f64 / 2.0, (i % 5) as f64 / 2.0, 0.25, -0.125])
                .collect::<Vec<_>>()
        } else {
            (0..=2800)
                .map(|i| vec![-(i as f64) / 4.0, 0.0, 0.25, -0.125])
                .collect::<Vec<_>>()
        };
        fs::write(
            dir.join("input.txt"),
            rows.iter()
                .flatten()
                .map(|v| v.to_bits().to_string())
                .collect::<Vec<_>>()
                .join(" "),
        )
        .unwrap();
        let reference_words = run(
            Command::new("python3")
                .args(["-c", oracle, name])
                .arg(dir.join("input.txt")),
            &dir,
            "decimal-oracle",
        )
        .unwrap()
        .lines()
        .map(|s| u64::from_str_radix(s, 16).unwrap())
        .collect::<Vec<_>>();
        assert_eq!(reference_words.len(), rows.len());
        let mut worst_relative_error = 0.0f64;
        for (i, (row, word)) in rows.iter().zip(&reference_words).enumerate() {
            let expected = f64::from_bits(*word);
            let result = program.evaluate_row(row);
            assert!(
                result.faults.is_empty(),
                "{name} row{i}: {:?}",
                result.faults
            );
            let actual = result.value.unwrap();
            assert!(
                actual > 0.0 && expected > 0.0,
                "{name} row{i}: lost positive tail"
            );
            let relative = (actual - expected).abs() / expected;
            assert!(
                relative <= 5e-14,
                "{name} row{i}: relative error {relative:e}"
            );
            worst_relative_error = worst_relative_error.max(relative);
        }
        let reference_rows = rows.len();
        // Outside the certified mathematical interval, only the existing guarded
        // evaluator contract is required. Fault words must match, not disappear.
        rows.extend(
            [
                -0.0,
                0.0,
                f64::from_bits(1),
                -f64::from_bits(1),
                f64::MAX,
                -f64::MAX,
                f64::INFINITY,
                f64::NEG_INFINITY,
                f64::NAN,
            ]
            .map(|v| vec![v, 0.0, 0.25, -0.125]),
        );
        let expected_words = rows
            .iter()
            .map(|row| program.evaluate_row(row).value.unwrap().to_bits())
            .collect::<Vec<_>>();
        let fault_words = rows
            .iter()
            .map(|row| program.evaluate_row(row).faults.bits())
            .collect::<Vec<_>>();
        assert!(fault_words[reference_rows..].iter().any(|bits| *bits != 0));
        let mut backends = Vec::new();
        for (language, compiler) in [
            (ScalarLanguage::Python, None),
            (ScalarLanguage::Rust, None),
            (ScalarLanguage::Cpp17, Some("g++")),
            (ScalarLanguage::Cpp17, Some("clang++")),
        ] {
            if let Some(compiler) = compiler {
                match Command::new(compiler)
                    .current_dir(&dir)
                    .arg("--version")
                    .output()
                {
                    Ok(output) => assert!(
                        output.status.success(),
                        "{compiler} --version failed: {}",
                        String::from_utf8_lossy(&output.stderr)
                    ),
                    Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                        backends.push(json!({
                            "language": format!("{language:?}"),
                            "compiler": compiler,
                            "status": "SKIPPED_COMPILER_UNAVAILABLE"
                        }));
                        continue;
                    }
                    Err(error) => panic!("failed to start {compiler}: {error}"),
                }
            }
            let words = check_backend(&program, language, compiler, &rows, &dir)
                .unwrap_or_else(|e| panic!("{name}/{language:?}/{compiler:?}: {e}"));
            assert_eq!(words, expected_words, "{name}/{language:?}/{compiler:?}");
            backends.push(json!({"language":format!("{language:?}"),"compiler":compiler,"status":"EXACT_WORD_AND_FAULT_MATCH","words":words}));
        }
        matrix.push(json!({"case":name,"tree":tree,"canonical":canonical,"reference_rows":reference_rows,"oracle_words":reference_words,"relative_bound":5e-14,"worst_relative_error":worst_relative_error,"native_words":expected_words,"fault_words":fault_words,"backends":backends}));
    }
    fs::write(
        root.join("matrix.json"),
        serde_json::to_vec_pretty(&matrix).unwrap(),
    )
    .unwrap();
}
