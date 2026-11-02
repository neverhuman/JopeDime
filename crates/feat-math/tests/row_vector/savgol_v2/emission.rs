#[test]
fn three_language_boundary_values_and_faults_are_bit_equal() {
    let root = Path::new(env!("CARGO_TARGET_TMPDIR")).join("savgol-boundary-v2");
    fs::create_dir_all(&root).unwrap();
    let fixture: serde_json::Value =
        serde_json::from_str(include_str!("../fixtures/math_boundary/v2/savgol.json")).unwrap();
    let mut rs = format!("{}\nfn main(){{\n", emit::RUST_HELPER);
    let mut py = format!("import math\n{}\n", emit::PYTHON_HELPER);
    let mut cc=format!("#include <algorithm>\n#include <cmath>\n#include <cstring>\n#include <cstdint>\n#include <stdexcept>\n#include <vector>\n#include <utility>\n#include <iostream>\n#include <iomanip>\n{}\nint main(){{\n",emit::CPP_HELPER);
    let mut expected = String::new();
    let mut count = 0;
    for case in fixture["cases"].as_array().unwrap() {
        let row = case["bits"]
            .as_array()
            .unwrap()
            .iter()
            .map(|b| f64::from_bits(u64::from_str_radix(b.as_str().unwrap(), 16).unwrap()))
            .collect::<Vec<_>>();
        let state = config(
            case["w"].as_u64().unwrap() as u8,
            case["p"].as_u64().unwrap() as u8,
            case["r"].as_u64().unwrap() as u8,
        )
        .fit(row.len())
        .unwrap();
        let out = state.apply(&row).unwrap();
        let pinned = case["expected_bits"]
            .as_array()
            .unwrap()
            .iter()
            .map(|b| u64::from_str_radix(b.as_str().unwrap(), 16).unwrap())
            .collect::<Vec<_>>();
        let pinned_faults = case["expected_faults"]
            .as_array()
            .unwrap()
            .iter()
            .map(|b| b.as_u64().unwrap() as u32)
            .collect::<Vec<_>>();
        assert_eq!(bits(&out.values), pinned);
        assert_eq!(out.faults, pinned_faults);
        for (v, f) in out.values.iter().zip(out.faults) {
            expected.push_str(&format!("{:016x} {f:08x}\n", v.to_bits()));
            count += 1;
        }
        let stage = FittedStage::Savgol(state);
        let values = bits(&row);
        let rsrow = values
            .iter()
            .map(|b| format!("f64::from_bits({b})"))
            .collect::<Vec<_>>()
            .join(",");
        let pyrow = values
            .iter()
            .map(|b| format!("struct.unpack('<d',struct.pack('<Q',{b}))[0]"))
            .collect::<Vec<_>>()
            .join(",");
        let ccrow = values
            .iter()
            .map(|b| format!("row_vector_v2_bits({b}ULL)"))
            .collect::<Vec<_>>()
            .join(",");
        rs.push_str(&format!("{{let(v,f)=row_vector_v2_values(&[vec![{rsrow}]],{});for (a,b) in v[0].iter().zip(&f[0]){{println!(\"{{:016x}} {{:08x}}\",a.to_bits(),b);}}}}\n",emit::arguments(&stage,RowRecipeLanguage::Rust)));
        py.push_str(&format!("v,f=row_vector_v2_values([[{pyrow}]],{})\nfor a,b in zip(v[0],f[0]): print('%016x %08x' % (struct.unpack('<Q',struct.pack('<d',a))[0],b))\n",emit::arguments(&stage,RowRecipeLanguage::Python)));
        cc.push_str(&format!("{{auto out=row_vector_v2_values({{{{{ccrow}}}}},{});for(std::size_t i=0;i<out.first[0].size();++i){{std::uint64_t b;double a=out.first[0][i];std::memcpy(&b,&a,8);std::cout<<std::hex<<std::setfill('0')<<std::setw(16)<<b<<' '<<std::setw(8)<<out.second[0][i]<<'\\n';}}}}\n",emit::arguments(&stage,RowRecipeLanguage::Cpp17)));
    }
    assert!(count >= 150);
    rs.push_str("}\n");
    cc.push_str("}\n");
    fs::write(root.join("boundary.rs"), rs).unwrap();
    fs::write(root.join("boundary.py"), py).unwrap();
    fs::write(root.join("boundary.cpp"), cc).unwrap();
    fs::write(root.join("expected.txt"), &expected).unwrap();
    run(Command::new("rustc")
        .arg("-O")
        .arg(root.join("boundary.rs"))
        .arg("-o")
        .arg(root.join("rust")));
    assert_eq!(
        run(&mut Command::new(root.join("rust"))).stdout,
        expected.as_bytes()
    );
    assert_eq!(
        run(Command::new("python3").arg(root.join("boundary.py"))).stdout,
        expected.as_bytes()
    );
    cpp(&root, "boundary.cpp", "cpp17", "g++", "-std=c++17", &[]);
    assert_eq!(
        run(&mut Command::new(root.join("cpp17"))).stdout,
        expected.as_bytes()
    );
    if std::env::var_os("JOPE_ROW_CPP_QUALIFY").is_some() {
        for (name, compiler, std, extra) in [
            ("clang17", "clang++", "-std=c++17", vec![]),
            ("cpp20", "g++", "-std=c++20", vec![]),
            (
                "sanitizer",
                "g++",
                "-std=c++17",
                vec![
                    "-O1",
                    "-fsanitize=address,undefined",
                    "-fno-omit-frame-pointer",
                    "-no-pie",
                ],
            ),
        ] {
            cpp(&root, "boundary.cpp", name, compiler, std, &extra);
            assert_eq!(
                run(&mut Command::new(root.join(name))).stdout,
                expected.as_bytes()
            );
        }
    }
}
