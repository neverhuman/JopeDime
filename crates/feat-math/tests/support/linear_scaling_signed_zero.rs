//! Compositions expose zero signs through a reciprocal and the final finite guard.
use feat_math::math_expr::MathExpr;
use std::fmt::Write;

pub struct Case {
    pub source: &'static str,
    pub train: Vec<Vec<f64>>,
    pub labels: Vec<f64>,
    pub test: Vec<Vec<f64>>,
}

pub fn cases() -> Vec<Case> {
    let mut cases = Vec::new();
    for source in [
        "min(min(c0,c1)^-1,1)",
        "min(max(c0,c1)^-1,1)",
        "min(row_sum^-1,1)",
        "min(row_mean^-1,1)",
        "min(row_min^-1,1)",
        "min(row_max^-1,1)",
        "min(floor(c0)^-1,1)",
        "min(round(c0)^-1,1)",
        "min(sqrt(c0)^-1,1)",
        "min(log1p(c0)^-1,1)",
        "min(sin(c0)^-1,1)",
        "min(tanh(c0)^-1,1)",
        "min(atan(c0)^-1,1)",
        "min(min((-1)^0.5,c0)^-1,1)",
        "min(max(c0,(-1)^0.5)^-1,1)",
    ] {
        let expression = MathExpr::parse(source).unwrap();
        for width in [2, 512] {
            let train = [1.0, 2.0, 4.0]
                .into_iter()
                .map(|v| vec![v; width])
                .collect::<Vec<_>>();
            let labels = train
                .iter()
                .map(|row| 3.0 * expression.eval_row(row).unwrap() + 7.0)
                .collect();
            let mut test = Vec::new();
            // Every prefix width, both mixed orders, and all-negative/all-positive zeros.
            for prefix in 1..=width {
                for negative_prefix in [false, true] {
                    test.push(
                        (0..width)
                            .map(|i| {
                                if (i < prefix) == negative_prefix {
                                    -0.0
                                } else {
                                    0.0
                                }
                            })
                            .collect(),
                    );
                }
            }
            for v in [
                -f64::from_bits(1),
                f64::from_bits(1),
                -0.49999999999999994,
                0.49999999999999994,
                -0.5,
                0.5,
                -1.5,
                1.5,
            ] {
                for background in [-0.0, 0.0] {
                    let mut row = vec![background; width];
                    row[0] = v;
                    test.push(row);
                }
            }
            cases.push(Case {
                source,
                train,
                labels,
                test,
            });
        }
    }
    cases
}

pub fn encoded(values: &[f64], backend: usize) -> String {
    let mut out = String::new();
    for value in values {
        let bits = value.to_bits();
        match backend {
            0 => write!(out, "f64::from_bits(0x{bits:016x}),").unwrap(),
            1 => write!(out, "unpack('>d',bytes.fromhex('{bits:016x}'))[0],").unwrap(),
            _ => write!(out, "from_bits(0x{bits:016x}ULL),").unwrap(),
        }
    }
    match backend {
        0 => format!("vec![{out}]"),
        1 => format!("[{out}]"),
        _ => format!("{{{out}}}"),
    }
}

pub fn table(rows: &[Vec<f64>], backend: usize) -> String {
    if rows.iter().map(Vec::len).sum::<usize>() > 4096 {
        // Encode runs by bits, so +0 and -0 remain distinct. Runtime expansion
        // avoids optimizing millions of repeated float literals in the driver.
        let mut data = String::new();
        for (i, row) in rows.iter().enumerate() {
            if i > 0 {
                data.push(';');
            }
            let mut start = 0;
            while start < row.len() {
                let word = row[start].to_bits();
                let mut end = start + 1;
                while end < row.len() && row[end].to_bits() == word {
                    end += 1;
                }
                if start > 0 {
                    data.push(',');
                }
                write!(data, "{word:016x}:{}", end - start).unwrap();
                start = end;
            }
        }
        return match backend {
            0 => format!("\"{data}\".split(';').map(|row| row.split(',').flat_map(|run| {{ let (word,n)=run.split_once(':').unwrap(); std::iter::repeat_n(f64::from_bits(u64::from_str_radix(word,16).unwrap()), n.parse::<usize>().unwrap()) }}).collect::<Vec<_>>()).collect::<Vec<_>>()"),
            1 => format!("[[v for run in row.split(',') for v in [unpack('>d',bytes.fromhex(run.split(':')[0]))[0]]*int(run.split(':')[1])] for row in '{data}'.split(';')]"),
            _ => format!("[](){{ const std::string data=\"{data}\"; std::vector<std::vector<double>> rows; std::size_t start=0; while(start<data.size()){{ auto end=data.find(';',start); if(end==std::string::npos)end=data.size(); std::vector<double> row; while(start<end){{ auto colon=data.find(':',start), stop=data.find(',',colon); if(stop==std::string::npos || stop>end)stop=end; row.insert(row.end(),std::stoull(data.substr(colon+1,stop-colon-1)),from_bits(std::stoull(data.substr(start,colon-start),nullptr,16))); start=stop+1; }} rows.push_back(row); start=end+1; }} return rows; }}()"),
        };
    }
    let out = rows
        .iter()
        .map(|row| encoded(row, backend))
        .collect::<Vec<_>>()
        .join(",");
    match backend {
        0 => format!("vec![{out}]"),
        1 => format!("[{out}]"),
        _ => format!("{{{out}}}"),
    }
}
