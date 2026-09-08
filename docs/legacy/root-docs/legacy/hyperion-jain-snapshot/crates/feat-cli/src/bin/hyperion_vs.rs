//! Write synchronized Hyperion cell scores and row embeddings for a headerless train/test CSV pair.
//!
//! Each input CSV is expected to have the target in the final column. The target is dropped before
//! Hyperion sees the data. Quantile/sweep state is fit on train and replayed for test through the
//! same API the main pipeline uses. Cell-score percentiles are also fit on train only.

#[cfg(feature = "hyperion-cpu")]
mod imp {
    use anyhow::{bail, Context};
    use feat_core::data::Matrix;
    use feat_core::hyperion::{
        fit_transform_pair, hyperion_device_label, HyperionConfig, HyperionEngine, HyperionMode,
        DEFAULT_MANIFEST, DEFAULT_OUTPUT_DIM, DEFAULT_WEIGHTS,
    };
    use feat_core::progress::NullSink;
    use std::cmp::Ordering;
    use std::io::{BufWriter, Write};
    use std::path::{Path, PathBuf};
    use std::time::Instant;

    #[derive(Debug)]
    struct Args {
        train: PathBuf,
        test: PathBuf,
        out_cell_train: PathBuf,
        out_cell_test: PathBuf,
        out_row_train: PathBuf,
        out_row_test: PathBuf,
        weights: PathBuf,
        manifest: PathBuf,
        sweeps: usize,
        scales: Vec<usize>,
        seed: Option<u64>,
        batch: usize,
    }

    struct OutputBlock<'a> {
        path: &'a Path,
        data: &'a [f32],
        rows: usize,
        cols: usize,
        unit_interval: bool,
    }

    fn take(argv: &[String], i: &mut usize, flag: &str) -> anyhow::Result<String> {
        *i += 1;
        argv.get(*i)
            .cloned()
            .with_context(|| format!("{flag} requires a value"))
    }

    fn parse_scales(raw: &str) -> anyhow::Result<Vec<usize>> {
        let scales: Vec<usize> = raw
            .split(',')
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(|s| {
                s.parse::<usize>()
                    .with_context(|| format!("invalid scale '{s}'"))
            })
            .collect::<anyhow::Result<_>>()?;
        if scales.is_empty() {
            bail!("--scales must contain at least one integer");
        }
        Ok(scales)
    }

    fn parse_args() -> anyhow::Result<Args> {
        let argv: Vec<String> = std::env::args().collect();
        let mut train = None;
        let mut test = None;
        let mut out_cell_train = None;
        let mut out_cell_test = None;
        let mut out_row_train = None;
        let mut out_row_test = None;
        let mut weights = PathBuf::from(DEFAULT_WEIGHTS);
        let mut manifest = PathBuf::from(DEFAULT_MANIFEST);
        let mut sweeps = 3usize;
        let mut scales = vec![4, 7, 11, 15];
        let mut seed = None;
        let mut batch = 256usize;

        let mut i = 1;
        while i < argv.len() {
            match argv[i].as_str() {
                "--train" => train = Some(PathBuf::from(take(&argv, &mut i, "--train")?)),
                "--test" => test = Some(PathBuf::from(take(&argv, &mut i, "--test")?)),
                "--out-cell-train" => {
                    out_cell_train = Some(PathBuf::from(take(&argv, &mut i, "--out-cell-train")?))
                }
                "--out-cell-test" => {
                    out_cell_test = Some(PathBuf::from(take(&argv, &mut i, "--out-cell-test")?))
                }
                "--out-row-train" => {
                    out_row_train = Some(PathBuf::from(take(&argv, &mut i, "--out-row-train")?))
                }
                "--out-row-test" => {
                    out_row_test = Some(PathBuf::from(take(&argv, &mut i, "--out-row-test")?))
                }
                "--weights" => weights = PathBuf::from(take(&argv, &mut i, "--weights")?),
                "--manifest" => manifest = PathBuf::from(take(&argv, &mut i, "--manifest")?),
                "--sweeps" => {
                    sweeps = take(&argv, &mut i, "--sweeps")?
                        .parse::<usize>()
                        .context("invalid --sweeps")?
                        .max(1)
                }
                "--scales" => scales = parse_scales(&take(&argv, &mut i, "--scales")?)?,
                "--seed" => {
                    seed = Some(
                        take(&argv, &mut i, "--seed")?
                            .parse()
                            .context("invalid --seed")?,
                    )
                }
                "--batch" => {
                    batch = take(&argv, &mut i, "--batch")?
                        .parse::<usize>()
                        .context("invalid --batch")?
                        .max(1)
                }
                "--help" | "-h" => {
                    println!(
                        "usage: hyperion_vs --train train.csv --test test.csv \
                         --out-cell-train train3.vs --out-cell-test test3.vs \
                         --out-row-train hyperion3train.vs --out-row-test hyperion3test.vs \
                         [--weights model.safetensors --manifest manifest.json] \
                         [--sweeps 3 --scales 4,7,11,15 --seed N --batch 256]"
                    );
                    std::process::exit(0);
                }
                other => bail!("unknown argument '{other}'"),
            }
            i += 1;
        }

        Ok(Args {
            train: train.context("--train is required")?,
            test: test.context("--test is required")?,
            out_cell_train: out_cell_train.context("--out-cell-train is required")?,
            out_cell_test: out_cell_test.context("--out-cell-test is required")?,
            out_row_train: out_row_train.context("--out-row-train is required")?,
            out_row_test: out_row_test.context("--out-row-test is required")?,
            weights,
            manifest,
            sweeps,
            scales,
            seed,
            batch,
        })
    }

    fn parse_feature_csv_text(text: &str, label: &str) -> anyhow::Result<Matrix> {
        let mut rows = Vec::new();
        let mut width = None;
        for (lineno, line) in text.lines().enumerate() {
            if line.trim().is_empty() {
                continue;
            }
            let row: Vec<f32> = line
                .split(',')
                .map(|f| f.trim().parse::<f32>().unwrap_or(f32::NAN))
                .collect();
            if let Some(expected) = width {
                if row.len() != expected {
                    bail!(
                        "{label} row {} has {} columns, expected {expected}",
                        lineno + 1,
                        row.len()
                    );
                }
            } else {
                width = Some(row.len());
            }
            rows.push(row);
        }
        if rows.is_empty() {
            bail!("{label} has no data rows");
        }
        let full_cols = width.unwrap_or(0);
        if full_cols < 2 {
            bail!("{label} needs at least one feature plus target");
        }
        let n = rows.len();
        let c = full_cols - 1;
        let mut data = vec![0.0f32; n * c];
        for (r, row) in rows.iter().enumerate() {
            for (j, value) in row.iter().take(c).enumerate() {
                data[j * n + r] = *value;
            }
        }
        Ok(Matrix::new(data, n, c))
    }

    fn read_feature_csv(path: &Path) -> anyhow::Result<Matrix> {
        let text =
            std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
        parse_feature_csv_text(&text, &path.display().to_string())
    }

    fn temp_path(path: &Path) -> anyhow::Result<PathBuf> {
        let file_name = path
            .file_name()
            .and_then(|s| s.to_str())
            .context("output path needs a UTF-8 file name")?;
        Ok(path.with_file_name(format!(".{file_name}.tmp.{}", std::process::id())))
    }

    fn write_block_tmp(block: &OutputBlock<'_>) -> anyhow::Result<PathBuf> {
        if let Some(parent) = block.path.parent() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("creating {}", parent.display()))?;
        }
        anyhow::ensure!(
            block.data.len() == block.rows * block.cols,
            "block shape mismatch for {}: {} values for {}x{}",
            block.path.display(),
            block.data.len(),
            block.rows,
            block.cols
        );
        let tmp = temp_path(block.path)?;
        let file = std::fs::File::create(&tmp)
            .with_context(|| format!("creating temp {}", tmp.display()))?;
        let mut out = BufWriter::new(file);
        for row in 0..block.rows {
            for col in 0..block.cols {
                if col > 0 {
                    out.write_all(b",")?;
                }
                let v = block.data[col * block.rows + row];
                anyhow::ensure!(
                    v.is_finite(),
                    "non-finite Hyperion value at {} row {row} col {col}",
                    block.path.display()
                );
                if block.unit_interval {
                    anyhow::ensure!(
                        (0.0..=1.0).contains(&v),
                        "cell percentile outside [0,1] at {} row {row} col {col}: {v}",
                        block.path.display()
                    );
                }
                write!(out, "{v:.9}")?;
            }
            out.write_all(b"\n")?;
        }
        out.flush()?;
        Ok(tmp)
    }

    fn rename_tmp(tmp: &Path, dst: &Path) -> anyhow::Result<()> {
        std::fs::rename(tmp, dst)
            .with_context(|| format!("renaming {} to {}", tmp.display(), dst.display()))
    }

    fn cleanup(paths: impl IntoIterator<Item = PathBuf>) {
        for path in paths {
            let _ = std::fs::remove_file(path);
        }
    }

    fn write_all_outputs(blocks: &[OutputBlock<'_>]) -> anyhow::Result<()> {
        let mut temps: Vec<(PathBuf, PathBuf)> = Vec::new();
        for block in blocks {
            match write_block_tmp(block) {
                Ok(tmp) => temps.push((tmp, block.path.to_path_buf())),
                Err(err) => {
                    cleanup(temps.into_iter().map(|(tmp, _)| tmp));
                    return Err(err);
                }
            }
        }

        let mut renamed = Vec::new();
        for (tmp, dst) in &temps {
            if let Err(err) = rename_tmp(tmp, dst) {
                cleanup(temps.iter().map(|(tmp, _)| tmp.clone()));
                cleanup(renamed);
                return Err(err);
            }
            renamed.push(dst.clone());
        }
        Ok(())
    }

    fn train_column_percentiles(raw: &Matrix) -> anyhow::Result<Matrix> {
        let n = raw.n;
        anyhow::ensure!(n > 0, "cannot percentile-scale empty train matrix");
        let mut out = vec![0.0f32; raw.n * raw.c];
        for col in 0..raw.c {
            let offset = col * raw.n;
            let values = &raw.data[offset..offset + raw.n];
            anyhow::ensure!(
                values.iter().all(|v| v.is_finite()),
                "non-finite train cell score in column {col}"
            );
            let mut idx: Vec<usize> = (0..n).collect();
            idx.sort_by(|&a, &b| values[a].partial_cmp(&values[b]).unwrap_or(Ordering::Equal));
            for (rank, &row) in idx.iter().enumerate() {
                out[offset + row] = (rank as f32 + 0.5) / n as f32;
            }
        }
        Ok(Matrix::new(out, raw.n, raw.c))
    }

    fn train_fitted_test_percentiles(
        train_raw: &Matrix,
        test_raw: &Matrix,
    ) -> anyhow::Result<Matrix> {
        anyhow::ensure!(
            train_raw.c == test_raw.c,
            "cell-score feature count mismatch: train has {}, test has {}",
            train_raw.c,
            test_raw.c
        );
        anyhow::ensure!(
            train_raw.n > 0,
            "cannot map test percentiles with empty train matrix"
        );
        let n = train_raw.n;
        let lo_clamp = 0.5f32 / n as f32;
        let hi_clamp = 1.0f32 - lo_clamp;
        let mut out = vec![0.0f32; test_raw.n * test_raw.c];
        for col in 0..train_raw.c {
            let train_offset = col * train_raw.n;
            let test_offset = col * test_raw.n;
            let mut sorted = train_raw.data[train_offset..train_offset + train_raw.n].to_vec();
            anyhow::ensure!(
                sorted.iter().all(|v| v.is_finite()),
                "non-finite train cell score in column {col}"
            );
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
            for row in 0..test_raw.n {
                let v = test_raw.data[test_offset + row];
                anyhow::ensure!(
                    v.is_finite(),
                    "non-finite test cell score at row {row} col {col}"
                );
                let lo = sorted.partition_point(|x| *x < v);
                let hi = sorted.partition_point(|x| *x <= v);
                let pct = ((lo + hi) as f32 / (2.0 * n as f32)).clamp(lo_clamp, hi_clamp);
                out[test_offset + row] = pct;
            }
        }
        Ok(Matrix::new(out, test_raw.n, test_raw.c))
    }

    fn run(args: Args) -> anyhow::Result<()> {
        let train = read_feature_csv(&args.train)?;
        let test = read_feature_csv(&args.test)?;
        if test.c != train.c {
            bail!(
                "feature count mismatch: train has {} features, test has {}",
                train.c,
                test.c
            );
        }

        let input_names = (0..train.c).map(|j| format!("f{j:05}")).collect::<Vec<_>>();
        let mut cfg = HyperionConfig {
            mode: HyperionMode::On,
            weights_path: args.weights,
            manifest_path: args.manifest,
            output_dim: DEFAULT_OUTPUT_DIM,
            sweeps: args.sweeps,
            engine: HyperionEngine::V7,
            scales: args.scales,
            batch: args.batch,
            ..HyperionConfig::default()
        };
        if let Some(seed) = args.seed {
            cfg.seed = seed;
        }

        eprintln!(
            "[hyperion_vs] train={} test={} features={} sweeps={} scales={:?} device={}",
            train.n,
            test.n,
            train.c,
            cfg.sweeps,
            cfg.scales,
            hyperion_device_label()
        );
        let aug = fit_transform_pair(&train, &test, &input_names, &cfg, &NullSink, Instant::now())?;
        anyhow::ensure!(aug.state.is_some(), "Hyperion did not produce state");
        anyhow::ensure!(
            aug.train.c == train.c + DEFAULT_OUTPUT_DIM
                && aug.hold.c == test.c + DEFAULT_OUTPUT_DIM,
            "unexpected augmented widths: train {} hold {}",
            aug.train.c,
            aug.hold.c
        );

        let train_conf = aug
            .confidence
            .as_ref()
            .context("missing train confidence surface")?;
        let test_conf = aug
            .hold_confidence
            .as_ref()
            .context("missing test confidence surface")?;
        anyhow::ensure!(
            (train_conf.n, train_conf.c) == (train.n, train.c),
            "unexpected train confidence shape: {}x{} expected {}x{}",
            train_conf.n,
            train_conf.c,
            train.n,
            train.c
        );
        anyhow::ensure!(
            (test_conf.n, test_conf.c) == (test.n, test.c),
            "unexpected test confidence shape: {}x{} expected {}x{}",
            test_conf.n,
            test_conf.c,
            test.n,
            test.c
        );

        let train_cell = train_column_percentiles(train_conf)?;
        let test_cell = train_fitted_test_percentiles(train_conf, test_conf)?;
        let train_row_offset = train.c * train.n;
        let test_row_offset = test.c * test.n;
        let train_row = &aug.train.data[train_row_offset..];
        let test_row = &aug.hold.data[test_row_offset..];

        let blocks = [
            OutputBlock {
                path: &args.out_cell_train,
                data: train_cell.data.as_ref(),
                rows: train.n,
                cols: train.c,
                unit_interval: true,
            },
            OutputBlock {
                path: &args.out_cell_test,
                data: test_cell.data.as_ref(),
                rows: test.n,
                cols: test.c,
                unit_interval: true,
            },
            OutputBlock {
                path: &args.out_row_train,
                data: train_row,
                rows: train.n,
                cols: DEFAULT_OUTPUT_DIM,
                unit_interval: false,
            },
            OutputBlock {
                path: &args.out_row_test,
                data: test_row,
                rows: test.n,
                cols: DEFAULT_OUTPUT_DIM,
                unit_interval: false,
            },
        ];
        write_all_outputs(&blocks)?;
        eprintln!(
            "[hyperion_vs] wrote {}, {}, {}, {}",
            args.out_cell_train.display(),
            args.out_cell_test.display(),
            args.out_row_train.display(),
            args.out_row_test.display()
        );
        Ok(())
    }

    pub fn main_impl() -> anyhow::Result<()> {
        run(parse_args()?)
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn final_target_column_is_dropped() {
            let a = parse_feature_csv_text("1,2,0\n3,4,1\n", "a").unwrap();
            let b = parse_feature_csv_text("1,2,99\n3,4,-7\n", "b").unwrap();
            assert_eq!((a.n, a.c), (2, 2));
            assert_eq!(a.data.as_ref(), b.data.as_ref());
        }

        #[test]
        fn test_percentiles_use_train_cdf_with_midpoint_ties() {
            let train = Matrix::new(vec![10.0, 20.0, 20.0, 40.0], 4, 1);
            let test = Matrix::new(vec![5.0, 20.0, 30.0, 50.0], 4, 1);
            let train_pct = train_column_percentiles(&train).unwrap();
            assert_eq!(train_pct.data.as_ref(), &[0.125, 0.375, 0.625, 0.875]);

            let test_pct = train_fitted_test_percentiles(&train, &test).unwrap();
            assert_eq!(test_pct.data.as_ref(), &[0.125, 0.5, 0.75, 0.875]);
        }
    }
}

#[cfg(feature = "hyperion-cpu")]
fn main() -> anyhow::Result<()> {
    imp::main_impl()
}

#[cfg(not(feature = "hyperion-cpu"))]
fn main() {
    eprintln!(
        "hyperion_vs requires the hyperion-cpu or hyperion-cuda feature:\n  \
         cargo build -p feat-cli --features hyperion-cpu --bin hyperion_vs"
    );
    std::process::exit(2);
}
