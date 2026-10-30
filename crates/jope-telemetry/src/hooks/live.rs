//! `live-metrics.jsonl` for the DIME score trainer (`jope-prime.dime-live-metrics.v1`), ported
//! from `dime_score/train.py::_LiveMetricsWriter`.
//!
//! Rows are averaged over a window of `1 / hz` seconds and the file rotates to `<name>.1` past
//! the size cap. Each line is spelled the way `json.dumps(record, sort_keys=True)` spelled it
//! (`", "`/`": "` separators, `repr` floats, `NaN`, ASCII escapes), and the window means sum in
//! numpy's pairwise order, so `dime_score/tui.py` and anything diffing old runs see the same text.

use std::fs::{self, OpenOptions};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

pub const LIVE_SCHEMA: &str = "jope-prime.dime-live-metrics.v1";
pub const ROTATE_BYTES: u64 = 8 * 1024 * 1024;

/// Averaged over the window, in the trainer's order (`loader_wait_ms` .. `optimizer_ms` also make
/// up the `loader_wait_pct` denominator).
pub const MEAN_FIELDS: [&str; 9] = [
    "total_loss",
    "huber_loss",
    "bradley_terry_loss",
    "validity_loss",
    "loader_wait_ms",
    "h2d_ms",
    "compute_ms",
    "optimizer_ms",
    "gradient_norm",
];

/// Copied from the window's latest row.
pub const LATEST_FIELDS: [&str; 7] = [
    "epoch",
    "step",
    "steps",
    "global_step",
    "batch_shape",
    "learning_rate",
    "eta_seconds",
];

const LOADER_WAIT: usize = 4;

/// One training step, as the trainer hands it over.
#[derive(Clone, Debug, PartialEq)]
pub struct LiveRow {
    pub batch_samples: i64,
    pub starved: bool,
    pub means: [f64; 9],
    pub queue_depth: Option<i64>,
    /// `LATEST_FIELDS`, already spelled as Python JSON (`null` when the row lacks one).
    pub latest: [String; 7],
}

pub struct LiveMetricsWriter {
    path: PathBuf,
    period: f64,
    rotate_bytes: u64,
    window_started: Instant,
    rows: Vec<LiveRow>,
}

impl LiveMetricsWriter {
    pub fn new(path: impl Into<PathBuf>, hz: f64, rotate_bytes: u64) -> io::Result<Self> {
        if hz.is_nan() || hz <= 0.0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "live metrics frequency must be positive",
            ));
        }
        Ok(Self {
            path: path.into(),
            period: 1.0 / hz,
            rotate_bytes,
            window_started: Instant::now(),
            rows: Vec::new(),
        })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn add(&mut self, row: LiveRow) -> io::Result<()> {
        self.rows.push(row);
        if self.window_started.elapsed().as_secs_f64() >= self.period {
            self.flush()?;
        }
        Ok(())
    }

    pub fn flush(&mut self) -> io::Result<()> {
        if self.rows.is_empty() {
            return Ok(());
        }
        let ended = Instant::now();
        let window = (ended - self.window_started).as_secs_f64();
        let mut payload = aggregate(&self.rows, window, unix_seconds());
        payload.push('\n');
        let size = match fs::metadata(&self.path) {
            Ok(meta) => meta.len(),
            Err(e) if e.kind() == io::ErrorKind::NotFound => 0,
            Err(e) => return Err(e),
        };
        if size > 0 && size + payload.len() as u64 > self.rotate_bytes {
            let mut rotated = self.path.clone().into_os_string();
            rotated.push(".1");
            fs::rename(&self.path, rotated)?;
        }
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.path)?;
        file.write_all(payload.as_bytes())?;
        file.flush()?;
        self.rows.clear();
        self.window_started = ended;
        Ok(())
    }

    pub fn close(&mut self) -> io::Result<()> {
        self.flush()
    }
}

/// One window record (no trailing newline). `window` is the elapsed seconds of the window.
pub fn aggregate(rows: &[LiveRow], window: f64, time_unix_seconds: f64) -> String {
    let latest = &rows[rows.len() - 1].latest;
    let window = py_max(window, 1e-9);
    let samples: i64 = rows.iter().map(|r| r.batch_samples).sum();
    let starvations = rows.iter().filter(|r| r.starved).count();
    let means: Vec<f64> = (0..MEAN_FIELDS.len())
        .map(|k| np_mean(&rows.iter().map(|r| r.means[k]).collect::<Vec<_>>()))
        .collect();
    let total_ms = 0.0 + means[4] + means[5] + means[6] + means[7];
    let mut fields: Vec<(&str, String)> = vec![
        ("schema", py_str(LIVE_SCHEMA)),
        ("time_unix_seconds", py_float(time_unix_seconds)),
        ("window_seconds", py_float(window)),
        ("steps_in_window", rows.len().to_string()),
        ("samples", samples.to_string()),
        ("queue_starvations", starvations.to_string()),
        ("samples_per_second", py_float(samples as f64 / window)),
        (
            "queue_depth",
            rows.iter()
                .filter_map(|r| r.queue_depth)
                .min()
                .map_or_else(|| "null".to_owned(), |d| d.to_string()),
        ),
        (
            "loader_wait_pct",
            py_float(100.0 * means[LOADER_WAIT] / py_max(total_ms, 1e-9)),
        ),
    ];
    fields.extend(
        MEAN_FIELDS
            .iter()
            .zip(&means)
            .map(|(k, v)| (*k, py_float(*v))),
    );
    fields.extend(
        LATEST_FIELDS
            .iter()
            .zip(latest)
            .map(|(k, v)| (*k, v.clone())),
    );
    fields.sort_by(|a, b| a.0.cmp(b.0));
    let mut out = String::from("{");
    for (i, (key, value)) in fields.iter().enumerate() {
        if i > 0 {
            out.push_str(", ");
        }
        out.push_str(&py_str(key));
        out.push_str(": ");
        out.push_str(value);
    }
    out.push('}');
    out
}

fn unix_seconds() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0.0, |d| d.as_secs_f64())
}

/// Python's `max(a, b)`: `b` only when it is strictly greater, so a NaN `a` stays NaN.
pub fn py_max(a: f64, b: f64) -> f64 {
    if b > a {
        b
    } else {
        a
    }
}

/// `np.mean` of a float64 list: `0.0 + pairwise_sum(values)`, divided by the count.
pub fn np_mean(values: &[f64]) -> f64 {
    (0.0 + pairwise_sum(values)) / values.len() as f64
}

/// numpy's `pairwise_sum` for float64: a plain loop below 8 values, eight accumulators up to the
/// 128-value block, and halves split on a multiple of 8 above it.
fn pairwise_sum(a: &[f64]) -> f64 {
    let n = a.len();
    if n < 8 {
        let mut res = 0.0;
        for v in a {
            res += v;
        }
        res
    } else if n <= 128 {
        let mut r = [0.0; 8];
        r.copy_from_slice(&a[..8]);
        let mut i = 8;
        while i < n - n % 8 {
            for (j, acc) in r.iter_mut().enumerate() {
                *acc += a[i + j];
            }
            i += 8;
        }
        let mut res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        for v in &a[i..] {
            res += v;
        }
        res
    } else {
        let mut n2 = n / 2;
        n2 -= n2 % 8;
        pairwise_sum(&a[..n2]) + pairwise_sum(&a[n2..])
    }
}

/// `repr(float)` as `json.dumps` writes it: shortest round-trip digits, exponent form outside
/// `1e-4 <= |v| < 1e16`, and `NaN` / `Infinity` for non-finite values.
pub fn py_float(v: f64) -> String {
    if v.is_nan() {
        return "NaN".to_owned();
    }
    if v.is_infinite() {
        return if v > 0.0 { "Infinity" } else { "-Infinity" }.to_owned();
    }
    let sci = format!("{v:e}");
    let (mantissa, exp) = sci.split_once('e').unwrap_or((&sci, "0"));
    let exp: i32 = exp.parse().unwrap_or(0);
    let (sign, mantissa) = match mantissa.strip_prefix('-') {
        Some(m) => ("-", m),
        None => ("", mantissa),
    };
    let digits: String = mantissa.chars().filter(|c| *c != '.').collect();
    if (-4..16).contains(&exp) {
        if exp >= 0 {
            let int_len = exp as usize + 1;
            if digits.len() <= int_len {
                format!("{sign}{digits}{}.0", "0".repeat(int_len - digits.len()))
            } else {
                format!("{sign}{}.{}", &digits[..int_len], &digits[int_len..])
            }
        } else {
            format!("{sign}0.{}{digits}", "0".repeat((-exp - 1) as usize))
        }
    } else {
        let frac = if digits.len() > 1 {
            format!(".{}", &digits[1..])
        } else {
            String::new()
        };
        let exp_sign = if exp < 0 { '-' } else { '+' };
        format!("{sign}{}{frac}e{exp_sign}{:02}", &digits[..1], exp.abs())
    }
}

/// A JSON string the way `json.dumps` (with `ensure_ascii=True`) writes it.
pub fn py_str(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{8}' => out.push_str("\\b"),
            '\u{c}' => out.push_str("\\f"),
            ' '..='~' => out.push(c),
            _ => {
                let mut units = [0_u16; 2];
                for unit in c.encode_utf16(&mut units) {
                    out.push_str(&format!("\\u{unit:04x}"));
                }
            }
        }
    }
    out.push('"');
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(samples: i64, loss: f64, wait: f64, depth: Option<i64>, step: u32) -> LiveRow {
        LiveRow {
            batch_samples: samples,
            starved: depth == Some(0),
            means: [loss, loss / 2.0, 0.25, 0.125, wait, 1.0, 2.0, 1.0, 0.5],
            queue_depth: depth,
            latest: [
                "3".to_owned(),
                step.to_string(),
                "10".to_owned(),
                (100 + step).to_string(),
                "[4, 64]".to_owned(),
                py_float(3e-4),
                "null".to_owned(),
            ],
        }
    }

    #[test]
    fn float_repr_matches_python() {
        for (v, want) in [
            (0.0, "0.0"),
            (-0.0, "-0.0"),
            (0.1, "0.1"),
            (123.0, "123.0"),
            (1234567.125, "1234567.125"),
            (0.0001, "0.0001"),
            (1e-5, "1e-05"),
            (-2.5e-7, "-2.5e-07"),
            (1e15, "1000000000000000.0"),
            (1e16, "1e+16"),
            (1.5e300, "1.5e+300"),
            (5e-324, "5e-324"),
            (f64::MAX, "1.7976931348623157e+308"),
            (f64::NAN, "NaN"),
            (f64::NEG_INFINITY, "-Infinity"),
        ] {
            assert_eq!(py_float(v), want, "{v:e}");
        }
    }

    #[test]
    fn strings_escape_like_ensure_ascii() {
        assert_eq!(
            py_str("a\"b\\c\n\u{7f}\u{e9}\u{1f600}"),
            "\"a\\\"b\\\\c\\n\\u007f\\u00e9\\ud83d\\ude00\""
        );
    }

    #[test]
    fn pairwise_sum_follows_numpy_blocks() {
        let values: Vec<f64> = (0..300).map(|i| 1.0 / (i as f64 + 3.0)).collect();
        // Below 8 values numpy adds left to right.
        assert_eq!(
            pairwise_sum(&values[..7]),
            values[..7].iter().fold(0.0, |a, v| a + v)
        );
        // Eight accumulators: lane j sums values j, j + 8, ... and the tail is added last.
        let n = 21;
        let mut lanes = [0.0; 8];
        lanes.copy_from_slice(&values[..8]);
        for (j, lane) in lanes.iter_mut().enumerate() {
            *lane += values[8 + j];
        }
        let mut want = ((lanes[0] + lanes[1]) + (lanes[2] + lanes[3]))
            + ((lanes[4] + lanes[5]) + (lanes[6] + lanes[7]));
        for v in &values[16..n] {
            want += v;
        }
        assert_eq!(pairwise_sum(&values[..n]), want);
        // Above the 128-value block the split point is n / 2 rounded down to a multiple of 8.
        assert_eq!(
            pairwise_sum(&values),
            pairwise_sum(&values[..144]) + pairwise_sum(&values[144..])
        );
        assert_eq!(np_mean(&[-0.0, -0.0]).to_bits(), 0.0_f64.to_bits());
    }

    #[test]
    fn aggregate_writes_the_python_record() {
        let rows = vec![
            row(4, 1.0, 3.0, Some(2), 1),
            row(4, 2.0, 1.0, Some(0), 2),
            row(2, 3.0, 2.0, None, 3),
        ];
        let line = aggregate(&rows, 0.5, 1_700_000_000.25);
        assert_eq!(
            line,
            "{\"batch_shape\": [4, 64], \"bradley_terry_loss\": 0.25, \"compute_ms\": 2.0, \
             \"epoch\": 3, \"eta_seconds\": null, \"global_step\": 103, \"gradient_norm\": 0.5, \
             \"h2d_ms\": 1.0, \"huber_loss\": 1.0, \"learning_rate\": 0.0003, \
             \"loader_wait_ms\": 2.0, \"loader_wait_pct\": 33.333333333333336, \
             \"optimizer_ms\": 1.0, \"queue_depth\": 0, \"queue_starvations\": 1, \
             \"samples\": 10, \"samples_per_second\": 20.0, \"schema\": \
             \"jope-prime.dime-live-metrics.v1\", \"step\": 3, \"steps\": 10, \
             \"steps_in_window\": 3, \"time_unix_seconds\": 1700000000.25, \
             \"total_loss\": 2.0, \"validity_loss\": 0.125, \"window_seconds\": 0.5}"
        );
        let tiny = aggregate(&rows[2..], 0.0, 0.0);
        assert!(tiny.contains("\"window_seconds\": 1e-09"), "{tiny}");
        assert!(tiny.contains("\"queue_depth\": null"), "{tiny}");
    }

    #[test]
    fn writer_rotates_past_the_cap() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("live-metrics.jsonl");
        assert!(LiveMetricsWriter::new(&path, 0.0, ROTATE_BYTES).is_err());
        let mut writer = LiveMetricsWriter::new(&path, 1e-6, 600).unwrap();
        writer.add(row(4, 1.0, 1.0, Some(1), 1)).unwrap();
        assert!(!path.exists(), "the window is still open");
        writer.flush().unwrap();
        writer.flush().unwrap();
        let first = fs::read_to_string(&path).unwrap();
        assert_eq!(first.lines().count(), 1);
        writer.add(row(4, 1.0, 1.0, Some(1), 2)).unwrap();
        writer.close().unwrap();
        let rotated = dir.path().join("live-metrics.jsonl.1");
        assert_eq!(fs::read_to_string(&rotated).unwrap(), first);
        let second = fs::read_to_string(&path).unwrap();
        assert_eq!(second.lines().count(), 1);
        assert!(second.contains("\"step\": 2"));
    }
}
