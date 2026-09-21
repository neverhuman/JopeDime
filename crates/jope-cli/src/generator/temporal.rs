use super::{row, Random};
use crate::schema::{Dataset, Split, ROWS};
use serde_json::Value;

pub(super) fn populate(data: &mut Dataset, scenario: usize, random: &mut Random) {
    let mut previous = [0_i64; 3];
    let mut second_previous = [0_i64; 3];
    let mut timestamp = 0;
    data.columns[0].name = if matches!(scenario, 2 | 8) {
        "lag_1"
    } else {
        "signal"
    }
    .into();
    data.columns[1].name = if scenario == 8 { "lag_2" } else { "exogenous" }.into();
    data.columns[0].nullable = scenario == 5;
    for index in 0..ROWS {
        let series = if scenario == 4 { index % 3 } else { 0 };
        let tick = if scenario == 4 { index / 3 } else { index };
        let signal = random.noise();
        let noise = random.noise();
        let shifted = signal
            + if scenario == 9 && index >= 80 {
                4000
            } else {
                0
            };
        let lag = previous[series];
        let lag2 = second_previous[series];
        let target = match scenario {
            0 => tick as i64 * 100 + noise / 8,
            1 => {
                let phase = tick as i64 % 12;
                (if phase < 6 { phase } else { 12 - phase }) * 200 - 600 + noise / 20
            }
            2 => 3 * lag / 4 + noise / 2,
            3 => 2 * signal + 3 * lag / 5,
            4 => series as i64 * 1000 + tick as i64 * 40 + noise / 4,
            7 => {
                if index < 64 {
                    signal
                } else {
                    -3 * signal + 5000
                }
            }
            8 => 2 * lag - lag2 + noise / 10,
            _ => 2 * shifted + noise / 4,
        };
        let mut features = vec![
            Value::from(if matches!(scenario, 2 | 8) {
                lag
            } else {
                shifted
            }),
            Value::from(if scenario == 8 { lag2 } else { signal }),
        ];
        if scenario == 5 && index.is_multiple_of(7) {
            features[0] = Value::Null;
        }
        let mut current = row(index, features, target);
        timestamp += match scenario {
            6 => 1 + (random.next() % 119) as i64,
            5 if index.is_multiple_of(12) => 120,
            _ => 60,
        };
        current.timestamp = Some(if scenario == 4 {
            tick as i64 * 60
        } else {
            timestamp
        });
        current.series = Some(format!("series-{series}"));
        if scenario == 4 {
            current.split = match tick {
                0..20 => Split::Train,
                20..26 => Split::Validation,
                _ => Split::Test,
            };
        }
        second_previous[series] = lag;
        previous[series] = target;
        data.rows.push(current);
    }
}
