use super::{columns, row, Random};
use crate::schema::{Dataset, Kind, Task, ROWS};
use serde_json::Value;

const COLORS: [&str; 3] = ["red", "green", "blue"];

pub(super) fn populate(data: &mut Dataset, scenario: usize, random: &mut Random) {
    let task = data.task;
    let wide =
        (task == Task::Regression && scenario == 4) || (task == Task::Multiclass && scenario == 7);
    if wide {
        data.columns = columns(12);
    }
    if task == Task::Regression && scenario == 8 {
        data.columns = columns(3);
    }
    let categorical = matches!(
        (task, scenario),
        (Task::Binary, 4) | (Task::Regression, 5) | (Task::Multiclass, 5)
    );
    let missing = matches!(
        (task, scenario),
        (Task::Binary, 5) | (Task::Regression, 6) | (Task::Multiclass, 6)
    );
    if categorical {
        data.columns[1].kind = Kind::Categorical;
        data.columns[1].categories = COLORS.iter().map(|color| (*color).into()).collect();
    }
    data.columns[1].nullable = missing;
    data.classes = match task {
        Task::Binary => vec![0, 1],
        Task::Multiclass => (0..match scenario {
            1 => 7,
            4 => 4,
            8 => 5,
            _ => 3,
        })
            .collect(),
        _ => Vec::new(),
    };
    for index in 0..ROWS {
        let mut values: Vec<i64> = (0..data.columns.len()).map(|_| random.noise()).collect();
        if task == Task::Binary && scenario == 6 {
            values[1] = 42;
        }
        if task == Task::Binary && scenario == 8 && index.is_multiple_of(12) {
            values[0] *= 30;
        }
        if task == Task::Regression && scenario == 8 {
            values[1] = 2 * values[0];
            values[2] = 3 * values[0];
        }
        if task == Task::Regression && scenario == 9 {
            values[0] = match index {
                0..64 => -(index as i64) * 16,
                64..80 => 100 + (index as i64 - 64) * 25,
                _ => 1000 + (index as i64 - 80) * 64,
            };
        }
        let noise = random.noise();
        let category = index % COLORS.len();
        let target = match task {
            Task::Binary => binary_target(scenario, index, &values, noise, category),
            Task::Regression => regression_target(scenario, index, &values, noise, category),
            Task::Multiclass => multiclass_target(scenario, index, &values, noise, category),
            Task::TimeSeries => unreachable!("time series has a separate generator"),
        };
        let mut features: Vec<Value> = values.into_iter().map(Value::from).collect();
        if categorical {
            features[1] = COLORS[category].into();
        }
        if missing && index.is_multiple_of(7) {
            features[1] = Value::Null;
        }
        let mut current = row(index, features, target);
        if matches!(task, Task::Binary | Task::Multiclass) && scenario == 9 {
            current.group = Some(format!("group-{:02}", index / 4));
        }
        if task == Task::Binary && scenario == 7 && index % 2 == 1 {
            current.features.clone_from(&data.rows[index - 1].features);
            current.target = data.rows[index - 1].target;
        }
        data.rows.push(current);
    }
}

fn binary_target(scenario: usize, index: usize, x: &[i64], noise: i64, category: usize) -> i64 {
    i64::from(match scenario {
        1 => x[0] * x[0] + x[1] * x[1] > 600_000,
        2 => index.is_multiple_of(12),
        3 => x[0] + 3 * noise > 0,
        4 => category == 0,
        9 => x[0] + (index as i64 / 4 - 12) * 40 > 0,
        _ => x[0] + x[1] > 0,
    })
}

fn regression_target(scenario: usize, index: usize, x: &[i64], noise: i64, category: usize) -> i64 {
    let linear = 3 * x[0] - 2 * x[1];
    match scenario {
        1 => x[0] * x[0] / 1000 + 2 * x[1],
        2 => linear + noise * x[0].abs() / 1000,
        // A bounded heavy-tail stress sample: sparse large impulses among small residuals.
        3 => {
            linear
                + if index.is_multiple_of(11) {
                    noise * 50
                } else {
                    noise / 10
                }
        }
        4 => 3 * x[0] - 2 * x[11],
        5 => 3 * x[0] + category as i64 * 2000,
        7 => 7,
        _ => linear,
    }
}

fn multiclass_target(scenario: usize, index: usize, x: &[i64], noise: i64, category: usize) -> i64 {
    match scenario {
        1 => (index % 7) as i64,
        2 => {
            if index.is_multiple_of(12) {
                2
            } else {
                i64::from(index.is_multiple_of(4))
            }
        }
        3 => (x[0] + 3 * noise + 4000) / 2667,
        4 => match x[0] * x[0] + x[1] * x[1] {
            0..250_000 => 0,
            250_000..600_000 => 1,
            600_000..1_000_000 => 2,
            _ => 3,
        },
        5 => category as i64,
        8 => {
            if matches!(index, 0 | 32 | 64 | 80) {
                4
            } else {
                (index % 4) as i64
            }
        }
        _ => (index % 3) as i64,
    }
}
