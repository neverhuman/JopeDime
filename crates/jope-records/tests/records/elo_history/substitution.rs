use super::*;

const V1: &str = "H-same-cut-v1-deep";

fn flip_winner(data: Vec<u8>) -> Vec<u8> {
    let text = String::from_utf8(data).unwrap();
    let mut rows = pycsv::reader(&text, Newlines::Universal).unwrap();
    let best = (1..rows.len())
        .filter(|&i| rows[i][6] == "ok")
        .fold(None::<usize>, |best, i| match best {
            Some(b) if py::float(&rows[b][5]).unwrap() <= py::float(&rows[i][5]).unwrap() => {
                Some(b)
            }
            _ => Some(i),
        })
        .unwrap();
    rows[best][5] = pyjson::float_repr(py::float(&rows[best][5]).unwrap() * 10.0);
    let mut out = String::new();
    for row in &rows {
        pycsv::write_row(&mut out, row);
    }
    out.into_bytes()
}

fn swap_rows(data: Vec<u8>) -> Vec<u8> {
    let text = String::from_utf8(data).unwrap();
    let mut lines: Vec<String> = text.split_inclusive('\n').map(str::to_owned).collect();
    let first: Vec<String> = lines[1].splitn(3, ',').map(str::to_owned).collect();
    let second: Vec<String> = lines[2].splitn(3, ',').map(str::to_owned).collect();
    lines[1] = [first[0].clone(), first[1].clone(), second[2].clone()].join(",");
    lines[2] = [second[0].clone(), second[1].clone(), first[2].clone()].join(",");
    lines.concat().into_bytes()
}

fn snapshot_elo(data: Vec<u8>) -> Vec<u8> {
    let mut snapshot = pyjson::parse_bytes(&data).unwrap();
    set(nth(at(&mut snapshot, "rows"), 0), "elo", "1700.0".into());
    pyjson::dumps(&snapshot, false, None).into_bytes()
}

type Mutate = fn(&mut Copy);

#[test]
fn positive_control_and_substitutions() {
    let mutations: [(&str, Mutate, &str); 7] = [
        (
            "unchanged bytes, wrong digest",
            |c| {
                set(
                    c.item(V1, &["replay", "inputs", "scores"]),
                    "sha256",
                    "0".repeat(64).into(),
                )
            },
            "digest mismatch",
        ),
        (
            "re-pinned substituted scores",
            |c| {
                let root = c.root.clone();
                rewrite(
                    &root,
                    c.item(V1, &["replay", "inputs", "scores"]),
                    flip_winner,
                );
            },
            "differs from replay|in the committed ladder but",
        ),
        (
            "re-pinned duplicate delivery",
            |c| {
                let root = c.root.clone();
                rewrite(
                    &root,
                    c.item(V1, &["replay", "inputs", "scores"]),
                    |mut d| {
                        let first = d.split_inclusive(|b| *b == b'\n').nth(1).unwrap().to_vec();
                        d.extend(first);
                        d
                    },
                );
            },
            "duplicate delivery",
        ),
        (
            "re-pinned reordered leaderboard",
            |c| {
                let root = c.root.clone();
                rewrite(
                    &root,
                    c.item(V1, &["replay", "outputs", "leaderboard"]),
                    swap_rows,
                );
            },
            "in the committed ladder but",
        ),
        (
            "re-pinned snapshot rating",
            |c| {
                let root = c.root.clone();
                rewrite(
                    &root,
                    c.item(V1, &["replay", "outputs", "snapshot"]),
                    snapshot_elo,
                );
            },
            "differs from replay",
        ),
        (
            "input other than the track declares",
            |c| {
                let stranger = c
                    .item(
                        "H-same-cut-v2-deep",
                        &["replay", "inputs", "scores", "path"],
                    )
                    .clone();
                set(
                    c.item(V1, &["replay", "inputs", "scores"]),
                    "path",
                    stranger,
                );
            },
            "not the track's declared input",
        ),
        (
            "path traversal",
            |c| {
                let retained = c.item("H-legacy-round1-internal-elo", &["retained"]);
                set(nth(retained, 0), "path", "../elo_trajectory.csv".into());
            },
            "path traversal",
        ),
    ];
    Copy::new(fast(record()))
        .verify()
        .expect("positive control");
    for (label, mutate, reason) in mutations {
        let mut copy = Copy::new(fast(record()));
        mutate(&mut copy);
        let result = copy.verify().map(|_| ());
        assert!(result.is_err(), "{label}: accepted");
        assert_refused(result, reason);
    }
}
