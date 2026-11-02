use super::*;

// ---- S3 Inventory (object-level facts) ----

/// One inventory row: key, latest?, delete marker?, size, last modified (RFC 3339), class.
type InvRow = (String, bool, bool, i64, &'static str, &'static str);

fn inv_columns() -> Vec<ColumnSpec> {
    let c = |name: &str, ty: ColumnType, required: bool| ColumnSpec {
        name: name.into(),
        ty,
        required,
    };
    vec![
        c("bucket", ColumnType::Utf8, true),
        c("key", ColumnType::Utf8, true),
        c("version_id", ColumnType::Utf8, false),
        c("is_latest", ColumnType::Boolean, false),
        c("is_delete_marker", ColumnType::Boolean, false),
        c("size", ColumnType::Int64, false),
        c("last_modified_date", ColumnType::TimestampMillis, false),
        c("e_tag", ColumnType::Utf8, false),
        c("storage_class", ColumnType::Utf8, false),
    ]
}

fn inv_table(bucket: &str, rows: &[InvRow]) -> Table {
    let rows = rows
        .iter()
        .enumerate()
        .map(|(i, (key, latest, dm, size, lm, class))| {
            let ms = jope_learning::civil::parse_rfc3339(lm).unwrap() as i64 * 1000;
            vec![
                Cell::Str(bucket.into()),
                Cell::Str(key.clone()),
                Cell::Str(format!("v{i}")),
                Cell::Bool(*latest),
                Cell::Bool(*dm),
                if *dm { Cell::Null } else { Cell::I64(*size) },
                Cell::I64(ms),
                if *dm {
                    Cell::Null
                } else {
                    Cell::Str(format!("{:032x}", size))
                },
                if *dm {
                    Cell::Null
                } else {
                    Cell::Str((*class).into())
                },
            ]
        })
        .collect();
    Table {
        columns: inv_columns(),
        rows,
        metadata: vec![],
    }
}

/// Writes a report the way S3 lays it out and returns its manifest path.
fn inv_report(root: &Path, bucket: &str, table: &Table) -> PathBuf {
    let opts = WriteOptions {
        codec: Codec::Snappy,
        row_group_rows: 700,
        page_rows: 300,
        dictionary: true,
    };
    let bytes = write_with(table, &opts, "fixture").unwrap();
    let cfg = root.join(format!("inventory/{bucket}/jopedime-daily-parquet"));
    put(&cfg.join("data/f1.parquet"), &bytes);
    let manifest = json!({
        "sourceBucket": bucket, "destinationBucket": "arn:aws:s3:::inv", "version": "2016-11-30",
        "creationTimestamp": "1791766800000", "fileFormat": "Parquet",
        "fileSchema": "message s3.inventory { required binary bucket (STRING); required binary key (STRING); optional binary version_id (STRING); optional boolean is_latest; optional boolean is_delete_marker; optional int64 size; optional int64 last_modified_date (TIMESTAMP(MILLIS,true)); optional binary e_tag (STRING); optional binary storage_class (STRING);}",
        "files": [{"key": format!("inventory/{bucket}/{bucket}/jopedime-daily-parquet/data/f1.parquet"), "size": bytes.len(), "MD5checksum": md5_hex(&bytes)}]
    });
    let m = serde_json::to_vec(&manifest).unwrap();
    let mpath = cfg.join("2026-10-12T01-00Z/manifest.json");
    put(&mpath, &m);
    put(
        &cfg.join("2026-10-12T01-00Z/manifest.checksum"),
        format!("{}\n", md5_hex(&m)).as_bytes(),
    );
    mpath
}

fn learn_inventory_rows() -> Vec<InvRow> {
    let o = |k: String, size: i64, lm: &'static str, class: &'static str| {
        (k, true, false, size, lm, class)
    };
    let mut rows = vec![
        o(
            "experiments/a.txt".into(),
            100,
            "2026-09-01T00:00:00Z",
            "STANDARD",
        ),
        o(
            "experiments/b.txt".into(),
            100,
            "2026-09-01T00:00:00Z",
            "STANDARD",
        ),
        // frozen24's bucketless pin: its latest version is a delete marker.
        (
            "experiments/gp-teacher-v6/contexts/aaaa/obj.json".into(),
            true,
            true,
            0,
            "2026-10-02T00:00:00Z",
            "",
        ),
        o(
            format!("experiments/gp-policy-dl/v2-trainonly/gp-hwm/ds/f0/{C1}/{P1}.json"),
            2_000,
            "2026-10-07T00:00:00Z",
            "STANDARD",
        ),
    ];
    // program-results: 100 current objects, none of them the val263 pin obj1, plus 5 old versions.
    for i in 0..100 {
        rows.push(o(
            format!("experiments/gp-policy-dl/v2-trainonly/program-results/p{i}"),
            100_000,
            "2026-10-09T00:00:00Z",
            "STANDARD",
        ));
    }
    for _ in 0..5 {
        rows.push((
            "experiments/gp-policy-dl/v2-trainonly/program-results/p0".into(),
            false,
            false,
            1_000_000,
            "2026-10-01T00:00:00Z",
            "STANDARD",
        ));
    }
    for i in 0..4 {
        rows.push(o(
            format!("experiments/mystery/m{i}"),
            1_000,
            "2026-09-25T00:00:00Z",
            "STANDARD",
        ));
    }
    rows.push(o(
        "experiments/mystery/file.json".into(),
        1_000,
        "2026-09-25T00:00:00Z",
        "STANDARD",
    ));
    for i in 0..2 {
        rows.push(o(
            format!("staging/lineage-a/preserved-archives-20260908-F-D-E/t{i}.tar"),
            10_000_000_000,
            "2026-09-18T00:00:00Z",
            "DEEP_ARCHIVE",
        ));
    }
    // The rest of aaaa and gp-hwm, so their ledger counts match the report.
    for i in 0..1000 {
        rows.push(o(
            format!("experiments/gp-teacher-v6/contexts/aaaa/k{i}"),
            50_000,
            "2026-10-01T00:00:00Z",
            "STANDARD",
        ));
    }
    for i in 0..49 {
        rows.push(o(
            format!("experiments/gp-policy-dl/v2-trainonly/gp-hwm/ds/f0/extra{i}.json"),
            2_000,
            "2026-10-07T00:00:00Z",
            "STANDARD",
        ));
    }
    // bbbb was written after the report (ledger: 10 objects, last write 2026-10-13); 9 are here.
    for i in 0..9 {
        rows.push(o(
            format!("experiments/gp-teacher-v6/contexts/bbbb/y{i}"),
            100_000,
            "2026-10-10T00:00:00Z",
            "STANDARD",
        ));
    }
    // cccc: the ledger has 2000 objects, the report 1999 and no delete marker, yet the ledger's
    // last write (2026-10-01) predates the report. No later write explains that: a conflict.
    for i in 0..1999 {
        rows.push(o(
            format!("experiments/gp-teacher-v6/contexts/cccc/x{i}"),
            30_000,
            "2026-10-01T00:00:00Z",
            "STANDARD",
        ));
    }
    rows
}

#[test]
fn streaming_reader_decodes_inventory_shaped_files() {
    let table = inv_table(LEARN, &learn_inventory_rows());
    let opts = WriteOptions {
        codec: Codec::Snappy,
        row_group_rows: 700,
        page_rows: 300,
        dictionary: true,
    };
    let bytes = write_with(&table, &opts, "fixture").unwrap();
    let r = FileReader::open(&bytes).unwrap();
    assert_eq!(r.num_rows, table.rows.len() as u64);
    assert_eq!(r.row_groups.len(), table.rows.len().div_ceil(700));
    assert_eq!(r.columns[6].timestamp_ms_divisor, Some(1));
    let mut pages = PageCounts::default();
    for (ci, _) in table.columns.iter().enumerate() {
        let mut got: Vec<Cell> = Vec::new();
        for g in 0..r.row_groups.len() {
            let (v, pc) = r.read_column(g, ci).unwrap();
            pages.add(&pc);
            match v {
                Values::Bool(x) => {
                    got.extend(x.into_iter().map(|b| b.map_or(Cell::Null, Cell::Bool)))
                }
                Values::I64(x) => {
                    got.extend(x.into_iter().map(|b| b.map_or(Cell::Null, Cell::I64)))
                }
                Values::F64(x) => {
                    got.extend(x.into_iter().map(|b| b.map_or(Cell::Null, Cell::F64)))
                }
                Values::Bytes(x) => {
                    got.extend(x.into_iter().map(|b| {
                        b.map_or(Cell::Null, |b| Cell::Str(String::from_utf8(b).unwrap()))
                    }))
                }
            }
        }
        let want: Vec<Cell> = table.rows.iter().map(|row| row[ci].clone()).collect();
        assert_eq!(got, want, "column {}", table.columns[ci].name);
    }
    assert!(pages.dictionary > 0 && pages.data_v1 > r.row_groups.len() as u64);
    assert_eq!(pages.codecs, [1].into_iter().collect(), "Snappy");
    assert_eq!(
        pages.encodings,
        [0, 2].into_iter().collect(),
        "PLAIN + PLAIN_DICTIONARY"
    );
    // The default layout still reads with the one-row-group reader.
    let plain = write_parquet(&table, false, "fixture").unwrap();
    assert_eq!(read(&plain).unwrap().rows, table.rows);
}

#[test]
fn inventory_report_adds_object_facts_and_pin_checks() {
    let f = fixture();
    let manifest = inv_report(&f.root, LEARN, &inv_table(LEARN, &learn_inventory_rows()));
    let mut cfg = config(&f);
    let without = build(&cfg).unwrap();
    cfg.inventories = vec![spec("inventory-manifest", manifest)];
    let b = build(&cfg).unwrap();
    let l = |p: &str| format!("s3://{LEARN}/{p}");

    // Exact dates: the direct objects under experiments/ are old, so they are not "live" any more.
    let d = row(&b, &l("experiments/#direct"));
    let inv = d.inventory.as_ref().unwrap();
    assert!(inv.unchanged_since_report);
    assert!(!d.live, "{}", d.reason);
    assert!(
        d.reason
            .contains("direct objects last written 2026-09-01T00:00:00Z"),
        "{}",
        d.reason
    );
    assert!(row(&without, &l("experiments/#direct"))
        .reason
        .contains("dates are unknown until S3 Inventory"));

    // Exact classes: the archive is already in Deep Archive.
    let a = row(
        &b,
        &l("staging/lineage-a/preserved-archives-20260908-F-D-E/"),
    );
    assert_eq!(a.storage_class_basis, "inventory");
    assert_eq!(a.cost.class_now, "DEEP_ARCHIVE");
    assert!(
        (a.cost.now_usd_mo - 0.0198).abs() < 1e-9,
        "{}",
        a.cost.now_usd_mo
    );
    assert!(
        row(
            &without,
            &l("staging/lineage-a/preserved-archives-20260908-F-D-E/")
        )
        .cost
        .now_usd_mo
            > 0.4
    );

    // A group written after the report keeps prefix-level reasoning.
    let bb = l("experiments/gp-teacher-v6/contexts/bbbb/");
    let c = row(&b, &bb);
    assert!(!c.inventory.as_ref().unwrap().unchanged_since_report);
    assert_eq!(c.storage_class_basis, "inventory-changed-since-report");
    assert_eq!(c.action, row(&without, &bb).action);
    assert!(
        c.reason.contains("inventory changed since report"),
        "{}",
        c.reason
    );

    // A difference no later write explains: one input is wrong, so COMPACT becomes HOLD and the
    // group is a hazard. Only cccc is such a group, and only it carries the row field.
    let cc = l("experiments/gp-teacher-v6/contexts/cccc/");
    let c = row(&b, &cc);
    assert_eq!(row(&without, &cc).action, Action::Compact);
    assert_eq!(c.action, Action::Hold, "{}", c.reason);
    assert_eq!(c.storage_class_basis, "inventory-ledger-conflict");
    assert!(
        c.reason.contains("disagree with no write after the report")
            && c.reason
                .contains("disagrees with the ledger without a later write"),
        "{}",
        c.reason
    );
    let flagged: Vec<String> = b
        .rows
        .iter()
        .filter_map(|r| {
            let v = serde_json::to_value(r).unwrap();
            (v["inventory"]["ledger_conflict"] == json!(true)).then(|| r.group_id.clone())
        })
        .collect();
    assert_eq!(flagged, vec![cc.clone()]);
    assert!(b
        .summary
        .hazards
        .iter()
        .any(|h| h.kind == "ledger-inventory-conflict" && h.groups == 1));
    let sv = serde_json::to_value(&b.summary).unwrap();
    assert!(
        sv["inventory"]["ledger_conflicts"][0]
            .as_str()
            .is_some_and(
                |s| s.contains("ledger 2000 obj") && s.contains("inventory 1999 current obj")
            )
    );

    // Noncurrent versions: counted per group; inside T0 groups they are a D2 note.
    let pr = row(
        &b,
        &l("experiments/gp-policy-dl/v2-trainonly/program-results/"),
    );
    assert_eq!(pr.action, Action::KeepT0);
    assert_eq!(pr.inventory.as_ref().unwrap().facts.noncurrent_versions, 5);
    assert!(pr
        .e_candidates
        .iter()
        .any(|e| e.starts_with("E5: 5 noncurrent versions")));
    assert!(b
        .summary
        .notes
        .iter()
        .any(|n| n.kind == "t0-noncurrent-versions" && n.gb > 0.0049));

    // Pins: val263 obj1 is absent (hazard), frozen24's key is a delete marker (hazard).
    let s = b.summary.inventory.as_ref().unwrap();
    assert_eq!(s.pins["val263"]["absent"], 1);
    assert!(
        s.pins_not_present_sample
            .iter()
            .any(|p| p.contains("program-results/obj1") && p.contains("validation.jsonl")),
        "{:?}",
        s.pins_not_present_sample
    );
    assert_eq!(s.pins["train"]["present"], 1);
    assert_eq!(s.pins["frozen24"]["deleted"], 1);
    assert_eq!(s.pins["main-manifest"]["present"], 1);
    let kinds: Vec<&str> = b.summary.hazards.iter().map(|h| h.kind.as_str()).collect();
    assert!(kinds.contains(&"t0-pin-not-present"), "{kinds:?}");
    assert!(kinds.contains(&"active-pin-not-present"), "{kinds:?}");
    assert!(
        without.summary.hazards.is_empty()
            || !without
                .summary
                .hazards
                .iter()
                .any(|h| h.kind.contains("pin"))
    );

    // Totals per bucket, and buckets without a report.
    let lb = &s.buckets[0];
    assert_eq!(
        (
            lb.rows,
            lb.current_objects,
            lb.noncurrent_versions,
            lb.delete_markers
        ),
        (3173, 3167, 5, 1)
    );
    assert_eq!(
        s.ledger_buckets_without_report,
        vec![SEALED.to_string(), DATA.to_string()],
        "in-scope buckets without a report, sorted"
    );
    assert_eq!(b.lock.inventory.len(), 1);
    assert!(b.lock.inventory[0]
        .manifest
        .check
        .contains("manifest.checksum MD5 ok"));
    assert!(b
        .summary
        .granularity
        .contains("S3 Inventory reports (2026-10-12T01:00:00Z)"));
    assert_eq!(
        b.summary.by_action["EXPIRE"].groups, 0,
        "inventory proves no E-rule"
    );
    // Without --inventory nothing about the output changes shape.
    assert!(
        without.rows.iter().all(|r| r.inventory.is_none()) && without.summary.inventory.is_none()
    );
}

#[test]
fn inventory_inputs_fail_closed() {
    let f = fixture();
    let rows = learn_inventory_rows();
    let manifest = inv_report(&f.root, LEARN, &inv_table(LEARN, &rows));
    let check = manifest.with_file_name("manifest.checksum");
    let data = manifest
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("data/f1.parquet");
    let ok_check = fs::read(&check).unwrap();
    let ok_data = fs::read(&data).unwrap();
    load_report(&manifest, None).unwrap();

    put(&check, b"00000000000000000000000000000000\n");
    assert!(
        load_report(&manifest, None).is_err(),
        "manifest checksum mismatch"
    );
    put(&check, &ok_check);

    let mut bad = ok_data.clone();
    let mid = bad.len() / 2;
    bad[mid] ^= 1;
    put(&data, &bad);
    assert!(
        load_report(&manifest, None).is_err(),
        "data file MD5 mismatch"
    );
    put(&data, &ok_data);
    assert!(
        load_report(&manifest, Some("ffffffff")).is_err(),
        "manifest sha256 pin"
    );

    // A row for another bucket inside this bucket's report.
    let mut cfg = config(&f);
    let mut t = inv_table(LEARN, &rows);
    t.rows[3][0] = Cell::Str(DATA.into());
    let m2 = inv_report(&f.path("other"), LEARN, &t);
    cfg.inventories = vec![spec("inventory-manifest", m2)];
    assert!(build(&cfg).is_err(), "foreign bucket row");

    // A report without a column the brain reads.
    let mut t = inv_table(LEARN, &rows);
    t.columns.pop();
    for r in &mut t.rows {
        r.pop();
    }
    let m3 = inv_report(&f.path("nocol"), LEARN, &t);
    cfg.inventories = vec![spec("inventory-manifest", m3)];
    assert!(build(&cfg).is_err(), "missing storage_class column");

    // Two reports for one bucket.
    cfg.inventories = vec![
        spec("inventory-manifest", manifest.clone()),
        spec("inventory-manifest", manifest.clone()),
    ];
    assert!(build(&cfg).is_err(), "duplicate bucket report");

    // Not Parquet.
    let mut v: Value = serde_json::from_slice(&fs::read(&manifest).unwrap()).unwrap();
    v["fileFormat"] = json!("CSV");
    let m = serde_json::to_vec(&v).unwrap();
    put(&manifest, &m);
    put(&check, format!("{}\n", md5_hex(&m)).as_bytes());
    assert!(load_report(&manifest, None).is_err(), "CSV report");
}
