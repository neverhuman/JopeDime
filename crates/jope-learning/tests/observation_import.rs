use jope_learning::canon::sha256_hex;
use jope_learning::replay::index_producer::{
    self, HistoricalFixture, HistoricalStatus, ProducerClosure, ProducerInput, ProducerSpec,
};
use jope_learning::replay::observation::{
    DependencyEvidence, Disposition, DurationObservation, Measurement, ObservationArchive,
    ObservedOutcome,
};
use jope_learning::retention::{
    distilled,
    parquet::{
        self, Cell, Codec, ColumnSpec, ColumnType, FileReader, StatisticValue, Table, Values,
        WriteOptions,
    },
};
use serde_json::{json, Value};
use std::collections::{BTreeMap, BTreeSet};

const INDEX: &[u8] = include_bytes!("fixtures/observation-import-v2/index.json");
const SPEC: &[u8] = include_bytes!("fixtures/observation-import-v2/producer-spec.json");
fn known() -> BTreeSet<String> {
    BTreeSet::from(["validation_rmse".into()])
}
fn spec() -> ProducerSpec {
    serde_json::from_slice(SPEC).unwrap()
}
fn import(bytes: &[u8]) -> ObservationArchive {
    distilled::import_gp_results_index_complete(bytes, None, &known(), &BTreeMap::new(), None)
        .unwrap()
}
fn manifest(bytes: &[u8], producer: ProducerSpec) -> Vec<u8> {
    serde_json::to_vec(&index_producer::produce_manifest(bytes, producer).unwrap()).unwrap()
}
fn bound(bytes: &[u8], manifest: &[u8], pin: &str) -> ObservationArchive {
    distilled::import_gp_results_index_complete(
        bytes,
        Some(ProducerInput {
            manifest_bytes: manifest,
            authenticated_manifest_sha256: Some(pin),
        }),
        &known(),
        &BTreeMap::new(),
        None,
    )
    .unwrap()
}
fn opts() -> WriteOptions {
    WriteOptions {
        codec: Codec::Zstd,
        row_group_rows: 3,
        page_rows: 1,
        dictionary: true,
    }
}

include!("observation_import/admission.rs");
include!("observation_import/archive.rs");
include!("observation_import/fixture.rs");
