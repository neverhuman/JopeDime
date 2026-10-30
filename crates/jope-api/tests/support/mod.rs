use jope_api::*;
use schemars::{schema_for, JsonSchema};
use serde_json::{Map, Value};

pub fn schemas() -> Value {
    fn add<T: JsonSchema>(map: &mut Map<String, Value>, name: &str) {
        map.insert(name.into(), serde_json::to_value(schema_for!(T)).unwrap());
    }
    let mut map = Map::new();
    add::<RunSpec>(&mut map, "RunSpec");
    add::<Plan>(&mut map, "Plan");
    add::<RunSnapshot>(&mut map, "RunSnapshot");
    add::<PublicEvent>(&mut map, "PublicEvent");
    add::<TelemetryEvent>(&mut map, "TelemetryEvent");
    add::<Algorithm>(&mut map, "Algorithm");
    add::<ExportRequest>(&mut map, "ExportRequest");
    add::<ExportManifest>(&mut map, "ExportManifest");
    add::<PredictRequest>(&mut map, "PredictRequest");
    add::<SystemInfo>(&mut map, "SystemInfo");
    add::<DatasetProfile>(&mut map, "DatasetProfile");
    add::<Warning>(&mut map, "Warning");
    add::<MetricSample>(&mut map, "MetricSample");
    Value::Object(map)
}
