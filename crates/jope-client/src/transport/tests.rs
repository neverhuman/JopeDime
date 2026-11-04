use super::*;
use serde_json::Value;

struct GoldenTransport<'a> {
    request: Option<&'a Value>,
    response: &'a Value,
}
impl Transport for GoldenTransport<'_> {
    fn execute(&self, request: Request) -> Result<Response> {
        request.validate()?;
        if let Some(expected) = self.request {
            assert_eq!(serde_json::from_slice::<Value>(request.body())?, *expected);
        } else {
            assert!(request.body().is_empty());
        }
        let mut headers = HeaderMap::new();
        headers.insert("jope-api-version", HeaderValue::from_static("1.0"));
        headers.insert(
            header::CONTENT_TYPE,
            HeaderValue::from_static("application/json"),
        );
        Ok(Response::new(
            200,
            headers,
            io::Cursor::new(serde_json::to_vec(self.response)?),
        ))
    }
}

fn exchange_goldens() -> Value {
    serde_json::from_str(include_str!(
        "../../../jope-api/tests/snapshots/exchanges.json"
    ))
    .unwrap()
}

#[test]
fn canonical_request_dtos_preserve_api_owner_golden_wire_values() {
    let goldens = exchange_goldens();
    let positive = &goldens["positive"];
    let profile = &positive["DatasetPage"][0]["datasets"][0];
    let snapshot = &positive["RunPage"][0]["runs"][0];
    let run: RunId = "run_01ARZ3NDEKTSV4RRFFQ69G5FAV".parse().unwrap();
    macro_rules! request_cases {
            ($input:ty, $output:ty, $reply:expr, $method:ident $(, $arg:expr)*) => {
                for value in positive[stringify!($input)].as_array().unwrap() {
                    let body: $input = serde_json::from_value(value.clone()).unwrap();
                    let transport = GoldenTransport { request: Some(value), response: $reply };
                    let result: $output = transport.$method($($arg,)* &body, Some("golden-retry")).unwrap();
                    assert_eq!(serde_json::to_value(result).unwrap(), *$reply);
                }
            };
        }
    request_cases!(
        DatasetRegistration,
        DatasetProfile,
        profile,
        dataset_register
    );
    request_cases!(
        ObjectiveValidationRequest,
        ObjectiveValidation,
        &positive["ObjectiveValidation"][0],
        objective_validate
    );
    request_cases!(RunControlRequest, RunSnapshot, snapshot, run_control, &run);
    request_cases!(RunForkRequest, RunSnapshot, snapshot, run_fork, &run);
    request_cases!(GcRequest, GcReport, &positive["GcReport"][0], store_gc);
}

#[test]
fn canonical_response_dtos_round_trip_and_reject_api_owner_negative_goldens() {
    let goldens = exchange_goldens();
    let run: RunId = "run_01ARZ3NDEKTSV4RRFFQ69G5FAV".parse().unwrap();
    let dataset: DatasetId = "ds_00112233445566778899aabb".parse().unwrap();
    macro_rules! response_cases {
            ($output:ty, $method:ident $(, $arg:expr)*) => {
                for value in goldens["positive"][stringify!($output)].as_array().unwrap() {
                    let transport = GoldenTransport { request: None, response: value };
                    let result: $output = transport.$method($($arg),*).unwrap();
                    assert_eq!(serde_json::to_value(result).unwrap(), *value);
                }
                for value in goldens["negative"][stringify!($output)].as_array().unwrap() {
                    let transport = GoldenTransport { request: None, response: value };
                    assert!(matches!(transport.$method($($arg),*), Err(Error::Json(_))));
                }
            };
        }
    response_cases!(DatasetPage, datasets);
    response_cases!(RunPage, runs);
    response_cases!(AlgorithmPage, algorithms, &run);
    response_cases!(DeleteReceipt, dataset_delete, &dataset, "delete-dataset");
    response_cases!(DeleteReceipt, run_delete, &run, "delete-run");
    response_cases!(
        EventPage,
        run_events,
        &run,
        jope_api::decimal::DecimalU64(0),
        256
    );
    response_cases!(TrainingFeedback, feedback_export, &run);

    // POST responses have request bodies as well as the newly typed results.
    let validation: ObjectiveValidationRequest =
        serde_json::from_value(goldens["positive"]["ObjectiveValidationRequest"][0].clone())
            .unwrap();
    for name in ["positive", "negative"] {
        for value in goldens[name]["ObjectiveValidation"].as_array().unwrap() {
            let transport = GoldenTransport {
                request: Some(&goldens["positive"]["ObjectiveValidationRequest"][0]),
                response: value,
            };
            let result: Result<ObjectiveValidation> =
                transport.objective_validate(&validation, None);
            if name == "positive" {
                assert_eq!(serde_json::to_value(result.unwrap()).unwrap(), *value);
            } else {
                assert!(matches!(result, Err(Error::Json(_))));
            }
        }
    }
    let gc: GcRequest =
        serde_json::from_value(goldens["positive"]["GcRequest"][0].clone()).unwrap();
    for value in goldens["negative"]["GcReport"].as_array().unwrap() {
        let transport = GoldenTransport {
            request: Some(&goldens["positive"]["GcRequest"][0]),
            response: value,
        };
        assert!(matches!(
            transport.store_gc(&gc, Some("gc-retry")),
            Err(Error::Json(_))
        ));
    }
}

#[test]
fn registry_routes_and_headers_are_validated_before_dispatch() {
    for op in OPERATIONS {
        for (route, template) in op.paths.iter().enumerate() {
            let mut params = vec![];
            if template.contains("{id}") {
                params.push(("id", "a/b ü"));
            }
            if template.contains("{alg}") {
                params.push(("alg", "alg_0001"));
            }
            let request = Request::new(op.id, route, &params).unwrap();
            assert_eq!(request.method, op.method);
            assert!(!request.path.contains('{'));
            assert!(!request.path.contains("a/b"));
            assert_eq!(request.validate().is_err(), op.requires_idempotency_key);
            request.key(Some("cli_test")).unwrap().validate().unwrap();
        }
    }
    assert_eq!(encode("a/b ü"), "a%2Fb%20%C3%BC");
    assert_eq!(
        OPERATIONS.len(),
        17,
        "new operations need typed client bindings"
    );
    assert_eq!(OPERATIONS.iter().map(|o| o.paths.len()).sum::<usize>(), 20);
    assert!(Request::new(7, 1, &[]).is_err());
    assert!(Request::new(1, 0, &[("id", "x")]).is_err());
    for key in ["", "line\nbreak", "\u{7f}", &"x".repeat(129)] {
        assert!(Request::new(6, 0, &[]).unwrap().key(Some(key)).is_err());
    }
    let mut duplicate = Request::new(6, 0, &[]).unwrap().key(Some("one")).unwrap();
    duplicate
        .headers
        .append("idempotency-key", HeaderValue::from_static("two"));
    assert!(duplicate.validate().is_err());
}
#[test]
fn response_requires_media_version_and_matching_problem_status() {
    let make = |status, media, bytes: &[u8]| {
        let mut headers = HeaderMap::new();
        headers.insert("jope-api-version", HeaderValue::from_static("1.0"));
        headers.insert(header::CONTENT_TYPE, HeaderValue::from_static(media));
        Response::new(status, headers, io::Cursor::new(bytes.to_vec()))
    };
    assert_eq!(
        make(200, "application/json", b"{\"ok\":true}")
            .json::<Value>()
            .unwrap()["ok"],
        true
    );
    assert!(make(200, "text/plain", b"{}").json::<Value>().is_err());
    let mut response = make(200, "application/json", b"{}");
    response.headers.remove("jope-api-version");
    assert!(response.json::<Value>().is_err());
    assert!(matches!(
        make(503, "text/plain", b"busy").json::<Value>(),
        Err(Error::Status(503))
    ));
    let problem = Problem {
        r#type: "about:blank".into(),
        title: "Invalid input".into(),
        status: 400,
        detail: "bad field".into(),
        instance: None,
        code: jope_api::error::ErrorCode::InvalidInput,
        retriable: false,
        remedy: "supply a valid field".into(),
        field: Some("field".into()),
    };
    let bytes = serde_json::to_vec(&problem).unwrap();
    assert!(
        matches!(make(400,Problem::CONTENT_TYPE,&bytes).json::<Value>(),
            Err(Error::Problem(p)) if *p==problem)
    );
    assert!(matches!(
        make(409, Problem::CONTENT_TYPE, &bytes).json::<Value>(),
        Err(Error::Protocol(_))
    ));
    assert!(make(200, "Application/JSON", b"{}").json::<Value>().is_ok());
}
