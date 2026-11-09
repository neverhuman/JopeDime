//! A mock veox Cloud Workspace speaking the documented connected-CLI contract (ai-veox-app
//! `docs/connected-cli.md`): plain HTTP on 127.0.0.1, one request per connection, every request
//! recorded for the test to inspect.
#![allow(dead_code)]
use std::{
    collections::BTreeMap,
    io::{Read, Write},
    net::TcpListener,
    sync::{Arc, Mutex},
    thread,
    time::Duration,
};

#[derive(Clone, Debug)]
pub struct Recorded {
    pub method: String,
    pub path: String,
    pub headers: BTreeMap<String, String>,
    pub body: Vec<u8>,
}

impl Recorded {
    pub fn text(&self) -> String {
        String::from_utf8_lossy(&self.body).into_owned()
    }
}

pub type Handler = dyn Fn(&Recorded, &[Recorded]) -> (u16, String) + Send + Sync;

pub struct Workspace {
    pub origin: String,
    pub log: Arc<Mutex<Vec<Recorded>>>,
}

impl Workspace {
    pub fn start(handler: Box<Handler>) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let origin = format!("http://{}", listener.local_addr().unwrap());
        let log = Arc::new(Mutex::new(Vec::<Recorded>::new()));
        let shared = Arc::clone(&log);
        thread::spawn(move || {
            for stream in listener.incoming() {
                let Ok(mut stream) = stream else { return };
                let _ = stream.set_read_timeout(Some(Duration::from_secs(10)));
                let mut head = Vec::new();
                let mut byte = [0u8];
                while !head.ends_with(b"\r\n\r\n") {
                    match stream.read(&mut byte) {
                        Ok(1) => head.push(byte[0]),
                        _ => break,
                    }
                }
                let text = String::from_utf8_lossy(&head).into_owned();
                let mut lines = text.split("\r\n");
                let mut first = lines.next().unwrap_or("").split_whitespace();
                let method = first.next().unwrap_or("").to_owned();
                let path = first.next().unwrap_or("").to_owned();
                let headers: BTreeMap<String, String> = lines
                    .filter_map(|l| l.split_once(':'))
                    .map(|(k, v)| (k.trim().to_ascii_lowercase(), v.trim().to_owned()))
                    .collect();
                let length: usize = headers
                    .get("content-length")
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(0);
                let mut body = vec![0u8; length];
                let _ = stream.read_exact(&mut body);
                let request = Recorded {
                    method,
                    path,
                    headers,
                    body,
                };
                let (status, reply) = {
                    let previous = shared.lock().unwrap().clone();
                    handler(&request, &previous)
                };
                shared.lock().unwrap().push(request);
                let _ = write!(
                    stream,
                    "HTTP/1.1 {status} X\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{reply}",
                    reply.len()
                );
            }
        });
        Self { origin, log }
    }

    pub fn requests(&self) -> Vec<Recorded> {
        self.log.lock().unwrap().clone()
    }

    /// `METHOD path` of every request, in order.
    pub fn calls(&self) -> Vec<String> {
        self.requests()
            .iter()
            .map(|r| format!("{} {}", r.method, r.path))
            .collect()
    }
}

pub const ACCOUNT: &str = "6f2c1a7e-3b4d-4e5f-8a9b-0c1d2e3f4a5b";
pub const CLOUD_RUN: &str = "a1b2c3d4-e5f6-4789-8abc-def012345678";
pub const SESSION: &str = "11111111-2222-4333-8444-555555555555";

pub fn account(tier: &str) -> String {
    format!(
        r#"{{"schema":"jain-cli-account/v1","accountId":"{ACCOUNT}","email":"ada@example.test","tier":"{tier}","mayRun":true,"termsVersion":"2026-11-03","termsCurrent":true,"mayWrite":true,"mayPublish":true,"localEngine":true}}"#
    )
}
