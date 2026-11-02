//! Real CLI, terminal PTY, MCP reconnect and engine crash acceptance.
//! The generated Python fixtures use only the standard library and real processes.

use std::{fs, path::PathBuf, process::Command};

pub(super) fn run() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let runtime_base = std::env::var_os("JOPE_INT02_RUNTIME_ROOT")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(std::env::var_os("HOME").expect("HOME")).join(".local/state/jope-int02")
        });
    fs::create_dir_all(&runtime_base).unwrap();
    let scratch = tempfile::Builder::new()
        .prefix("i-")
        .tempdir_in(&runtime_base)
        .unwrap();
    let out = std::env::var_os("JOPE_INT02_EVIDENCE_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| scratch.path().join("evidence"));
    fs::create_dir_all(&out).unwrap();
    let driver = scratch.path().join("live_probe.py");
    fs::write(&driver, DRIVER).unwrap();
    fs::write(scratch.path().join("observe_http.py"), OBSERVER).unwrap();
    let binary = std::env::var_os("JOPE_INT02_BINARY")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(env!("CARGO_BIN_EXE_jopedime")));
    let version = Command::new(&binary)
        .args(["--version", "--json"])
        .output()
        .unwrap();
    assert!(version.status.success());
    let version: serde_json::Value = serde_json::from_slice(&version.stdout).unwrap();
    let source = std::env::var("JOPE_INT02_SOURCE_SHA")
        .unwrap_or_else(|_| version["source_sha40"].as_str().unwrap().to_owned());
    let mut command = Command::new("python3");
    command
        .arg(&driver)
        .arg("--binary")
        .arg(&binary)
        .arg("--source")
        .arg(&source)
        .arg("--source-root")
        .arg(&root)
        .arg("--out")
        .arg(&out)
        .arg("--runtime-root")
        .arg(scratch.path().join("runtime"))
        .args([
            "--all-live",
            "--capture-sse",
            "--train-rows",
            "96",
            "--s2-evaluations",
            "2",
            "--wall-seconds",
            "300",
            "--assert-model-identity",
            "--assert-s2",
        ]);
    if std::env::var("INT02_ENGINE_BOUNDARY_CUTS").as_deref() == Ok("1") {
        command.arg("--boundary-matrix");
    }
    if std::env::var("INT02_SAME_STORE").as_deref() == Ok("1") {
        command.arg("--same-store");
    }
    let output = command
        .output()
        .expect("native INT-02 acceptance requires python3");
    fs::write(out.join("driver.stdout"), &output.stdout).unwrap();
    fs::write(out.join("driver.stderr"), &output.stderr).unwrap();
    if std::env::var("JOPE_INT02_EVIDENCE_DIR").is_ok() {
        // Retain owned per-run stores with the external qualification receipts.
        let path = scratch.keep();
        fs::write(
            out.join("runtime-location.txt"),
            path.to_string_lossy().as_bytes(),
        )
        .unwrap();
    }
    assert!(
        output.status.success(),
        "INT-02 live acceptance failed: {}\n{}\nreceipts: {}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr),
        out.display()
    );
    let report: serde_json::Value =
        serde_json::from_slice(&fs::read(out.join("report.json")).unwrap()).unwrap();
    for gate in [
        "projection_oracle_controls",
        "r4_qualified",
        "s2_qualified",
        "r1_coarse_native_pass",
    ] {
        assert_eq!(report[gate], true, "missing actual acceptance: {gate}");
    }
    assert_eq!(
        report["driver_qualified"], true,
        "driver did not qualify its required gates"
    );
    assert!(report["cleanup_errors"]
        .as_array()
        .is_some_and(Vec::is_empty));
    eprintln!(
        "INT-02 R4/S2/R1 actual native acceptance PASS; receipts {}",
        out.display()
    );
}

const DRIVER: &str = concat!(
    include_str!("live_driver/driver_part1.py"),
    include_str!("live_driver/driver_part2.py"),
);

const OBSERVER: &str = r####""""Transparent loopback observer; retain response events, never auth headers."""
import http.client
import http.server
import json
import os
import threading
import urllib.parse
from pathlib import Path

class Observer:
    def __init__(self, home):
        self.home = Path(home)
        record = json.loads((self.home/'daemon.json').read_text())
        address = urllib.parse.urlsplit(record['url'])
        assert address.hostname == '127.0.0.1'
        upstream_port = address.port
        self.streams = []
        owner = self
        class Handler(http.server.BaseHTTPRequestHandler):
            protocol_version = 'HTTP/1.1'
            def log_message(self, *args):
                pass
            def relay(self):
                connection = http.client.HTTPConnection('127.0.0.1', upstream_port, timeout=90)
                headers = {k:v for k,v in self.headers.items() if k.lower() not in ['host','connection']}
                headers['Host'] = f'127.0.0.1:{upstream_port}'
                body = self.rfile.read(int(self.headers.get('Content-Length','0')))
                connection.request(self.command,self.path,body or None,headers)
                response = connection.getresponse()
                sse = 'text/event-stream' in response.getheader('Content-Type','')
                self.send_response(response.status)
                for key,value in response.getheaders():
                    if key.lower() not in ['content-length','transfer-encoding','connection']:
                        self.send_header(key,value)
                self.send_header('Connection','close')
                self.close_connection = True
                if not sse:
                    data=response.read();self.send_header('Content-Length',str(len(data)))
                    self.end_headers();self.wfile.write(data);connection.close();return
                self.send_header('Transfer-Encoding','chunked');self.end_headers()
                trace={'path':self.path,'after':self.headers.get('Last-Event-ID'),
                       'events':[],'complete':False,'client_closed':False}
                owner.streams.append(trace)
                frame=[]
                try:
                    while True:
                        line=response.readline()
                        if not line:break
                        self.wfile.write(f'{len(line):x}\r\n'.encode()+line+b'\r\n');self.wfile.flush()
                        if line.strip():
                            if line.startswith(b'data:'):frame.append(line[5:].strip())
                        elif frame:
                            value=json.loads(b'\n'.join(frame));frame=[]
                            if 'sequence' in value:trace['events'].append(value)
                    self.wfile.write(b'0\r\n\r\n');self.wfile.flush();trace['complete']=True
                except (BrokenPipeError,ConnectionResetError):
                    trace['client_closed']=True
                finally:
                    connection.close()
            do_GET=relay
            do_POST=relay
            do_DELETE=relay
        self.server=http.server.ThreadingHTTPServer(('127.0.0.1',0),Handler)
        self.server.daemon_threads=True
        self.thread=threading.Thread(target=self.server.serve_forever,daemon=True)
        self.thread.start()
        # Only this isolated engine's private descriptor changes. Upstream bearer
        # remains in memory and the private daemon file, never in receipts.
        record['url']=f'http://127.0.0.1:{self.server.server_port}'
        temporary=self.home/'observer-descriptor.tmp'
        fd=os.open(temporary,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
        with os.fdopen(fd,'w') as stream:json.dump(record,stream)
        os.replace(temporary,self.home/'daemon.json')
    def close(self):
        self.server.shutdown();self.server.server_close();self.thread.join(timeout=3)
"####;
