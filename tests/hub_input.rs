//! `modelq quantize hf:<owner>/<name>` against a local stand-in for the Hub.
//!
//! A small HTTP server on 127.0.0.1 serves model metadata and files, supports
//! Range requests, counts the requests it receives and can require a token.
//! No test contacts the real Hub.

use std::{
    collections::HashMap,
    env, fs,
    io::{BufRead, BufReader, Write},
    net::{Shutdown, TcpListener, TcpStream},
    path::{Path, PathBuf},
    process::{Command, Output, id},
    sync::{
        Arc, Mutex,
        atomic::{AtomicU64, Ordering},
    },
    thread,
};

use serde_json::{Map, Value, json};
use sha1::Sha1;
use sha2::{Digest, Sha256};

const COMMIT: &str = "1111111111111111111111111111111111111111";
const TOKEN: &str = "test-token-value";

static NEXT_ID: AtomicU64 = AtomicU64::new(0);

struct TestDir(PathBuf);

impl TestDir {
    fn new(label: &str) -> Self {
        let serial = NEXT_ID.fetch_add(1, Ordering::Relaxed);
        let path = env::temp_dir().join(format!("modelq-hub-{label}-{}-{serial}", id()));
        let _ = fs::remove_dir_all(&path);
        fs::create_dir_all(&path).unwrap();
        Self(path)
    }

    fn join(&self, name: &str) -> PathBuf {
        self.0.join(name)
    }
}

impl Drop for TestDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

/// A SafeTensors file with the given F32 tensors.
fn safetensors(tensors: &[(&str, Vec<usize>, Vec<f32>)]) -> Vec<u8> {
    let mut header = Map::new();
    let mut data = Vec::new();
    for (name, shape, values) in tensors {
        let begin = data.len();
        for value in values {
            data.extend_from_slice(&value.to_le_bytes());
        }
        header.insert(
            (*name).to_owned(),
            json!({"dtype": "F32", "shape": shape, "data_offsets": [begin, data.len()]}),
        );
    }
    let mut json = serde_json::to_vec(&Value::Object(header)).unwrap();
    while json.len() % 8 != 0 {
        json.push(b' ');
    }
    let mut bytes = (json.len() as u64).to_le_bytes().to_vec();
    bytes.extend_from_slice(&json);
    bytes.extend_from_slice(&data);
    bytes
}

fn matrix(seed: f32) -> Vec<f32> {
    (0..64 * 64)
        .map(|index| ((index as f32) * 0.37 + seed).sin() * 0.5)
        .collect()
}

fn weights(seed: f32) -> Vec<u8> {
    safetensors(&[
        ("w", vec![64, 64], matrix(seed)),
        ("bias", vec![64], vec![0.25; 64]),
    ])
}

fn sha256_hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

fn git_sha1_hex(bytes: &[u8]) -> String {
    let mut hasher = Sha1::new();
    hasher.update(format!("blob {}\0", bytes.len()).as_bytes());
    hasher.update(bytes);
    hasher
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

#[derive(Clone)]
struct Route {
    body: Vec<u8>,
    /// When set, the file is served only with `Authorization: Bearer <TOKEN>`.
    gated: bool,
    /// When set, the served bytes are replaced by this, so the published checksum is wrong.
    corrupt: bool,
}

#[derive(Default)]
struct Hub {
    routes: HashMap<String, Route>,
    json: HashMap<String, Value>,
    hits: HashMap<String, usize>,
    ranges: Vec<(String, String)>,
    auth_seen: Vec<String>,
}

struct Server {
    endpoint: String,
    state: Arc<Mutex<Hub>>,
}

impl Server {
    fn start(hub: Hub) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let endpoint = format!("http://{}", listener.local_addr().unwrap());
        let state = Arc::new(Mutex::new(hub));
        let shared = Arc::clone(&state);
        thread::spawn(move || {
            for stream in listener.incoming() {
                let Ok(stream) = stream else { continue };
                let shared = Arc::clone(&shared);
                thread::spawn(move || handle(stream, &shared));
            }
        });
        Self { endpoint, state }
    }

    fn hits(&self, path: &str) -> usize {
        self.state
            .lock()
            .unwrap()
            .hits
            .get(path)
            .copied()
            .unwrap_or(0)
    }

    fn ranges(&self) -> Vec<(String, String)> {
        self.state.lock().unwrap().ranges.clone()
    }

    fn auth_seen(&self) -> Vec<String> {
        self.state.lock().unwrap().auth_seen.clone()
    }
}

fn handle(mut stream: TcpStream, shared: &Mutex<Hub>) {
    let mut reader = BufReader::new(stream.try_clone().unwrap());
    let mut request_line = String::new();
    if reader.read_line(&mut request_line).is_err() {
        return;
    }
    // The client sends query strings (for example `?blobs=true`); routes are keyed by path.
    let path = request_line
        .split_whitespace()
        .nth(1)
        .unwrap_or("/")
        .split('?')
        .next()
        .unwrap_or("/")
        .to_owned();
    let mut range = None;
    let mut authorization = None;
    loop {
        let mut line = String::new();
        if reader.read_line(&mut line).is_err() || line.trim().is_empty() {
            break;
        }
        if let Some((name, value)) = line.split_once(':') {
            let value = value.trim().to_owned();
            if name.eq_ignore_ascii_case("range") {
                range = Some(value);
            } else if name.eq_ignore_ascii_case("authorization") {
                authorization = Some(value);
            }
        }
    }
    let (status, headers, body) =
        respond(shared, &path, range.as_deref(), authorization.as_deref());
    let mut head = format!("HTTP/1.1 {status} {}\r\n", reason(status));
    head.push_str(&format!(
        "Content-Length: {}\r\nConnection: close\r\n",
        body.len()
    ));
    for (name, value) in headers {
        head.push_str(&format!("{name}: {value}\r\n"));
    }
    head.push_str("\r\n");
    let _ = stream.write_all(head.as_bytes());
    let _ = stream.write_all(&body);
    let _ = stream.flush();
    let _ = stream.shutdown(Shutdown::Both);
}

fn reason(status: u16) -> &'static str {
    match status {
        200 => "OK",
        206 => "Partial Content",
        401 => "Unauthorized",
        404 => "Not Found",
        416 => "Range Not Satisfiable",
        _ => "Status",
    }
}

fn respond(
    shared: &Mutex<Hub>,
    path: &str,
    range: Option<&str>,
    authorization: Option<&str>,
) -> (u16, Vec<(String, String)>, Vec<u8>) {
    let mut hub = shared.lock().unwrap();
    *hub.hits.entry(path.to_owned()).or_insert(0) += 1;
    if let Some(value) = authorization {
        hub.auth_seen.push(value.to_owned());
    }
    if let Some(value) = json_route(&hub, path) {
        return (200, vec![], value.to_string().into_bytes());
    }
    let Some(route) = hub.routes.get(path).cloned() else {
        return (404, vec![], b"not found".to_vec());
    };
    if route.gated && authorization != Some(&format!("Bearer {TOKEN}")) {
        return (401, vec![], b"unauthorized".to_vec());
    }
    let body = if route.corrupt {
        let mut corrupted = route.body.clone();
        corrupted[0] ^= 0xff;
        corrupted
    } else {
        route.body.clone()
    };
    let Some(range) = range else {
        return (200, vec![], body);
    };
    hub.ranges.push((path.to_owned(), range.to_owned()));
    let start: usize = range
        .strip_prefix("bytes=")
        .and_then(|rest| rest.strip_suffix('-'))
        .and_then(|digits| digits.parse().ok())
        .unwrap_or(0);
    if start >= body.len() {
        return (416, vec![], vec![]);
    }
    let headers = vec![(
        "Content-Range".to_owned(),
        format!("bytes {start}-{}/{}", body.len() - 1, body.len()),
    )];
    (206, headers, body[start..].to_vec())
}

fn json_route(hub: &Hub, path: &str) -> Option<Value> {
    hub.json.get(path).cloned()
}

/// The model metadata the Hub returns for a revision, with `files` as siblings.
fn metadata(license: Option<&str>, gated: bool, files: &[(&str, Vec<u8>)]) -> Value {
    let siblings: Vec<Value> = files
        .iter()
        .map(|(name, bytes)| {
            if name.ends_with(".json") {
                json!({"rfilename": name, "size": bytes.len(), "blobId": git_sha1_hex(bytes)})
            } else {
                json!({
                    "rfilename": name,
                    "size": bytes.len(),
                    "lfs": {"sha256": sha256_hex(bytes), "size": bytes.len()}
                })
            }
        })
        .collect();
    let mut value = json!({"sha": COMMIT, "siblings": siblings, "gated": if gated { json!("auto") } else { json!(false) }});
    if let Some(license) = license {
        value["cardData"] = json!({"license": license});
    }
    value
}

/// A fake Hub with three repositories: a single-file model, a sharded model
/// and a gated one. `acme/bad` publishes a checksum that its file does not match.
fn hub_fixture() -> (Hub, Vec<u8>) {
    let single = weights(1.0);
    // Each shard holds exactly the tensors the index assigns to it.
    let shard_one = safetensors(&[("w", vec![64, 64], matrix(2.0))]);
    let shard_two = safetensors(&[("bias", vec![64], vec![0.25; 64])]);
    let index = serde_json::to_vec(&json!({
        "metadata": {},
        "weight_map": {
            "w": "model-00001-of-00002.safetensors",
            "bias": "model-00002-of-00002.safetensors"
        }
    }))
    .unwrap();
    let gated = weights(4.0);
    let mut hub = Hub::default();
    let mut add_repo =
        |owner_name: &str, files: &[(&str, Vec<u8>)], license: Option<&str>, gated_repo: bool| {
            hub.json.insert(
                format!("/api/models/{owner_name}/revision/main"),
                metadata(license, gated_repo, files),
            );
            for (name, bytes) in files {
                hub.routes.insert(
                    format!("/{owner_name}/resolve/{COMMIT}/{name}"),
                    Route {
                        body: bytes.clone(),
                        gated: gated_repo,
                        corrupt: false,
                    },
                );
            }
        };
    add_repo(
        "acme/tiny",
        &[
            ("config.json", b"{}".to_vec()),
            ("model.safetensors", single.clone()),
        ],
        Some("apache-2.0"),
        false,
    );
    add_repo(
        "acme/sharded",
        &[
            ("model.safetensors.index.json", index),
            ("model-00001-of-00002.safetensors", shard_one),
            ("model-00002-of-00002.safetensors", shard_two),
        ],
        Some("mit"),
        false,
    );
    add_repo("acme/gated", &[("model.safetensors", gated)], None, true);
    add_repo(
        "acme/bad",
        &[("model.safetensors", single.clone())],
        Some("mit"),
        false,
    );
    add_repo(
        "acme/none",
        &[("config.json", b"{}".to_vec())],
        Some("mit"),
        false,
    );
    hub.routes
        .get_mut(&format!("/acme/bad/resolve/{COMMIT}/model.safetensors"))
        .unwrap()
        .corrupt = true;
    (hub, single)
}

fn modelq(endpoint: &str, cache: &Path) -> Command {
    let mut command = Command::new(env!("CARGO_BIN_EXE_modelq"));
    command
        .env("MODELQ_HUB_ENDPOINT", endpoint)
        .env("MODELQ_CACHE", cache)
        .env_remove("HF_TOKEN");
    command
}

fn stdout(output: &Output) -> String {
    String::from_utf8_lossy(&output.stdout).into_owned()
}

fn stderr(output: &Output) -> String {
    String::from_utf8_lossy(&output.stderr).into_owned()
}

fn cached_commit_dir(cache: &Path, repo: &str) -> PathBuf {
    cache.join("hub").join(repo).join(COMMIT)
}

#[test]
fn a_single_file_model_is_fetched_verified_and_quantized() {
    let (hub, _) = hub_fixture();
    let server = Server::start(hub);
    let dir = TestDir::new("single");
    let output = dir.join("out.safetensors");

    let result = modelq(&server.endpoint, &dir.0)
        .args(["quantize", "hf:acme/tiny", "--format", "int8", "--output"])
        .arg(&output)
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}{}",
        stdout(&result),
        stderr(&result)
    );
    let text = stdout(&result);
    assert!(text.contains("license apache-2.0"), "{text}");
    assert!(
        text.contains("stored  model.safetensors (verified)"),
        "{text}"
    );
    assert!(output.is_file(), "the quantized output was written");
    assert!(
        cached_commit_dir(&dir.0, "acme/tiny")
            .join("model.safetensors")
            .is_file()
    );
    assert_eq!(
        server.hits(&format!("/acme/tiny/resolve/{COMMIT}/model.safetensors")),
        1
    );
}

#[test]
fn a_second_run_reuses_the_verified_cache() {
    let (hub, _) = hub_fixture();
    let server = Server::start(hub);
    let dir = TestDir::new("reuse");
    let run = |output: &Path| {
        modelq(&server.endpoint, &dir.0)
            .args(["quantize", "hf:acme/tiny", "--format", "int8", "--output"])
            .arg(output)
            .output()
            .unwrap()
    };
    assert!(run(&dir.join("first.safetensors")).status.success());
    let second = run(&dir.join("second.safetensors"));
    assert!(second.status.success(), "{}", stderr(&second));
    assert!(
        stdout(&second).contains("cached  model.safetensors (verified)"),
        "{}",
        stdout(&second)
    );
    assert_eq!(
        server.hits(&format!("/acme/tiny/resolve/{COMMIT}/model.safetensors")),
        1,
        "the weights were downloaded once"
    );
}

#[test]
fn a_sharded_model_fetches_the_index_and_every_shard() {
    let (hub, _) = hub_fixture();
    let server = Server::start(hub);
    let dir = TestDir::new("sharded");
    let output = dir.join("out.safetensors");
    let result = modelq(&server.endpoint, &dir.0)
        .args([
            "quantize",
            "hf:acme/sharded",
            "--format",
            "int8",
            "--output",
        ])
        .arg(&output)
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}{}",
        stdout(&result),
        stderr(&result)
    );
    let commit_dir = cached_commit_dir(&dir.0, "acme/sharded");
    for name in [
        "model.safetensors.index.json",
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
    ] {
        assert!(commit_dir.join(name).is_file(), "{name} is cached");
    }
    assert!(
        stdout(&result).contains("license mit"),
        "{}",
        stdout(&result)
    );
}

#[test]
fn a_checksum_mismatch_is_refused_and_nothing_is_kept() {
    let (hub, _) = hub_fixture();
    let server = Server::start(hub);
    let dir = TestDir::new("mismatch");
    let output = dir.join("out.safetensors");
    let result = modelq(&server.endpoint, &dir.0)
        .args(["quantize", "hf:acme/bad", "--format", "int8", "--output"])
        .arg(&output)
        .output()
        .unwrap();
    assert!(!result.status.success());
    assert!(
        stderr(&result).contains("checksum mismatch"),
        "{}",
        stderr(&result)
    );
    assert!(!output.exists());
    let commit_dir = cached_commit_dir(&dir.0, "acme/bad");
    assert!(
        !commit_dir.join("model.safetensors").exists(),
        "the unverified file was not kept"
    );
    assert!(
        !commit_dir.join("model.safetensors.part").exists(),
        "the partial file was removed"
    );
}

#[test]
fn a_gated_repository_explains_what_to_do_and_accepts_a_token() {
    let (hub, _) = hub_fixture();
    let server = Server::start(hub);
    let dir = TestDir::new("gated");
    let output = dir.join("out.safetensors");

    let refused = modelq(&server.endpoint, &dir.0)
        .args(["quantize", "hf:acme/gated", "--format", "int8", "--output"])
        .arg(&output)
        .output()
        .unwrap();
    assert!(!refused.status.success());
    assert!(
        stderr(&refused).contains("needs access"),
        "{}",
        stderr(&refused)
    );
    assert!(
        stderr(&refused).contains("HF_TOKEN"),
        "{}",
        stderr(&refused)
    );

    let accepted = modelq(&server.endpoint, &dir.0)
        .env("HF_TOKEN", TOKEN)
        .args(["quantize", "hf:acme/gated", "--format", "int8", "--output"])
        .arg(&output)
        .output()
        .unwrap();
    assert!(
        accepted.status.success(),
        "{}{}",
        stdout(&accepted),
        stderr(&accepted)
    );
    assert!(
        server
            .auth_seen()
            .iter()
            .any(|value| value == &format!("Bearer {TOKEN}"))
    );
    assert!(
        !stdout(&accepted).contains(TOKEN),
        "the token is never printed"
    );
    assert!(
        !stderr(&accepted).contains(TOKEN),
        "the token is never printed"
    );
}

#[test]
fn an_interrupted_download_resumes_with_a_range_request() {
    let (hub, single) = hub_fixture();
    let server = Server::start(hub);
    let dir = TestDir::new("resume");
    let commit_dir = cached_commit_dir(&dir.0, "acme/tiny");
    fs::create_dir_all(&commit_dir).unwrap();
    let half = single.len() / 2;
    fs::write(commit_dir.join("model.safetensors.part"), &single[..half]).unwrap();

    let output = dir.join("out.safetensors");
    let result = modelq(&server.endpoint, &dir.0)
        .args(["quantize", "hf:acme/tiny", "--format", "int8", "--output"])
        .arg(&output)
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}{}",
        stdout(&result),
        stderr(&result)
    );
    let expected_range = format!("bytes={half}-");
    assert!(
        server
            .ranges()
            .iter()
            .any(|(_, range)| range == &expected_range),
        "the server saw a resume request for {expected_range}: {:?}",
        server.ranges()
    );
    assert_eq!(
        fs::read(commit_dir.join("model.safetensors")).unwrap(),
        single
    );
}

#[test]
fn a_repository_without_weights_is_reported() {
    let (hub, _) = hub_fixture();
    let server = Server::start(hub);
    let dir = TestDir::new("none");
    let result = modelq(&server.endpoint, &dir.0)
        .args(["quantize", "hf:acme/none", "--format", "int8", "--output"])
        .arg(dir.join("out.safetensors"))
        .output()
        .unwrap();
    assert!(!result.status.success());
    assert!(
        stderr(&result).contains("has no SafeTensors weights"),
        "{}",
        stderr(&result)
    );
}

#[test]
fn an_unknown_repository_is_reported() {
    let (hub, _) = hub_fixture();
    let server = Server::start(hub);
    let dir = TestDir::new("missing");
    let result = modelq(&server.endpoint, &dir.0)
        .args([
            "quantize",
            "hf:acme/missing",
            "--format",
            "int8",
            "--output",
        ])
        .arg(dir.join("out.safetensors"))
        .output()
        .unwrap();
    assert!(!result.status.success());
    assert!(
        stderr(&result).contains("was not found"),
        "{}",
        stderr(&result)
    );
}

#[test]
fn a_malformed_model_id_is_refused_before_any_request() {
    let (hub, _) = hub_fixture();
    let server = Server::start(hub);
    let dir = TestDir::new("bad-id");
    let result = modelq(&server.endpoint, &dir.0)
        .args(["quantize", "hf:../escape", "--format", "int8", "--output"])
        .arg(dir.join("out.safetensors"))
        .output()
        .unwrap();
    assert!(!result.status.success());
    assert!(
        stderr(&result).contains("not a valid model id"),
        "{}",
        stderr(&result)
    );
    assert!(
        server.state.lock().unwrap().hits.is_empty(),
        "no request was sent"
    );
}

#[test]
fn a_local_checkpoint_never_contacts_the_hub() {
    let dir = TestDir::new("local");
    let input = dir.join("local.safetensors");
    fs::write(&input, weights(9.0)).unwrap();
    let output = dir.join("out.safetensors");
    // Nothing listens on port 1; any network use would fail the run.
    let result = modelq("http://127.0.0.1:1", &dir.0)
        .args(["quantize"])
        .arg(&input)
        .args(["--format", "int8", "--output"])
        .arg(&output)
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}{}",
        stdout(&result),
        stderr(&result)
    );
    assert!(output.is_file());
}

#[test]
fn a_missing_local_path_is_not_treated_as_a_hub_id() {
    let dir = TestDir::new("typo");
    let result = modelq("http://127.0.0.1:1", &dir.0)
        .args(["quantize", "models/qwen", "--format", "int8", "--output"])
        .arg(dir.join("out.safetensors"))
        .output()
        .unwrap();
    assert!(!result.status.success());
    assert!(!stderr(&result).contains("hf:"), "{}", stderr(&result));
}
