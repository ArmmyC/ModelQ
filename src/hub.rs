//! Hugging Face Hub input (ADR 0029, milestone M2).
//!
//! `modelq quantize hf:<owner>/<name>` fetches the SafeTensors weights of a
//! model into a local cache, verifies every file against the checksum the Hub
//! publishes (SHA-256 for LFS files, the git blob SHA-1 for small text files),
//! and returns the directory, which the quantizer reads like a local
//! checkpoint. Nothing is downloaded unless the user names the model with the
//! `hf:` prefix, and the model's license is printed before any weight is
//! fetched.

use std::{
    fs::{self, File, OpenOptions},
    io::{self, BufReader, Read, Write},
    path::{Path, PathBuf},
};

use serde_json::Value;
use sha1::Sha1;
use sha2::{Digest, Sha256};

/// Hub base URL, unless `MODELQ_HUB_ENDPOINT` overrides it.
pub const DEFAULT_ENDPOINT: &str = "https://huggingface.co";
/// Environment variable that overrides the Hub base URL (mirrors and tests).
pub const ENDPOINT_ENV: &str = "MODELQ_HUB_ENDPOINT";
/// Environment variable with a personal access token for gated or private repositories.
pub const TOKEN_ENV: &str = "HF_TOKEN";
/// Environment variable that overrides the cache directory.
pub const CACHE_ENV: &str = "MODELQ_CACHE";
/// The prefix that marks a Hub model in the `model` argument.
pub const PREFIX: &str = "hf:";

const SINGLE_FILE: &str = "model.safetensors";
const INDEX_FILE: &str = "model.safetensors.index.json";
const ATTEMPTS: u32 = 3;
const USER_AGENT: &str = concat!("modelq/", env!("CARGO_PKG_VERSION"));

/// A model repository, `owner/name`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RepoId {
    owner: String,
    name: String,
}

impl RepoId {
    /// Parses `owner/name`. Both parts must be plain names, so the id can be
    /// used as a directory name without any path tricks.
    pub fn parse(text: &str) -> Result<Self, String> {
        let mut parts = text.split('/');
        let (Some(owner), Some(name), None) = (parts.next(), parts.next(), parts.next()) else {
            return Err(format!("{text:?} is not a model id; expected owner/name"));
        };
        if !plain_name(owner) || !plain_name(name) {
            return Err(format!("{text:?} is not a valid model id"));
        }
        Ok(Self {
            owner: owner.to_owned(),
            name: name.to_owned(),
        })
    }

    /// `owner/name`.
    pub fn path(&self) -> String {
        format!("{}/{}", self.owner, self.name)
    }
}

fn plain_name(part: &str) -> bool {
    !part.is_empty()
        && part != "."
        && part != ".."
        && part
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.'))
}

/// Everything a fetch needs. Built from the command line and environment.
#[derive(Debug, Clone)]
pub struct Request {
    pub repo: RepoId,
    /// A branch, tag or commit; `main` by default.
    pub revision: String,
    /// Where files are cached: `<cache_dir>/hub/<owner>/<name>/<commit>/`.
    pub cache_dir: PathBuf,
    /// Hub base URL.
    pub endpoint: String,
    /// Sent as a bearer token when present. Never printed.
    pub token: Option<String>,
}

/// The Hub's record of one file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RepoFile {
    pub name: String,
    pub size: Option<u64>,
    /// SHA-256 of the content, for LFS files.
    pub sha256: Option<String>,
    /// The git blob SHA-1 (`sha1("blob <len>\0" + content)`), for other files.
    pub git_sha1: Option<String>,
}

/// The Hub's record of one revision of a model.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RepoInfo {
    pub commit: String,
    pub license: Option<String>,
    pub gated: bool,
    pub files: Vec<RepoFile>,
}

/// The base URL from `MODELQ_HUB_ENDPOINT`, or the default.
pub fn endpoint_from_env() -> String {
    std::env::var(ENDPOINT_ENV)
        .ok()
        .filter(|value| !value.trim().is_empty())
        .unwrap_or_else(|| DEFAULT_ENDPOINT.to_owned())
        .trim_end_matches('/')
        .to_owned()
}

/// The token from `HF_TOKEN`, if set and not empty.
pub fn token_from_env() -> Option<String> {
    std::env::var(TOKEN_ENV)
        .ok()
        .filter(|value| !value.trim().is_empty())
}

/// `MODELQ_CACHE`, then the platform cache directory for ModelQ.
pub fn default_cache_dir() -> Result<PathBuf, String> {
    if let Some(dir) = std::env::var_os(CACHE_ENV).filter(|value| !value.is_empty()) {
        return Ok(PathBuf::from(dir));
    }
    if cfg!(windows) {
        if let Some(base) = std::env::var_os("LOCALAPPDATA").filter(|value| !value.is_empty()) {
            return Ok(PathBuf::from(base).join("modelq").join("cache"));
        }
    } else {
        if let Some(base) = std::env::var_os("XDG_CACHE_HOME").filter(|value| !value.is_empty()) {
            return Ok(PathBuf::from(base).join("modelq"));
        }
        if let Some(home) = std::env::var_os("HOME").filter(|value| !value.is_empty()) {
            return Ok(PathBuf::from(home).join(".cache").join("modelq"));
        }
    }
    Err(format!(
        "no cache directory: pass --cache-dir or set {CACHE_ENV}"
    ))
}

/// Downloads (or reuses) the weights of one model revision and returns the
/// directory that holds them. Prints the license and every step.
pub fn fetch(request: &Request) -> Result<PathBuf, String> {
    validate_revision(&request.revision)?;
    let agent = agent();
    let info = repo_info(&agent, request)?;
    announce(request, &info);

    let mut files = vec![select_weights(request, &info)?];
    if files[0].name == INDEX_FILE {
        let index = files.remove(0);
        let shards = shard_names(&read_index(&agent, request, &info, &index)?)?;
        files.push(index);
        for shard in shards {
            let file = info
                .files
                .iter()
                .find(|candidate| candidate.name == shard)
                .ok_or_else(|| {
                    format!("the index names {shard}, which the repository does not have")
                })?;
            files.push(file.clone());
        }
    }

    let directory = request
        .cache_dir
        .join("hub")
        .join(&request.repo.owner)
        .join(&request.repo.name)
        .join(&info.commit);
    for file in &files {
        ensure_file(&agent, request, &info.commit, file, &directory)?;
    }
    println!("ready   {}", directory.display());
    Ok(directory)
}

fn validate_revision(revision: &str) -> Result<(), String> {
    let ok = !revision.is_empty()
        && !revision.contains("..")
        && !revision.starts_with('/')
        && revision
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.' | '/'));
    if ok {
        Ok(())
    } else {
        Err(format!("{revision:?} is not a valid revision"))
    }
}

fn agent() -> ureq::Agent {
    // Only the native TLS provider is compiled in (see ADR 0029), so it must be
    // selected explicitly; the default provider is Rustls.
    let tls = ureq::tls::TlsConfig::builder()
        .provider(ureq::tls::TlsProvider::NativeTls)
        .build();
    let config = ureq::Agent::config_builder()
        .http_status_as_error(false)
        .tls_config(tls)
        .build();
    ureq::Agent::new_with_config(config)
}

fn get(
    agent: &ureq::Agent,
    request: &Request,
    url: &str,
) -> Result<ureq::http::Response<ureq::Body>, String> {
    let mut builder = agent.get(url).header("User-Agent", USER_AGENT);
    if let Some(token) = &request.token {
        builder = builder.header("Authorization", format!("Bearer {token}"));
    }
    builder
        .call()
        .map_err(|error| format!("could not reach {}: {error}", request.endpoint))
}

fn repo_info(agent: &ureq::Agent, request: &Request) -> Result<RepoInfo, String> {
    let url = format!(
        "{}/api/models/{}/revision/{}?blobs=true",
        request.endpoint,
        request.repo.path(),
        encode_path(&request.revision)
    );
    let mut response = get(agent, request, &url)?;
    match response.status().as_u16() {
        200 => {}
        401 | 403 => return Err(access_message(request)),
        404 => {
            return Err(format!(
                "model {} at revision {} was not found on {} (check the id and the revision)",
                request.repo.path(),
                request.revision,
                request.endpoint
            ));
        }
        status => return Err(format!("HTTP {status} from {url}")),
    }
    let mut text = String::new();
    response
        .body_mut()
        .as_reader()
        .read_to_string(&mut text)
        .map_err(|error| format!("could not read the Hub response: {error}"))?;
    parse_repo_info(&text)
}

/// Parses the Hub's model metadata. Public for tests.
pub fn parse_repo_info(text: &str) -> Result<RepoInfo, String> {
    let value: Value = serde_json::from_str(text)
        .map_err(|error| format!("the Hub response is not JSON: {error}"))?;
    let commit = value
        .get("sha")
        .and_then(Value::as_str)
        .filter(|sha| sha.len() == 40 && sha.chars().all(|c| c.is_ascii_hexdigit()))
        .ok_or_else(|| "the Hub response has no commit id".to_owned())?
        .to_owned();
    let license = value
        .pointer("/cardData/license")
        .and_then(Value::as_str)
        .map(str::to_owned)
        .or_else(|| {
            value
                .get("tags")
                .and_then(Value::as_array)?
                .iter()
                .filter_map(Value::as_str)
                .find_map(|tag| tag.strip_prefix("license:").map(str::to_owned))
        });
    let gated = match value.get("gated") {
        None | Some(Value::Null) | Some(Value::Bool(false)) => false,
        Some(_) => true,
    };
    let mut files = Vec::new();
    for sibling in value
        .get("siblings")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
    {
        let Some(name) = sibling.get("rfilename").and_then(Value::as_str) else {
            continue;
        };
        if !safe_relative_name(name) {
            continue;
        }
        files.push(RepoFile {
            name: name.to_owned(),
            size: sibling.get("size").and_then(Value::as_u64),
            sha256: sibling
                .pointer("/lfs/sha256")
                .and_then(Value::as_str)
                .map(str::to_owned),
            git_sha1: sibling
                .get("blobId")
                .and_then(Value::as_str)
                .map(str::to_owned),
        });
    }
    Ok(RepoInfo {
        commit,
        license,
        gated,
        files,
    })
}

fn announce(request: &Request, info: &RepoInfo) {
    println!("model   {} @ {}", request.repo.path(), &info.commit[..12]);
    match &info.license {
        Some(license) => println!(
            "license {license} (check the model page before you use the result: {}/{})",
            request.endpoint,
            request.repo.path()
        ),
        None => println!(
            "license not stated in the model card; check the model page first: {}/{}",
            request.endpoint,
            request.repo.path()
        ),
    }
    if info.gated {
        println!(
            "gated   this repository asks you to accept its terms; set {TOKEN_ENV} if it is required"
        );
    }
}

fn access_message(request: &Request) -> String {
    format!(
        "{} needs access: accept its terms on {}/{} and set {TOKEN_ENV} to a token that has access",
        request.repo.path(),
        request.endpoint,
        request.repo.path()
    )
}

fn select_weights(request: &Request, info: &RepoInfo) -> Result<RepoFile, String> {
    if let Some(file) = info.files.iter().find(|file| file.name == SINGLE_FILE) {
        return Ok(file.clone());
    }
    if let Some(file) = info.files.iter().find(|file| file.name == INDEX_FILE) {
        return Ok(file.clone());
    }
    Err(format!(
        "{} has no SafeTensors weights ({SINGLE_FILE} or {INDEX_FILE})",
        request.repo.path()
    ))
}

fn read_index(
    agent: &ureq::Agent,
    request: &Request,
    info: &RepoInfo,
    index: &RepoFile,
) -> Result<String, String> {
    let directory = request
        .cache_dir
        .join("hub")
        .join(&request.repo.owner)
        .join(&request.repo.name)
        .join(&info.commit);
    ensure_file(agent, request, &info.commit, index, &directory)?;
    fs::read_to_string(directory.join(&index.name))
        .map_err(|error| format!("could not read {}: {error}", index.name))
}

/// The distinct shard file names in an index's `weight_map`, sorted.
pub fn shard_names(index_text: &str) -> Result<Vec<String>, String> {
    let value: Value = serde_json::from_str(index_text)
        .map_err(|error| format!("the weights index is not JSON: {error}"))?;
    let map = value
        .get("weight_map")
        .and_then(Value::as_object)
        .ok_or_else(|| "the weights index has no weight_map".to_owned())?;
    let mut shards: Vec<String> = map
        .values()
        .filter_map(Value::as_str)
        .map(str::to_owned)
        .collect();
    shards.sort();
    shards.dedup();
    if shards.is_empty() {
        return Err("the weights index names no shards".to_owned());
    }
    for shard in &shards {
        if !safe_relative_name(shard) {
            return Err(format!("the weights index names an unsafe file {shard:?}"));
        }
    }
    Ok(shards)
}

/// A repository file name that can be joined under a directory: relative,
/// no empty, `.` or `..` parts, and no drive or backslash tricks.
pub fn safe_relative_name(name: &str) -> bool {
    !name.is_empty()
        && !name.contains('\\')
        && !name.contains(':')
        && !name.starts_with('/')
        && name
            .split('/')
            .all(|part| !part.is_empty() && part != "." && part != "..")
}

/// Makes sure `file` is in `directory` and matches its published checksum.
/// A complete, verified file is reused; anything else is downloaded again.
fn ensure_file(
    agent: &ureq::Agent,
    request: &Request,
    commit: &str,
    file: &RepoFile,
    directory: &Path,
) -> Result<(), String> {
    let target = directory.join(&file.name);
    if target.is_file() && verify(&target, file)? {
        println!("cached  {} (verified)", file.name);
        return Ok(());
    }
    if let Some(parent) = target.parent() {
        fs::create_dir_all(parent)
            .map_err(|error| format!("could not create {}: {error}", parent.display()))?;
    }
    let part = partial_path(&target);
    let url = format!(
        "{}/{}/resolve/{}/{}",
        request.endpoint,
        request.repo.path(),
        commit,
        encode_path(&file.name)
    );
    println!("fetch   {} ({})", file.name, size_text(file.size));

    let mut attempt = 0;
    loop {
        attempt += 1;
        match download_once(agent, request, &url, file, &part) {
            Ok(()) => break,
            Err(Failure::Retry(message)) if attempt < ATTEMPTS => {
                eprintln!("retry   {}: {message}", file.name);
            }
            Err(Failure::Retry(message) | Failure::Fatal(message)) => return Err(message),
        }
    }

    if !verify(&part, file)? {
        let _ = fs::remove_file(&part);
        return Err(format!(
            "checksum mismatch for {}; the download was discarded, try again",
            file.name
        ));
    }
    if target.exists() {
        fs::remove_file(&target)
            .map_err(|error| format!("could not replace {}: {error}", target.display()))?;
    }
    fs::rename(&part, &target)
        .map_err(|error| format!("could not store {}: {error}", target.display()))?;
    println!("stored  {} (verified)", file.name);
    Ok(())
}

enum Failure {
    /// A network or server problem that another attempt may fix.
    Retry(String),
    /// A problem another attempt cannot fix, such as missing access.
    Fatal(String),
}

fn download_once(
    agent: &ureq::Agent,
    request: &Request,
    url: &str,
    file: &RepoFile,
    part: &Path,
) -> Result<(), Failure> {
    let mut existing = fs::metadata(part).map(|meta| meta.len()).unwrap_or(0);
    if let Some(size) = file.size {
        if existing == size {
            return Ok(());
        }
        if existing > size {
            let _ = fs::remove_file(part);
            existing = 0;
        }
    }

    let mut builder = agent.get(url).header("User-Agent", USER_AGENT);
    if let Some(token) = &request.token {
        builder = builder.header("Authorization", format!("Bearer {token}"));
    }
    if existing > 0 {
        builder = builder.header("Range", format!("bytes={existing}-"));
    }
    let mut response = builder
        .call()
        .map_err(|error| Failure::Retry(format!("network error: {error}")))?;
    let status = response.status().as_u16();
    match status {
        200 | 206 => {}
        416 if file.size == Some(existing) => return Ok(()),
        416 => {
            let _ = fs::remove_file(part);
            return Err(Failure::Retry(
                "the server rejected the resume request".to_owned(),
            ));
        }
        401 | 403 => return Err(Failure::Fatal(access_message(request))),
        404 => {
            return Err(Failure::Fatal(format!(
                "{} was not found at revision {}",
                file.name, request.revision
            )));
        }
        500..=599 => return Err(Failure::Retry(format!("HTTP {status} from the Hub"))),
        other => return Err(Failure::Fatal(format!("HTTP {other} for {}", file.name))),
    }

    let resumed = status == 206 && existing > 0;
    let mut options = OpenOptions::new();
    options.create(true).write(true);
    if resumed {
        options.append(true);
    } else {
        options.truncate(true);
    }
    let mut output = options
        .open(part)
        .map_err(|error| Failure::Fatal(format!("could not write {}: {error}", part.display())))?;
    let mut body = response.body_mut().as_reader();
    io::copy(&mut body, &mut output)
        .map_err(|error| Failure::Retry(format!("the download stopped: {error}")))?;
    output
        .flush()
        .map_err(|error| Failure::Fatal(format!("could not write {}: {error}", part.display())))?;
    Ok(())
}

/// Checks a local file against the Hub's checksum. A file without any
/// published checksum is refused rather than trusted.
fn verify(path: &Path, file: &RepoFile) -> Result<bool, String> {
    let metadata = fs::metadata(path)
        .map_err(|error| format!("could not read {}: {error}", path.display()))?;
    if let Some(size) = file.size {
        if metadata.len() != size {
            return Ok(false);
        }
    }
    if let Some(expected) = &file.sha256 {
        return Ok(sha256_file(path)?.eq_ignore_ascii_case(expected));
    }
    if let Some(expected) = &file.git_sha1 {
        return Ok(git_blob_sha1(path, metadata.len())?.eq_ignore_ascii_case(expected));
    }
    Err(format!(
        "the Hub publishes no checksum for {}; refusing to use it",
        file.name
    ))
}

/// SHA-256 of a file, as lowercase hex.
pub fn sha256_file(path: &Path) -> Result<String, String> {
    let mut reader = BufReader::new(
        File::open(path).map_err(|error| format!("could not open {}: {error}", path.display()))?,
    );
    let mut hasher = Sha256::new();
    let mut buffer = vec![0_u8; 1 << 20];
    loop {
        let read = reader
            .read(&mut buffer)
            .map_err(|error| format!("could not read {}: {error}", path.display()))?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(hex(&hasher.finalize()))
}

/// The git blob id of a file: `sha1("blob <len>\0" + content)`, as lowercase hex.
fn git_blob_sha1(path: &Path, length: u64) -> Result<String, String> {
    let mut reader = BufReader::new(
        File::open(path).map_err(|error| format!("could not open {}: {error}", path.display()))?,
    );
    let mut hasher = Sha1::new();
    hasher.update(format!("blob {length}\0").as_bytes());
    let mut buffer = vec![0_u8; 1 << 20];
    loop {
        let read = reader
            .read(&mut buffer)
            .map_err(|error| format!("could not read {}: {error}", path.display()))?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(hex(&hasher.finalize()))
}

fn hex(bytes: &[u8]) -> String {
    use std::fmt::Write as _;
    bytes
        .iter()
        .fold(String::with_capacity(bytes.len() * 2), |mut text, byte| {
            let _ = write!(text, "{byte:02x}");
            text
        })
}

fn partial_path(target: &Path) -> PathBuf {
    let mut name = target
        .file_name()
        .map(|name| name.to_os_string())
        .unwrap_or_default();
    name.push(".part");
    target.with_file_name(name)
}

/// Percent-encodes everything except unreserved characters and `/`.
fn encode_path(text: &str) -> String {
    let mut encoded = String::with_capacity(text.len());
    for byte in text.bytes() {
        if byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.' | b'~' | b'/') {
            encoded.push(char::from(byte));
        } else {
            encoded.push_str(&format!("%{byte:02X}"));
        }
    }
    encoded
}

fn size_text(size: Option<u64>) -> String {
    match size {
        Some(bytes) if bytes >= 1 << 30 => {
            format!("{:.2} GB", bytes as f64 / f64::from(1_u32 << 30))
        }
        Some(bytes) if bytes >= 1 << 20 => {
            format!("{:.1} MB", bytes as f64 / f64::from(1_u32 << 20))
        }
        Some(bytes) => format!("{bytes} bytes"),
        None => "unknown size".to_owned(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn repo_ids_must_be_two_plain_names() {
        assert_eq!(
            RepoId::parse("Qwen/Qwen2.5-0.5B").unwrap().path(),
            "Qwen/Qwen2.5-0.5B"
        );
        for bad in [
            "", "Qwen", "a/b/c", "../etc", "a/..", "a/b c", "C:/x", "a\\b", "/a/b",
        ] {
            assert!(RepoId::parse(bad).is_err(), "{bad:?} should be rejected");
        }
    }

    #[test]
    fn unsafe_file_names_are_refused() {
        assert!(safe_relative_name("model.safetensors"));
        assert!(safe_relative_name("shards/model-00001.safetensors"));
        for bad in [
            "",
            "/etc/passwd",
            "../x",
            "a/../b",
            "C:x",
            "a\\b",
            "a//b",
            "./a",
        ] {
            assert!(!safe_relative_name(bad), "{bad:?} should be refused");
        }
    }

    #[test]
    fn revisions_cannot_escape_the_url() {
        assert!(validate_revision("main").is_ok());
        assert!(validate_revision("refs/pr/3").is_ok());
        for bad in ["", "../x", "a?b", "a b", "/main"] {
            assert!(validate_revision(bad).is_err(), "{bad:?}");
        }
    }

    #[test]
    fn checksums_match_known_values() {
        let dir = std::env::temp_dir().join(format!("modelq-hub-unit-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join("abc.txt");
        fs::write(&path, b"abc").unwrap();
        assert_eq!(
            sha256_file(&path).unwrap(),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        // git's id of the empty blob is a well-known constant.
        let empty = dir.join("empty.txt");
        fs::write(&empty, b"").unwrap();
        assert_eq!(
            git_blob_sha1(&empty, 0).unwrap(),
            "e69de29bb2d1d6434b8b29ae775ad8c2e48c5391"
        );
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn metadata_yields_commit_license_and_files() {
        let text = r#"{
            "sha": "060db6499f32faf8b98477b0a26969ef7d8b9987",
            "cardData": {"license": "apache-2.0"},
            "gated": false,
            "siblings": [
                {"rfilename": "config.json", "size": 659, "blobId": "aa"},
                {"rfilename": "model.safetensors", "size": 988097824,
                 "lfs": {"sha256": "88c142557820ccad55bb59756bfcfcf891de9cc6202816bd346445188a0ed342"}},
                {"rfilename": "../escape.safetensors", "size": 1}
            ]
        }"#;
        let info = parse_repo_info(text).unwrap();
        assert_eq!(info.commit.len(), 40);
        assert_eq!(info.license.as_deref(), Some("apache-2.0"));
        assert!(!info.gated);
        assert_eq!(info.files.len(), 2, "the unsafe name is dropped");
        assert_eq!(info.files[1].size, Some(988_097_824));
        assert!(info.files[1].sha256.is_some());
    }

    #[test]
    fn a_license_tag_is_used_when_the_card_has_none() {
        let text = r#"{"sha": "060db6499f32faf8b98477b0a26969ef7d8b9987", "tags": ["text-generation", "license:mit"]}"#;
        assert_eq!(
            parse_repo_info(text).unwrap().license.as_deref(),
            Some("mit")
        );
    }

    #[test]
    fn a_commit_that_is_not_a_hash_is_refused() {
        assert!(parse_repo_info(r#"{"sha": "../../x"}"#).is_err());
    }

    #[test]
    fn shard_names_are_distinct_sorted_and_safe() {
        let index = r#"{"weight_map": {"a": "model-2.safetensors", "b": "model-1.safetensors", "c": "model-1.safetensors"}}"#;
        assert_eq!(
            shard_names(index).unwrap(),
            ["model-1.safetensors", "model-2.safetensors"]
        );
        let unsafe_index = r#"{"weight_map": {"a": "../model.safetensors"}}"#;
        assert!(shard_names(unsafe_index).is_err());
        assert!(shard_names(r#"{"weight_map": {}}"#).is_err());
    }

    #[test]
    fn paths_are_percent_encoded_per_byte() {
        assert_eq!(encode_path("refs/pr 3"), "refs/pr%203");
        assert_eq!(encode_path("model.safetensors"), "model.safetensors");
    }
}
