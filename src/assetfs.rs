//! In-memory asset filesystem for shells that cannot see disk files
//! (web: one assets zip dropped before boot). Entries are keyed by
//! their `images/…` / `fonts/…` / `sounds/…` tail, the same path rule
//! the APK `assets/` table uses, so `read_asset_bytes` resolves
//! identically on every platform.

use std::collections::HashMap;
use std::io::Read;
use std::path::Path;
use std::sync::OnceLock;

static ZIPPED: OnceLock<HashMap<String, Vec<u8>>> = OnceLock::new();

/// Tail of an asset path from the first `images`/`fonts` segment, so
/// `…/assets/images/anims.ron` and `images/anims.ron` both address
/// the same entry.
pub fn tail(path: &Path) -> String {
    let parts: Vec<String> = path
        .components()
        .map(|c| c.as_os_str().to_string_lossy().into_owned())
        .collect();
    let start = parts
        .iter()
        .position(|p| p == "images" || p == "fonts" || p == "sounds")
        .unwrap_or(0);
    parts[start..].join("/")
}

/// Unpack an assets zip into the process-wide store. Fails when the
/// zip carries no animation catalog (not an NT assets zip) or when
/// assets were already installed.
pub fn install(bytes: &[u8]) -> anyhow::Result<()> {
    let mut archive = zip::ZipArchive::new(std::io::Cursor::new(bytes))?;
    let mut entries: HashMap<String, Vec<u8>> = HashMap::new();
    for i in 0..archive.len() {
        let mut file = archive.by_index(i)?;
        if file.is_dir() {
            continue;
        }
        let name = file.name().replace('\\', "/");
        if !name.contains("images/") && !name.contains("fonts/") && !name.contains("sounds/") {
            continue;
        }
        let key = tail(Path::new(&name));
        let mut buf = Vec::with_capacity(file.size() as usize);
        file.read_to_end(&mut buf)?;
        entries.insert(key, buf);
    }
    if !entries.contains_key("images/anims.ron") && !entries.contains_key("images/anims.json") {
        anyhow::bail!("not an NT assets zip (images/anims.ron missing)");
    }
    ZIPPED
        .set(entries)
        .map_err(|_| anyhow::anyhow!("assets already installed"))?;
    Ok(())
}

pub fn installed() -> bool {
    ZIPPED.get().is_some()
}

pub fn get(path: &Path) -> Option<Vec<u8>> {
    ZIPPED.get()?.get(&tail(path)).cloned()
}

pub fn has(path: &Path) -> bool {
    ZIPPED
        .get()
        .is_some_and(|entries| entries.contains_key(&tail(path)))
}

pub fn keys() -> impl Iterator<Item = &'static str> {
    ZIPPED
        .get()
        .into_iter()
        .flat_map(|entries| entries.keys().map(String::as_str))
}
