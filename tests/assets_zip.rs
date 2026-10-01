//! Assets-zip golden: a zip of the packaged assets (the format the
//! web shell drops) installs into the in-memory store and boots
//! `App::load_assets` with no disk reads.

use std::path::Path;

use nt_rewrite::assetfs;

fn pack(dir: &Path, prefix: &str, out: &mut zip::ZipWriter<std::io::Cursor<Vec<u8>>>) {
    let entries = std::fs::read_dir(dir).expect("read assets dir");
    for entry in entries {
        let entry = entry.expect("dir entry");
        let path = entry.path();
        let name = format!("{prefix}/{}", path.file_name().unwrap().to_string_lossy());
        if path.is_dir() {
            pack(&path, &name, out);
        } else {
            let options = zip::write::SimpleFileOptions::default()
                .compression_method(zip::CompressionMethod::Deflated);
            out.start_file(&name, options).expect("start entry");
            let bytes = std::fs::read(&path).expect("read file");
            std::io::Write::write_all(out, &bytes).expect("write entry");
        }
    }
}

#[test]
fn assets_zip_installs_and_loads() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("packaging/assets");
    let catalog = root.join("images/anims.json");
    if !catalog.is_file() {
        return;
    }
    let mut writer = zip::ZipWriter::new(std::io::Cursor::new(Vec::new()));
    pack(&root, "assets", &mut writer);
    let options = zip::write::SimpleFileOptions::default()
        .compression_method(zip::CompressionMethod::Deflated);
    writer
        .start_file("assets/sounds/probe/probe.ogg", options)
        .expect("start sound entry");
    std::io::Write::write_all(&mut writer, b"OggS").expect("write sound entry");
    let zip = writer.finish().expect("zip finished").into_inner();

    assetfs::install(&zip).expect("zip installs");
    assert!(
        assetfs::get(Path::new("/nt-assets/images/anims.json")).is_some(),
        "catalog resolves through the in-memory store"
    );
    assert!(
        assetfs::has(Path::new("/nt-assets/sounds/probe/probe.ogg")),
        "sound stems survive the install"
    );
    assert!(
        assetfs::keys().any(|k| k == "sounds/probe/probe.ogg"),
        "sound keys are visible to the stem scan"
    );
    let mut app = nt_rewrite::App::new();
    app.load_assets().expect("catalog loads from the zip");
}
