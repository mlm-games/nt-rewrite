//! Web entry: the shell page drops the assets zip before the game
//! loads; [`start`] awaits it through `window.__ntAssetsPromise`,
//! installs it into [`assetfs`](crate::assetfs), then boots the same
//! desktop frame loop on `repame_shell::run_web`.

use web_time::Duration;

use wasm_bindgen::prelude::*;

use crate::{App, root_view};

fn boot_status(message: &str, failed: bool) {
    let global = js_sys::global();
    let Ok(func) = js_sys::Reflect::get(&global, &JsValue::from_str("__ntBootStatus")) else {
        return;
    };
    let Ok(func) = func.dyn_into::<js_sys::Function>() else {
        return;
    };
    let _ = func.call2(
        &global,
        &JsValue::from_str(message),
        &JsValue::from_bool(failed),
    );
}

async fn shell_assets() -> anyhow::Result<Option<Vec<u8>>> {
    let global = js_sys::global();
    let promise = js_sys::Reflect::get(&global, &JsValue::from_str("__ntAssetsPromise"))
        .map_err(|_| anyhow::anyhow!("assets channel missing"))?;
    if promise.dyn_ref::<js_sys::Promise>().is_none() {
        return Ok(None);
    }
    let value = wasm_bindgen_futures::JsFuture::from(js_sys::Promise::from(promise))
        .await
        .map_err(|e| anyhow::anyhow!("assets read failed: {e:?}"))?;
    if value.is_null() || value.is_undefined() {
        return Ok(None);
    }
    Ok(Some(js_sys::Uint8Array::new(&value).to_vec()))
}

fn run() -> anyhow::Result<()> {
    let mut app = App::new();
    app.load_assets()
        .map_err(|e| anyhow::anyhow!("assets failed: {e}"))?;
    let save_path = crate::savedata_part::save_file_path();
    let save = app.load_save(&save_path);
    log::info!(
        "nt: save loaded from {} (version {})",
        save_path.display(),
        save.version
    );
    let mut audio = crate::audio_host::AudioHost::new();
    let mut poller = repame_shell::GamepadPoller::new();
    let mut bank = repame_shell::PadBank::default();
    let mut last = web_time::Instant::now();
    repame_shell::run_web(move |sched, ctx| {
        bank.feed(poller.poll());
        for pad in bank.drain() {
            app.stage_gamepad(pad);
        }
        let now = web_time::Instant::now();
        let dt = now.duration_since(last).min(Duration::from_secs_f32(0.25));
        last = now;
        let view = root_view(sched, ctx, &mut app, dt);
        audio.pump(dt.as_secs_f32(), &mut app);
        view
    })
    .map_err(|e| anyhow::anyhow!("{e:?}"))?;
    Ok(())
}

#[wasm_bindgen(start)]
pub fn start() {
    wasm_bindgen_futures::spawn_local(async {
        boot_status("loading assets", false);
        let zip = match shell_assets().await {
            Ok(Some(bytes)) => bytes,
            Ok(None) => {
                boot_status(
                    "no assets found; drop the assets zip on the start page",
                    true,
                );
                return;
            }
            Err(e) => {
                boot_status(&format!("assets unavailable: {e}"), true);
                return;
            }
        };
        if let Err(e) = crate::assetfs::install(&zip) {
            boot_status(&format!("bad assets zip: {e}"), true);
            return;
        }
        match run() {
            Ok(()) => boot_status("ready", false),
            Err(e) => boot_status(&format!("boot failed: {e}"), true),
        }
    });
}
