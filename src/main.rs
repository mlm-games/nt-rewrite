//! Entry: desktop window plus Android NativeActivity (`cargo rapk`). Assets when
//! present, placeholders otherwise (`NT_ASSETS` overrides the search; see
//! `nt_rewrite` docs). Hardware gamepads drain through `repame-shell`'s
//! [`GamepadPoller`] into backend-neutral [`GamepadState`] snapshots, which
//! `App::stage_gamepad` reads one per tick; touch arrives as viewport
//! `PickEvent`s. GML has no poll/snapshot split: it reads the pad inline
//! (`InputHandling/InputHandling.gml:438-445` for the aim stick,
//! `scripts/scrGamepadUIControl/scrGamepadUIControl.gml:2-3,33-34,53-57`
//! for menu cursor rows), so the fan-out is port-only.

#[cfg(all(not(target_os = "android"), not(target_arch = "wasm32")))]
use web_time::{Duration, Instant};

#[cfg(all(not(target_os = "android"), not(target_arch = "wasm32")))]
use nt_rewrite::{App, audio_host::AudioHost, root_view};

#[cfg(all(not(target_os = "android"), not(target_arch = "wasm32")))]
fn boot() -> App {
    boot_at(None)
}

#[cfg(all(not(target_os = "android"), not(target_arch = "wasm32")))]
fn boot_at(files_dir: Option<std::path::PathBuf>) -> App {
    let debug_state = std::env::var("NT_DEBUG_STATE").ok();
    // Debug boots must not touch the real mid-run save: `setup_run` writes
    // one as soon as the first floor generates.
    if debug_state.is_some() && std::env::var("NT_RUN_SAVE_DIR").is_err() {
        // Safety: boot runs before any thread starts (edition 2024 set_var).
        unsafe {
            std::env::set_var(
                "NT_RUN_SAVE_DIR",
                std::env::temp_dir().join("nt-rewrite-debug"),
            );
        }
    }
    // An `ingame` debug boot takes the live-run seam (as if Loading just
    // ran `setup_run`): the plain boot carries no run, so the state swap
    // alone would sit the debug entry in an empty world.
    let mut app = if debug_state.as_deref() == Some("ingame") {
        App::new_with_seed(rand::random::<u64>())
    } else {
        App::new()
    };
    if let Some(st) = debug_state {
        let state = match st.as_str() {
            "title" => nt_rewrite::state::AppState::Title,
            "ingame" => nt_rewrite::state::AppState::InGame,
            _ => nt_rewrite::state::AppState::MainMenu,
        };
        nt_rewrite::state::goto_state(&mut app.sim.world, state);
    }
    #[cfg(target_os = "android")]
    if let Some(dir) = files_dir.as_ref() {
        if app.load_assets_from(&dir.join("assets")).is_ok() {
            eprintln!("nt: assets loaded from {}", dir.join("assets").display());
        } else if let Ok(found) = app.load_assets() {
            eprintln!("nt: assets loaded from {}", found.display());
        } else {
            eprintln!("nt: running without assets; placeholder renderer");
        }
    }
    #[cfg(not(target_os = "android"))]
    let _ = &files_dir;
    #[cfg(not(target_os = "android"))]
    match app.load_assets() {
        Ok(dir) => eprintln!("nt: assets loaded from {}", dir.display()),
        Err(e) => eprintln!("nt: running without assets ({e}); placeholder renderer"),
    }
    let save_path = match files_dir {
        Some(dir) => dir.join(nt_rewrite::savedata_part::save_file_name()),
        None => nt_rewrite::savedata_part::save_file_path(),
    };
    let save = app.load_save(&save_path);
    eprintln!(
        "nt: save loaded from {} (version {})",
        save_path.display(),
        save.version
    );
    app
}

#[cfg(all(not(target_os = "android"), not(target_arch = "wasm32")))]
fn main() -> anyhow::Result<()> {
    use repame_shell::{GamepadPoller, PadBank};

    let mut app = boot();
    let mut poller = GamepadPoller::new();
    let mut bank = PadBank::default();
    let mut audio = AudioHost::new();
    let mut last = Instant::now();
    let size = std::env::var("NT_WINDOW")
        .ok()
        .and_then(|v| {
            let (a, b) = v.split_once('x')?;
            Some((a.parse().ok()?, b.parse().ok()?))
        })
        .unwrap_or((1280, 720));
    repame_shell::run_desktop("NT (repame)", size, move |sched, ctx| {
        bank.feed(poller.poll());
        for pad in bank.drain() {
            app.stage_gamepad(pad);
        }
        let now = Instant::now();
        let dt = now.duration_since(last).min(Duration::from_secs_f32(0.25));
        last = now;
        let view = root_view(sched, ctx, &mut app, dt);
        audio.pump(dt.as_secs_f32(), &mut app);
        view
    })
}

#[cfg(any(target_os = "android", target_arch = "wasm32"))]
fn main() {}
