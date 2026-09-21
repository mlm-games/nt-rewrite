//! Entry: desktop window plus Android NativeActivity (`cargo rapk`).
//! Assets when present, placeholders otherwise (`NT_ASSETS` overrides the
//! search; see `nt_rewrite` docs). Hardware gamepads drain through
//! `repame-shell`'s [`GamepadPoller`] into backend-neutral [`GamepadState`]
//! snapshots (bevy `sample_input` pad-section parity); touch arrives as
//! viewport `PickEvent`s.

use std::time::{Duration, Instant};

use nt_rewrite::{App, root_view};

fn boot() -> App {
    boot_at(None)
}

fn boot_at(files_dir: Option<std::path::PathBuf>) -> App {
    let mut app = App::new();
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

#[cfg(not(target_os = "android"))]
fn main() -> anyhow::Result<()> {
    use repame_shell::{GamepadPoller, PadBank};

    let mut app = boot();
    let mut poller = GamepadPoller::new();
    let mut bank = PadBank::default();
    let mut audio = repame_audio::Audio::noop();
    let mut last = Instant::now();
    repame_shell::run_desktop("NT (repame)", (1280, 720), move |sched, ctx| {
        bank.feed(poller.poll());
        for pad in bank.drain() {
            app.stage_gamepad(pad);
        }
        let view = root_view(sched, ctx, &mut app, {
            let now = Instant::now();
            let dt = now.duration_since(last).min(Duration::from_secs_f32(0.25));
            last = now;
            dt
        });
        for cue in app.drain_audio_cues() {
            audio.play(cue.name);
        }
        view
    })
}

#[cfg(target_os = "android")]
fn main() {}
