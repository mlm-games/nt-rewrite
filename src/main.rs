//! Desktop entry: 1280x720 window, assets when present, placeholders
//! otherwise (`NT_ASSETS` overrides the search; see `nt_rewrite` docs).
//! Hardware gamepads drain through `repame-shell`'s [`GamepadPoller`]
//! into backend-neutral [`GamepadState`] snapshots (bevy `sample_input`
//! pad-section parity); touch arrives as viewport `PickEvent`s.

use std::time::{Duration, Instant};

use nt_rewrite::{App, root_view};
use repame_shell::{GamepadPoller, PadBank};

fn main() -> anyhow::Result<()> {
    let mut app = App::new();
    match app.load_assets() {
        Ok(dir) => eprintln!("nt: assets loaded from {}", dir.display()),
        Err(e) => eprintln!("nt: running without assets ({e}); placeholder renderer"),
    }
    let save_path = nt_rewrite::savedata_part::save_file_path();
    let save = app.load_save(&save_path);
    eprintln!(
        "nt: save loaded from {} (version {})",
        save_path.display(),
        save.version
    );
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
