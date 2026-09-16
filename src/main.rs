//! Desktop entry: 1280x720 window, assets when present, placeholders
//! otherwise (`NT_ASSETS` overrides the search; see `nt_rewrite` docs).
//! Hardware gamepads drain through `repame-shell`'s [`GamepadPoller`]
//! into backend-neutral [`GamepadState`] snapshots (bevy `sample_input`
//! pad-section parity); touch arrives as viewport `PickEvent`s.

use std::collections::HashMap;
use std::time::{Duration, Instant};

use glam::Vec2;
use nt_rewrite::input::GamepadState;
use nt_rewrite::{App, root_view};
use repame_shell::{GamepadEvent, GamepadId, GamepadPoller};
use repose_core::input::{GamepadAxis, GamepadButton};

/// Trigger axis threshold mapping to bevy's trigger *buttons*
/// (`RightTrigger2` fire / `LeftTrigger2` ability-spec): held past
/// half-press, edge on the rising crossing.
const TRIGGER_HELD: f32 = 0.5;

/// Per-pad hardware accumulation: axes + held buttons persist across
/// frames, pressed-edges clear after staging (bevy `pressed` vs
/// `just_pressed` split).
#[derive(Default)]
struct PadBridge {
    lx: f32,
    ly: f32,
    rx: f32,
    ry: f32,
    lt_held: bool,
    rt_held: bool,
    south: bool,
    east: bool,
    dpad_l: bool,
    dpad_u: bool,
    dpad_r: bool,
    north: bool,
    lt_edge: bool,
    rt_edge: bool,
    south_edge: bool,
    east_edge: bool,
    dpad_l_edge: bool,
    dpad_u_edge: bool,
    dpad_r_edge: bool,
    north_edge: bool,
}

impl PadBridge {
    fn button(&mut self, button: GamepadButton, pressed: bool) {
        let (held, edge) = match button {
            GamepadButton::South => (&mut self.south, &mut self.south_edge),
            GamepadButton::East => (&mut self.east, &mut self.east_edge),
            GamepadButton::North => (&mut self.north, &mut self.north_edge),
            GamepadButton::DPadLeft => (&mut self.dpad_l, &mut self.dpad_l_edge),
            GamepadButton::DPadUp => (&mut self.dpad_u, &mut self.dpad_u_edge),
            GamepadButton::DPadRight => (&mut self.dpad_r, &mut self.dpad_r_edge),
            _ => return,
        };
        if pressed && !*held {
            *edge = true;
        }
        *held = pressed;
    }

    fn axis(&mut self, axis: GamepadAxis, value: f32) {
        match axis {
            GamepadAxis::LeftStickX => self.lx = value,
            GamepadAxis::LeftStickY => self.ly = value,
            GamepadAxis::RightStickX => self.rx = value,
            GamepadAxis::RightStickY => self.ry = value,
            GamepadAxis::LeftTrigger => {
                let held = value > TRIGGER_HELD;
                if held && !self.lt_held {
                    self.lt_edge = true;
                }
                self.lt_held = held;
            }
            GamepadAxis::RightTrigger => {
                let held = value > TRIGGER_HELD;
                if held && !self.rt_held {
                    self.rt_edge = true;
                }
                self.rt_held = held;
            }
        }
    }

    fn snapshot(&self) -> GamepadState {
        GamepadState {
            left_stick: Vec2::new(self.lx, self.ly),
            right_stick: Vec2::new(self.rx, self.ry),
            right_trigger_held: self.rt_held,
            right_trigger_pressed: self.rt_edge,
            left_trigger_held: self.lt_held,
            left_trigger_pressed: self.lt_edge,
            south_pressed: self.south_edge,
            east_pressed: self.east_edge,
            dpad_left_pressed: self.dpad_l_edge,
            dpad_up_pressed: self.dpad_u_edge,
            dpad_right_pressed: self.dpad_r_edge,
            north_pressed: self.north_edge,
        }
    }

    fn clear_edges(&mut self) {
        self.lt_edge = false;
        self.rt_edge = false;
        self.south_edge = false;
        self.east_edge = false;
        self.dpad_l_edge = false;
        self.dpad_u_edge = false;
        self.dpad_r_edge = false;
        self.north_edge = false;
    }
}

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
    let mut pads: HashMap<GamepadId, PadBridge> = HashMap::new();
    let mut last = Instant::now();
    repame_shell::run_desktop("NT (repame)", (1280, 720), move |sched, ctx| {
        for ev in poller.poll() {
            match ev {
                GamepadEvent::Connected { .. } => {}
                GamepadEvent::Disconnected { id } => {
                    pads.remove(&id);
                }
                GamepadEvent::Button { id, button, pressed } => {
                    pads.entry(id).or_default().button(button, pressed);
                }
                GamepadEvent::Axis { id, axis, value } => {
                    pads.entry(id).or_default().axis(axis, value);
                }
            }
        }
        // Stage in stable id order (bevy `gamepads` query order).
        let mut ids: Vec<GamepadId> = pads.keys().copied().collect();
        ids.sort_by_key(|id| id.0);
        for id in ids {
            if let Some(bridge) = pads.get_mut(&id) {
                app.stage_gamepad(bridge.snapshot());
                bridge.clear_edges();
            }
        }
        let now = Instant::now();
        let dt = now.duration_since(last).min(Duration::from_secs_f32(0.25));
        last = now;
        root_view(sched, ctx, &mut app, dt)
    })
}
