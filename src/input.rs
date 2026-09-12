//! Player input state. `NtInput` is pure data with take-once pulse
//! semantics (identical to the bevy build); the samplers below fill it
//! from keyboard/mouse/gamepad/touch (bevy `sample_input` layers
//! verbatim). Shells stage backend-neutral snapshots (`MouseState`,
//! `GamepadState`, `TouchContact`); winit/web event wiring is the only
//! shell-side piece.

use bevy_ecs::prelude::*;
use glam::Vec2;
use std::collections::HashSet;

use crate::comps_a::MutationChoice;

/// Sampled player intent for one tick.
#[derive(Resource, Debug, Clone)]
pub struct NtInput {
    pub move_axis: Vec2,
    pub aim_axis: Vec2,
    pub fire_held: bool,

    fire_pressed: bool,
    ability_pressed: bool,
    interact_pressed: bool,

    pub spec_held: bool,
    spec_pressed: bool,
    weapon_slot: Option<usize>,
    cycle_weapon: i8,
}

impl Default for NtInput {
    fn default() -> Self {
        Self {
            move_axis: Vec2::ZERO,
            aim_axis: Vec2::ZERO,
            fire_held: false,
            fire_pressed: false,
            ability_pressed: false,
            interact_pressed: false,
            spec_held: false,
            spec_pressed: false,
            weapon_slot: None,
            cycle_weapon: 0,
        }
    }
}

impl NtInput {
    pub fn take_fire_pressed(&mut self) -> bool {
        std::mem::take(&mut self.fire_pressed)
    }

    pub fn take_ability_pressed(&mut self) -> bool {
        std::mem::take(&mut self.ability_pressed)
    }

    pub fn take_spec_pressed(&mut self) -> bool {
        std::mem::take(&mut self.spec_pressed)
    }

    pub fn take_interact_pressed(&mut self) -> bool {
        std::mem::take(&mut self.interact_pressed)
    }

    pub fn peek_interact_pressed(&self) -> bool {
        self.interact_pressed
    }

    pub fn take_weapon_slot(&mut self) -> Option<usize> {
        self.weapon_slot.take()
    }

    pub fn take_cycle_weapon(&mut self) -> i8 {
        std::mem::take(&mut self.cycle_weapon)
    }

    /// Test/sampler hook: inject a pulse the systems will take once.
    pub fn press_fire(&mut self) {
        self.fire_pressed = true;
    }

    /// Test/sampler hook: inject an ability pulse the systems will take once.
    pub fn press_ability(&mut self) {
        self.ability_pressed = true;
    }

    /// Test/sampler hook: inject an interact pulse the systems will take once.
    pub fn press_interact(&mut self) {
        self.interact_pressed = true;
    }

    /// Test/sampler hook: queue a direct weapon-slot select.
    pub fn select_weapon(&mut self, slot: usize) {
        self.weapon_slot = Some(slot);
    }

    /// Test/sampler hook: queue a cycle step (+1 / -1).
    pub fn cycle_weapon(&mut self, dir: i8) {
        self.cycle_weapon = dir;
    }

    pub fn clear_transient(&mut self) {
        self.fire_pressed = false;
        self.ability_pressed = false;
        self.interact_pressed = false;
        self.spec_pressed = false;
        self.spec_held = false;
        self.weapon_slot = None;
        self.cycle_weapon = 0;
    }
}

/// Gamepad stick dead zone with rescaled response (bevy parity).
pub fn dead_zone(value: Vec2) -> Vec2 {
    const DEAD_ZONE: f32 = 0.22;

    let length = value.length();
    if length <= DEAD_ZONE {
        return Vec2::ZERO;
    }

    let scaled = ((length - DEAD_ZONE) / (1.0 - DEAD_ZONE)).clamp(0.0, 1.0);
    value.normalize_or_zero() * scaled
}

/// Drop pulses at the end of every tick (bevy `clear_input_pulses`).
pub fn clear_input_pulses(mut input: ResMut<NtInput>) {
    input.clear_transient();
}

/// Minimal backend-neutral key codes covering every key bevy
/// `sample_input` / `handle_mutation_choice` read. Any shell (winit,
/// web, test harness) maps its native codes onto these; no winit/bevy
/// dependency.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum KeyCode {
    KeyW,
    KeyA,
    KeyS,
    KeyD,
    ArrowUp,
    ArrowDown,
    ArrowLeft,
    ArrowRight,
    Space,
    ShiftLeft,
    ShiftRight,
    KeyE,
    KeyF,
    KeyQ,
    KeyG,
    Tab,
    Digit1,
    Digit2,
    Digit3,
    Digit4,
}

/// Backend-neutral mouse button state for one tick: held vs pressed
/// this tick (bevy `pressed` vs `just_pressed`).
#[derive(Clone, Copy, Debug, Default)]
pub struct MouseState {
    pub left_held: bool,
    pub left_pressed: bool,
    pub right_held: bool,
    pub right_pressed: bool,
}

/// Backend-neutral gamepad snapshot for one tick: raw stick axes plus
/// held/pressed edges for exactly the buttons bevy `sample_input`
/// reads (right/left trigger 2, South/East, D-pad L/U/R, North).
/// Any shell (repame-shell `GamepadPoller`, winit, test harness) maps
/// its native events onto this; the sim law below is bevy-verbatim.
#[derive(Clone, Copy, Debug, Default)]
pub struct GamepadState {
    pub left_stick: Vec2,
    pub right_stick: Vec2,
    pub right_trigger_held: bool,
    pub right_trigger_pressed: bool,
    pub left_trigger_held: bool,
    pub left_trigger_pressed: bool,
    pub south_pressed: bool,
    pub east_pressed: bool,
    pub dpad_left_pressed: bool,
    pub dpad_up_pressed: bool,
    pub dpad_right_pressed: bool,
    pub north_pressed: bool,
}

/// Backend-neutral touch contact: `start` is the touchdown point,
/// `pos` the current point (both in screen px, y-down from the top),
/// `just_pressed` marks contacts that began this tick (bevy
/// `iter_just_pressed` vs `iter`).
#[derive(Clone, Copy, Debug, Default)]
pub struct TouchContact {
    pub start: Vec2,
    pub pos: Vec2,
    pub just_pressed: bool,
}

/// WASD/arrows move vector (bevy `keyboard_move` law verbatim:
/// y-up, opposing pairs cancel, normalized diagonals).
pub fn keyboard_move(held: &HashSet<KeyCode>) -> Vec2 {
    let mut value = Vec2::ZERO;

    if held.contains(&KeyCode::KeyW) || held.contains(&KeyCode::ArrowUp) {
        value.y += 1.0;
    }
    if held.contains(&KeyCode::KeyS) || held.contains(&KeyCode::ArrowDown) {
        value.y -= 1.0;
    }
    if held.contains(&KeyCode::KeyA) || held.contains(&KeyCode::ArrowLeft) {
        value.x -= 1.0;
    }
    if held.contains(&KeyCode::KeyD) || held.contains(&KeyCode::ArrowRight) {
        value.x += 1.0;
    }

    value.normalize_or_zero()
}

/// Stick response shared by future gamepad/touch shells (bevy law:
/// `dead_zone` rescale, then length clamp — `sample_input` applied the
/// clamp when writing `move_axis`/`aim_axis`).
pub fn apply_stick(raw: Vec2) -> Vec2 {
    dead_zone(raw).clamp_length_max(1.0)
}

/// Backend-neutral port of bevy `sample_input`'s keyboard+mouse path.
/// `held` = keys down now, `just_pressed` = pressed-this-tick edges;
/// accumulation into `output` is bevy-verbatim (axes/held overwritten,
/// pulses OR-ed so a shell can layer gamepad on top, weapon slot replaced
/// only when a digit edge fires, cycle saturating-added).
///
/// Not covered here (different layers): mouse-cursor aim (bevy computed
/// that from a viewport ray in the render layer, never in `sample_input`
/// — keyboard/mouse leaves `aim_axis` zero, same as bevy; the `App`
/// hover block replicates it), gamepad sticks/triggers/d-pad (see
/// `sample_gamepad`), touch zones (see `sample_touch`).
pub fn sample_keyboard(
    held: &HashSet<KeyCode>,
    just_pressed: &HashSet<KeyCode>,
    mouse: &MouseState,
    output: &mut NtInput,
) {
    let move_axis = keyboard_move(held);
    let aim_axis = Vec2::ZERO;

    let fire_held = mouse.left_held || held.contains(&KeyCode::Space);
    let fire_pressed = mouse.left_pressed || just_pressed.contains(&KeyCode::Space);
    let ability_pressed = mouse.right_pressed
        || just_pressed.contains(&KeyCode::ShiftLeft)
        || just_pressed.contains(&KeyCode::ShiftRight);

    let spec_held = mouse.right_held
        || held.contains(&KeyCode::ShiftLeft)
        || held.contains(&KeyCode::ShiftRight);
    let spec_pressed = mouse.right_pressed
        || just_pressed.contains(&KeyCode::ShiftLeft)
        || just_pressed.contains(&KeyCode::ShiftRight);
    let interact_pressed = just_pressed.contains(&KeyCode::KeyE)
        || just_pressed.contains(&KeyCode::KeyF)
        || just_pressed.contains(&KeyCode::KeyQ)
        || just_pressed.contains(&KeyCode::KeyG)
        || just_pressed.contains(&KeyCode::Tab);

    let mut weapon_slot = None;
    if just_pressed.contains(&KeyCode::Digit1) {
        weapon_slot = Some(0);
    } else if just_pressed.contains(&KeyCode::Digit2) {
        weapon_slot = Some(1);
    } else if just_pressed.contains(&KeyCode::Digit3) {
        weapon_slot = Some(2);
    }
    // Keyboard contributes no cycle step (bevy: gamepad North / touch
    // button only).
    let cycle_weapon = 0_i8;

    output.move_axis = move_axis.clamp_length_max(1.0);
    output.aim_axis = aim_axis.clamp_length_max(1.0);
    output.fire_held = fire_held;
    output.spec_held = spec_held;

    output.fire_pressed |= fire_pressed;
    output.ability_pressed |= ability_pressed;
    output.interact_pressed |= interact_pressed;
    output.spec_pressed |= spec_pressed;

    if weapon_slot.is_some() {
        output.weapon_slot = weapon_slot;
    }
    output.cycle_weapon = output.cycle_weapon.saturating_add(cycle_weapon);
}

/// Backend-neutral port of bevy `sample_input`'s per-gamepad loop.
/// Nonzero dead-zoned sticks overwrite the axes (left = move, right =
/// aim); triggers/south/east OR into held/pulses; D-pad edges replace
/// the weapon slot. Returns this pad's cycle step (North = +1); the
/// caller applies the bevy overwrite law (last pad wins, added once —
/// see `sample_gamepads`).
pub fn sample_gamepad(pad: &GamepadState, output: &mut NtInput) -> i8 {
    let left = dead_zone(pad.left_stick);
    let right = dead_zone(pad.right_stick);

    if left != Vec2::ZERO {
        output.move_axis = left.clamp_length_max(1.0);
    }
    if right != Vec2::ZERO {
        output.aim_axis = right.clamp_length_max(1.0);
    }

    let mut fire_held = false;
    let mut fire_pressed = false;
    let mut ability_pressed = false;
    let mut spec_held_now = false;
    let mut spec_pressed_now = false;
    let mut interact_pressed = false;
    let mut weapon_slot = None;
    let mut cycle_weapon = 0_i8;

    fire_held |= pad.right_trigger_held;
    fire_pressed |= pad.right_trigger_pressed;
    ability_pressed |= pad.left_trigger_pressed;
    spec_held_now |= pad.left_trigger_held;
    spec_pressed_now |= pad.left_trigger_pressed;
    interact_pressed |= pad.south_pressed || pad.east_pressed;

    if pad.dpad_left_pressed {
        weapon_slot = Some(0);
    } else if pad.dpad_up_pressed {
        weapon_slot = Some(1);
    } else if pad.dpad_right_pressed {
        weapon_slot = Some(2);
    }

    if pad.north_pressed {
        cycle_weapon = 1;
    }

    // Bevy `sample_input` layers the pad loop over the keyboard/mouse
    // writes: axes overwritten above, held/pulses OR-accumulated,
    // slot replaced on edge (the cycle step returns for the
    // once-after-loop add).
    output.fire_held |= fire_held;
    output.spec_held |= spec_held_now;

    output.fire_pressed |= fire_pressed;
    output.ability_pressed |= ability_pressed;
    output.interact_pressed |= interact_pressed;
    output.spec_pressed |= spec_pressed_now;

    if weapon_slot.is_some() {
        output.weapon_slot = weapon_slot;
    }
    cycle_weapon
}

/// Bevy `sample_input` gamepad section verbatim: the per-pad body runs
/// in query order sharing one `cycle_weapon` local (plain assignment,
/// so the last pad with a North edge wins), and the step is
/// `saturating_add`ed exactly once after the loop.
pub fn sample_gamepads(pads: &[GamepadState], output: &mut NtInput) {
    let mut cycle_weapon = 0_i8;
    for pad in pads {
        cycle_weapon = sample_gamepad(pad, output);
    }
    output.cycle_weapon = output.cycle_weapon.saturating_add(cycle_weapon);
}

/// Backend-neutral port of bevy `sample_input`'s touch zones.
/// `window_width` is the viewport width in screen px. Top-right
/// 96px corners are the ability button (outer) and weapon-cycle
/// button (inner); other fresh touches on the right half fire;
/// held contacts become virtual sticks (`(pos - start)` y-flipped,
/// /56px, clamped, dead-zoned): left half steers move, right half
/// holds fire and steers aim. Contacts starting in the top button
/// strip never become sticks.
pub fn sample_touch(contacts: &[TouchContact], window_width: f32, output: &mut NtInput) {
    let width = window_width;

    for touch in contacts.iter().filter(|t| t.just_pressed) {
        let start = touch.start;

        if start.y < 96.0 && start.x >= width - 96.0 {
            output.ability_pressed |= true;
        } else if start.y < 96.0 && start.x >= width - 192.0 {
            output.cycle_weapon = output.cycle_weapon.saturating_add(1);
        } else if start.x >= width * 0.5 {
            output.fire_pressed |= true;
        }
    }

    for touch in contacts {
        let start = touch.start;

        if start.y < 96.0 && start.x >= width - 192.0 {
            continue;
        }

        let screen_delta = touch.pos - start;
        let stick = Vec2::new(screen_delta.x, -screen_delta.y) / 56.0;
        let stick = dead_zone(stick.clamp_length_max(1.0));

        if start.x < width * 0.5 {
            if stick != Vec2::ZERO {
                output.move_axis = stick;
            }
        } else {
            output.fire_held = true;
            if stick != Vec2::ZERO {
                output.aim_axis = stick;
            }
        }
    }

    output.move_axis = output.move_axis.clamp_length_max(1.0);
    output.aim_axis = output.aim_axis.clamp_length_max(1.0);
}

/// Digit1-4 -> pending mutation pick (headless equivalent of the digit
/// read at the top of bevy `handle_mutation_choice`: independent `if`s,
/// so a later digit wins when several edges land the same tick; never
/// overwrites a pick the UI already wrote).
pub fn sample_mutation_digits(just_pressed: &HashSet<KeyCode>, choice: &mut MutationChoice) {
    if choice.0.is_some() {
        return;
    }
    let mut picked: Option<usize> = None;
    if just_pressed.contains(&KeyCode::Digit1) {
        picked = Some(0);
    }
    if just_pressed.contains(&KeyCode::Digit2) {
        picked = Some(1);
    }
    if just_pressed.contains(&KeyCode::Digit3) {
        picked = Some(2);
    }
    if just_pressed.contains(&KeyCode::Digit4) {
        picked = Some(3);
    }
    if picked.is_some() {
        choice.0 = picked;
    }
}

/// Drop everything when the sim isn't live (paused or out of game —
/// bevy `clear_input_when_inactive` parity).
pub fn clear_input_when_inactive(
    paused: Res<crate::state::Paused>,
    state: Res<crate::state::AppState>,
    mut input: ResMut<NtInput>,
) {
    use crate::state::AppState;
    if paused.0 || *state != AppState::InGame {
        input.clear_transient();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pulses_take_once() {
        let mut input = NtInput::default();
        input.press_fire();
        assert!(input.take_fire_pressed());
        assert!(!input.take_fire_pressed());
        assert!(!input.take_interact_pressed());
    }

    #[test]
    fn clear_transient_resets_pulses() {
        let mut input = NtInput::default();
        input.press_fire();
        input.spec_held = true;
        input.clear_transient();
        assert!(!input.take_fire_pressed());
        assert!(!input.spec_held);
    }

    #[test]
    fn dead_zone_kills_drift_rescales_rest() {
        assert_eq!(dead_zone(Vec2::new(0.1, 0.1)), Vec2::ZERO);
        let out = dead_zone(Vec2::new(1.0, 0.0));
        assert!((out.x - 1.0).abs() < 1e-6 && out.y == 0.0);
        let mid = dead_zone(Vec2::new(0.61, 0.0));
        assert!((mid.x - 0.5).abs() < 1e-6, "got {mid:?}");
    }

    fn held_of(keys: &[KeyCode]) -> HashSet<KeyCode> {
        keys.iter().copied().collect()
    }

    #[test]
    fn keyboard_move_8way_normalized() {
        use KeyCode::*;
        // Single axis.
        assert_eq!(keyboard_move(&held_of(&[KeyD])), Vec2::new(1.0, 0.0));
        assert_eq!(keyboard_move(&held_of(&[KeyW])), Vec2::new(0.0, 1.0));
        assert_eq!(keyboard_move(&held_of(&[ArrowLeft])), Vec2::new(-1.0, 0.0));
        assert_eq!(keyboard_move(&held_of(&[ArrowDown])), Vec2::new(0.0, -1.0));
        // Opposing pairs cancel (bevy law).
        assert_eq!(keyboard_move(&held_of(&[KeyW, KeyS])), Vec2::ZERO);
        assert_eq!(keyboard_move(&held_of(&[KeyA, KeyD])), Vec2::ZERO);
        // Diagonal normalized.
        let diag = keyboard_move(&held_of(&[KeyW, KeyD]));
        let k = std::f32::consts::FRAC_1_SQRT_2;
        assert!((diag.x - k).abs() < 1e-6 && (diag.y - k).abs() < 1e-6, "got {diag:?}");
        assert!((diag.length() - 1.0).abs() < 1e-6);
        // Arrows mix with WASD.
        assert_eq!(
            keyboard_move(&held_of(&[ArrowUp, ArrowRight])),
            keyboard_move(&held_of(&[KeyW, KeyD]))
        );
        assert_eq!(keyboard_move(&HashSet::new()), Vec2::ZERO);
    }

    #[test]
    fn sample_keyboard_pulse_edges_fire_once_then_clear() {
        use KeyCode::*;
        let mut input = NtInput::default();
        let mouse = MouseState::default();

        // Tick 1: Space edge -> pulse set, held set.
        sample_keyboard(
            &held_of(&[Space]),
            &held_of(&[Space]),
            &mouse,
            &mut input,
        );
        assert!(input.fire_held);
        assert!(input.take_fire_pressed(), "edge fires");
        assert!(!input.take_fire_pressed(), "take-once");

        // Tick 2: still held, no new edge -> no pulse, held stays.
        sample_keyboard(&held_of(&[Space]), &HashSet::new(), &mouse, &mut input);
        assert!(input.fire_held);
        assert!(!input.take_fire_pressed(), "held alone must not re-fire");

        // End-of-tick clear drops held-derived levels only via fresh sample;
        // pulse gates clear explicitly.
        input.clear_transient();
        assert!(!input.take_fire_pressed());
    }

    #[test]
    fn sample_keyboard_maps_ability_spec_interact_and_slots() {
        use KeyCode::*;
        let mut input = NtInput::default();
        let mouse = MouseState {
            right_held: true,
            right_pressed: true,
            ..MouseState::default()
        };
        sample_keyboard(
            &held_of(&[ShiftLeft, KeyE, Digit2]),
            &held_of(&[ShiftLeft, KeyE, Digit2]),
            &mouse,
            &mut input,
        );
        assert!(input.take_ability_pressed(), "shift edge = ability");
        assert!(input.take_spec_pressed(), "right/shift edge = spec");
        assert!(input.spec_held, "shift held = spec held");
        assert!(input.take_interact_pressed(), "E edge = interact");
        assert_eq!(input.take_weapon_slot(), Some(1), "Digit2 = slot 1");
        assert_eq!(input.take_weapon_slot(), None, "slot takes once");

        // No digit edge -> previous slot selection preserved until consumed.
        input.select_weapon(0);
        sample_keyboard(&HashSet::new(), &HashSet::new(), &MouseState::default(), &mut input);
        assert_eq!(input.take_weapon_slot(), Some(0));
    }

    #[test]
    fn sample_keyboard_mouse_fire_and_stick_aim() {
        let mut input = NtInput::default();
        let down = MouseState {
            left_held: true,
            left_pressed: true,
            ..MouseState::default()
        };
        sample_keyboard(&HashSet::new(), &HashSet::new(), &down, &mut input);
        assert!(input.fire_held);
        assert!(input.take_fire_pressed());
        // Keyboard/mouse contributes no stick aim (bevy parity: mouse aim is
        // cursor-ray in the shell layer).
        assert_eq!(input.aim_axis, Vec2::ZERO);
        // Gamepad-shell helper keeps the dead_zone + clamp law.
        assert_eq!(apply_stick(Vec2::new(0.1, 0.0)), Vec2::ZERO);
        assert_eq!(apply_stick(Vec2::new(1.0, 0.0)), Vec2::new(1.0, 0.0));
    }

    #[test]
    fn mutation_digits_1_to_4_map_in_order() {        use KeyCode::*;
        let mut choice = MutationChoice(None);
        sample_mutation_digits(&held_of(&[Digit1]), &mut choice);
        assert_eq!(choice.0, Some(0));
        let mut choice = MutationChoice(None);
        sample_mutation_digits(&held_of(&[Digit2]), &mut choice);
        assert_eq!(choice.0, Some(1));
        let mut choice = MutationChoice(None);
        sample_mutation_digits(&held_of(&[Digit3]), &mut choice);
        assert_eq!(choice.0, Some(2));
        let mut choice = MutationChoice(None);
        sample_mutation_digits(&held_of(&[Digit4]), &mut choice);
        assert_eq!(choice.0, Some(3));

        // Later digit wins on multi-edge ticks (bevy independent-ifs law).
        let mut choice = MutationChoice(None);
        sample_mutation_digits(&held_of(&[Digit1, Digit4]), &mut choice);
        assert_eq!(choice.0, Some(3));

        // Never overwrite a pick already written; empty edges keep None.
        let mut choice = MutationChoice(Some(0));
        sample_mutation_digits(&held_of(&[Digit3]), &mut choice);
        assert_eq!(choice.0, Some(0));
        let mut choice = MutationChoice(None);
        sample_mutation_digits(&HashSet::new(), &mut choice);
        assert_eq!(choice.0, None);
    }

    #[test]
    fn sample_gamepad_sticks_buttons_verbatim() {
        // Sticks overwrite axes only when nonzero past the dead zone.
        let mut input = NtInput::default();
        sample_gamepad(
            &GamepadState {
                left_stick: Vec2::new(1.0, 0.0),
                right_stick: Vec2::new(0.0, 1.0),
                ..GamepadState::default()
            },
            &mut input,
        );
        assert_eq!(input.move_axis, Vec2::new(1.0, 0.0));
        assert_eq!(input.aim_axis, Vec2::new(0.0, 1.0));

        // Drift sticks leave prior axes alone.
        sample_gamepad(&GamepadState::default(), &mut input);
        assert_eq!(input.move_axis, Vec2::new(1.0, 0.0));

        // Triggers: RT2 fires, LT2 abilities + spec-holds.
        let mut input = NtInput::default();
        sample_gamepad(
            &GamepadState {
                right_trigger_held: true,
                right_trigger_pressed: true,
                left_trigger_held: true,
                left_trigger_pressed: true,
                ..GamepadState::default()
            },
            &mut input,
        );
        assert!(input.fire_held);
        assert!(input.take_fire_pressed());
        assert!(input.take_ability_pressed());
        assert!(input.spec_held);
        assert!(input.take_spec_pressed());

        // South/East interact, D-pad slots, North cycles once.
        let mut input = NtInput::default();
        sample_gamepads(
            &[GamepadState {
                south_pressed: true,
                dpad_up_pressed: true,
                north_pressed: true,
                ..GamepadState::default()
            }],
            &mut input,
        );
        assert!(input.take_interact_pressed());
        assert_eq!(input.take_weapon_slot(), Some(1));
        assert_eq!(input.take_cycle_weapon(), 1);

        // Two pads both on North still step once (bevy overwrite law).
        let mut input = NtInput::default();
        let north = GamepadState {
            north_pressed: true,
            ..GamepadState::default()
        };
        sample_gamepads(&[north, north], &mut input);
        assert_eq!(input.take_cycle_weapon(), 1);
    }

    #[test]
    fn sample_touch_zones_verbatim() {
        let w = 1280.0;
        // Outer corner = ability, inner corner = cycle step.
        let mut input = NtInput::default();
        sample_touch(
            &[
                TouchContact {
                    start: Vec2::new(w - 48.0, 48.0),
                    pos: Vec2::new(w - 48.0, 48.0),
                    just_pressed: true,
                },
                TouchContact {
                    start: Vec2::new(w - 144.0, 48.0),
                    pos: Vec2::new(w - 144.0, 48.0),
                    just_pressed: true,
                },
            ],
            w,
            &mut input,
        );
        assert!(input.take_ability_pressed());
        assert_eq!(input.take_cycle_weapon(), 1);
        // Button-strip contacts never become sticks.
        assert_eq!(input.move_axis, Vec2::ZERO);
        assert!(!input.fire_held);

        // Fresh right-half touch fires; held right contact holds fire
        // and steers aim (drag up = +y, /56px, dead-zoned).
        let mut input = NtInput::default();
        sample_touch(
            &[TouchContact {
                start: Vec2::new(w * 0.75, 400.0),
                pos: Vec2::new(w * 0.75, 400.0),
                just_pressed: true,
            }],
            w,
            &mut input,
        );
        assert!(input.take_fire_pressed());
        sample_touch(
            &[TouchContact {
                start: Vec2::new(w * 0.75, 400.0),
                pos: Vec2::new(w * 0.75, 344.0),
                just_pressed: false,
            }],
            w,
            &mut input,
        );
        assert!(input.fire_held);
        assert!(input.aim_axis.y > 0.9, "got {:?}", input.aim_axis);

        // Left-half drag steers move.
        let mut input = NtInput::default();
        sample_touch(
            &[TouchContact {
                start: Vec2::new(w * 0.25, 400.0),
                pos: Vec2::new(w * 0.25 + 56.0, 400.0),
                just_pressed: false,
            }],
            w,
            &mut input,
        );
        assert!(!input.fire_held);
        assert!(input.move_axis.x > 0.9, "got {:?}", input.move_axis);
    }
}
