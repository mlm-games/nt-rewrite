//! Player input state. `NtInput` is pure data with take-once pulse
//! semantics (identical to the bevy build); the samplers below fill it
//! from keyboard/mouse/gamepad/touch (bevy `sample_input` layers
//! verbatim). Shells stage backend-neutral snapshots (`MouseState`,
//! `GamepadState`, `TouchContact`); winit/web event wiring is the only
//! shell-side piece.

use bevy_ecs::prelude::*;
use glam::Vec2;
use repose_core::input::PhysicalKey;
use std::collections::HashSet;

/// Sampled player intent for one tick.
#[derive(Resource, Debug, Clone)]
pub struct NtInput {
    pub move_axis: Vec2,
    pub aim_axis: Vec2,
    pub fire_held: bool,
    /// GML `JoystickAttack.dis` (px deflection, `min(rad, mdis) * 2`):
    /// trigger distance for the touch attack stick. Mouse/keyboard
    /// leave 0; touch aims read it for spread/crosshair distance.
    pub touch_dis: f32,
    /// GML `JoystickMove` stick anchor (GUI px). `None` until the first
    /// touch claims the move stick; persists while claimed.
    pub move_stick: Option<TouchStick>,
    /// GML `JoystickAttack` stick anchor (GUI px). Lerps toward the
    /// claiming touch at 0.8 (`scrStickRegions`).
    pub attack_stick: Option<TouchStick>,

    fire_pressed: bool,
    fire_released: bool,
    ability_pressed: bool,
    interact_pressed: bool,

    pub spec_held: bool,
    spec_pressed: bool,
    weapon_slot: Option<usize>,
    cycle_weapon: i8,
    /// Menu cursor steps staged by the shell feed (`feed_input`) for
    /// keyboard navigation where no gameplay channel exists: vertical
    /// (Up/Down: main-menu rows, settings rows) and horizontal
    /// (Left/Right: settings sliders/cycles). Take-once, like
    /// `cycle_weapon`; consumed by `tick_menus`.
    menu_nav_v: i8,
    menu_nav_h: i8,
    touch_released_swap: bool,
    touch_released_fire: bool,
    /// Fingers lifted while a stick claimed them, by shell finger id.
    /// Drained on the next `sample_touch`: a missing id matching a
    /// live claim fires that element's lift edge (attack finger →
    /// swapped `press_fire` + release consumers; anything else → no
    /// edge, GML reads releases off the claiming element only).
    pub touch_lifted: Vec<i64>,
}

/// GML `MobileUI` stick claim verbatim: GUI-px anchor, claimed touch
/// id (`index`, -1 = free), deflection (`dis`, px from anchor), heading
/// (`dir`, degrees), and the move-stick direction-hold ramp
/// (`current_move_direction_time`: +1/tick within 10° of the held
/// heading, −3/tick otherwise).
///
/// Touch ids here are the shell finger ids (`TouchContact.id`, GML
/// touch slot 0-4): claims key on the stable finger, never on the
/// contact's position in the per-frame slice.
#[derive(Clone, Copy, Debug, Default)]
pub struct TouchStick {
    pub anchor: Vec2,
    pub touch: i64,
    pub dis: f32,
    pub dir: f32,
    pub hold_time: f32,
}

/// GML `JoystickAttack/Create_0` verbatim.
pub const ATTACK_BUTTON_DEADZONE: f32 = 0.4125;
/// GML stick radius (`JoystickMove/Create_0`, `JoystickAttack/Create_0`).
pub const TOUCH_STICK_RADIUS: f32 = 32.0;
/// GML `ButtonAct`/`ButtonSwap` radius (`rad = 25`).
pub const TOUCH_BUTTON_RADIUS: f32 = 25.0;

impl Default for NtInput {
    fn default() -> Self {
        Self {
            move_axis: Vec2::ZERO,
            aim_axis: Vec2::ZERO,
            fire_held: false,
            touch_dis: 0.0,
            move_stick: None,
            attack_stick: None,
            fire_pressed: false,
            fire_released: false,
            ability_pressed: false,
            interact_pressed: false,
            spec_held: false,
            spec_pressed: false,
            weapon_slot: None,
            cycle_weapon: 0,
            menu_nav_v: 0,
            menu_nav_h: 0,
            touch_released_swap: false,
            touch_released_fire: false,
            touch_lifted: Vec::new(),
        }
    }
}

impl NtInput {
    pub fn take_fire_pressed(&mut self) -> bool {
        std::mem::take(&mut self.fire_pressed)
    }

    /// GML `release_fire`: the attack-finger lift edge (the stick swaps
    /// press/release by design; `JoystickAttack` sets `press_fire` on
    /// release).
    pub fn take_fire_released(&mut self) -> bool {
        std::mem::take(&mut self.fire_released)
    }

    pub fn peek_fire_pressed(&self) -> bool {
        self.fire_pressed
    }

    /// Take the latched touch-release fire edge. `sample_touch` latches
    /// it two ways: the attack stick's swapped press frame writes
    /// `touch_released_fire` itself, and the finger-lift path stages
    /// `fire_released` directly — so consumers take both.
    pub fn take_touch_released_fire(&mut self) -> bool {
        self.fire_released || std::mem::take(&mut self.touch_released_fire)
    }

    /// Test/sampler hook: stage the lift edge directly (mirrors the
    /// attack finger lifting off past the deadzone).
    pub fn release_fire(&mut self) {
        self.fire_released = true;
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

    /// Take staged menu cursor steps `(vertical, horizontal)`.
    pub fn take_menu_nav(&mut self) -> (i8, i8) {
        (
            std::mem::take(&mut self.menu_nav_v),
            std::mem::take(&mut self.menu_nav_h),
        )
    }

    /// Stage a menu cursor step (saturating; the menu tick takes it once).
    pub fn push_menu_nav(&mut self, dv: i8, dh: i8) {
        self.menu_nav_v = self.menu_nav_v.saturating_add(dv);
        self.menu_nav_h = self.menu_nav_h.saturating_add(dh);
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

    /// Drain the peek-only interact pulse (Cleanup tail, live play).
    pub(crate) fn clear_interact_pulse(&mut self) {
        self.interact_pressed = false;
    }

    pub fn clear_transient(&mut self) {
        self.move_axis = Vec2::ZERO;
        self.aim_axis = Vec2::ZERO;
        self.fire_held = false;
        self.fire_pressed = false;
        self.fire_released = false;
        self.touch_dis = 0.0;
        self.ability_pressed = false;
        self.interact_pressed = false;
        self.spec_pressed = false;
        self.spec_held = false;
        self.weapon_slot = None;
        self.cycle_weapon = 0;
        self.menu_nav_v = 0;
        self.menu_nav_h = 0;
        self.touch_released_swap = false;
        self.touch_released_fire = false;
    }

    /// Shell `TouchUp` latch: the shell drops contacts on lift, so the
    /// lifted finger id is staged here and `sample_touch` matches it
    /// against its live claims (see `touch_lifted`). Sticks already
    /// released stay released; unknown ids latch nothing.
    pub fn note_touch_released(&mut self, id: i64) {
        if id >= 0 && !self.touch_lifted.contains(&id) {
            self.touch_lifted.push(id);
        }
    }
}

/// Gamepad stick dead zone with rescaled response (bevy parity).
pub fn dead_zone(value: Vec2) -> Vec2 {
    repame_input::dead_zone(value)
}

/// Drop pulses at the end of every tick (bevy `clear_input_pulses`).
pub fn clear_input_pulses(mut input: ResMut<NtInput>) {
    input.clear_transient();
}

/// Drain the peek-only interact pulse after every live-play consumer
/// ran (`collect_pickups`, `tick_throne_sit` peek it; nothing takes
/// it). Without this one E tap stays true forever and re-equips
/// nearby guns each tick. Folded into `clear_input_when_inactive`'s
/// (state, input) params so no second `ResMut<NtInput>` conflicts.
fn drain_interact_pulse(state: &crate::state::AppState, input: &mut NtInput) {
    use crate::state::AppState;
    if *state == AppState::InGame {
        input.clear_interact_pulse();
    }
}

/// Minimal backend-neutral key codes covering every key bevy
/// `sample_input` / `handle_mutation_choice` read. Any shell (winit,
/// web, test harness) maps its native codes onto these; no winit/bevy
/// dependency. Physical winit `KeyCode` debug names map 1:1 here
/// (`physical_key_name` in `repose-platform`), so games can poll
/// layout-independent positions instead of characters.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum KeyCode {
    KeyA,
    KeyB,
    KeyC,
    KeyD,
    KeyE,
    KeyF,
    KeyG,
    KeyH,
    KeyI,
    KeyJ,
    KeyK,
    KeyL,
    KeyM,
    KeyN,
    KeyO,
    KeyP,
    KeyQ,
    KeyR,
    KeyS,
    KeyT,
    KeyU,
    KeyV,
    KeyW,
    KeyX,
    KeyY,
    KeyZ,
    ArrowUp,
    ArrowDown,
    ArrowLeft,
    ArrowRight,
    Space,
    ShiftLeft,
    ShiftRight,
    Backquote,
    Tab,
    Digit0,
    Digit1,
    Digit2,
    Digit3,
    Digit4,
    Digit5,
    Digit6,
    Digit7,
    Digit8,
    Digit9,
}

/// `Scheduler::held_keys` position to the backend-neutral [`KeyCode`].
/// Derived from the W3C `name()`: letters/digits/Backquote spell their
/// own [`KeyCode`], the rest match by name. Returns `None` for keys
/// the sim never reads (F-keys, punctuation, numpad, modifiers...).
pub fn keycode_for_physical(key: PhysicalKey) -> Option<KeyCode> {
    let name = key.name();
    if let Some(tail) = name
        .strip_prefix("Key")
        .or_else(|| name.strip_prefix("Digit"))
    {
        let mut chars = tail.chars();
        match (chars.next(), chars.next()) {
            (Some(c), None) => {
                let upper = c.to_ascii_uppercase();
                if upper.is_ascii_alphabetic() {
                    return keycode_for_glyph(upper.to_ascii_lowercase());
                }
                if upper.is_ascii_digit() {
                    return keycode_for_glyph(upper);
                }
                return None;
            }
            _ => return None,
        }
    }
    Some(match key {
        PhysicalKey::ArrowUp => KeyCode::ArrowUp,
        PhysicalKey::ArrowDown => KeyCode::ArrowDown,
        PhysicalKey::ArrowLeft => KeyCode::ArrowLeft,
        PhysicalKey::ArrowRight => KeyCode::ArrowRight,
        PhysicalKey::Space => KeyCode::Space,
        PhysicalKey::ShiftLeft => KeyCode::ShiftLeft,
        PhysicalKey::ShiftRight => KeyCode::ShiftRight,
        PhysicalKey::Tab => KeyCode::Tab,
        PhysicalKey::Backquote => KeyCode::Backquote,
        _ => return None,
    })
}

pub use repame_input::{GamepadState, MouseState, TouchContact, apply_stick};

/// WASD/arrows move vector in world space. The world is y-down
/// (GML convention: north is −y, see `worldgen::Maker::step_delta`),
/// so W/Up is −y and S/Down is +y — the bevy build's y-up signs
/// flipped for this port's [`Pos`](crate::spatial::Pos) space.
/// Opposing pairs cancel, diagonals normalize.
pub fn keyboard_move(held: &HashSet<KeyCode>) -> Vec2 {
    let mut value = Vec2::ZERO;

    if held.contains(&KeyCode::KeyW) || held.contains(&KeyCode::ArrowUp) {
        value.y -= 1.0;
    }
    if held.contains(&KeyCode::KeyS) || held.contains(&KeyCode::ArrowDown) {
        value.y += 1.0;
    }
    if held.contains(&KeyCode::KeyA) || held.contains(&KeyCode::ArrowLeft) {
        value.x -= 1.0;
    }
    if held.contains(&KeyCode::KeyD) || held.contains(&KeyCode::ArrowRight) {
        value.x += 1.0;
    }

    value.normalize_or_zero()
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
    sample_keyboard_mapped(held, just_pressed, mouse, None, output)
}

/// [`sample_keyboard`] with an optional remap table. `held`/`just`
/// carry physical positions; each rebound entry is tested by position
/// (key entries) or button (mouse entries), so a rebound FIRE key
/// steers the same action GML's per-key poll would. Unbound sides
/// read inactive; arrows/digits/Tab stay fixed (UI channels, never
/// rebound rows).
pub fn sample_keyboard_mapped(
    held: &HashSet<KeyCode>,
    just_pressed: &HashSet<KeyCode>,
    mouse: &MouseState,
    keymap: Option<&crate::keymap::InputMapState>,
    output: &mut NtInput,
) {
    let move_axis = match keymap {
        Some(state) => keymap_move(&state.session.map, held),
        None => keyboard_move(held),
    };
    let aim_axis = Vec2::ZERO;

    let (fire_held, fire_pressed, spec_held_now, spec_pressed_now, swap_pressed, pick_pressed) =
        match keymap {
            Some(state) => {
                let fire = state.session.map.active(&crate::keymap::NtAction::Fire, false);
                let spec = state.session.map.active(&crate::keymap::NtAction::Spec, false);
                let swap = state.session.map.active(&crate::keymap::NtAction::Swap, false);
                let pick = state.session.map.active(&crate::keymap::NtAction::Pick, false);
                let fire_edge = entry_pressed(&fire, just_pressed, mouse, true);
                let spec_edge = entry_pressed(&spec, just_pressed, mouse, false);
                let shift_held = held.contains(&KeyCode::ShiftLeft)
                    || held.contains(&KeyCode::ShiftRight);
                let shift_edge = just_pressed.contains(&KeyCode::ShiftLeft)
                    || just_pressed.contains(&KeyCode::ShiftRight);
                (
                    entry_held(&fire, held, mouse) || mouse.left_held || fire_edge,
                    fire_edge || mouse.left_pressed,
                    entry_held(&spec, held, mouse)
                        || mouse.right_held
                        || spec_edge
                        || shift_held,
                    spec_edge || mouse.right_pressed || shift_edge,
                    entry_pressed(&swap, just_pressed, mouse, true),
                    entry_pressed(&pick, just_pressed, mouse, true),
                )
            }
            None => (
                mouse.left_held || held.contains(&KeyCode::Space),
                mouse.left_pressed || just_pressed.contains(&KeyCode::Space),
                mouse.right_held
                    || held.contains(&KeyCode::ShiftLeft)
                    || held.contains(&KeyCode::ShiftRight),
                mouse.right_pressed
                    || just_pressed.contains(&KeyCode::ShiftLeft)
                    || just_pressed.contains(&KeyCode::ShiftRight),
                just_pressed.contains(&KeyCode::Space),
                just_pressed.contains(&KeyCode::KeyE)
                    || just_pressed.contains(&KeyCode::KeyF)
                    || just_pressed.contains(&KeyCode::KeyQ)
                    || just_pressed.contains(&KeyCode::KeyG),
            ),
        };
    let swap_pressed = match keymap {
        Some(state)
            if state.session.map.keyboard(&crate::keymap::NtAction::Swap)
                != repame_input::KeymapEntry::None =>
        {
            swap_pressed
        }
        _ => swap_pressed || just_pressed.contains(&KeyCode::Space),
    };

    let ability_pressed = spec_pressed_now;
    let spec_held = spec_held_now;
    let spec_pressed = spec_pressed_now;
    let mut interact_pressed = just_pressed.contains(&KeyCode::Tab);
    if keymap.is_none() {
        interact_pressed |= just_pressed.contains(&KeyCode::KeyE)
            || just_pressed.contains(&KeyCode::KeyF)
            || just_pressed.contains(&KeyCode::KeyQ)
            || just_pressed.contains(&KeyCode::KeyG);
    } else {
        interact_pressed |= pick_pressed;
    }

    let mut weapon_slot = None;
    if just_pressed.contains(&KeyCode::Digit1) {
        weapon_slot = Some(0);
    } else if just_pressed.contains(&KeyCode::Digit2) {
        weapon_slot = Some(1);
    } else if just_pressed.contains(&KeyCode::Digit3) {
        weapon_slot = Some(2);
    } else if just_pressed.contains(&KeyCode::Digit4) {
        weapon_slot = Some(3);
    } else if just_pressed.contains(&KeyCode::Digit5) {
        // Main-menu 5th row (QUIT); gameplay consumers ignore it.
        weapon_slot = Some(4);
    }

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
    // GML `press_swap` is a pulse the player step consumes for the gun
    // swap (`Step_0:30`): route it as a +1 cycle step so the shared
    // `weapon_switch` path fires it.
    if swap_pressed && keymap.is_some() {
        output.cycle_weapon = output.cycle_weapon.saturating_add(1);
    }
}

/// Remapped move vector: each direction row reads its rebound entry by
/// position (key entries) — arrows always count on top (menu nav shares
/// them; GML reads `vk_*` on top of the rebinds). World space is y-down
/// (see [`keyboard_move`]), so north rows steer −y. Opposing pairs
/// cancel, diagonals normalize (same law as [`keyboard_move`]).
pub fn keymap_move(
    map: &repame_input::Keymap<crate::keymap::NtAction>,
    held: &HashSet<KeyCode>,
) -> Vec2 {
    use crate::keymap::NtAction;
    let mut value = Vec2::ZERO;
    if entry_held_pos(&map.active(&NtAction::North, false), held)
        || held.contains(&KeyCode::ArrowUp)
    {
        value.y -= 1.0;
    }
    if entry_held_pos(&map.active(&NtAction::South, false), held)
        || held.contains(&KeyCode::ArrowDown)
    {
        value.y += 1.0;
    }
    if entry_held_pos(&map.active(&NtAction::West, false), held)
        || held.contains(&KeyCode::ArrowLeft)
    {
        value.x -= 1.0;
    }
    if entry_held_pos(&map.active(&NtAction::East, false), held)
        || held.contains(&KeyCode::ArrowRight)
    {
        value.x += 1.0;
    }
    value.normalize_or_zero()
}

fn keycode_for_entry(entry: &repame_input::KeymapEntry) -> Option<KeyCode> {
    use repame_input::KeymapEntry;
    match entry {
        KeymapEntry::Key(chord) => keycode_for_chord(&chord.key),
        KeymapEntry::Physical(key) => keycode_for_physical(*key),
        KeymapEntry::None | KeymapEntry::Mouse(_) | KeymapEntry::Pad(_) | KeymapEntry::Axis { .. } => {
            None
        }
    }
}

fn keycode_for_chord(key: &repose_core::input::Key) -> Option<KeyCode> {
    use repose_core::input::Key;
    Some(match key {
        Key::Character(c) => keycode_for_glyph(*c)?,
        Key::Space => KeyCode::Space,
        Key::ShiftLeft => KeyCode::ShiftLeft,
        Key::ShiftRight => KeyCode::ShiftRight,
        Key::Tab => KeyCode::Tab,
        Key::ArrowUp => KeyCode::ArrowUp,
        Key::ArrowDown => KeyCode::ArrowDown,
        Key::ArrowLeft => KeyCode::ArrowLeft,
        Key::ArrowRight => KeyCode::ArrowRight,
        _ => return None,
    })
}

/// NT glyph coverage: the rebound chord alphabet is letters, digits
/// and backquote (see `default_keymap` + `chord_for_physical`). Chords
/// outside it (punctuation, F-keys) are unstageable by design —
// `keycode_for_entry` maps them to `None` and the sampler skips them.
fn keycode_for_glyph(c: char) -> Option<KeyCode> {
    Some(match c {
        'a' => KeyCode::KeyA,
        'b' => KeyCode::KeyB,
        'c' => KeyCode::KeyC,
        'd' => KeyCode::KeyD,
        'e' => KeyCode::KeyE,
        'f' => KeyCode::KeyF,
        'g' => KeyCode::KeyG,
        'h' => KeyCode::KeyH,
        'i' => KeyCode::KeyI,
        'j' => KeyCode::KeyJ,
        'k' => KeyCode::KeyK,
        'l' => KeyCode::KeyL,
        'm' => KeyCode::KeyM,
        'n' => KeyCode::KeyN,
        'o' => KeyCode::KeyO,
        'p' => KeyCode::KeyP,
        'q' => KeyCode::KeyQ,
        'r' => KeyCode::KeyR,
        's' => KeyCode::KeyS,
        't' => KeyCode::KeyT,
        'u' => KeyCode::KeyU,
        'v' => KeyCode::KeyV,
        'w' => KeyCode::KeyW,
        'x' => KeyCode::KeyX,
        'y' => KeyCode::KeyY,
        'z' => KeyCode::KeyZ,
        '0' => KeyCode::Digit0,
        '1' => KeyCode::Digit1,
        '2' => KeyCode::Digit2,
        '3' => KeyCode::Digit3,
        '4' => KeyCode::Digit4,
        '5' => KeyCode::Digit5,
        '6' => KeyCode::Digit6,
        '7' => KeyCode::Digit7,
        '8' => KeyCode::Digit8,
        '9' => KeyCode::Digit9,
        '`' => KeyCode::Backquote,
        _ => return None,
    })
}

/// Physical keys driving one [`KeyCode`] (`Scheduler::held_keys`
/// entries). Derived from the W3C `name()`: single letters/digits map
/// to their own positions, the rest by name. Used only to drop stuck
/// levels in the polled repair — never to stage edges.
pub fn physical_key_for_code(code: KeyCode) -> Option<PhysicalKey> {
    physical_keys_for_code(&code)
        .and_then(|keys| keys.first().copied())
}
pub fn physical_keys_for_code(code: &KeyCode) -> Option<&'static [PhysicalKey]> {
    use repose_core::input::PhysicalKey as P;
    Some(match code {
        KeyCode::ShiftLeft => &[P::ShiftLeft],
        KeyCode::ShiftRight => &[P::ShiftRight],
        KeyCode::KeyA => &[P::KeyA],
        KeyCode::KeyB => &[P::KeyB],
        KeyCode::KeyC => &[P::KeyC],
        KeyCode::KeyD => &[P::KeyD],
        KeyCode::KeyE => &[P::KeyE],
        KeyCode::KeyF => &[P::KeyF],
        KeyCode::KeyG => &[P::KeyG],
        KeyCode::KeyH => &[P::KeyH],
        KeyCode::KeyI => &[P::KeyI],
        KeyCode::KeyJ => &[P::KeyJ],
        KeyCode::KeyK => &[P::KeyK],
        KeyCode::KeyL => &[P::KeyL],
        KeyCode::KeyM => &[P::KeyM],
        KeyCode::KeyN => &[P::KeyN],
        KeyCode::KeyO => &[P::KeyO],
        KeyCode::KeyP => &[P::KeyP],
        KeyCode::KeyQ => &[P::KeyQ],
        KeyCode::KeyR => &[P::KeyR],
        KeyCode::KeyS => &[P::KeyS],
        KeyCode::KeyT => &[P::KeyT],
        KeyCode::KeyU => &[P::KeyU],
        KeyCode::KeyV => &[P::KeyV],
        KeyCode::KeyW => &[P::KeyW],
        KeyCode::KeyX => &[P::KeyX],
        KeyCode::KeyY => &[P::KeyY],
        KeyCode::KeyZ => &[P::KeyZ],
        KeyCode::ArrowUp => &[P::ArrowUp],
        KeyCode::ArrowDown => &[P::ArrowDown],
        KeyCode::ArrowLeft => &[P::ArrowLeft],
        KeyCode::ArrowRight => &[P::ArrowRight],
        KeyCode::Space => &[P::Space],
        KeyCode::Tab => &[P::Tab],
        KeyCode::Backquote => &[P::Backquote],
        KeyCode::Digit0 => &[P::Digit0],
        KeyCode::Digit1 => &[P::Digit1],
        KeyCode::Digit2 => &[P::Digit2],
        KeyCode::Digit3 => &[P::Digit3],
        KeyCode::Digit4 => &[P::Digit4],
        KeyCode::Digit5 => &[P::Digit5],
        KeyCode::Digit6 => &[P::Digit6],
        KeyCode::Digit7 => &[P::Digit7],
        KeyCode::Digit8 => &[P::Digit8],
        KeyCode::Digit9 => &[P::Digit9],
    })
}

fn entry_held_pos(entry: &repame_input::KeymapEntry, held: &HashSet<KeyCode>) -> bool {
    keycode_for_entry(entry).is_some_and(|code| held.contains(&code))
}

fn entry_held(
    entry: &repame_input::KeymapEntry,
    held: &HashSet<KeyCode>,
    _mouse: &MouseState,
) -> bool {
    use repame_input::KeymapEntry;
    match entry {
        // Button-specific: a Mouse entry only reads the button its
        // action binds (GML `fire` = mb_left only, `spec` = mb_right
        // only). The sampler ORs the matching `mouse.*` channel
        // explicitly, so a blanket `left || right` here would cross-talk
        // (LMB raising spec, RMB raising fire).
        KeymapEntry::Mouse(_) => false,
        _ => entry_held_pos(entry, held),
    }
}

fn entry_pressed(
    entry: &repame_input::KeymapEntry,
    just: &HashSet<KeyCode>,
    mouse: &MouseState,
    left: bool,
) -> bool {
    use repame_input::KeymapEntry;
    match entry {
        KeymapEntry::Mouse(_) => {
            if left {
                mouse.left_pressed
            } else {
                mouse.right_pressed
            }
        }
        _ => keycode_for_entry(entry).is_some_and(|code| just.contains(&code)),
    }
}

/// Backend-neutral port of bevy `sample_input`'s per-gamepad loop.
/// Nonzero dead-zoned sticks overwrite the axes (left = move, right =
/// aim); the remapped pad rows (default: Fire/Swap = RightShoulder,
/// Spec = LeftShoulder, Pick = South) OR into held/pulses; D-pad edges
/// replace the weapon slot. Triggers keep their hardcoded bevy role
/// (RT = fire, LT = spec/ability) alongside the rows. Returns this
/// pad's cycle step (North = +1); the caller applies the bevy overwrite
/// law (last pad wins, added once — see `sample_gamepads`).
///
/// Stick Y arrives screen-down (gilrs/SDL convention matches this
/// port's y-down world), so unlike the y-up bevy build no flip is
/// applied: stick-up (−y) moves north.
pub fn sample_gamepad(
    pad: &GamepadState,
    keymap: Option<&crate::keymap::InputMapState>,
    output: &mut NtInput,
) -> i8 {
    sample_gamepad_mapped(pad, keymap, output)
}

/// [`sample_gamepad`] with an explicit remap table (`None` = bevy
/// hardcoded behavior verbatim). Pad-side `Pad` entries read the
/// snapshot below; keyboard/mouse/axis entries on the pad side are
/// inert here (they belong to the keyboard sampler).
pub fn sample_gamepad_mapped(
    pad: &GamepadState,
    keymap: Option<&crate::keymap::InputMapState>,
    output: &mut NtInput,
) -> i8 {
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

    if let Some(state) = keymap {
        use crate::keymap::NtAction;
        let map = &state.session.map;
        let (fire_row, spec_row, swap_row, pick_row) = (
            map.gamepad(&NtAction::Fire),
            map.gamepad(&NtAction::Spec),
            map.gamepad(&NtAction::Swap),
            map.gamepad(&NtAction::Pick),
        );
        fire_held |= pad_held(&fire_row, pad);
        fire_pressed |= pad_pressed(&fire_row, pad);
        spec_held_now |= pad_held(&spec_row, pad);
        spec_pressed_now |= pad_pressed(&spec_row, pad);
        ability_pressed |= pad_pressed(&spec_row, pad);
        interact_pressed |= pad_pressed(&pick_row, pad);
        if pad_pressed(&swap_row, pad) {
            cycle_weapon = cycle_weapon.saturating_add(1);
        }
    }

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
    sample_gamepads_mapped(pads, None, output)
}

/// [`sample_gamepads`] with the live remap table: pad rows consult the
/// saved bindings (Fire/Swap shoulders, Pick south, ...), triggers
/// keep their hardcoded role underneath.
pub fn sample_gamepads_mapped(
    pads: &[GamepadState],
    keymap: Option<&crate::keymap::InputMapState>,
    output: &mut NtInput,
) {
    let mut cycle_weapon = 0_i8;
    for pad in pads {
        cycle_weapon = sample_gamepad_mapped(pad, keymap, output);
    }
    output.cycle_weapon = output.cycle_weapon.saturating_add(cycle_weapon);
}

/// Pad-side `Pad` entry -> snapshot held channel. Anything else is
/// inert (keyboard/mouse/axis rows belong to other samplers).
fn pad_held(entry: &repame_input::KeymapEntry, pad: &GamepadState) -> bool {
    use repame_input::KeymapEntry;
    use repose_core::input::GamepadButton;
    match entry {
        KeymapEntry::Pad(GamepadButton::RightShoulder) => pad.right_shoulder_held,
        KeymapEntry::Pad(GamepadButton::LeftShoulder) => pad.left_shoulder_held,
        _ => pad_pressed(entry, pad),
    }
}

/// Pad-side `Pad` entry -> snapshot press edge. Anything else is inert.
fn pad_pressed(entry: &repame_input::KeymapEntry, pad: &GamepadState) -> bool {
    use repame_input::KeymapEntry;
    use repose_core::input::GamepadButton;
    match entry {
        KeymapEntry::Pad(GamepadButton::South) => pad.south_pressed,
        KeymapEntry::Pad(GamepadButton::East) => pad.east_pressed,
        KeymapEntry::Pad(GamepadButton::West) => pad.west_pressed,
        KeymapEntry::Pad(GamepadButton::North) => pad.north_pressed,
        KeymapEntry::Pad(GamepadButton::LeftShoulder) => pad.left_shoulder_pressed,
        KeymapEntry::Pad(GamepadButton::RightShoulder) => pad.right_shoulder_pressed,
        KeymapEntry::Pad(GamepadButton::DPadLeft) => pad.dpad_left_pressed,
        KeymapEntry::Pad(GamepadButton::DPadUp) => pad.dpad_up_pressed,
        KeymapEntry::Pad(GamepadButton::DPadRight) => pad.dpad_right_pressed,
        KeymapEntry::Pad(GamepadButton::DPadDown) => pad.dpad_down_pressed,
        _ => false,
    }
}

/// Backend-neutral port of bevy `sample_input`'s touch zones.
///
/// GML law sources (`JoystickMove/Other_10`, `JoystickAttack/Other_10`,
/// `ButtonAct/Swap/Active/Attack/Other_10`, `get_nearest_touch`,
/// `scrStickRegions`):
/// - Sticks capture within `rad * 1.75 * (controls_scale + 0.5)`
///   (`rad` 32); touches claimed by another element lose
///   (`get_nearest_touch` exclusion).
/// - Move stick snaps to a free touch anywhere in the left half;
///   attack stick repositions toward its press point once, at claim
///   time only (`scrStickRegions`, 0.8 lerp on the press edge — never
///   a per-tick chase).
/// - Move: `moving = min(1, hold_ramp + dis/rad)`, direction hold ramp
///   `current_move_direction_time` (+1/tick within 10° of the held
///   direction, −3/tick otherwise, normalized `(t−10)/20`).
/// - Attack deflects `dis = min(rad, mdis) * 2` (Crystal TB: half view
///   range scaled); fires only past `ATTACK_BUTTON_DEADZONE` 0.4125.
///   Press/release edges are swapped by design (`release_fire` on the
///   press edge, `press_fire` on the lift edge).
/// - Corner zones: outer top-right pulses ability on the press edge,
///   inner top-right pulses swap on the press edge (GML
///   `Player/Step_0:22` swaps on `press_swap` only).
/// - `ButtonAct` pulses interact on the press edge; `ButtonActive`
///   holds spec while held plus press/release edges every tick;
///   `ButtonAttack` (splitfire) mirrors held/press/release raw.
///
/// Port adaptations: `TouchContact` carries the stable shell finger id
/// (GML touch slot), so stick claims key on it; lift edges arrive via
/// `note_touch_released` (the shell drops contacts on lift). `output`
///'s stick anchors, claim ids, and hidden claim-state persist across
/// ticks (GML `x/y` + `index` on the stick objects). The caller must
/// run this every tick — even with zero contacts — so claims release
/// and lift edges fire on the lift frame.
pub fn sample_touch(
    contacts: &[TouchContact],
    window_width: f32,
    scale: f32,
    split_fire: bool,
    output: &mut NtInput,
) {
    sample_touch_full(
        contacts,
        window_width,
        scale,
        split_fire,
        false,
        output,
    )
}

/// [`sample_touch`] with the stick-regions gate. GML
/// `opt_stickregions` off (default): sticks never reposition and only
/// claim touches already on them; free left/right-half touches drive
/// the move stick (snap) / attack stick (fixed home) directly.
pub fn sample_touch_full(
    contacts: &[TouchContact],
    window_width: f32,
    scale: f32,
    split_fire: bool,
    stick_regions: bool,
    output: &mut NtInput,
) {
    let width = window_width;
    let btn_capture = TOUCH_BUTTON_RADIUS * (scale + 0.5);

    // Default stick anchors from the GUI size (`JoystickMove/Create_0`,
    // `JoystickAttack/Create_0`: move `(view-max)/2 + 64, h - 64`,
    // attack `w + (max-w)/2 - 64, h - 64`). The view here is the GUI
    // width; `global.view_width_max` is the widened cap — the port
    // centers on `width` directly (no widen split).
    let gui_h = 240.0;
    let move_home = Vec2::new(64.0, gui_h - 64.0);
    let attack_home = Vec2::new(width - 64.0, gui_h - 64.0);
    // `ButtonAct` (w/2, 48), `ButtonSwap` ((w-max)/2+64, h/2-48),
    // `ButtonActive` (w+(max-w)/2-64, h/2-48), `ButtonAttack`
    // (w+(max-w)/2-48, h/2) — same centering simplification.
    let act_home = Vec2::new(width * 0.5, 48.0);
    let swap_home = Vec2::new(64.0, gui_h * 0.5 - 48.0);
    let active_home = Vec2::new(width - 64.0, gui_h * 0.5 - 48.0);
    let attack_btn_home = Vec2::new(width - 48.0, gui_h * 0.5);

    // Claim ids: stable shell finger ids (GML touch slot), -1 = free.
    // `get_nearest_touch` picks the touch whose *current* position is
    // nearest the claimant; `scrStickRegions` runs only on the press
    // edge, so the claim runs on `just_pressed` contacts there.
    let mut move_stick = output.move_stick.unwrap_or(TouchStick {
        anchor: move_home,
        touch: -1,
        ..Default::default()
    });
    move_stick.anchor = if move_stick.touch < 0 {
        move_home
    } else {
        move_stick.anchor
    };
    let mut attack_stick = output.attack_stick.unwrap_or(TouchStick {
        anchor: attack_home,
        touch: -1,
        ..Default::default()
    });

    let claimed = |id: i64| -> Option<&TouchContact> {
        if id < 0 {
            return None;
        }
        contacts.iter().find(|c| c.id as i64 == id)
    };
    // Nearest free contact: nearest by *current* position within `rad`
    // of `at`, skipping contacts another element claims
    // (`get_nearest_touch` + MobileUI exclusion; the GML nearest-MobileUI
    // tiebreak collapses because a claimed id is excluded everywhere).
    let nearest_free = |at: Vec2, rad: f32, held: &[i64]| -> Option<usize> {
        let mut best: Option<(usize, f32)> = None;
        for (i, c) in contacts.iter().enumerate() {
            if held.contains(&(c.id as i64)) {
                continue;
            }
            let d = c.pos.distance(at);
            if d <= rad && best.is_none_or(|(_, bd)| d < bd) {
                best = Some((i, d));
            }
        }
        best.map(|(i, _)| i)
    };

    // Fixed buttons first (they steal touches from sticks):
    // ability outer corner, cycle inner corner, act, swap, active,
    // attack button in splitfire.
    let mut held: Vec<i64> = Vec::new();
    if move_stick.touch >= 0 {
        held.push(move_stick.touch);
    }
    if attack_stick.touch >= 0 {
        held.push(attack_stick.touch);
    }
    // Lifted ids still in flight (shell `TouchUp` raced the contact
    // snapshot, or the viewport path latched them): treat them as
    // missing everywhere below, then drain. Matching a live claim
    // fires that element's lift edge at its update site.
    let lifted: Vec<i64> = std::mem::take(&mut output.touch_lifted);
    let gone = |id: i64| lifted.contains(&id);

    // Ability corner (outer top-right): press edge only.
    if contacts.iter().any(|c| {
        c.just_pressed && c.start.y < 96.0 && c.start.x >= width - 96.0
    }) {
        output.ability_pressed = true;
    }
    // Cycle corner (inner top-right) folds into the swap-button
    // claim below (same GML gesture).
    // Act button (`ButtonAct/Other_10`): press_pick on press edge.
    let act_idx = nearest_free(act_home, btn_capture, &held);
    if let Some(i) = act_idx {
        held.push(contacts[i].id as i64);
        if contacts[i].just_pressed {
            output.press_interact();
        }
    }
    // Swap button (`ButtonSwap/Other_10`): press edge only. GML swaps
    // on `press_swap` (`Player/Step_0:22`); `release_swap` feeds the
    // disabled wepstick handoff, never the swap itself — so the lift
    // latch must NOT cycle here (one tap = one swap).
    // The top-right cycle corner counts as a swap-button tap (same
    // `get_nearest_touch(rad)` claim in GML — the corner tap below and
    // the button claim are one gesture).
    let swap_idx = nearest_free(swap_home, btn_capture, &held).or_else(|| {
        contacts.iter().position(|c| {
            c.just_pressed && c.start.y < 96.0 && c.start.x >= width - 192.0
        })
    });
    if let Some(i) = swap_idx {
        held.push(contacts[i].id as i64);
        if contacts[i].just_pressed {
            output.cycle_weapon = output.cycle_weapon.saturating_add(1);
        }
    }
    // Active button (`ButtonActive/Other_10`): hold while held plus
    // press/release edges every tick (`hold_spec` drives hold
    // abilities; the press edge routes the tap ability).
    let active_idx = nearest_free(active_home, btn_capture, &held);
    if let Some(i) = active_idx {
        held.push(contacts[i].id as i64);
        output.spec_held = true;
        if contacts[i].just_pressed {
            output.spec_pressed = true;
            output.ability_pressed = true;
        }
    }
    // Splitfire attack button (`ButtonAttack/Other_10`): raw fire edges.
    // In splitfire the stick is the AIM JOYSTICK, so the button also
    // claims a live finger by id for the release edge below.
    let mut split_btn_touch: i64 = -1;
    if split_fire {
        let btn_idx = nearest_free(attack_btn_home, TOUCH_STICK_RADIUS * (scale + 0.5), &held);
        if let Some(i) = btn_idx {
            split_btn_touch = contacts[i].id as i64;
            held.push(split_btn_touch);
            output.fire_held = true;
            if contacts[i].just_pressed {
                output.press_fire();
            }
        }
        if lifted.iter().any(|id| *id == split_btn_touch && split_btn_touch >= 0)
            || output.touch_released_fire
        {
            output.fire_held = false;
            output.fire_released = true;
        }
    }

    // Move stick (`JoystickMove/Other_10`): press-edge reposition
    // anywhere in the left half under `opt_stickregions`
    // (`scrStickRegions` move arm, gated on
    // `device_mouse_check_button_pressed` — a free touch beats the
    // `get_nearest_touch` result). Without stick regions the stick
    // never moves and claims only touches already on it; a free left
    // touch that misses the home capture falls through (no plain-tap
    // move zone in GML — `KeyCont.moving` just stays 0). The touch
    // claims by its start here: the press frame's `pos == start`, so
    // the claim is the pressed finger wherever it landed in the left
    // half.
    if move_stick.touch < 0 {
        if let Some(i) = contacts.iter().position(|c| {
            c.just_pressed
                && c.start.x < width * 0.5
                && !held.contains(&(c.id as i64))
                && (stick_regions || {
                    let r = TOUCH_STICK_RADIUS * 1.75 * (scale + 0.5);
                    c.start.distance(move_stick.anchor) <= r
                })
        }) {
            let c = &contacts[i];
            move_stick.touch = c.id as i64;
            if stick_regions {
                move_stick.anchor = c.start;
            }
            move_stick.dis = 0.0;
            move_stick.dir = 0.0;
            held.push(c.id as i64);
        }
    }
    output.move_axis = Vec2::ZERO;
    if gone(move_stick.touch) {
        move_stick.touch = -1;
        move_stick.anchor = move_home;
    }
    if let Some(c) = claimed(move_stick.touch) {
        let d = c.pos - move_stick.anchor;
        let dis = d.length();
        if dis > 0.0 {
            let dir_deg = d.y.atan2(d.x).to_degrees();
            let prev = move_stick.dir;
            let mut diff = (dir_deg - prev) % 360.0;
            if diff > 180.0 {
                diff -= 360.0;
            }
            if diff < -180.0 {
                diff += 360.0;
            }
            if diff.abs() < 10.0 {
                if move_stick.hold_time < 30.0 {
                    move_stick.hold_time += 1.0;
                }
            } else if move_stick.hold_time > 0.0 {
                move_stick.hold_time = (move_stick.hold_time - 3.0).max(0.0);
            }
            move_stick.dir = dir_deg;
            let same = (move_stick.hold_time - 10.0).max(0.0) / 20.0;
            let moving = (same + dis / TOUCH_STICK_RADIUS).min(1.0);
            if moving > 0.0 {
                output.move_axis = d.normalize_or_zero() * moving;
            }
            move_stick.dis = dis;
        } else {
            move_stick.dis = 0.0;
        }
    } else {
        if move_stick.touch >= 0 {
            move_stick.touch = -1;
            move_stick.anchor = move_home;
        }
        if move_stick.hold_time > 0.0 {
            move_stick.hold_time = (move_stick.hold_time - 3.0).max(0.0);
        }
    }
    output.move_stick = Some(move_stick);

    // Attack stick (`JoystickAttack/Other_10`): claim + one-shot
    // reposition on the press edge (`scrStickRegions` attack arm runs
    // only under `device_mouse_check_button_pressed`: anchor chases
    // the press point once, `x = lerp(x, mx, 0.8)`). After that the
    // anchor stays fixed — the deflection is measured from it every
    // tick (`dis = min(rad, mdis) * 2`, fires past the 0.4125
    // deadzone with swapped press/release edges). Aim follows the
    // stick heading; `touch_dis` carries the deflection for spread.
    // In splitfire the stick keeps aiming while the separate button
    // fires (`ButtonAttack` owns `hold_fire`; the stick is the AIM
    // JOYSTICK there) — only the fire block is gated below.
    // `scrStickRegions` only repositions under `opt_stickregions`:
    // without it the attack stick stays at home and claims only
    // touches already within the capture radius of the home anchor.
    if attack_stick.touch < 0 {
        if let Some(i) = contacts.iter().position(|c| {
            c.just_pressed
                && c.start.x >= width * 0.5
                && c.start.y >= 96.0
                && !held.contains(&(c.id as i64))
                && (stick_regions || {
                    let r = TOUCH_STICK_RADIUS * 1.75 * (scale + 0.5);
                    c.start.distance(attack_stick.anchor) <= r
                })
        }) {
            let c = &contacts[i];
            if stick_regions {
                attack_stick.anchor += (c.start - attack_stick.anchor) * 0.8;
            }
            attack_stick.touch = c.id as i64;
            held.push(c.id as i64);
        }
    }
    output.touch_dis = 0.0;
    let attack_was = attack_stick.touch;
    let attack_lifted = attack_was >= 0 && gone(attack_was);
    if let Some(c) = claimed(attack_stick.touch) {
        let raw = c.pos - attack_stick.anchor;
        // GML releases the claim past `rad * 3` from the anchor
        // (`distance_to_point(mx, my) > rad * 3`).
        if raw.length() > TOUCH_STICK_RADIUS * 3.0 {
            attack_stick.touch = -1;
            attack_stick.dis = 0.0;
        } else {
            let mdis = raw.length().min(TOUCH_STICK_RADIUS);
            if raw.length_squared() > 0.0 {
                let dir_deg = raw.y.atan2(raw.x).to_degrees();
                attack_stick.dir = dir_deg;
                output.aim_axis = raw.normalize_or_zero();
            }
            let dis = mdis * 2.0;
            attack_stick.dis = dis;
            output.touch_dis = dis;
            if !split_fire && dis / TOUCH_STICK_RADIUS > ATTACK_BUTTON_DEADZONE {
                output.fire_held = true;
                // Edges are swapped by design (`Other_10` comment):
                // the press frame reports `release_fire`, the lift
                // frame reports `press_fire`.
                if c.just_pressed {
                    output.touch_released_fire = true;
                }
            }
        }
    } else {
        // GML `index = -1` on `!device_mouse_check_button(...)` (the
        // finger lifted). The lift frame reports `press_fire`
        // (swapped edges); `fire_held` drops.
        if attack_stick.touch >= 0 {
            attack_stick.touch = -1;
        }
        attack_stick.dis = 0.0;
        if attack_lifted {
            output.fire_held = false;
            output.fire_released = true;
        }
    }
    output.attack_stick = Some(attack_stick);
    output.touch_released_swap = false;
    output.touch_released_fire = false;

    output.move_axis = output.move_axis.clamp_length_max(1.0);
    output.aim_axis = output.aim_axis.clamp_length_max(1.0);
    // A bare tap (no stick/button claim — menus, splash advance) is a
    // fire edge: GML `Vlambeer/Draw_0` advances on any press
    // (`mouse_ui_clicked`, `keyboard_anykey`, gamepad anykey) with no
    // touch object involved. The stick/button claims above only fire
    // for claimed fingers, so an unclaimed tap would otherwise vanish.
    if !output.fire_pressed
        && !output.fire_released
        && !output.touch_released_fire
        && contacts.iter().any(|c| c.just_pressed)
    {
        output.fire_released = true;
    }
}

/// Drop everything when the sim isn't live (paused, overlay open, or out
/// of game — bevy `clear_input_when_inactive` parity plus the overlay
/// conjunct: an overlay opened without the `Paused` flag must still
/// swallow gameplay pulses like ability/spec).
///
/// Menu states (Splash/Loading/MainMenu/Title) deliberately keep their
/// advance edges: `feed_input` stages the tap into
/// fire/interact/spec/ability pulses, the sim tick runs, then
/// `tick_menus` consumes them. Clearing here would eat the tap before
/// the menu tick ever sees it (the stuck-splash-4 bug).
pub fn clear_input_when_inactive(
    paused: Res<crate::state::Paused>,
    state: Res<crate::state::AppState>,
    overlay: Option<Res<crate::state::OverlayMenu>>,
    mut input: ResMut<NtInput>,
) {
    use crate::state::AppState;
    let overlay_open = overlay.is_some_and(|o| *o != crate::state::OverlayMenu::None);
    if paused.0 || overlay_open || *state != AppState::InGame {
        if matches!(*state, AppState::Splash | AppState::Loading) {
            drain_interact_pulse(&state, &mut input);
            return;
        }
        input.clear_transient();
    } else {
        drain_interact_pulse(&state, &mut input);
    }
}

#[cfg(test)]
mod keymap_tests {
    use super::*;
    use crate::keymap::{InputMapState, NtAction};
    use repame_input::{KeymapDevice, KeymapEntry};
    use repose_core::input::{Key, Modifiers, PointerButton};
    use repose_core::shortcuts::KeyChord;

    fn contact(start: [f32; 2], pos: [f32; 2], just_pressed: bool) -> TouchContact {
        // Stable finger id per (start, press) gesture so multi-frame
        // claims survive: a held finger keeps its press frame's start,
        // so reuse the last id while the start matches.
        static NEXT_ID: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
        thread_local! {
            static LAST: std::cell::Cell<(u64, [u32; 2])> =
                const { std::cell::Cell::new((0, [0, 0])) };
        }
        let bits = [start[0].to_bits(), start[1].to_bits()];
        let id = LAST.with(|last| {
            let (id, prev) = last.get();
            if id != 0 && prev == bits {
                id
            } else {
                let id = NEXT_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                last.set((id, bits));
                id
            }
        });
        contact_id(id, start, pos, just_pressed)
    }
    fn contact_id(
        id: u64,
        start: [f32; 2],
        pos: [f32; 2],
        just_pressed: bool,
    ) -> TouchContact {
        TouchContact {
            id,
            start: Vec2::from(start),
            pos: Vec2::from(pos),
            just_pressed,
        }
    }

    #[test]
    fn move_stick_magnitude_ramps_with_deflection() {
        let mut out = NtInput::default();
        // Stick regions on (GML default off, but the claim + snap law
        // under test is the stickregions arm): the stick repositions
        // to the press point, then the drag deflects from it.
        sample_touch_full(
            &[contact([40.0, 200.0], [40.0, 200.0], true)],
            320.0,
            0.5,
            false,
            true,
            &mut out,
        );
        sample_touch_full(
            &[contact([40.0, 200.0], [40.0, 168.0], false)],
            320.0,
            0.5,
            false,
            true,
            &mut out,
        );
        assert!(out.move_axis.length() > 0.9, "got {:?}", out.move_axis);
        assert!(out.move_axis.y < 0.0);
        assert!(!out.fire_held);
    }

    #[test]
    fn attack_stick_deadzone_gates_fire() {
        let mut out = NtInput::default();
        // Touch starts on the stick home and drags 6px: aims without
        // firing (0.4125 deadzone on dis/rad).
        sample_touch_full(
            &[contact([256.0, 176.0], [256.0, 176.0], true)],
            320.0,
            0.5,
            false,
            true,
            &mut out,
        );
        sample_touch_full(
            &[contact([256.0, 176.0], [262.0, 176.0], false)],
            320.0,
            0.5,
            false,
            true,
            &mut out,
        );
        assert!(!out.fire_held, "grazing the stick must not fire");
        assert!(out.aim_axis.x > 0.0);
        // Full 32px drag fires; the deflection persists while held
        // (GML measures from the fixed anchor — no per-tick chase).
        sample_touch_full(
            &[contact([256.0, 176.0], [288.0, 176.0], false)],
            320.0,
            0.5,
            false,
            true,
            &mut out,
        );
        assert!(out.fire_held);
        assert!(out.touch_dis > 0.0);
        for _ in 0..6 {
            sample_touch_full(
                &[contact([256.0, 176.0], [288.0, 176.0], false)],
                320.0,
                0.5,
                false,
                true,
                &mut out,
            );
        }
        assert!(out.fire_held, "held deflection must keep firing");
        assert!(out.touch_dis > 0.0);
        // Lift: claim releases and the swapped press edge fires (the
        // finger id stages through `note_touch_released`, like the
        // shell `TouchUp` path — the contact itself is already gone).
        let attack_id = out.attack_stick.map(|s| s.touch).unwrap_or(-1);
        let mut lifted = NtInput::default();
        lifted.move_stick = out.move_stick;
        lifted.attack_stick = out.attack_stick;
        lifted.note_touch_released(attack_id);
        sample_touch_full(&[], 320.0, 0.5, false, true, &mut lifted);
        assert!(lifted.attack_stick.is_some_and(|s| s.touch < 0));
        assert!(lifted.take_fire_released(), "lift must report press_fire");
    }

    #[test]
    fn swap_tap_cycles_once() {
        let mut out = NtInput::default();
        sample_touch_full(
            &[contact([64.0, 72.0], [64.0, 72.0], true)],
            320.0,
            0.5,
            false,
            true,
            &mut out,
        );
        assert_eq!(out.take_cycle_weapon(), 1);
        // Release frame: no second cycle (GML swaps on press only).
        sample_touch_full(&[], 320.0, 0.5, false, true, &mut out);
        assert_eq!(out.take_cycle_weapon(), 0);
        assert!(!out.take_fire_released());
    }

    #[test]
    fn two_fingers_keep_their_claims() {
        let mut out = NtInput::default();
        sample_touch_full(
            &[
                contact_id(1, [40.0, 200.0], [40.0, 200.0], true),
                contact_id(2, [256.0, 176.0], [256.0, 176.0], true),
            ],
            320.0,
            0.5,
            false,
            true,
            &mut out,
        );
        let move_id = out.move_stick.map(|s| s.touch).unwrap_or(-1);
        let attack_id = out.attack_stick.map(|s| s.touch).unwrap_or(-1);
        assert!(move_id >= 0 && attack_id >= 0 && move_id != attack_id);
        // Reordered + dragged: claims follow the finger ids, not the
        // slice order.
        sample_touch_full(
            &[
                contact_id(2, [256.0, 176.0], [288.0, 176.0], false),
                contact_id(1, [40.0, 200.0], [40.0, 168.0], false),
            ],
            320.0,
            0.5,
            false,
            true,
            &mut out,
        );
        assert_eq!(out.move_stick.map(|s| s.touch), Some(move_id));
        assert_eq!(out.attack_stick.map(|s| s.touch), Some(attack_id));
        assert!(out.move_axis.length() > 0.5);
        assert!(out.fire_held);
        assert!(out.touch_dis > 0.0);
    }

    #[test]
    fn split_fire_aims_while_button_fires() {
        let mut out = NtInput::default();
        // Aim finger claims the stick right of the fire button zone.
        sample_touch_full(
            &[
                contact_id(1, [250.0, 176.0], [250.0, 176.0], true),
                contact_id(2, [280.0, 120.0], [280.0, 120.0], true),
            ],
            320.0,
            0.5,
            true,
            true,
            &mut out,
        );
        assert!(out.fire_held, "button finger fires");
        sample_touch_full(
            &[
                contact_id(1, [250.0, 176.0], [282.0, 176.0], false),
                contact_id(2, [280.0, 120.0], [280.0, 120.0], false),
            ],
            320.0,
            0.5,
            true,
            true,
            &mut out,
        );
        assert!(out.fire_held);
        assert!(out.aim_axis.x > 0.0, "stick still aims in splitfire");
        assert!(out.touch_dis > 0.0);
    }

    #[test]
    fn split_fire_button_fires_without_stick() {
        let mut out = NtInput::default();
        sample_touch_full(
            &[contact([280.0, 120.0], [280.0, 120.0], true)],
            320.0,
            0.5,
            true,
            true,
            &mut out,
        );
        assert!(out.fire_held);
        assert_eq!(out.touch_dis, 0.0);
    }

    #[test]
    fn act_button_pulses_interact() {
        let mut out = NtInput::default();
        sample_touch_full(
            &[contact([160.0, 48.0], [160.0, 48.0], true)],
            320.0,
            0.5,
            false,
            true,
            &mut out,
        );
        assert!(out.take_interact_pressed());
    }

    fn chord(c: char) -> KeymapEntry {
        KeymapEntry::Key(KeyChord::new(Key::Character(c), Modifiers::default()))
    }

    fn state() -> InputMapState {
        InputMapState::default()
    }

    #[test]
    fn default_map_keeps_hardcoded_parity() {
        let s = state();
        let held: HashSet<KeyCode> = [KeyCode::KeyW, KeyCode::KeyD].into_iter().collect();
        let mut out = NtInput::default();
        sample_keyboard_mapped(
            &held,
            &HashSet::new(),
            &MouseState::default(),
            Some(&s),
            &mut out,
        );
        // Y-down world: W steers −y, D steers +x (diagonal).
        assert!((out.move_axis.x + out.move_axis.y).abs() < 1e-6);
        assert!(out.move_axis.x > 0.7 && out.move_axis.y < -0.7);
    }

    #[test]
    fn rebound_move_key_steers() {
        let mut s = state();
        s.session.map.set_keyboard(NtAction::North, chord('z'));
        let held: HashSet<KeyCode> = [KeyCode::KeyW].into_iter().collect();
        let mut out = NtInput::default();
        sample_keyboard_mapped(
            &held,
            &HashSet::new(),
            &MouseState::default(),
            Some(&s),
            &mut out,
        );
        // W no longer moves north after the rebind (arrows still would).
        assert_eq!(out.move_axis, Vec2::ZERO);
    }

    #[test]
    fn rebound_fire_key_fires() {
        let mut s = state();
        s.session.map.set_keyboard(NtAction::Fire, chord('f'));
        let just: HashSet<KeyCode> = [KeyCode::KeyF].into_iter().collect();
        let mut out = NtInput::default();
        sample_keyboard_mapped(
            &HashSet::new(),
            &just,
            &MouseState::default(),
            Some(&s),
            &mut out,
        );
        assert!(out.fire_pressed);
        assert!(out.fire_held);
    }

    #[test]
    fn mouse_rebound_fire_still_clicks() {
        let mut s = state();
        s.session.map
            .set_keyboard(NtAction::Fire, KeymapEntry::Mouse(PointerButton::Primary));
        let mouse = MouseState {
            left_held: true,
            left_pressed: true,
            ..MouseState::default()
        };
        let mut out = NtInput::default();
        sample_keyboard_mapped(&HashSet::new(), &HashSet::new(), &mouse, Some(&s), &mut out);
        assert!(out.fire_pressed && out.fire_held);
    }

    #[test]
    fn swap_pulse_cycles() {
        let s = state();
        let just: HashSet<KeyCode> = [KeyCode::Space].into_iter().collect();
        let mut out = NtInput::default();
        sample_keyboard_mapped(
            &HashSet::new(),
            &just,
            &MouseState::default(),
            Some(&s),
            &mut out,
        );
        assert_eq!(out.cycle_weapon, 1);
    }

    #[test]
    fn capture_rebinds_keyboard_side() {
        let mut s = state();
        s.begin_capture(NtAction::North, KeymapDevice::KeyboardMouse);
        s.resolve_capture(Some(chord('z')));
        assert_eq!(s.session.map.keyboard(&NtAction::North), chord('z'));
        assert_eq!(s.session.map.gamepad(&NtAction::North), s.session.map.gamepad(&NtAction::North));
    }

    #[test]
    fn rmb_raises_spec_not_fire() {
        let s = state();
        let mouse = MouseState {
            right_held: true,
            right_pressed: true,
            ..MouseState::default()
        };
        let mut out = NtInput::default();
        sample_keyboard_mapped(&HashSet::new(), &HashSet::new(), &mouse, Some(&s), &mut out);
        assert!(out.spec_held && out.spec_pressed && out.ability_pressed);
        assert!(!out.fire_held && !out.fire_pressed);
    }

    #[test]
    fn lmb_raises_fire_not_spec() {
        let s = state();
        let mouse = MouseState {
            left_held: true,
            left_pressed: true,
            ..MouseState::default()
        };
        let mut out = NtInput::default();
        sample_keyboard_mapped(&HashSet::new(), &HashSet::new(), &mouse, Some(&s), &mut out);
        assert!(out.fire_held && out.fire_pressed);
        assert!(!out.spec_held && !out.spec_pressed && !out.ability_pressed);
    }
}
