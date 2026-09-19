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

    fire_pressed: bool,
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
            menu_nav_v: 0,
            menu_nav_h: 0,
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

    pub fn clear_transient(&mut self) {
        self.fire_pressed = false;
        self.ability_pressed = false;
        self.interact_pressed = false;
        self.spec_pressed = false;
        self.spec_held = false;
        self.weapon_slot = None;
        self.cycle_weapon = 0;
        self.menu_nav_v = 0;
        self.menu_nav_h = 0;
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
        // `Physical` entries never occur: `capture_physical_press`
        // resolves every capture through `chord_for_physical` into a
        // `Key` chord, so only glyph chords reach the sampler.
        KeymapEntry::Key(chord) => keycode_for_chord(&chord.key),
        KeymapEntry::None
        | KeymapEntry::Physical(_)
        | KeymapEntry::Mouse(_)
        | KeymapEntry::Pad(_)
        | KeymapEntry::Axis { .. } => None,
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
/// aim); triggers/south/east OR into held/pulses; D-pad edges replace
/// the weapon slot. Returns this pad's cycle step (North = +1); the
/// caller applies the bevy overwrite law (last pad wins, added once —
/// see `sample_gamepads`).
///
/// Stick Y arrives screen-down (gilrs/SDL convention matches this
/// port's y-down world), so unlike the y-up bevy build no flip is
/// applied: stick-up (−y) moves north.
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
/// held contacts become virtual sticks (`(pos - start)` unflipped —
/// screen y-down IS world y-down here, so drag-up (−y) steers north,
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
        let stick = screen_delta / 56.0;
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

/// Drop everything when the sim isn't live (paused, overlay open, or out
/// of game — bevy `clear_input_when_inactive` parity plus the overlay
/// conjunct: an overlay opened without the `Paused` flag must still
/// swallow gameplay pulses like ability/spec).
pub fn clear_input_when_inactive(
    paused: Res<crate::state::Paused>,
    state: Res<crate::state::AppState>,
    overlay: Option<Res<crate::state::OverlayMenu>>,
    mut input: ResMut<NtInput>,
) {
    use crate::state::AppState;
    let overlay_open = overlay.is_some_and(|o| *o != crate::state::OverlayMenu::None);
    if paused.0 || overlay_open || *state != AppState::InGame {
        input.clear_transient();
    }
}

#[cfg(test)]
mod keymap_tests {
    use super::*;
    use crate::keymap::{InputMapState, NtAction};
    use repame_input::{KeymapDevice, KeymapEntry};
    use repose_core::input::{Key, Modifiers, PointerButton};
    use repose_core::shortcuts::KeyChord;

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
