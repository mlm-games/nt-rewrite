use bevy_ecs::prelude::*;
use repame_input::{Keymap, KeymapCapture, KeymapDevice, KeymapEntry, KeymapRow};
use repose_core::input::{GamepadButton, Key, Modifiers, PointerButton};
use repose_core::shortcuts::KeyChord;
use serde::{Deserialize, Serialize};

/// Remappable gameplay controls, GML `scrKeymapsSetup` parity.
///
/// Actions match the GML `Key` struct rows (`fire`, `spec`, `swap`,
/// `pick`, `north`, `south`, `west`, `east`); `chat`/`console` stay
/// out — chat has no co-op peer in this build and the debug console
/// keeps its hardcoded backtick. Each action carries the GML default
/// keyboard/mouse entry plus the gamepad fallback
/// (`fire: [mb_left, gp_shoulderr]`, ...). Overrides persist per
/// device family in the save file; the REMAP page arms a capture and
/// the next pressed input resolves it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum NtAction {
    Fire,
    Spec,
    Swap,
    Pick,
    North,
    South,
    West,
    East,
}

impl NtAction {
    pub const ALL: [NtAction; 8] = [
        NtAction::Fire,
        NtAction::Spec,
        NtAction::Swap,
        NtAction::Pick,
        NtAction::North,
        NtAction::South,
        NtAction::West,
        NtAction::East,
    ];

    pub fn name(self) -> &'static str {
        match self {
            NtAction::Fire => "fire",
            NtAction::Spec => "spec",
            NtAction::Swap => "swap",
            NtAction::Pick => "pick",
            NtAction::North => "north",
            NtAction::South => "south",
            NtAction::West => "west",
            NtAction::East => "east",
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            NtAction::Fire => "FIRE",
            NtAction::Spec => "ACTIVE",
            NtAction::Swap => "SWAP",
            NtAction::Pick => "PICK/USE",
            NtAction::North => "WALK UP",
            NtAction::South => "WALK DOWN",
            NtAction::West => "WALK LEFT",
            NtAction::East => "WALK RIGHT",
        }
    }

    pub fn from_name(name: &str) -> Option<NtAction> {
        Some(match name {
            "fire" => NtAction::Fire,
            "spec" => NtAction::Spec,
            "swap" => NtAction::Swap,
            "pick" => NtAction::Pick,
            "north" => NtAction::North,
            "south" => NtAction::South,
            "west" => NtAction::West,
            "east" => NtAction::East,
            _ => return None,
        })
    }
}

fn chord(c: char) -> KeymapEntry {
    KeymapEntry::Key(KeyChord::new(
        Key::Character(c),
        Modifiers::default(),
    ))
}

fn chord_shift(c: char) -> KeymapEntry {
    KeymapEntry::Key(KeyChord::new(
        Key::Character(c),
        Modifiers {
            shift: true,
            ..Modifiers::default()
        },
    ))
}

/// GML `scrKeymapsSetup` defaults verbatim, minus the debug `horn`
/// row: `spec` is RMB (GML default `mb_right`; Shift is the keyboard
/// fallback the sampler ORs in next to the row), `swap` is Space,
/// `pick` is E.
pub fn default_keymap() -> Keymap<NtAction> {
    let mut map = Keymap::new();
    map.set_keyboard(
        NtAction::Fire,
        KeymapEntry::Mouse(PointerButton::Primary),
    );
    map.set_keyboard(
        NtAction::Spec,
        KeymapEntry::Mouse(PointerButton::Secondary),
    );
    map.set_keyboard(
        NtAction::Swap,
        KeymapEntry::Key(KeyChord::new(Key::Space, Modifiers::default())),
    );
    map.set_keyboard(NtAction::Pick, chord('e'));
    map.set_keyboard(NtAction::North, chord('w'));
    map.set_keyboard(NtAction::South, chord('s'));
    map.set_keyboard(NtAction::West, chord('a'));
    map.set_keyboard(NtAction::East, chord('d'));
    map.set_gamepad(
        NtAction::Fire,
        KeymapEntry::Pad(GamepadButton::RightShoulder),
    );
    map.set_gamepad(
        NtAction::Spec,
        KeymapEntry::Pad(GamepadButton::LeftShoulder),
    );
    map.set_gamepad(
        NtAction::Swap,
        KeymapEntry::Pad(GamepadButton::RightShoulder),
    );
    map.set_gamepad(NtAction::Pick, KeymapEntry::Pad(GamepadButton::South));
    map.set_gamepad(NtAction::North, KeymapEntry::Pad(GamepadButton::DPadUp));
    map.set_gamepad(
        NtAction::South,
        KeymapEntry::Pad(GamepadButton::DPadDown),
    );
    map.set_gamepad(NtAction::West, KeymapEntry::Pad(GamepadButton::DPadLeft));
    map.set_gamepad(
        NtAction::East,
        KeymapEntry::Pad(GamepadButton::DPadRight),
    );
    // Shift is the keyboard-side `spec` fallback next to RMB: GML
    // binds `spec` to `mb_right` only. The sampler ORs the Shift
    // levels in next to the row, so a rebound Shift key keeps working.
    let _ = chord_shift;
    map
}

/// Serializable keymap rows for the save file (`keyboard.<action>`,
/// `gamepad.<action>` sections in GML `saveData`).
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct KeyBindings {
    pub rows: Vec<KeymapRow>,
}

impl KeyBindings {
    pub fn to_keymap(&self) -> Keymap<NtAction> {
        let mut map = default_keymap();
        for row in &self.rows {
            let Some(action) = NtAction::from_name(&row.action) else {
                continue;
            };
            if !row.keyboard.is_empty() {
                map.set_keyboard(
                    action,
                    repame_input::decode_keymap_entry(&row.keyboard),
                );
            }
            if !row.gamepad.is_empty() {
                map.set_gamepad(action, repame_input::decode_keymap_entry(&row.gamepad));
            }
        }
        map
    }

    pub fn from_keymap(map: &Keymap<NtAction>) -> Self {
        let mut rows: Vec<KeymapRow> = NtAction::ALL
            .iter()
            .map(|action| KeymapRow {
                action: action.name().to_string(),
                keyboard: repame_input::encode_keymap_entry(&map.keyboard(action)),
                gamepad: repame_input::encode_keymap_entry(&map.gamepad(action)),
            })
            .collect();
        rows.sort_by(|a, b| a.action.cmp(&b.action));
        Self { rows }
    }
}

/// Live input-mapping state: the editable keymap plus the pending
/// REMAP capture (`None` outside the remap gesture). Keyed off the
/// save file on boot (`KeyBindings`), written back on every rebind.
#[derive(Resource, Clone, Debug)]
pub struct InputMapState {
    pub map: Keymap<NtAction>,
    pub capture: Option<KeymapCapture<NtAction>>,
}

impl Default for InputMapState {
    fn default() -> Self {
        Self {
            map: default_keymap(),
            capture: None,
        }
    }
}

impl InputMapState {
    pub fn begin_capture(&mut self, action: NtAction, device: KeymapDevice) {
        self.capture = Some(self.map.begin_capture(action, device));
    }

    pub fn cancel_capture(&mut self) {
        self.capture = None;
    }

    pub fn resolve_capture(&mut self, pressed: Option<KeymapEntry>) {
        if let Some(capture) = self.capture.take() {
            self.map.resolve_capture(&capture, pressed);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gml_defaults_match_scrkeymapssetup() {
        let map = default_keymap();
        assert_eq!(
            map.keyboard(&NtAction::Fire),
            KeymapEntry::Mouse(PointerButton::Primary)
        );
        assert_eq!(
            map.keyboard(&NtAction::Spec),
            KeymapEntry::Mouse(PointerButton::Secondary)
        );
        assert_eq!(
            map.keyboard(&NtAction::Swap),
            KeymapEntry::Key(KeyChord::new(Key::Space, Modifiers::default()))
        );
        assert_eq!(map.keyboard(&NtAction::Pick), chord('e'));
        assert_eq!(map.keyboard(&NtAction::North), chord('w'));
        assert_eq!(
            map.gamepad(&NtAction::Fire),
            KeymapEntry::Pad(GamepadButton::RightShoulder)
        );
        assert_eq!(
            map.gamepad(&NtAction::Pick),
            KeymapEntry::Pad(GamepadButton::South)
        );
    }

    #[test]
    fn bindings_round_trip_through_save_rows() {
        let mut map = default_keymap();
        map.set_keyboard(NtAction::North, chord('z'));
        let saved = KeyBindings::from_keymap(&map);
        let back = saved.to_keymap();
        assert_eq!(back.keyboard(&NtAction::North), chord('z'));
        assert_eq!(
            back.keyboard(&NtAction::Fire),
            KeymapEntry::Mouse(PointerButton::Primary)
        );
    }

    #[test]
    fn capture_rebinds_one_side() {
        let mut state = InputMapState::default();
        state.begin_capture(NtAction::North, KeymapDevice::KeyboardMouse);
        assert!(state.capture.is_some());
        state.resolve_capture(Some(chord('z')));
        assert!(state.capture.is_none());
        assert_eq!(state.map.keyboard(&NtAction::North), chord('z'));
        state.resolve_capture(Some(chord('x')));
        assert_eq!(state.map.keyboard(&NtAction::North), chord('z'));
    }
}
